import gc
import os
import math
import torch
import shutil
import random
import datetime
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch_optimizer as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.patches as patches
import torch.nn.utils.prune as prune
import torchvision.transforms.functional as TF

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from torchvision import transforms
from torch.nn import functional as F
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from mobileyolov3 import MobileYOLOv3, DSConv, Resizer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using training device: {device}")

effective_batch_size = 128
num_workers = 4
num_classes = 1
learning_rate = 4e-4
num_epochs = 400
lr_warmup = num_epochs * 0.1
weight_decay = 1e-4
optim_k = 5
optim_alpha = 0.3
prune_amount = 0.2
dropout_rate = 0.4
t_arch = str(device)
global_img_size = (600, 600)
anchors = [(0.28, 0.35), (0.43, 0.58), (0.62, 0.78)]
num_anchors = len(anchors)
model_path = 'mobileyolov3_coco-text.pth'

down_dir = Path("/mnt/data/")
target_path = down_dir / "COCO-Text"
cache_path = down_dir / "COCO-Text_Cache"

if not os.path.exists(target_path):
    os.makedirs(target_path)
    os.makedirs(cache_path, exist_ok=True)

    dataset_id = "howard-hou/COCO-Text"

    print(f"Downloading {dataset_id}...")

    while True:
        try:
            snapshot_download(dataset_id, repo_type="dataset", cache_dir=str(cache_path), local_dir=str(target_path))
            break
        except Exception as _:
            continue
else:
    print(f"Dataset already exists at {target_path}. Proceed.")

class COCOText(Dataset):
    """
    COCO-Text Dataset adapter for YOLOv3 training, loading from a local directory.
    """
    def __init__(self, num_classes=1, num_anchors=3, img_size=global_img_size, anchors=anchors, split="train"):
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.img_size = img_size
        self.anchors = anchors
        self.batch_count = 0
        self.coarse = (7, 7, self.num_anchors * (5 + self.num_classes))
        self.medium = (14, 14, self.num_anchors * (5 + self.num_classes))
        self.fine = (28, 28, self.num_anchors * (5 + self.num_classes))
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=3.0, p=0.9),
            transforms.ToTensor()
        ])

        # Load the dataset from the local directory 
        # akin to https://stackoverflow.com/questions/77020278/how-to-load-a-huggingface-dataset-from-local-path
        # I named the split exactly like the original, you can change it to your liking
        self.dataset = load_dataset(
            "parquet",
            data_files={
                "train": str(target_path / "data" / "train-*.parquet"),
                "validation": str(target_path / "data" / "validation-*.parquet"),
            },
            cache_dir=str(cache_path),
            split=split,
        )

    def _calculate_iou(self, box1, box2):
        # Intersection over Union (IoU) between two bounding boxes, range of returns is [0, 1]
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[0] + box1[2], box2[0] + box2[2]), min(box1[1] + box1[3], box2[1] + box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area, box2_area = box1[2] * box1[3], box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0 # Avoiding zero division
    
    def _to_grid(self, grid, box):
        # Convert bounding boxes to grid coordinates
        # BBox format: [x, y, w, h, obj, cls]
        # Anchor idxs == num_anchors: 0, 1, 2 
        # Assign BBox to the anchor (and its idx in grid coords therefore) with highest IoU
        grid_h, grid_w = grid.size(0), grid.size(1)
        x, y, w, h, obj, _ = box
        grid_x, grid_y = int(x * grid_w), int(y * grid_h)
        x, y = x * grid_w - grid_x, y * grid_h - grid_y
        best_iou, best_anchor_idx = 0, -1
        for anchor_idx, (anchor_w, anchor_h) in enumerate(self.anchors):
            anchor_box = torch.tensor([x, y, w / anchor_w, h / anchor_h])
            iou = self._calculate_iou(anchor_box.numpy(), [x, y, w, h])
            if iou > best_iou:
                best_iou = iou
                best_anchor_idx = anchor_idx
        if best_anchor_idx >= 0:
            anchor_slice = slice(best_anchor_idx * (5 + self.num_classes), (best_anchor_idx + 1) * (5 + self.num_classes))
            grid[grid_y, grid_x, anchor_slice][:4] = torch.tensor([x, y, w / self.anchors[best_anchor_idx][0], h / self.anchors[best_anchor_idx][1]])
            grid[grid_y, grid_x, anchor_slice][4] = obj

    def _parse_label(self, label):
        # Read Label data, parse it, apply it to (7, 14, 28) grid, resolutions,
        # Concatenate them, flatten and return them as a single tensor
        coarse_labels = torch.zeros(self.coarse)
        medium_labels = torch.zeros(self.medium)
        fine_labels   = torch.zeros(self.fine)
        boxes = [[word["bounding_box"]["top_left_x"], word["bounding_box"]["top_left_y"], word["bounding_box"]["width"], word["bounding_box"]["height"]] for word in label]
        centers = [[box[0] + box[2] / 2, box[1] + box[3] / 2] for box in boxes] # these are all normalized already, so no need to divide by image size
        for i in range(len(boxes)):
            x, y = centers[i]
            w, h = boxes[i][2], boxes[i][3]
            obj, cls = 1.0, 0.0
            box = torch.tensor([x, y, w, h, obj, cls])
            self._to_grid(coarse_labels, box)
            self._to_grid(medium_labels, box)
            self._to_grid(fine_labels, box)
        return torch.cat([coarse_labels.flatten(), medium_labels.flatten(), fine_labels.flatten()], dim=0)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = self.transform(item["image"])
        label = self._parse_label(item["ocr_info"])
        return img, label
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        item = self[self.index]
        self.index += 1
        return item
    
    def get_batch(self, batch_size, randomized=True):
        # returns pairs of (batch_size, 3, 600, 600) image with (batch_size, 18522) label
        # 18522 = (7*7*3*6) + (14*14*3*6) + (28*28*3*6)
        if randomized:
            indices = np.random.choice(len(self), batch_size, replace=False)
        else:
            indices = np.arange(self.batch_count, self.batch_count + batch_size) % len(self)
            self.batch_count += batch_size
        indices = [int(i) for i in indices]
        batch_images = torch.stack([self[i][0] for i in indices], dim=0)
        batch_labels = torch.stack([self[i][1] for i in indices], dim=0)
        return batch_images, batch_labels

    @staticmethod
    def collate_fn(batch):
        # collate_fn purpose is to convert list of (image, label) pairs into a single tensor
        # Gets used internally by DataLoader to prepare batches for training
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, labels

class YoLoss(nn.Module):
    """
    YOLOv3 Custom Loss Function.
    """
    def __init__(self, num_classes=num_classes, num_anchors=num_anchors, lambda_coord=1.0, lambda_obj=3.0, lambda_noobj=0.1,
                 lambda_class=2.0, iou_threshold=0.5, focal_alpha=0.5, focal_gamma=1.0, label_smoothing=0.1, anchors=anchors):
        super(YoLoss, self).__init__()
        self.num_classes = num_classes      # Amount of associable classes
        self.num_anchors = num_anchors      # Count of distinctly predictable objects per grid tile
        self.lambda_coord = lambda_coord    # BBox Coord Loss Weight
        self.lambda_obj = lambda_obj        # Objectness Loss Weight
        self.lambda_noobj = lambda_noobj    # Non-Objectness Loss Weight
        self.lambda_class = lambda_class    # Classification Loss Weight
        self.iou_threshold = iou_threshold  # Intersection over Union Threshold
        self.focal_alpha = focal_alpha      # Focal Loss weight of the positive class
        self.focal_gamma = focal_gamma      # Focal Loss weight of the negative class
        self.label_smoothing = label_smoothing  # Percentage of noise to add to the labels
        self.anchors = torch.tensor(anchors)    # Anchor boxes
        self.eps = 1e-7                     # Imprecision Avoidance Factor
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def focal_loss(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_factor = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)
        modulating_factor = (1.0 - p_t).pow(self.focal_gamma)
        loss = self.bce_loss(pred, target)
        weight = torch.where(target == 1, torch.tensor(60.0).to(pred.device), torch.tensor(1.0).to(pred.device))
        return weight * (alpha_factor * modulating_factor * loss)

    def bbox_iou(self, box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        if xywh:
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        # Area of Bbox Intersection
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        # Union Area
        union = (w1 * h1 + w2 * h2 - inter) + eps
        iou = inter / union
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            if CIoU or DIoU:
                c2 = (cw ** 2 + ch ** 2) + eps
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                if DIoU:
                    return iou - rho2 / c2
                elif CIoU:
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)
            else:
                c_area = cw * ch + eps
                return iou - (c_area - union) / c_area
        else:
            return iou

    def get_box_loss(self, predictions, targets):
        box_loss = 0
        for pi, ti in zip(predictions, targets):
            mask = ti[..., 4] == 1
            p_boxes, t_boxes = pi[mask][..., :4], ti[mask][..., :4]
            if p_boxes.numel() > 0:
                iou = self.bbox_iou(p_boxes, t_boxes, CIoU=True)
                box_loss += torch.mean(1.0 - iou)
                box_loss += F.smooth_l1_loss(p_boxes, t_boxes, reduction='mean')
        return box_loss

    def get_obj_loss(self, predictions, targets):
        obj_loss = 0.0
        noobj_loss = 0.0
        for pi, ti in zip(predictions, targets):
            pred_obj = pi[..., 4]
            target_obj = ti[..., 4]
            obj_loss += torch.mean(self.focal_loss(pred_obj, target_obj))
            noobj_mask = target_obj == 0
            if noobj_mask.any():
                noobj_loss += torch.mean(self.bce_loss(pred_obj[noobj_mask], target_obj[noobj_mask]))
        return self.lambda_obj * obj_loss + self.lambda_noobj * noobj_loss

    def get_cls_loss(self, predictions, targets):
        cls_loss = 0
        if self.num_classes > 1:
            for pi, ti in zip(predictions, targets):
                pred_cls = pi[..., 5:]
                target_cls = ti[..., 5:]
                target_cls = (1 - self.label_smoothing) * target_cls + self.label_smoothing / self.num_classes
                cls_loss += torch.mean(self.focal_loss(pred_cls, target_cls))
        return cls_loss

    def forward(self, predictions, targets):
        b_size = targets.size(0) # Batch size can't be expected to be static
        coarse_size = 7 * 7 * self.num_anchors * (5 + self.num_classes)
        medium_size = 14 * 14 * self.num_anchors * (5 + self.num_classes)
        fine_size = 28 * 28 * self.num_anchors * (5 + self.num_classes)

        flat_coarse, flat_medium, flat_fine = torch.split(targets, [coarse_size, medium_size, fine_size], dim=1)
        t_coarse = flat_coarse.view(b_size, 7, 7, self.num_anchors, (5 + self.num_classes))
        t_medium = flat_medium.view(b_size, 14, 14, self.num_anchors, (5 + self.num_classes))
        t_fine = flat_fine.view(b_size, 28, 28, self.num_anchors, (5 + self.num_classes))

        targets_split = [t_coarse, t_medium, t_fine]
        
        box_loss = self.get_box_loss(predictions, targets_split)
        obj_loss = self.get_obj_loss(predictions, targets_split)
        cls_loss = self.get_cls_loss(predictions, targets_split)
                
        total_loss = self.lambda_coord * box_loss * obj_loss + self.lambda_class * cls_loss

        if torch.isnan(total_loss):
            # Debugging purposes (I saw all of these go NaN at some point, fun times)
            print(f'box_loss={box_loss}, obj_loss={obj_loss}, cls_loss={cls_loss}')
            total_loss = torch.where(torch.isnan(total_loss), torch.zeros_like(total_loss), total_loss)

        return total_loss

class LossTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
    
    def update(self, epoch, train_loss, val_loss=None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
    
    def plot(self, save_path='loss_plot.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            val_epochs = self.epochs[::5]  # Since validation is done every 5 epochs
            plt.plot(val_epochs, self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=str(device)):
                outputs = model(images)
                loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def mixup_data(x, y, alpha=0.1):
    # Intentionally mess up image-to-label relationship for alpha percentage of batch entries
    # Helps to avoid overfitting (when not overdone)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    else:
        return x, y, y, 1

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=10))
    except Exception as e:
        cleanup()
        raise RuntimeError(f"Failed to initialize process group: {e}")

def cleanup():
    dist.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    

def save_checkpoint(state, is_best, filename='checkpoints/latest.pt'):
    # Run only by rank 0 as single source of truth for model state
    torch.save(state, filename)
    if is_best:
        # Save twice on 'best model occasion'
        shutil.copyfile(filename, 'checkpoints/best.pt')

def all_reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    # Sums up the tensor values from all GPUs
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    # Divide by world_size for averaged loss
    return rt / world_size

def train(rank, world_size):
    setup(rank, world_size)
    
    # Have devices not be completely identical (and not completely random either)
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)
    
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)
    
    loss_tracker = LossTracker() if rank == 0 else None

    train_dataset = COCOText(split="train")
    val_dataset = COCOText(split="validation")
    
    # Ensures each GPU gets different data
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    # Scale down batch size by device count
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size // world_size, sampler=train_sampler, num_workers=num_workers, collate_fn=COCOText.collate_fn, pin_memory=pin_memory, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size // world_size, sampler=val_sampler, num_workers=num_workers, collate_fn=COCOText.collate_fn, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    # Wrapping with DDP helps automatically sync gradients/grad descent across GPUs
    model = MobileYOLOv3(num_classes=num_classes, dropout_rate=dropout_rate, anchors=torch.tensor(anchors, dtype=torch.float32).to(device)).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False, broadcast_buffers=False)

    criterion = YoLoss().to(device)
    
    param_groups = [
        {'params': [], 'weight_decay': 0.0},          # no weight decay
        {'params': [], 'weight_decay': weight_decay}  # apply weight decay
    ]
    
    for name, param in model.named_parameters():
        if any(x in name for x in ['bias', 'bn']):
            param_groups[0]['params'].append(param)
        else:
            param_groups[1]['params'].append(param)
    
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4)
    scaler = torch.cuda.amp.GradScaler()
    
    # Work with checkpoints now, load if one exists (not doing this can quickly become expensive now)
    start_epoch = 0
    best_loss = float('inf')
    if os.path.exists('checkpoints/latest.pt'):
        try:
            checkpoint = torch.load('checkpoints/latest.pt', map_location=device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            if rank == 0 and loss_tracker is not None:
                loss_tracker.train_losses = checkpoint.get('train_losses', [])
                loss_tracker.val_losses = checkpoint.get('val_losses', [])
                loss_tracker.epochs = checkpoint.get('epochs', [])
        except Exception as e:
            print(f"Checkpoint loading failed\n{e}")
            start_epoch = 0
            best_loss = float('inf')

    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_loss = torch.zeros(1, device=device)
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        else:
            progress = train_loader
        
        for batch_idx, (images, targets) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mixed_images, targets_a, targets_b, lam = mixup_data(images, targets)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(mixed_images)
                loss_a = criterion(outputs, targets_a)
                loss_b = criterion(outputs, targets_b)
                loss = lam * loss_a + (1 - lam) * loss_b
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            if rank == 0 and batch_idx % 100 == 0:
                # Log loss and learning rate every 100 batches
                lr = scheduler.get_last_lr()[0]
                progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{lr:.6f}'
                })
        
        # Combines and averages losses from all GPUs
        epoch_loss = epoch_loss / len(train_loader)
        reduced_loss = all_reduce_tensor(epoch_loss, world_size)
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = torch.zeros(1, device=device)
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.detach()
            
            val_loss = val_loss / len(val_loader)
            reduced_val_loss = all_reduce_tensor(val_loss, world_size)
            
            # Only the main/0 process is checkpointing
            if rank == 0:
                print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {reduced_loss.item():.4f} | Val Loss: {reduced_val_loss.item():.4f}')
                
                # Update loss tracker and plot
                loss_tracker.update(epoch + 1, reduced_loss.item(), reduced_val_loss.item())
                loss_tracker.plot()
                
                # Save checkpoint
                is_best = reduced_val_loss.item() < best_loss
                best_loss = min(reduced_val_loss.item(), best_loss)
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_loss': best_loss,
                    'train_losses': loss_tracker.train_losses,
                    'val_losses': loss_tracker.val_losses,
                    'epochs': loss_tracker.epochs
                }, is_best)
        else:
            if rank == 0:
                # Update loss tracker with training loss only
                loss_tracker.update(epoch + 1, reduced_loss.item())
                loss_tracker.plot()
        # Synchronize devices
        dist.barrier()
    # Disintegrate with grace
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(0, 1)

if __name__ == "__main__":
    main()

# Should not yet work, but most important stuff is here.