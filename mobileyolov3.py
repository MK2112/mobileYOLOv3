import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class DSConv(nn.Module):
    # Depthwise Separable Convolution, i.e. Seperate, individually 'easier' convolutions for spatial and channel dimensions
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DSConv, self).__init__()
        # Depthwise (input channels)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=in_channels, bias=False)
        # Layer Pointwise (output channels) onto depthwise result
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Normalize network output using batch output's mean over batch output's stddev; scale + shift that learnably
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.softplus(x, beta=0.5)

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float)) + b) / gamma))
        self.conv = nn.Conv1d(1, 1, kernel_size=t, padding=(t - 1) // 2, bias=False)

    def forward(self, x):
        y = torch.mean(x, dim=(2, 3), keepdim=True)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)

class Resizer(nn.Module):
    def __init__(self, in_channel, out_channel, grid_size):
        super(Resizer, self).__init__()
        self.conv = DSConv(in_channel, out_channel, kernel_size=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(grid_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return F.softplus(x, beta=0.5)

class MobileYOLOv3(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3, anchors=None):
        super(MobileYOLOv3, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.dropout_rate = dropout_rate
        self.num_anchors = len(anchors)
        # Backbone
        self.backbone = nn.Sequential(*list(models.mobilenet_v3_small(weights='IMAGENET1K_V1', pretrained=True).features))
        # Coarse Detection Head
        self.conv_7 = DSConv(576, 1024)
        self.eca_7 = ECA(1024)
        self.dropout_7 = nn.Dropout2d(p=self.dropout_rate)
        self.det1 = nn.Conv2d(1024, self.num_anchors * (5 + num_classes), kernel_size=1)
        # Medium Detection Head
        self.r_1024_128 = Resizer(1024, 64, (14, 14))
        self.r_48_128 = Resizer(48, 64, (14, 14))
        self.conv_14 = DSConv(128, 512)
        self.eca_14 = ECA(512)
        self.dropout_14 = nn.Dropout2d(p=self.dropout_rate)
        self.det2 = nn.Conv2d(512, self.num_anchors * (5 + num_classes), kernel_size=1)
        # Fine Detection Head
        self.r_512_64 = Resizer(512, 64, (28, 28))
        self.r_24_64 = Resizer(24, 64, (28, 28))
        self.conv_28 = DSConv(128, 512)
        self.eca_28 = ECA(512)
        self.dropout_28 = nn.Dropout2d(p=self.dropout_rate)
        self.det3 = nn.Conv2d(512, self.num_anchors * (5 + num_classes), kernel_size=1)

    def _activate(self, det):
        # Reshape to (batch_size, grid_x, grid_y, num_anchors, 5 + num_classes)
        det = det.view(det.shape[0], det.shape[1], det.shape[2], self.num_anchors, 5 + self.num_classes)
        det[..., 0:2] = torch.sigmoid(det[..., 0:2]) # Sigmoid for x, y
        det[..., 2:4] = torch.sigmoid(det[..., 2:4]) * self.anchors # Sigmoid for w, h
        det[..., 4] = torch.sigmoid(det[..., 4])   # Objectness
        det[..., 5:] = torch.sigmoid(det[..., 5:]) # Class probabilities
        return det

    def forward(self, x):
        skip_out_14, skip_out_28 = None, None
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 3:
                skip_out_28 = x # (batch_size, 24, 28, 28)
            if i == 8:
                skip_out_14 = x # (batch_size, 48, 14, 14)
        
        # Coarse Detection 7x7
        x = self.conv_7(x) # (batch_size, 1024, 7, 7)
        x = self.eca_7(x)  # (batch_size, 1024, 7, 7)
        det1 = self.det1(x).permute(0, 2, 3, 1).contiguous() # (batch_size, 7, 7, num_anchors * (5 + num_classes))
    
        # Medium Detection 14x14
        x = torch.cat([self.r_1024_128(x), self.r_48_128(skip_out_14)], dim=1) # (batch_size, 128, 14, 14)
        x = self.conv_14(x) # (batch_size, 512, 14, 14)
        x = self.eca_14(x)  # (batch_size, 512, 14, 14)
        det2 = self.det2(x).permute(0, 2, 3, 1).contiguous() # (batch_size, 14, 14, num_anchors * (5 + num_classes))
        
        # Fine Detection 28x28
        x = torch.cat([self.r_512_64(x), self.r_24_64(skip_out_28)], dim=1) # (batch_size, 128, 28, 28)
        x = self.conv_28(x) # (batch_size, 512, 28, 28)
        x = self.eca_28(x)  # (batch_size, 512, 28, 28)
        det3 = self.det3(x).permute(0, 2, 3, 1).contiguous() # (batch_size, 28, 28, num_anchors * (5 + num_classes))
        
        return [self._activate(det1), self._activate(det2), self._activate(det3)]