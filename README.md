# mobileYOLOv3

mobileYOLOv3 is YOLOv3 via a MobileNetV2 backbone for text detection, pruned and quantized for deployment on mobile devices.

## Roadmap
- [x] Pretrained MobileNetV2 backbone
- [x] YOLOv3 top end
- [x] Basic Pruning, Quantization integration
- [x] Training pipeline (for ICDAR 2015)
- [ ] Basic Inference
- [ ] Deflate Jupyter Notebook into file structure
- [ ] Advanced training pipeline (COCO-Text dataset, batch augmentation, mixed precision, etc.)
- [ ] Advanced Pruning and quantization
- [ ] Live-Feed Processing

MobileNetv2 and YOLOv3 are used as foundation because they are well established, well documented and allow to focus on pipeline optimization.<br>
If the above steps are completed, the project will be bumped to MobileNetv3 and YOLOv4.

## References
1. [Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement.](https://arxiv.org/abs/1804.02767)
2. [Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.](https://arxiv.org/abs/1801.04381)
3. [Howard, A. G., et al. (2019). Searching for MobileNetV3.](https://arxiv.org/abs/1905.02244)
4. [Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection.](https://arxiv.org/abs/2004.10934)
5. [Terven, J., & Cordova-Esparza, D. (2023). A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS.](https://arxiv.org/abs/2304.00501v7)
6. [Wang, A., et al. (2024). YOLOv10: Real-Time End-to-End Object Detection.](https://arxiv.org/abs/2405.14458)
