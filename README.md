# mobileYOLOv3

mobileYOLOv3: YOLOv3 via a MobileNetV3 backbone for text detection, pruned and quantized for deployment on mobile devices.

## Roadmap
- [x] Pretrained MobileNetV2 backbone
- [x] YOLOv3 top end
- [x] Basic Pruning, Quantization integration
- [x] Training pipeline (for ICDAR 2015)
- [x] Switch backbone to MobileNetV3
- [x] Mixed Precision Training
- [x] Advanced Pruning and quantization
- [ ] Advanced training pipeline (COCO-Text dataset, batch augmentation, etc.)
- [ ] Live Image-Feed Inference

## References
1. [Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement.](https://arxiv.org/abs/1804.02767)
2. [Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.](https://arxiv.org/abs/1801.04381)
3. [Howard, A. G., et al. (2019). Searching for MobileNetV3.](https://arxiv.org/abs/1905.02244)
4. [Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection.](https://arxiv.org/abs/2004.10934)
5. [Terven, J., & Cordova-Esparza, D. (2023). A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS.](https://arxiv.org/abs/2304.00501v7)
6. [Wang, A., et al. (2024). YOLOv10: Real-Time End-to-End Object Detection.](https://arxiv.org/abs/2405.14458)
7. [Ramachandran, P., et al. (2017). Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
8. [Zheng, Z., et al. (2019). Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)
9. [Yuan, D., et al. (2020). Accurate Bounding-box Regression with Distance-IoU Loss for Visual Tracking](https://arxiv.org/abs/2007.01864)
10. [Rezatofighi, H., et al. (2019). Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)
