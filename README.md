# mobileYOLOv3

YOLOv3 via a MobileNetV3 backbone for text detection; pruned, quantized, optimized, and explained for deployment on mobile devices. Primarily intended as a single source for learning about YOLO(v3) in an applied manner.

## Roadmap
- [x] Pretrained MobileNetV2 backbone
- [x] Introduce the YOLOv3 paradigm
- [x] Basic Pruning, Quantization integration
- [x] Training pipeline (for ICDAR 2015)
- [x] Switch backbone to MobileNetV3
- [x] Mixed Precision Training
- [x] Pruning and quantization
- [x] Add textbook-style explanations for YOLOv3

## References

- [\[1\]](https://arxiv.org/abs/1804.02767) YOLOv3 - Farhadi, A., & Redmon, J. (2018, June). Yolov3: An incremental improvement. In Computer vision and pattern recognition (Vol. 1804, pp. 1-6). Berlin/Heidelberg, Germany: Springer.
- [\[2\]](https://www.kaggle.com/datasets/bestofbests9/icdar2015) ICDAR 2015 - Kaggle.com
- [\[3\]](https://link.springer.com/content/pdf/10.1007/s11036-020-01650-z.pdf) Mobile App Use Cases - Sarker, I. H., Hoque, M. M., Uddin, M. K., & Alsanoosy, T. (2021). Mobile data science and intelligent apps: concepts, AI-based modeling and research directions. Mobile Networks and Applications, 26(1), 285-303.
- [\[4\]](https://arxiv.org/abs/1506.01497) R-CNN - Ren, S., He, K., Girshick, R., & Sun, J. (2016). Faster R-CNN: Towards real-time object detection with region proposal networks. IEEE transactions on pattern analysis and machine intelligence, 39(6), 1137-1149.
- [\[5\]](http://vision.stanford.edu/teaching/cs231b_spring1213/papers/CVPR05_DalalTriggs.pdf) Sliding Window Detectors - Dalal, N., & Triggs, B. (2005, June). Histograms of oriented gradients for human detection. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR'05) (Vol. 1, pp. 886-893). Ieee.
- [\[6\]](https://arxiv.org/abs/1911.08287) CIoU - Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020, April). Distance-IoU loss: Faster and better learning for bounding box regression. In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, No. 07, pp. 12993-13000).
- [\[7\]](https://arxiv.org/abs/1708.02002) Focal Loss - Ross, T. Y., & Dollár, G. K. H. P. (2017, July). Focal loss for dense object detection. In proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2980-2988).
- [\[8\]](https://arxiv.org/abs/2003.02819) Label Smoothing - Lukasik, M., Bhojanapalli, S., Menon, A., & Kumar, S. (2020, November). Does label smoothing mitigate label noise?. In International Conference on Machine Learning (pp. 6448-6458). PMLR.
- [\[9\]](https://arxiv.org/abs/1710.05941) Activation Functions - Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for activation functions. arXiv preprint arXiv:1710.05941.
- [\[10\]](https://arxiv.org/abs/1905.02244) MobileNetV3 - Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for mobilenetv3. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 1314-1324).
- [\[11\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf) ECA - Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., & Hu, Q. (2020). ECA-Net: Efficient channel attention for deep convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 11534-11542).
- [\[12\]](https://arxiv.org/abs/1610.02357) DConv - Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1251-1258).
- [\[13\]](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) Dropout - Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1), 1929-1958.
- [\[14\]](https://openreview.net/forum?id=r1Ddp1-Rb) Zhang, H. (2017). mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412.
- [\[15\]](https://arxiv.org/abs/1708.07120) OneCycleLR - Smith, L. N., & Topin, N. (2019, May). Super-convergence: Very fast training of neural networks using large learning rates. In Artificial intelligence and machine learning for multi-domain operations applications (Vol. 11006, pp. 369-386). SPIE.
- [\[16\]](https://arxiv.org/abs/1907.08610) Lookahead - Zhang, M., Lucas, J., Ba, J., & Hinton, G. E. (2019). Lookahead optimizer: k steps forward, 1 step back. Advances in neural information processing systems, 32.
- [\[17\]](https://arxiv.org/abs/2011.00241) Pruning - Vadera, S., & Ameen, S. (2022). Methods for pruning deep neural networks. IEEE Access, 10, 63280-63300.
- [\[18\]](https://arxiv.org/abs/1510.00149) Quantization - Han, S., Mao, H., & Dally, W. J. (2015). Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149.
