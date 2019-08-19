# gles_app
This repository contains several applications which invoke DNN inference with **TensorFlow Lite** and visualizes its result with **OpenGLES**.

### applications
| App name    | Descriptions |
|:-----------:|:------------:|
| [gl2detection](https://github.com/terryky/tflite_gles_app/tree/master/gl2detection)| ![img](gl2detection/gl2detection.png " image") <br> Object Detection using MobileNet SSD.|
| [gl2posenet](https://github.com/terryky/tflite_gles_app/tree/master/gl2posenet)| ![img](gl2posenet/gl2posenet.png " image") <br> Pose Estimation.|
| [gl2segmentation](https://github.com/terryky/tflite_gles_app/tree/master/gl2segmentation)| ![img](gl2segmentation/gl2segmentation.gif " image") <br> Semantic image segmentation using Deeplab.|

### tested platforms
You can select the platform by editing [Makefile.env](Makefile.env).
- Linux PC (XWindow/Wayland/DRM)
- NVIDIA Jetson TX2 (XWindow/Wayland/EGLStream)
- RaspberryPi (Dispmanx)
