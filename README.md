# gles_app
This repository contains several applications which invoke DNN inference with **TensorFlow Lite** and visualizes its result with **OpenGLES**.

### applications
| App name    | Descriptions |
|:-----------:|:------------:|
| gl2detection| ![img](gl2detection/gl2detection.png " image") <br> Object Detection using MobileNet SSD.|

### tested platforms
You can select the platform by editing [Makefile.env](Makefile.env).
- Linux PC (XWindow/Wayland/DRM)
- NVIDIA Jetson TX2 (XWindow/Wayland/EGLStream)
- RaspberryPi (Dispmanx)
