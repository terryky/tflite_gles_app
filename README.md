# gles_app
This repository contains several applications which invoke DNN inference with **TensorFlow Lite** and visualizes its result with **OpenGLES**.

### applications
| App name    | Descriptions |
|:-----------:|:------------:|
| [gl2blazeface](https://github.com/terryky/tflite_gles_app/tree/master/gl2blazeface)| ![img](gl2blazeface/gl2blazeface.png " image") <br> lightweight face detection.|
| [gl2detection](https://github.com/terryky/tflite_gles_app/tree/master/gl2detection)| ![img](gl2detection/gl2detection.png " image") <br> Object Detection using MobileNet SSD.|
| [gl2posenet](https://github.com/terryky/tflite_gles_app/tree/master/gl2posenet)| ![img](gl2posenet/gl2posenet.png " image") <br> Pose Estimation.|
| [gl2segmentation](https://github.com/terryky/tflite_gles_app/tree/master/gl2segmentation)| ![img](gl2segmentation/gl2segmentation.gif " image") <br> Semantic image segmentation using Deeplab.|
| [gl2style_transfer](https://github.com/terryky/tflite_gles_app/tree/master/gl2style_transfer)| ![img](gl2style_transfer/gl2style_transfer.png " image") <br> Artistic Style Transfer.|

### tested platforms
You can select the platform by editing [Makefile.env](Makefile.env).
- Linux PC (X11)
- NVIDIA Jetson Nano (X11)
- NVIDIA Jetson TX2 (X11)
- RaspberryPi4 (X11)
- RaspberryPi3 (Dispmanx)
- Coral EdgeTPU Devboard (Wayland)

### How to Build & Run
For Linux X11:
```
> sudo apt install libgles2-mesa-dev 
> make TARGET_ENV=x11
> cd gl2detection
> ./gl2detection
```

For Jetson Nano (native build on Jetson Nano)
```
> make TARGET_ENV=jetson_nano
> cd gl2detection
> ./gl2detection
```

For Raspberry Pi 4 (native build on RaaspberryPi)
```
> sudo apt install libgles2-mesa-dev 
> sudo apt-get install libdrm-dev
> make TARGET_ENV=raspi4
> cd gl2detection
> ./gl2detection
```
