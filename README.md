# GPU accelerated TensorFlow Lite applications.
This repository contains several applications which invoke DNN inference with **TensorFlow Lite GPU Delegate** and visualizes its result with **OpenGLES**.

Target platform: Linux PC / NVIDIA Jetson / RaspberryPi.

## 1. Applications

### [gl2blazeface](https://github.com/terryky/tflite_gles_app/tree/master/gl2blazeface)
- Lightweight Face Detection.<br>
[<img src="gl2blazeface/gl2blazeface.png" width=500>](https://github.com/terryky/tflite_gles_app/tree/master/gl2blazeface)

### [gl2detection](https://github.com/terryky/tflite_gles_app/tree/master/gl2detection)
- Object Detection using MobileNet SSD.<br>
[<img src="gl2detection/gl2detection.png" width=500>](https://github.com/terryky/tflite_gles_app/tree/master/gl2detection)

### [gl2facemesh](https://github.com/terryky/tflite_gles_app/tree/master/gl2facemesh)
- 3D Facial Surface Geometry estimation and face replacement.<br>
[<img src="gl2facemesh/gl2facemesh.png" width=700>](https://github.com/terryky/tflite_gles_app/tree/master/gl2facemesh)

### [gl2hair_segmentation](https://github.com/terryky/tflite_gles_app/tree/master/gl2hair_segmentation)
- Hair segmentation and recoloring.<br>
[<img src="gl2hair_segmentation/gl2hair_segmentation.png" width=700>](https://github.com/terryky/tflite_gles_app/tree/master/gl2hair_segmentation)

### [gl2handpose](https://github.com/terryky/tflite_gles_app/tree/master/gl2handpose)
- 3D Handpose Estimation from single RGB images.<br>
[<img src="gl2handpose/gl2handpose.png" width=600>](https://github.com/terryky/tflite_gles_app/tree/master/gl2handpose)

### [gl2objectron](https://github.com/terryky/tflite_gles_app/tree/master/gl2objectron)
- 3D Object Detection.<br>
[<img src="gl2objectron/gl2objectron.png" width=300>](https://github.com/terryky/tflite_gles_app/tree/master/gl2objectron)

### [gl2posenet](https://github.com/terryky/tflite_gles_app/tree/master/gl2posenet)
- Pose Estimation.<br>
[<img src="gl2posenet/gl2posenet.png" width=500>](https://github.com/terryky/tflite_gles_app/tree/master/gl2posenet)

### [gl2segmentation](https://github.com/terryky/tflite_gles_app/tree/master/gl2segmentation)
- Semantic image segmentation using Deeplab.<br>
[<img src="gl2segmentation/gl2segmentation.png" width=600>](https://github.com/terryky/tflite_gles_app/tree/master/gl2segmentation)

### [gl2style_transfer](https://github.com/terryky/tflite_gles_app/tree/master/gl2style_transfer)
- Artistic Style Transfer.<br>
[<img src="gl2style_transfer/gl2style_transfer.png" width=600>](https://github.com/terryky/tflite_gles_app/tree/master/gl2style_transfer)




## 2. How to Build & Run

- [Build for x86_64 Linux](#build_for_x86_64)
- [Build for Jetson Nano (aarch64 )](#build_for_jetson_nano)
- [Build for Raspberry Pi 4 (armv7l)](#build_for_raspi4)


### <a name="build_for_x86_64">2.1. Build for x86_64 Linux</a>

##### 2.1.1. setup environment
```
$ sudo apt install libgles2-mesa-dev 
$
$ wget https://github.com/bazelbuild/bazel/releases/download/2.0.0/bazel-2.0.0-installer-linux-x86_64.sh
$ chmod 755 bazel-2.0.0-installer-linux-x86_64.sh
$ sudo ./bazel-2.0.0-installer-linux-x86_64.sh
```

##### 2.1.2. build TensorFlow Lite library.

```
$ cd ~/work 
$ git clone https://github.com/terryky/tflite_gles_app.git
$ ./tflite_gles_app/tools/scripts/tf2.2/build_libtflite_r2.2.sh

(Tensorflow configure will start after a while. Please enter according to your environment)

$
$ ln -s tensorflow_r2.2 ./tensorflow
```

##### 2.1.3. build an application.

```
$ cd ~/work/tflite_gles_app/gl2handpose
$ make -j4
```

##### 2.1.4. run an application.

```
$ cd ~/work/tflite_gles_app/gl2handpose
$ ./gl2handpose
```



### <a name="build_for_jetson_nano">2.2. Build for Jetson Nano (aarch64)</a>

##### 2.2.1. build TensorFlow Lite library on **Host PC**.

```
(HostPC)$ wget https://github.com/bazelbuild/bazel/releases/download/2.0.0/bazel-2.0.0-installer-linux-x86_64.sh
(HostPC)$ chmod 755 bazel-2.0.0-installer-linux-x86_64.sh
(HostPC)$ sudo ./bazel-2.0.0-installer-linux-x86_64.sh
(HostPC)$
(HostPC)$ cd ~/work 
(HostPC)$ git clone https://github.com/terryky/tflite_gles_app.git
(HostPC)$ ./tflite_gles_app/tools/scripts/tf2.2/build_libtflite_r2.2_with_gpu_delegate_aarch64.sh

(Tensorflow configure will start after a while. Please enter according to your environment)
```

##### 2.2.2. copy libtensorflow-lite.a to target Jetson.

```
(HostPC)scp ~/work/tensorflow_r2.2/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a jetson@192.168.11.11:/home/jetson/
```

##### 2.2.3. clone Tensorflow repository on **target Jetson**.

```
(Jetson)$ cd ~/work
(Jetson)$ git clone https://github.com/tensorflow/tensorflow.git
(Jetson)$ cd tensorflow
(Jetson)$ git checkout r2.2
(Jetson)$ ./tensorflow/lite/tools/make/download_dependencies.sh
```


##### 2.2.4. build an application.

```
(Jetson)$ cd ~/work 
(Jetson)$ git clone https://github.com/terryky/tflite_gles_app.git
(Jetson)$ cd ~/work/tflite_gles_app/gl2handpose
(Jetson)$ cp ~/libtensorflow-lite.a .
(Jetson)$ make -j4 TARGET_ENV=jetson_nano TFLITE_DELEGATE=GPU_DELEGATEV2
```

##### 2.2.5. run an application.

```
(Jetson)$ cd ~/work/tflite_gles_app/gl2handpose
(Jetson)$ ./gl2handpose
```

##### about VSYNC
On Jetson Nano, display sync to vblank (VSYNC) is enabled to avoid the tearing by default .
To enable/disable VSYNC, run app with the following command.
```
# enable VSYNC (default).
(Jetson)$ export __GL_SYNC_TO_VBLANK=1; ./gl2handpose

# disable VSYNC. framerate improves, but tearing occurs.
(Jetson)$ export __GL_SYNC_TO_VBLANK=0; ./gl2handpose
```


### <a name="build_for_raspi4">2.3 Build for Raspberry Pi 4 (armv7l)</a>

##### 2.3.1. build TensorFlow Lite library on **Host PC**.

```
(HostPC)$ wget https://github.com/bazelbuild/bazel/releases/download/2.0.0/bazel-2.0.0-installer-linux-x86_64.sh
(HostPC)$ chmod 755 bazel-2.0.0-installer-linux-x86_64.sh
(HostPC)$ sudo ./bazel-2.0.0-installer-linux-x86_64.sh
(HostPC)$
(HostPC)$ cd ~/work 
(HostPC)$ git clone https://github.com/terryky/tflite_gles_app.git
(HostPC)$ ./tflite_gles_app/tools/scripts/tf2.2/build_libtflite_r2.2_with_gpu_delegate_rpi.sh

(Tensorflow configure will start after a while. Please enter according to your environment)
```

##### 2.3.2. copy libtensorflow-lite.a to target Raspberry Pi 4.

```
(HostPC)scp ~/work/tensorflow_r2.2/tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a pi@192.168.11.11:/home/pi/
```

##### 2.3.3. setup environment on **Raspberry Pi 4**.

```
(Raspi)$ sudo apt install libgles2-mesa-dev libegl1-mesa-dev xorg-dev
(Raspi)$ sudo apt update
(Raspi)$ sudo apt upgrade
```


##### 2.3.4. clone Tensorflow repository on **target Raspi**.

```
(Raspi)$ cd ~/work
(Raspi)$ git clone https://github.com/tensorflow/tensorflow.git
(Raspi)$ cd tensorflow
(Raspi)$ git checkout r2.2
(Raspi)$ ./tensorflow/lite/tools/make/download_dependencies.sh
```


##### 2.3.5. build an application on **target Raspi**..

```
(Raspi)$ cd ~/work 
(Raspi)$ git clone https://github.com/terryky/tflite_gles_app.git
(Raspi)$ cd ~/work/tflite_gles_app/gl2handpose
(Raspi)$ cp ~/libtensorflow-lite.a .
(Raspi)$ make -j4 TARGET_ENV=raspi4  #disable GPUDelegate. (recommended)

#enable GPUDelegate. but it cause low performance on Raspi4.
(Raspi)$ make -j4 TARGET_ENV=raspi4 TFLITE_DELEGATE=GPU_DELEGATEV2
```


##### 2.3.6. run an application on **target Raspi**..

```
(Raspi)$ cd ~/work/tflite_gles_app/gl2handpose
(Raspi)$ ./gl2handpose
```


for more detail infomation, please refer [this article](https://qiita.com/terryky/items/fa18bd10cfead076b39f).



## 3. About Input video stream

Both **Live camera** and **video file** are supported as input methods.
- [Live UVC Camera](#uvc_camera)
- [Recorded Video file](#video_file)





### <a name="uvc_camera">3.1. Live UVC Camera (default)</a>


- UVC(USB Video Class) camera capture is supported. 

<img src="gl2handpose/gl2handpose_mov.gif" width="500">

- Use ```v4l2-ctl``` command to configure the capture resolution.

	- lower the resolution, higher the framerate.

```
(Target)$ sudo apt-get install v4l-utils

# confirm current resolution settings
(Target)$ v4l2-ctl --all

# query available resolutions
(Target)$ v4l2-ctl --list-formats-ext

# set capture resolution (160x120)
(Target)$ v4l2-ctl --set-fmt-video=width=160,height=120

# set capture resolution (640x480)
(Target)$ v4l2-ctl --set-fmt-video=width=640,height=480
```

- currently, only YUYV pixelformat is supported. 

	- If you have error messages like below:

```
-------------------------------
 capture_devie  : /dev/video0
 capture_devtype: V4L2_CAP_VIDEO_CAPTURE
 capture_buftype: V4L2_BUF_TYPE_VIDEO_CAPTURE
 capture_memtype: V4L2_MEMORY_MMAP
 WH(640, 480), 4CC(MJPG), bpl(0), size(341333)
-------------------------------
ERR: camera_capture.c(87): pixformat(MJPG) is not supported.
ERR: camera_capture.c(87): pixformat(MJPG) is not supported.
...
```

please try to change your camera settings to use YUYV pixelformat like following command :

```
$ sudo apt-get install v4l-utils
$ v4l2-ctl --set-fmt-video=width=640,height=480,pixelformat=YUYV --set-parm=30
```

- to disable camera
	- If your camera doesn't support YUYV, please run the apps in camera_disabled_mode with argument ```-x```


```
$ ./gl2handpose -x
```


### <a name="video_file">3.2 Recorded Video file</a>
- FFmpeg (libav) video decode is supported. 
- If you want to use a recorded video file instead of a live camera, follow these steps:

```
# setup dependent libralies.
(Target)$ sudo apt install libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavresample-dev libavutil-dev

# build an app with ENABLE_VDEC options
(Target)$ cd ~/work/tflite_gles_app/gl2facemesh
(Target)$ make -j4 ENABLE_VDEC=true

# run an app with a video file name as an argument.
(Target)$ ./gl2facemesh -v assets/sample_video.mp4
```





## 4. Tested platforms
You can select the platform by editing [Makefile.env](Makefile.env).
- Linux PC (X11)
- NVIDIA Jetson Nano (X11)
- NVIDIA Jetson TX2 (X11)
- RaspberryPi4 (X11)
- RaspberryPi3 (Dispmanx)
- Coral EdgeTPU Devboard (Wayland)
