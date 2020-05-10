# GPU accelerated TensorFlow Lite applications.
This repository contains several applications which invoke DNN inference with **TensorFlow Lite with GPU Delegate** and visualizes its result with **OpenGLES**.

## applications
| App name    | Descriptions |
|:-----------:|:------------:|
| [gl2blazeface](https://github.com/terryky/tflite_gles_app/tree/master/gl2blazeface)| ![img](gl2blazeface/gl2blazeface.png " image") <br> lightweight face detection.|
| [gl2detection](https://github.com/terryky/tflite_gles_app/tree/master/gl2detection)| ![img](gl2detection/gl2detection.png " image") <br> Object Detection using MobileNet SSD.|
| [gl2handpose](https://github.com/terryky/tflite_gles_app/tree/master/gl2handpose)| ![img](gl2handpose/gl2handpose.png " image") <br> 3D Handpose Estimation from single RGB images.|
| [gl2posenet](https://github.com/terryky/tflite_gles_app/tree/master/gl2posenet)| ![img](gl2posenet/gl2posenet.png " image") <br> Pose Estimation.|
| [gl2segmentation](https://github.com/terryky/tflite_gles_app/tree/master/gl2segmentation)| ![img](gl2segmentation/gl2segmentation.png " image") <br> Semantic image segmentation using Deeplab.|
| [gl2style_transfer](https://github.com/terryky/tflite_gles_app/tree/master/gl2style_transfer)| ![img](gl2style_transfer/gl2style_transfer.png " image") <br> Artistic Style Transfer.|



## How to Build & Run

- [Build for x86_64 Linux](#build_for_x86_64)
- [Build for Jetson Nano](#build_for_jetson_nano)
- [Build for Raspberry Pi 4](#build_for_raspi4)


### <a name="build_for_x86_64">Build for x86_64 Linux</a>

##### 1.setup environment
```
$ sudo apt install libgles2-mesa-dev 
$
$ wget https://github.com/bazelbuild/bazel/releases/download/2.0.0/bazel-2.0.0-installer-linux-x86_64.sh
$ chmod 755 bazel-2.0.0-installer-linux-x86_64.sh
$ sudo ./bazel-2.0.0-installer-linux-x86_64.sh
```

##### 2.build TensorFlow Lite library.

```
$ cd ~/work 
$ git clone https://github.com/terryky/tflite_gles_app.git
$ ./tflite_gles_app/tools/scripts/tf2.2/build_libtflite_r2.2.sh

(Tensorflow configure will start after a while. Please enter according to your environment)
```

##### 3.build an application.

```
$ cd ~/work/tflite_gles_app/gl2handpose
$ make -j4
```

##### 4.run an application.

```
$ cd ~/work/tflite_gles_app/gl2handpose
$ ./gl2handpose
```



### <a name="build_for_jetson_nano">Build for Jetson Nano</a>

##### 1.build TensorFlow Lite library on **Host PC**.

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

##### 2.copy libtensorflow-lite.a to target Jetson.

```
(HostPC)scp ~/work/tensorflow_r2.2/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a jetson@192.168.11.11:/home/jetson/
```

##### 3.clone Tensorflow repository on **target Jetson**.

```
(Jetson)$ cd ~/work
(Jetson)$ git clone https://github.com/tensorflow/tensorflow.git
(Jetson)$ cd tensorflow
(Jetson)$ git checkout r2.2
(Jetson)$ ./tensorflow/lite/tools/make/download_dependencies.sh
```


##### 4.build an application.

```
(Jetson)$ cd ~/work 
(Jetson)$ git clone https://github.com/terryky/tflite_gles_app.git
(Jetson)$ cd ~/work/tflite_gles_app/gl2handpose
(Jetson)$ cp ~/libtensorflow-lite.a .
(Jetson)$ make -j4 TARGET_ENV=jetson_nano TFLITE_DELEGATE=GPU_DELEGATEV2
```

##### 5.run an application.

```
(Jetson)$ cd ~/work/tflite_gles_app/gl2handpose
(Jetson)$ ./gl2handpose
```



### <a name="build_for_raspi4">Build for Raspberry Pi 4</a>

##### 1.build TensorFlow Lite library on **Host PC**.

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

##### 2.copy libtensorflow-lite.a to target Raspberry Pi 4.

```
(HostPC)scp ~/work/tensorflow_r2.2/tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a pi@192.168.11.11:/home/pi/
```

##### 3.setup environment on **Raspberry Pi 4**.

```
(Raspi)$ sudo apt install libgles2-mesa-dev libegl1-mesa-dev xorg-dev
(Raspi)$ sudo apt update
(Raspi)$ sudo apt upgrade
```


##### 4.clone Tensorflow repository on **target Raspi**.

```
(Raspi)$ cd ~/work
(Raspi)$ git clone https://github.com/tensorflow/tensorflow.git
(Raspi)$ cd tensorflow
(Raspi)$ git checkout r2.2
(Raspi)$ ./tensorflow/lite/tools/make/download_dependencies.sh
```


##### 5.build an application on **target Raspi**..

```
(Raspi)$ cd ~/work 
(Raspi)$ git clone https://github.com/terryky/tflite_gles_app.git
(Raspi)$ cd ~/work/tflite_gles_app/gl2handpose
(Raspi)$ cp ~/libtensorflow-lite.a .
(Raspi)$ make -j4 TARGET_ENV=raspi4  #disable GPUDelegate. (recommended)

#enable GPUDelegate. but it cause low performance on Raspi4.
(Raspi)$ make -j4 TARGET_ENV=raspi4 TFLITE_DELEGATE=GPU_DELEGATEV2
```


##### 6.run an application on **target Raspi**..

```
(Raspi)$ cd ~/work/tflite_gles_app/gl2handpose
(Raspi)$ ./gl2handpose
```


for more detail infomation, please refer [this article](https://qiita.com/terryky/items/fa18bd10cfead076b39f).




## tested platforms
You can select the platform by editing [Makefile.env](Makefile.env).
- Linux PC (X11)
- NVIDIA Jetson Nano (X11)
- NVIDIA Jetson TX2 (X11)
- RaspberryPi4 (X11)
- RaspberryPi3 (Dispmanx)
- Coral EdgeTPU Devboard (Wayland)
