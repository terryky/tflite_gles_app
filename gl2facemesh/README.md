# gl2facemesh
3D Facial Surface Geometry Estimation and Face Replacement.
- This application use the pre-trained tflite model of [Google Mediapipe](https://github.com/google/mediapipe/tree/master/mediapipe/models).
- But this app directly call the TensorFlow Lite C++ api instead of  Mediapipe framework.

 ![capture image](gl2facemesh_mov.gif "capture image")

### To use a recorded video file instead of a live UVC camera

By default, this app uses a UVC camera for the input stream.
If you want to use a recorded video file instead of a live camera, follow these steps:

#### 1) edit Makefile
```
(before) ENABLE_VDEC := false
(after ) ENABLE_VDEC := true
```

#### 2) rebuild
```
$ sudo apt install libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavresample-dev libavutil-dev
$ make -j4
```

#### 3) run with options
```
$ ./gl2facemesh -e -v assets/sample_video.mp4
```


### video of running on Jetson Nano
[youtube](https://www.youtube.com/watch?v=QOTV5_6-Ycc)
