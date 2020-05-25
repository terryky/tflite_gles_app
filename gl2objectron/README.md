# gl2objectron
3D Object Detection from single RGB images.
- This application use the pre-trained tflite model of [Google Mediapipe](https://github.com/google/mediapipe/tree/master/mediapipe/models).
- But this app directly call the TensorFlow Lite C++ api instead of  Mediapipe framework.


 ![capture image](gl2objectron_mov.gif "capture image")


## use int8 quantized tflite for better performance.
```
$ ./gl2objectron -q

