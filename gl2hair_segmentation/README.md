# gl2hair_segmentation
Hair segmentation and recoloring.
- This application use the pre-trained tflite model of [Google Mediapipe](https://github.com/google/mediapipe/tree/master/mediapipe/models).
- But this app directly call the TensorFlow Lite C++ api instead of  Mediapipe framework.

 ![capture image](gl2hair_segmentation.jpg "capture image")


#### stream input example

```
$ ./gl2hair_segmentation -v assets/pexels_person.mp4
```
 ![capture image](gl2hair_segmentation_mov.gif "capture image")
