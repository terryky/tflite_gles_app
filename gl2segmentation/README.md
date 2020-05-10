# gl2segmentation
TensorflowLite Semantic image segmentation application.
 - Load JPEG file as source image for pose estimation.
 - Invoke Tensorflow Lite with [Deeplab](https://www.tensorflow.org/lite/models/segmentation/overview).
 - visualize the result of detection.

 ![capture image](gl2segmentation.png "capture image")

## Visualize Heatmap
To visualize the heatmap of each classes, edit just one line.

```
"main.c"

#if 1
render_deeplab_heatmap (draw_x, draw_y, draw_w, draw_h, &deeplab_ret);
#endif
```
