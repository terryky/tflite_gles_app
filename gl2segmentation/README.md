# gl2segmentation
Semantic image segmentation.
 - Invoke Tensorflow Lite with [Deeplab](https://www.tensorflow.org/lite/models/segmentation/overview).

 ![capture image](gl2segmentation.png "capture image")


#### stream input example

```
$  ./gl2segmentation -v assets/pexels_video.mp4
```
 ![capture image](gl2segmentation_mov.gif "capture image")



#### Visualize Heatmap
To visualize the heatmap of each classes, edit just one line.

```
"main.c"

#if 1
render_deeplab_heatmap (draw_x, draw_y, draw_w, draw_h, &deeplab_ret);
#endif
```
