/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_DEEPLAB_H_
#define TFLITE_DEEPLAB_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DETECT_CLASS 20


typedef struct _deeplab_result_t
{
    float *segmentmap;
    int   segmentmap_dims[3];
} deeplab_result_t;



int   init_tflite_deeplab ();
int   get_deeplab_input_type ();
void  *get_deeplab_input_buf (int *w, int *h);
char  *get_deeplab_class_name (int class_idx);
float *get_deeplab_class_color (int class_idx);

int   invoke_deeplab (deeplab_result_t *deeplab_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_DEEPLAB_H_ */
