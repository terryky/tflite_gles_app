/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_DETECT_H_
#define TFLITE_DETECT_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DETECT_CLASS 20


typedef struct _deeplab_result_t
{
    float *segmentmap;
    int   segmentmap_dims[3]; 
} deeplab_result_t;



extern int init_tflite_deeplab ();
extern void  *get_deeplab_input_buf (int *w, int *h);
extern float *get_deeplab_class_color (int class_idx);
extern char  *get_deeplab_class_name (int class_idx);
    
extern int invoke_deeplab (deeplab_result_t *deeplab_result);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_DETECT_H_ */
