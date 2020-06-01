/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_HAIR_SEGMENTATION_H_
#define TFLITE_HAIR_SEGMENTATION_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_SEGMENT_CLASS 2


typedef struct _segmentation_result_t
{
    float *segmentmap;
    int   segmentmap_dims[3]; 
} segmentation_result_t;



extern int init_tflite_segmentation ();
extern void  *get_segmentation_input_buf (int *w, int *h);

extern int invoke_segmentation (segmentation_result_t *segment_result);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_HAIR_SEGMENTATION_H_ */
