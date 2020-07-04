/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_DETECT_H_
#define TFLITE_DETECT_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DETECT_OBJS     100
#define MAX_DETECT_CLASS    90
    
typedef struct _detect_obj_t
{
    float x1, x2, y1, y2;
    float score;
    int det_class;
} detect_obj_t;

typedef struct _detect_result_t
{
    int num;
    detect_obj_t obj[MAX_DETECT_OBJS];
} detect_result_t;



int   init_tflite_detection (int use_quantized_tflite);
int   get_detect_input_type ();
void  *get_detect_input_buf (int *w, int *h);
char  *get_detect_class_name (int class_idx);
float *get_detect_class_color (int class_idx);

int invoke_detect(detect_result_t *detection);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_DETECT_H_ */
