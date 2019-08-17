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



extern int init_tflite_detection();
extern void  *get_detect_src_buf ();
extern char  *get_detect_class_name (int class_idx);
extern float *get_detect_class_color (int class_idx);

extern int invoke_detect(detect_result_t *detection);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_DETECT_H_ */
