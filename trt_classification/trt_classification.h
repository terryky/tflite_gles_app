/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TRT_CLASSIFICATION_H_
#define TRT_CLASSIFICATION_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CLASS_NUM  1001

typedef struct _classify_t
{
    int     id;
    float   score;
    char    name[64];
} classify_t;

typedef struct _classification_result_t
{
    int num;
    classify_t classify[MAX_CLASS_NUM];
} classification_result_t;



extern int init_trt_classification ();
extern void  *get_classification_input_buf (int *w, int *h);

extern int invoke_classification (classification_result_t *class_result);
    
#ifdef __cplusplus
}
#endif

#endif /* TRT_CLASSIFICATION_H_ */
