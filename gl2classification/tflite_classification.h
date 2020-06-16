/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_CLASSIFICATION_H_
#define TFLITE_CLASSIFICATION_H_

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



int   init_tflite_classification (int use_quantized_tflite);
int   get_classification_input_type ();
void  *get_classification_input_buf (int *w, int *h);

int   invoke_classification (classification_result_t *class_result);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_CLASSIFICATION_H_ */
