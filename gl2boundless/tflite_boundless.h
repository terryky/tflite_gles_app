/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_MIRNET_H_
#define TFLITE_MIRNET_H_

#ifdef __cplusplus
extern "C" {
#endif


typedef struct _boundless_t
{
    int w, h;
    void *buf_mask;
    void *buf_gen;
} boundless_t;


int init_tflite_boundless (int use_quantized_tflite);
void  *get_boundless_input_buf (int *w, int *h);

int  invoke_boundless (boundless_t *boundless_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_MIRNET_H_ */
