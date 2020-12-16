/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_MIRNET_H_
#define TFLITE_MIRNET_H_

#ifdef __cplusplus
extern "C" {
#endif


typedef struct _mirnet_t
{
    int w, h;
    void *param;
} mirnet_t;


int init_tflite_mirnet (int use_quantized_tflite);
void  *get_mirnet_input_buf (int *w, int *h);

int  invoke_mirnet (mirnet_t *mirnet_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_MIRNET_H_ */
