/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TRT_DENSE_DEPTH_H_
#define TRT_DENSE_DEPTH_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _dense_depth_result_t
{
    float *depthmap;
    int   depthmap_dims[3];
} dense_depth_result_t;

int init_trt_dense_depth ();

void  *get_dense_depth_input_buf (int *w, int *h);
int invoke_dense_depth (dense_depth_result_t *dense_depth_result);

#ifdef __cplusplus
}
#endif

#endif /* TRT_DENSE_DEPTH_H_ */
