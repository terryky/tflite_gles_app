/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef SSBO_TENSOR_H_
#define SSBO_TENSOR_H_


typedef struct _ssbo_t
{
    int width;              /* SSBO buffer size */
    int height;
    int active_width;       /* Tensor size */
    int active_height;
    int ssbo_id;
} ssbo_t;


extern ssbo_t *init_ssbo_tensor (int img_w, int img_h);
extern int resize_texture_to_ssbo (int texid, ssbo_t *ssbo);
extern int visualize_ssbo (ssbo_t *ssbo);


#endif
