/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef CAMERA_CAPTURE_H_
#define CAMERA_CAPTURE_H_

#include <stdint.h>

#define CAPTURE_SQUARED_CROP        (1 << 0)
#define CAPTURE_PIXFORMAT_RGBA      (1 << 1)

int init_capture (uint32_t flags);
int get_capture_dimension (int *width, int *height);
int get_capture_pixformat (uint32_t *pixformat);
int get_capture_buffer (void ** buf);

int start_capture ();


#endif
