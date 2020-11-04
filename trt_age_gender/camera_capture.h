/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef CAMERA_CAPTURE_H_
#define CAMERA_CAPTURE_H_

#include <stdint.h>

int init_capture ();
int get_capture_dimension (int *width, int *height);
int get_capture_pixformat (uint32_t *pixformat);
int get_capture_buffer (void ** buf);

int start_capture ();


#endif
