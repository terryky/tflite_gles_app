/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_IMAGE_JPG_H_
#define _UTIL_IMAGE_JPG_H_

#include <sys/types.h>

int open_jpeg (u_char *data, int size, unsigned int *w, unsigned int *h);
int decode_jpeg (u_char *data, int src_size, u_char *dst);

int open_jpeg_from_file (char *fname, u_int *w, u_int *h);
void decode_jpeg_from_file (char *fname, u_char *dst);

#endif /* _UTIL_IMAGE_JPG_H_ */
