/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_IMAGE_PNG_H_
#define _UTIL_IMAGE_PNG_H_

#include <sys/types.h>
#include <png.h>

int open_png (u_char *data, int size, unsigned int *w, unsigned int *h, int *ctype);
int decode_png (u_char *src, int src_size, u_char *dst);

int  open_png_from_file (char *fname, u_int *w, u_int *h, int *ctype);
void decode_png_from_file (char *fname, u_char *dst);

#endif /* _UTIL_IMAGE_PNG_H_ */
