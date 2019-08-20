/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_IMAGE_TGA_H_
#define _UTIL_IMAGE_TGA_H_

#include <sys/types.h>

int open_tga (u_char *data, int size, unsigned int *w, unsigned int *h);
int decode_tga (u_char *data, int src_size, u_char *dst);

int open_tga_from_file (char *fname, u_int *w, u_int *h);
void decode_tga_from_file (char *fname, u_char *dst);

int save_to_tga_file (char *fname, u_char *src, int width, int height);

#endif /* _UTIL_IMAGE_TGA_H_ */
