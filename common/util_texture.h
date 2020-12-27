/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TEXTURE_UTIL_H
#define TEXTURE_UTIL_H

#include <stdint.h>

#define pixfmt_fourcc(a, b, c, d)\
    ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))

typedef struct _texture_2d_t
{
    uint32_t    texid;
    int         width;
    int         height;
    uint32_t    format;
} texture_2d_t;


#ifdef __cplusplus
extern "C" {
#endif

int load_png_texture (char *name, int *lpTexID, int *width, int *height);
int load_jpg_texture (char *name, int *lpTexID, int *width, int *height);

uint32_t create_2d_texture (void *imgbuf, int width, int height);

int create_2d_texture_ex (texture_2d_t *tex2d, void *imgbuf, int w, int h, uint32_t fmt);

#if defined (USE_INPUT_CAMERA_CAPTURE)
int  create_capture_texture (texture_2d_t *captex);
void update_capture_texture (texture_2d_t *captex);
#endif

#if defined (USE_INPUT_VIDEO_DECODE)
int  create_video_texture (texture_2d_t *vidtex, const char *fname);
void update_video_texture (texture_2d_t *vidtex);
#endif

#ifdef __cplusplus
}
#endif
#endif /* TEXTURE_UTIL_H */
