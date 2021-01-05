/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef UTIL_RENDER_TARGET_H
#define UTIL_RENDER_TARGET_H


#define RTARGET_DEFAULT     (0 << 0)
#define RTARGET_COLOR       (1 << 0)
#define RTARGET_DEPTH       (1 << 1)

typedef struct _render_target_t
{
    GLuint texc_id; /* color */
    GLuint texz_id; /* depth */
    GLuint fbo_id;
    int width;
    int height;
} render_target_t;


#ifdef __cplusplus
extern "C" {
#endif


int create_render_target (render_target_t *rtarget, int w, int h, unsigned int flags);
int destroy_render_target (render_target_t *rtarget);
int set_render_target (render_target_t *rtarget);
int get_render_target (render_target_t *rtarget);
int blit_render_target (render_target_t *rtarget_src, int x, int y, int w, int h);

#ifdef __cplusplus
}
#endif
#endif /* UTIL_RENDER_TARGET_H */
