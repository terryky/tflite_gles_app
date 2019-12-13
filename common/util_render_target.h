/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_RENDER_TARGET_H_
#define _UTIL_RENDER_TARGET_H_

typedef struct render_target_t
{
  int texid;
  int fboid;
  int width;
  int height;
} render_target_t;

#define RENDER_TARGET_COLOR (1 << 0)
#define RENDER_TARGET_DEPTH (1 << 1)

int create_render_target (render_target_t *rtarget, int w, int h, unsigned int flag);
int destroy_render_target (render_target_t *rtarget);
int set_render_target (render_target_t *rtarget);

#endif /* _UTIL_RENDER_TARGET_H_ */
