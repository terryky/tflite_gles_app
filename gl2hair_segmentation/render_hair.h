/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef RENDER_HAIR_H_
#define RENDER_HAIR_H_

int init_hair_renderer (int w, int h);
int draw_colored_hair (texture_2d_t *tex, int hair_texid, int x, int y, int w, int h, int upsidedown, float *color);



#endif /* RENDER_HAIR_H_ */
