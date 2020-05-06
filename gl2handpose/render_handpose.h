/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _RENDER_HANDPOSE_H_
#define _RENDER_HANDPOSE_H_

int init_cube (float aspect);
int draw_cube (float *mtxGlobal, float *color);
int draw_floor (float *mtxGlobal);
int draw_line (float *mtxGlobal, float *p0, float *p1, float *color);
int draw_triangle (float *mtxGlobal, float *p0, float *p1, float *p2, float *color);

#endif /* _RENDER_HANDPOSE_H_ */
 