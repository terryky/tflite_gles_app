/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _RENDER_HANDPOSE_H_
#define _RENDER_HANDPOSE_H_
 
int init_cube (float aspect);
int draw_cube (float *mtxGlobal);

int draw_floor (float *mtxGlobal);
int draw_line (float *mtxGlobal, float *p0, float *p1);

#endif /* _RENDER_HANDPOSE_H_ */
 