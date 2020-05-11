/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _RENDER_FACEMESH_H_
#define _RENDER_FACEMESH_H_

int init_cube (float aspect);
int draw_cube (float *mtxGlobal, float *color);
int draw_floor (float *mtxGlobal);
int draw_line (float *mtxGlobal, float *p0, float *p1, float *color);
int draw_triangle (float *mtxGlobal, float *p0, float *p1, float *p2, float *color);
int draw_bone (float *mtxGlobal, float *p0, float *p1, float radius, float *color);
int draw_sphere (float *mtxGlobal, float *p0, float radius, float *color);

int init_face_2d_renderer (int w, int h);
int draw_tri_tex_indexed (int texid, float *vtx, float *uv, int *idx, int counts);

#endif /* _RENDER_FACEMESH_H_ */
 