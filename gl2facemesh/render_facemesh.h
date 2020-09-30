/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _RENDER_FACEMESH_H_
#define _RENDER_FACEMESH_H_

int init_cube (float aspect);
int draw_floor (float *mtxGlobal, float div_u, float div_v);

int init_facemesh_renderer (int w, int h);
int draw_facemesh_tri_tex (int texid, fvec3 *vtx, fvec3 *uv, float *color, int drill_eye_hole);
int draw_facemesh_line (fvec3 *joint, float *color, int drill_eye_hole);

#endif /* _RENDER_FACEMESH_H_ */
 