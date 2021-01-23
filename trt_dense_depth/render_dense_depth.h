/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _RENDER_HANDPOSE_H_
#define _RENDER_HANDPOSE_H_

typedef struct _mesh_obj_t
{
    float           *vtx_array;
    float           *uv_array;
    unsigned short  *idx_array;

    GLuint vbo_vtx;
    GLuint vbo_uv;
    GLuint vbo_idx;

    int num_tile_w;
    int num_tile_h;
    int num_idx;
} mesh_obj_t;

int init_cube (float aspect);
int draw_cube (float *mtxGlobal, float *color);
int draw_floor (float *mtxGlobal, float div_u, float div_v);
int draw_line (float *mtxGlobal, float *p0, float *p1, float *color);
int draw_triangle (float *mtxGlobal, float *p0, float *p1, float *p2, float *color);

int draw_point_arrays (float *mtxGlobal, float *vtx, float *uv, int num, int texid, float *color);

int create_mesh (mesh_obj_t *mobj, int num_tile_w, int num_tile_h);

#endif /* _RENDER_HANDPOSE_H_ */
 