/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _SHAPES_H_
#define _SHAPES_H_

#ifndef M_PI
#define M_PI (3.1415926535f)
#endif

#define SHAPE_TORUS			(1)
#define SHAPE_MOEBIUS		(2)
#define SHAPE_KLEINBOTTLE	(3)
#define SHAPE_BOYSURFACE	(4)
#define SHAPE_DINISURFACE	(5)
#define SHAPE_SPHERE        (6)
#define SHAPE_CYLINDER      (7)

typedef struct shape_obj_t
{
    GLuint  vbo_vtx;
    GLuint  vbo_col;
    GLuint  vbo_nrm;
    GLuint  vbo_uv;
    GLuint  vbo_tng;
    GLuint  vbo_idx;
    int     num_faces;
} shape_obj_t;

int
shape_create (int type, int nDivU, int nDivV, shape_obj_t *shape);


#endif /* _SHAPES_H_ */
