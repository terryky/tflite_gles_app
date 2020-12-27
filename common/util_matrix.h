/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _MATRIX_UTIL_H_
#define _MATRIX_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif


#ifndef M_PI
#define M_PI (3.141592654f)
#endif

#define DEG_TO_RAD(degree) ( (M_PI/180.0f) * (degree) )
#define RAD_TO_DEG(rad)    ( (rad) * (180.0f/M_PI) )


void matrix_translate (float *m, float x, float y, float z);
void matrix_rotate (float *m, float angle, float x, float y, float z);
void matrix_scale (float *m, float x, float y, float z);
void matrix_skew  (float *m, float x, float y);
void matrix_mult (float *m, float *m1, float *m2);
void matrix_identity (float *m);
void matrix_perspective (float *m, float depth);
void matrix_projectto2d (float *m);

void matrix_modellookat (float *m, float *src_pos, float *tgt_pos, float twist);

int  matrix_isidentity (float *m);
int  matrix_is2d (float *m);
int  matrix_is2d_scale_trans (float *m);

void matrix_multvec2 (float *m, float *svec, float *dvec);
void matrix_multvec4 (float *m, float *svec, float *dvec);

void matrix_print (float *m);

void matrix_copy( float *d, float *s );
void matrix_proj_frustum( float *mat, float left, float right, float bottom, float top, float znear, float zfar );
void matrix_proj_perspective( float *mat, float fovy, float aspect, float znear, float zfar );
void matrix_proj_ortho ( float *mat, float left, float right, float bottom, float top, float znear, float zfar );

void matrix_transpose (float *m);
void matrix_invert (float *m);

float vec3_length (float *v);
float vec3_normalize (float *v);

void quaternion_mult(float *lpR, float *lpP, float *lpQ);
void quaternion_to_matrix(float *lpM, float *lpQ);
void quaternion_rotate(float *lpQ, float rad, float ax, float ay, float az);
void quaternion_identity(float *lpQ);
void quaternion_copy (float *lpTo, float *lpFrom);

float vector_normalize(float *lpV);

#ifdef __cplusplus
}
#endif
#endif /* _MATRIX_UTIL_H_ */
