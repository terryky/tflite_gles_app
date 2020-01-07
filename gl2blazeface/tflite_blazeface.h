/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_BLAZEFACE_H_
#define TFLITE_BLAZEFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_FACE_NUM  10

typedef struct fvec2
{
    float x, y;
} fvec2;
    
typedef struct _face_t
{
    fvec2 topleft;
    fvec2 btmright;
} face_t;

typedef struct _blazeface_result_t
{
    int num;
    face_t faces[MAX_FACE_NUM];
} blazeface_result_t;



extern int init_tflite_blazeface ();
extern void  *get_blazeface_input_buf (int *w, int *h);

extern int invoke_blazeface (blazeface_result_t *blazeface_result);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_BLAZEFACE_H_ */
