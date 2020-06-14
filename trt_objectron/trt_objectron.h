/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TRT_CLASSIFICATION_H_
#define TRT_CLASSIFICATION_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_OBJECT_NUM    10

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct fvec3
{
    float x, y, z;
} fvec3;

typedef struct _object_t
{
    float belief;
    float center_x;
    float center_y;
    fvec2 bbox[8];

    fvec3 center3d;
    fvec3 bbox3d[8];
    fvec2 bbox2d[8];
} object_t;

typedef struct _objectron_result_t
{
    int num;
    object_t objects[MAX_OBJECT_NUM];
} objectron_result_t;


int  init_trt_objectron ();

void *get_objectron_input_buf (int *w, int *h);
int  invoke_objectron (objectron_result_t *objectron_result);

#ifdef __cplusplus
}
#endif

#endif /* TRT_CLASSIFICATION_H_ */
