/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TRT_DBFACE_H_
#define TRT_DBFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_FACE_NUM  100

enum face_key_id {
    kRightEye = 0,  //  0
    kLeftEye,       //  1
    kNose,          //  2
    kMouth,         //  3
    kRightEar,      //  4

    kFaceKeyNum
};

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct _face_t
{
    float score;
    fvec2 topleft;
    fvec2 btmright;
    fvec2 keys[kFaceKeyNum];
} face_t;

typedef struct _dbface_result_t
{
    int num;
    face_t faces[MAX_FACE_NUM];
} dbface_result_t;

typedef struct _dbface_config_t
{
    float score_thresh;
    float iou_thresh;
} dbface_config_t;


int  init_trt_dbface (dbface_config_t *config);

void *get_dbface_input_buf (int *w, int *h);
int invoke_dbface (dbface_result_t *face_result, dbface_config_t *config);

#ifdef __cplusplus
}
#endif

#endif /* TRT_DBFACE_H_ */
