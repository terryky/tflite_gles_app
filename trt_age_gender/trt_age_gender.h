/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TRT_AGE_GENDER_H_
#define TRT_AGE_GENDER_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_FACE_NUM     10

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

    float rotation;
    float face_cx;
    float face_cy;
    float face_w;
    float face_h;
    fvec2 face_pos[4];
} face_t;

typedef struct _age_t
{
    int   age;
    float score;
} age_t;

typedef struct _gender_t
{
    float score_m;
    float score_f;
} gender_t;

typedef struct _face_detect_result_t
{
    int num;
    face_t faces[MAX_FACE_NUM];
} face_detect_result_t;

typedef struct _face_detect_config_t
{
    float score_thresh;
    float iou_thresh;
} face_detect_config_t;


typedef struct _age_gender_result_t
{
    age_t    age;
    gender_t gender;
} age_gender_result_t;



int init_trt_age_gender (face_detect_config_t *config);

void *get_face_detect_input_buf (int *w, int *h);
int  invoke_face_detect (face_detect_result_t *facedet_result, face_detect_config_t *config);

void  *get_age_gender_input_buf (int *w, int *h);
int invoke_age_gender (age_gender_result_t *age_gender_result);

#ifdef __cplusplus
}
#endif

#endif /* TRT_AGE_GENDER_H_ */
