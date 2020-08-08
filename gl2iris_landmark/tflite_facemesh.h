/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_HAND_LANDMARK_H_
#define TFLITE_HAND_LANDMARK_H_

#ifdef __cplusplus
extern "C" {
#endif


#define MAX_FACE_NUM    10
#define FACE_KEY_NUM    468

enum face_key_id {
    kRightEye = 0,  //  0
    kLeftEye,       //  1
    kNose,          //  2
    kMouth,         //  3
    kRightEar,      //  4
    kLeftEar,       //  5

    kFaceKeyNum
};

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct fvec3
{
    float x, y, z;
} fvec3;

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

typedef struct _face_detect_result_t
{
    int num;
    face_t faces[MAX_FACE_NUM];
} face_detect_result_t;



typedef struct rect_t
{
    fvec2 topleft;
    fvec2 btmright;
} rect_t;

typedef struct _eye_region_t
{
    float rotation;
    fvec2 center;
    fvec2 size;
} eye_region_t;

typedef struct _face_landmark_result_t
{
    float score;
    fvec3 joint[FACE_KEY_NUM];

    eye_region_t    eye_rgn[2];
    fvec2           eye_pos[2][4];
} face_landmark_result_t;


typedef struct _irismesh_result_t
{
    fvec3 eye_landmark[71];
    fvec3 iris_landmark[5];
} irismesh_result_t;


int  init_tflite_facemesh (int use_quantized_tflite);

void *get_face_detect_input_buf (int *w, int *h);
int  invoke_face_detect (face_detect_result_t *facedet_result);

void *get_facemesh_landmark_input_buf (int *w, int *h);
int  invoke_facemesh_landmark (face_landmark_result_t *facemesh_result);

void *get_irismesh_landmark_input_buf (int *w, int *h);
int  invoke_irismesh_landmark (irismesh_result_t *eyemesh_result);

int
get_static_facemesh_landmark (face_detect_result_t   *facedet_result,
                              face_landmark_result_t *facemesh_result);

int *get_facemesh_tri_indicies (int *num_tris, int flags);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_HAND_LANDMARK_H_ */
