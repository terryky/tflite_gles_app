/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_FACE_SEGMENTATION_H_
#define TFLITE_FACE_SEGMENTATION_H_

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
    kLeftEar,       //  5

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

typedef struct _face_detect_result_t
{
    int num;
    face_t faces[MAX_FACE_NUM];
} face_detect_result_t;


typedef struct _bisenetv2_result_t
{
    int64_t *segmentmap;
    int   segmentmap_dims[3]; 
} bisenetv2_result_t;



int init_tflite_bisenetv2 (int use_quantized_tflite);

void *get_face_detect_input_buf (int *w, int *h);
int  invoke_face_detect (face_detect_result_t *facedet_result);

void  *get_bisenetv2_input_buf (int *w, int *h);
int invoke_bisenetv2 (bisenetv2_result_t *bisenetv2_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_FACE_SEGMENTATION_H_ */
