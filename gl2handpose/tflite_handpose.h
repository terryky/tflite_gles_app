/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_HAND_LANDMARK_H_
#define TFLITE_HAND_LANDMARK_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_PALM_NUM   4
#define HAND_JOINT_NUM 21

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct fvec3
{
    float x, y, z;
} fvec3;

typedef struct rect_t
{
    fvec2 topleft;
    fvec2 btmright;
} rect_t;

typedef struct _palm_t
{
    float  score;
    rect_t rect;
    fvec2  keys[7];
    float  rotation;

    float  hand_cx;
    float  hand_cy;
    float  hand_w;
    float  hand_h;
    fvec2  hand_pos[4];
} palm_t;

typedef struct _palm_detection_result_t
{
    int num;
    palm_t palms[MAX_PALM_NUM];
} palm_detection_result_t;

typedef struct _hand_landmark_result_t
{
    float score;
    fvec3 joint[HAND_JOINT_NUM];
} hand_landmark_result_t;



typedef struct _pose3d_config_t
{
    float score_thresh;
    float iou_thresh;
} pose3d_config_t;

int   init_tflite_hand_landmark (int use_quantized_tflite);

void  *get_palm_detection_input_buf (int *w, int *h);
int   invoke_palm_detection (palm_detection_result_t *palm_result, int flag);

void  *get_hand_landmark_input_buf (int *w, int *h);
int   invoke_hand_landmark (hand_landmark_result_t *hand_landmark_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_HAND_LANDMARK_H_ */
