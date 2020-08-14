/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_BLAZEPOSE_H_
#define TFLITE_BLAZEPOSE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_POSE_NUM  100
#define POSE_JOINT_NUM 25

enum pose_detect_key_id {
    kMidHipCenter = 0,      //  0
    kFullBodySizeRot,       //  1
    kMidShoulderCenter,     //  2
    kUpperBodySizeRot,      //  3

    kPoseDetectKeyNum
};

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct fvec3
{
    float x, y, z;
} fvec3;

typedef struct _detect_region_t
{
    float score;
    fvec2 topleft;
    fvec2 btmright;
    fvec2 keys[kPoseDetectKeyNum];

    float  rotation;
    float  detect_cx;
    float  detect_cy;
    float  detect_w;
    float  detect_h;
    fvec2  detect_pos[4];
} detect_region_t;

typedef struct _pose_detect_result_t
{
    int num;
    detect_region_t poses[MAX_POSE_NUM];
} pose_detect_result_t;

typedef struct _pose_landmark_result_t
{
    float score;
    fvec3 joint[POSE_JOINT_NUM];
} pose_landmark_result_t;


typedef struct _blazepose_config_t
{
    float score_thresh;
    float iou_thresh;
} blazepose_config_t;

int init_tflite_blazepose (int use_quantized_tflite, blazepose_config_t *config);

void *get_pose_detect_input_buf (int *w, int *h);
int  invoke_pose_detect (pose_detect_result_t *detect_result, blazepose_config_t *config);

void  *get_pose_landmark_input_buf (int *w, int *h);
int   invoke_pose_landmark (pose_landmark_result_t *pose_landmark_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_BLAZEPOSE_H_ */
