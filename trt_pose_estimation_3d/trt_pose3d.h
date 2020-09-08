/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TRT_POSE3D_H_
#define TRT_POSE3D_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_POSE_NUM  10

enum pose_key_id {
    kNose = 0,          //  0
    kNeck,              //  1

    kRightShoulder,     //  2
    kRightElbow,        //  3
    kRightWrist,        //  4
    
    kLeftShoulder,      //  5
    kLeftElbow,         //  6
    kLeftWrist,         //  7

    kRightHip,          //  8
    kRightKnee,         //  9
    kRightAnkle,        // 10

    kLeftHip,           // 11
    kLeftKnee,          // 12
    kLeftAnkle,         // 13

    kLeftEye,           // 14
    kRightEye,          // 15
    kLeftEar,           // 16
    kRightEar,          // 17

    kPad,               // 18

    kPoseKeyNum
};

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct fvec3
{
    float x, y, z;
} fvec3;


typedef struct _pose_key_t
{
    float x;
    float y;
    float z;
    float score;
} pose_key_t;

typedef struct _pose_t
{
    pose_key_t key  [kPoseKeyNum];
    pose_key_t key3d[kPoseKeyNum];
    float pose_score;

    void *heatmap;
    int   heatmap_dims[2];  /* heatmap resolution. (9x9) */
} pose_t;

typedef struct _posenet_result_t
{
    int num;
    pose_t pose[MAX_POSE_NUM];
} posenet_result_t;



typedef struct _pose3d_config_t
{
    float score_thresh;
    float iou_thresh;
} pose3d_config_t;


int  init_trt_pose3d (pose3d_config_t *config);

void *get_pose3d_input_buf (int *w, int *h);
int invoke_pose3d (posenet_result_t *pose_result);

#ifdef __cplusplus
}
#endif

#endif /* TRT_POSE3D_H_ */
