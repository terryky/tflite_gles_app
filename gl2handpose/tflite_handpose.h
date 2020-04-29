/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_HAND_LANDMARK_H_
#define TFLITE_HAND_LANDMARK_H_

#ifdef __cplusplus
extern "C" {
#endif

#define HAND_JOINT_NUM 21

typedef struct fvec3
{
    float x, y, z;
} fvec3;


typedef struct _hand_landmark_result_t
{
    float score;
    fvec3 joint[HAND_JOINT_NUM];
} hand_landmark_result_t;



extern int init_tflite_hand_landmark ();
extern void  *get_hand_landmark_input_buf (int *w, int *h);
extern int invoke_hand_landmark (hand_landmark_result_t *hand_landmark_result);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_HAND_LANDMARK_H_ */
