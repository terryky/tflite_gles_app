/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_EAST_TEXTDET_H_
#define TFLITE_EAST_TEXTDET_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_TEXT_NUM  100

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct _detect_region_t
{
    float score;
    fvec2 topleft;
    fvec2 btmright;
    float angle;
} detect_region_t;

typedef struct _detect_result_t
{
    int num;
    detect_region_t texts[MAX_TEXT_NUM];
} detect_result_t;

typedef struct _detect_config_t
{
    float score_thresh;
    float iou_thresh;
} detect_config_t;

extern int init_tflite_textdet (int use_quantized_tflite, detect_config_t *config);
extern void  *get_textdet_input_buf (int *w, int *h);

extern int invoke_textdet (detect_result_t *detect_result, detect_config_t *config);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_EAST_TEXTDET_H_ */
