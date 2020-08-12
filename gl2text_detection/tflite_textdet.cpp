/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_textdet.h"
#include <list>

/* 
 * https://tfhub.dev/sayakpaul/lite-model/east-text-detector/int8/1
 */
#define EAST_TEXTDET_MODEL_PATH        "./east_textdet_model/lite-model_east-text-detector_fp16_1.tflite"
#define EAST_TEXTDET_QUANT_MODEL_PATH  "./east_textdet_model/east_text_detection_320x320_integer_quant.tflite"
//#define EAST_TEXTDET_QUANT_MODEL_PATH  "./east_textdet_model/lite-model_east-text-detector_int8_1.tflite"

static tflite_interpreter_t s_detect_interpreter;
static tflite_tensor_t      s_detect_tensor_input;
static tflite_tensor_t      s_detect_tensor_scores;
static tflite_tensor_t      s_detect_tensor_geometry;
static tflite_tensor_t      s_detect_tensor_angle;




/* -------------------------------------------------- *
 *  Create TFLite Interpreter
 * -------------------------------------------------- */
int
init_tflite_textdet(int use_quantized_tflite, detect_config_t *config)
{
    const char *textdet_model;

    if (use_quantized_tflite)
    {
        textdet_model = EAST_TEXTDET_QUANT_MODEL_PATH;

        /* Angle and geometry are independent */
        tflite_create_interpreter_from_file (&s_detect_interpreter, textdet_model);
        tflite_get_tensor_by_name (&s_detect_interpreter, 0, "input_images",                  &s_detect_tensor_input);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "feature_fusion/Conv_7/Sigmoid", &s_detect_tensor_scores);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "feature_fusion/mul_6",          &s_detect_tensor_geometry);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "feature_fusion/mul_7",          &s_detect_tensor_angle);
    }
    else
    {
        textdet_model = EAST_TEXTDET_MODEL_PATH;

        /* Angle and geometry are concatinated */
        tflite_create_interpreter_from_file (&s_detect_interpreter, textdet_model);
        tflite_get_tensor_by_name (&s_detect_interpreter, 0, "input_images",                  &s_detect_tensor_input);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "feature_fusion/Conv_7/Sigmoid", &s_detect_tensor_scores);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "feature_fusion/concat_3",       &s_detect_tensor_geometry);
    }


    config->score_thresh = 0.75f;
    config->iou_thresh   = 0.3f;

    return 0;
}

void *
get_textdet_input_buf (int *w, int *h)
{
    *w = s_detect_tensor_input.dims[2];
    *h = s_detect_tensor_input.dims[1];
    return s_detect_tensor_input.ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite (Text detection)
 * -------------------------------------------------- */
static float *
get_geometry_ptr (int x, int y)
{
    int geom_w = s_detect_tensor_geometry.dims[2];
    int geom_c = s_detect_tensor_geometry.dims[3];
    int idx = (y * geom_w * geom_c) + (x * geom_c);
    float *geom_ptr = (float *)s_detect_tensor_geometry.ptr;

    return &geom_ptr[idx];
}

static float *
get_angle_ptr (int x, int y)
{
    int geom_c = s_detect_tensor_geometry.dims[3];

    /* concatinated geometry (geom[4] + angle[1]) */
    if (geom_c > 4)
    {
        int geom_w = s_detect_tensor_geometry.dims[2];
        int idx = (y * geom_w * geom_c) + (x * geom_c);
        float *geom_ptr = (float *)s_detect_tensor_geometry.ptr;

        return &geom_ptr[idx + 4];
    }
    /* angle independent of geometry */
    else
    {
        int angle_w = s_detect_tensor_angle.dims[2];
        int angle_c = s_detect_tensor_angle.dims[3];
        int idx = (y * angle_w * angle_c) + (x * angle_c);
        float *geom_ptr = (float *)s_detect_tensor_angle.ptr;
        return &geom_ptr[idx];
    }
}

/*
 * https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/EAST_TFLite.ipynb
 */
static int
decode_bounds (std::list<detect_region_t> &detect_list, float score_thresh, int input_img_w, int input_img_h)
{
    detect_region_t detect_item;
    float  *scores_ptr = (float *)s_detect_tensor_scores.ptr;
    float img_w = (float)s_detect_tensor_input.dims[2];
    float img_h = (float)s_detect_tensor_input.dims[1];
    int score_w = s_detect_tensor_scores.dims[2];
    int score_h = s_detect_tensor_scores.dims[1];

    for (int y = 0; y < score_h; y ++) 
    {
        for (int x = 0; x < score_w; x ++)
        {
            float score = scores_ptr[score_w * y + x];

            if (score < score_thresh)
                continue;

            float *geom_ptr  = get_geometry_ptr (x, y);
            float *angle_ptr = get_angle_ptr (x, y);

            float offset_x = x * 4;
            float offset_y = y * 4;
            float angle = angle_ptr[0];
            float h = geom_ptr[0] + geom_ptr[2];
            float w = geom_ptr[1] + geom_ptr[3];

            float end_x = offset_x + cos(angle) * geom_ptr[1] + sin(angle) * geom_ptr[2];
            float end_y = offset_y - sin(angle) * geom_ptr[1] + cos(angle) * geom_ptr[2];
            float start_x = end_x - w;
            float start_y = end_y - h;

            fvec2 topleft, btmright;
            topleft.x  = start_x / img_w;
            topleft.y  = start_y / img_h;
            btmright.x = end_x   / img_w;
            btmright.y = end_y   / img_h;

            detect_item.score    = score;
            detect_item.topleft  = topleft;
            detect_item.btmright = btmright;
            detect_item.angle    = angle;

            detect_list.push_back (detect_item);
        }
    }
    return 0;
}

/* -------------------------------------------------- *
 *  Apply NonMaxSuppression:
 *      https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/image_ops.ts
 * -------------------------------------------------- */
static float
calc_intersection_over_union (detect_region_t &region0, detect_region_t &region1)
{
    float sx0 = region0.topleft.x;
    float sy0 = region0.topleft.y;
    float ex0 = region0.btmright.x;
    float ey0 = region0.btmright.y;
    float sx1 = region1.topleft.x;
    float sy1 = region1.topleft.y;
    float ex1 = region1.btmright.x;
    float ey1 = region1.btmright.y;
    
    float xmin0 = std::min (sx0, ex0);
    float ymin0 = std::min (sy0, ey0);
    float xmax0 = std::max (sx0, ex0);
    float ymax0 = std::max (sy0, ey0);
    float xmin1 = std::min (sx1, ex1);
    float ymin1 = std::min (sy1, ey1);
    float xmax1 = std::max (sx1, ex1);
    float ymax1 = std::max (sy1, ey1);
    
    float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
    float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
    if (area0 <= 0 || area1 <= 0)
        return 0.0f;

    float intersect_xmin = std::max (xmin0, xmin1);
    float intersect_ymin = std::max (ymin0, ymin1);
    float intersect_xmax = std::min (xmax0, xmax1);
    float intersect_ymax = std::min (ymax0, ymax1);

    float intersect_area = std::max (intersect_ymax - intersect_ymin, 0.0f) *
                           std::max (intersect_xmax - intersect_xmin, 0.0f);
    
    return intersect_area / (area0 + area1 - intersect_area);
}

static bool
compare (detect_region_t &v1, detect_region_t &v2)
{
    if (v1.score > v2.score)
        return true;
    else
        return false;
}

static int
non_max_suppression (std::list<detect_region_t> &detect_list, std::list<detect_region_t> &detect_sel_list, float iou_thresh)
{
    detect_list.sort (compare);

    for (auto itr = detect_list.begin(); itr != detect_list.end(); itr ++)
    {
        detect_region_t detect_candidate = *itr;

        int ignore_candidate = false;
        for (auto itr_sel = detect_sel_list.rbegin(); itr_sel != detect_sel_list.rend(); itr_sel ++)
        {
            detect_region_t detect_sel = *itr_sel;

            float iou = calc_intersection_over_union (detect_candidate, detect_sel);
            if (iou >= iou_thresh)
            {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate)
        {
            detect_sel_list.push_back(detect_candidate);
            if (detect_sel_list.size() >= MAX_TEXT_NUM)
                break;
        }
    }

    return 0;
}

static void
pack_detect_result (detect_result_t *detect_result, std::list<detect_region_t> &detect_list)
{
    int num_detects = 0;
    for (auto itr = detect_list.begin(); itr != detect_list.end(); itr ++)
    {
        detect_region_t detect = *itr;
        memcpy (&detect_result->texts[num_detects], &detect, sizeof (detect));
        num_detects ++;
        detect_result->num = num_detects;

        if (num_detects >= MAX_TEXT_NUM)
            break;
    }
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_textdet (detect_result_t *detect_result, detect_config_t *config)
{
    if (s_detect_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* decode boundary box and landmark keypoints */
    float score_thresh = config->score_thresh;
    std::list<detect_region_t> detect_list;

    int input_img_w = s_detect_tensor_input.dims[2];
    int input_img_h = s_detect_tensor_input.dims[1];
    decode_bounds (detect_list, score_thresh, input_img_w, input_img_h);

#if 1 /* USE NMS */
    float iou_thresh = config->iou_thresh;
    std::list<detect_region_t> detect_nms_list;

    non_max_suppression (detect_list, detect_nms_list, iou_thresh);
    pack_detect_result (detect_result, detect_nms_list);
#else
    pack_detect_result (detect_result, detect_list);
#endif

    return 0;
}

