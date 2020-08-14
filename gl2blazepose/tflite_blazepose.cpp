/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_blazepose.h"
#include <list>

/* 
 * https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_detection
 */
#define POSE_DETECT_MODEL_PATH        "./model/pose_detection.tflite"

static tflite_interpreter_t s_detect_interpreter;
static tflite_tensor_t      s_detect_tensor_input;
static tflite_tensor_t      s_detect_tensor_scores;
static tflite_tensor_t      s_detect_tensor_bboxes;

static std::list<fvec2> s_anchors;

/*
 * determine where the anchor points are scatterd.
 *   https://github.com/tensorflow/tfjs-models/blob/master/blazeface/src/face.ts
 */
static int
create_ssd_anchors(int input_w, int input_h)
{
    /* ANCHORS_CONFIG  */
    int strides[2] = {8, 16};
    int anchors[2] = {2,  6};

    int numtotal = 0;

    for (int i = 0; i < 2; i ++)
    {
        int stride = strides[i];
        int gridCols = (input_w + stride -1) / stride;
        int gridRows = (input_h + stride -1) / stride;
        int anchorNum = anchors[i];

        fvec2 anchor;
        for (int gridY = 0; gridY < gridRows; gridY ++)
        {
            anchor.y = stride * (gridY + 0.5f);
            for (int gridX = 0; gridX < gridCols; gridX ++)
            {
                anchor.x = stride * (gridX + 0.5f);
                for (int n = 0; n < anchorNum; n ++)
                {
                    s_anchors.push_back (anchor);
                    numtotal ++;
                }
            }
        }
    }
    return numtotal;
}



/* -------------------------------------------------- *
 *  Create TFLite Interpreter
 * -------------------------------------------------- */
int
init_tflite_blazepose(int use_quantized_tflite, blazepose_config_t *config)
{
    const char *detectpose_model;

    if (use_quantized_tflite)
    {
        detectpose_model = POSE_DETECT_MODEL_PATH;
    }
    else
    {
        detectpose_model = POSE_DETECT_MODEL_PATH;
    }

    /* Pose detect */
    tflite_create_interpreter_from_file (&s_detect_interpreter, detectpose_model);
    tflite_get_tensor_by_name (&s_detect_interpreter, 0, "input",          &s_detect_tensor_input);
    tflite_get_tensor_by_name (&s_detect_interpreter, 1, "regressors",     &s_detect_tensor_bboxes);
    tflite_get_tensor_by_name (&s_detect_interpreter, 1, "classificators", &s_detect_tensor_scores);

    int det_input_w = s_detect_tensor_input.dims[2];
    int det_input_h = s_detect_tensor_input.dims[1];
    create_ssd_anchors (det_input_w, det_input_h);

    config->score_thresh = 0.75f;
    config->iou_thresh   = 0.3f;

    return 0;
}

void *
get_pose_detect_input_buf (int *w, int *h)
{
    *w = s_detect_tensor_input.dims[2];
    *h = s_detect_tensor_input.dims[1];
    return s_detect_tensor_input.ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite (Pose detection)
 * -------------------------------------------------- */
static float *
get_bbox_ptr (int anchor_idx)
{
    /*
     *  cx, cy, width, height
     *  key0_x, key0_y
     *  key1_x, key1_y
     *  key2_x, key2_y
     *  key3_x, key3_y
     */
    int numkey = kPoseDetectKeyNum;
    int idx = (4 + 2 * numkey) * anchor_idx;
    float *bboxes_ptr = (float *)s_detect_tensor_bboxes.ptr;

    return &bboxes_ptr[idx];
}

static int
decode_bounds (std::list<detect_region_t> &region_list, float score_thresh, int input_img_w, int input_img_h)
{
    detect_region_t region;
    float  *scores_ptr = (float *)s_detect_tensor_scores.ptr;

    int i = 0;
    for (auto itr = s_anchors.begin(); itr != s_anchors.end(); i ++, itr ++)
    {
        fvec2 anchor = *itr;
        float score0 = scores_ptr[i];
        float score = 1.0f / (1.0f + exp(-score0));

        if (score > score_thresh)
        {
            float *p = get_bbox_ptr (i);

            /* boundary box */
            float sx = p[0];
            float sy = p[1];
            float w  = p[2];
            float h  = p[3];

            float cx = sx + anchor.x;
            float cy = sy + anchor.y;

            cx /= (float)input_img_w;
            cy /= (float)input_img_h;
            w  /= (float)input_img_w;
            h  /= (float)input_img_h;

            fvec2 topleft, btmright;
            topleft.x  = cx - w * 0.5f;
            topleft.y  = cy - h * 0.5f;
            btmright.x = cx + w * 0.5f;
            btmright.y = cy + h * 0.5f;

            region.score    = score;
            region.topleft  = topleft;
            region.btmright = btmright;

            /* landmark positions (6 keys) */
            for (int j = 0; j < kPoseDetectKeyNum; j ++)
            {
                float lx = p[4 + (2 * j) + 0];
                float ly = p[4 + (2 * j) + 1];
                lx += anchor.x;
                ly += anchor.y;
                lx /= (float)input_img_w;
                ly /= (float)input_img_h;

                region.keys[j].x = lx;
                region.keys[j].y = ly;
            }

            region_list.push_back (region);
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
non_max_suppression (std::list<detect_region_t> &region_list, std::list<detect_region_t> &region_sel_list, float iou_thresh)
{
    region_list.sort (compare);

    for (auto itr = region_list.begin(); itr != region_list.end(); itr ++)
    {
        detect_region_t region_candidate = *itr;

        int ignore_candidate = false;
        for (auto itr_sel = region_sel_list.rbegin(); itr_sel != region_sel_list.rend(); itr_sel ++)
        {
            detect_region_t region_sel = *itr_sel;

            float iou = calc_intersection_over_union (region_candidate, region_sel);
            if (iou >= iou_thresh)
            {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate)
        {
            region_sel_list.push_back(region_candidate);
            if (region_sel_list.size() >= MAX_POSE_NUM)
                break;
        }
    }

    return 0;
}

static void
pack_detect_result (pose_detect_result_t *detect_result, std::list<detect_region_t> &region_list)
{
    int num_regions = 0;
    for (auto itr = region_list.begin(); itr != region_list.end(); itr ++)
    {
        detect_region_t region = *itr;
        memcpy (&detect_result->poses[num_regions], &region, sizeof (region));
        num_regions ++;
        detect_result->num = num_regions;

        if (num_regions >= MAX_POSE_NUM)
            break;
    }
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_pose_detect (pose_detect_result_t *detect_result, blazepose_config_t *config)
{
    if (s_detect_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* decode boundary box and landmark keypoints */
    float score_thresh = config->score_thresh;
    std::list<detect_region_t> region_list;

    int input_img_w = s_detect_tensor_input.dims[2];
    int input_img_h = s_detect_tensor_input.dims[1];
    decode_bounds (region_list, score_thresh, input_img_w, input_img_h);


#if 1 /* USE NMS */
    float iou_thresh = config->iou_thresh;
    std::list<detect_region_t> region_nms_list;

    non_max_suppression (region_list, region_nms_list, iou_thresh);
    pack_detect_result (detect_result, region_nms_list);
#else
    pack_detect_result (detect_result, region_list);
#endif

    return 0;
}

