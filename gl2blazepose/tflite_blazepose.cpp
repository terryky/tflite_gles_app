/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_blazepose.h"
#include "glue_mediapipe.h"
#include <list>

/* 
 * https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_detection
 * https://github.com/PINTO0309/PINTO_model_zoo/tree/master/53_BlazePose/03_integer_quantization
 */
#define POSE_DETECT_MODEL_PATH      "./model/pose_detection.tflite"
#define POSE_LANDMARK_MODEL_PATH    "./model/pose_landmark_upper_body.tflite"

#define POSE_DETECT_QUANT_MODEL_PATH    "./model/pose_detection_128x128_integer_quant.tflite"

static tflite_interpreter_t s_detect_interpreter;
static tflite_tensor_t      s_detect_tensor_input;
static tflite_tensor_t      s_detect_tensor_scores;
static tflite_tensor_t      s_detect_tensor_bboxes;

static tflite_interpreter_t s_landmark_interpreter;
static tflite_tensor_t      s_landmark_tensor_input;
static tflite_tensor_t      s_landmark_tensor_landmark;
static tflite_tensor_t      s_landmark_tensor_landmarkflag;

static std::vector<Anchor>  s_anchors;


static int
create_ssd_anchors(int input_w, int input_h)
{
    /*
     *  Anchor parameters are based on:
     *      mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
     */
    SsdAnchorsCalculatorOptions anchor_options;
    anchor_options.num_layers = 4;
    anchor_options.min_scale = 0.1484375;
    anchor_options.max_scale = 0.75;
    anchor_options.input_size_height = 128;
    anchor_options.input_size_width  = 128;
    anchor_options.anchor_offset_x  = 0.5f;
    anchor_options.anchor_offset_y  = 0.5f;
//  anchor_options.feature_map_width .push_back(0);
//  anchor_options.feature_map_height.push_back(0);
    anchor_options.strides.push_back( 8);
    anchor_options.strides.push_back(16);
    anchor_options.strides.push_back(16);
    anchor_options.strides.push_back(16);
    anchor_options.aspect_ratios.push_back(1.0);
    anchor_options.reduce_boxes_in_lowest_layer = false;
    anchor_options.interpolated_scale_aspect_ratio = 1.0;
    anchor_options.fixed_anchor_size = true;

    GenerateAnchors (&s_anchors, anchor_options);

    return 0;
}



/* -------------------------------------------------- *
 *  Create TFLite Interpreter
 * -------------------------------------------------- */
int
init_tflite_blazepose(int use_quantized_tflite, blazepose_config_t *config)
{
    const char *detectpose_model;
    const char *landmark_model;

    if (use_quantized_tflite)
    {
        detectpose_model = POSE_DETECT_QUANT_MODEL_PATH;
        landmark_model   = POSE_LANDMARK_MODEL_PATH;

        /* Pose detect */
        tflite_create_interpreter_from_file (&s_detect_interpreter, detectpose_model);
        tflite_get_tensor_by_name (&s_detect_interpreter, 0, "input",          &s_detect_tensor_input);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "Identity_1",     &s_detect_tensor_bboxes);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "Identity",       &s_detect_tensor_scores);
    }
    else
    {
        detectpose_model = POSE_DETECT_MODEL_PATH;
        landmark_model   = POSE_LANDMARK_MODEL_PATH;

        /* Pose detect */
        tflite_create_interpreter_from_file (&s_detect_interpreter, detectpose_model);
        tflite_get_tensor_by_name (&s_detect_interpreter, 0, "input",          &s_detect_tensor_input);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "regressors",     &s_detect_tensor_bboxes);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "classificators", &s_detect_tensor_scores);
    }

    /* Pose Landmark */
    tflite_create_interpreter_from_file (&s_landmark_interpreter, landmark_model);
    tflite_get_tensor_by_name (&s_landmark_interpreter, 0, "input_1",         &s_landmark_tensor_input);
    tflite_get_tensor_by_name (&s_landmark_interpreter, 1, "ld_3d",           &s_landmark_tensor_landmark);
    tflite_get_tensor_by_name (&s_landmark_interpreter, 1, "output_poseflag", &s_landmark_tensor_landmarkflag);

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


void *
get_pose_landmark_input_buf (int *w, int *h)
{
    *w = s_landmark_tensor_input.dims[2];
    *h = s_landmark_tensor_input.dims[1];
    return s_landmark_tensor_input.ptr;
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
        Anchor anchor = *itr;
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

            float cx = sx + anchor.x_center * input_img_w;
            float cy = sy + anchor.y_center * input_img_h;

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
                lx += anchor.x_center * input_img_w;
                ly += anchor.y_center * input_img_h;
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
 *  extract ROI
 *  based on:
 *   - mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
 *       AlignmentPointsRectsCalculator::DetectionToNormalizedRect()
 *   - mediapipe\calculators\util\rect_transformation_calculator.cc
 *       RectTransformationCalculator::TransformNormalizedRect()
 * -------------------------------------------------- */
static float
normalize_radians (float angle)
{
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

static void
compute_rotation (detect_region_t &region)
{
    float x0 = region.keys[kMidHipCenter].x;
    float y0 = region.keys[kMidHipCenter].y;
    float x1 = region.keys[kMidShoulderCenter].x;
    float y1 = region.keys[kMidShoulderCenter].y;

    float target_angle = M_PI * 0.5f;
    float rotation = target_angle - std::atan2(-(y1 - y0), x1 - x0);

    region.rotation = normalize_radians (rotation);
}

static void
rot_vec (fvec2 &vec, float rotation)
{
    float sx = vec.x;
    float sy = vec.y;
    vec.x = sx * std::cos(rotation) - sy * std::sin(rotation);
    vec.y = sx * std::sin(rotation) + sy * std::cos(rotation);
}

static void
compute_detect_to_roi (detect_region_t &region)
{
    int input_img_w = s_detect_tensor_input.dims[2];
    int input_img_h = s_detect_tensor_input.dims[1];
    float x_center = region.keys[kMidShoulderCenter].x * input_img_w;
    float y_center = region.keys[kMidShoulderCenter].y * input_img_h;
    float x_scale  = region.keys[kUpperBodySizeRot] .x * input_img_w;
    float y_scale  = region.keys[kUpperBodySizeRot] .y * input_img_h;

    // Bounding box size as double distance from center to scale point.
    float box_size = std::sqrt((x_scale - x_center) * (x_scale - x_center) +
                               (y_scale - y_center) * (y_scale - y_center)) * 2.0;

    /* RectTransformationCalculator::TransformNormalizedRect() */
    float width    = box_size;
    float height   = box_size;
    float rotation = region.rotation;
    float shift_x =  0.0f;
    float shift_y =  0.0f;
    float roi_cx;
    float roi_cy;

    if (rotation == 0.0f)
    {
        roi_cx = x_center + (width  * shift_x);
        roi_cy = y_center + (height * shift_y);
    }
    else
    {
        float dx = (width  * shift_x) * std::cos(rotation) -
                   (height * shift_y) * std::sin(rotation);
        float dy = (width  * shift_x) * std::sin(rotation) +
                   (height * shift_y) * std::cos(rotation);
        roi_cx = x_center + dx;
        roi_cy = y_center + dy;
    }

    /*
     *  calculate ROI width and height.
     *  scale parameter is based on
     *      "mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt"
     */
    float scale_x = 1.5f;
    float scale_y = 1.5f;
    float long_side = std::max (width, height);
    float roi_w = long_side * scale_x;
    float roi_h = long_side * scale_y;

    region.roi_center.x = roi_cx / (float)input_img_w;
    region.roi_center.y = roi_cy / (float)input_img_h;
    region.roi_size.x   = roi_w  / (float)input_img_w;
    region.roi_size.y   = roi_h  / (float)input_img_h;

    /* calculate ROI coordinates */
    float dx = roi_w * 0.5f;
    float dy = roi_h * 0.5f;
    region.roi_coord[0].x = - dx;  region.roi_coord[0].y = - dy;
    region.roi_coord[1].x = + dx;  region.roi_coord[1].y = - dy;
    region.roi_coord[2].x = + dx;  region.roi_coord[2].y = + dy;
    region.roi_coord[3].x = - dx;  region.roi_coord[3].y = + dy;

    for (int i = 0; i < 4; i ++)
    {
        rot_vec (region.roi_coord[i], rotation);
        region.roi_coord[i].x += roi_cx;
        region.roi_coord[i].y += roi_cy;

        region.roi_coord[i].x /= (float)input_img_h;
        region.roi_coord[i].y /= (float)input_img_h;
    }
}


static void
pack_detect_result (pose_detect_result_t *detect_result, std::list<detect_region_t> &region_list)
{
    int num_regions = 0;
    for (auto itr = region_list.begin(); itr != region_list.end(); itr ++)
    {
        detect_region_t region = *itr;

        compute_rotation (region);
        compute_detect_to_roi (region);

        memcpy (&detect_result->poses[num_regions], &region, sizeof (region));
        num_regions ++;
        detect_result->num = num_regions;

        if (num_regions >= MAX_POSE_NUM)
            break;
    }
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite (Pose detection)
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



/* -------------------------------------------------- *
 * Invoke TensorFlow Lite (Pose landmark)
 * -------------------------------------------------- */
int
invoke_pose_landmark (pose_landmark_result_t *landmark_result)
{
    if (s_landmark_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    float *poseflag_ptr = (float *)s_landmark_tensor_landmarkflag.ptr;
    float *landmark_ptr = (float *)s_landmark_tensor_landmark.ptr;
    int img_w = s_landmark_tensor_input.dims[2];
    int img_h = s_landmark_tensor_input.dims[1];

    landmark_result->score = *poseflag_ptr;
    for (int i = 0; i < POSE_JOINT_NUM; i ++)
    {
        landmark_result->joint[i].x = landmark_ptr[4 * i + 0] / (float)img_w;
        landmark_result->joint[i].y = landmark_ptr[4 * i + 1] / (float)img_h;
        landmark_result->joint[i].z = landmark_ptr[4 * i + 2];
        //fprintf (stderr, "[%2d] (%8.1f, %8.1f, %8.1f)\n", i,
        //    landmark_ptr[4 * i + 0], landmark_ptr[4 * i + 1], landmark_ptr[4 * i + 2]);
    }

    return 0;
}

