/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_handpose.h"
#include "custom_ops/transpose_conv_bias.h"
#include <list>

/* 
 * https://github.com/google/mediapipe/tree/master/mediapipe/models/hand_landmark_3d.tflite
 */
#define PALM_DETECTION_MODEL_PATH  "./handpose_model/palm_detection.tflite"
#define HAND_LANDMARK_MODEL_PATH   "./handpose_model/hand_landmark_3d.tflite"

static tflite_interpreter_t s_palm_interpreter;
static tflite_tensor_t      s_palm_tensor_input;
static tflite_tensor_t      s_palm_tensor_scores;
static tflite_tensor_t      s_palm_tensor_points;

static tflite_interpreter_t s_hand_interpreter;
static tflite_tensor_t      s_hand_tensor_input;
static tflite_tensor_t      s_hand_tensor_landmark;
static tflite_tensor_t      s_hand_tensor_handflag;


typedef struct Anchor
{
    float x_center, y_center, w, h;
} Anchor;

static std::vector<Anchor>  s_anchors;

typedef struct SsdAnchorsCalculatorOptions 
{
    int input_size_width;
    int input_size_height;
    float min_scale;
    float max_scale;
    float anchor_offset_x;
    float anchor_offset_y;

    int num_layers;
    std::vector<int> feature_map_width;
    std::vector<int> feature_map_height;

    std::vector<int>   strides;
    std::vector<float> aspect_ratios;
    bool reduce_boxes_in_lowest_layer;
    float interpolated_scale_aspect_ratio;
    bool fixed_anchor_size;

} SsdAnchorsCalculatorOptions;



/* ---------------------------------------------------------------------- *
 *   mediapipe/calculators/tflite/ssd_anchors_calculator_test.cc
 * ---------------------------------------------------------------------- */
 
// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


float
CalculateScale(float min_scale, float max_scale, int stride_index, int num_strides) 
{
    return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
}


int
GenerateAnchors(std::vector<Anchor>* anchors, const SsdAnchorsCalculatorOptions& options)
{
    int layer_id = 0;
    while (layer_id < (int)options.strides.size()) {
        std::vector<float> anchor_height;
        std::vector<float> anchor_width;
        std::vector<float> aspect_ratios;
        std::vector<float> scales;

        // For same strides, we merge the anchors in the same order.
        int last_same_stride_layer = layer_id;
        while (last_same_stride_layer < (int)options.strides.size() &&
               options.strides[last_same_stride_layer] == options.strides[layer_id]) 
        {
          const float scale =
              CalculateScale(options.min_scale, options.max_scale,
                last_same_stride_layer, options.strides.size());
          if (last_same_stride_layer == 0 && options.reduce_boxes_in_lowest_layer) {
            // For first layer, it can be specified to use predefined anchors.
            aspect_ratios.push_back(1.0);
            aspect_ratios.push_back(2.0);
            aspect_ratios.push_back(0.5);
            scales.push_back(0.1);
            scales.push_back(scale);
            scales.push_back(scale);
          } else {
            for (int aspect_ratio_id = 0;
                aspect_ratio_id < (int)options.aspect_ratios.size();
                 ++aspect_ratio_id) {
              aspect_ratios.push_back(options.aspect_ratios[aspect_ratio_id]);
              scales.push_back(scale);
            }
            if (options.interpolated_scale_aspect_ratio > 0.0) {
              const float scale_next =
                last_same_stride_layer == (int)options.strides.size() - 1
                      ? 1.0f
                      : CalculateScale(options.min_scale, options.max_scale,
                                       last_same_stride_layer + 1,
                                       options.strides.size());
              scales.push_back(std::sqrt(scale * scale_next));
              aspect_ratios.push_back(options.interpolated_scale_aspect_ratio);
            }
          }
          last_same_stride_layer++;
        }

        for (int i = 0; i < (int)aspect_ratios.size(); ++i) {
          const float ratio_sqrts = std::sqrt(aspect_ratios[i]);
          anchor_height.push_back(scales[i] / ratio_sqrts);
          anchor_width .push_back(scales[i] * ratio_sqrts);
        }

        int feature_map_height = 0;
        int feature_map_width  = 0;
        if (options.feature_map_height.size()) {
          feature_map_height = options.feature_map_height[layer_id];
          feature_map_width  = options.feature_map_width [layer_id];
        } else {
          const int stride = options.strides[layer_id];
          feature_map_height = std::ceil(1.0f * options.input_size_height / stride);
          feature_map_width  = std::ceil(1.0f * options.input_size_width  / stride);
        }

        for (int y = 0; y < feature_map_height; ++y) {
          for (int x = 0; x < feature_map_width; ++x) {
            for (int anchor_id = 0; anchor_id < (int)anchor_height.size(); ++anchor_id) {
              // TODO: Support specifying anchor_offset_x, anchor_offset_y.
              const float x_center = (x + options.anchor_offset_x) * 1.0f / feature_map_width;
              const float y_center = (y + options.anchor_offset_y) * 1.0f / feature_map_height;

              Anchor new_anchor;
              new_anchor.x_center = x_center;
              new_anchor.y_center = y_center;

              if (options.fixed_anchor_size) {
                new_anchor.w = 1.0f;
                new_anchor.h = 1.0f;
              } else {
                new_anchor.w = anchor_width [anchor_id];
                new_anchor.h = anchor_height[anchor_id];
              }
              anchors->push_back(new_anchor);
            }
          }
        }
        layer_id = last_same_stride_layer;
    }
    return 0;
}


static int
generate_ssd_anchors ()
{
    SsdAnchorsCalculatorOptions anchor_options;
    anchor_options.num_layers = 5;
    anchor_options.min_scale = 0.1171875;
    anchor_options.max_scale = 0.75;
    anchor_options.input_size_height = 256;
    anchor_options.input_size_width  = 256;
    anchor_options.anchor_offset_x  = 0.5f;
    anchor_options.anchor_offset_y  = 0.5f;
//  anchor_options.feature_map_width .push_back(0);
//  anchor_options.feature_map_height.push_back(0);
    anchor_options.strides.push_back( 8);
    anchor_options.strides.push_back(16);
    anchor_options.strides.push_back(32);
    anchor_options.strides.push_back(32);
    anchor_options.strides.push_back(32);
    anchor_options.aspect_ratios.push_back(1.0);
    anchor_options.reduce_boxes_in_lowest_layer = false;
    anchor_options.interpolated_scale_aspect_ratio = 1.0;
    anchor_options.fixed_anchor_size = true;

    GenerateAnchors (&s_anchors, anchor_options);

#if 0
    for (int i = 0; i < (int)s_anchors.size(); i ++)
    {
        fprintf (stderr, "[%4d](%f, %f, %f, %f)\n", i,
            s_anchors[i].x_center, s_anchors[i].y_center, s_anchors[i].w, s_anchors[i].h);
    }
#endif

    return 0;
}


/* -------------------------------------------------- *
 *  Create TFLite Interpreter
 * -------------------------------------------------- */
int
init_tflite_hand_landmark()
{
    /* Palm Detection */
    s_palm_interpreter.resolver.AddCustom("Convolution2DTransposeBias",
            mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());

    tflite_create_interpreter_from_file (&s_palm_interpreter, PALM_DETECTION_MODEL_PATH);
    tflite_get_tensor_by_name (&s_palm_interpreter, 0, "input",           &s_palm_tensor_input);
    tflite_get_tensor_by_name (&s_palm_interpreter, 1, "classificators",  &s_palm_tensor_scores);
    tflite_get_tensor_by_name (&s_palm_interpreter, 1, "regressors",      &s_palm_tensor_points);

    /* Hand Landmark */
    tflite_create_interpreter_from_file (&s_hand_interpreter, HAND_LANDMARK_MODEL_PATH);
    tflite_get_tensor_by_name (&s_hand_interpreter, 0, "input_1",         &s_hand_tensor_input);
    tflite_get_tensor_by_name (&s_hand_interpreter, 1, "ld_21_3d",        &s_hand_tensor_landmark);
    tflite_get_tensor_by_name (&s_hand_interpreter, 1, "output_handflag", &s_hand_tensor_handflag);

    generate_ssd_anchors ();

    return 0;
}

void *
get_palm_detection_input_buf (int *w, int *h)
{
    *w = s_palm_tensor_input.dims[2];
    *h = s_palm_tensor_input.dims[1];
    return s_palm_tensor_input.ptr;
}

void *
get_hand_landmark_input_buf (int *w, int *h)
{
    *w = s_hand_tensor_input.dims[2];
    *h = s_hand_tensor_input.dims[1];
    return s_hand_tensor_input.ptr;
}


/* -------------------------------------------------- *
 *  Decode palm detection result
 * -------------------------------------------------- */static int
decode_keypoints (std::list<palm_t> &palm_list, float score_thresh)
{
    palm_t palm_item;
    float *scores_ptr = (float *)s_palm_tensor_scores.ptr;
    float *points_ptr = (float *)s_palm_tensor_points.ptr;
    int img_w = s_palm_tensor_input.dims[2];
    int img_h = s_palm_tensor_input.dims[1];

    int i = 0;
    for (auto itr = s_anchors.begin(); itr != s_anchors.end(); i ++, itr ++)
    {
        Anchor anchor = *itr;
        float score0 = scores_ptr[i];
        float score = 1.0f / (1.0f + exp(-score0));

        if (score > score_thresh)
        {
            float *p = points_ptr + (i * 18);

            /* boundary box */
            float sx = p[0];
            float sy = p[1];
            float w  = p[2];
            float h  = p[3];

            float cx = sx + anchor.x_center * img_w;
            float cy = sy + anchor.y_center * img_w;

            cx /= (float)img_w;
            cy /= (float)img_h;
            w  /= (float)img_w;
            h  /= (float)img_h;

            fvec2 topleft, btmright;
            topleft.x  = cx - w * 0.5f;
            topleft.y  = cy - h * 0.5f;
            btmright.x = cx + w * 0.5f;
            btmright.y = cy + h * 0.5f;

            palm_item.score         = score;
            palm_item.rect.topleft  = topleft;
            palm_item.rect.btmright = btmright;

            /* landmark positions (7 keys) */
            for (int j = 0; j < 7; j ++)
            {
                float lx = p[4 + (2 * j) + 0];
                float ly = p[4 + (2 * j) + 1];
                lx += anchor.x_center * img_w;
                ly += anchor.y_center * img_w;
                lx /= (float)img_w;
                ly /= (float)img_h;

                palm_item.keys[j].x = lx;
                palm_item.keys[j].y = ly;
            }

            palm_list.push_back (palm_item);
        }
    }
    return 0;
}



/* -------------------------------------------------- *
 *  Apply NonMaxSuppression:
 * -------------------------------------------------- */
static float
calc_intersection_over_union (rect_t &rect0, rect_t &rect1)
{
    float sx0 = rect0.topleft.x;
    float sy0 = rect0.topleft.y;
    float ex0 = rect0.btmright.x;
    float ey0 = rect0.btmright.y;
    float sx1 = rect1.topleft.x;
    float sy1 = rect1.topleft.y;
    float ex1 = rect1.btmright.x;
    float ey1 = rect1.btmright.y;
    
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
compare (palm_t &v1, palm_t &v2)
{
    if (v1.score > v2.score)
        return true;
    else
        return false;
}

static int
non_max_suppression (std::list<palm_t> &face_list, std::list<palm_t> &face_sel_list, float iou_thresh)
{
    face_list.sort (compare);

    for (auto itr = face_list.begin(); itr != face_list.end(); itr ++)
    {
        palm_t face_candidate = *itr;

        int ignore_candidate = false;
        for (auto itr_sel = face_sel_list.rbegin(); itr_sel != face_sel_list.rend(); itr_sel ++)
        {
            palm_t face_sel = *itr_sel;

            float iou = calc_intersection_over_union (face_candidate.rect, face_sel.rect);
            if (iou >= iou_thresh)
            {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate)
        {
            face_sel_list.push_back(face_candidate);
            if (face_sel_list.size() >= MAX_PALM_NUM)
                break;
        }
    }

    return 0;
}


/* -------------------------------------------------- *
 *  Expand palm to hand
 * -------------------------------------------------- */
static float
normalize_radians (float angle)
{
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

static void
compute_rotation (palm_t &palm)
{
    float x0 = palm.keys[0].x;  // Center of wrist.
    float y0 = palm.keys[0].y;
    float x1 = palm.keys[2].x;  // MCP of middle finger.
    float y1 = palm.keys[2].y;

    float target_angle = M_PI * 0.5f;
    float rotation = target_angle - std::atan2(-(y1 - y0), x1 - x0);
    
    palm.rotation = normalize_radians (rotation);
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
compute_hand_rect (palm_t &palm)
{
    float width    = palm.rect.btmright.x - palm.rect.topleft.x;
    float height   = palm.rect.btmright.y - palm.rect.topleft.y;
    float palm_cx  = palm.rect.topleft.x + width  * 0.5f;
    float palm_cy  = palm.rect.topleft.y + height * 0.5f;
    float hand_cx;
    float hand_cy;
    float rotation = palm.rotation;
    float shift_x =  0.0f;
    float shift_y = -0.5f;
    
    if (rotation == 0.0f)
    {
        hand_cx = palm_cx + (width  * shift_x);
        hand_cy = palm_cy + (height * shift_y);
    }
    else
    {
        float dx = (width  * shift_x) * std::cos(rotation) -
                   (height * shift_y) * std::sin(rotation);
        float dy = (width  * shift_x) * std::sin(rotation) +
                   (height * shift_y) * std::cos(rotation);
        hand_cx = palm_cx + dx;
        hand_cy = palm_cy + dy;
    }

    float long_side = std::max (width, height);
    width  = long_side;
    height = long_side;
    float hand_w = width  * 2.6f;
    float hand_h = height * 2.6f;

    palm.hand_cx = hand_cx;
    palm.hand_cy = hand_cy;
    palm.hand_w  = hand_w;
    palm.hand_h  = hand_h;

    float dx = hand_w * 0.5f;
    float dy = hand_h * 0.5f;

    palm.hand_pos[0].x = - dx;  palm.hand_pos[0].y = - dy;
    palm.hand_pos[1].x = + dx;  palm.hand_pos[1].y = - dy;
    palm.hand_pos[2].x = + dx;  palm.hand_pos[2].y = + dy;
    palm.hand_pos[3].x = - dx;  palm.hand_pos[3].y = + dy;

    for (int i = 0; i < 4; i ++)
    {
        rot_vec (palm.hand_pos[i], rotation);
        palm.hand_pos[i].x += hand_cx;
        palm.hand_pos[i].y += hand_cy;
    }
}

static void
pack_palm_result (palm_detection_result_t *palm_result, std::list<palm_t> &palm_list)
{
    int num_palms = 0;
    for (auto itr = palm_list.begin(); itr != palm_list.end(); itr ++)
    {
        palm_t palm = *itr;
        
        compute_rotation (palm);
        compute_hand_rect (palm);

        memcpy (&palm_result->palms[num_palms], &palm, sizeof (palm));
        num_palms ++;
        palm_result->num = num_palms;

        if (num_palms >= MAX_PALM_NUM)
            break;
    }
}



/* -------------------------------------------------- *
 * Invoke TensorFlow Lite (Palm detection)
 * -------------------------------------------------- */
static int
detect_palm (palm_detection_result_t *palm_result)
{
    if (s_palm_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    float score_thresh = 0.7f;
    std::list<palm_t> palm_list;

    decode_keypoints (palm_list, score_thresh);

#if 1 /* USE NMS */
    float iou_thresh = 0.3f;
    std::list<palm_t> palm_nms_list;

    non_max_suppression (palm_list, palm_nms_list, iou_thresh);
    pack_palm_result (palm_result, palm_nms_list);
#else
    pack_palm_result (palm_result, palm_list);
#endif

    return 0;
}

static int
detect_palm_stub (palm_detection_result_t *palm_result)
{
    palm_result->num = 1;

    palm_t *palm = &palm_result->palms[0];
    palm->hand_pos[0].x = 0.0f; palm->hand_pos[0].y = 0.0f; //    0--------1
    palm->hand_pos[1].x = 1.0f; palm->hand_pos[1].y = 0.0f; //    |        |
    palm->hand_pos[2].x = 1.0f; palm->hand_pos[2].y = 1.0f; //    |        |
    palm->hand_pos[3].x = 0.0f; palm->hand_pos[3].y = 1.0f; //    3--------2
    palm->hand_cx  = 0.5f;
    palm->hand_cy  = 0.5f;
    palm->hand_w   = 1.0f;
    palm->hand_h   = 1.0f;
    palm->rotation = 0.0f;

    return 0;
}

int
invoke_palm_detection (palm_detection_result_t *palm_result, int flag)
{
    if (flag == 0)
    {
        return detect_palm (palm_result);
    }
    else
    {
        return detect_palm_stub (palm_result);
    }
}



/* -------------------------------------------------- *
 * Invoke TensorFlow Lite (Hand landmark)
 * -------------------------------------------------- */
int
invoke_hand_landmark (hand_landmark_result_t *hand_result)
{
    if (s_hand_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    float *handflag_ptr = (float *)s_hand_tensor_handflag.ptr;
    float *landmark_ptr = (float *)s_hand_tensor_landmark.ptr;
    int img_w = s_hand_tensor_input.dims[2];
    int img_h = s_hand_tensor_input.dims[1];
    
    hand_result->score = *handflag_ptr;
    //fprintf (stderr, "handflag = %f\n", *handflag_ptr);
    
    for (int i = 0; i < HAND_JOINT_NUM; i ++)
    {
        hand_result->joint[i].x = landmark_ptr[3 * i + 0] / (float)img_w;
        hand_result->joint[i].y = landmark_ptr[3 * i + 1] / (float)img_h;
        hand_result->joint[i].z = landmark_ptr[3 * i + 2];
        //fprintf (stderr, "[%2d] (%8.1f, %8.1f, %8.1f)\n", i, 
        //    landmark_ptr[3 * i + 0], landmark_ptr[3 * i + 1], landmark_ptr[3 * i + 2]);
    }

    return 0;
}

