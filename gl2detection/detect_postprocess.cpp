/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "detect_postprocess.h"

static float    *s_anchors;
static int      s_anchors_count;

static float    *s_decoded_boxes;
static uint8_t  *s_active_candidate;

/* Attrubutes of TFLite_Detection_PostProcess */
#define ATTR_X_SCALE                      10.0
#define ATTR_Y_SCALE                      10.0
#define ATTR_W_SCALE                       5.0
#define ATTR_H_SCALE                       5.0
#define ATTR_NUM_CLASSES                  90
#define ATTR_MAX_CLASSES_PER_DETECTION    1
#define ATTR_DETECTIONS_PER_CLASS         100
#define ATTR_MAX_DETECTIONS               100
#define ATTR_NMS_SCORE_THRESHOLD          0.5
#define ATTR_NMS_IOU_THRESHOLD            0.6
#define ATTR_USE_REGULAR_NMS              false

/* -------------------------------------------------------------------- *
 *  Decode detection boxes and apply NMS.
 *    These functions are clone codes of: 
 *    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/detection_postprocess.cc
 * -------------------------------------------------------------------- */

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

struct BoxCornerEncoding {
    float ymin;
    float xmin;
    float ymax;
    float xmax;
};

struct CenterSizeEncoding {
    float y;
    float x;
    float h;
    float w;
};

int
DecodeCenterSizeBoxes (float *decoded_boxes, const float *input_box_encodings) 
{
    int num_boxes        = s_anchors_count;
    float *input_anchors = s_anchors;

    // Decode the boxes to get (ymin, xmin, ymax, xmax) based on the anchors
    CenterSizeEncoding box_centersize;
    CenterSizeEncoding scale_values = {ATTR_X_SCALE, ATTR_Y_SCALE,
                                       ATTR_W_SCALE, ATTR_H_SCALE};
    CenterSizeEncoding anchor;

    for (int idx = 0; idx < num_boxes; ++idx) 
    {
        box_centersize = reinterpret_cast<const CenterSizeEncoding*>(input_box_encodings)[idx];
        anchor         = reinterpret_cast<const CenterSizeEncoding*>(input_anchors)[idx];

        float ycenter = box_centersize.y / scale_values.y * anchor.h + anchor.y;
        float xcenter = box_centersize.x / scale_values.x * anchor.w + anchor.x;
        float half_h =
            0.5f * static_cast<float>(std::exp(box_centersize.h / scale_values.h)) *
            anchor.h;
        float half_w =
            0.5f * static_cast<float>(std::exp(box_centersize.w / scale_values.w)) *
            anchor.w;

        auto& box = reinterpret_cast<BoxCornerEncoding*>(decoded_boxes)[idx];
        box.ymin = ycenter - half_h;
        box.xmin = xcenter - half_w;
        box.ymax = ycenter + half_h;
        box.xmax = xcenter + half_w;
    }
    return 0;
}


void DecreasingPartialArgSort(const float* values, int num_values,
                              int num_to_sort, int* indices) {
  std::iota(indices, indices + num_values, 0);
  std::partial_sort(
      indices, indices + num_to_sort, indices + num_values,
      [&values](const int i, const int j) { return values[i] > values[j]; });
}


void SelectDetectionsAboveScoreThreshold(const std::vector<float>& values,
                                         const float threshold,
                                         std::vector<float>* keep_values,
                                         std::vector<int>* keep_indices) {
  for (unsigned int i = 0; i < values.size(); i++) {
    if (values[i] >= threshold) {
      keep_values->emplace_back(values[i]);
      keep_indices->emplace_back(i);
    }
  }
}


float ComputeIntersectionOverUnion(const float* decoded_boxes,
                                   const int i, const int j) {
  auto& box_i = reinterpret_cast<const BoxCornerEncoding*>(decoded_boxes)[i];
  auto& box_j = reinterpret_cast<const BoxCornerEncoding*>(decoded_boxes)[j];
  const float area_i = (box_i.ymax - box_i.ymin) * (box_i.xmax - box_i.xmin);
  const float area_j = (box_j.ymax - box_j.ymin) * (box_j.xmax - box_j.xmin);
  if (area_i <= 0 || area_j <= 0) return 0.0;
  const float intersection_ymin = std::max<float>(box_i.ymin, box_j.ymin);
  const float intersection_xmin = std::max<float>(box_i.xmin, box_j.xmin);
  const float intersection_ymax = std::min<float>(box_i.ymax, box_j.ymax);
  const float intersection_xmax = std::min<float>(box_i.xmax, box_j.xmax);
  const float intersection_area =
      std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
      std::max<float>(intersection_xmax - intersection_xmin, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}


// NonMaxSuppressionSingleClass() prunes out the box locations with high overlap
// before selecting the highest scoring boxes (max_detections in number)
// It assumes all boxes are good in beginning and sorts based on the scores.
// If lower-scoring box has too much overlap with a higher-scoring box,
// we get rid of the lower-scoring box.
// Complexity is O(N^2) pairwise comparison between boxes
int
NonMaxSuppressionSingleClassHelper(const float *decoded_boxes,
                                   const std::vector<float>& scores, 
                                   std::vector<int>* selected, int max_detections) {

    const float non_max_suppression_score_threshold = ATTR_NMS_SCORE_THRESHOLD;
    const float intersection_over_union_threshold   = ATTR_NMS_IOU_THRESHOLD;

    // threshold scores
    std::vector<int> keep_indices;
    // TODO (chowdhery): Remove the dynamic allocation and replace it
    // with temporaries, esp for std::vector<float>
    std::vector<float> keep_scores;
    SelectDetectionsAboveScoreThreshold(
        scores, non_max_suppression_score_threshold, &keep_scores, &keep_indices);

    int num_scores_kept = keep_scores.size();
    std::vector<int> sorted_indices;
    sorted_indices.resize(num_scores_kept);
    DecreasingPartialArgSort(keep_scores.data(), num_scores_kept, num_scores_kept,
                                sorted_indices.data());
    const int num_boxes_kept = num_scores_kept;
    const unsigned int output_size = std::min(num_boxes_kept, max_detections);
    selected->clear();

    int num_active_candidate = num_boxes_kept;
    uint8_t* active_box_candidate = s_active_candidate;
    for (int row = 0; row < num_boxes_kept; row++) {
        active_box_candidate[row] = 1;
    }

    for (int i = 0; i < num_boxes_kept; ++i) {
        if (num_active_candidate == 0 || selected->size() >= output_size) break;
        if (active_box_candidate[i] == 1) {
            selected->push_back(keep_indices[sorted_indices[i]]);
            active_box_candidate[i] = 0;
            num_active_candidate--;
        } else {
            continue;
        }

        for (int j = i + 1; j < num_boxes_kept; ++j) {
            if (active_box_candidate[j] == 1) {
                float intersection_over_union = ComputeIntersectionOverUnion(
                    decoded_boxes, keep_indices[sorted_indices[i]],
                    keep_indices[sorted_indices[j]]);

                if (intersection_over_union > intersection_over_union_threshold) {
                    active_box_candidate[j] = 0;
                    num_active_candidate--;
                }
            }
        }
    }
    return 0;
}


// This function implements a regular version of Non Maximal Suppression (NMS)
// for multiple classes where
// 1) we do NMS separately for each class across all anchors and
// 2) keep only the highest anchor scores across all classes
// 3) The worst runtime of the regular NMS is O(K*N^2)
// where N is the number of anchors and K the number of
// classes.
int
NonMaxSuppressionMultiClassRegularHelper(std::vector<DetectionBox> &detection_boxes, 
                                         const float *decoded_boxes, const float* scores) {
    const int num_boxes   = s_anchors_count;
    const int num_classes = ATTR_NUM_CLASSES;
    const int num_detections_per_class = ATTR_DETECTIONS_PER_CLASS;
    const int max_detections = ATTR_MAX_DETECTIONS;

    // The row index offset is 1 if background class is included and 0 otherwise.
    const int label_offset = 1;
    const int num_classes_with_background = num_classes + label_offset;

    // For each class, perform non-max suppression.
    std::vector<float> class_scores(num_boxes);

    std::vector<int> box_indices_after_regular_non_max_suppression(num_boxes + max_detections);
    std::vector<float> scores_after_regular_non_max_suppression(num_boxes +  max_detections);

    int size_of_sorted_indices = 0;
    std::vector<int> sorted_indices;
    sorted_indices.resize(num_boxes + max_detections);
    std::vector<float> sorted_values;
    sorted_values.resize(max_detections);

    for (int col = 0; col < num_classes; col++) {
        for (int row = 0; row < num_boxes; row++) {
            // Get scores of boxes corresponding to all anchors for single class
            class_scores[row] =
                *(scores + row * num_classes_with_background + col + label_offset);
        }
        // Perform non-maximal suppression on single class
        std::vector<int> selected;
        NonMaxSuppressionSingleClassHelper(decoded_boxes, class_scores, &selected, num_detections_per_class);

        // Add selected indices from non-max suppression of boxes in this class
        int output_index = size_of_sorted_indices;
        for (const auto& selected_index : selected) {
            box_indices_after_regular_non_max_suppression[output_index] =
                (selected_index * num_classes_with_background + col + label_offset);
            scores_after_regular_non_max_suppression[output_index] =
                class_scores[selected_index];
            output_index++;
        }

        // Sort the max scores among the selected indices
        // Get the indices for top scores
        int num_indices_to_sort = std::min(output_index, max_detections);
        DecreasingPartialArgSort(scores_after_regular_non_max_suppression.data(),
                             output_index, num_indices_to_sort,
                             sorted_indices.data());

        // Copy values to temporary vectors
        for (int row = 0; row < num_indices_to_sort; row++) {
            int temp = sorted_indices[row];
            sorted_indices[row] = box_indices_after_regular_non_max_suppression[temp];
            sorted_values[row] = scores_after_regular_non_max_suppression[temp];
        }
        // Copy scores and indices from temporary vectors
        for (int row = 0; row < num_indices_to_sort; row++) {
            box_indices_after_regular_non_max_suppression[row] = sorted_indices[row];
            scores_after_regular_non_max_suppression[row] = sorted_values[row];
        }
        size_of_sorted_indices = num_indices_to_sort;
    }

    // Allocate output tensors
    for (int output_box_index = 0; output_box_index < max_detections; output_box_index++) {
        if (output_box_index < size_of_sorted_indices) {
            const int anchor_index = floor(
                        box_indices_after_regular_non_max_suppression[output_box_index] /
                        num_classes_with_background);
            const int class_index =
                        box_indices_after_regular_non_max_suppression[output_box_index] -
                        anchor_index * num_classes_with_background - label_offset;
            const float selected_score =
                        scores_after_regular_non_max_suppression[output_box_index];

            BoxCornerEncoding box = reinterpret_cast<const BoxCornerEncoding*>(s_decoded_boxes)[anchor_index];

            detection_boxes.push_back({box.xmin, box.ymin,
                                       box.xmax, box.ymax,
                                       selected_score, class_index});
        } else {
        }
    }

    box_indices_after_regular_non_max_suppression.clear();
    scores_after_regular_non_max_suppression.clear();

    return 0;
}


// This function implements a fast version of Non Maximal Suppression for
// multiple classes where
// 1) we keep the top-k scores for each anchor and
// 2) during NMS, each anchor only uses the highest class score for sorting.
// 3) Compared to standard NMS, the worst runtime of this version is O(N^2)
// instead of O(KN^2) where N is the number of anchors and K the number of
// classes.
int
NonMaxSuppressionMultiClassFastHelper (std::vector<DetectionBox> &detection_boxes, 
                                       const float *decoded_boxes, const float* scores) {
    const int num_boxes   = s_anchors_count;
    const int num_classes = ATTR_NUM_CLASSES;
    const int max_categories_per_anchor = ATTR_MAX_CLASSES_PER_DETECTION;

    // The row index offset is 1 if background class is included and 0 otherwise.
    const int label_offset = 1;
    const int num_classes_with_background = num_classes + label_offset;
    const int num_categories_per_anchor   = std::min(max_categories_per_anchor, num_classes);

    std::vector<float> max_scores;
    max_scores.resize(num_boxes);
    std::vector<int> sorted_class_indices;
    sorted_class_indices.resize(num_boxes * num_classes);

    for (int row = 0; row < num_boxes; row++) {
        const float* box_scores =
                    scores + row * num_classes_with_background + label_offset;
        int* class_indices = sorted_class_indices.data() + row * num_classes;
        DecreasingPartialArgSort(box_scores, num_classes, num_categories_per_anchor,
                             class_indices);
        max_scores[row] = box_scores[class_indices[0]];
    }

    // Perform non-maximal suppression on max scores
    std::vector<int> selected;
    NonMaxSuppressionSingleClassHelper(decoded_boxes, max_scores, &selected, ATTR_MAX_DETECTIONS);

    // Allocate output tensors
    for (const auto& selected_index : selected) {
        const float* box_scores =
                scores + selected_index * num_classes_with_background + label_offset;
        const int* class_indices =
                sorted_class_indices.data() + selected_index * num_classes;

        for (int col = 0; col < num_categories_per_anchor; ++col) {

            // detection_boxes
            BoxCornerEncoding box = reinterpret_cast<const BoxCornerEncoding*>(decoded_boxes)[selected_index];

            // detection_classes
            int class_index = class_indices[col];

            // detection_scores
            float score = box_scores[class_index];

            detection_boxes.push_back({box.xmin, box.ymin,
                                       box.xmax, box.ymax,
                                       score, class_index});
        }
    }

    return 0;
}




/* -------------------------------------------------------------------- *
 *  software routine for "TFLite_Detection_PostProcess" Op.
 * -------------------------------------------------------------------- */
float *
read_anchors_file (std::string filename, int& size) 
{
    std::vector<std::string> lines;
    std::ifstream file (filename);

    std::string s;
    while (getline (file, s))
    {
        lines.push_back (s);
    }

    size = lines.size();
    float *result = new float[size * 4]();

    for (int i = 0; i < size; i++) 
    {
        int index = i * 4;
        std::stringstream(lines[i]) >> result[index]
                                    >> result[index + 1]
                                    >> result[index + 2]
                                    >> result[index + 3];
    }
    return result;
}


int
init_detect_postprocess (std::string filename)
{
    s_anchors = read_anchors_file (filename, s_anchors_count);

#if 0
    for (int i = 0; i < s_anchors_count; i ++)
    {
        fprintf (stderr, "[%3d] %f %f %f %f\n", i,
            s_anchors[i * 4 + 0], s_anchors[i * 4 + 1], s_anchors[i * 4 + 2], s_anchors[i * 4 + 3]);
    }
#endif

    s_decoded_boxes    = new float  [s_anchors_count * 4];
    s_active_candidate = new uint8_t[s_anchors_count];

    return 0;
}


int
invoke_detection_postprocess (std::vector<DetectionBox> &detection_boxes,  /* [OUT] */
                              const float *boxes_ptr,                      /* [IN ] */
                              const float *scores_ptr)                     /* [IN ] */
{
    float *decoded_boxes = s_decoded_boxes;

    /*
     *  decode detected bbox. 
     *      (decoded_boxes) = (boxes_ptr) * (anchor.wh) + (anchor.xy);
     */
    DecodeCenterSizeBoxes (decoded_boxes, boxes_ptr);

    if (ATTR_USE_REGULAR_NMS)
    {
        NonMaxSuppressionMultiClassRegularHelper(detection_boxes, decoded_boxes, scores_ptr);
    }
    else
    {
        NonMaxSuppressionMultiClassFastHelper (detection_boxes, decoded_boxes, scores_ptr);
    }

    return 0;
}

