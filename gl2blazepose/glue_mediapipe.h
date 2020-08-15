/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */

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

#ifndef GLUE_MEDIAPIPE_H_
#define GLUE_MEDIAPIPE_H_

#include <list>
#include <vector>


typedef struct Anchor
{
    float x_center, y_center, w, h;
} Anchor;

/*  
 *  SsdAnchorsCalculatorOptions
 *      mediapipe/calculators/tflite/ssd_anchors_calculator.proto
 */
typedef struct SsdAnchorsCalculatorOptions 
{
    // Size of input images.
    int input_size_width;                   // [required]
    int input_size_height;                  // [required]

    // Min and max scales for generating anchor boxes on feature maps.
    float min_scale;                        // [required]
    float max_scale;                        // [required]

    // The offset for the center of anchors. The value is in the scale of stride.
    // E.g. 0.5 meaning 0.5 * |current_stride| in pixels.
    float anchor_offset_x;                  // default = 0.5
    float anchor_offset_y;                  // default = 0.5

    // Number of output feature maps to generate the anchors on.
    int num_layers;                         // [required]

    // Sizes of output feature maps to create anchors. Either feature_map size or
    // stride should be provided.
    std::vector<int> feature_map_width;
    std::vector<int> feature_map_height;

    // Strides of each output feature maps.
    std::vector<int>   strides;

    // List of different aspect ratio to generate anchors.
    std::vector<float> aspect_ratios;

    // A boolean to indicate whether the fixed 3 boxes per location is used in the
    // lowest layer.
    bool reduce_boxes_in_lowest_layer;      // default = false

    // An additional anchor is added with this aspect ratio and a scale
    // interpolated between the scale for a layer and the scale for the next layer
    // (1.0 for the last layer). This anchor is not included if this value is 0.
    float interpolated_scale_aspect_ratio;  // default = 1.0

    // Whether use fixed width and height (e.g. both 1.0f) for each anchor.
    // This option can be used when the predicted anchor width and height are in
    // pixels.
    bool fixed_anchor_size;                 // default = false
} SsdAnchorsCalculatorOptions;


int GenerateAnchors(std::vector<Anchor>* anchors, const SsdAnchorsCalculatorOptions& options);


#endif /* GLUE_MEDIAPIPE_H_ */
