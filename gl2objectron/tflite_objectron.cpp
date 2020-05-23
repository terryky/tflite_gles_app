/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */

// Copyright 2020 The MediaPipe Authors.
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


#include "util_tflite.h"
#include "tflite_objectron.h"
#include <list>
#include "Eigen/Dense"

/* 
 * https://github.com/google/mediapipe/blob/master/mediapipe/models/object_detection_3d_chair.tflite
 * https://github.com/google/mediapipe/blob/master/mediapipe/models/object_detection_3d_sneakers.tflite
 * https://github.com/PINTO0309/PINTO_model_zoo/blob/master/36_Objectron/03_integer_quantization
 */
#if 1 /* [1] chair or [0] shoes */
#define DETECT_3D_MODEL_PATH        "./objectron_model/object_detection_3d_chair.tflite"
#define DETECT_3D_QUANT_MODEL_PATH  "./objectron_model/object_detection_3d_chair_640x480_integer_quant.tflite"
#else
#define DETECT_3D_MODEL_PATH        "./objectron_model/object_detection_3d_sneakers.tflite"
#define DETECT_3D_QUANT_MODEL_PATH  "./objectron_model/object_detection_3d_sneakers_640x480_integer_quant.tflite"
#endif

static tflite_interpreter_t s_detect_interpreter;
static tflite_tensor_t      s_detect_tensor_input;
static tflite_tensor_t      s_detect_tensor_offsetmap;
static tflite_tensor_t      s_detect_tensor_heatmap;

static int s_need_post_logistic = 0;

/*
 * https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection_3d/calculators/tflite_tensors_to_objects_calculator.cc
 */
Eigen::Matrix<float, 4, 4, Eigen::RowMajor> projection_matrix_;
Eigen::Matrix<float, 8, 4, Eigen::RowMajor> epnp_alpha_;

/* -------------------------------------------------- *
 *  Create TFLite Interpreter
 * -------------------------------------------------- */
int
init_tflite_objectron (int use_quantized_tflite)
{
    if (use_quantized_tflite)
    {
        tflite_create_interpreter_from_file (&s_detect_interpreter, DETECT_3D_QUANT_MODEL_PATH);
        tflite_get_tensor_by_name (&s_detect_interpreter, 0, "input",      &s_detect_tensor_input);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "model/belief/Conv2D/conv2d", &s_detect_tensor_heatmap);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "Identity_1", &s_detect_tensor_offsetmap); /*  40x 30x16 */
        s_need_post_logistic = 1;
    }
    else
    {
        tflite_create_interpreter_from_file (&s_detect_interpreter, DETECT_3D_MODEL_PATH);
        tflite_get_tensor_by_name (&s_detect_interpreter, 0, "input",      &s_detect_tensor_input);
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "Identity",   &s_detect_tensor_heatmap);   /*  40x 30x 1 */
        tflite_get_tensor_by_name (&s_detect_interpreter, 1, "Identity_1", &s_detect_tensor_offsetmap); /*  40x 30x16 */
    }

    projection_matrix_ <<
      1.5731,     0,       0,    0,
      0,     2.0975,       0,    0,
      0,          0, -1.0002, -0.2,
      0,          0,      -1,    0;

    epnp_alpha_ << 4.0f, -1.0f, -1.0f, -1.0f, 2.0f, -1.0f, -1.0f, 1.0f, 2.0f,
        -1.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 1.0f,  2.0f,  1.0f, -1.0f, -1.0f,
         0.0f, 1.0f, -1.0f, 1.0f,  0.0f, 1.0f, 1.0f, -1.0f, -2.0f,  1.0f,  1.0f,
         1.0f;

    return 0;
}

void *
get_objectron_input_buf (int *w, int *h)
{
    *w = s_detect_tensor_input.dims[2];
    *h = s_detect_tensor_input.dims[1];
    return s_detect_tensor_input.ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite (3D Object detection)
 * -------------------------------------------------- */
static float
get_heatmap_val (int x, int y)
{
    int hmp_w = s_detect_tensor_heatmap.dims[2];
    float *heatmap = (float *)s_detect_tensor_heatmap.ptr;

    float val = heatmap[hmp_w * y + x];

    if (s_need_post_logistic)
    {
        const float cutoff_upper = 16.619047164916992188f;
        const float cutoff_lower = -9.f;

        if (val > cutoff_upper)
            val = 1.0f;
        else if (val < cutoff_lower)
            val = std::exp(val);
        else
            val = 1.f / (1.f + std::exp(-val));
    }

    return val;
}

static float
get_max_value (int cx, int cy, int hmp_w, int hmp_h, int kern_size)
{
    int sx = cx - (kern_size / 2);
    int ex = cx + (kern_size / 2) + 1;
    int sy = cy - (kern_size / 2);
    int ey = cy + (kern_size / 2) + 1;

    sx = std::max (sx, 0);
    ex = std::min (ex, hmp_w);
    sy = std::max (sy, 0);
    ey = std::min (ey, hmp_h);

    float max_val = 0;
    for (int y = sy; y < ey; y ++)
    {
        for (int x = sx; x < ex; x ++)
        {
            float val = get_heatmap_val (x, y);
            max_val = std::max (val, max_val);
        }
    }

    return max_val;
}

static void
dilate_heatmap (float *dst_hmp, int hmp_w, int hmp_h, int kern_size)
{
    for (int y = 0; y < hmp_h; y ++)
    {
        for (int x = 0; x < hmp_w; x ++)
        {
            dst_hmp[hmp_w * y + x] = get_max_value (x, y, hmp_w, hmp_h, kern_size);
        }
    }
}

static void
extract_center_keypoints (std::list<fvec2> &center_points)
{
    int hmp_w = s_detect_tensor_heatmap.dims[2];
    int hmp_h = s_detect_tensor_heatmap.dims[1];

    float *max_filtered_heatmap = (float *)malloc (hmp_w * hmp_h * sizeof (float));

    /* apply (5x5) MAX filter */
    int local_max_distance = 2;
    int kernel_size = static_cast<int>(local_max_distance * 2 + 1 + 0.5f);
    dilate_heatmap (max_filtered_heatmap, hmp_w, hmp_h, kernel_size);

    float heatmap_threshold = 0.6f;
    for (int y = 0; y < hmp_h; y ++)
    {
        for (int x = 0; x < hmp_w; x ++)
        {
            float center_hmp_val = get_heatmap_val (x, y);
            float max_hmp_val    = max_filtered_heatmap[hmp_w * y + x];

            if ((center_hmp_val >= heatmap_threshold) &&
                (center_hmp_val >= max_hmp_val))
            {
                fvec2 locations;
                locations.x = x;
                locations.y = y;
                center_points.push_back (locations);
            }
        }
    }

    free (max_filtered_heatmap);
}

/*
 *  (cx, cy): Tile id. (cx: 0-30), (cy: 0-40)
 */
void
decode_by_voting (int cx, int cy, float offset_scale_x, float offset_scale_y, object_t *obj)
{
    float *offsetmap = (float *)s_detect_tensor_offsetmap.ptr;
    int   map_w = s_detect_tensor_offsetmap.dims[2];
    int   map_h = s_detect_tensor_offsetmap.dims[1];
    float *center_offset = &offsetmap[16 * ((cy * map_w) + cx)];

    /* transform BBOX offsetmap. (relative offset) --> (absolute offset) */
    float *center_votes = (float *)malloc (16 * map_w * map_h * sizeof (float));
    for (int i = 0; i < 8; i ++)
    {
        center_votes[2 * i    ] = cx + center_offset[2 * i    ] * offset_scale_x;
        center_votes[2 * i + 1] = cy + center_offset[2 * i + 1] * offset_scale_y;
    }

    /* Voting Window */
    int voting_radius = 2;
    int x_min  = std::max (0, cx - voting_radius);
    int y_min  = std::max (0, cy - voting_radius);
    int width  = std::min (map_w - x_min, voting_radius * 2 + 1);
    int height = std::min (map_h - y_min, voting_radius * 2 + 1);

    float voting_threshold = 0.2f;
    float voting_allowance = 1.0f;
    for (int i = 0; i < 8; i ++)
    {
        float x_sum = 0.0f;
        float y_sum = 0.0f;
        float votes = 0.0f;

        for (int r = 0; r < height; r ++)
        {
            for (int c = 0; c < width; c ++)
            {
                int idx_x = c + x_min;
                int idx_y = r + y_min;

                float belief = get_heatmap_val (idx_x, idx_y);
                if (belief < voting_threshold)
                    continue;

                float *cur_offsetmap = &offsetmap[16 * ((idx_y * map_w) + idx_x)];

                float offset_x = cur_offsetmap[2 * i    ] * offset_scale_x;
                float offset_y = cur_offsetmap[2 * i + 1] * offset_scale_y;
                float vote_x   = c + x_min + offset_x;
                float vote_y   = r + y_min + offset_y;
                float x_diff   = std::abs (vote_x - center_votes[2 * i    ]);
                float y_diff   = std::abs (vote_y - center_votes[2 * i + 1]);

                if (x_diff > voting_allowance || y_diff > voting_allowance)
                    continue;

                x_sum += vote_x * belief;
                y_sum += vote_y * belief;
                votes += belief;
            }
        }
        obj->bbox[i].x = x_sum / votes;
        obj->bbox[i].y = y_sum / votes;
    }

    free (center_votes);
}


static bool
IsIdentical (const object_t& box_1, const object_t& box_2)
{
    float voting_allowance = 1.0f;
    for (int i = 0; i < 8; i ++)
    {
        const float x_diff = std::abs(box_1.bbox[i].x - box_2.bbox[i].x);
        const float y_diff = std::abs(box_1.bbox[i].y - box_2.bbox[i].y);
        if (x_diff > voting_allowance || y_diff > voting_allowance)
        {
            return false;
        }
    }
    return true;
}


static bool
IsNewBox (std::list<object_t> *obj_list, object_t *obj_item)
{
    for (auto& b : *obj_list)
    {
        if (IsIdentical (b, *obj_item))
        {
            if (b.belief < obj_item->belief)
            {
                std::swap (b, *obj_item);
            }
            return false;
        }
    }
    return true;
}



static int
Lift2DTo3D(
    const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>& projection_matrix,
    bool portrait, object_t *obj)
{
    const float fx = projection_matrix(0, 0);
    const float fy = projection_matrix(1, 1);
    const float cx = projection_matrix(0, 2);
    const float cy = projection_matrix(1, 2);

    Eigen::Matrix<float, 16, 12, Eigen::RowMajor> m = Eigen::Matrix<float, 16, 12, Eigen::RowMajor>::Zero(16, 12);

    float u, v;
    for (int i = 0; i < 8; ++i)
    {
        fvec2 keypoint2d = obj->bbox[i];
        if (portrait)
        {
            // swap x and y given that our image is in portrait orientation
            u = keypoint2d.y * 2 - 1;
            v = keypoint2d.x * 2 - 1;
        } else 
        {
            u = keypoint2d.x * 2 - 1;
            v = 1 - keypoint2d.y * 2;  // (1 - keypoint2d.y()) * 2 - 1
        }
        
        for (int j = 0; j < 4; ++j)
        {
            // For each of the 4 control points, formulate two rows of the
            // m matrix (two equations).
            const float control_alpha = epnp_alpha_(i, j);
            m(i * 2,     j * 3    ) =       fx * control_alpha;
            m(i * 2,     j * 3 + 2) = (cx + u) * control_alpha;
            m(i * 2 + 1, j * 3 + 1) =       fy * control_alpha;
            m(i * 2 + 1, j * 3 + 2) = (cy + v) * control_alpha;
        }
    }

    // This is a self adjoint matrix. Use SelfAdjointEigenSolver for a fast
    // and stable solution.
    Eigen::Matrix<float, 12, 12, Eigen::RowMajor> mt_m = m.transpose() * m;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 12, 12, Eigen::RowMajor>> eigen_solver(mt_m);
    if (eigen_solver.info() != Eigen::Success)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    if (12 != eigen_solver.eigenvalues().size())
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    // Eigenvalues are sorted in increasing order for SelfAdjointEigenSolver
    // only! If you use other Eigen Solvers, it's not guaranteed to be in
    // increasing order. Here, we just take the eigen vector corresponding
    // to first/smallest eigen value, since we used SelfAdjointEigenSolver.
    Eigen::VectorXf eigen_vec = eigen_solver.eigenvectors().col(0);
    Eigen::Map<Eigen::Matrix<float, 4, 3, Eigen::RowMajor>> control_matrix(
        eigen_vec.data());
    if (control_matrix(0, 2) > 0) {
      control_matrix = -control_matrix;
    }

    // First set the center keypoint.
    obj->center3d.x = control_matrix(0, 0);
    obj->center3d.y = control_matrix(0, 1);
    obj->center3d.z = control_matrix(0, 2);

    // Then set the 8 vertices.
    Eigen::Matrix<float, 8, 3, Eigen::RowMajor> vertices = epnp_alpha_ * control_matrix;

    for (int i = 0; i < 8; ++i) 
    {
        obj->bbox3d[i].x = vertices(i, 0);
        obj->bbox3d[i].y = vertices(i, 1);
        obj->bbox3d[i].z = vertices(i, 2);
    }

    return 0;
}

static void
Project3DTo2D (bool portrait, object_t *object)
{
    fvec3 *bbox3d = object->bbox3d;
    fvec2 *bbox2d = object->bbox2d;

    for (int i = 0; i < 8; i ++)
    {
        Eigen::Vector4f point3d;
        point3d << bbox3d[i].x, bbox3d[i].y, bbox3d[i].z, 1.0f;

        Eigen::Vector4f point3d_projection = projection_matrix_ * point3d;

        float u, v;
        const float inv_w = 1.0f / point3d_projection(3);

        if (portrait)
        {
            u = (point3d_projection(1) * inv_w + 1.0f) * 0.5f;
            v = (point3d_projection(0) * inv_w + 1.0f) * 0.5f;
        }
        else
        {
            u = (point3d_projection(0) * inv_w + 1.0f) * 0.5f;
            v = (1.0f - point3d_projection(1) * inv_w) * 0.5f;
        }

        bbox2d[i].x = u;
        bbox2d[i].y = v;
    }
}



static void
pack_objectron_result (objectron_result_t *objectron_result, std::list<object_t> &bbox_list)
{
    int num_obj = 0;
    for (auto itr = bbox_list.begin(); itr != bbox_list.end(); itr ++)
    {
        object_t obj_item = *itr;

        memcpy (&objectron_result->objects[num_obj], &obj_item, sizeof (obj_item));

        num_obj++;
        objectron_result->num = num_obj;

        if (num_obj >= MAX_OBJECT_NUM)
            break;
    }
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_objectron (objectron_result_t *objectron_result)
{
    if (s_detect_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    int ofstmap_w = s_detect_tensor_offsetmap.dims[2];
    int ofstmap_h = s_detect_tensor_offsetmap.dims[1];
#if 0
    float offset_scalex = std::min (ofstmap_w, ofstmap_h);
    float offset_scaley = std::min (ofstmap_w, ofstmap_h);
#else
    float offset_scalex = ofstmap_w;
    float offset_scaley = ofstmap_h;
#endif

    std::list<fvec2> center_points;
    extract_center_keypoints (center_points);

    std::list<object_t> obj_list;
    for (auto &center_point : center_points)
    {
        int cx = static_cast<int>(std::round(center_point.x));
        int cy = static_cast<int>(std::round(center_point.y));
        object_t obj_item = {0};

        obj_item.belief = get_heatmap_val (cx, cy);
        decode_by_voting (cx, cy, offset_scalex, offset_scaley, &obj_item);

        /* eliminate duplicate bbox */
        if (!IsNewBox (&obj_list, &obj_item))
        {
            continue;
        }

        for (int i = 0; i < 8; i ++)
        {
            obj_item.bbox[i].x /= offset_scalex;
            obj_item.bbox[i].y /= offset_scaley;
        }

        Lift2DTo3D (projection_matrix_, /*portrait*/ true, &obj_item);
        Project3DTo2D (/*portrait*/ true, &obj_item);

        obj_item.center_x = center_point.x / (float)ofstmap_w;
        obj_item.center_y = center_point.y / (float)ofstmap_h;
        obj_list.push_back (obj_item);
    }

    pack_objectron_result (objectron_result, obj_list);

    return 0;
}
