/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_trt.h"
#include "trt_age_gender.h"
#include <unistd.h>

/* 
 * https://github.com/yu4u/age-gender-estimation
 */
#define FACE_DETECT_ONNX_MODEL_PATH     "./models/dbface_keras_256x256_float32_nhwc.onnx"
#define FACE_DETECT_PLAN_MODEL_PATH     "./models/dbface_keras_256x256_float32_nhwc.plan"

#define ONNX_MODEL_PATH     "./models/EfficientNetB3_224_weights.11-3.44.onnx"
#define PLAN_MODEL_PATH     "./models/EfficientNetB3_224_weights.11-3.44.plan"


static IExecutionContext   *s_detect_trt_context;
static trt_tensor_t         s_detect_tensor_input;
static trt_tensor_t         s_detect_tensor_hm;
static trt_tensor_t         s_detect_tensor_box;
static trt_tensor_t         s_detect_tensor_landmark;
static std::vector<void *>  s_detect_gpu_buffers;

static IExecutionContext   *s_trt_context;
static trt_tensor_t         s_tensor_input;
static trt_tensor_t         s_tensor_age;
static trt_tensor_t         s_tensor_gender;
static std::vector<void *>  s_gpu_buffers;


/* -------------------------------------------------- *
 *  create cuda engine
 * -------------------------------------------------- */
static int
convert_onnx_to_plan (const std::string &plan_file_name, const std::string &uff_file_name)
{
    ICudaEngine *engine;
    engine = trt_create_engine_from_onnx (uff_file_name);
    if (!engine)
    {
        fprintf (stderr, "ERR:%s(%d): Failed to load graph from file.\n", __FILE__, __LINE__);
        return -1;
    }

    trt_emit_plan_file (engine, plan_file_name);

    engine->destroy();

    return 0;
}


/* -------------------------------------------------- *
 *  Create TensorRT Interpreter
 * -------------------------------------------------- */
static ICudaEngine *
load_trt_engine (const char *plan_path, const char *onnx_path)
{
    ICudaEngine *engine = NULL;

    /* Try to load Prebuilt TensorRT Engine */
    fprintf (stderr, "loading prebuilt TensorRT engine...\n");
    engine = trt_load_plan_file (plan_path);

    /* Build TensorRT Engine */
    if (engine == NULL)
    {
        convert_onnx_to_plan (plan_path, onnx_path);

        engine = trt_load_plan_file (plan_path);
        if (engine == NULL)
        {
            fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
            return NULL;
        }
    }
    return engine;
}

int
init_trt_age_gender (face_detect_config_t *config)
{
    trt_initialize ();

    /* ---------------------------------- *
     *  TRT-Engine for Face Detect
     * ---------------------------------- */
    {
        ICudaEngine *engine = load_trt_engine (FACE_DETECT_PLAN_MODEL_PATH, FACE_DETECT_ONNX_MODEL_PATH);

        /* Allocate IO tensors */
        trt_get_tensor_by_name (engine, "input:0",      &s_detect_tensor_input);
        trt_get_tensor_by_name (engine, "Identity_2:0", &s_detect_tensor_hm);         /* 1, h, w,  1 */
        trt_get_tensor_by_name (engine, "Identity_1:0", &s_detect_tensor_box);        /* 1, h, w,  4 */
        trt_get_tensor_by_name (engine, "Identity:0",   &s_detect_tensor_landmark);   /* 1, h, w, 10 */

        int num_bindings = engine->getNbBindings();
        s_detect_gpu_buffers.resize (num_bindings);
        s_detect_gpu_buffers[s_detect_tensor_input   .bind_idx] = s_detect_tensor_input   .gpu_mem;
        s_detect_gpu_buffers[s_detect_tensor_hm      .bind_idx] = s_detect_tensor_hm      .gpu_mem;
        s_detect_gpu_buffers[s_detect_tensor_box     .bind_idx] = s_detect_tensor_box     .gpu_mem;
        s_detect_gpu_buffers[s_detect_tensor_landmark.bind_idx] = s_detect_tensor_landmark.gpu_mem;

        s_detect_trt_context = engine->createExecutionContext();
    }

    /* ---------------------------------- *
     *  TRT-Engine for Age Gender estimation
     * ---------------------------------- */
    {
        ICudaEngine *engine = load_trt_engine (PLAN_MODEL_PATH, ONNX_MODEL_PATH);

        /* Allocate IO tensors */
        trt_get_tensor_by_name (engine, "input_1:0",    &s_tensor_input);
        trt_get_tensor_by_name (engine, "Identity:0",   &s_tensor_age);         /* 1, 101 */
        trt_get_tensor_by_name (engine, "Identity_1:0", &s_tensor_gender);      /* 1, 2 */

        int num_bindings = engine->getNbBindings();
        s_gpu_buffers.resize (num_bindings);
        s_gpu_buffers[s_tensor_input .bind_idx] = s_tensor_input .gpu_mem;
        s_gpu_buffers[s_tensor_age   .bind_idx] = s_tensor_age   .gpu_mem;
        s_gpu_buffers[s_tensor_gender.bind_idx] = s_tensor_gender.gpu_mem;

        s_trt_context = engine->createExecutionContext();
    }

    /* config parameters */
    config->score_thresh = 0.3f;
    config->iou_thresh   = 0.3f;

    return 0;
}

void *
get_face_detect_input_buf (int *w, int *h)
{
    *w = s_detect_tensor_input.dims.d[2];
    *h = s_detect_tensor_input.dims.d[1];
    return s_detect_tensor_input.cpu_mem;
}

void *
get_age_gender_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims.d[2];
    *h = s_tensor_input.dims.d[1];
    return s_tensor_input.cpu_mem;
}


/* -------------------------------------------------- *
 * Invoke TensorRT (Face detection)
 * -------------------------------------------------- */
static float *
get_bbox_ptr (int anchor_idx)
{
    int idx = 4 * anchor_idx;
    float *bboxes_ptr = (float *)s_detect_tensor_box.cpu_mem;

    return &bboxes_ptr[idx];
}

static float *
get_landmark_ptr (int anchor_idx)
{
    int idx = 10 * anchor_idx;
    float *landmark_ptr = (float *)s_detect_tensor_landmark.cpu_mem;

    return &landmark_ptr[idx];
}

static float
_exp (float v)
{
    if (fabs (v) < 1.0f)
        return v * expf (1.0f);

    if (v > 0.0f)
        return expf (v);
    else
        return -expf (-v);
}


static int
decode_bounds (std::list<face_t> &face_list, float score_thresh)
{
    face_t face_item;
    float  *scores_ptr = (float *)s_detect_tensor_hm.cpu_mem;
    int score_w = s_detect_tensor_hm.dims.d[2];
    int score_h = s_detect_tensor_hm.dims.d[1];

    for (int y = 0; y < score_h; y ++)
    {
        for (int x = 0; x < score_w; x ++)
        {
            int idx = y * score_w + x;
            float score = scores_ptr[idx];

            if (score < score_thresh)
                continue;

            float *p = get_bbox_ptr (idx);
            float bx = p[0];
            float by = p[1];
            float bw = p[2];
            float bh = p[3];

            fvec2 topleft, btmright;
            topleft.x  = (x - bx) / (float)score_w;
            topleft.y  = (y - by) / (float)score_h;
            btmright.x = (x + bw) / (float)score_w;
            btmright.y = (y + bh) / (float)score_h;

            face_item.score    = score;
            face_item.topleft  = topleft;
            face_item.btmright = btmright;

            /* landmark positions (5 keys) */
            float *lm = get_landmark_ptr (idx);
            for (int j = 0; j < kFaceKeyNum; j ++)
            {
                float lx = lm[j    ] * 4;
                float ly = lm[j + 5] * 4;
                lx = (_exp (lx) + x) / (float)score_w;
                ly = (_exp (ly) + y) / (float)score_h;

                face_item.keys[j].x = lx;
                face_item.keys[j].y = ly;
            }

            face_list.push_back (face_item);
        }
    }
    return 0;
}


/* -------------------------------------------------- *
 *  Apply NonMaxSuppression:
 *      https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/image_ops.ts
 * -------------------------------------------------- */
static float
calc_intersection_over_union (face_t &face0, face_t &face1)
{
    float sx0 = face0.topleft.x;
    float sy0 = face0.topleft.y;
    float ex0 = face0.btmright.x;
    float ey0 = face0.btmright.y;
    float sx1 = face1.topleft.x;
    float sy1 = face1.topleft.y;
    float ex1 = face1.btmright.x;
    float ey1 = face1.btmright.y;
    
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
compare (face_t &v1, face_t &v2)
{
    if (v1.score > v2.score)
        return true;
    else
        return false;
}

static int
non_max_suppression (std::list<face_t> &face_list, std::list<face_t> &face_sel_list, float iou_thresh)
{
    face_list.sort (compare);

    for (auto itr = face_list.begin(); itr != face_list.end(); itr ++)
    {
        face_t face_candidate = *itr;

        int ignore_candidate = false;
        for (auto itr_sel = face_sel_list.rbegin(); itr_sel != face_sel_list.rend(); itr_sel ++)
        {
            face_t face_sel = *itr_sel;

            float iou = calc_intersection_over_union (face_candidate, face_sel);
            if (iou >= iou_thresh)
            {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate)
        {
            face_sel_list.push_back(face_candidate);
            if (face_sel_list.size() >= MAX_FACE_NUM)
                break;
        }
    }

    return 0;
}

/* -------------------------------------------------- *
 *  Compute ROI region
 * -------------------------------------------------- */
static float
normalize_radians (float angle)
{
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

static void
compute_rotation (face_t &face)
{
    float x0 = face.keys[kRightEye].x;
    float y0 = face.keys[kRightEye].y;
    float x1 = face.keys[kLeftEye].x;
    float y1 = face.keys[kLeftEye].y;

    float target_angle = 0;//M_PI * 0.5f;
    float rotation = target_angle - std::atan2(-(y1 - y0), x1 - x0);

    face.rotation = normalize_radians (rotation);
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
compute_face_rect (face_t &face)
{
#define CROP_SCALE 2.0
    float width    = face.btmright.x - face.topleft.x;
    float height   = face.btmright.y - face.topleft.y;
    float palm_cx  = face.topleft.x + width  * 0.5f;
    float palm_cy  = face.topleft.y + height * 0.5f;
    float face_cx;
    float face_cy;
    float rotation = face.rotation;
    float shift_x = 0;// 0.0f;
    float shift_y = -0.3f;

    if (rotation == 0.0f)
    {
        face_cx = palm_cx + (width  * shift_x);
        face_cy = palm_cy + (height * shift_y);
    }
    else
    {
        float dx = (width  * shift_x) * std::cos(rotation) -
                   (height * shift_y) * std::sin(rotation);
        float dy = (width  * shift_x) * std::sin(rotation) +
                   (height * shift_y) * std::cos(rotation);
        face_cx = palm_cx + dx;
        face_cy = palm_cy + dy;
    }

    float long_side = std::max (width, height);
    width  = long_side;
    height = long_side;
    float face_w = width  * CROP_SCALE;
    float face_h = height * CROP_SCALE;

    face.face_cx = face_cx;
    face.face_cy = face_cy;
    face.face_w  = face_w;
    face.face_h  = face_h;

    float dx = face_w * 0.5f;
    float dy = face_h * 0.5f;

    face.face_pos[0].x = - dx;  face.face_pos[0].y = - dy;
    face.face_pos[1].x = + dx;  face.face_pos[1].y = - dy;
    face.face_pos[2].x = + dx;  face.face_pos[2].y = + dy;
    face.face_pos[3].x = - dx;  face.face_pos[3].y = + dy;

    for (int i = 0; i < 4; i ++)
    {
        rot_vec (face.face_pos[i], rotation);
        face.face_pos[i].x += face_cx;
        face.face_pos[i].y += face_cy;
    }
}


static void
pack_face_result (face_detect_result_t *facedet_result, std::list<face_t> &face_list)
{
    int num_faces = 0;
    for (auto itr = face_list.begin(); itr != face_list.end(); itr ++)
    {
        face_t face = *itr;

        compute_rotation (face);
        compute_face_rect (face);

        memcpy (&facedet_result->faces[num_faces], &face, sizeof (face));
        num_faces ++;
        facedet_result->num = num_faces;

        if (num_faces >= MAX_FACE_NUM)
            break;
    }
}


/* -------------------------------------------------- *
 * Invoke TensorRT
 * -------------------------------------------------- */
int
invoke_face_detect (face_detect_result_t *facedet_result, face_detect_config_t *config)
{
    /* copy to CUDA buffer */
    trt_copy_tensor_to_gpu (s_detect_tensor_input);

    /* invoke inference */
    int batchSize = 1;
    s_detect_trt_context->execute (batchSize, &s_detect_gpu_buffers[0]);

    /* copy from CUDA buffer */
    trt_copy_tensor_from_gpu (s_detect_tensor_hm);
    trt_copy_tensor_from_gpu (s_detect_tensor_box);
    trt_copy_tensor_from_gpu (s_detect_tensor_landmark);


    /* decode boundary box and landmark keypoints */
    float score_thresh = config->score_thresh;
    std::list<face_t> face_list;

    decode_bounds (face_list, score_thresh);

#if 1 /* USE NMS */
    float iou_thresh = config->iou_thresh;
    std::list<face_t> face_nms_list;

    non_max_suppression (face_list, face_nms_list, iou_thresh);
    pack_face_result (facedet_result, face_nms_list);
#else
    pack_face_result (facedet_result, face_list);
#endif

    return 0;
}


static bool
compare_age (age_t &v1, age_t &v2)
{
    if (v1.score > v2.score)
        return true;
    else
        return false;
}

static void
decode_ages (std::list<age_t> &age_list)
{
    age_t age_item;
    float *ages_ptr = (float *)s_tensor_age.cpu_mem;
    int num_age     = s_tensor_age.dims.d[1];
    for (int i = 0; i < num_age; i ++)
    {
        age_item.age   = i;
        age_item.score = ages_ptr[i];
        age_list.push_back (age_item);
    }

    age_list.sort (compare_age);
}

int
invoke_age_gender (age_gender_result_t *age_gender_result)
{
    /* copy to CUDA buffer */
    trt_copy_tensor_to_gpu (s_tensor_input);

    /* invoke inference */
    int batchSize = 1;
    s_trt_context->execute (batchSize, &s_gpu_buffers[0]);

    /* copy from CUDA buffer */
    trt_copy_tensor_from_gpu (s_tensor_age);
    trt_copy_tensor_from_gpu (s_tensor_gender);

    std::list<age_t> age_list;
    decode_ages (age_list);

    //for (auto itr = age_list.begin(); itr != age_list.end(); itr ++)
    //{
    //    age_t age_item = *itr;
    //    fprintf (stderr, "%2d: %f\n", age_item.age, age_item.score);
    //}
    
    float *gender_ptr = (float *)s_tensor_gender.cpu_mem;
    float score_m = gender_ptr[1];
    float score_f = gender_ptr[0];
    //fprintf (stderr, "gender(%f, %f)\n", score_m, score_f);

    age_t age_item = age_list.front();
    age_gender_result->age.age   = age_item.age;
    age_gender_result->age.score = age_item.score;
    age_gender_result->gender.score_m = score_m;
    age_gender_result->gender.score_f = score_f;
    return 0;
}

