/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_dbface.h"
#include <list>

/* 
 * https://github.com/PINTO0309/PINTO_model_zoo/tree/master/041_DBFace/01_float32
 * https://github.com/PINTO0309/PINTO_model_zoo/tree/master/041_DBFace/03_integer_quantization
 */
//#define DBFACE_MODEL_PATH        "./model/dbface_keras_256x256_float32_nhwc.tflite"
//#define DBFACE_QUANT_MODEL_PATH  "./model/dbface_keras_256x256_integer_quant_nhwc.tflite"

#define DBFACE_MODEL_PATH        "./model/dbface_keras_480x640_float32_nhwc.tflite"
#define DBFACE_QUANT_MODEL_PATH  "./model/dbface_keras_480x640_integer_quant_nhwc.tflite"

static tflite_interpreter_t s_detect_interpreter;
static tflite_tensor_t      s_detect_tensor_input;
static tflite_tensor_t      s_detect_tensor_hm;
static tflite_tensor_t      s_detect_tensor_box;
static tflite_tensor_t      s_detect_tensor_landmark;





/* -------------------------------------------------- *
 *  Create TFLite Interpreter
 * -------------------------------------------------- */
int
init_tflite_dbface(int use_quantized_tflite, dbface_config_t *config)
{
    const char *dbface_model;

    if (use_quantized_tflite)
    {
        dbface_model = DBFACE_QUANT_MODEL_PATH;
    }
    else
    {
        dbface_model = DBFACE_MODEL_PATH;
    }

    /* Face detect */
    tflite_create_interpreter_from_file (&s_detect_interpreter, dbface_model);
    tflite_get_tensor_by_name (&s_detect_interpreter, 0, "input",          &s_detect_tensor_input);
    tflite_get_tensor_by_name (&s_detect_interpreter, 1, "Identity_2",     &s_detect_tensor_hm);
    tflite_get_tensor_by_name (&s_detect_interpreter, 1, "Identity_1",     &s_detect_tensor_box);
    tflite_get_tensor_by_name (&s_detect_interpreter, 1, "Identity",       &s_detect_tensor_landmark);

    config->score_thresh = 0.3f;
    config->iou_thresh   = 0.3f;

    return 0;
}

void *
get_dbface_input_buf (int *w, int *h)
{
    *w = s_detect_tensor_input.dims[2];
    *h = s_detect_tensor_input.dims[1];
    return s_detect_tensor_input.ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite (Face detection)
 * -------------------------------------------------- */
static float *
get_bbox_ptr (int anchor_idx)
{
    int idx = 4 * anchor_idx;
    float *bboxes_ptr = (float *)s_detect_tensor_box.ptr;

    return &bboxes_ptr[idx];
}

static float *
get_landmark_ptr (int anchor_idx)
{
    int idx = 10 * anchor_idx;
    float *landmark_ptr = (float *)s_detect_tensor_landmark.ptr;

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
    float  *scores_ptr = (float *)s_detect_tensor_hm.ptr;
    int score_w = s_detect_tensor_hm.dims[2];
    int score_h = s_detect_tensor_hm.dims[1];

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

static void
pack_face_result (dbface_result_t *face_result, std::list<face_t> &face_list)
{
    int num_faces = 0;
    for (auto itr = face_list.begin(); itr != face_list.end(); itr ++)
    {
        face_t face = *itr;
        memcpy (&face_result->faces[num_faces], &face, sizeof (face));
        num_faces ++;
        face_result->num = num_faces;

        if (num_faces >= MAX_FACE_NUM)
            break;
    }
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_dbface (dbface_result_t *face_result, dbface_config_t *config)
{
    if (s_detect_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* decode boundary box and landmark keypoints */
    float score_thresh = config->score_thresh;
    std::list<face_t> face_list;

    decode_bounds (face_list, score_thresh);

#if 1 /* USE NMS */
    float iou_thresh = config->iou_thresh;
    std::list<face_t> face_nms_list;

    non_max_suppression (face_list, face_nms_list, iou_thresh);
    pack_face_result (face_result, face_nms_list);
#else
    pack_face_result (face_result, face_list);
#endif

    return 0;
}

