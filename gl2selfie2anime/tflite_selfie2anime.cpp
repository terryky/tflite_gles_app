/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_selfie2anime.h"
#include <list>

/* 
 * https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite
 * https://github.com/margaretmz/selfie2anime-e2e-tutorial/blob/master/android/selfie2anime/app/src/main/ml/selfie2anime.tflite
 */
#define FACE_DETECT_MODEL_PATH      "./model/face_detection_front.tflite"
#define SELFIE2ANIME_MODEL_PATH     "./model/selfie2anime.tflite"


static tflite_interpreter_t s_detect_interpreter;
static tflite_tensor_t      s_detect_tensor_input;
static tflite_tensor_t      s_detect_tensor_scores;
static tflite_tensor_t      s_detect_tensor_bboxes;

static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_segment;

static std::list<fvec2> s_anchors;

/*
 * determine where the anchor points are scatterd.
 *   https://github.com/tensorflow/tfjs-models/blob/master/blazeface/src/face.ts
 */
static int
create_blazeface_anchors(int input_w, int input_h)
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
init_tflite_selfie2anime()
{
    /* Face detect */
    tflite_create_interpreter_from_file (&s_detect_interpreter, FACE_DETECT_MODEL_PATH);
    tflite_get_tensor_by_name (&s_detect_interpreter, 0, "input",          &s_detect_tensor_input);
    tflite_get_tensor_by_name (&s_detect_interpreter, 1, "regressors",     &s_detect_tensor_bboxes);
    tflite_get_tensor_by_name (&s_detect_interpreter, 1, "classificators", &s_detect_tensor_scores);

    /* Selfie2Anime */
    tflite_create_interpreter_from_file (&s_interpreter, SELFIE2ANIME_MODEL_PATH);
    tflite_get_tensor_by_name (&s_interpreter, 0, "test_domain_A",  &s_tensor_input);
    tflite_get_tensor_by_name (&s_interpreter, 1, "generator_B/Tanh",  &s_tensor_segment);

    int det_input_w = s_detect_tensor_input.dims[2];
    int det_input_h = s_detect_tensor_input.dims[1];
    create_blazeface_anchors (det_input_w, det_input_h);

    return 0;
}

int
get_selfie2anime_input_type ()
{
    if (s_tensor_input.type == kTfLiteUInt8)
        return 1;
    else
        return 0;
}

void *
get_face_detect_input_buf (int *w, int *h)
{
    *w = s_detect_tensor_input.dims[2];
    *h = s_detect_tensor_input.dims[1];
    return s_detect_tensor_input.ptr;
}

void *
get_selfie2anime_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return s_tensor_input.ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite (Face detection)
 * -------------------------------------------------- */
static float *
get_bbox_ptr (int anchor_idx)
{
    int idx = 16 * anchor_idx;
    float *bboxes_ptr = (float *)s_detect_tensor_bboxes.ptr;

    return &bboxes_ptr[idx];
}

static int
decode_bounds (std::list<face_t> &face_list, float score_thresh, int input_img_w, int input_img_h)
{
    face_t face_item;
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

            face_item.score    = score;
            face_item.topleft  = topleft;
            face_item.btmright = btmright;

            /* landmark positions (6 keys) */
            for (int j = 0; j < kFaceKeyNum; j ++)
            {
                float lx = p[4 + (2 * j) + 0];
                float ly = p[4 + (2 * j) + 1];
                lx += anchor.x;
                ly += anchor.y;
                lx /= (float)input_img_w;
                ly /= (float)input_img_h;

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
 *  Scale bbox
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
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_face_detect (face_detect_result_t *facedet_result)
{
    if (s_detect_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* decode boundary box and landmark keypoints */
    float score_thresh = 0.75f;
    std::list<face_t> face_list;

    int input_img_w = s_detect_tensor_input.dims[2];
    int input_img_h = s_detect_tensor_input.dims[1];
    decode_bounds (face_list, score_thresh, input_img_w, input_img_h);


#if 1 /* USE NMS */
    float iou_thresh = 0.3f;
    std::list<face_t> face_nms_list;

    non_max_suppression (face_list, face_nms_list, iou_thresh);
    pack_face_result (facedet_result, face_nms_list);
#else
    pack_face_result (facedet_result, face_list);
#endif

    return 0;
}


int
invoke_selfie2anime (selfie2anime_result_t *selfie2anime_result)
{
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 1
    int w = s_tensor_segment.dims[2];
    int h = s_tensor_segment.dims[1];
    int c = s_tensor_segment.dims[3];
    int memsize = w * h * c * sizeof (float);

    if (selfie2anime_result->segmentmap == NULL)
    {
        selfie2anime_result->segmentmap = (float *)malloc (memsize);
    }
    memcpy (selfie2anime_result->segmentmap, s_tensor_segment.ptr, memsize);
#else
    selfie2anime_result->segmentmap         = (float *)s_tensor_segment.ptr;
#endif
    selfie2anime_result->segmentmap_dims[0] = s_tensor_segment.dims[2];
    selfie2anime_result->segmentmap_dims[1] = s_tensor_segment.dims[1];
    selfie2anime_result->segmentmap_dims[2] = s_tensor_segment.dims[3];

    return 0;
}

