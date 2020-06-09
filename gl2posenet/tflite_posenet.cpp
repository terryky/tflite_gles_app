/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_posenet.h"
#include "ssbo_tensor.h"
#include <list>
#include <float.h>

/* 
 * [float]
 *   https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite
 *   https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_513x513_multi_kpt_stripped.tflite
 *
 * [int8 quant]
 *   https://github.com/PINTO0309/PINTO_model_zoo/tree/master/03_posenet/01_posenet_v1/03_integer_quantization
 */
#define POSENET_MODEL_PATH          "./posenet_model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
#define POSENET_QUANT_MODEL_PATH    "./posenet_model/model-mobilenet_v1_101_257_integer_quant.tflite"

static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_heatmap;
static tflite_tensor_t      s_tensor_offsets;
static tflite_tensor_t      s_tensor_fw_disp;
static tflite_tensor_t      s_tensor_bw_disp;

static int     s_img_w = 0;
static int     s_img_h = 0;
static int     s_hmp_w = 0;
static int     s_hmp_h = 0;
static int     s_edge_num = 0;

typedef struct part_score_t {
    float score;
    int   idx_x;
    int   idx_y;
    int   key_id;
} part_score_t;

typedef struct keypoint_t {
    float pos_x;
    float pos_y;
    float score;
    int   valid;
} keypoint_t;


static int pose_edges[][2] =
{
    /* parent,        child */
    { kNose,          kLeftEye      },  //  0
    { kLeftEye,       kLeftEar      },  //  1
    { kNose,          kRightEye     },  //  2
    { kRightEye,      kRightEar     },  //  3
    { kNose,          kLeftShoulder },  //  4
    { kLeftShoulder,  kLeftElbow    },  //  5
    { kLeftElbow,     kLeftWrist    },  //  6
    { kLeftShoulder,  kLeftHip      },  //  7
    { kLeftHip,       kLeftKnee     },  //  8
    { kLeftKnee,      kLeftAnkle    },  //  9
    { kNose,          kRightShoulder},  // 10
    { kRightShoulder, kRightElbow   },  // 11
    { kRightElbow,    kRightWrist   },  // 12
    { kRightShoulder, kRightHip     },  // 13
    { kRightHip,      kRightKnee    },  // 14
    { kRightKnee,     kRightAnkle   },  // 15
};


int
init_tflite_posenet(int use_quantized_tflite, ssbo_t *ssbo)
{
    const char *posenet_model;

    if (use_quantized_tflite)
    {
        posenet_model = POSENET_QUANT_MODEL_PATH;
        tflite_create_interpreter_from_file (&s_interpreter, posenet_model);
        tflite_get_tensor_by_name (&s_interpreter, 0, "image",              &s_tensor_input);
        tflite_get_tensor_by_name (&s_interpreter, 1, "heatmap",            &s_tensor_heatmap);
        tflite_get_tensor_by_name (&s_interpreter, 1, "offset_2",           &s_tensor_offsets);
        tflite_get_tensor_by_name (&s_interpreter, 1, "displacement_fwd_2", &s_tensor_fw_disp);
        tflite_get_tensor_by_name (&s_interpreter, 1, "displacement_bwd_2", &s_tensor_bw_disp);
    }
    else
    {
        posenet_model = POSENET_MODEL_PATH;
        tflite_create_interpreter_from_file (&s_interpreter, posenet_model);
        tflite_get_tensor_by_name (&s_interpreter, 0, "sub_2",                                  &s_tensor_input);
        tflite_get_tensor_by_name (&s_interpreter, 1, "MobilenetV1/heatmap_2/BiasAdd",          &s_tensor_heatmap);
        tflite_get_tensor_by_name (&s_interpreter, 1, "MobilenetV1/offset_2/BiasAdd",           &s_tensor_offsets);
        tflite_get_tensor_by_name (&s_interpreter, 1, "MobilenetV1/displacement_fwd_2/BiasAdd", &s_tensor_fw_disp);
        tflite_get_tensor_by_name (&s_interpreter, 1, "MobilenetV1/displacement_bwd_2/BiasAdd", &s_tensor_bw_disp);
    }

    /* input image dimention */
    s_img_w = s_tensor_input.dims[2];
    s_img_h = s_tensor_input.dims[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* heatmap dimention */
    s_hmp_w = s_tensor_heatmap.dims[2];
    s_hmp_h = s_tensor_heatmap.dims[1];
    fprintf (stderr, "heatmap size: (%d, %d)\n", s_hmp_w, s_hmp_h);

    /* displacement forward vector dimention */
    s_edge_num = s_tensor_fw_disp.dims[3] / 2;

    return 0;
}

void *
get_posenet_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return s_tensor_input.ptr;
}

static float
get_heatmap_score (int idx_y, int idx_x, int key_id)
{
    int idx = (idx_y * s_hmp_w * kPoseKeyNum) + (idx_x * kPoseKeyNum) + key_id;
    float *heatmap_ptr = (float *)s_tensor_heatmap.ptr;
    return heatmap_ptr[idx];
}

static void
get_displacement_vector (void *disp_buf, float *dis_x, float *dis_y, int idx_y, int idx_x, int edge_id)
{
    int idx0 = (idx_y * s_hmp_w * s_edge_num*2) + (idx_x * s_edge_num*2) + (edge_id + s_edge_num);
    int idx1 = (idx_y * s_hmp_w * s_edge_num*2) + (idx_x * s_edge_num*2) + (edge_id);

    float *disp_buf_fp = (float *)disp_buf;
    *dis_x = disp_buf_fp[idx0];
    *dis_y = disp_buf_fp[idx1];
}

static void
get_offset_vector (float *ofst_x, float *ofst_y, int idx_y, int idx_x, int pose_id)
{
    int idx0 = (idx_y * s_hmp_w * kPoseKeyNum*2) + (idx_x * kPoseKeyNum*2) + (pose_id + kPoseKeyNum);
    int idx1 = (idx_y * s_hmp_w * kPoseKeyNum*2) + (idx_x * kPoseKeyNum*2) + (pose_id);
    float *offsets_ptr = (float *)s_tensor_offsets.ptr;

    *ofst_x = offsets_ptr[idx0];
    *ofst_y = offsets_ptr[idx1];
}

/* enqueue an item in descending order. */
static void
enqueue_score (std::list<part_score_t> &queue, int x, int y, int key, float score)
{
    std::list<part_score_t>::iterator itr;
    for (itr = queue.begin(); itr != queue.end(); itr++)
    {
        if (itr->score < score)
            break;
    }

    part_score_t item;
    item.score = score;
    item.idx_x = x;
    item.idx_y = y;
    item.key_id= key;
    queue.insert(itr, item);
}

/*
 * If the score is the highest in local window, return true.
 *
 *    xs    xe
 *   +--+--+--+
 *   |  |  |  | ys
 *   +--+--+--+
 *   |  |##|  |         ##: (idx_x, idx_y)
 *   +--+--+--+
 *   |  |  |  | ye
 *   +--+--+--+
 */
static bool
score_is_max_in_local_window (int key, float score, int idx_y, int idx_x, int max_rad)
{
    int xs = std::max (idx_x - max_rad,     0);
    int ys = std::max (idx_y - max_rad,     0);
    int xe = std::min (idx_x + max_rad + 1, s_hmp_w);
    int ye = std::min (idx_y + max_rad + 1, s_hmp_h);

    for (int y = ys; y < ye; y ++)
    {
        for (int x = xs; x < xe; x ++)
        {
            /* if a higher score is found, return false */
            if (get_heatmap_score (y, x, key) > score)
                return false;
        }
    }
    return true;
}

static void
build_score_queue (std::list<part_score_t> &queue, float thresh, int max_rad)
{
    for (int y = 0; y < s_hmp_h; y ++)
    {
        for (int x = 0; x < s_hmp_w; x ++)
        {
            for (int key = 0; key < kPoseKeyNum; key ++)
            {
                float score = get_heatmap_score (y, x, key);

                /* if this score is lower than thresh, skip this pixel. */
                if (score < thresh)
                    continue;

                /* if there is a higher score near this pixel, skip this pixel. */
                if (!score_is_max_in_local_window (key, score, y, x, max_rad))
                    continue;

                enqueue_score (queue, x, y, key, score);
            }
        }
    }
}

/*
 *  0      28.5    57.1    85.6   114.2   142.7   171.3   199.9   228.4   257   [pos_x]
 *  |---+---|---+---|---+---|---+---|---+---|---+---|---+---|---+---|---+---|
 *     0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0      [hmp_pos_x]
 */
static void
get_pos_to_near_index (float pos_x, float pos_y, int *idx_x, int *idx_y)
{
    float ratio_x = pos_x / (float)s_img_w;
    float ratio_y = pos_y / (float)s_img_h;

    float hmp_pos_x = ratio_x * (s_hmp_w - 1);
    float hmp_pos_y = ratio_y * (s_hmp_h - 1);

    int hmp_idx_x = roundf (hmp_pos_x);
    int hmp_idx_y = roundf (hmp_pos_y);

    hmp_idx_x = std::min (hmp_idx_x, s_hmp_w -1);
    hmp_idx_y = std::min (hmp_idx_y, s_hmp_h -1);
    hmp_idx_x = std::max (hmp_idx_x, 0);
    hmp_idx_y = std::max (hmp_idx_y, 0);

    *idx_x = hmp_idx_x;
    *idx_y = hmp_idx_y;
}

static void
get_index_to_pos (int idx_x, int idx_y, int key_id, float *pos_x, float *pos_y)
{
    float ofst_x, ofst_y;
    get_offset_vector (&ofst_x, &ofst_y, idx_y, idx_x, key_id);

    float rel_x = (float)idx_x / (float)(s_hmp_w -1);
    float rel_y = (float)idx_y / (float)(s_hmp_h -1);

    float pos0_x = rel_x * s_img_w;
    float pos0_y = rel_y * s_img_h;

    *pos_x = pos0_x + ofst_x;
    *pos_y = pos0_y + ofst_y;
}


static keypoint_t
traverse_to_tgt_key(int edge, keypoint_t src_key, int tgt_key_id, void *disp)
{
    float src_pos_x = src_key.pos_x;
    float src_pos_y = src_key.pos_y;

    int src_idx_x, src_idx_y;
    get_pos_to_near_index (src_pos_x, src_pos_y, &src_idx_x, &src_idx_y);

    /* get displacement vector from source to target */
    float disp_x, disp_y;
    get_displacement_vector (disp, &disp_x, &disp_y, src_idx_y, src_idx_x, edge);

    /* calculate target position */
    float tgt_pos_x = src_pos_x + disp_x;
    float tgt_pos_y = src_pos_y + disp_y;

    int tgt_idx_x, tgt_idx_y;
    int offset_refine_step = 2;
    for (int i = 0; i < offset_refine_step; i ++)
    {
        get_pos_to_near_index (tgt_pos_x, tgt_pos_y, &tgt_idx_x, &tgt_idx_y);
        get_index_to_pos (tgt_idx_x, tgt_idx_y, tgt_key_id, &tgt_pos_x, &tgt_pos_y);
    }

    keypoint_t tgt_key = {0};
    tgt_key.pos_x = tgt_pos_x;
    tgt_key.pos_y = tgt_pos_y;
    tgt_key.score = get_heatmap_score (tgt_idx_y, tgt_idx_x, tgt_key_id);
    tgt_key.valid = 1;

    return tgt_key;
}

static void
decode_pose (part_score_t &root, keypoint_t *keys)
{
    /* calculate root key position. */
    int idx_x = root.idx_x;
    int idx_y = root.idx_y;
    int keyid = root.key_id;
    float *fw_disp_ptr = (float *)s_tensor_fw_disp.ptr;
    float *bw_disp_ptr = (float *)s_tensor_bw_disp.ptr;

    float pos_x, pos_y;
    get_index_to_pos (idx_x, idx_y, keyid, &pos_x, &pos_y);

    keys[keyid].pos_x = pos_x;
    keys[keyid].pos_y = pos_y;
    keys[keyid].score = root.score;
    keys[keyid].valid = 1;

    for (int edge = s_edge_num - 1; edge >= 0; edge --)
    {
        int src_key_id = pose_edges[edge][1];
        int tgt_key_id = pose_edges[edge][0];

        if ( keys[src_key_id].valid &&
            !keys[tgt_key_id].valid)
        {
            keys[tgt_key_id] = traverse_to_tgt_key(edge, keys[src_key_id], tgt_key_id, bw_disp_ptr);
        }
    }

    for (int edge = 0; edge < s_edge_num; edge ++)
    {
        int src_key_id = pose_edges[edge][0];
        int tgt_key_id = pose_edges[edge][1];

        if ( keys[src_key_id].valid &&
            !keys[tgt_key_id].valid)
        {
            keys[tgt_key_id] = traverse_to_tgt_key(edge, keys[src_key_id], tgt_key_id, fw_disp_ptr);
        }
    }
}

static bool
within_nms_of_corresponding_point (posenet_result_t *pose_result,
                        float pos_x, float pos_y, int key_id, float nms_rad)
{
    for (int i = 0; i < pose_result->num; i ++)
    {
        pose_t *pose = &pose_result->pose[i];
        float prev_pos_x = pose->key[key_id].x * s_img_w;
        float prev_pos_y = pose->key[key_id].y * s_img_h;

        float dx = pos_x - prev_pos_x;
        float dy = pos_y - prev_pos_y;
        float len = (dx * dx) + (dy * dy);

        if (len <= (nms_rad * nms_rad))
            return true;
    }
    return false;
}

static float
get_instance_score (posenet_result_t *pose_result, keypoint_t *keys, float nms_rad)
{
    float score_total = 0.0f;
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float pos_x = keys[i].pos_x;
        float pos_y = keys[i].pos_y;
        if (within_nms_of_corresponding_point (pose_result, pos_x, pos_y, i, nms_rad))
            continue;

        score_total += keys[i].score;
    }
    return score_total / (float)kPoseKeyNum;
}

static int
regist_detected_pose (posenet_result_t *pose_result, keypoint_t *keys, float score)
{
    int pose_id = pose_result->num;
    if (pose_id >= MAX_POSE_NUM)
    {
        fprintf (stderr, "ERR: %s(%d): pose_num overflow.\n", __FILE__, __LINE__);
        return -1;
    }

    for (int i = 0; i < kPoseKeyNum; i++)
    {
        pose_result->pose[pose_id].key[i].x     = keys[i].pos_x / (float)s_img_w;
        pose_result->pose[pose_id].key[i].y     = keys[i].pos_y / (float)s_img_h;
        pose_result->pose[pose_id].key[i].score = keys[i].score;
    }

    pose_result->pose[pose_id].pose_score = score;
    pose_result->num ++;

    return 0;
}


static void
decode_multiple_poses (posenet_result_t *pose_result)
{
    std::list<part_score_t> queue;

    float score_thresh  = 0.5f;
    int   local_max_rad = 1;
    build_score_queue (queue, score_thresh, local_max_rad);

    memset (pose_result, 0, sizeof (posenet_result_t));
    while (pose_result->num < MAX_POSE_NUM && !queue.empty())
    {
        part_score_t &root = queue.front();

        float pos_x, pos_y;
        get_index_to_pos (root.idx_x, root.idx_y, root.key_id, &pos_x, &pos_y);

        float nms_rad = 20.0f;
        if (within_nms_of_corresponding_point (pose_result, pos_x, pos_y, root.key_id, nms_rad))
        {
            queue.pop_front();
            continue;
        }

        keypoint_t key_points[kPoseKeyNum] = {0};
        decode_pose (root, key_points);

        float score = get_instance_score (pose_result, key_points, nms_rad);
        regist_detected_pose (pose_result, key_points, score);

        queue.pop_front();
    }
}

static void
decode_single_pose (posenet_result_t *pose_result)
{
    int   max_block_idx[kPoseKeyNum][2] = {0};
    float max_block_cnf[kPoseKeyNum]    = {0};

    /* find the highest heatmap block for each key */
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float max_confidence = -FLT_MAX;
        for (int y = 0; y < s_hmp_h; y ++)
        {
            for (int x = 0; x < s_hmp_w; x ++)
            {
                float confidence = get_heatmap_score (y, x, i);
                if (confidence > max_confidence)
                {
                    max_confidence = confidence;
                    max_block_cnf[i] = confidence;
                    max_block_idx[i][0] = x;
                    max_block_idx[i][1] = y;
                }
            }
        }
    }

    /* find the offset vector and calculate the keypoint coordinates. */
    for (int i = 0; i < kPoseKeyNum;i ++ )
    {
        int idx_x = max_block_idx[i][0];
        int idx_y = max_block_idx[i][1];
        float key_posex, key_posey;
        get_index_to_pos (idx_x, idx_y, i, &key_posex, &key_posey);

        pose_result->pose[0].key[i].x     = key_posex / (float)s_img_w;
        pose_result->pose[0].key[i].y     = key_posey / (float)s_img_h;
        pose_result->pose[0].key[i].score = max_block_cnf[i];
    }
    pose_result->num = 1;
    pose_result->pose[0].pose_score = 1.0f;
}

int
invoke_posenet (posenet_result_t *pose_result)
{
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /*
     * decode algorithm is from:
     *   https://github.com/tensorflow/tfjs-models/tree/master/posenet/src/multi_pose
     */
    if (1)
        decode_multiple_poses (pose_result);
    else
        decode_single_pose (pose_result);

    pose_result->pose[0].heatmap = s_tensor_heatmap.ptr;
    pose_result->pose[0].heatmap_dims[0] = s_hmp_w;
    pose_result->pose[0].heatmap_dims[1] = s_hmp_h;

    return 0;
}

