/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_trt.h"
#include "trt_pose3d.h"
#include <unistd.h>
#include <float.h>

#define UFF_MODEL_PATH      "./models/human-pose-estimation-3d.onnx"
#define PLAN_MODEL_PATH     "./models/human-pose-estimation-3d.plan"


static IExecutionContext   *s_trt_context;
static trt_tensor_t         s_tensor_input;
static trt_tensor_t         s_tensor_heatmap;
static trt_tensor_t         s_tensor_offsets;
static trt_tensor_t         s_tensor_pafs;

static std::vector<void *>  s_gpu_buffers;

static int     s_img_w = 0;
static int     s_img_h = 0;
static int     s_hmp_w = 0;
static int     s_hmp_h = 0;

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
int
init_trt_pose3d (pose3d_config_t *config)
{
    ICudaEngine *engine = NULL;

    trt_initialize ();

    /* Try to load Prebuilt TensorRT Engine */
    fprintf (stderr, "loading prebuilt TensorRT engine...\n");
    engine = trt_load_plan_file (PLAN_MODEL_PATH);

    /* Build TensorRT Engine */
    if (engine == NULL)
    {
        convert_onnx_to_plan (PLAN_MODEL_PATH, UFF_MODEL_PATH);

        engine = trt_load_plan_file (PLAN_MODEL_PATH);
        if (engine == NULL)
        {
            fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    }

    s_trt_context = engine->createExecutionContext();


    /* Allocate IO tensors */
    trt_get_tensor_by_name (engine, "data:0",       &s_tensor_input);   /* (1, 256, 448,  3) */
    trt_get_tensor_by_name (engine, "Identity:0",   &s_tensor_offsets); /* (1,  32,  56, 57) */
    trt_get_tensor_by_name (engine, "Identity_1:0", &s_tensor_heatmap); /* (1,  32,  56  19) */
    trt_get_tensor_by_name (engine, "Identity_2:0", &s_tensor_pafs);    /* (1,  32,  56, 38) */

    int num_bindings = engine->getNbBindings();
    s_gpu_buffers.resize (num_bindings);
    s_gpu_buffers[s_tensor_input  .bind_idx] = s_tensor_input  .gpu_mem;
    s_gpu_buffers[s_tensor_heatmap.bind_idx] = s_tensor_heatmap.gpu_mem;
    s_gpu_buffers[s_tensor_offsets.bind_idx] = s_tensor_offsets.gpu_mem;
    s_gpu_buffers[s_tensor_pafs   .bind_idx] = s_tensor_pafs   .gpu_mem;

    config->score_thresh = 0.3f;
    config->iou_thresh   = 0.3f;

    /* input image dimention */
    s_img_w = s_tensor_input.dims.d[2];
    s_img_h = s_tensor_input.dims.d[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* heatmap dimention */
    s_hmp_w = s_tensor_heatmap.dims.d[2];
    s_hmp_h = s_tensor_heatmap.dims.d[1];
    fprintf (stderr, "heatmap size: (%d, %d)\n", s_hmp_w, s_hmp_h);

    return 0;
}

void *
get_pose3d_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims.d[2];
    *h = s_tensor_input.dims.d[1];
    return s_tensor_input.cpu_mem;
}


/* -------------------------------------------------- *
 * Invoke TensorRT
 * -------------------------------------------------- */
static float
get_heatmap_score (int idx_y, int idx_x, int key_id)
{
    int idx = (idx_y * s_hmp_w * kPoseKeyNum) + (idx_x * kPoseKeyNum) + key_id;
    float *heatmap_ptr = (float *)s_tensor_heatmap.cpu_mem;
    return heatmap_ptr[idx];
}

static void
get_offset_vector (float *ofst_x, float *ofst_y, float *ofst_z, int idx_y, int idx_x, int pose_id_)
{
    int map_id_to_panoptic[] = {1, 0,  9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18, 2};
    int pose_id = map_id_to_panoptic[pose_id_];

    int idx0 = (idx_y * s_hmp_w * kPoseKeyNum*3) + (idx_x * kPoseKeyNum*3) + (3 * pose_id + 0);
    int idx1 = (idx_y * s_hmp_w * kPoseKeyNum*3) + (idx_x * kPoseKeyNum*3) + (3 * pose_id + 1);
    int idx2 = (idx_y * s_hmp_w * kPoseKeyNum*3) + (idx_x * kPoseKeyNum*3) + (3 * pose_id + 2);

    float *offsets_ptr = (float *)s_tensor_offsets.cpu_mem;

    *ofst_x = offsets_ptr[idx0];
    *ofst_y = offsets_ptr[idx1];
    *ofst_z = offsets_ptr[idx2];
}




static void
get_index_to_pos (int idx_x, int idx_y, int key_id, fvec2 *pos2d, fvec3 *pos3d)
{
    float ofst_x, ofst_y, ofst_z;
    get_offset_vector (&ofst_x, &ofst_y, &ofst_z, idx_y, idx_x, key_id);

    /* pos 2D */
    pos2d->x = (float)idx_x / (float)(s_hmp_w -1);
    pos2d->y = (float)idx_y / (float)(s_hmp_h -1);

    /* pos 3D */
    pos3d->x = ofst_x;
    pos3d->y = ofst_y;
    pos3d->z = ofst_z;
}




static void
decode_multiple_poses (posenet_result_t *pose_result)
{
    memset (pose_result, 0, sizeof (posenet_result_t));
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

#if 0
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        fprintf (stderr, "---------[%d] --------\n", i);
        for (int y = 0; y < s_hmp_h; y ++)
        {
            fprintf (stderr, "[%d] ", y);
            for (int x = 0; x < s_hmp_w; x ++)
            {
                float confidence = get_heatmap_score (y, x, i);
                fprintf (stderr, "%6.3f ", confidence);

                if (x == max_block_idx[i][0] && y == max_block_idx[i][1])
                    fprintf (stderr, "#");
                else
                    fprintf (stderr, " ");
            }
            fprintf (stderr, "\n");
        }
    }
#endif

    /* find the offset vector and calculate the keypoint coordinates. */
    for (int i = 0; i < kPoseKeyNum;i ++ )
    {
        int idx_x = max_block_idx[i][0];
        int idx_y = max_block_idx[i][1];
        fvec2 pos2d;
        fvec3 pos3d;
        get_index_to_pos (idx_x, idx_y, i, &pos2d, &pos3d);

        pose_result->pose[0].key[i].x     = pos2d.x;
        pose_result->pose[0].key[i].y     = pos2d.y;
        pose_result->pose[0].key[i].score = max_block_cnf[i];

        pose_result->pose[0].key3d[i].x   = pos3d.x;
        pose_result->pose[0].key3d[i].y   = pos3d.y;
        pose_result->pose[0].key3d[i].z   = pos3d.z;
        pose_result->pose[0].key3d[i].score = max_block_cnf[i];
    }
    pose_result->num = 1;
    pose_result->pose[0].pose_score = 1.0f;
}


/* -------------------------------------------------- *
 * Invoke TensorRT
 * -------------------------------------------------- */
int
invoke_pose3d (posenet_result_t *pose_result)
{
    /* copy to CUDA buffer */
    trt_copy_tensor_to_gpu (s_tensor_input);

    /* invoke inference */
    int batchSize = 1;
    s_trt_context->execute (batchSize, &s_gpu_buffers[0]);

    /* copy from CUDA buffer */
    trt_copy_tensor_from_gpu (s_tensor_heatmap);
    trt_copy_tensor_from_gpu (s_tensor_offsets);
    trt_copy_tensor_from_gpu (s_tensor_pafs);


    if (0)
        decode_multiple_poses (pose_result);
    else
        decode_single_pose (pose_result);

    pose_result->pose[0].heatmap = s_tensor_heatmap.cpu_mem;
    pose_result->pose[0].heatmap_dims[0] = s_hmp_w;
    pose_result->pose[0].heatmap_dims[1] = s_hmp_h;

    return 0;
}

