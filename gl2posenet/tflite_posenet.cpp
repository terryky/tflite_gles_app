/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tflite_posenet.h"
#include <float.h>

/* 
 * https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite
 * https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_513x513_multi_kpt_stripped.tflite
 */
#define POSENET_MODEL_PATH  "./posenet_model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

using namespace std;
using namespace tflite;

unique_ptr<FlatBufferModel> model;
unique_ptr<Interpreter> interpreter;
ops::builtin::BuiltinOpResolver resolver;

static float   *in_ptr;
static float   *heatmap_ptr;
static float   *offsets_ptr;
static float   *fw_disp_ptr;
static float   *bw_disp_ptr;
static int     s_img_w = 0;
static int     s_img_h = 0;
static int     s_hmp_w = 0;
static int     s_hmp_h = 0;

static void
print_tensor_dim (int tensor_id)
{
    TfLiteIntArray *dim = interpreter->tensor(tensor_id)->dims;
    
    for (int i = 0; i < dim->size; i ++)
    {
        if (i > 0)
            fprintf (stderr, "x");
        fprintf (stderr, "%d", dim->data[i]);
    }
    fprintf (stderr, "\n");
}

static void
print_tensor_info ()
{
    int i, idx;
    int in_size  = interpreter->inputs().size();
    int out_size = interpreter->outputs().size();

    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, "tensors size     : %ld\n", interpreter->tensors_size());
    fprintf (stderr, "nodes   size     : %ld\n", interpreter->nodes_size());
    fprintf (stderr, "number of inputs : %d\n", in_size);
    fprintf (stderr, "number of outputs: %d\n", out_size);
    fprintf (stderr, "input(0) name    : %s\n", interpreter->GetInputName(0));

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, "                     name                     bytes  type  scale   zero_point\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    int t_size = interpreter->tensors_size();
    for (i = 0; i < t_size; i++) 
    {
        fprintf (stderr, "Tensor[%2d] %-32s %8ld, %2d, %f, %3d\n", i, 
            interpreter->tensor(i)->name, 
            interpreter->tensor(i)->bytes,
            interpreter->tensor(i)->type,
            interpreter->tensor(i)->params.scale,
            interpreter->tensor(i)->params.zero_point);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, " Input Tensor Dimension\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    for (i = 0; i < in_size; i ++)
    {
        idx = interpreter->inputs()[i];
        fprintf (stderr, "Tensor[%2d]: ", idx);
        print_tensor_dim (idx);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, " Output Tensor Dimension\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    for (i = 0; i < out_size; i ++)
    {
        idx = interpreter->outputs()[i];
        fprintf (stderr, "Tensor[%2d]: ", idx);
        print_tensor_dim (idx);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    PrintInterpreterState(interpreter.get());
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
}

int
init_tflite_posenet()
{
    model = FlatBufferModel::BuildFromFile(POSENET_MODEL_PATH);
    if (!model)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 1 /* for debug */
    print_tensor_info ();
#endif
    
    in_ptr      = interpreter->typed_input_tensor<float>(0);
    heatmap_ptr = interpreter->typed_output_tensor<float>(0);
    offsets_ptr = interpreter->typed_output_tensor<float>(1);
    fw_disp_ptr = interpreter->typed_output_tensor<float>(2);
    bw_disp_ptr = interpreter->typed_output_tensor<float>(3);

    /* input image dimention */
    int input_idx = interpreter->inputs()[0];
    TfLiteIntArray *dim = interpreter->tensor(input_idx)->dims;
    s_img_w = dim->data[2];
    s_img_h = dim->data[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* heatmap dimention */
    int heatmap_idx = interpreter->outputs()[0];
    TfLiteIntArray *hmp_dim = interpreter->tensor(heatmap_idx)->dims;
    s_hmp_w = hmp_dim->data[2];
    s_hmp_h = hmp_dim->data[1];
    fprintf (stderr, "heatmap size: (%d, %d)\n", s_hmp_w, s_hmp_h);

    return 0;
}

void *
get_posenet_input_buf (int *w, int *h)
{
    *w = s_img_w;
    *h = s_img_h;
    return in_ptr;
}

#if 0 /* for debug */
static void
load_dummy_img()
{
    FILE *fp;
    
    if ((fp = fopen ("000000000139_RGB888_SIZE300x300.img", "rb" )) == NULL)
    {
        fprintf (stderr, "Can't open img file.\n");
        return;
    }

    fseek (fp, 0, SEEK_END);
    size_t len = ftell (fp);
    fseek( fp, 0, SEEK_SET );

    void *buf = malloc(len);
    if (fread (buf, 1, len, fp) < len)
    {
        fprintf (stderr, "Can't read img file.\n");
        return;
    }

    memcpy (in_ptr, buf, 300 * 300 * 3);

    free (buf);
}

static void
save_img ()
{
    static int s_cnt = 0;
    FILE *fp;
    char strFName[ 128 ];
    int w = s_img_w;
    int h = s_img_h;
    
    sprintf (strFName, "detect%04d_RGB888_SIZE%dx%d.img", s_cnt, w, h);
    s_cnt ++;

    fp = fopen (strFName, "wb");
    if (fp == NULL)
    {
        fprintf (stderr, "FATAL ERROR at %s(%d)\n", __FILE__, __LINE__);
        return;
    }

    fwrite (in_ptr, 3, w * h, fp);
    fclose (fp);
}
#endif


int
invoke_posenet (posenet_result_t *pose_result)
{
#if 0 /* load dummy image. */
    load_dummy_img();
#endif

#if 0 /* dump image of DL network input. */
    save_img ();
#endif

    if (interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    float *heatmap = heatmap_ptr;
    float *offsets = offsets_ptr;
    int n_poses = 1;
    int x, y;
    int   max_block_idx[kPoseKeyNum][2] = {0};
    float max_block_cnf[kPoseKeyNum]    = {0};
    
    /* find the highest heatmap block for each key */
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float max_confidence = -FLT_MAX;
        for (y = 0; y < s_hmp_h; y ++)
        {
            for (x = 0; x < s_hmp_w; x ++)
            {
                float confidence = heatmap[(y * s_hmp_w * kPoseKeyNum)+ (x * kPoseKeyNum) + i];
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
        float ofst_x = offsets[(idx_y * s_hmp_w * kPoseKeyNum*2) + (idx_x * kPoseKeyNum*2) + (i + kPoseKeyNum)];
        float ofst_y = offsets[(idx_y * s_hmp_w * kPoseKeyNum*2) + (idx_x * kPoseKeyNum*2) + (i)              ];
        float key_posex = (float)idx_x / (float)(s_hmp_w - 1) * s_img_w + ofst_x;
        float key_posey = (float)idx_y / (float)(s_hmp_h - 1) * s_img_h + ofst_y;

        pose_result->pose[0].key[i].x     = key_posex / (float)s_img_w;
        pose_result->pose[0].key[i].y     = key_posey / (float)s_img_h;
        pose_result->pose[0].key[i].score = max_block_cnf[i];
    }

    pose_result->num = n_poses;
    pose_result->pose[0].pose_score = 1.0f;
    pose_result->pose[0].heatmap    = heatmap;
    pose_result->pose[0].heatmap_dims[0] = s_hmp_w;
    pose_result->pose[0].heatmap_dims[1] = s_hmp_h;

    return 0;
}

