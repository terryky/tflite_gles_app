/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#if defined (USE_GL_DELEGATE)
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif
#if defined (USE_GPU_DELEGATEV2)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif
#include "tflite_handpose.h"
#include <float.h>

/* 
 * https://github.com/google/mediapipe/tree/master/mediapipe/models/hand_landmark_3d.tflite
 */
#define BLAZEFACE_MODEL_PATH  "./handpose_model/hand_landmark_3d.tflite"

using namespace std;
using namespace tflite;

unique_ptr<FlatBufferModel> model;
unique_ptr<Interpreter> interpreter;
ops::builtin::BuiltinOpResolver resolver;

static float   *in_ptr;
static float   *landmark_ptr;
static float   *handflag_ptr;

static int     s_img_w = 0;
static int     s_img_h = 0;


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
    fprintf (stderr, "tensors size     : %zu\n", interpreter->tensors_size());
    fprintf (stderr, "nodes   size     : %zu\n", interpreter->nodes_size());
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
        fprintf (stderr, "Tensor[%2d] %-32s %8zu, %2d, %f, %3d\n", i,
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
init_tflite_hand_landmark()
{
    model = FlatBufferModel::BuildFromFile(BLAZEFACE_MODEL_PATH);
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

#if defined (USE_GL_DELEGATE)
    const TfLiteGpuDelegateOptions options = {
        .metadata = NULL,
        .compile_options = {
            .precision_loss_allowed = 1,  // FP16
            .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
            .dynamic_batch_enabled = 0,   // Not fully functional yet
        },
    };
    auto* delegate = TfLiteGpuDelegateCreate(&options);

    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
#endif
#if defined (USE_GPU_DELEGATEV2)
    const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 1, // FP16
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
        .inference_priority1  = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
        .inference_priority2  = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
        .inference_priority3  = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    };
    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
#endif

    interpreter->SetNumThreads(4);
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 0 /* for debug */
    print_tensor_info ();
#endif

    in_ptr       = interpreter->typed_input_tensor<float>(0);
    landmark_ptr = interpreter->typed_output_tensor<float>(0);
    handflag_ptr = interpreter->typed_output_tensor<float>(1);

    /* input image dimention */
    int input_idx = interpreter->inputs()[0];
    TfLiteIntArray *dim = interpreter->tensor(input_idx)->dims;
    s_img_w = dim->data[2];
    s_img_h = dim->data[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* landmark dimention */
    int landmark_idx = interpreter->outputs()[0];
    TfLiteIntArray *landmark_dim = interpreter->tensor(landmark_idx)->dims;
    int l0 = landmark_dim->data[0];
    int l1 = landmark_dim->data[1];
    int l2 = landmark_dim->data[2];
    fprintf (stderr, "landmark dim    : (%dx%dx%d)\n", l0, l1, l2);

    /* handflag dimention */
    int handflag_idx = interpreter->outputs()[1];
    TfLiteIntArray *handflag_dim = interpreter->tensor(handflag_idx)->dims;
    int h0 = handflag_dim->data[0];
    int h1 = handflag_dim->data[1];
    int h2 = handflag_dim->data[2];
    fprintf (stderr, "handflag dim    : (%dx%dx%d)\n", h0, h1, h2);

    return 0;
}

void *
get_hand_landmark_input_buf (int *w, int *h)
{
    *w = s_img_w;
    *h = s_img_h;
    return in_ptr;
}



/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_hand_landmark (hand_landmark_result_t *hand_result)
{
    if (interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    hand_result->score = *handflag_ptr;
    //fprintf (stderr, "handflag = %f\n", *handflag_ptr);
    
    for (int i = 0; i < HAND_JOINT_NUM; i ++)
    {
        hand_result->joint[i].x = landmark_ptr[3 * i + 0] / (float)s_img_w;
        hand_result->joint[i].y = landmark_ptr[3 * i + 1] / (float)s_img_h;
        hand_result->joint[i].z = landmark_ptr[3 * i + 2];
        //fprintf (stderr, "[%2d] (%8.1f, %8.1f, %8.1f)\n", i, 
        //    landmark_ptr[3 * i + 0], landmark_ptr[3 * i + 1], landmark_ptr[3 * i + 2]);
    }

    return 0;
}

