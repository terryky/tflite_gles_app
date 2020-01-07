/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
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
#include "tflite_blazeface.h"
#include <float.h>

/* 
 * https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite
 */
#define BLAZEFACE_MODEL_PATH  "./blazeface_model/face_detection_front.tflite"

using namespace std;
using namespace tflite;

unique_ptr<FlatBufferModel> model;
unique_ptr<Interpreter> interpreter;
ops::builtin::BuiltinOpResolver resolver;

static float   *in_ptr;
static float   *classific_ptr;
static float   *regressor_ptr;

static int     s_img_w = 0;
static int     s_img_h = 0;
static int     s_anchor_num = 0;



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
init_tflite_blazeface()
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
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
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

#if 1 /* for debug */
    print_tensor_info ();
#endif

    in_ptr        = interpreter->typed_input_tensor<float>(0);
    regressor_ptr = interpreter->typed_output_tensor<float>(0);
    classific_ptr = interpreter->typed_output_tensor<float>(1);

    /* input image dimention */
    int input_idx = interpreter->inputs()[0];
    TfLiteIntArray *dim = interpreter->tensor(input_idx)->dims;
    s_img_w = dim->data[2];
    s_img_h = dim->data[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* regressors dimention */
    int regressor_idx = interpreter->outputs()[0];
    TfLiteIntArray *rgr_dim = interpreter->tensor(regressor_idx)->dims;
    int r0 = rgr_dim->data[0];
    int r1 = rgr_dim->data[1];
    int r2 = rgr_dim->data[2];
    fprintf (stderr, "regressors dim    : (%dx%dx%d)\n", r0, r1, r2);

    /* classificators dimention */
    int classificator_idx = interpreter->outputs()[1];
    TfLiteIntArray *cls_dim = interpreter->tensor(classificator_idx)->dims;
    int c0 = cls_dim->data[0];
    int c1 = cls_dim->data[1];
    int c2 = cls_dim->data[2];
    fprintf (stderr, "classificators dim: (%dx%dx%d)\n", c0, c1, c2);

    s_anchor_num = c1;

    return 0;
}

void *
get_blazeface_input_buf (int *w, int *h)
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


static float *
get_anchor_ptr (int anchor_idx)
{
    int idx = 16 * anchor_idx;
    return &regressor_ptr[idx];
}

int
invoke_blazeface (blazeface_result_t *face_result)
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

#if 0
    for (int i = 0; i < s_anchor_num; i ++)
    {
        if ((i % 10 == 0))
            fprintf (stderr, "\n[%3d] ", i);
        if (classific_ptr[i] > 0)
            fprintf (stderr, "%6.1f ", classific_ptr[i]);
        else
            fprintf (stderr, "%6.1f ", 0.0f);
    }
#endif

    for (int i = 0; i < s_anchor_num; i ++)
    {
        if (classific_ptr[i] > 0)
        {
            float *p = get_anchor_ptr (i);
            fprintf (stderr, "[%3d] %6.1f: (%f, %f) (%f, %f)\n", i, classific_ptr[i], p[0], p[1], p[2], p[3]);
        }
    }
    
    return 0;
}

