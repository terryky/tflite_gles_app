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
#include "tflite_style_transfer.h"
#include <float.h>

#if 1
#define STYLE_PREDICT_MODEL_PATH  "./style_transfer_model/style_predict_quantized_256.tflite"
//#define STYLE_TRANSFER_MODEL_PATH "./style_transfer_model/style_transfer_quantized_dynamic.tflite"
#define STYLE_TRANSFER_MODEL_PATH "./style_transfer_model/style_transfer_quantized_384.tflite"
#else
#define STYLE_PREDICT_MODEL_PATH  "./style_transfer_model/style_predict_f16_256.tflite"
#define STYLE_TRANSFER_MODEL_PATH "./style_transfer_model/style_transfer_f16_384.tflite"
#endif

using namespace std;
using namespace tflite;

typedef struct tflite_interpreter_t
{
    unique_ptr<FlatBufferModel>     model;
    unique_ptr<Interpreter>         interpreter;
    ops::builtin::BuiltinOpResolver resolver;
} tflite_interpreter_t;


static tflite_interpreter_t s_interpreter_style_predict;
static float    *s_style_predict_in_ptr;
static int      s_style_predict_img_w = 0;
static int      s_style_predict_img_h = 0;


static tflite_interpreter_t s_interpreter_style_transfer;
static float    *s_style_transfer_style_in_ptr;
static float    *s_style_transfer_content_in_ptr;
static int      s_style_transfer_style_dim;
static int      s_style_transfer_img_w = 0;
static int      s_style_transfer_img_h = 0;



static void
print_tensor_dim (unique_ptr<Interpreter> &interpreter, int tensor_id)
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
print_tensor_info (unique_ptr<Interpreter> &interpreter)
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
        print_tensor_dim (interpreter, idx);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, " Output Tensor Dimension\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    for (i = 0; i < out_size; i ++)
    {
        idx = interpreter->outputs()[i];
        fprintf (stderr, "Tensor[%2d]: ", idx);
        print_tensor_dim (interpreter, idx);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    PrintInterpreterState(interpreter.get());
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
}


static int
create_interpreter(tflite_interpreter_t *p, const char *model_path)
{
    p->model = FlatBufferModel::BuildFromFile(model_path);
    if (!p->model)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    InterpreterBuilder(*(p->model), p->resolver)(&(p->interpreter));
    if (!p->interpreter)
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

    if (p->interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
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
    if (p->interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
#endif

    p->interpreter->SetNumThreads(4);
    if (p->interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 0 /* for debug */
    print_tensor_info (p->interpreter);
#endif

    return 0;
}

static int
init_tflite_style_predict(tflite_interpreter_t *p, const char *model_path)
{
    create_interpreter (p, model_path);

    s_style_predict_in_ptr  = p->interpreter->typed_input_tensor<float>(0);

    /* input image dimention */
    int input_idx = p->interpreter->inputs()[0];
    TfLiteIntArray *in_dim = p->interpreter->tensor(input_idx)->dims;
    s_style_predict_img_w = in_dim->data[2];
    s_style_predict_img_h = in_dim->data[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_style_predict_img_w, s_style_predict_img_h);

    /* style output  dimention */
    int output_idx = p->interpreter->outputs()[0];
    TfLiteIntArray *out_dim = p->interpreter->tensor(output_idx)->dims;
    int s0 = out_dim->data[0];
    int s1 = out_dim->data[1];
    int s2 = out_dim->data[2];
    int s3 = out_dim->data[3];
    fprintf (stderr, "style output dim: (%dx%dx%dx%d)\n", s0, s1, s2, s3);

    return 0;
}

static int
init_tflite_style_trans(tflite_interpreter_t *p, const char *model_path)
{
    create_interpreter (p, model_path);

    s_style_transfer_content_in_ptr = p->interpreter->typed_input_tensor<float>(0);
    s_style_transfer_style_in_ptr   = p->interpreter->typed_input_tensor<float>(1);

    /* input image dimention */
    int input_content_idx = p->interpreter->inputs()[0];
    TfLiteIntArray *in_content_dim = p->interpreter->tensor(input_content_idx)->dims;
    s_style_transfer_img_w = in_content_dim->data[2];
    s_style_transfer_img_h = in_content_dim->data[1];
    fprintf (stderr, "input image size: [%d](%d, %d)\n", input_content_idx, s_style_transfer_img_w, s_style_transfer_img_h);

    int input_style_idx = p->interpreter->inputs()[1];
    TfLiteIntArray *in_style_dim = p->interpreter->tensor(input_style_idx)->dims;
    s_style_transfer_style_dim = in_style_dim->data[3];
    fprintf (stderr, "input image size: [%d](%d)\n", input_style_idx, s_style_transfer_style_dim);

    int output_style_idx = p->interpreter->outputs()[0];
    TfLiteIntArray *out_dim = p->interpreter->tensor(output_style_idx)->dims;
    int s0 = out_dim->data[0];
    int s1 = out_dim->data[1];
    int s2 = out_dim->data[2];
    int s3 = out_dim->data[3];
    fprintf (stderr, "style output dim: (%dx%dx%dx%d)\n", s0, s1, s2, s3);

    return 0;
}


int
init_tflite_style_transfer ()
{
    init_tflite_style_predict (&s_interpreter_style_predict,  STYLE_PREDICT_MODEL_PATH);
    init_tflite_style_trans   (&s_interpreter_style_transfer, STYLE_TRANSFER_MODEL_PATH);

    return 0;
}

void *
get_style_predict_input_buf (int *w, int *h)
{
    *w = s_style_predict_img_w;
    *h = s_style_predict_img_h;
    return s_style_predict_in_ptr;
}

void *
get_style_transfer_style_input_buf (int *size)
{
    s_style_transfer_style_in_ptr = s_interpreter_style_transfer.interpreter->typed_input_tensor<float>(1);

    *size = s_style_transfer_style_dim;
    return s_style_transfer_style_in_ptr;
}

void *
get_style_transfer_content_input_buf (int *w, int *h)
{
    s_style_transfer_content_in_ptr = s_interpreter_style_transfer.interpreter->typed_input_tensor<float>(0);

    *w = s_style_transfer_img_w;
    *h = s_style_transfer_img_h;
    return s_style_transfer_content_in_ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_style_predict (style_predict_t *predict_result)
{
    if (s_interpreter_style_predict.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    int output_idx = s_interpreter_style_predict.interpreter->outputs()[0];
    TfLiteIntArray *out_dim = s_interpreter_style_predict.interpreter->tensor(output_idx)->dims;
    //int s0 = out_dim->data[0];
    //int s1 = out_dim->data[1];
    //int s2 = out_dim->data[2];
    int s3 = out_dim->data[3];

    predict_result->size  = s3;
    predict_result->param = s_interpreter_style_predict.interpreter->typed_output_tensor<float>(0); 
    
    return 0;
}


int
invoke_style_transfer (style_transfer_t *transfered_result)
{
    if (s_interpreter_style_transfer.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    int output_idx = s_interpreter_style_transfer.interpreter->outputs()[0];
    TfLiteIntArray *out_dim = s_interpreter_style_transfer.interpreter->tensor(output_idx)->dims;
    //int s0 = out_dim->data[0];
    int s1 = out_dim->data[1];
    int s2 = out_dim->data[2];
    //int s3 = out_dim->data[3];
    
    transfered_result->h = s1;
    transfered_result->w = s2;
    transfered_result->img = s_interpreter_style_transfer.interpreter->typed_output_tensor<float>(0); 

    return 0;
}
