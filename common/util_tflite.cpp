/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"


using namespace std;
using namespace tflite;


typedef struct tflite_obj_in_t
{
    unique_ptr<FlatBufferModel>     model;
    unique_ptr<Interpreter>         interpreter;
    ops::builtin::BuiltinOpResolver resolver;
} tflite_obj_in_t;



static void
print_tensor_dim (TfLiteTensor *tensor)
{
    TfLiteIntArray *dim = tensor->dims;

    fprintf (stderr, "[");
    for (int i = 0; i < dim->size; i ++)
    {
        if (i > 0)
            fprintf (stderr, "x");
        fprintf (stderr, "%d", dim->data[i]);
    }
    fprintf (stderr, "]");
}

static const char *
get_tflite_type_str (TfLiteType type)
{
    switch (type)
    {
    case kTfLiteNoType:     return "none";
    case kTfLiteFloat32:    return "fp32";
    case kTfLiteInt32:      return " i32";
    case kTfLiteUInt8:      return "ui32";
    case kTfLiteInt64:      return " i64";
    case kTfLiteString:     return "str ";
    case kTfLiteBool:       return "bool";
    case kTfLiteInt16:      return " i16";
    case kTfLiteComplex64:  return "cp64";
    case kTfLiteInt8:       return " i8 ";
    case kTfLiteFloat16 :   return "fp16";
    }

    return "xxxx";
}

static void
print_tensor (TfLiteTensor *tensor, int idx)
{
    fprintf (stderr, "Tensor[%3d] %8zu, %2d(%s), (%3d, %8.6f) %-32s ", idx,
        tensor->bytes,
        tensor->type,
        get_tflite_type_str (tensor->type),
        tensor->params.zero_point,
        tensor->params.scale,
        tensor->name);
    
    print_tensor_dim (tensor);
    fprintf (stderr, "\n");
}

void
dump_tflite_model (tflite_obj_t tflite_obj)
{
    tflite_obj_in_t *pobj = (tflite_obj_in_t *)tflite_obj;
    unique_ptr<Interpreter> &interpreter = pobj->interpreter;

    int i, idx;
    int in_size  = interpreter->inputs().size();
    int out_size = interpreter->outputs().size();

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, "       T E N S O R S\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, "tensors size     : %zu\n", interpreter->tensors_size());
    fprintf (stderr, "nodes   size     : %zu\n", interpreter->nodes_size());
    fprintf (stderr, "number of inputs : %d\n", in_size);
    fprintf (stderr, "number of outputs: %d\n", out_size);

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, "               bytes,   type  , (quant params),     name,                dimention \n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    int t_size = interpreter->tensors_size();
    for (i = 0; i < t_size; i++) 
    {
        TfLiteTensor *tensor = interpreter->tensor(i);
        print_tensor (tensor, i);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, " Input Tensor Dimension\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    for (i = 0; i < in_size; i ++)
    {
        idx = interpreter->inputs()[i];
        TfLiteTensor *tensor = interpreter->tensor(idx);
        print_tensor (tensor, idx);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, " Output Tensor Dimension\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    for (i = 0; i < out_size; i ++)
    {
        idx = interpreter->outputs()[i];
        TfLiteTensor *tensor = interpreter->tensor(idx);
        print_tensor (tensor, idx);
    }
    fprintf (stderr, "\n");

#if 0
    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    PrintInterpreterState(interpreter.get());
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
#endif
}





tflite_obj_t
create_tflite_inferer (const char *model_path)
{
    unique_ptr<FlatBufferModel>     model;
    unique_ptr<Interpreter>         interpreter;
    ops::builtin::BuiltinOpResolver resolver;

    fprintf (stderr, "MODEL_PATH: %s\n", model_path);
    model = FlatBufferModel::BuildFromFile (model_path);
    if (!model)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    InterpreterBuilder (*model, resolver)(&interpreter);
    if (!interpreter)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
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
        return NULL;
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
        return NULL;
    }
#endif

    interpreter->SetNumThreads(4);
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    tflite_obj_in_t *pobj = new tflite_obj_in_t;
    if (pobj == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    pobj->model       = std::move (model);
    pobj->interpreter = std::move (interpreter);
    pobj->resolver    = resolver;

    return (tflite_obj_t)pobj;
}








