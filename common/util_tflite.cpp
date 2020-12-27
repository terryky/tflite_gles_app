/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "util_debug.h"
#include <thread>

using namespace tflite;



static std::string
tflite_get_tensor_dim_str (TfLiteTensor *tensor)
{
    TfLiteIntArray *dim = tensor->dims;
    std::string str_dim;

    if (dim == NULL)
    {
        str_dim = "[]";
        return str_dim;
    }

    str_dim = "[";
    for (int i = 0; i < dim->size; i ++)
    {
        if (i > 0)
            str_dim += "x";
        str_dim += std::to_string (dim->data[i]);
    }
    str_dim += "]";

    return str_dim;
}

static const char *
tflite_get_type_str (TfLiteType type)
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
    case kTfLiteFloat16:    return "fp16";
    case kTfLiteFloat64:    return "fp64";
    default:                return "????";
    }

    return "xxxx";
}

static void
tflite_print_tensor (TfLiteTensor *tensor, int idx)
{
    if (tensor == NULL)
    {
        DBG_LOG ("Tensor[%3d]  \n", idx );
        return;
    }

    std::string str_dim = tflite_get_tensor_dim_str (tensor);
    DBG_LOG ("Tensor[%3d] %8zu, %2d(%s), (%3d, %8.6f) %-32s %s", idx,
        tensor->bytes,
        tensor->type,
        tflite_get_type_str (tensor->type),
        tensor->params.zero_point,
        tensor->params.scale,
        tensor->name,
        str_dim.c_str());

    DBG_LOG ("\n");
}

void
tflite_print_tensor_info (std::unique_ptr<Interpreter> &interpreter)
{
    int i, idx;
    int in_size  = interpreter->inputs().size();
    int out_size = interpreter->outputs().size();

    DBG_LOG ("-----------------------------------------------------------------------------\n");
    DBG_LOG ("       T E N S O R S\n");
    DBG_LOG ("-----------------------------------------------------------------------------\n");
    DBG_LOG ("tensors size     : %zu\n", interpreter->tensors_size());
    DBG_LOG ("nodes   size     : %zu\n", interpreter->nodes_size());
    DBG_LOG ("number of inputs : %d\n", in_size);
    DBG_LOG ("number of outputs: %d\n", out_size);
#if 0
    DBG_LOG ("\n");
    DBG_LOG ("-----------------------------------------------------------------------------\n");
    DBG_LOG ("                     name                     bytes  type  scale   zero_point\n");
    DBG_LOG ("-----------------------------------------------------------------------------\n");
    int t_size = interpreter->tensors_size();
    for (i = 0; i < t_size; i++) 
    {
        TfLiteTensor *tensor = interpreter->tensor(i);
        tflite_print_tensor (tensor, i);
    }
#endif
    DBG_LOG ("\n");
    DBG_LOG ("-----------------------------------------------------------------------------\n");
    DBG_LOG (" Input Tensor Dimension\n");
    DBG_LOG ("-----------------------------------------------------------------------------\n");
    for (i = 0; i < in_size; i ++)
    {
        idx = interpreter->inputs()[i];
        TfLiteTensor *tensor = interpreter->tensor(idx);
        tflite_print_tensor (tensor, idx);
    }

    DBG_LOG ("\n");
    DBG_LOG ("-----------------------------------------------------------------------------\n");
    DBG_LOG (" Output Tensor Dimension\n");
    DBG_LOG ("-----------------------------------------------------------------------------\n");
    for (i = 0; i < out_size; i ++)
    {
        idx = interpreter->outputs()[i];
        TfLiteTensor *tensor = interpreter->tensor(idx);
        tflite_print_tensor (tensor, idx);
    }
    DBG_LOG ("\n");

#if 0
    DBG_LOG ("\n");
    DBG_LOG ("-----------------------------------------------------------------------------\n");
    PrintInterpreterState(interpreter.get());
    DBG_LOG ("-----------------------------------------------------------------------------\n");
#endif
}


static int
modify_graph_with_delegate (tflite_interpreter_t *p, tflite_createopt_t *opt)
{
    TfLiteDelegate *delegate = NULL;

#if defined (USE_GL_DELEGATE)
    const TfLiteGpuDelegateOptions options = {
        .metadata = NULL,
        .compile_options = {
            .precision_loss_allowed = 1,  // FP16
            .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
            .dynamic_batch_enabled = 0,   // Not fully functional yet
        },
    };
    delegate = TfLiteGpuDelegateCreate(&options);

#if defined (USE_INPUT_SSBO)
    if (opt && opt->gpubuffer)
    {
        int ssbo_id = opt->gpubuffer;
        int tensor_index = p->interpreter->inputs()[0];

        TfLiteIntArray *dim = interpreter->tensor(tensor_index)->dims;
        if (TfLiteGpuDelegateBindBufferToTensor(delegate, ssbo_id, tensor_index) != kTfLiteOk)
        {
            DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    }
#endif
#endif

#if defined (USE_GPU_DELEGATEV2)
    const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 1, // FP16
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
        .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
        .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
        .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    };
    delegate = TfLiteGpuDelegateV2Create(&options);
#endif

#if defined (USE_NNAPI_DELEGATE)
    delegate = tflite::NnApiDelegate ();
#endif


#if defined (USE_HEXAGON_DELEGATE)
    // Assuming shared libraries are under "/data/local/tmp/"
    // If files are packaged with native lib in android App then it
    // will typically be equivalent to the path provided by
    // "getContext().getApplicationInfo().nativeLibraryDir"

    //const char library_directory_path[] = "/data/local/tmp/";
    //TfLiteHexagonInitWithPath(library_directory_path);  // Needed once at startup.

    TfLiteHexagonInit();  // Needed once at startup.
    TfLiteHexagonDelegateOptions params = {0};

    // 'delegate_ptr' Need to outlive the interpreter. For example,
    // If use case will need to resize input or anything that can trigger
    // re-applying delegates then 'delegate_ptr' need to outlive the interpreter.
    auto* delegate_ptr = TfLiteHexagonDelegateCreate(&params);
    tflite::Interpreter::TfLiteDelegatePtr delegatep(delegate_ptr,
        [](TfLiteDelegate* delegatep) {
            TfLiteHexagonDelegateDelete(delegatep);
        });

    delegate = delegatep.get();
#endif

#if defined (USE_XNNPACK_DELEGATE)
    int num_threads = std::thread::hardware_concurrency();
    char *env_tflite_num_threads = getenv ("FORCE_TFLITE_NUM_THREADS");
    if (env_tflite_num_threads)
    {
        num_threads = atoi (env_tflite_num_threads);
        DBG_LOGI ("@@@@@@ FORCE_TFLITE_NUM_THREADS(XNNPACK)=%d\n", num_threads);
    }

    // IMPORTANT: initialize options with TfLiteXNNPackDelegateOptionsDefault() for
    // API-compatibility with future extensions of the TfLiteXNNPackDelegateOptions
    // structure.
    TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_options.num_threads = num_threads;

    delegate = TfLiteXNNPackDelegateCreate (&xnnpack_options);
    if (!delegate)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
    }
#endif

    if (!delegate)
        return 0;

    if (p->interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    return 0;
}


int
tflite_create_interpreter_from_file (tflite_interpreter_t *p, const char *model_path)
{
    p->model = FlatBufferModel::BuildFromFile (model_path);
    if (!p->model)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    InterpreterBuilder(*(p->model), p->resolver)(&(p->interpreter));
    if (!p->interpreter)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    int num_threads = std::thread::hardware_concurrency();
    char *env_tflite_num_threads = getenv ("FORCE_TFLITE_NUM_THREADS");
    if (env_tflite_num_threads)
    {
        num_threads = atoi (env_tflite_num_threads);
        DBG_LOGI ("@@@@@@ FORCE_TFLITE_NUM_THREADS=%d\n", num_threads);
    }
    DBG_LOG ("@@@@@@ TFLITE_NUM_THREADS=%d\n", num_threads);
    p->interpreter->SetNumThreads(num_threads);

    if (modify_graph_with_delegate (p, NULL) < 0)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        //return -1;
    }

    if (p->interpreter->AllocateTensors() != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 1 /* for debug */
    DBG_LOG ("\n");
    DBG_LOG ("##### LOAD TFLITE FILE: \"%s\"\n", model_path);
    tflite_print_tensor_info (p->interpreter);
#endif

    return 0;
}

int
tflite_create_interpreter_ex_from_file (tflite_interpreter_t *p, const char *model_path, tflite_createopt_t *opt)
{
    p->model = FlatBufferModel::BuildFromFile (model_path);
    if (!p->model)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    InterpreterBuilder(*(p->model), p->resolver)(&(p->interpreter));
    if (!p->interpreter)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    int num_threads = std::thread::hardware_concurrency();
    char *env_tflite_num_threads = getenv ("FORCE_TFLITE_NUM_THREADS");
    if (env_tflite_num_threads)
    {
        num_threads = atoi (env_tflite_num_threads);
        DBG_LOGI ("@@@@@@ FORCE_TFLITE_NUM_THREADS=%d\n", num_threads);
    }
    DBG_LOG ("@@@@@@ TFLITE_NUM_THREADS=%d\n", num_threads);
    p->interpreter->SetNumThreads(num_threads);

#if 0
    std::vector<int> sizes = {1, 1280, 1280, 3};
    int input_id = p->interpreter->inputs()[0];
    p->interpreter->ResizeInputTensor(input_id, sizes);
#endif

    if (modify_graph_with_delegate (p, opt) < 0)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        //return -1;
    }

    if (p->interpreter->AllocateTensors() != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 1 /* for debug */
    DBG_LOG ("\n");
    DBG_LOG ("##### LOAD TFLITE FILE: \"%s\"\n", model_path);
    tflite_print_tensor_info (p->interpreter);
#endif

    return 0;
}


int
tflite_create_interpreter (tflite_interpreter_t *p, const char *model_buf, size_t model_size)
{
    p->model = FlatBufferModel::BuildFromBuffer(model_buf, model_size);
    if (!p->model)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    InterpreterBuilder(*(p->model), p->resolver)(&(p->interpreter));
    if (!p->interpreter)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    int num_threads = std::thread::hardware_concurrency();
    char *env_tflite_num_threads = getenv ("FORCE_TFLITE_NUM_THREADS");
    if (env_tflite_num_threads)
    {
        num_threads = atoi (env_tflite_num_threads);
        DBG_LOGI ("@@@@@@ FORCE_TFLITE_NUM_THREADS=%d\n", num_threads);
    }
    DBG_LOG ("@@@@@@ TFLITE_NUM_THREADS=%d\n", num_threads);
    p->interpreter->SetNumThreads(num_threads);

    if (modify_graph_with_delegate (p, NULL) < 0)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        //return -1;
    }

    if (p->interpreter->AllocateTensors() != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 1 /* for debug */
    DBG_LOG ("\n");
    DBG_LOG ("##### LOAD TFLITE: %p: %zu[byte]\n", model_buf, model_size);
    tflite_print_tensor_info (p->interpreter);
#endif

    return 0;
}

int
tflite_create_interpreter_ex (tflite_interpreter_t *p, const char *model_buf, size_t model_size, tflite_createopt_t *opt)
{
    p->model = FlatBufferModel::BuildFromBuffer(model_buf, model_size);
    if (!p->model)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    InterpreterBuilder(*(p->model), p->resolver)(&(p->interpreter));
    if (!p->interpreter)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    int num_threads = std::thread::hardware_concurrency();
    char *env_tflite_num_threads = getenv ("FORCE_TFLITE_NUM_THREADS");
    if (env_tflite_num_threads)
    {
        num_threads = atoi (env_tflite_num_threads);
        DBG_LOGI ("@@@@@@ FORCE_TFLITE_NUM_THREADS=%d\n", num_threads);
    }
    DBG_LOG ("@@@@@@ TFLITE_NUM_THREADS=%d\n", num_threads);
    p->interpreter->SetNumThreads(num_threads);

#if 0
    std::vector<int> sizes = {1, 1280, 1280, 3};
    int input_id = p->interpreter->inputs()[0];
    p->interpreter->ResizeInputTensor(input_id, sizes);
#endif

    if (modify_graph_with_delegate (p, opt) < 0)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        //return -1;
    }

    if (p->interpreter->AllocateTensors() != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 1 /* for debug */
    DBG_LOG ("\n");
    DBG_LOG ("##### LOAD TFLITE: %p: %zu[byte]\n", model_buf, model_size);
    tflite_print_tensor_info (p->interpreter);
#endif

    return 0;
}


int
tflite_get_tensor_by_name (tflite_interpreter_t *p, int io, const char *name, tflite_tensor_t *ptensor)
{
    std::unique_ptr<Interpreter> &interpreter = p->interpreter;

    memset (ptensor, 0, sizeof (*ptensor));

    int tensor_idx;
    int io_idx = -1;
    int num_tensor = (io == 0) ? interpreter->inputs ().size() :
                                 interpreter->outputs().size();

    for (int i = 0; i < num_tensor; i ++)
    {
        tensor_idx = (io == 0) ? interpreter->inputs ()[i] :
                                 interpreter->outputs()[i];

        const char *tensor_name = interpreter->tensor(tensor_idx)->name;
        if (strcmp (tensor_name, name) == 0)
        {
            io_idx = i;
            break;
        }
    }

    if (io_idx < 0)
    {
        DBG_LOGE ("can't find tensor: \"%s\"\n", name);
        return -1;
    }

    void *ptr = NULL;
    TfLiteTensor *tensor = interpreter->tensor(tensor_idx);
    switch (tensor->type)
    {
    case kTfLiteUInt8:
        ptr = (io == 0) ? interpreter->typed_input_tensor <uint8_t>(io_idx) :
                          interpreter->typed_output_tensor<uint8_t>(io_idx);
        break;
    case kTfLiteFloat32:
        ptr = (io == 0) ? interpreter->typed_input_tensor <float>(io_idx) :
                          interpreter->typed_output_tensor<float>(io_idx);
        break;
    case kTfLiteInt64:
        ptr = (io == 0) ? interpreter->typed_input_tensor <int64_t>(io_idx) :
                          interpreter->typed_output_tensor<int64_t>(io_idx);
        break;
    default:
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    ptensor->idx    = tensor_idx;
    ptensor->io     = io;
    ptensor->io_idx = io_idx;
    ptensor->type   = tensor->type;
    ptensor->ptr    = ptr;
    ptensor->quant_scale = tensor->params.scale;
    ptensor->quant_zerop = tensor->params.zero_point;

    for (int i = 0; (i < 4) && (i < tensor->dims->size); i ++)
    {
        ptensor->dims[i] = tensor->dims->data[i];
    }

    return 0;
}

