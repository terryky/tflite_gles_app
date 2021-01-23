/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_trt.h"
#include "trt_dense_depth.h"
#include <unistd.h>

#define UFF_MODEL_PATH      "models/dense_depth_640x480.onnx"
#define PLAN_MODEL_PATH     "models/dense_depth_640x480.plan"


static IExecutionContext   *s_trt_context;
static trt_tensor_t         s_tensor_input;
static trt_tensor_t         s_tensor_depth;

static std::vector<void *>  s_gpu_buffers;


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
init_trt_dense_depth ()
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
    trt_get_tensor_by_name (engine, "input_1:0",    &s_tensor_input);
    trt_get_tensor_by_name (engine, "Identity:0",   &s_tensor_depth);

    int num_bindings = engine->getNbBindings();
    s_gpu_buffers.resize (num_bindings);
    s_gpu_buffers[s_tensor_input.bind_idx] = s_tensor_input.gpu_mem;
    s_gpu_buffers[s_tensor_depth.bind_idx] = s_tensor_depth.gpu_mem;

    return 0;
}


void *
get_dense_depth_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims.d[2];
    *h = s_tensor_input.dims.d[1];
    return s_tensor_input.cpu_mem;
}



/* -------------------------------------------------- *
 * Invoke TensorRT
 * -------------------------------------------------- */
int
invoke_dense_depth (dense_depth_result_t *dense_depth_result)
{
    /* copy to CUDA buffer */
    trt_copy_tensor_to_gpu (s_tensor_input);

    /* invoke inference */
    int batchSize = 1;
    s_trt_context->execute (batchSize, &s_gpu_buffers[0]);

    /* copy from CUDA buffer */
    trt_copy_tensor_from_gpu (s_tensor_depth);


    
    dense_depth_result->depthmap         = (float *)s_tensor_depth.cpu_mem;
    dense_depth_result->depthmap_dims[0] = s_tensor_depth.dims.d[2];
    dense_depth_result->depthmap_dims[1] = s_tensor_depth.dims.d[1];

    return 0;
}

