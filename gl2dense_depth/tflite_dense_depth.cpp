/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_dense_depth.h"
#include <list>

/* 
 * https://github.com/PINTO0309/PINTO_model_zoo/tree/master/064_Dense_Depth/nyu/01_float32
 */
#define DENSEDEPTH_MODEL_PATH         "./model/dense_depth_nyu_480x640_float32.tflite"
#define DENSEDEPTH_QUANT_MODEL_PATH   "./model/dense_depth_nyu_480x640_float32.tflite"

static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_depth;


/* -------------------------------------------------- *
 *  Create TFLite Interpreter
 * -------------------------------------------------- */
int
init_tflite_dense_depth (int use_quantized_tflite)
{
    const char *densedepth_model;

    if (use_quantized_tflite)
    {
        densedepth_model = DENSEDEPTH_QUANT_MODEL_PATH;
    }
    else
    {
        densedepth_model = DENSEDEPTH_MODEL_PATH;
    }

    tflite_create_interpreter_from_file (&s_interpreter, densedepth_model);
    tflite_get_tensor_by_name (&s_interpreter, 0, "input_1",  &s_tensor_input);
    tflite_get_tensor_by_name (&s_interpreter, 1, "Identity", &s_tensor_depth);

    return 0;
}


void *
get_dense_depth_input_buf (int *w, int *h)
{
    /* need to retrieve the input tensor again ? (dynamic shape) */
    tflite_get_tensor_by_name (&s_interpreter, 0, "input_1", &s_tensor_input);

    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return s_tensor_input.ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_dense_depth (dense_depth_result_t *dense_depth_result)
{
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* need to retrieve the output tensor again ? (dynamic shape) */
    tflite_get_tensor_by_name (&s_interpreter, 1, "Identity", &s_tensor_depth);
    
    dense_depth_result->depthmap         = (float *)s_tensor_depth.ptr;
    dense_depth_result->depthmap_dims[0] = s_tensor_depth.dims[2];
    dense_depth_result->depthmap_dims[1] = s_tensor_depth.dims[1];

    return 0;
}

