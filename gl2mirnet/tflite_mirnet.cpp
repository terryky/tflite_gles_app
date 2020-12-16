/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_mirnet.h"


#define MIRNET_MODEL_PATH        "./model/lite-model_mirnet-fixed_fp16_1.tflite"
#define MIRNET_QUANT_MODEL_PATH  "./model/lite-model_mirnet-fixed_integer_1.tflite"


static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_output;


int
init_tflite_mirnet (int use_quantized_tflite)
{
    const char *mirnet_model;

    if (use_quantized_tflite)
    {
        mirnet_model = MIRNET_QUANT_MODEL_PATH;
    }
    else
    {
        mirnet_model = MIRNET_MODEL_PATH;
    }

    tflite_create_interpreter_from_file (&s_interpreter, mirnet_model);
    tflite_get_tensor_by_name (&s_interpreter, 0, "input_1",  &s_tensor_input);
    tflite_get_tensor_by_name (&s_interpreter, 1, "Identity", &s_tensor_output);

    return 0;
}

void *
get_mirnet_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return (float *)s_tensor_input.ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_mirnet (mirnet_t *predict_result)
{
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    predict_result->param = s_tensor_output.ptr;
    predict_result->w     = s_tensor_output.dims[2];
    predict_result->h     = s_tensor_output.dims[1];

    return 0;
}

