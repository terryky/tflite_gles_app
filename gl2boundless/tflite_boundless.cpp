/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_boundless.h"


#define BOUNDLESS_MODEL_PATH        "./model/boundless_half_dr.tflite"
#define BOUNDLESS_QUANT_MODEL_PATH  "./model/boundless_half_dr.tflite"


static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_mask;
static tflite_tensor_t      s_tensor_output;


int
init_tflite_boundless (int use_quantized_tflite)
{
    const char *mirnet_model;

    if (use_quantized_tflite)
    {
        mirnet_model = BOUNDLESS_MODEL_PATH;
    }
    else
    {
        mirnet_model = BOUNDLESS_MODEL_PATH;
    }

    tflite_create_interpreter_from_file (&s_interpreter, mirnet_model);
    tflite_get_tensor_by_name (&s_interpreter, 0, "Placeholder", &s_tensor_input);
    tflite_get_tensor_by_name (&s_interpreter, 1, "mul_2",       &s_tensor_mask);
    tflite_get_tensor_by_name (&s_interpreter, 1, "mul_1",       &s_tensor_output);

    return 0;
}

void *
get_boundless_input_buf (int *w, int *h)
{
    tflite_get_tensor_by_name (&s_interpreter, 0, "Placeholder", &s_tensor_input);
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    fprintf (stderr, "(%d, %d): %p\n", *w, *h, s_tensor_input.ptr);
    return (float *)s_tensor_input.ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_boundless (boundless_t *predict_result)
{
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    tflite_get_tensor_by_name (&s_interpreter, 1, "mul_1", &s_tensor_output);
    tflite_get_tensor_by_name (&s_interpreter, 1, "mul_2", &s_tensor_mask);
    predict_result->buf_gen  = s_tensor_output.ptr;
    predict_result->buf_mask = s_tensor_mask.ptr;
    predict_result->w        = s_tensor_output.dims[2];
    predict_result->h        = s_tensor_output.dims[1];

    return 0;
}

