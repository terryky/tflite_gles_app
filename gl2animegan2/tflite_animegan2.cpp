/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_animegan2.h"


#define ANIMEGAN2_MODEL_PATH        "./model/animeganv2_hayao_256x256.tflite"
#define ANIMEGAN2_QUANT_MODEL_PATH  "./model/animeganv2_hayao_256x256_integer_quant.tflite"


static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_output;


int
init_tflite_animegan2 (int use_quantized_tflite)
{
    const char *animegan2_model;

    if (use_quantized_tflite)
    {
        animegan2_model = ANIMEGAN2_QUANT_MODEL_PATH;
    }
    else
    {
        animegan2_model = ANIMEGAN2_MODEL_PATH;
    }

    tflite_create_interpreter_from_file (&s_interpreter, animegan2_model);
    tflite_get_tensor_by_name (&s_interpreter, 0, "input",                            &s_tensor_input);
    tflite_get_tensor_by_name (&s_interpreter, 1, "generator/G_MODEL/out_layer/Tanh", &s_tensor_output);

    return 0;
}

void *
get_animegan2_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return (float *)s_tensor_input.ptr;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_animegan2 (animegan2_t *predict_result)
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

