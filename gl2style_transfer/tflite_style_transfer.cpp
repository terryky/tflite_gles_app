/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_style_transfer.h"
#include <list>


#if 0
#define STYLE_PREDICT_MODEL_PATH  "./style_transfer_model/style_predict_quantized_256.tflite"
//#define STYLE_TRANSFER_MODEL_PATH "./style_transfer_model/style_transfer_quantized_dynamic.tflite"
#define STYLE_TRANSFER_MODEL_PATH "./style_transfer_model/style_transfer_quantized_384.tflite"
#else
#define STYLE_PREDICT_MODEL_PATH  "./style_transfer_model/style_predict_f16_256.tflite"
#define STYLE_TRANSFER_MODEL_PATH "./style_transfer_model/style_transfer_f16_384.tflite"
#endif


static tflite_interpreter_t s_interpreter_style_predict;
static tflite_tensor_t      s_predict_tensor_input;
static tflite_tensor_t      s_predict_tensor_output;

static tflite_interpreter_t s_interpreter_style_transfer;
static tflite_tensor_t      s_transfer_tensor_content_in;
static tflite_tensor_t      s_transfer_tensor_style_in;
static tflite_tensor_t      s_transfer_tensor_output;


int
init_tflite_style_transfer ()
{
    tflite_interpreter_t *p;

    /* predict */
    p = &s_interpreter_style_predict;
    tflite_create_interpreter_from_file (p, STYLE_PREDICT_MODEL_PATH);
    tflite_get_tensor_by_name (p, 0, "style_image",                 &s_predict_tensor_input);
    tflite_get_tensor_by_name (p, 1, "mobilenet_conv/Conv/BiasAdd", &s_predict_tensor_output);

    /* transfeer */
    p = &s_interpreter_style_transfer;
    tflite_create_interpreter_from_file (p, STYLE_TRANSFER_MODEL_PATH);
    tflite_get_tensor_by_name (p, 0, "content_image",               &s_transfer_tensor_content_in);
    tflite_get_tensor_by_name (p, 0, "mobilenet_conv/Conv/BiasAdd", &s_transfer_tensor_style_in);
    tflite_get_tensor_by_name (p, 1, "transformer/expand/conv3/conv/Sigmoid", &s_transfer_tensor_output);

    return 0;
}

void *
get_style_predict_input_buf (int *w, int *h)
{
    *w = s_predict_tensor_input.dims[2];
    *h = s_predict_tensor_input.dims[1];
    return (float *)s_predict_tensor_input.ptr;
}

void *
get_style_transfer_style_input_buf (int *size)
{
    *size = s_transfer_tensor_style_in.dims[3];
    return (float *)s_transfer_tensor_style_in.ptr;
}

void *
get_style_transfer_content_input_buf (int *w, int *h)
{
    *w = s_transfer_tensor_content_in.dims[2];
    *h = s_transfer_tensor_content_in.dims[1];
    return (float *)s_transfer_tensor_content_in.ptr;
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

    //int s0 = s_predict_tensor_output.dims[0];
    //int s1 = s_predict_tensor_output.dims[1];
    //int s2 = s_predict_tensor_output.dims[2];
    int s3 = s_predict_tensor_output.dims[3];
    //fprintf (stderr, "style output dim: (%dx%dx%dx%d)\n", s0, s1, s2, s3);

    predict_result->size  = s3;
    predict_result->param = s_predict_tensor_output.ptr;

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

    //int s0 = s_transfer_tensor_output.dims[0];
    int s1 = s_transfer_tensor_output.dims[1];
    int s2 = s_transfer_tensor_output.dims[2];
    //int s3 = s_transfer_tensor_output.dims[3];
    //fprintf (stderr, "style output dim: (%dx%dx%dx%d)\n", s0, s1, s2, s3);

    transfered_result->h = s1;
    transfered_result->w = s2;
    transfered_result->img = s_transfer_tensor_output.ptr;

    return 0;
}
