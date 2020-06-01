/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_hair_segmentation.h"
#include "custom_ops/max_pool_argmax.h"
#include "custom_ops/max_unpooling.h"
#include "custom_ops/transpose_conv_bias.h"

/* 
 * https://github.com/google/mediapipe/tree/master/mediapipe/models/hair_segmentation.tflite
 */
#define SEGMENTATION_MODEL_PATH  "./hair_segmentation_model/hair_segmentation.tflite"


static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_segment;


int
init_tflite_segmentation ()
{
    s_interpreter.resolver.AddCustom("MaxPoolingWithArgmax2D",
            mediapipe::tflite_operations::RegisterMaxPoolingWithArgmax2D());

    s_interpreter.resolver.AddCustom("MaxUnpooling2D",
            mediapipe::tflite_operations::RegisterMaxUnpooling2D());

    s_interpreter.resolver.AddCustom("Convolution2DTransposeBias",
            mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());

    tflite_create_interpreter_from_file (&s_interpreter, SEGMENTATION_MODEL_PATH);

    /* get in/out tensor */
    tflite_get_tensor_by_name (&s_interpreter, 0, "input_1",  &s_tensor_input);
    tflite_get_tensor_by_name (&s_interpreter, 1, "conv2d_transpose_4",  &s_tensor_segment);

    return 0;
}


void *
get_segmentation_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return s_tensor_input.ptr;
}


int
invoke_segmentation (segmentation_result_t *segment_result)
{
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    segment_result->segmentmap         = (float *)s_tensor_segment.ptr;
    segment_result->segmentmap_dims[0] = s_tensor_segment.dims[2];
    segment_result->segmentmap_dims[1] = s_tensor_segment.dims[1];
    segment_result->segmentmap_dims[2] = s_tensor_segment.dims[3];

    return 0;
}

