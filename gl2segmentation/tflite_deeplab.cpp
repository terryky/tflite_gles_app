/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_deeplab.h"
#include "util_debug.h"

/* 
 * https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite
 */
#define DEEPLAB_MODEL_PATH  "./deeplab_model/deeplabv3_257_mv_gpu.tflite"


static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_segment;

static float s_class_color[MAX_DETECT_CLASS + 1][4];
static char  s_class_name [MAX_DETECT_CLASS + 1][64] =
{
    "background",   // 0
    "aeroplane",    // 1
    "bicycle",      // 2
    "bird",         // 3
    "boat",         // 4
    "bottle",       // 5
    "bus",          // 6
    "car",          // 7
    "cat",          // 8
    "chair",        // 9
    "cow",          // 10
    "diningtable",  // 11
    "dog",          // 12
    "horse",        // 13
    "motorbike",    // 14
    "person",       // 15
    "pottedplant",  // 16
    "sheep",        // 17
    "sofa",         // 18
    "train",        // 19
    "tvmonitor"     // 20
};

static int
init_class_color ()
{
    for (int i = 0; i < MAX_DETECT_CLASS + 1; i ++)
    {
        float *col = s_class_color[i];
        if (i == 0)
        {
            col[0] = col[1] = col[2] = 0.0f;
        }
        else
        {
            col[0] = (rand () % 255) / 255.0f;
            col[1] = (rand () % 255) / 255.0f;
            col[2] = (rand () % 255) / 255.0f;
        }
        col[3] = 0.8f;
    }
    return 0;
}


int
init_tflite_deeplab()
{
    tflite_create_interpreter_from_file (&s_interpreter, DEEPLAB_MODEL_PATH);

    /* get input tensor */
    tflite_get_tensor_by_name (&s_interpreter, 0, "sub_7",  &s_tensor_input);

    /* get output tensor */
    tflite_get_tensor_by_name (&s_interpreter, 1, "ResizeBilinear_3",  &s_tensor_segment);

    init_class_color ();

    return 0;
}

int
get_deeplab_input_type ()
{
    if (s_tensor_input.type == kTfLiteUInt8)
        return 1;
    else
        return 0;
}

void *
get_deeplab_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return s_tensor_input.ptr;
}

char *
get_deeplab_class_name (int class_idx)
{
    return s_class_name[class_idx];
}

float *
get_deeplab_class_color (int class_idx)
{
    return s_class_color[class_idx];
}


int
invoke_deeplab (deeplab_result_t *deeplab_result)
{
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    deeplab_result->segmentmap         = (float *)s_tensor_segment.ptr;
    deeplab_result->segmentmap_dims[0] = s_tensor_segment.dims[2];
    deeplab_result->segmentmap_dims[1] = s_tensor_segment.dims[1];
    deeplab_result->segmentmap_dims[2] = s_tensor_segment.dims[3];

    return 0;
}

