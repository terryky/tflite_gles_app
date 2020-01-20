/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_TFLITE_H_
#define _UTIL_TFLITE_H_

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#if defined (USE_GL_DELEGATE)
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif
#if defined (USE_GPU_DELEGATEV2)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif


typedef struct tflite_tensor_t
{
    int         idx;        /* whole  tensor index */
    int         io;         /* [0] input_tensor, [1] output_tensor */
    int         io_idx;     /* in/out tensor index */
    TfLiteType  type;       /* [1] kTfLiteFloat32, [2] kTfLiteInt32, [3] kTfLiteUInt8 */
    void        *ptr;
    int         dims[4];
    float       quant_scale;
    int         quant_zerop;
} tflite_tensor_t;




typedef void *tflite_obj_t;

tflite_obj_t create_tflite_inferer (const char *model_fname);
void dump_tflite_model (tflite_obj_t tflite_obj);

int get_tflite_tensor_by_name (tflite_obj_t tobj, int io, const char *name, tflite_tensor_t *ptensor);

int invoke_tflite (tflite_obj_t tobj);

#ifdef __cplusplus
}
#endif

#endif /* _UTIL_TFLITE_H_ */

