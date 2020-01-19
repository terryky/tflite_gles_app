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





typedef void *tflite_obj_t;

tflite_obj_t create_tflite_inferer (const char *model_fname);
void dump_tflite_model (tflite_obj_t tflite_obj);





#ifdef __cplusplus
}
#endif

#endif /* _UTIL_TFLITE_H_ */

