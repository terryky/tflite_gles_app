/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef UTIL_TRT_H
#define UTIL_TRT_H

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvUtils.h"
#include "NvUffParser.h"
#include "NvOnnxParser.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <string>
#include <utility>
#include <vector>
#include <list>

using namespace nvinfer1;
using namespace nvuffparser;
using namespace plugin;


class Logger : public nvinfer1::ILogger
{
public:
    Logger (Severity severity = Severity::kINFO)
        : reportableSeverity(severity)
    {
    }

    void log (Severity severity, const char* msg) override
    {
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR:          std::cerr << "ERROR  : "; break;
        case Severity::kWARNING:        std::cerr << "WARNING: "; break;
        case Severity::kINFO:           std::cerr << "INFO   : "; break;
        default:                        std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

typedef struct trt_tensor_t
{
    int         io;         /* [0] input_tensor, [1] output_tensor */
    int         bind_idx;   /* in/out bind index */
    DataType    dtype;      /* kINT32, kFLOAT, kHALF, kINT8 */
    Dims        dims;
    size_t      memsize;
    void        *gpu_mem;
    void        *cpu_mem;
} trt_tensor_t;


typedef struct trt_uff_inputdef_t
{
    std::string     name;
    nvinfer1::Dims  dims;
    UffInputOrder   order;
} trt_uff_inputdef_t;

typedef struct trt_uff_outputdef_t
{
    std::string     name;
} trt_uff_outputdef_t;


/* initialize */
int trt_initialize ();


/* create TensorRT engine */
ICudaEngine *
trt_create_engine_from_uff (const std::string &uff_file,
                            std::vector<trt_uff_inputdef_t>  &input_array, 
                            std::vector<trt_uff_outputdef_t> &output_array);

ICudaEngine *
trt_create_engine_from_onnx (const std::string &onnx_file);


/* emit/load PLAN file */
int trt_emit_plan_file (ICudaEngine *engine, const std::string &plan_file_name);
ICudaEngine *trt_load_plan_file (const std::string &plan_file_name);

/* get pointer to input/output Tensor */
int trt_get_tensor_by_name (ICudaEngine *engine, const char *name, trt_tensor_t *ptensor);

/* Tensor memory transaction */
int trt_copy_tensor_to_gpu   (trt_tensor_t &tensor);
int trt_copy_tensor_from_gpu (trt_tensor_t &tensor);


#endif // UTIL_TRT_H
