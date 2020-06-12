#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvUtils.h"
#include "NvUffParser.h"
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

#endif // TENSORRT_COMMON_H
