/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_trt.h"
#include "util_debug.h"
#include <thread>

static Logger s_Logger;


/* -------------------------------------------------- *
 *  wait counter thread
 * -------------------------------------------------- */
static pthread_t s_wait_thread;
static int       s_is_waiting = 1;

static void *
waitcounter_thread_main (void *)
{
    int cnt = 0;
    while (s_is_waiting)
    {
        fprintf (stderr, "\r%d", cnt++);
        std::this_thread::sleep_for (std::chrono::seconds (1));
    }
    return NULL;
}


static int
start_waitcounter_thread ()
{
    s_is_waiting = 1;
    pthread_create (&s_wait_thread, NULL, waitcounter_thread_main, NULL);
    return 0;
}

static int
stop_waitcounter_thread ()
{
    s_is_waiting = 0;
    return 0;
}


/* -------------------------------------------------- *
 *  allocate tensor buffer
 * -------------------------------------------------- */
static int
get_element_count (const Dims &dims)
{
    int elem_count = 1;
    for (int i = 0; i < dims.nbDims; i++)
        elem_count *= dims.d[i];

    return elem_count;
}


static unsigned int
get_element_size (DataType t)
{
    switch (t)
    {
    case DataType::kINT32: return 4;
    case DataType::kFLOAT: return 4;
    case DataType::kHALF:  return 2;
    case DataType::kINT8:  return 1;
    default:               return 0;
    }
}


static void
print_dims (const Dims &dims)
{
    for (int i = 0; i < dims.nbDims; i++)
    {
        if (i > 0)
            fprintf (stderr, "x");

        fprintf (stderr, "%d", dims.d[i]);
    }
}

static const char *
get_type_str (DataType t)
{
    switch (t)
    {
    case DataType::kINT32: return "kINT32";
    case DataType::kFLOAT: return "kFLOAT";
    case DataType::kHALF:  return "kHALF";
    case DataType::kINT8:  return "kINT8";
    default:               return "UNKNOWN";
    }
}


void *
safeCudaMalloc (size_t memSize)
{
    void* deviceMem;
    cudaError_t err;

    err = cudaMalloc(&deviceMem, memSize);
    if (err != 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}


/* -------------------------------------------------- *
 *  create cuda engine
 * -------------------------------------------------- */
#define MAX_WORKSPACE (1 << 30)

ICudaEngine *
trt_create_engine_from_uff (const std::string &uff_file, 
                            std::vector<trt_uff_inputdef_t>  &input_array, 
                            std::vector<trt_uff_outputdef_t> &output_array)
{
    IBuilder           *builder = createInferBuilder (s_Logger);
    INetworkDefinition *network = builder->createNetwork ();

    IUffParser *parser = nvuffparser::createUffParser();

    for (auto itr = input_array.begin(); itr != input_array.end(); itr++)
    {
        parser->registerInput (itr->name.c_str(), itr->dims, itr->order);
    }

    for (auto itr = output_array.begin(); itr != output_array.end(); itr++)
    {
        parser->registerOutput(itr->name.c_str());
    }

#if 1
    if (!parser->parse(uff_file.c_str(), *network, nvinfer1::DataType::kHALF))
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }
    builder->setFp16Mode(true);
#else
    if (!parser->parse(uff_file.c_str(), *network, nvinfer1::DataType::kFLOAT))
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }
#endif

    // builder->setInt8Mode (true);
    // builder->setInt8Calibrator (NULL);
    fprintf (stderr, " - HasFastFp16(): %d\n", builder->platformHasFastFp16());
    fprintf (stderr, " - HasFastInt8(): %d\n", builder->platformHasFastInt8());

    /* create the engine */
    builder->setMaxBatchSize (1);
    builder->setMaxWorkspaceSize (MAX_WORKSPACE);

    fprintf (stderr, "Building CUDA Engine. (This takes more than 10 minutes. please wait...)\n");
    start_waitcounter_thread();

    ICudaEngine *engine = builder->buildCudaEngine (*network);
    if (!engine)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    stop_waitcounter_thread();

    network->destroy();
    builder->destroy();
    parser->destroy();

    return engine;
}


ICudaEngine *
trt_create_engine_from_onnx (const std::string &onnx_file)
{
    IBuilder           *builder = createInferBuilder (s_Logger);
    //INetworkDefinition *network = builder->createNetwork ();
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig     *config  = builder->createBuilderConfig ();

    nvonnxparser::IParser *parser = nvonnxparser::createParser (*network, s_Logger);

    int severity = static_cast<int>(ILogger::Severity::kWARNING);
    auto parsed = parser->parseFromFile(onnx_file.c_str(), severity);
    if (!parsed)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    // builder->setInt8Mode (true);
    // builder->setInt8Calibrator (NULL);
    fprintf (stderr, " - HasFastFp16(): %d\n", builder->platformHasFastFp16());
    fprintf (stderr, " - HasFastInt8(): %d\n", builder->platformHasFastInt8());

    /* create the engine */
    builder->setMaxBatchSize (1);
    builder->setMaxWorkspaceSize (MAX_WORKSPACE);

    config->setFlag (BuilderFlag::kFP16);
//  config->setFlag (BuilderFlag::kINT8);

    fprintf (stderr, "Building CUDA Engine. (This takes more than 10 minutes. please wait...)\n");
    start_waitcounter_thread();

    ICudaEngine *engine = builder->buildEngineWithConfig (*network, *config);
    if (!engine)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    stop_waitcounter_thread();

    network->destroy();
    builder->destroy();
    parser->destroy();

    return engine;
}


/* -------------------------------------------------- *
 *  save/load "plan file"
 * -------------------------------------------------- */
int
trt_emit_plan_file (ICudaEngine *engine, const std::string &plan_file_name)
{
    IHostMemory *serialized_engine = engine->serialize();

    std::ofstream planfile;
    planfile.open (plan_file_name);
    planfile.write((char *)serialized_engine->data(), serialized_engine->size());
    planfile.close();

    return 0;
}


ICudaEngine *
trt_load_plan_file (const std::string &plan_file_name)
{
    std::ifstream planfile (plan_file_name);
    if (!planfile.is_open())
    {
        fprintf (stderr, "ERR:%s(%d) Could not open plan file.\n", __FILE__, __LINE__);
        return NULL;
    }

    std::stringstream planbuffer;
    planbuffer << planfile.rdbuf();
    std::string plan = planbuffer.str();

    IRuntime *runtime = createInferRuntime (s_Logger);

    ICudaEngine *engine;
    engine = runtime->deserializeCudaEngine ((void*)plan.data(), plan.size(), nullptr);

    return engine;
}


/* -------------------------------------------------- *
 *  get pointer to input/output Tensor.
 * -------------------------------------------------- */
int
trt_get_tensor_by_name (ICudaEngine *engine, const char *name, trt_tensor_t *ptensor)
{
    memset (ptensor, 0, sizeof (*ptensor));

    int bind_idx = -1;
    bind_idx = engine->getBindingIndex (name);
    if (bind_idx < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    ptensor->bind_idx = bind_idx;
    ptensor->dtype    = engine->getBindingDataType   (bind_idx);
    ptensor->dims     = engine->getBindingDimensions (bind_idx);

    int element_size  = get_element_size  (ptensor->dtype);
    int element_count = get_element_count (ptensor->dims);
    ptensor->memsize  = element_size * element_count;

    ptensor->gpu_mem = safeCudaMalloc(ptensor->memsize);
    ptensor->cpu_mem = malloc (ptensor->memsize);

    fprintf (stderr, "------------------------------\n");
    fprintf (stderr, "[%s]\n", name);
    fprintf (stderr, " bind_idx = %d\n", ptensor->bind_idx);
    fprintf (stderr, " data type= %s\n", get_type_str (ptensor->dtype));
    fprintf (stderr, " dimension= "); print_dims (ptensor->dims);
    fprintf (stderr, "\n");
    fprintf (stderr, "------------------------------\n");

    return 0;
}


/* -------------------------------------------------- *
 *  Memory transaction
 * -------------------------------------------------- */
int
trt_copy_tensor_to_gpu (trt_tensor_t &tensor)
{
    cudaError_t err;

    err = cudaMemcpy (tensor.gpu_mem, tensor.cpu_mem, tensor.memsize, cudaMemcpyHostToDevice);
    if (err != 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    return 0;
}

int
trt_copy_tensor_from_gpu (trt_tensor_t &tensor)
{
    cudaError_t err;

    err = cudaMemcpy (tensor.cpu_mem, tensor.gpu_mem, tensor.memsize, cudaMemcpyDeviceToHost);
    if (err != 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    return 0;
}



/* -------------------------------------------------- *
 *  Initialize TensorRT
 * -------------------------------------------------- */
int
trt_initialize ()
{
    fprintf (stderr, "TensorRT version: %d.%d.%d.%d\n",
        NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);

    return 0;
}
