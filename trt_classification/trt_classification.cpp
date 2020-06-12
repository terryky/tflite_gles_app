/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "trt_common.h"
#include "trt_classification.h"
#include <unistd.h>


#define UFF_MODEL_PATH      "./models/mobilenet_v1_1.0_224.uff"
#define PLAN_MODEL_PATH     "./models/mobilenet_v1_1.0_224.plan"
#define LABEL_MAP_PATH      "./models/class_label.txt"

static Logger               gLogger;
static IExecutionContext   *s_trt_context;
static trt_tensor_t         s_tensor_input;
static trt_tensor_t         s_tensor_output;
static std::vector<void *>  s_gpu_buffers;

static char                 s_class_name [MAX_CLASS_NUM][64];


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
        sleep(1);
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
 *  load class labels
 * -------------------------------------------------- */
static int
load_label_map ()
{
    FILE *fp = fopen (LABEL_MAP_PATH, "r");
    if (fp == NULL)
        return 0;

    int id = 1;
    char buf[512];
    while (fgets (buf, 512, fp))
    {
        int len = strlen(buf);
        buf[len-1] = '\0';
        memcpy (&s_class_name[id], buf, sizeof (s_class_name[id]));
        //fprintf (stderr, "ID[%d] %s\n", id, s_class_name[id]);
        id ++;
    }

    fclose (fp);

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
 *  save/load "plan file"
 * -------------------------------------------------- */
static int
emit_plan_file (ICudaEngine *engine, const std::string &plan_file_name)
{
    IHostMemory *serialized_engine = engine->serialize();

    std::ofstream planfile;
    planfile.open (plan_file_name);
    planfile.write((char *)serialized_engine->data(), serialized_engine->size());
    planfile.close();

    return 0;
}


static ICudaEngine *
load_plan_file (const std::string &plan_file_name)
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

    IRuntime *runtime = createInferRuntime (gLogger);

    ICudaEngine *engine;
    engine = runtime->deserializeCudaEngine ((void*)plan.data(), plan.size(), nullptr);

    return engine;
}


/* -------------------------------------------------- *
 *  create cuda engine
 * -------------------------------------------------- */
#define MAX_WORKSPACE (1 << 30)

ICudaEngine *
create_engine_from_uff (const std::string &uff_file)
{
    IBuilder           *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetwork();

    const std::string input_layer_name  = "input";
    const std::string output_layer_name = "MobilenetV1/Predictions/Reshape_1";

    IUffParser *parser = nvuffparser::createUffParser();
    parser->registerInput (input_layer_name.c_str(),
                           nvinfer1::Dims3(224, 224, 3), 
                           nvuffparser::UffInputOrder::kNHWC);
    parser->registerOutput(output_layer_name.c_str());

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

    fprintf (stderr, "Building CUDA Engine. please wait...\n");
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


int
convert_uff_to_plan (const std::string &plan_file_name, const std::string &uff_file_name)
{
    ICudaEngine *engine;

    engine = create_engine_from_uff (uff_file_name);
    if (!engine)
    {
        fprintf (stderr, "ERR:%s(%d): Failed to load graph from file.\n", __FILE__, __LINE__);
        return -1;
    }

    emit_plan_file (engine, plan_file_name);

    engine->destroy();

    return 0;
}



/* -------------------------------------------------- *
 *  Create TensorRT Interpreter
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

int
init_trt_classification ()
{
    fprintf (stderr, "TensorRT version: %d.%d.%d.%d\n",
        NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);

    /* Build TensorRT Engine */
    if (0)
    {
        convert_uff_to_plan (PLAN_MODEL_PATH, UFF_MODEL_PATH);
    }

    /* Load Prebuilt TensorRT Engine */
    fprintf (stderr, "loading cuda engine...\n");
    ICudaEngine *engine = load_plan_file (PLAN_MODEL_PATH);

    s_trt_context = engine->createExecutionContext();


    /* Allocate IO tensors */
    trt_get_tensor_by_name (engine, "input",                             &s_tensor_input);
    trt_get_tensor_by_name (engine, "MobilenetV1/Predictions/Reshape_1", &s_tensor_output);


    int num_bindings = engine->getNbBindings();
    s_gpu_buffers.resize (num_bindings);
    s_gpu_buffers[s_tensor_input .bind_idx] = s_tensor_input .gpu_mem;
    s_gpu_buffers[s_tensor_output.bind_idx] = s_tensor_output.gpu_mem;


    load_label_map ();

    return 0;
}



void *
get_classification_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims.d[1];
    *h = s_tensor_input.dims.d[0];
    return s_tensor_input.cpu_mem;
}



/* -------------------------------------------------- *
 * Invoke TensorRT
 * -------------------------------------------------- */
static bool
compare (classify_t &v1, classify_t &v2)
{
    if (v1.score > v2.score)
        return true;
    else
        return false;
}


int
invoke_classification (classification_result_t *class_ret)
{
    float *in_ptr  = (float *)s_tensor_input.cpu_mem;
    float *out_ptr = (float *)s_tensor_output.cpu_mem;
    cudaError_t err;

    /* copy to CUDA buffer */
    err = cudaMemcpy (s_tensor_input.gpu_mem, in_ptr, s_tensor_input.memsize, cudaMemcpyHostToDevice);
    if (err != 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /* invoke inference */
    int batchSize = 1;
    s_trt_context->execute (batchSize, &s_gpu_buffers[0]);

    /* copy from CUDA buffer */
    err = cudaMemcpy (out_ptr, s_tensor_output.gpu_mem, s_tensor_output.memsize, cudaMemcpyDeviceToHost);
    if (err != 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }


    std::list<classify_t> classify_list;
    for (int i = 0; i < MAX_CLASS_NUM; i ++)
    {
        classify_t item;
        item.id = i;
        item.score = out_ptr[i];

        classify_list.push_back (item);
    }

    classify_list.sort (compare);

    int count = 0;
    for (auto itr = classify_list.begin(); itr != classify_list.end(); itr ++)
    {
        classify_t *item = &class_ret->classify[count];

        item->id    = itr->id;
        item->score = itr->score;
        memcpy (item->name, s_class_name[itr->id], 64);

        class_ret->num = count + 1;

        count ++;
        if (count >= 5)
            break;
    }

    return 0;
}
