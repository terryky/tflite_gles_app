/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_trt.h"
#include "trt_classification.h"


#define UFF_MODEL_PATH      "./models/mobilenet_v1_1.0_224.uff"
#define PLAN_MODEL_PATH     "./models/mobilenet_v1_1.0_224.plan"
#define LABEL_MAP_PATH      "./models/class_label.txt"

static IExecutionContext   *s_trt_context;
static trt_tensor_t         s_tensor_input;
static trt_tensor_t         s_tensor_output;
static std::vector<void *>  s_gpu_buffers;

static char                 s_class_name [MAX_CLASS_NUM][64];


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
 *  create TensorRT engine.
 * -------------------------------------------------- */
static int
convert_uff_to_plan (const std::string &plan_file_name, const std::string &uff_file_name)
{
    std::vector<trt_uff_inputdef_t>  uff_input_array;
    trt_uff_inputdef_t uff_input;
    uff_input.name  = "input";
    uff_input.dims  = nvinfer1::Dims3(224, 224, 3);
    uff_input.order = nvuffparser::UffInputOrder::kNHWC;
    uff_input_array.push_back (uff_input);

    std::vector<trt_uff_outputdef_t> uff_output_array;
    trt_uff_outputdef_t uff_output;
    uff_output.name = "MobilenetV1/Predictions/Reshape_1";
    uff_output_array.push_back (uff_output);

    ICudaEngine *engine;
    engine = trt_create_engine_from_uff (uff_file_name, uff_input_array, uff_output_array);
    if (!engine)
    {
        fprintf (stderr, "ERR:%s(%d): Failed to load graph from file.\n", __FILE__, __LINE__);
        return -1;
    }

    trt_emit_plan_file (engine, plan_file_name);

    engine->destroy();

    return 0;
}



int
init_trt_classification ()
{
    ICudaEngine *engine = NULL;

    trt_initialize ();

    /* Try to load Prebuilt TensorRT Engine */
    fprintf (stderr, "loading prebuilt TensorRT engine...\n");
    engine = trt_load_plan_file (PLAN_MODEL_PATH);

    /* Build TensorRT Engine */
    if (engine == NULL)
    {
        convert_uff_to_plan (PLAN_MODEL_PATH, UFF_MODEL_PATH);

        engine = trt_load_plan_file (PLAN_MODEL_PATH);
        if (engine == NULL)
        {
            fprintf (stderr, "%s(%d)\n", __FILE__, __LINE__);
            return -1;
        }
    }

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
static float
get_scoreval (int class_id)
{
    float *val = (float *)s_tensor_output.cpu_mem;
    return val[class_id];
}

static int
push_listitem (std::list<classify_t> &class_list, classify_t &item, size_t topn)
{
    size_t idx = 0;

    /* search insert point */
    for (auto itr = class_list.begin(); itr != class_list.end(); itr ++)
    {
        if (item.score > itr->score)
        {
            class_list.insert (itr, item);
            if (class_list.size() > topn)
            {
                class_list.pop_back();
            }
            return 0;
        }

        idx ++;
        if (idx >= topn)
        {
            return 0;
        }
    }

    /* if list is not full, add item to the bottom */
    if (class_list.size() < topn)
    {
        class_list.push_back (item);
    }
    return 0;
}

int
invoke_classification (classification_result_t *class_ret)
{
    size_t topn = 5;

    /* copy to CUDA buffer */
    trt_copy_tensor_to_gpu (s_tensor_input);

    /* invoke inference */
    int batchSize = 1;
    s_trt_context->execute (batchSize, &s_gpu_buffers[0]);

    /* copy from CUDA buffer */
    trt_copy_tensor_from_gpu (s_tensor_output);


    std::list<classify_t> classify_list;
    for (int i = 0; i < MAX_CLASS_NUM; i ++)
    {
        classify_t item;
        item.id = i;
        item.score = get_scoreval (i);

        push_listitem (classify_list, item, topn);
    }

    int count = 0;
    for (auto itr = classify_list.begin(); itr != classify_list.end(); itr ++)
    {
        classify_t *item = &class_ret->classify[count];

        item->id    = itr->id;
        item->score = itr->score;
        memcpy (item->name, s_class_name[itr->id], 64);

        count ++;
        class_ret->num = count;
    }

    return 0;
}
