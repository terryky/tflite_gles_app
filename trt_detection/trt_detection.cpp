/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_trt.h"
#include "util_debug.h"
#include "trt_detection.h"
#include <unistd.h>


#define UFF_MODEL_PATH      "./models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.uff"
#define PLAN_MODEL_PATH     "./models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.plan"
#define LABEL_MAP_PATH      "./models/ssd_mobilenet_v1_coco_2018_01_28/mscoco_label_map.pbtxt"

static Logger               gLogger;
static IExecutionContext   *s_trt_context;
static trt_tensor_t         s_tensor_input;
static trt_tensor_t         s_tensor_output;
static trt_tensor_t         s_tensor_numdet;
static std::vector<void *>  s_gpu_buffers;

static char  s_class_name [MAX_DETECT_CLASS + 1][128];
static float s_class_color[MAX_DETECT_CLASS + 1][4];


/* -------------------------------------------------- *
 *  load class labels
 * -------------------------------------------------- */
static char *
get_token (char *lpSrc, char *lpToken)
{
    int nTokenLength = 0;

    if (lpSrc == NULL)
    {
        *lpToken = '\0';
        return NULL;
    }

    /* skip space */
    while ( *lpSrc != '\0' )
    {
        if (isspace (*lpSrc) == 0)
            break;

        lpSrc ++;
    }

    /* if empty, no token. */
    if ( *lpSrc == '\0' )
    {
        *lpToken = '\0';
        return NULL;
    }

    /* !"#$%&'()=~|`@{[+;*:}]<,>.?/\ */
    if ( !isalnum( *lpSrc ) && *lpSrc != '_' )
    {
        nTokenLength = 1;
    }
    else if (isalnum (*lpSrc) || *lpSrc == '_' )
    {
        for (nTokenLength = 1; isalnum (lpSrc[ nTokenLength ]) || lpSrc[ nTokenLength ] == '_'; nTokenLength ++)
            ;
    }
    else
    {
        for (nTokenLength = 1; isalnum (lpSrc[ nTokenLength ]); nTokenLength ++)
            ;
    }

    if (nTokenLength == 0)
    {
        *lpToken = '\0';
        return NULL;
    }

    strncpy (lpToken, lpSrc, nTokenLength);
    lpToken[ nTokenLength ] = '\0';

    lpSrc += nTokenLength;
    return lpSrc;
}

static int
load_label_map ()
{
    char buf[512];
    char buf_token[512];
    char *p, *q;
    int id = 0;

    FILE *fp = fopen (LABEL_MAP_PATH, "r");
    if (fp == NULL)
        return 0;
    
    while (fgets (buf, 512, fp))
    {
        p = buf;
        p = get_token (p, buf_token);
        if (strcmp (buf_token, "id") == 0 )
        {
            p = get_token (p, buf_token);
            p = get_token (p, buf_token);
            id = atoi (buf_token);
        }
        if (strcmp (buf_token, "display_name") == 0 )
        {
            p = get_token (p, buf_token);
            p = get_token (p, buf_token);
            q = s_class_name[id];

            while ( *p != '"')
                *q ++ = *p ++;
            LOG ("ID[%d] %s\n", id, s_class_name[id]);
        }
    }

    fclose (fp);

    return 0;
}

static int
init_class_color ()
{
    for (int i = 0; i < MAX_DETECT_CLASS; i ++)
    {
        float *col = s_class_color[i];
        col[0] = (rand () % 255) / 255.0f;
        col[1] = (rand () % 255) / 255.0f;
        col[2] = (rand () % 255) / 255.0f;
        col[3] = 0.8f;
    }
    return 0;
}


/* -------------------------------------------------- *
 *  create cuda engine
 * -------------------------------------------------- */
static int
convert_uff_to_plan (const std::string &plan_file_name, const std::string &uff_file_name)
{
    std::vector<trt_uff_inputdef_t>  uff_input_array;
    trt_uff_inputdef_t uff_input;
    uff_input.name  = "Input";
    uff_input.dims  = nvinfer1::DimsCHW(3, 300, 300),
    uff_input.order = nvuffparser::UffInputOrder::kNCHW;
    uff_input_array.push_back (uff_input);

    std::vector<trt_uff_outputdef_t> uff_output_array;
    trt_uff_outputdef_t uff_output;
    uff_output.name = "NMS";
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



/* -------------------------------------------------- *
 *  Create TensorRT Interpreter
 * -------------------------------------------------- */
int
init_trt_detection ()
{
    ICudaEngine *engine = NULL;

    trt_initialize ();

    initLibNvInferPlugins (&gLogger, "");

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
    trt_get_tensor_by_name (engine, "Input", &s_tensor_input);
    trt_get_tensor_by_name (engine, "NMS",   &s_tensor_output);
    trt_get_tensor_by_name (engine, "NMS_1", &s_tensor_numdet);


    int num_bindings = engine->getNbBindings();
    s_gpu_buffers.resize (num_bindings);
    s_gpu_buffers[s_tensor_input .bind_idx] = s_tensor_input .gpu_mem;
    s_gpu_buffers[s_tensor_output.bind_idx] = s_tensor_output.gpu_mem;
    s_gpu_buffers[s_tensor_numdet.bind_idx] = s_tensor_numdet.gpu_mem;

    load_label_map ();
    init_class_color ();

    return 0;
}

void *
get_detect_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims.d[2];
    *h = s_tensor_input.dims.d[1];
    return s_tensor_input.cpu_mem;
}

char *
get_detect_class_name (int class_idx)
{
    return s_class_name[class_idx + 1];
}

float *
get_detect_class_color (int class_idx)
{
    return s_class_color[class_idx + 1];
}


/* -------------------------------------------------- *
 * Invoke TensorRT
 * -------------------------------------------------- */
int
invoke_detect (detect_result_t *detection)
{
    /* copy to CUDA buffer */
    trt_copy_tensor_to_gpu (s_tensor_input);

    /* invoke inference */
    int batchSize = 1;
    s_trt_context->execute (batchSize, &s_gpu_buffers[0]);

    /* copy from CUDA buffer */
    trt_copy_tensor_from_gpu (s_tensor_output);
    trt_copy_tensor_from_gpu (s_tensor_numdet);

    float *out_ptr = (float *)s_tensor_output .cpu_mem;
    int   *num_det = (int   *)s_tensor_numdet.cpu_mem;

    int count = 0;
    detection->num = 0;
    for (int i = 0; i < *num_det; i ++)
    {
        float *det = out_ptr + (7 * i);
        float score = det[2];

        if (score > 0.5f)
        {
            detection->obj[count].x1        = det[3];
            detection->obj[count].y1        = det[4];
            detection->obj[count].x2        = det[5];
            detection->obj[count].y2        = det[6];
            detection->obj[count].score     = score;
            detection->obj[count].det_class = int(det[1]) - 1;

            count ++;
            detection->num = count;
        }
    }

    return 0;
}
