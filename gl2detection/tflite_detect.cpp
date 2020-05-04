/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "util_debug.h"
#include "tflite_detect.h"
#include "detect_postprocess.h"

/* 
 * https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
 */
#define LABEL_MAP_PATH           "./detect_model/mscoco_label_map.pbtxt"

#if 1       /* Mobilenet SSD V1 with PostProcess (quant) */
#define MOBILNET_SSD_MODEL_PATH  "./detect_model/detect_regular_nms_quant.tflite"

#elif 0     /* Mobilenet SSD V1 without PostProcess (quant) */
#define MOBILNET_SSD_MODEL_PATH  "./detect_model/detect_no_nms_quant.tflite"
#define ANCHORS_FILE             "./detect_model/anchors.txt"
#define INVOKE_POSTPROCESS_AFTER_TFLITE 1

#elif 0     /* Mobilenet SSD V3 without PostProcess (float) */
#define MOBILNET_SSD_MODEL_PATH  "./detect_model/mobilenetv3_small/ssd_mobilenet_v3_small_coco_float.tflite"
#define ANCHORS_FILE             "./detect_model/mobilenetv3_small/anchors.txt"
#define INVOKE_POSTPROCESS_AFTER_TFLITE 1

#elif 0     /* Mobilenet SSD V3 without PostProcess (float) */
#define MOBILNET_SSD_MODEL_PATH  "./detect_model/mobilenetv3_large/ssd_mobilenet_v3_large_coco_float.tflite"
#define ANCHORS_FILE             "./detect_model/mobilenetv3_large/anchors.txt"
#define INVOKE_POSTPROCESS_AFTER_TFLITE 1

#else
#error "Please choose a TFLite Model."
#endif


static tflite_interpreter_t s_interpreter;
static tflite_tensor_t  s_tensor_input;

#if defined (INVOKE_POSTPROCESS_AFTER_TFLITE)
static tflite_tensor_t  s_tensor_boxes;
static tflite_tensor_t  s_tensor_scores;
static float            *s_boxes_buf;
static float            *s_scores_buf;
#else
static tflite_tensor_t  s_tensor_boxes;
static tflite_tensor_t  s_tensor_scores;
static tflite_tensor_t  s_tensor_classes;
static tflite_tensor_t  s_tensor_num;
#endif

static char  s_class_name [MAX_DETECT_CLASS + 1][128];
static float s_class_color[MAX_DETECT_CLASS + 1][4];

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


int
init_tflite_detection()
{
    tflite_create_interpreter_from_file (&s_interpreter, MOBILNET_SSD_MODEL_PATH);

    /* get input tensor */
    tflite_get_tensor_by_name (&s_interpreter, 0, "normalized_input_image_tensor",  &s_tensor_input);

#if defined (INVOKE_POSTPROCESS_AFTER_TFLITE)
    /* get output tensor */
    tflite_get_tensor_by_name (&s_interpreter, 1, "raw_outputs/box_encodings",     &s_tensor_boxes);
    tflite_get_tensor_by_name (&s_interpreter, 1, "raw_outputs/class_predictions", &s_tensor_scores);

    /* if it's a quantized model, allocate buffers for (uint8 -> float) convertion */
    if (s_tensor_scores.type == kTfLiteUInt8)
    {
        int num_anchors = s_tensor_scores.dims[1];
        int num_classes = s_tensor_scores.dims[2];

        s_scores_buf = new float[num_anchors * num_classes];
        s_boxes_buf  = new float[num_anchors * 4]; /* float4 {x0, y0, x1, y1} */
    }

    init_detect_postprocess (ANCHORS_FILE);
#else
    /* get output tensor */
    tflite_get_tensor_by_name (&s_interpreter, 1, "TFLite_Detection_PostProcess",   &s_tensor_boxes);
    tflite_get_tensor_by_name (&s_interpreter, 1, "TFLite_Detection_PostProcess:1", &s_tensor_classes);
    tflite_get_tensor_by_name (&s_interpreter, 1, "TFLite_Detection_PostProcess:2", &s_tensor_scores);
    tflite_get_tensor_by_name (&s_interpreter, 1, "TFLite_Detection_PostProcess:3", &s_tensor_num);
#endif

    load_label_map ();
    init_class_color ();

    return 0;
}

int
get_detect_input_type ()
{
    if (s_tensor_input.type == kTfLiteUInt8)
        return 1;
    else
        return 0;
}

void *
get_detect_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return s_tensor_input.ptr;
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


int
invoke_detect (detect_result_t *detection)
{
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if defined (INVOKE_POSTPROCESS_AFTER_TFLITE)
    std::vector<DetectionBox> detection_boxes = {};
    float *scores = (float *)s_tensor_scores.ptr;
    float *boxes  = (float *)s_tensor_boxes.ptr;

    /* if it's a quantized model, convert uint8 -> float */
    if (s_tensor_scores.type == kTfLiteUInt8)
    {
        int num_anchors = s_tensor_scores.dims[1];
        int num_classes = s_tensor_scores.dims[2];
        uint8_t *scores_u8 = (uint8_t *)s_tensor_scores.ptr;
        uint8_t *boxes_u8  = (uint8_t *)s_tensor_boxes.ptr;

        scores = s_scores_buf;
        boxes  = s_boxes_buf;

        for (int i = 0; i < num_anchors * num_classes; i ++)
            scores[i] = (scores_u8[i] - s_tensor_scores.quant_zerop) * s_tensor_scores.quant_scale;

        for (int i = 0; i < num_anchors * 4; i ++)
            boxes[i] = (boxes_u8[i] - s_tensor_boxes.quant_zerop) * s_tensor_boxes.quant_scale;
    }

    invoke_detection_postprocess (detection_boxes, boxes, scores);

    int num = detection_boxes.size();
    num = std::min (num, MAX_DETECT_OBJS);

    detection->num = num;
    for (int i = 0; i < num; i ++)
    {
        detection->obj[i].x1        = detection_boxes[i].x1;
        detection->obj[i].y1        = detection_boxes[i].y1;
        detection->obj[i].x2        = detection_boxes[i].x2;
        detection->obj[i].y2        = detection_boxes[i].y2;
        detection->obj[i].score     = detection_boxes[i].score;
        detection->obj[i].det_class = detection_boxes[i].class_id;
    }
#else
    float *boxes   = (float *)s_tensor_boxes.ptr;
    float *classes = (float *)s_tensor_classes.ptr;
    float *scores  = (float *)s_tensor_scores.ptr;
    float *numf    = (float *)s_tensor_num.ptr;
    int num = (int)*numf;
    num = std::min (num, MAX_DETECT_OBJS);

    detection->num = num;
    for (int i = 0; i < num; i ++)
    {
        detection->obj[i].y1        = boxes[i * sizeof(float)    ];
        detection->obj[i].x1        = boxes[i * sizeof(float) + 1];
        detection->obj[i].y2        = boxes[i * sizeof(float) + 2];
        detection->obj[i].x2        = boxes[i * sizeof(float) + 3];
        detection->obj[i].score     = scores[i];
        detection->obj[i].det_class = int(classes[i]);
    }
#endif

    return 0;
}

