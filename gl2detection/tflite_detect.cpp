/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tflite_detect.h"

/* 
 * https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
 */
#define LABEL_MAP_PATH           "./detect_model/mscoco_label_map.pbtxt"
#define MOBILNET_SSD_MODEL_PATH  "./detect_model/detect_regular_nms_quant.tflite"

using namespace std;
using namespace tflite;

unique_ptr<FlatBufferModel> model;
unique_ptr<Interpreter> interpreter;
ops::builtin::BuiltinOpResolver resolver;

static uint8_t *in_ptr;
static float   *boxes_ptr;
static float   *classes_ptr;
static float   *scores_ptr;
static float   *num_ptr;

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
            fprintf (stderr, "ID[%d] %s\n", id, s_class_name[id]);
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


static void
print_tensor_dim (int tensor_id)
{
    TfLiteIntArray *dim = interpreter->tensor(tensor_id)->dims;
    
    for (int i = 0; i < dim->size; i ++)
    {
        if (i > 0)
            fprintf (stderr, "x");
        fprintf (stderr, "%d", dim->data[i]);
    }
    fprintf (stderr, "\n");
}

static void
print_tensor_info ()
{
    int i, idx;
    int in_size  = interpreter->inputs().size();
    int out_size = interpreter->outputs().size();

    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, "tensors size     : %ld\n", interpreter->tensors_size());
    fprintf (stderr, "nodes   size     : %ld\n", interpreter->nodes_size());
    fprintf (stderr, "number of inputs : %d\n", in_size);
    fprintf (stderr, "number of outputs: %d\n", out_size);
    fprintf (stderr, "input(0) name    : %s\n", interpreter->GetInputName(0));

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, "                     name                     bytes  type  scale   zero_point\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    int t_size = interpreter->tensors_size();
    for (i = 0; i < t_size; i++) 
    {
        fprintf (stderr, "Tensor[%2d] %-32s %8ld, %2d, %f, %3d\n", i, 
            interpreter->tensor(i)->name, 
            interpreter->tensor(i)->bytes,
            interpreter->tensor(i)->type,
            interpreter->tensor(i)->params.scale,
            interpreter->tensor(i)->params.zero_point);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, " Input Tensor Dimension\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    for (i = 0; i < in_size; i ++)
    {
        idx = interpreter->inputs()[i];
        fprintf (stderr, "Tensor[%2d]: ", idx);
        print_tensor_dim (idx);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, " Output Tensor Dimension\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    for (i = 0; i < out_size; i ++)
    {
        idx = interpreter->outputs()[i];
        fprintf (stderr, "Tensor[%2d]: ", idx);
        print_tensor_dim (idx);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    PrintInterpreterState(interpreter.get());
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
}

int
init_tflite_detection()
{
    model = FlatBufferModel::BuildFromFile(MOBILNET_SSD_MODEL_PATH);
    if (!model)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 1 /* for debug */
    print_tensor_info ();
#endif
    
    in_ptr      = interpreter->typed_input_tensor<uint8_t>(0);
    boxes_ptr   = interpreter->typed_output_tensor<float>(0);
    classes_ptr = interpreter->typed_output_tensor<float>(1);
    scores_ptr  = interpreter->typed_output_tensor<float>(2);
    num_ptr     = interpreter->typed_output_tensor<float>(3);

    load_label_map ();
    init_class_color ();

    return 0;
}

void *
get_detect_src_buf ()
{
    return in_ptr;
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


void
load_dummy_img()
{
    FILE *fp;
    
    if ((fp = fopen ("000000000139_RGB888_SIZE300x300.img", "rb" )) == NULL)
    {
        fprintf (stderr, "Can't open img file.\n");
        return;
    }

    fseek (fp, 0, SEEK_END);
    size_t len = ftell (fp);
    fseek( fp, 0, SEEK_SET );

    void *buf = malloc(len);
    if (fread (buf, 1, len, fp) < len)
    {
        fprintf (stderr, "Can't read img file.\n");
        return;
    }

    memcpy (in_ptr, buf, 300 * 300 * 3);

    free (buf);
}

void
save_img ()
{
    static int s_cnt = 0;
    FILE *fp;
    char strFName[ 128 ];
    int w = 300;
    int h = 300;
    
    sprintf (strFName, "detect%04d_RGB888_SIZE%dx%d.img", s_cnt, w, h);
    s_cnt ++;

    fp = fopen (strFName, "wb");
    if (fp == NULL)
    {
        fprintf (stderr, "FATAL ERROR at %s(%d)\n", __FILE__, __LINE__);
        return;
    }

    fwrite (in_ptr, 3, w * h, fp);
    fclose (fp);
}



int
invoke_detect (detect_result_t *detection)
{
#if 0 /* load dummy image. */
    load_dummy_img();
#endif

#if 0 /* dump image of DL network input. */
    save_img ();
#endif

    if (interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    float *boxes   = boxes_ptr;
    float *classes = classes_ptr;
    float *scores  = scores_ptr;
    int num = (int)*num_ptr;
    num = min (num, MAX_DETECT_OBJS);
    
    detection->num = num;
    for (int i = 0; i < num; i ++)
    {
        float y1 = boxes[i * sizeof(float)    ];
        float x1 = boxes[i * sizeof(float) + 1];
        float y2 = boxes[i * sizeof(float) + 2];
        float x2 = boxes[i * sizeof(float) + 3];
        float score = scores[i] ;
        int detected_class = int(classes[i]);

        detection->obj[i].x1 = x1;
        detection->obj[i].y1 = y1;
        detection->obj[i].x2 = x2;
        detection->obj[i].y2 = y2;
        detection->obj[i].score = score;
        detection->obj[i].det_class = detected_class;

#if 0
        fprintf (stderr, "[%2d/%2d](%8.5f, %8.5f, %8.5f, %8.5f): %4.1f, (%2d) %s\n", 
                    i, num,
                    x1, y1, x2, y2, score * 100, detected_class,
                    get_detect_class_name (detected_class));
#endif
    }

    return 0;
}

