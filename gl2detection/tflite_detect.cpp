/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#if defined (USE_GL_DELEGATE)
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif
#if defined (USE_GPU_DELEGATEV2)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif
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

#elif 1     /* Mobilenet SSD V3 without PostProcess (float) */
#define MOBILNET_SSD_MODEL_PATH  "./detect_model/mobilenetv3_small/ssd_mobilenet_v3_small_coco_float.tflite"
#define ANCHORS_FILE             "./detect_model/mobilenetv3_small/anchors.txt"
#define INVOKE_POSTPROCESS_AFTER_TFLITE 1
#endif


using namespace std;
using namespace tflite;

unique_ptr<FlatBufferModel> model;
unique_ptr<Interpreter> interpreter;
ops::builtin::BuiltinOpResolver resolver;

typedef struct tensor_info_t
{
    TfLiteType  type;        /* [1] kTfLiteFloat32, [2] kTfLiteInt32, [3] kTfLiteUInt8 */
    float       scale;
    int         zero_point;
} tensor_info_t;


static void   *in_ptr;

#if defined (INVOKE_POSTPROCESS_AFTER_TFLITE)
static uint8_t *boxes_u8_ptr;
static uint8_t *scores_u8_ptr;
static float   *boxes_ptr;
static float   *scores_ptr;
static tensor_info_t s_tensor_boxes;
static tensor_info_t s_tensor_scores;
static int     s_num_anchors = 0;
static int     s_num_classes = 0;
#else
static float   *boxes_ptr;
static float   *classes_ptr;
static float   *scores_ptr;
static float   *num_ptr;
#endif

static int     s_img_w = 0;
static int     s_img_h = 0;

static tensor_info_t s_tensor_input;

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
    fprintf (stderr, "tensors size     : %zu\n", interpreter->tensors_size());
    fprintf (stderr, "nodes   size     : %zu\n", interpreter->nodes_size());
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
        fprintf (stderr, "Tensor[%2d] %-32s %8zu, %2d, %f, %3d\n", i,
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
get_outtensor_idx_by_name (const char *key_name)
{
    int out_size = interpreter->outputs().size();

    for (int i = 0; i < out_size; i ++)
    {
        int idx = interpreter->outputs()[i];
        const char *tensor_name = interpreter->tensor(idx)->name;

        if (strcmp (tensor_name, key_name) == 0)
            return i;
    }
    return -1;
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

#if defined (USE_GL_DELEGATE)
    const TfLiteGpuDelegateOptions options = {
        .metadata = NULL,
        .compile_options = {
            .precision_loss_allowed = 1,  // FP16
            .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
            .dynamic_batch_enabled = 0,   // Not fully functional yet
        },
    };
    auto* delegate = TfLiteGpuDelegateCreate(&options);

    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
#endif

#if defined (USE_GPU_DELEGATEV2)
    const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 1, // FP16
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
    };
    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
#endif

    interpreter->SetNumThreads(4);
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 1 /* for debug */
    print_tensor_info ();
#endif

    /* input image dimention */
    int input_idx = interpreter->inputs()[0];
    TfLiteTensor *input_tensor = interpreter->tensor(input_idx);
    TfLiteIntArray *dim = input_tensor->dims;
    s_img_w = dim->data[2];
    s_img_h = dim->data[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);
    s_tensor_input.type = input_tensor->type;
    
    if (s_tensor_input.type == kTfLiteUInt8)
        in_ptr = interpreter->typed_input_tensor<uint8_t>(0);
    else
        in_ptr = interpreter->typed_input_tensor<float>(0);


#if defined (INVOKE_POSTPROCESS_AFTER_TFLITE)
    int boxes_out_idx  = get_outtensor_idx_by_name ("raw_outputs/box_encodings");
    int scores_out_idx = get_outtensor_idx_by_name ("raw_outputs/class_predictions");
    fprintf (stderr, "boxes_out_idx  = %d\n", boxes_out_idx);
    fprintf (stderr, "scores_out_idx = %d\n", scores_out_idx);

    /* scores dimention [1, 1917, 91] */
    int scores_idx = interpreter->outputs()[scores_out_idx];
    TfLiteTensor *scores_tensor = interpreter->tensor(scores_idx);
    TfLiteIntArray *scores_dim  = scores_tensor->dims;
    s_num_anchors = scores_dim->data[1];
    s_num_classes = scores_dim->data[2];
    fprintf (stderr, "num_anchors = %d\n", s_num_anchors);
    fprintf (stderr, "num_classes = %d\n", s_num_classes);

    s_tensor_scores.type       = scores_tensor->type;
    s_tensor_scores.scale      = scores_tensor->params.scale;
    s_tensor_scores.zero_point = scores_tensor->params.zero_point;

    /* boxes dimention [1, 1917, 4] */
    int boxes_idx = interpreter->outputs()[boxes_out_idx];
    TfLiteTensor *boxes_tensor = interpreter->tensor(boxes_idx);

    s_tensor_boxes.type       = boxes_tensor->type;
    s_tensor_boxes.scale      = boxes_tensor->params.scale;
    s_tensor_boxes.zero_point = boxes_tensor->params.zero_point;

    if (s_tensor_scores.type == kTfLiteUInt8)
    {
        scores_u8_ptr = interpreter->typed_output_tensor<uint8_t>(scores_out_idx);
        boxes_u8_ptr  = interpreter->typed_output_tensor<uint8_t>(boxes_out_idx);

        /* allocate buffers for float convertion */
        scores_ptr = new float[s_num_anchors * s_num_classes];
        boxes_ptr  = new float[s_num_anchors * 4]; /* float4 {x0, y0, x1, y1} */
    }
    else
    {
        scores_ptr = interpreter->typed_output_tensor<float>(scores_out_idx);
        boxes_ptr  = interpreter->typed_output_tensor<float>(boxes_out_idx);
    }
#else
    boxes_ptr   = interpreter->typed_output_tensor<float>(0);
    classes_ptr = interpreter->typed_output_tensor<float>(1);
    scores_ptr  = interpreter->typed_output_tensor<float>(2);
    num_ptr     = interpreter->typed_output_tensor<float>(3);
#endif

    load_label_map ();
    init_class_color ();

#if defined (INVOKE_POSTPROCESS_AFTER_TFLITE)
    init_detect_postprocess (ANCHORS_FILE);
#endif

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
    *w = s_img_w;
    *h = s_img_h;
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

#if defined (INVOKE_POSTPROCESS_AFTER_TFLITE)
    std::vector<DetectionBox> detection_boxes = {};

    if (s_tensor_scores.type == kTfLiteUInt8)
    {
        for (int i = 0; i < s_num_anchors * s_num_classes; i ++)
        {
            scores_ptr[i] = (scores_u8_ptr[i] - s_tensor_scores.zero_point) * s_tensor_scores.scale;
        }
    }

    if (s_tensor_boxes.type == kTfLiteUInt8)
    {
        for (int i = 0; i < s_num_anchors * 4; i ++)
        {
            boxes_ptr[i] = (boxes_u8_ptr[i] - s_tensor_boxes.zero_point) * s_tensor_boxes.scale;
        }
    }

    invoke_detection_postprocess (detection_boxes, boxes_ptr, scores_ptr);

    int num = detection_boxes.size();
    num = min (num, MAX_DETECT_OBJS);
    detection->num = num;

    for (int i = 0; i < num; i ++)
    {
        float x1 = detection_boxes[i].x1;
        float y1 = detection_boxes[i].y1;
        float x2 = detection_boxes[i].x2;
        float y2 = detection_boxes[i].y2;
        float score = detection_boxes[i].score;
        int detected_class = detection_boxes[i].class_id;

        detection->obj[i].x1 = x1;
        detection->obj[i].y1 = y1;
        detection->obj[i].x2 = x2;
        detection->obj[i].y2 = y2;
        detection->obj[i].score = score;
        detection->obj[i].det_class = detected_class;

        //fprintf (stderr, "[%2d] (%f, %f, %f, %f) %f %d\n", i, x1, y1, x2, y2, score, detected_class);
    }
#else
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
#endif

    return 0;
}

