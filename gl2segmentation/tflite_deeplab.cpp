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
#include "tflite_deeplab.h"
#include <float.h>

/* 
 * https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite
 */
#define DEEPLAB_MODEL_PATH  "./posenet_model/deeplabv3_257_mv_gpu.tflite"

using namespace std;
using namespace tflite;

unique_ptr<FlatBufferModel> model;
unique_ptr<Interpreter> interpreter;
ops::builtin::BuiltinOpResolver resolver;

static float   *in_ptr;
static float   *segment_ptr;
static int     s_img_w = 0;
static int     s_img_h = 0;
static int     s_seg_w = 0;
static int     s_seg_h = 0;
static int     s_seg_c = 0;

static float s_class_color[MAX_DETECT_CLASS + 1][4];
static char  s_class_name [MAX_DETECT_CLASS + 1][64] = 
{
    "background",   // 0
    "aeroplane",    // 1
    "bicycle",      // 2
    "bird",         // 3
    "boat",         // 4
    "bottle",       // 5
    "bus",          // 6
    "car",          // 7
    "cat",          // 8
    "chair",        // 9
    "cow",          // 10
    "diningtable",  // 11
    "dog",          // 12
    "horse",        // 13
    "motorbike",    // 14
    "person",       // 15
    "pottedplant",  // 16
    "sheep",        // 17
    "sofa",         // 18
    "train",        // 19
    "tvmonitor"     // 20
};

static int
init_class_color ()
{
    for (int i = 0; i < MAX_DETECT_CLASS + 1; i ++)
    {
        float *col = s_class_color[i];
        if (i == 0)
        {
            col[0] = col[1] = col[2] = 0.0f;
        }
        else
        {
            col[0] = (rand () % 255) / 255.0f;
            col[1] = (rand () % 255) / 255.0f;
            col[2] = (rand () % 255) / 255.0f;
        }
        col[3] = 0.9f;
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
init_tflite_deeplab()
{
    model = FlatBufferModel::BuildFromFile(DEEPLAB_MODEL_PATH);
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
            .precision_loss_allowed = 0,  // FP16
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

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

#if 1 /* for debug */
    print_tensor_info ();
#endif
    
    in_ptr      = interpreter->typed_input_tensor<float>(0);
    segment_ptr = interpreter->typed_output_tensor<float>(0);

    /* input image dimention */
    int input_idx = interpreter->inputs()[0];
    TfLiteIntArray *dim = interpreter->tensor(input_idx)->dims;
    s_img_w = dim->data[2];
    s_img_h = dim->data[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* heatmap dimention */
    int segment_idx = interpreter->outputs()[0];
    TfLiteIntArray *seg_dim = interpreter->tensor(segment_idx)->dims;
    s_seg_c = seg_dim->data[3];
    s_seg_w = seg_dim->data[2];
    s_seg_h = seg_dim->data[1];
    fprintf (stderr, "segment_map size: (%d, %d), (%d)\n", s_seg_w, s_seg_h, s_seg_c);

    init_class_color ();
    
    return 0;
}

void *
get_deeplab_input_buf (int *w, int *h)
{
    *w = s_img_w;
    *h = s_img_h;
    return in_ptr;
}

char *
get_deeplab_class_name (int class_idx)
{
    return s_class_name[class_idx];
}

float *
get_deeplab_class_color (int class_idx)
{
    return s_class_color[class_idx];
}


#if 0 /* for debug */
static void
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

static void
save_img ()
{
    static int s_cnt = 0;
    FILE *fp;
    char strFName[ 128 ];
    int w = s_img_w;
    int h = s_img_h;
    
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
#endif


int
invoke_deeplab (deeplab_result_t *deeplab_result)
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
    
    float *segmentmap = segment_ptr;

    deeplab_result->segmentmap         = segmentmap;
    deeplab_result->segmentmap_dims[0] = s_seg_w;
    deeplab_result->segmentmap_dims[1] = s_seg_h;
    deeplab_result->segmentmap_dims[2] = s_seg_c;

    return 0;
}

