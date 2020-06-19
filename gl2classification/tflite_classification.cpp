/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "tflite_classification.h"
#include <list>

/* 
 * https://www.tensorflow.org/lite/guide/hosted_models
 */
#define CLASSIFY_MODEL_PATH        "./classification_model/mobilenet_v1_1.0_224.tflite"
#define CLASSIFY_QUANT_MODEL_PATH  "./classification_model/mobilenet_v1_1.0_224_quant.tflite"
#define CLASSIFY_LABEL_MAP_PATH    "./classification_model/class_label.txt"

static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_output;

static char                 s_class_name [MAX_CLASS_NUM][64];


/* -------------------------------------------------- *
 *  load class labels
 * -------------------------------------------------- */
static int
load_label_map ()
{
    FILE *fp = fopen (CLASSIFY_LABEL_MAP_PATH, "r");
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
 *  Create TFLite Interpreter
 * -------------------------------------------------- */
int
init_tflite_classification(int use_quantized_tflite)
{
    const char *model;

    if (use_quantized_tflite)
    {
        model = CLASSIFY_QUANT_MODEL_PATH;
    }
    else
    {
        model = CLASSIFY_MODEL_PATH;
    }

    /* Face detect */
    tflite_create_interpreter_from_file (&s_interpreter, model);
    tflite_get_tensor_by_name (&s_interpreter, 0, "input",                             &s_tensor_input);
    tflite_get_tensor_by_name (&s_interpreter, 1, "MobilenetV1/Predictions/Reshape_1", &s_tensor_output);

    load_label_map ();

    return 0;
}

int
get_classification_input_type ()
{
    if (s_tensor_input.type == kTfLiteUInt8)
        return 1;
    else
        return 0;
}

void *
get_classification_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return s_tensor_input.ptr;
}



/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
static float
get_scoreval (int class_id)
{
    if (s_tensor_output.type == kTfLiteFloat32)
    {
        float *val = (float *)s_tensor_output.ptr;
        return val[class_id];
    }

    if (s_tensor_output.type == kTfLiteUInt8)
    {
        uint8_t *val8 = (uint8_t *)s_tensor_output.ptr;
        float scale = s_tensor_output.quant_scale;
        float zerop = s_tensor_output.quant_zerop;
        float fval = (val8[class_id] - zerop) * scale;
        return fval;
    }

    return 0;
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

    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }


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
