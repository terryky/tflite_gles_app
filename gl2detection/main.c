/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <GLES2/gl2.h>
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_texture.h"
#include "util_render2d.h"
#include "tflite_detect.h"

#define UNUSED(x) (void)(x)


/* resize image to (300x300) for input image of MobileNet SSD */
void
feed_detect_image_uint8 (int texid, int win_w, int win_h)
{
    int w, h;
    uint8_t *buf_u8 = (uint8_t *)get_detect_input_buf (&w, &h);

    draw_2d_texture (texid, 0, win_h - h, w, h, 1);

#if 0 /* if your platform supports glReadPixles(GL_RGB), use this code. */
    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    glReadPixels (0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, buf);
#else /* if your platform supports only glReadPixels(GL_RGBA), try this code. */
    {
        int x, y;
        unsigned char *bufRGBA = (unsigned char *)malloc (w * h * 4);
        unsigned char *pRGBA = bufRGBA;
        glPixelStorei (GL_PACK_ALIGNMENT, 4);
        glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, bufRGBA);

        for (y = 0; y < h; y ++)
        {
            for (x = 0; x < w; x ++)
            {
                int r = *pRGBA ++;
                int g = *pRGBA ++;
                int b = *pRGBA ++;
                pRGBA ++;          /* skip alpha */

                *buf_u8 ++ = r;
                *buf_u8 ++ = g;
                *buf_u8 ++ = b;
            }
        }
        free (bufRGBA);
    }
#endif
}

void
feed_detect_image_float (int texid, int win_w, int win_h)
{
    int w, h;
    float *buf_fp32 = (float *)get_detect_input_buf (&w, &h);

    draw_2d_texture (texid, 0, win_h - h, w, h, 1);

    int x, y;
    unsigned char *bufRGBA = (unsigned char *)malloc (w * h * 4);
    unsigned char *pRGBA = bufRGBA;
    glPixelStorei (GL_PACK_ALIGNMENT, 4);
    glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, bufRGBA);

    /* convert UI8 [0, 255] ==> FP32 [-1, 1] */
    float mean = 128.0f;
    float std  = 128.0f;
    for (y = 0; y < h; y ++)
    {
        for (x = 0; x < w; x ++)
        {
            int r = *pRGBA ++;
            int g = *pRGBA ++;
            int b = *pRGBA ++;
            pRGBA ++;          /* skip alpha */
            *buf_fp32 ++ = (float)(r - mean) / std;
            *buf_fp32 ++ = (float)(g - mean) / std;
            *buf_fp32 ++ = (float)(b - mean) / std;
        }
    }
    free (bufRGBA);
}

void
feed_detect_image(int texid, int win_w, int win_h)
{
    int type = get_detect_input_type ();
    if (type)
        feed_detect_image_uint8 (texid, win_w, win_h);
    else
        feed_detect_image_float (texid, win_w, win_h);
}

void
render_detect_region (int ofstx, int ofsty, int texw, int texh, detect_result_t *detection)
{
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

    for (int i = 0; i < detection->num; i ++)
    {
        float x1 = detection->obj[i].x1 * texw + ofstx;
        float y1 = detection->obj[i].y1 * texh + ofsty;
        float x2 = detection->obj[i].x2 * texw + ofstx;
        float y2 = detection->obj[i].y2 * texh + ofsty;
        float score   = detection->obj[i].score;
        int det_class = detection->obj[i].det_class;

        /* rectangle region */
        float *col = get_detect_class_color(det_class);
        draw_2d_rect (x1, y1, x2-x1, y2-y1, col, 2.0f);

        /* class name */
        char *name = get_detect_class_name (det_class);
        char buf[512];
        sprintf (buf, "%s(%d)", name, (int)(score * 100));
        draw_dbgstr_ex (buf, x1, y1, 1.0f, col_white, col);
    }
}

/* Adjust the texture size to fit the window size
 *
 *                      Portrait
 *     Landscape        +------+
 *     +-+------+-+     +------+
 *     | |      | |     |      |
 *     | |      | |     |      |
 *     +-+------+-+     +------+
 *                      +------+
 */
static void
adjust_texture (int win_w, int win_h, int texw, int texh, 
                int *dx, int *dy, int *dw, int *dh)
{
    float win_aspect = (float)win_w / (float)win_h;
    float tex_aspect = (float)texw  / (float)texh;
    float scale;
    float scaled_w, scaled_h;
    float offset_x, offset_y;

    if (win_aspect > tex_aspect)
    {
        scale = (float)win_h / (float)texh;
        scaled_w = scale * texw;
        scaled_h = scale * texh;
        offset_x = (win_w - scaled_w) * 0.5f;
        offset_y = 0;
    }
    else
    {
        scale = (float)win_w / (float)texw;
        scaled_w = scale * texw;
        scaled_h = scale * texh;
        offset_x = 0;
        offset_y = (win_h - scaled_h) * 0.5f;
    }

    *dx = (int)offset_x;
    *dy = (int)offset_y;
    *dw = (int)scaled_w;
    *dh = (int)scaled_h;
}


/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
int
main(int argc, char *argv[])
{
    char input_name_default[] = "food.jpg";
    char *input_name = input_name_default;
    int count;
    int win_w = 960;
    int win_h = 540;
    int texid;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    double ttime0 = 0, ttime1 = 0, interval;
    UNUSED (argc);
    UNUSED (*argv);

    if (argc > 1)
        input_name = argv[1];

    egl_init_with_platform_window_surface (2, 0, 0, 0, win_w, win_h);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);
    init_tflite_detection ();

#if defined (USE_GL_DELEGATE) || defined (USE_GPU_DELEGATEV2)
    /* we need to recover framebuffer because GPU Delegate changes the context */
    glBindFramebuffer (GL_FRAMEBUFFER, 0);
    glViewport (0, 0, win_w, win_h);
#endif

    load_jpg_texture (input_name, &texid, &texw, &texh);
    adjust_texture (win_w, win_h, texw, texh, &draw_x, &draw_y, &draw_w, &draw_h);

    glClearColor (0.7f, 0.7f, 0.7f, 1.0f);

    for (count = 0; ; count ++)
    {
        detect_result_t detection;
        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime1 = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime1 - ttime0 : 0;
        ttime0 = ttime1;

        glClear (GL_COLOR_BUFFER_BIT);

        /* invoke object detection using TensorflowLite */
        feed_detect_image (texid, win_w, win_h);
        invoke_detect (&detection);

        /* visualize the object detection results. */
        draw_2d_texture (texid,  draw_x, draw_y, draw_w, draw_h, 0);
        render_detect_region (draw_x, draw_y, draw_w, draw_h, &detection);

        draw_pmeter (0, 40);

        sprintf (strbuf, "%.1f [ms]\n", interval);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

