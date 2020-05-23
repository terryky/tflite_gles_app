/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <GLES2/gl2.h>
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_texture.h"
#include "util_render2d.h"
#include "util_matrix.h"
#include "tflite_objectron.h"
#include "camera_capture.h"

#define UNUSED(x) (void)(x)


#if defined (USE_INPUT_CAMERA_CAPTURE)
static void
update_capture_texture (texture_2d_t *captex)
{
    int   cap_w, cap_h;
    uint32_t cap_fmt;
    void *cap_buf;

    get_capture_dimension (&cap_w, &cap_h);
    get_capture_pixformat (&cap_fmt);
    get_capture_buffer (&cap_buf);
    if (cap_buf)
    {
        int texw = cap_w;
        int texh = cap_h;
        int texfmt = GL_RGBA;
        switch (cap_fmt)
        {
        case pixfmt_fourcc('Y', 'U', 'Y', 'V'):
            texw = cap_w / 2;
            break;
        default:
            break;
        }

        glBindTexture (GL_TEXTURE_2D, captex->texid);
        glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, texw, texh, texfmt, GL_UNSIGNED_BYTE, cap_buf);
    }
}

static int
init_capture_texture (texture_2d_t *captex)
{
    int      cap_w, cap_h;
    uint32_t cap_fmt;

    get_capture_dimension (&cap_w, &cap_h);
    get_capture_pixformat (&cap_fmt);

    create_2d_texture_ex (captex, NULL, cap_w, cap_h, cap_fmt);
    start_capture ();

    return 0;
}

#endif


void
feed_objectron_image(texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_objectron_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

    draw_2d_texture_ex (srctex, 0, win_h - h, w, h, 1);

    glPixelStorei (GL_PACK_ALIGNMENT, 4);
    glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf_ui8);

    /* convert UI8 [0, 255] ==> FP32 [0, 1] */
    float mean =   0.0f;
    float std  = 255.0f;
    for (y = 0; y < h; y ++)
    {
        for (x = 0; x < w; x ++)
        {
            int r = *buf_ui8 ++;
            int g = *buf_ui8 ++;
            int b = *buf_ui8 ++;
            buf_ui8 ++;          /* skip alpha */
            *buf_fp32 ++ = (float)(r - mean) / std;
            *buf_fp32 ++ = (float)(g - mean) / std;
            *buf_fp32 ++ = (float)(b - mean) / std;
        }
    }

    return;
}


static void
render_bbox_edge (int ofstx, int ofsty, int texw, int texh, object_t *obj, int idx0, int idx1)
{
    float col_green[] = {0.0f, 1.0f, 0.0f, 1.0f};

    float x0 = obj->bbox2d[idx0].x * texw + ofstx;
    float y0 = obj->bbox2d[idx0].y * texh + ofsty;
    float x1 = obj->bbox2d[idx1].x * texw + ofstx;
    float y1 = obj->bbox2d[idx1].y * texh + ofsty;

    draw_2d_line (x0, y0, x1, y1, col_green, 5.0f);
}

/*
 *              x                              x
 *      0 + + + + + + + + 4                 .-------
 *      +\                +\                |\
 *      + \ y             + \             z | \ y
 *      +  \              +  \              |  \
 *      +   2 + + + + + + + + 6
 *    z +   +             +   +
 *      +   +             +   +
 *      +   +     C       +   +
 *      +   +             +   +
 *      1 + + + + + + + + 5   +
 *       \  +              \  +
 *        \ +               \ +
 *         \+                \+
 *          3 + + + + + + + + 7
 */
static void
render_detect_region (int ofstx, int ofsty, int texw, int texh, objectron_result_t *detection)
{
    float col_red[]   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

    for (int i = 0; i < detection->num; i ++)
    {
        object_t *obj = &(detection->objects[i]);
        float x = obj->center_x * texw + ofstx;
        float y = obj->center_y * texh + ofsty;

        /* rectangle region */
        int r = 4;
        draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_red);

        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 0, 2);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 2, 3);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 3, 1);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 1, 0);

        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 4, 6);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 6, 7);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 7, 5);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 5, 4);

        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 0, 4);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 4, 6);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 6, 2);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 2, 0);

        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 1, 5);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 5, 7);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 7, 3);
        render_bbox_edge (ofstx, ofsty, texw, texh, obj, 3, 1);

        /* score */
        char buf[512];
        sprintf (buf, "%d", (int)(obj->belief * 100));

        x = obj->bbox2d[0].x * texw + ofstx;
        y = obj->bbox2d[0].y * texh + ofsty;
        draw_dbgstr_ex (buf, x, y, 1.0f, col_white, col_red);
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
    char input_name_default[] = "chair.jpg";
    char *input_name = input_name_default;
    int count;
    int win_w = 800;
    int win_h = 800;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    texture_2d_t captex = {0};
    double ttime[10] = {0}, interval, invoke_ms0 = 0;
    int use_quantized_tflite = 0;
    int enable_camera = 1;
    UNUSED (argc);
    UNUSED (*argv);

    {
        int c;
        const char *optstring = "qx";

        while ((c = getopt (argc, argv, optstring)) != -1) 
        {
            switch (c)
            {
            case 'q':
                use_quantized_tflite = 1;
                break;
            case 'x':
                enable_camera = 0;
                break;
            }
        }

        while (optind < argc) 
        {
            input_name = argv[optind];
            optind++;
        }
    }

    egl_init_with_platform_window_surface (2, 0, 0, 0, win_w, win_h);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

    init_tflite_objectron (use_quantized_tflite);

#if defined (USE_GL_DELEGATE) || defined (USE_GPU_DELEGATEV2)
    /* we need to recover framebuffer because GPU Delegate changes the FBO binding */
    glBindFramebuffer (GL_FRAMEBUFFER, 0);
    glViewport (0, 0, win_w, win_h);
#endif

#if defined (USE_INPUT_CAMERA_CAPTURE)
    /* initialize V4L2 capture function */
    if (enable_camera && init_capture () == 0)
    {
        init_capture_texture (&captex);
        texw = captex.width;
        texh = captex.height;
    }
    else
#endif
    {
        int texid;
        load_jpg_texture (input_name, &texid, &texw, &texh);
        captex.texid  = texid;
        captex.width  = texw;
        captex.height = texh;
        captex.format = pixfmt_fourcc ('R', 'G', 'B', 'A');
    }
    adjust_texture (win_w, win_h, texw, texh, &draw_x, &draw_y, &draw_w, &draw_h);


    glClearColor (0.f, 0.f, 0.f, 1.0f);
    glClear (GL_COLOR_BUFFER_BIT);
    glViewport (0, 0, win_w, win_h);


    /* --------------------------------------- *
     *  Render Loop
     * --------------------------------------- */
    for (count = 0; ; count ++)
    {
        objectron_result_t objectron_ret = {0};

        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime[1] = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime[1] - ttime[0] : 0;
        ttime[0] = ttime[1];

        glClear (GL_COLOR_BUFFER_BIT);
        glViewport (0, 0, win_w, win_h);

#if defined (USE_INPUT_CAMERA_CAPTURE)
        if (enable_camera)
        {
            update_capture_texture (&captex);
        }
#endif

        /* --------------------------------------- *
         *  3D object detection
         * --------------------------------------- */
        feed_objectron_image (&captex, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_objectron (&objectron_ret);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms0 = ttime[3] - ttime[2];

        /* --------------------------------------- *
         *  render scene (left half)
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* visualize the 3d object detection results. */
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);

        render_detect_region (draw_x, draw_y, draw_w, draw_h, &objectron_ret);

        /* --------------------------------------- *
         *  post process
         * --------------------------------------- */
        glViewport (0, 0, win_w, win_h);
        draw_pmeter (0, 40);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]",
            interval, invoke_ms0);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

