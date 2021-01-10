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
#include "tflite_boundless.h"
#include "util_camera_capture.h"
#include "util_video_decode.h"

#define UNUSED(x) (void)(x)





/* resize image to DNN network input size and convert to fp32. */
void
feed_tflite_image (texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = get_boundless_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

    draw_2d_texture_ex (srctex, 0, win_h - h, w, h, 1);

    glPixelStorei (GL_PACK_ALIGNMENT, 4);
    glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf_ui8);

    /* convert UI8 [0, 255] ==> FP32 [ 0, 1] */
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

static float
clamp (float s)
{
    s = fmax (s, 0.0f);
    s = fmin (s, 1.0f);
    return s;
}

/* upload style transfered image to OpenGLES texture */
static int
update_style_transfered_texture (boundless_t *transfer, int *texid_mask, int *texid_gen)
{
    static int s_texid_mask = 0;
    static int s_texid_gen  = 0;
    static uint8_t *s_texbuf_mask = NULL;
    static uint8_t *s_texbuf_gen  = NULL;
    int img_w = transfer->w;
    int img_h = transfer->h;

    if (s_texid_mask == 0)
    {
        s_texbuf_mask = (uint8_t *)calloc (1, img_w * img_h * 4);
        s_texbuf_gen  = (uint8_t *)calloc (1, img_w * img_h * 4);
        s_texid_mask  = create_2d_texture (s_texbuf_mask, img_w, img_h);
        s_texid_gen   = create_2d_texture (s_texbuf_gen,  img_w, img_h);
    }

    uint8_t *d_gen  = s_texbuf_gen;
    uint8_t *d_mask = s_texbuf_mask;
    float   *s_gen  = transfer->buf_gen;
    float   *s_mask = transfer->buf_mask;

    for (int y = 0; y < img_h; y ++)
    {
        for (int x = 0; x < img_w; x ++)
        {
            float r = *s_gen ++;
            float g = *s_gen ++;
            float b = *s_gen ++;
            *d_gen ++ = (uint8_t)(clamp(r) * 255);
            *d_gen ++ = (uint8_t)(clamp(g) * 255);
            *d_gen ++ = (uint8_t)(clamp(b) * 255);
            *d_gen ++ = 0xFF;

            r = *s_mask ++;
            g = *s_mask ++;
            b = *s_mask ++;
            *d_mask ++ = (uint8_t)(clamp(r) * 255);
            *d_mask ++ = (uint8_t)(clamp(g) * 255);
            *d_mask ++ = (uint8_t)(clamp(b) * 255);
            *d_mask ++ = 0xFF;
        }
    }

    glBindTexture (GL_TEXTURE_2D, s_texid_gen);
    glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, img_w, img_h, GL_RGBA, GL_UNSIGNED_BYTE, s_texbuf_gen);

    glBindTexture (GL_TEXTURE_2D, s_texid_mask);
    glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, img_w, img_h, GL_RGBA, GL_UNSIGNED_BYTE, s_texbuf_mask);

    *texid_mask = s_texid_mask;
    *texid_gen  = s_texid_gen;
    return 0;
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
    char input_name_default[] = "assets/PAK752_miyakomac14120023_TP_V.jpg";
    char *input_name = NULL;
    int count;
    int win_w = 600;
    int win_h = 400;
    int texid;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    texture_2d_t captex = {0};
    double ttime[10] = {0}, interval, invoke_ms;
    int use_quantized_tflite = 0;
    int enable_camera = 1;
    UNUSED (argc);
    UNUSED (*argv);
#if defined (USE_INPUT_VIDEO_DECODE)
    int enable_video = 0;
#endif

    {
        int c;
        const char *optstring = "qv:x";

        while ((c = getopt (argc, argv, optstring)) != -1)
        {
            switch (c)
            {
            case 'q':
                use_quantized_tflite = 1;
                break;
#if defined (USE_INPUT_VIDEO_DECODE)
            case 'v':
                enable_video = 1;
                input_name = optarg;
                break;
#endif
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

    if (input_name == NULL)
        input_name = input_name_default;

    egl_init_with_platform_window_surface (2, 0, 0, 0, win_w * 2, win_h * 2);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

    init_tflite_boundless (use_quantized_tflite);

#if defined (USE_GL_DELEGATE) || defined (USE_GPU_DELEGATEV2)
    /* we need to recover framebuffer because GPU Delegate changes the FBO binding */
    glBindFramebuffer (GL_FRAMEBUFFER, 0);
    glViewport (0, 0, win_w, win_h);
#endif

#if defined (USE_INPUT_VIDEO_DECODE)
    /* initialize FFmpeg video decode */
    if (enable_video && init_video_decode () == 0)
    {
        create_video_texture (&captex, input_name);
        texw = captex.width;
        texh = captex.height;
        enable_camera = 0;
    }
    else
#endif
#if defined (USE_INPUT_CAMERA_CAPTURE)
    /* initialize V4L2 capture function */
    if (enable_camera && init_capture (0) == 0)
    {
        create_capture_texture (&captex);
        texw = captex.width;
        texh = captex.height;
    }
    else
#endif
    {
        load_jpg_texture (input_name, &texid, &texw, &texh);
        captex.texid  = texid;
        captex.width  = texw;
        captex.height = texh;
        captex.format = pixfmt_fourcc ('R', 'G', 'B', 'A');
        enable_camera = 0;
    }
    adjust_texture (win_w, win_h, texw, texh, &draw_x, &draw_y, &draw_w, &draw_h);

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    /* --------------------------------------- *
     *  Style transfer
     * --------------------------------------- */
    for (count = 0; ; count ++)
    {
        boundless_t style_transfered = {0};
        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime[1] = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime[1] - ttime[0] : 0;
        ttime[0] = ttime[1];

        glClear (GL_COLOR_BUFFER_BIT);
        glViewport (0, 0, win_w, win_h);

#if defined (USE_INPUT_VIDEO_DECODE)
        /* initialize FFmpeg video decode */
        if (enable_video)
        {
            update_video_texture (&captex);
        }
#endif
#if defined (USE_INPUT_CAMERA_CAPTURE)
        if (enable_camera)
        {
            update_capture_texture (&captex);
        }
#endif

        /* --------------------------------------- *
         *  style transfer
         * --------------------------------------- */
        feed_tflite_image (&captex, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_boundless (&style_transfered);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        /* --------------------------------------- *
         *  render scene (left half)
         * --------------------------------------- */
        float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
        glClear (GL_COLOR_BUFFER_BIT);

        int texid_mask, texid_gen;
        update_style_transfered_texture (&style_transfered, &texid_mask, &texid_gen);

        glViewport (0, win_h, win_w, win_h);
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);
        draw_2d_rect (draw_x, draw_y, draw_w, draw_h, col_white, 2.0f);

        glViewport (win_w, win_h, win_w, win_h);
        draw_2d_texture (texid_mask, draw_x, draw_y, draw_w, draw_h, 0);
        draw_2d_rect (draw_x, draw_y, draw_w, draw_h, col_white, 2.0f);

        glViewport (win_w / 2, 0, win_w, win_h);
        draw_2d_texture (texid_gen, draw_x, draw_y, draw_w, draw_h, 0);
        draw_2d_rect (draw_x, draw_y, draw_w, draw_h, col_white, 2.0f);

        /* --------------------------------------- *
         *  post process
         * --------------------------------------- */
        glViewport (0, 0, win_w, win_h);
        draw_pmeter (0, 40);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]", interval, invoke_ms);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

