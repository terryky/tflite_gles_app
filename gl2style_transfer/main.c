/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>
#include <GLES2/gl2.h>
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_texture.h"
#include "util_render2d.h"
#include "tflite_style_transfer.h"
#include "camera_capture.h"
#include "video_decode.h"

#define UNUSED(x) (void)(x)


#if defined (USE_INPUT_CAMERA_CAPTURE)
static void
update_capture_texture (texture_2d_t *captex)
{
    int      cap_w, cap_h;
    uint32_t cap_fmt;
    void     *cap_buf;

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

#if defined (USE_INPUT_VIDEO_DECODE)
static void
update_video_texture (texture_2d_t *captex)
{
    int   video_w, video_h;
    uint32_t video_fmt;
    void *video_buf;

    get_video_dimension (&video_w, &video_h);
    get_video_pixformat (&video_fmt);
    get_video_buffer (&video_buf);

    if (video_buf)
    {
        int texw = video_w;
        int texh = video_h;
        int texfmt = GL_RGBA;
        switch (video_fmt)
        {
        case pixfmt_fourcc('Y', 'U', 'Y', 'V'):
            texw = video_w / 2;
            break;
        default:
            break;
        }

        glBindTexture (GL_TEXTURE_2D, captex->texid);
        glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, texw, texh, texfmt, GL_UNSIGNED_BYTE, video_buf);
    }
}

static int
init_video_texture (texture_2d_t *captex, const char *fname)
{
    int      vid_w, vid_h;
    uint32_t vid_fmt;

    open_video_file (fname);

    get_video_dimension (&vid_w, &vid_h);
    get_video_pixformat (&vid_fmt);

    create_2d_texture_ex (captex, NULL, vid_w, vid_h, vid_fmt);
    start_video_decode ();

    return 0;
}
#endif /* USE_INPUT_VIDEO_DECODE */


/* resize image to DNN network input size and convert to fp32. */
void
feed_style_transfer_image(int is_predict, texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32;
    unsigned char *buf_ui8 = NULL;
    static int buf_w = 0, buf_h = 0;
    static unsigned char *pui8 = NULL;

    if (is_predict)
        buf_fp32 = (float *)get_style_predict_input_buf (&w, &h);
    else
        buf_fp32 = (float *)get_style_transfer_content_input_buf (&w, &h);

    if (buf_w != w || buf_h != h)
    {
        if (pui8)
            free (pui8);
        pui8 = (unsigned char *)malloc(w * h * 4);
        buf_w = w;
        buf_h = h;
    }

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

void
store_style_predict (style_predict_t *style)
{
    int size = style->size;
    float *param = style->param;
    float *store_param = (float *)calloc (1, size * sizeof(float));

    style->param = store_param;

    while (size --> 0)
    {
        *store_param ++ = *param ++;
    }
}

void
feed_blend_style (style_predict_t *style0, style_predict_t *style1, float ratio)
{
    int size;
    float *s0 = style0->param;
    float *s1 = style1->param;
    float *d = get_style_transfer_style_input_buf (&size);

    if (style0->size != size || style1->size != size)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return;
    }
    
    for (int i = 0; i < size; i ++)
    {
        float src0 = *s0;
        float src1 = *s1;
        float src  = (ratio * src1) + ((1.0f - ratio) * src0);

        *d ++ = src;
        s0 ++;
        s1 ++;
    }
}


/* upload style transfered image to OpenGLES texture */
static int
update_style_transfered_texture (style_transfer_t *transfer)
{
    static int s_texid = 0;
    static uint8_t *s_texbuf = NULL;
    int img_w = transfer->w;
    int img_h = transfer->h;

    if (s_texid == 0)
    {
        s_texbuf = (uint8_t *)calloc (1, img_w * img_h * 4);
        s_texid  = create_2d_texture (s_texbuf, img_w, img_h);
    }

    uint8_t *d = s_texbuf;
    float   *s = transfer->img;

    for (int y = 0; y < img_h; y ++)
    {
        for (int x = 0; x < img_w; x ++)
        {
            float r = *s ++;
            float g = *s ++;
            float b = *s ++;
            *d ++ = (uint8_t)(r * 255);
            *d ++ = (uint8_t)(g * 255);
            *d ++ = (uint8_t)(b * 255);
            *d ++ = 0xFF;
        }
    }

    glBindTexture (GL_TEXTURE_2D, s_texid);
    glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, img_w, img_h, GL_RGBA, GL_UNSIGNED_BYTE, s_texbuf);

    return s_texid;
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
    char input_name_default[] = "pakutaso_famicom.jpg";
    char *input_name = NULL;
    char input_style_name_default[] = "munch_scream.jpg";
    char *input_style_name = NULL;
    int count;
    int win_w = 720 * 2;
    int win_h = 540;
    int texid;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    texture_2d_t captex = {0};
    texture_2d_t styletex = {0};
    float style_ratio = -0.1f;
    double ttime[10] = {0}, interval, invoke_ms;
    int enable_camera = 1;
    UNUSED (argc);
    UNUSED (*argv);
#if defined (USE_INPUT_VIDEO_DECODE)
    int enable_video = 0;
#endif

    /* gl2style_transfer [content_file_name] [style_file_name] */
    {
        int c;
        const char *optstring = "v:x";

        while ((c = getopt (argc, argv, optstring)) != -1)
        {
            switch (c)
            {
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
            if (input_name == NULL)
                input_name = argv[optind];
            else if (input_style_name == NULL)
                input_style_name = argv[optind];;
            optind++;
        }
    }

    if (input_name == NULL)
        input_name = input_name_default;
    if (input_style_name == NULL)
        input_style_name = input_style_name_default;

    egl_init_with_platform_window_surface (2, 0, 0, 0, win_w, win_h);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

    init_tflite_style_transfer ();

#if defined (USE_GL_DELEGATE) || defined (USE_GPU_DELEGATEV2)
    /* we need to recover framebuffer because GPU Delegate changes the context */
    glBindFramebuffer (GL_FRAMEBUFFER, 0);
    glViewport (0, 0, win_w, win_h);
#endif

#if defined (USE_INPUT_VIDEO_DECODE)
    /* initialize FFmpeg video decode */
    if (enable_video && init_video_decode () == 0)
    {
        init_video_texture (&captex, input_name);
        texw = captex.width;
        texh = captex.height;
        enable_camera = 0;
    }
    else
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
        load_jpg_texture (input_name, &texid, &texw, &texh);
        captex.texid  = texid;
        captex.width  = texw;
        captex.height = texh;
        captex.format = pixfmt_fourcc ('R', 'G', 'B', 'A');
    }
    adjust_texture (win_w, win_h, texw, texh, &draw_x, &draw_y, &draw_w, &draw_h);

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    /* --------------------------------------- *
     *  Style prediction
     * --------------------------------------- */
    style_predict_t style_predict[2] = {0};
    {
        int w, h;
        load_jpg_texture (input_style_name, &texid, &w, &h);
        styletex.texid  = texid;
        styletex.width  = texw;
        styletex.height = texh;
        styletex.format = pixfmt_fourcc ('R', 'G', 'B', 'A');

        /* predict style of original image */
        glClear (GL_COLOR_BUFFER_BIT);
        feed_style_transfer_image (1, &captex, win_w, win_h);
        invoke_style_predict (&style_predict[0]);
        store_style_predict (&style_predict[0]);

        /* predict style of target image */
        glClear (GL_COLOR_BUFFER_BIT);
        feed_style_transfer_image (1, &styletex, win_w, win_h);
        invoke_style_predict (&style_predict[1]);
        store_style_predict (&style_predict[1]);
    }

    /* --------------------------------------- *
     *  Style transfer
     * --------------------------------------- */
    for (count = 0; ; count ++)
    {
        style_transfer_t style_transfered = {0};
        char strbuf[512];

        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime[1] = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime[1] - ttime[0] : 0;
        ttime[0] = ttime[1];

        glClear (GL_COLOR_BUFFER_BIT);

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

#if 0
        /* 
         *  update style parameter blend ratio.
         *      0.0: apply 100[%] style of original image.
         *      1.0: apply 100[%] style of target image.
         */
        style_ratio += 0.1f;
        if (style_ratio > 1.01f)
            style_ratio = -0.1f;
#else
        style_ratio = 1.0f;
#endif

        /* feed style parameter and original image */
        feed_blend_style (&style_predict[0], &style_predict[1], style_ratio);
        feed_style_transfer_image (0, &captex, win_w, win_h);

        /* invoke pose estimation using TensorflowLite */
        ttime[2] = pmeter_get_time_ms ();
        invoke_style_transfer (&style_transfered);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        /* visualize the style transform results. */
        glClear (GL_COLOR_BUFFER_BIT);
        int transfered_texid = update_style_transfered_texture (&style_transfered);
#if 0
        if (style_ratio < 0.0f)     /* render original content image */
            draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);
        else                        /* render style transformed image */
            draw_2d_texture (transfered_texid,  draw_x, draw_y, draw_w, draw_h, 0);
#else
        draw_2d_texture_ex (&captex, 0, 0, 750, 540, 0);
        draw_2d_texture (transfered_texid, 720, 0, 720, 540, 0);
#endif        
        /* render the target style image */
        {
            float col_black[] = {1.0f, 1.0f, 1.0f, 1.0f};
            draw_2d_texture_ex (&styletex,  win_w - 200, 0, 200, 200, 0);
            draw_2d_rect (win_w - 200, 0, 200, 200, col_black, 2.0f);
        }

        draw_pmeter (0, 40);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]\nstyle_ratio=%.1f", 
                                interval, invoke_ms, style_ratio);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

