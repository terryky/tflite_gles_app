/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <GLES2/gl2.h>
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_texture.h"
#include "util_render2d.h"
#include "tflite_hair_segmentation.h"
#include "render_hair.h"
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
feed_segmentation_image (texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_segmentation_input_buf (&w, &h);
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
            *buf_fp32 ++ = 0.0f;
        }
    }

    return;
}

static void 
colormap_hsv (float h, float col[4])
{
    float s = 1.0f;
    float v = 1.0f;

    float r = v;
    float g = v;
    float b = v;

    h *= 6.0f;
    int i = (int) h;
    float f = h - (float) i;

    switch (i) 
    {
    default:
    case 0:
        g *= 1 - s * (1 - f);
        b *= 1 - s;
        break;
    case 1:
        r *= 1 - s * f;
        b *= 1 - s;
        break;
    case 2:
        r *= 1 - s;
        b *= 1 - s * (1 - f);
        break;
    case 3:
        r *= 1 - s;
        g *= 1 - s * f;
        break;
    case 4:
        r *= 1 - s * (1 - f);
        g *= 1 - s;
        break;
    case 5:
        g *= 1 - s;
        b *= 1 - s * f;
        break;
    }

    float lumi = (r * 0.299f + g * 0.587f + b * 0.114f);

    r *= lumi;
    g *= lumi;
    b *= lumi;

    col[0] = r;
    col[1] = g;
    col[2] = b;
    col[3] = 1.0f;
}



void
render_segment_result (int ofstx, int ofsty, int draw_w, int draw_h, 
                       texture_2d_t *srctex, segmentation_result_t *segment_ret)
{
    float *segmap = segment_ret->segmentmap;
    int segmap_w  = segment_ret->segmentmap_dims[0];
    int segmap_h  = segment_ret->segmentmap_dims[1];
    int segmap_c  = segment_ret->segmentmap_dims[2];
    int x, y, c;
    unsigned int imgbuf[segmap_h][segmap_w];
    float hair_color[4] = {0};
    float back_color[4] = {0};
    static float s_count = 0.0f;

    s_count += 0.05f;
    if (s_count >= 1.0f)
        s_count = 0.0f;

    colormap_hsv (s_count, hair_color);

    /* find the most confident class for each pixel. */
    for (y = 0; y < segmap_h; y ++)
    {
        for (x = 0; x < segmap_w; x ++)
        {
            int max_id;
            float conf_max = 0;
            for (c = 0; c < MAX_SEGMENT_CLASS; c ++)
            {
                float confidence = segmap[(y * segmap_w * segmap_c)+ (x * segmap_c) + c];
                if (c == 0 || confidence > conf_max)
                {
                    conf_max = confidence;
                    max_id = c;
                }
            }

            float *col;
            if (max_id > 0)
                col = hair_color;
            else
                col = back_color;

            unsigned char r = ((int)(col[0] * 255)) & 0xff;
            unsigned char g = ((int)(col[1] * 255)) & 0xff;
            unsigned char b = ((int)(col[2] * 255)) & 0xff;
            unsigned char a = ((int)(col[3] * 255)) & 0xff;

            imgbuf[y][x] = (a << 24) | (b << 16) | (g << 8) | (r);
        }
    }

    GLuint texid;
    glGenTextures (1, &texid );
    glBindTexture (GL_TEXTURE_2D, texid);

    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei (GL_UNPACK_ALIGNMENT, 4);

    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA,
        segmap_w, segmap_h, 0, GL_RGBA,
        GL_UNSIGNED_BYTE, imgbuf);

#if 0
    draw_colored_hair (srctex, texid, ofstx, ofsty, draw_w, draw_h, 0, hair_color);
#else
    draw_2d_texture_ex (srctex, ofstx, ofsty, draw_w, draw_h, 0);

    unsigned int blend_add  [] = {GL_ONE, GL_ONE, GL_ZERO, GL_ONE};
    draw_2d_texture_blendfunc (texid, ofstx, ofsty, draw_w, draw_h, 0, blend_add);
#endif

    glDeleteTextures (1, &texid);
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
    char input_name_default[] = "assets/pakutaso.jpg";
    char *input_name = NULL;
    int count;
    int win_w = 800;
    int win_h = 600;
    int texid;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    texture_2d_t captex = {0};
    double ttime[10] = {0}, interval, invoke_ms;
    int enable_camera = 1;
    UNUSED (argc);
    UNUSED (*argv);
#if defined (USE_INPUT_VIDEO_DECODE)
    int enable_video = 0;
#endif

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
            input_name = argv[optind];
            optind++;
        }
    }

    if (input_name == NULL)
        input_name = input_name_default;

    egl_init_with_platform_window_surface (2, 0, 0, 0, win_w, win_h);

    init_2d_renderer (win_w, win_h);
    init_hair_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);
    init_tflite_segmentation ();

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

    glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
    for (count = 0; ; count ++)
    {
        segmentation_result_t segment_result;
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

        /* invoke hair segmentation using TensorflowLite */
        feed_segmentation_image (&captex, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_segmentation (&segment_result);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        glClear (GL_COLOR_BUFFER_BIT);

        /* visualize the segmentation results. */
        render_segment_result (draw_x, draw_y, draw_w, draw_h, &captex, &segment_result);

        draw_pmeter (0, 40);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]", interval, invoke_ms);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

