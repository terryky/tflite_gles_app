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
#include "tflite_deeplab.h"
#include "util_camera_capture.h"
#include "util_video_decode.h"

#define UNUSED(x) (void)(x)





/* resize image to DNN network input size and convert to fp32. */
void
feed_deeplab_image(texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_deeplab_input_buf (&w, &h);
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


static void
render_deeplab_result (int ofstx, int ofsty, int draw_w, int draw_h,
                       deeplab_result_t *deeplab_ret)
{
    float *segmap = deeplab_ret->segmentmap;
    int segmap_w  = deeplab_ret->segmentmap_dims[0];
    int segmap_h  = deeplab_ret->segmentmap_dims[1];
    int segmap_c  = deeplab_ret->segmentmap_dims[2];
    int x, y, c;
    unsigned int imgbuf[segmap_h][segmap_w];

    /* find the most confident class for each pixel. */
    for (y = 0; y < segmap_h; y ++)
    {
        for (x = 0; x < segmap_w; x ++)
        {
            int max_id;
            float conf_max = 0;
            for (c = 0; c < 21; c ++)
            {
                float confidence = segmap[(y * segmap_w * segmap_c)+ (x * segmap_c) + c];
                if (c == 0 || confidence > conf_max)
                {
                    conf_max = confidence;
                    max_id = c;
                }
            }
            float *col = get_deeplab_class_color (max_id);
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

    draw_2d_texture (texid, ofstx, ofsty, draw_w, draw_h, 0);

    /* class name */
    for (c = 0; c < 21; c ++)
    {
        float col_str[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float *col = get_deeplab_class_color (c);
        char *name = get_deeplab_class_name (c);
        char buf[512];
        sprintf (buf, "%2d:%s", c, name);
        draw_dbgstr_ex (buf, ofstx, ofsty + c * 22 * 0.7, 0.7f, col_str, col);
    }

    glDeleteTextures (1, &texid);
}

void
render_deeplab_heatmap (int ofstx, int ofsty, int draw_w, int draw_h, deeplab_result_t *deeplab_ret)
{
    float *segmap = deeplab_ret->segmentmap;
    int segmap_w  = deeplab_ret->segmentmap_dims[0];
    int segmap_h  = deeplab_ret->segmentmap_dims[1];
    int segmap_c  = deeplab_ret->segmentmap_dims[2];
    int x, y;
    unsigned char imgbuf[segmap_h][segmap_w];
    static int s_count = 0;
    int key_id = (s_count /10)% 21;
    s_count ++;
    float conf_min, conf_max;


#if 1
    conf_min =  0.0f;
    conf_max = 50.0f;
#else
    conf_min =  FLT_MAX;
    conf_max = -FLT_MAX;
    for (y = 0; y < segmap_h; y ++)
    {
        for (x = 0; x < segmap_w; x ++)
        {
            float confidence = segmap[(y * segmap_w * segmap_c)+ (x * segmap_c) + key_id];
            if (confidence < conf_min) conf_min = confidence;
            if (confidence > conf_max) conf_max = confidence;
        }
    }
#endif

    for (y = 0; y < segmap_h; y ++)
    {
        for (x = 0; x < segmap_w; x ++)
        {
            float confidence = segmap[(y * segmap_w * segmap_c)+ (x * segmap_c) + key_id];
            confidence = (confidence - conf_min) / (conf_max - conf_min);
            if (confidence < 0.0f) confidence = 0.0f;
            if (confidence > 1.0f) confidence = 1.0f;
            imgbuf[y][x] = confidence * 255;
        }
    }
    
    GLuint texid;
    glGenTextures (1, &texid );
    glBindTexture (GL_TEXTURE_2D, texid);

    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei (GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D (GL_TEXTURE_2D, 0, GL_LUMINANCE,
        segmap_w, segmap_h, 0, GL_LUMINANCE,
        GL_UNSIGNED_BYTE, imgbuf);

    draw_2d_colormap (texid, ofstx, ofsty, draw_w, draw_h, 0.8f, 0);

    glDeleteTextures (1, &texid);

    {
        char strbuf[128];
        sprintf (strbuf, "%2d (%f, %f) %s\n", key_id, 
            conf_min, conf_max, get_deeplab_class_name (key_id));
        draw_dbgstr (strbuf, ofstx + 5, 5);
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
    char input_name_default[] = "ride_horse.jpg";
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

    egl_init_with_platform_window_surface (2, 0, 0, 0, win_w * 2, win_h);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

    init_tflite_deeplab ();

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
     *  Render Loop
     * --------------------------------------- */
    for (count = 0; ; count ++)
    {
        deeplab_result_t deeplab_result;
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
         *  semantic segmentation
         * --------------------------------------- */
        feed_deeplab_image (&captex, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_deeplab (&deeplab_result);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        /* --------------------------------------- *
         *  render scene
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT);
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);

        /* visualize the segmentation results. */
        glViewport (win_w, 0, win_w, win_h);
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);
        render_deeplab_result (draw_x, draw_y, draw_w, draw_h, &deeplab_result);

#if 0
        render_deeplab_heatmap (draw_x, draw_y, draw_w, draw_h, &deeplab_result);
#endif

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

