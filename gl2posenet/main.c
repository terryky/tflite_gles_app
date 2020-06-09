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
#include "tflite_posenet.h"
#include "ssbo_tensor.h"
#include "camera_capture.h"
#include "video_decode.h"
#include "particle.h"

#define UNUSED(x) (void)(x)

//#define USE_FACE_MASK
//#define USE_FIREBALL_PARTICLE


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
feed_posenet_image(texture_2d_t *srctex, ssbo_t *ssbo, int win_w, int win_h)
{
#if defined (USE_INPUT_SSBO)
    resize_texture_to_ssbo (srctex->texid, ssbo);
#else
    int x, y, w, h;
#if defined (USE_QUANT_TFLITE_MODEL)
    unsigned char *buf_u8 = (unsigned char *)get_posenet_input_buf (&w, &h);
#else
    float *buf_fp32 = (float *)get_posenet_input_buf (&w, &h);
#endif
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
#if defined (USE_QUANT_TFLITE_MODEL)
            *buf_u8 ++ = r;
            *buf_u8 ++ = g;
            *buf_u8 ++ = b;
#else
            *buf_fp32 ++ = (float)(r - mean) / std;
            *buf_fp32 ++ = (float)(g - mean) / std;
            *buf_fp32 ++ = (float)(b - mean) / std;
#endif
        }
    }

#endif
    return;
}


#if defined (USE_FACE_MASK)
static void
render_facemask (int x, int y, int w, int h, posenet_result_t *pose_ret)
{
    static int s_mask_texid = 0;
    static int s_mask_w;
    static int s_mask_h;

    if (s_mask_texid == 0)
    {
        load_png_texture ("facemask/facemask.png", &s_mask_texid, &s_mask_w, &s_mask_h);
    }

    for (int i = 0; i < pose_ret->num; i ++)
    {
        float rx = pose_ret->pose[i].key[kRightEar].x * w + x;
        float ry = pose_ret->pose[i].key[kRightEar].y * h + y;
        float lx = pose_ret->pose[i].key[kLeftEar ].x * w + x;
        float ly = pose_ret->pose[i].key[kLeftEar ].y * h + y;
        float cx = (rx + lx) * 0.5f;
        float cy = (ry + ly) * 0.5f;
        float scale = (rx - lx) / (float)s_mask_w * 2.5;
        float mask_w = s_mask_w * scale;
        float mask_h = s_mask_h * scale;
        draw_2d_texture (s_mask_texid,
                         cx - mask_w * 0.5f, cy - mask_h * 0.5f,
                         mask_w, mask_h, 1);
    }
}
#endif

/* render a bone of skelton. */
void
render_bone (int ofstx, int ofsty, int drw_w, int drw_h, 
             posenet_result_t *pose_ret, int pid, 
             enum pose_key_id id0, enum pose_key_id id1,
             float *col)
{
    float x0 = pose_ret->pose[pid].key[id0].x * drw_w + ofstx;
    float y0 = pose_ret->pose[pid].key[id0].y * drw_h + ofsty;
    float x1 = pose_ret->pose[pid].key[id1].x * drw_w + ofstx;
    float y1 = pose_ret->pose[pid].key[id1].y * drw_h + ofsty;
    float s0 = pose_ret->pose[pid].key[id0].score;
    float s1 = pose_ret->pose[pid].key[id1].score;

    /* if the confidence score is low, draw more transparently. */
    col[3] = (s0 + s1) * 0.5f;
    draw_2d_line (x0, y0, x1, y1, col, 5.0f);

    col[3] = 1.0f;
}

void
render_posenet_result (int x, int y, int w, int h, posenet_result_t *pose_ret)
{
    float col_red[]    = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_orange[] = {1.0f, 0.6f, 0.0f, 1.0f};
    float col_cyan[]   = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_lime[]   = {0.0f, 1.0f, 0.3f, 1.0f};
    float col_pink[]   = {1.0f, 0.0f, 1.0f, 1.0f};
    float col_blue[]   = {0.0f, 0.5f, 1.0f, 1.0f};
    
    for (int i = 0; i < pose_ret->num; i ++)
    {
        /* draw skelton */

        /* body */
        render_bone (x, y, w, h, pose_ret, i, kLeftShoulder,  kRightShoulder, col_cyan);
        render_bone (x, y, w, h, pose_ret, i, kLeftShoulder,  kLeftHip,       col_cyan);
        render_bone (x, y, w, h, pose_ret, i, kRightShoulder, kRightHip,      col_cyan);
        render_bone (x, y, w, h, pose_ret, i, kLeftHip,       kRightHip,      col_cyan);

        /* legs */
        render_bone (x, y, w, h, pose_ret, i, kLeftHip,       kLeftKnee,      col_pink);
        render_bone (x, y, w, h, pose_ret, i, kLeftKnee,      kLeftAnkle,     col_pink);
        render_bone (x, y, w, h, pose_ret, i, kRightHip,      kRightKnee,     col_blue);
        render_bone (x, y, w, h, pose_ret, i, kRightKnee,     kRightAnkle,    col_blue);
        
        /* arms */
        render_bone (x, y, w, h, pose_ret, i, kLeftShoulder,  kLeftElbow,     col_orange);
        render_bone (x, y, w, h, pose_ret, i, kLeftElbow,     kLeftWrist,     col_orange);
        render_bone (x, y, w, h, pose_ret, i, kRightShoulder, kRightElbow,    col_lime  );
        render_bone (x, y, w, h, pose_ret, i, kRightElbow,    kRightWrist,    col_lime  );

        /* draw key points */
        for (int j = 0; j < kPoseKeyNum; j ++)
        {
            float keyx = pose_ret->pose[i].key[j].x * w + x;
            float keyy = pose_ret->pose[i].key[j].y * h + y;
            int r = 9;
            draw_2d_fillrect (keyx - (r/2), keyy - (r/2), r, r, col_red);
        }

#if defined (USE_FIREBALL_PARTICLE)
        {
            float x0 = pose_ret->pose[i].key[kRightWrist].x * w + x;
            float y0 = pose_ret->pose[i].key[kRightWrist].y * h + y;
            float x1 = pose_ret->pose[i].key[kLeftWrist].x * w + x;
            float y1 = pose_ret->pose[i].key[kLeftWrist].y * h + y;
            render_posenet_particle (x0, y0, x1, y1);
        }
#endif
    }

#if defined (USE_FACE_MASK)
    render_facemask (x, y, w, h, pose_ret);
#endif
}

void
render_posenet_heatmap (int ofstx, int ofsty, int draw_w, int draw_h, posenet_result_t *pose_ret)
{
    float *heatmap = pose_ret->pose[0].heatmap;
    int heatmap_w  = pose_ret->pose[0].heatmap_dims[0];
    int heatmap_h  = pose_ret->pose[0].heatmap_dims[1];
    int x, y;
    unsigned char imgbuf[heatmap_w * heatmap_h];
    float conf_min, conf_max;
    static int s_count = 0;
    int key_id = (s_count /10)% 17;
    s_count ++;

#if 1
    conf_min = -5.0f;
    conf_max =  1.0f;
#else
    conf_min =  FLT_MAX;
    conf_max = -FLT_MAX;
    for (y = 0; y < heatmap_h; y ++)
    {
        for (x = 0; x < heatmap_w; x ++)
        {
            float confidence = heatmap[(y * heatmap_w * 17)+ (x * 17) + key_id];
            if (confidence < conf_min) conf_min = confidence;
            if (confidence > conf_max) conf_max = confidence;
        }
    }
#endif

    for (y = 0; y < heatmap_h; y ++)
    {
        for (x = 0; x < heatmap_w; x ++)
        {
            float confidence = heatmap[(y * heatmap_w * 17)+ (x * 17) + key_id];
            confidence = (confidence - conf_min) / (conf_max - conf_min);
            if (confidence < 0.0f) confidence = 0.0f;
            if (confidence > 1.0f) confidence = 1.0f;
            imgbuf[y * heatmap_w + x] = confidence * 255;
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
        heatmap_w, heatmap_h, 0, GL_LUMINANCE,
        GL_UNSIGNED_BYTE, imgbuf);

    draw_2d_colormap (texid, ofstx, ofsty, draw_w, draw_h, 0.8f, 0);

    glDeleteTextures (1, &texid);

    {
        char strKey[][32] = {"Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder",
                             "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip",
                             "LKnee", "RKnee", "LAnkle", "RAnkle"};
        draw_dbgstr (strKey[key_id], ofstx + 5, 5);
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
    char input_name_default[] = "pakutaso_person.jpg";
    char *input_name = NULL;
    int count;
    int win_w = 600;
    int win_h = 600;
    int texid;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    texture_2d_t captex = {0};
    ssbo_t *ssbo = NULL;
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

    egl_init_with_platform_window_surface (2, 0, 0, 0, win_w, win_h);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

#if defined (USE_INPUT_SSBO)
    ssbo = init_ssbo_tensor (512, 512);
#endif

    init_tflite_posenet (use_quantized_tflite, ssbo);

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

#if defined (USE_FIREBALL_PARTICLE)
    init_posenet_particle (win_w, win_h);
#endif

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    for (count = 0; ; count ++)
    {
        posenet_result_t pose_ret = {0};
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

        /* invoke pose estimation using TensorflowLite */
        feed_posenet_image (&captex, ssbo, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_posenet (&pose_ret);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        glClear (GL_COLOR_BUFFER_BIT);

#if defined (USE_INPUT_SSBO) /* for Debug. */
        /* visualize the contents of SSBO for input tensor. */
        visualize_ssbo (ssbo);
#endif
        /* visualize the object detection results. */
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);
        render_posenet_result (draw_x, draw_y, draw_w, draw_h, &pose_ret);

#if 0
        render_posenet_heatmap (draw_x, draw_y, draw_w, draw_h, &pose_ret);
#endif

        draw_pmeter (0, 40);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]", interval, invoke_ms);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

