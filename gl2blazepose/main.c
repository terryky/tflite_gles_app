/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
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
#include "util_matrix.h"
#include "tflite_blazepose.h"
#include "camera_capture.h"
#include "video_decode.h"
#include "render_imgui.h"

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
feed_pose_detect_image(texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_pose_detect_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

    draw_2d_texture_ex (srctex, 0, win_h - h, w, h, 1);

    glPixelStorei (GL_PACK_ALIGNMENT, 4);
    glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf_ui8);

    /* convert UI8 [0, 255] ==> FP32 [-1, 1] */
    float mean = 128.0f;
    float std  = 128.0f;
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
feed_pose_landmark_image(texture_2d_t *srctex, int win_w, int win_h, pose_detect_result_t *detection, unsigned int pose_id)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_pose_landmark_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

    float texcoord[] = { 0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 0.0f };

    if (detection->num > pose_id)
    {
        detect_region_t *region = &(detection->poses[pose_id]);
        float x0 = region->roi_coord[0].x;
        float y0 = region->roi_coord[0].y;
        float x1 = region->roi_coord[1].x; //    0--------1
        float y1 = region->roi_coord[1].y; //    |        |
        float x2 = region->roi_coord[2].x; //    |        |
        float y2 = region->roi_coord[2].y; //    3--------2
        float x3 = region->roi_coord[3].x;
        float y3 = region->roi_coord[3].y;
        texcoord[0] = x3;   texcoord[1] = y3;
        texcoord[2] = x0;   texcoord[3] = y0;
        texcoord[4] = x2;   texcoord[5] = y2;
        texcoord[6] = x1;   texcoord[7] = y1;
    }

    draw_2d_texture_ex_texcoord (srctex, 0, win_h - h, w, h, texcoord);

    glPixelStorei (GL_PACK_ALIGNMENT, 4);
    glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf_ui8);

    /* convert UI8 [0, 255] ==> FP32 [-1, 1] */
    float mean = 128.0f;
    float std  = 128.0f;
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
render_detect_region (int ofstx, int ofsty, int texw, int texh, pose_detect_result_t *detection, imgui_data_t *imgui_data)
{
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float col_red[]   = {1.0f, 0.0f, 0.0f, 1.0f};
    float *col_frame = imgui_data->frame_color;

    for (int i = 0; i < detection->num; i ++)
    {
        detect_region_t *region = &(detection->poses[i]);
        float x1 = region->topleft.x  * texw + ofstx;
        float y1 = region->topleft.y  * texh + ofsty;
        float x2 = region->btmright.x * texw + ofstx;
        float y2 = region->btmright.y * texh + ofsty;
        float score = region->score;

        /* rectangle region */
        draw_2d_rect (x1, y1, x2-x1, y2-y1, col_frame, 2.0f);

        /* class name */
        char buf[512];
        sprintf (buf, "%d", (int)(score * 100));
        draw_dbgstr_ex (buf, x1, y1, 1.0f, col_white, col_frame);

        /* key points */
        float hx = region->keys[kMidHipCenter]     .x * texw + ofstx;
        float hy = region->keys[kMidHipCenter]     .y * texh + ofsty;
        float sx = region->keys[kMidShoulderCenter].x * texw + ofstx;
        float sy = region->keys[kMidShoulderCenter].y * texh + ofsty;

        draw_2d_line (hx, hy, sx, sy, col_white, 2.0f);
        int r = 4;
        draw_2d_fillrect (hx - (r/2), hy - (r/2), r, r, col_frame);
        draw_2d_fillrect (sx - (r/2), sy - (r/2), r, r, col_frame);

        for (int j0 = 0; j0 < 4; j0 ++)
        {
            int j1 = (j0 + 1) % 4;
            float x1 = region->roi_coord[j0].x * texw + ofstx;
            float y1 = region->roi_coord[j0].y * texh + ofsty;
            float x2 = region->roi_coord[j1].x * texw + ofstx;
            float y2 = region->roi_coord[j1].y * texh + ofsty;

            draw_2d_line (x1, y1, x2, y2, col_red, 2.0f);
        }
        float cx =  region->roi_center.x * texw + ofstx;
        float cy =  region->roi_center.y * texh + ofsty;
        r = 10;
        draw_2d_fillrect (cx - (r/2), cy - (r/2), r, r, col_red);
    }
}

static void
transform_pose_landmark (fvec2 *transformed_pos, pose_landmark_result_t *landmark, detect_region_t *region)
{
    float scale_x = region->roi_size.x;
    float scale_y = region->roi_size.y;
    float pivot_x = region->roi_center.x;
    float pivot_y = region->roi_center.y;
    float rotation= region->rotation;

    float mat[16];
    matrix_identity (mat);
    matrix_translate (mat, pivot_x, pivot_y, 0);
    matrix_rotate (mat, RAD_TO_DEG(rotation), 0, 0, 1);
    matrix_scale (mat, scale_x, scale_y, 1.0f);
    matrix_translate (mat, -0.5f, -0.5f, 0);

    for (int i = 0; i < POSE_JOINT_NUM; i ++)
    {
        float vec[2] = {landmark->joint[i].x, landmark->joint[i].y};
        matrix_multvec2 (mat, vec, vec);

        transformed_pos[i].x = vec[0];
        transformed_pos[i].y = vec[1];
    }
}

static void
render_bone (int ofstx, int ofsty, int drw_w, int drw_h,
             fvec2 *transformed_pos, int id0, int id1, float *col)
{
    float x0 = transformed_pos[id0].x * drw_w + ofstx;
    float y0 = transformed_pos[id0].y * drw_h + ofsty;
    float x1 = transformed_pos[id1].x * drw_w + ofstx;
    float y1 = transformed_pos[id1].y * drw_h + ofsty;

    draw_2d_line (x0, y0, x1, y1, col, 5.0f);
}

static void
render_pose_landmark (int ofstx, int ofsty, int texw, int texh, pose_landmark_result_t *landmark,
                      pose_detect_result_t *detection, unsigned int pose_id)
{
    float col_red[]    = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_orange[] = {1.0f, 0.6f, 0.0f, 1.0f};
    float col_cyan[]   = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_lime[]   = {0.0f, 1.0f, 0.3f, 1.0f};
    float col_blue[]   = {0.0f, 0.5f, 1.0f, 1.0f};
    float col_white[]  = {1.0f, 1.0f, 1.0f, 1.0f};

    float score = landmark->score;
    char buf[512];
    sprintf (buf, "score:%4.1f", score * 100);
    draw_dbgstr_ex (buf, texw - 120, 0, 1.0f, col_white, col_red);

    fvec2 transformed_pos[POSE_JOINT_NUM];
    transform_pose_landmark (transformed_pos, landmark, &(detection->poses[pose_id]));

    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 11, 12, col_cyan);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 12, 24, col_cyan);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 24, 23, col_cyan);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 23, 11, col_cyan);

    /* right arm */
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 11, 13, col_orange);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 13, 15, col_orange);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 15, 21, col_orange);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 15, 19, col_orange);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 15, 17, col_orange);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 17, 19, col_orange);

    /* left arm */
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 12, 14, col_lime);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 14, 16, col_lime);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 16, 22, col_lime);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 16, 20, col_lime);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 16, 18, col_lime);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos, 18, 20, col_lime);

    /* face */
    render_bone (ofstx, ofsty, texw, texh, transformed_pos,  9, 10, col_blue);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos,  0,  1, col_blue);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos,  1,  2, col_blue);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos,  2,  3, col_blue);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos,  3,  7, col_blue);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos,  0,  4, col_blue);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos,  4,  5, col_blue);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos,  5,  6, col_blue);
    render_bone (ofstx, ofsty, texw, texh, transformed_pos,  6,  8, col_blue);

    for (int i = 0; i < POSE_JOINT_NUM; i ++)
    {
        float x = transformed_pos[i].x * texw + ofstx;
        float y = transformed_pos[i].y * texh + ofsty;

        int r = 9;
        draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_red);
    }
}


static void
render_cropped_pose_image (texture_2d_t *srctex, int ofstx, int ofsty, int texw, int texh,
                           pose_detect_result_t *detection, unsigned int pose_id)
{
    float texcoord[8];

    if (detection->num <= pose_id)
        return;

    detect_region_t *region = &(detection->poses[pose_id]);
    float x0 = region->roi_coord[0].x;
    float y0 = region->roi_coord[0].y;
    float x1 = region->roi_coord[1].x; //    0--------1
    float y1 = region->roi_coord[1].y; //    |        |
    float x2 = region->roi_coord[2].x; //    |        |
    float y2 = region->roi_coord[2].y; //    3--------2
    float x3 = region->roi_coord[3].x;
    float y3 = region->roi_coord[3].y;
    texcoord[0] = x0;   texcoord[1] = y0;
    texcoord[2] = x3;   texcoord[3] = y3;
    texcoord[4] = x1;   texcoord[5] = y1;
    texcoord[6] = x2;   texcoord[7] = y2;

    draw_2d_texture_ex_texcoord (srctex, ofstx, ofsty, texw, texh, texcoord);
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

#if defined (USE_IMGUI)
void
mousemove_cb (int x, int y)
{
    imgui_mousemove (x, y);
}

void
button_cb (int button, int state, int x, int y)
{
    imgui_mousebutton (button, state, x, y);
}

void
keyboard_cb (int key, int state, int x, int y)
{
}
#endif

void
setup_imgui (int win_w, int win_h, imgui_data_t *imgui_data)
{
#if defined (USE_IMGUI)
    egl_set_motion_func (mousemove_cb);
    egl_set_button_func (button_cb);
    egl_set_key_func    (keyboard_cb);

    init_imgui (win_w, win_h);
#endif

    imgui_data->frame_color[0] = 1.0f;
    imgui_data->frame_color[1] = 0.0f;
    imgui_data->frame_color[2] = 0.0f;
    imgui_data->frame_color[3] = 1.0f;
}


/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
int
main(int argc, char *argv[])
{
    char input_name_default[] = "assets/pexels-alexy-almond-3758048.jpg";
    char *input_name = NULL;
    int count;
    int win_w = 900;
    int win_h = 900;
    int texid;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    texture_2d_t captex = {0};
    double ttime[10] = {0}, interval, invoke_ms0 = 0, invoke_ms1 = 0;
    int use_quantized_tflite = 0;
    int enable_camera = 1;
    imgui_data_t imgui_data = {0};
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

    init_tflite_blazepose (use_quantized_tflite, &imgui_data.blazepose_config);

    setup_imgui (win_w, win_h, &imgui_data);

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

    for (count = 0; ; count ++)
    {
        pose_detect_result_t    detect_ret = {0};
        pose_landmark_result_t  landmark_ret[MAX_POSE_NUM] = {0};
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

        /* --------------------------------------- *
         *  Pose detection
         * --------------------------------------- */
        feed_pose_detect_image (&captex, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_pose_detect (&detect_ret, &imgui_data.blazepose_config);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms0 = ttime[3] - ttime[2];

        /* --------------------------------------- *
         *  Pose landmark
         * --------------------------------------- */
        invoke_ms1 = 0;
        for (int pose_id = 0; pose_id < detect_ret.num; pose_id ++)
        {
            feed_pose_landmark_image (&captex, win_w, win_h, &detect_ret, pose_id);

            ttime[4] = pmeter_get_time_ms ();
            invoke_pose_landmark (&landmark_ret[pose_id]);
            ttime[5] = pmeter_get_time_ms ();
            invoke_ms1 += ttime[5] - ttime[4];
        }


        /* --------------------------------------- *
         *  render scene
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT);

        /* visualize the object detection results. */
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);
        render_detect_region (draw_x, draw_y, draw_w, draw_h, &detect_ret, &imgui_data);
        render_pose_landmark (draw_x, draw_y, draw_w, draw_h, &landmark_ret[0], &detect_ret, 0);

        /* draw cropped image of the pose area */
        for (int pose_id = 0; pose_id < detect_ret.num; pose_id ++)
        {
            float w = 100;
            float h = 100;
            float x = win_w - w - 10;
            float y = h * pose_id + 10;
            float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

            render_cropped_pose_image (&captex, x, y, w, h, &detect_ret, pose_id);
            draw_2d_rect (x, y, w, h, col_white, 2.0f);
        }

        /* --------------------------------------- *
         *  post process
         * --------------------------------------- */
        draw_pmeter (0, 40);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite0 :%5.1f [ms]\nTFLite1 :%5.1f [ms]",
            interval, invoke_ms0, invoke_ms1);
        draw_dbgstr (strbuf, 10, 10);

#if defined (USE_IMGUI)
        invoke_imgui (&imgui_data);
#endif
        egl_swap();
    }

    return 0;
}

