/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <GLES2/gl2.h>
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_texture.h"
#include "util_render2d.h"
#include "util_matrix.h"
#include "tflite_dense_depth.h"
#include "camera_capture.h"
#include "video_decode.h"
#include "render_dense_depth.h"
#include "touch_event.h"
#include "render_imgui.h"

#define UNUSED(x) (void)(x)

static imgui_data_t s_gui_prop = {0};


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



void
feed_dense_depth_image(texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_dense_depth_input_buf (&w, &h);
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



float
fclampf (float val)
{
    val = fmaxf (0.0f, val);
    val = fminf (1.0f, val);
    return val;
}

static void
render_depth_image (texture_2d_t *srctex, int ofstx, int ofsty, int texw, int texh,
                    dense_depth_result_t *dense_depth_ret)
{
    float *depthmap = dense_depth_ret->depthmap;
    int depthmap_w  = dense_depth_ret->depthmap_dims[0];
    int depthmap_h  = dense_depth_ret->depthmap_dims[1];
    int x, y;
    unsigned int imgbuf[depthmap_h][depthmap_w];

    /* find the most confident class for each pixel. */
    for (y = 0; y < depthmap_h; y ++)
    {
        for (x = 0; x < depthmap_w; x ++)
        {
            float d = depthmap[y * depthmap_w + x];
            d -= 0.0;
            d /= 10.0;
            d = fclampf (d);

            unsigned char r = d * 255;
            unsigned char g = r;
            unsigned char b = r;
            unsigned char a = 255;

            imgbuf[y][x] = (a << 24) | (b << 16) | (g << 8) | (r);
        }
    }

    texture_2d_t animtex;
    create_2d_texture_ex (&animtex, imgbuf, depthmap_w, depthmap_h, pixfmt_fourcc ('R', 'G', 'B', 'A'));
    draw_2d_texture_ex (&animtex, ofstx, ofsty, texw, texh, 0);

    glDeleteTextures (1, &animtex.texid);
}


static void
render_depth_image_3d (texture_2d_t *srctex, int ofstx, int ofsty, int texw, int texh,
                       dense_depth_result_t *dense_depth_ret)
{
    float mtxGlobal[16], mtxTouch[16];
    static int s_is_first_render3d = 1;
    static mesh_obj_t s_depth_mesh;

    get_touch_event_matrix (mtxTouch);
    matrix_identity (mtxGlobal);
    matrix_translate (mtxGlobal, 0, 0, -s_gui_prop.camera_pos_z);
    matrix_mult (mtxGlobal, mtxGlobal, mtxTouch);

    float *depthmap = dense_depth_ret->depthmap;
    int depthmap_w  = dense_depth_ret->depthmap_dims[0];
    int depthmap_h  = dense_depth_ret->depthmap_dims[1];

    /* create mesh object */
    if (s_is_first_render3d)
    {
        create_mesh (&s_depth_mesh, depthmap_w - 1, depthmap_h - 1);
        s_is_first_render3d = 0;
    }
    float *vtx = s_depth_mesh.vtx_array;
    float *uv  = s_depth_mesh.uv_array;

    /* create 3D vertex coordinate */
    for (int y = 0; y < depthmap_h; y ++)
    {
        for (int x = 0; x < depthmap_w; x ++)
        {
            int   idx = (y * depthmap_w + x);
            float d = depthmap[idx];

            if (1)
            {
                d -= 0;//s_gui_prop.depth_min;
                d /= 10;//s_gui_prop.depth_max;
                d = (d * 2.0 - 1.0) * s_gui_prop.pose_scale_z;
            }
            else
            {
                //d = s_gui_prop.depth_max / d;   //  inf -> 1.0
                //d = 2 - d;                      // -inf -> 1.0
                //d = d * s_gui_prop.pose_scale_z;
            }

            vtx[3 * idx + 0] =  ((x / (float)depthmap_h) * 2.0f - 1.0f) * s_gui_prop.pose_scale_x;
            vtx[3 * idx + 1] = -((y / (float)depthmap_h) * 2.0f - 1.0f) * s_gui_prop.pose_scale_y;
            vtx[3 * idx + 2] =  d;

            uv [2 * idx + 0] = x / (float)depthmap_w;
            uv [2 * idx + 1] = y / (float)depthmap_h;
        }
    }
    float colb[] = {1.0, 1.0, 1.0, 1.0};
    draw_point_arrays (mtxGlobal, vtx, uv, depthmap_h * depthmap_w, srctex->texid, colb);

    if (s_gui_prop.draw_axis)
    {
        /* (xyz)-AXIS */
        for (int i = -1; i <= 1; i ++)
        {
            for (int j = -1; j <= 1; j ++)
            {
                float col_base[] = {0.1, 0.5, 0.5, 0.5};
                float dx = s_gui_prop.pose_scale_x;
                float dy = s_gui_prop.pose_scale_y;
                float dz = s_gui_prop.pose_scale_z;

                {
                    float v0[3] = {-dx, i * dy, j * dz};
                    float v1[3] = { dx, i * dy, j * dz};
                    float col_red[] = {1.0, 0.0, 0.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_red : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                }
                {
                    float v0[3] = {i * dx, -dy, j * dz};
                    float v1[3] = {i * dx,  dy, j * dz};
                    float col_green[] = {0.0, 1.0, 0.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_green : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                }
                {
                    float v0[3] = {i * dx, j * dy, -dz};
                    float v1[3] = {i * dx, j * dy,  dz};
                    float col_blue[] = {0.0, 0.0, 1.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_blue : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                }
            }
        }
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

void
mousemove_cb (int x, int y)
{
#if defined (USE_IMGUI)
    imgui_mousemove (x, y);
    if (imgui_is_anywindow_hovered ())
        return;
#endif

    touch_event_move (0, x, y);
}

void
button_cb (int button, int state, int x, int y)
{
#if defined (USE_IMGUI)
    imgui_mousebutton (button, state, x, y);
#endif

    if (state)
        touch_event_start (0, x, y);
    else
        touch_event_end (0);
}

void
keyboard_cb (int key, int state, int x, int y)
{
}

void
setup_imgui (int win_w, int win_h)
{
    egl_set_motion_func (mousemove_cb);
    egl_set_button_func (button_cb);
    egl_set_key_func    (keyboard_cb);

    init_touch_event (win_w, win_h);

#if defined (USE_IMGUI)
    init_imgui (win_w, win_h);
#endif

    s_gui_prop.pose_scale_x = 100.0f;
    s_gui_prop.pose_scale_y = 100.0f;
    s_gui_prop.pose_scale_z = 100.0f;
    s_gui_prop.camera_pos_z = 200.0f;
    s_gui_prop.draw_axis    = 0;
    s_gui_prop.draw_pmeter  = 1;
}


/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
int
main(int argc, char *argv[])
{
    char input_name_default[] = "assets/pexels.jpg";
    char *input_name = NULL;
    int count;
    int win_w = 900;
    int win_h = 900;
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

    egl_init_with_platform_window_surface (2, 8, 0, 0, win_w * 2, win_h);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);
    init_cube ((float)win_w / (float)win_h);

    init_tflite_dense_depth (use_quantized_tflite);
    setup_imgui (win_w * 2, win_h);

#if defined (USE_GL_DELEGATE) || defined (USE_GPU_DELEGATEV2)
    /* we need to recover framebuffer because GPU Delegate changes the FBO binding */
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
        int texid;
        load_jpg_texture (input_name, &texid, &texw, &texh);
        captex.texid  = texid;
        captex.width  = texw;
        captex.height = texh;
        captex.format = pixfmt_fourcc ('R', 'G', 'B', 'A');
    }
    adjust_texture (win_w, win_h, texw, texh, &draw_x, &draw_y, &draw_w, &draw_h);


    glClearColor (0.f, 0.f, 0.f, 1.0f);


    /* --------------------------------------- *
     *  Render Loop
     * --------------------------------------- */
    dense_depth_result_t dense_depth_result = {0};
    int is_first = 1;
    for (count = 0; ; count ++)
    {
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
         *  Dense Depth
         * --------------------------------------- */
        invoke_ms = 0;
        if (is_first)
        {
            feed_dense_depth_image (&captex, win_w, win_h);

            ttime[2] = pmeter_get_time_ms ();
            invoke_dense_depth (&dense_depth_result);
            ttime[3] = pmeter_get_time_ms ();
            invoke_ms = ttime[3] - ttime[2];

            is_first = 0;
        }

        /* --------------------------------------- *
         *  render scene (left half)
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);

        {
            float aspect = (float)dense_depth_result.depthmap_dims[0] / 
                           (float)dense_depth_result.depthmap_dims[1];
            int dw = 200 * aspect;
            int dh = 200;
            int dx = win_w - dw - 10;
            int dy = 10;
            float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

            render_depth_image (&captex, dx, dy, dw, dh, &dense_depth_result);
            draw_2d_rect (dx, dy, dw, dh, col_white, 2.0f);
        }

        /* --------------------------------------- *
         *  render scene  (right half)
         * --------------------------------------- */
        glViewport (win_w, 0, win_w, win_h);

        render_depth_image_3d (&captex, draw_x, draw_y, draw_w, draw_h, &dense_depth_result);

        /* --------------------------------------- *
         *  post process
         * --------------------------------------- */
        glViewport (0, 0, win_w, win_h);

        if (s_gui_prop.draw_pmeter)
        {
            draw_pmeter (0, 40);
        }

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite  :%5.1f [ms]", interval, invoke_ms);
        draw_dbgstr (strbuf, 10, 10);

#if defined (USE_IMGUI)
        invoke_imgui (&s_gui_prop);
#endif
        egl_swap();
    }

    return 0;
}

