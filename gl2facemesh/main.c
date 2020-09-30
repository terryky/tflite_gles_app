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
#include "tflite_facemesh.h"
#include "render_facemesh.h"
#include "camera_capture.h"
#include "video_decode.h"
#include "render_imgui.h"

#define UNUSED(x) (void)(x)

static imgui_data_t s_gui_prop = {0};

typedef struct maskimage_t
{
    char fname[64];
    int  is_static_mask;
} maskimage_t;

static maskimage_t s_maskimages[] =
{
    {"assets/mask/lena.jpg",     0},
    {"assets/mask/soseki.jpg",   0},
    {"assets/mask/bakatono.jpg", 0},
    {"assets/mask/khamun.jpg",   0},
    {"assets/mask/kin.jpg",      1}
};

static int s_num_maskimages = sizeof (s_maskimages) / sizeof (maskimage_t);



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
feed_face_detect_image(texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_face_detect_input_buf (&w, &h);
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
feed_face_landmark_image(texture_2d_t *srctex, int win_w, int win_h, face_detect_result_t *detection, unsigned int face_id)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_facemesh_landmark_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

    float texcoord[] = { 0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 0.0f };

    if (detection->num > face_id)
    {
        face_t *face = &(detection->faces[face_id]);
        float x0 = face->face_pos[0].x;
        float y0 = face->face_pos[0].y;
        float x1 = face->face_pos[1].x; //    0--------1
        float y1 = face->face_pos[1].y; //    |        |
        float x2 = face->face_pos[2].x; //    |        |
        float y2 = face->face_pos[2].y; //    3--------2
        float x3 = face->face_pos[3].x;
        float y3 = face->face_pos[3].y;
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


static void
render_detect_region (int ofstx, int ofsty, int texw, int texh, face_detect_result_t *detection)
{
    float col_red[]   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

    for (int i = 0; i < detection->num; i ++)
    {
        face_t *face = &(detection->faces[i]);
        float x1 = face->topleft.x  * texw + ofstx;
        float y1 = face->topleft.y  * texh + ofsty;
        float x2 = face->btmright.x * texw + ofstx;
        float y2 = face->btmright.y * texh + ofsty;
        float score = face->score;

        /* rectangle region */
        draw_2d_rect (x1, y1, x2-x1, y2-y1, col_red, 2.0f);

        /* class name */
        char buf[512];
        sprintf (buf, "%d", (int)(score * 100));
        draw_dbgstr_ex (buf, x1, y1, 1.0f, col_white, col_red);
#if 0
        /* key points */
        for (int j = 0; j < kFaceKeyNum; j ++)
        {
            float x = face->keys[j].x * texw + ofstx;
            float y = face->keys[j].y * texh + ofsty;

            int r = 4;
            draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_red);
        }
#endif
    }
}


static void
rot_vec (float *x, float *y, float rotation)
{
    *x = (*x) * cos(rotation) - (*y) * sin(rotation);
    *y = (*x) * sin(rotation) + (*y) * cos(rotation);
}

static void
compute_3d_face_pos (face_landmark_result_t *dst_facemesh, int texw, int texh,
                     face_landmark_result_t *src_facemesh, face_t *face)
{
    float xoffset   = face->face_cx;// - 0.5f;
    float yoffset   = face->face_cy;// - 0.5f;
    float xsize     = face->face_w;
    float ysize     = face->face_h;
    float rotation  = face->rotation;

    for (int i = 0; i < FACE_KEY_NUM; i ++)
    {
        float x = src_facemesh->joint[i].x;
        float y = src_facemesh->joint[i].y;
        float z = src_facemesh->joint[i].z;

        x = x - 0.5f;
        y = y - 0.5f;
        rot_vec (&x, &y, rotation);
        x *= xsize;
        y *= ysize;
        x += xoffset;
        y += yoffset;

        dst_facemesh->joint[i].x = x;
        dst_facemesh->joint[i].y = y;
        dst_facemesh->joint[i].z = z;
    }
}

static void
render_face_landmark (int ofstx, int ofsty, int texw, int texh,
                      face_landmark_result_t *facemesh, face_t *face,
                      int texid_mask,
                      face_landmark_result_t *facemesh_mask, face_t *face_mask,
                      int meshline)
{
    int eyehole = s_gui_prop.mask_eye_hole;

    face_landmark_result_t facemesh_draw;
    compute_3d_face_pos (&facemesh_draw, texw, texh, facemesh, face);

    face_landmark_result_t facemesh_draw_mask;
    compute_3d_face_pos (&facemesh_draw_mask, texw, texh, facemesh_mask, face_mask);

#if 0
    float col_red[]   = {0.0f, 1.0f, 0.0f, 1.0f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float score = facemesh->score;
    char buf[512];
    sprintf (buf, "score:%4.1f", score * 100);
    draw_dbgstr_ex (buf, texw - 120, 0, 1.0f, col_white, col_red);
#endif

    for (int i = 0; i < FACE_KEY_NUM; i ++)
    {
        float x = facemesh_draw.joint[i].x  * texw + ofstx;
        float y = facemesh_draw.joint[i].y  * texh + ofsty;
        facemesh_draw.joint[i].x = x;
        facemesh_draw.joint[i].y = y;

        //int r = 4;
        //draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_red);
    }

    float mask_color[] = {1.0f, 1.0f, 1.0f, s_gui_prop.mask_alpha};
    draw_facemesh_tri_tex (texid_mask, facemesh_draw.joint, facemesh_draw_mask.joint,
                           mask_color, eyehole);

    if (meshline)
    {
        float col_white[] = {1.0f, 1.0f, 1.0f, 0.3f};
        draw_facemesh_line (facemesh_draw.joint, col_white, eyehole);
    }
}

static void
render_3d_scene (int ofstx, int ofsty, int texw, int texh)
{
    float mtxGlobal[16];
    float floor_size_x = 300.0f;
    float floor_size_y = 300.0f;
    float floor_size_z = 300.0f;

    /* background */
    matrix_identity (mtxGlobal);
    matrix_translate (mtxGlobal, 0, floor_size_y * 0.9f, 0);
    matrix_scale  (mtxGlobal, floor_size_x, floor_size_y, floor_size_z);
    draw_floor (mtxGlobal, floor_size_x/10, floor_size_y/10);
}

static void
render_cropped_face_image (texture_2d_t *srctex, int ofstx, int ofsty, int texw, int texh,
                           face_detect_result_t *detection, unsigned int face_id)
{
    float texcoord[8];

    if (detection->num <= face_id)
        return;

    face_t *face = &(detection->faces[face_id]);
    float x0 = face->face_pos[0].x;
    float y0 = face->face_pos[0].y;
    float x1 = face->face_pos[1].x; //    0--------1
    float y1 = face->face_pos[1].y; //    |        |
    float x2 = face->face_pos[2].x; //    |        |
    float y2 = face->face_pos[2].y; //    3--------2
    float x3 = face->face_pos[3].x;
    float y3 = face->face_pos[3].y;
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

void
mousemove_cb (int x, int y)
{
#if defined (USE_IMGUI)
    imgui_mousemove (x, y);
    if (imgui_is_anywindow_hovered ())
        return;
#endif
}

void
button_cb (int button, int state, int x, int y)
{
#if defined (USE_IMGUI)
    imgui_mousebutton (button, state, x, y);
#endif
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

#if defined (USE_IMGUI)
    init_imgui (win_w, win_h);
#endif

    s_gui_prop.mask_alpha       = 0.7;
    s_gui_prop.mask_eye_hole    = 0;
    s_gui_prop.draw_mesh_line   = 0;
    s_gui_prop.draw_detect_rect = 0;
    s_gui_prop.draw_pmeter      = 1;

    s_gui_prop.mask_num    = s_num_maskimages;
    s_gui_prop.cur_mask_id = 0;
}


/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
int
main(int argc, char *argv[])
{
    char input_name_default[] = "pakutaso.jpg";
    char *input_name = input_name_default;
    int count;
    int win_w = 900;
    int win_h = 900;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    texture_2d_t captex = {0};
    double ttime[10] = {0}, interval, invoke_ms0 = 0, invoke_ms1 = 0;
    int use_quantized_tflite = 0;
    int enable_video = 0;
    int enable_camera = 1;
    int mask_eye_hole = 0;
    UNUSED (argc);
    UNUSED (*argv);

    {
        int c;
        const char *optstring = "eqv:x";

        while ((c = getopt (argc, argv, optstring)) != -1)
        {
            switch (c)
            {
            case 'e':
                mask_eye_hole = 1;
                break;
            case 'q':
                use_quantized_tflite = 1;
                break;
            case 'v':
                enable_video = 1;
                input_name = optarg;
                break;
            case 'x':
                enable_camera = 0;
                break;
            default:
                fprintf (stderr, "inavlid option: %c\n", optopt);
                exit (0);
            }
        }

        while (optind < argc)
        {
            input_name = argv[optind];
            optind++;
        }
    }

    egl_init_with_platform_window_surface (2, 0, 0, 0, win_w * 2, win_h);

    init_2d_renderer (win_w, win_h);
    init_facemesh_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);
    init_cube ((float)win_w / (float)win_h);

    init_tflite_facemesh (use_quantized_tflite);
    setup_imgui (win_w * 2, win_h);
    s_gui_prop.mask_eye_hole = mask_eye_hole;

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
        enable_video = 0;
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
     *  prepare "facemask"
     * --------------------------------------- */
    face_detect_result_t    *face_detect_mask;
    face_landmark_result_t  *face_mesh_mask;
    int *texid_mask;

    face_detect_mask = (face_detect_result_t *)calloc (s_num_maskimages, sizeof(face_detect_result_t));
    face_mesh_mask = (face_landmark_result_t *)calloc (s_num_maskimages, sizeof(face_landmark_result_t));
    texid_mask = (int *)calloc (s_num_maskimages, sizeof (int));

    for (int mask_id = 0; mask_id < s_num_maskimages; mask_id ++)
    {
        int tw, th;
        char *mask_fname    = s_maskimages[mask_id].fname;
        int  is_static_mask = s_maskimages[mask_id].is_static_mask;

        load_jpg_texture (mask_fname, &texid_mask[mask_id], &tw, &th);

        if (is_static_mask == 0)
        {
            texture_2d_t masktex = {0};
            masktex.texid  = texid_mask[mask_id];
            masktex.width  = tw;
            masktex.height = th;
            masktex.format = pixfmt_fourcc ('R', 'G', 'B', 'A');

            feed_face_detect_image (&masktex, win_w, win_h);
            invoke_face_detect (&face_detect_mask[mask_id]);

            int face_id = 0;
            feed_face_landmark_image (&masktex, win_w, win_h, &face_detect_mask[mask_id], face_id);

            invoke_facemesh_landmark (&face_mesh_mask[mask_id]);
        }
        else
        {
            get_static_facemesh_landmark (&face_detect_mask[mask_id], &face_mesh_mask[mask_id]);
        }
    }


    /* --------------------------------------- *
     *  Render Loop
     * --------------------------------------- */
    for (count = 0; ; count ++)
    {
        face_detect_result_t    face_detect_ret = {0};
        face_landmark_result_t  face_mesh_ret[MAX_FACE_NUM] = {0};

        int mask_id = (count / 100) % s_num_maskimages;
        mask_id = s_gui_prop.cur_mask_id;
        face_detect_result_t   *cur_face_detect_mask = &face_detect_mask[mask_id];
        face_landmark_result_t *cur_face_mesh_mask = &face_mesh_mask[mask_id];
        int cur_texid_mask = texid_mask[mask_id];

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
         *  face detection
         * --------------------------------------- */
        feed_face_detect_image (&captex, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_face_detect (&face_detect_ret);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms0 = ttime[3] - ttime[2];

        /* --------------------------------------- *
         *  face landmark
         * --------------------------------------- */
        invoke_ms1 = 0;
        for (int face_id = 0; face_id < face_detect_ret.num; face_id ++)
        {
            feed_face_landmark_image (&captex, win_w, win_h, &face_detect_ret, face_id);

            ttime[4] = pmeter_get_time_ms ();
            invoke_facemesh_landmark (&face_mesh_ret[face_id]);
            ttime[5] = pmeter_get_time_ms ();
            invoke_ms1 += ttime[5] - ttime[4];
        }

        /* --------------------------------------- *
         *  render scene (left half)
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* visualize the face pose estimation results. */
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);

        for (int face_id = 0; face_id < face_detect_ret.num; face_id ++)
        {
            render_face_landmark (draw_x, draw_y, draw_w, draw_h, &face_mesh_ret[face_id], &face_detect_ret.faces[face_id],
                                  cur_texid_mask, cur_face_mesh_mask, &cur_face_detect_mask->faces[0], 0);
        }

        if (s_gui_prop.draw_detect_rect)
        {
            render_detect_region (draw_x, draw_y, draw_w, draw_h, &face_detect_ret);
            
            /* draw cropped image of the face area */
            for (int face_id = 0; face_id < face_detect_ret.num; face_id ++)
            {
                float w = 100;
                float h = 100;
                float x = win_w - w - 10;
                float y = h * face_id + 10;
                float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

                render_cropped_face_image (&captex, x, y, w, h, &face_detect_ret, face_id);
                draw_2d_rect (x, y, w, h, col_white, 2.0f);
            }
        }

        /* --------------------------------------- *
         *  render scene  (right half)
         * --------------------------------------- */
        glViewport (win_w, 0, win_w, win_h);

        render_3d_scene (draw_x, draw_y, draw_w, draw_h);

        for (int face_id = 0; face_id < face_detect_ret.num; face_id ++)
        {
            render_face_landmark (draw_x, draw_y, draw_w, draw_h,
                                  &face_mesh_ret[face_id], &face_detect_ret.faces[face_id],
                                  cur_texid_mask, cur_face_mesh_mask, &cur_face_detect_mask->faces[0],
                                  s_gui_prop.draw_mesh_line);
        }

        /* current mask image */
        {
            float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
            int size = 200;

            draw_2d_texture (cur_texid_mask, 10, 10, size, size, 0);
            draw_2d_rect (10, 10, size, size, col_white, 2.0f);
        }


        /* --------------------------------------- *
         *  post process
         * --------------------------------------- */
        glViewport (0, 0, win_w, win_h);

        if (s_gui_prop.draw_pmeter)
        {
            draw_pmeter (0, 40);
        }

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite0 :%5.1f [ms]\nTFLite1 :%5.1f [ms]",
            interval, invoke_ms0, invoke_ms1);
        draw_dbgstr (strbuf, 10, 10);

#if defined (USE_IMGUI)
        invoke_imgui (&s_gui_prop);
#endif
        egl_swap();
    }

    return 0;
}

