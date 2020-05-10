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
#include "tflite_handpose.h"
#include "camera_capture.h"
#include "render_handpose.h"

#define UNUSED(x) (void)(x)

#if defined (USE_INPUT_CAMERA_CAPTURE)
static void
update_capture_texture (int texid)
{
    int   cap_w, cap_h;
    void *cap_buf;

    get_capture_dimension (&cap_w, &cap_h);
    get_capture_buffer (&cap_buf);

    if (cap_buf)
    {
        glBindTexture (GL_TEXTURE_2D, texid);
        glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, cap_w, cap_h, GL_RGBA, GL_UNSIGNED_BYTE, cap_buf);
    }
}
#endif

/* resize image to DNN network input size and convert to fp32. */
void
feed_palm_detection_image(int texid, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_palm_detection_input_buf (&w, &h);
    unsigned char *buf_ui8, *pui8;

    pui8 = buf_ui8 = (unsigned char *)malloc(w * h * 4);

    draw_2d_texture (texid, 0, win_h - h, w, h, 1);

    glPixelStorei (GL_PACK_ALIGNMENT, 1);
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

    free (pui8);
    return;
}


void
feed_hand_landmark_image(int texid, int win_w, int win_h, palm_detection_result_t *detection, unsigned int hand_id)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_hand_landmark_input_buf (&w, &h);
    unsigned char *buf_ui8, *pui8;

    pui8 = buf_ui8 = (unsigned char *)malloc(w * h * 4);

    float texcoord[] = { 0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 0.0f };

    if (detection->num > hand_id)
    {
        palm_t *palm = &(detection->palms[hand_id]);
        float x0 = palm->hand_pos[0].x;
        float y0 = palm->hand_pos[0].y;
        float x1 = palm->hand_pos[1].x; //    0--------1
        float y1 = palm->hand_pos[1].y; //    |        |
        float x2 = palm->hand_pos[2].x; //    |        |
        float y2 = palm->hand_pos[2].y; //    3--------2
        float x3 = palm->hand_pos[3].x;
        float y3 = palm->hand_pos[3].y;
        texcoord[0] = x3;   texcoord[1] = y3;
        texcoord[2] = x0;   texcoord[3] = y0;
        texcoord[4] = x2;   texcoord[5] = y2;
        texcoord[6] = x1;   texcoord[7] = y1;
    }
    
    draw_2d_texture_texcoord (texid, 0, win_h - h, w, h, texcoord);

    glPixelStorei (GL_PACK_ALIGNMENT, 1);
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

    free (pui8);
    return;
}

static void
render_palm_region (int ofstx, int ofsty, int texw, int texh, palm_detection_result_t *detection)
{
    float col_blue[]  = {0.0f, 0.0f, 1.0f, 1.0f};
    float col_red[]   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

    for (int i = 0; i < detection->num; i ++)
    {
        palm_t *palm = &(detection->palms[i]);
        float x1 = palm->rect.topleft.x  * texw + ofstx;
        float y1 = palm->rect.topleft.y  * texh + ofsty;
        float x2 = palm->rect.btmright.x * texw + ofstx;
        float y2 = palm->rect.btmright.y * texh + ofsty;
        float score = palm->score;

        /* rectangle region */
        draw_2d_rect (x1, y1, x2-x1, y2-y1, col_blue, 2.0f);

        /* class name */
        char buf[512];
        sprintf (buf, "%d", (int)(score * 100));
        draw_dbgstr_ex (buf, x1, y1, 1.0f, col_white, col_blue);

        /* key points */
        for (int j = 0; j < 7; j ++)
        {
            float x = palm->keys[j].x * texw + ofstx;
            float y = palm->keys[j].y * texh + ofsty;

            int r = 4;
            draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_blue);
        }

        for (int j0 = 0; j0 < 4; j0 ++)
        {
            int j1 = (j0 + 1) % 4;
            float x1 = palm->hand_pos[j0].x * texw + ofstx;
            float y1 = palm->hand_pos[j0].y * texh + ofsty;
            float x2 = palm->hand_pos[j1].x * texw + ofstx;
            float y2 = palm->hand_pos[j1].y * texh + ofsty;

            draw_2d_line (x1, y1, x2, y2, col_red, 2.0f);
        }
    }
}

static void
render_hand_landmark2d (int ofstx, int ofsty, int texw, int texh, hand_landmark_result_t *hand_landmark)
{
    float col_red[]   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

    float score = hand_landmark->score;
    char buf[512];
    sprintf (buf, "score:%4.1f", score * 100);
    draw_dbgstr_ex (buf, texw - 120, 0, 1.0f, col_white, col_red);

    for (int i = 0; i < HAND_JOINT_NUM; i ++)
    {
        float x = hand_landmark->joint[i].x  * texw + ofstx;
        float y = hand_landmark->joint[i].y  * texh + ofsty;

        int r = 4;
        draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_red);
    }
}

static void
compute_3d_hand_pos (hand_landmark_result_t *dst_hand, int texw, int texh, hand_landmark_result_t *src_hand, palm_t *palm)
{
    float xoffset   = palm->hand_cx - 0.5f;
    float yoffset   = palm->hand_cy - 0.5f;
    float zoffset   = 1000 * (1.0f / palm->hand_w);

    xoffset *= (1.0f / palm->hand_w);

    //fprintf (stderr, "------------------------------\n");
    for (int i = 0; i < HAND_JOINT_NUM; i ++)
    {
        float x = src_hand->joint[i].x;
        float y = src_hand->joint[i].y;
        float z = src_hand->joint[i].z;

        x = (x + xoffset) - 0.5f;
        y = (y + yoffset) - 0.5f;
        z = (z + zoffset);
        y = -y;
        z = -z;
        x = x * texw;
        y = y * texh;

        dst_hand->joint[i].x = x;
        dst_hand->joint[i].y = y;
        dst_hand->joint[i].z = z;
        //fprintf (stderr, "(%8.1f, %8.1f, %8.1f)\n", x, y, z);
    }
}

static void
render_node (float *mtxGlobal, hand_landmark_result_t *hand_landmark, int idx0, int idx1, float *color)
{
    float *pos0 = (float *)&hand_landmark->joint[idx0];
    float *pos1 = (float *)&hand_landmark->joint[idx1];
    draw_bone (mtxGlobal, pos0, pos1, 5.0f, color);
}

static void
render_palm_tri (float *mtxGlobal, hand_landmark_result_t *hand_landmark, int idx0, int idx1, int idx2, float *color)
{
    float *pos0 = (float *)&hand_landmark->joint[idx0];
    float *pos1 = (float *)&hand_landmark->joint[idx1];
    float *pos2 = (float *)&hand_landmark->joint[idx2];

    draw_triangle (mtxGlobal, pos0, pos1, pos2, color);
}

static void
shadow_matrix (float *m, float *light_dir, float *ground_pos, float *ground_nrm)
{
    vec3_normalize (light_dir);
    vec3_normalize (ground_nrm);

    float a = ground_nrm[0];
    float b = ground_nrm[1];
    float c = ground_nrm[2];
    float d = 0;
    float ex = light_dir[0];
    float ey = light_dir[1];
    float ez = light_dir[2];

    m[ 0] =  b * ey + c * ez;
    m[ 1] = -a * ey;
    m[ 2] = -a * ez;
    m[ 3] = 0;

    m[ 4] = -b * ex;
    m[ 5] =  a * ex + c * ez;
    m[ 6] = -b * ez;
    m[ 7] = 0;

    m[ 8] = -c * ex;
    m[ 9] = -c * ey;
    m[10] =  a * ex + b * ey;
    m[11] = 0;

    m[12] = -d * ex;
    m[13] = -d * ey;
    m[14] = -d * ez;
    m[15] =  a * ex + b * ey + c * ey;
}

static void
render_hand_landmark3d (int ofstx, int ofsty, int texw, int texh, 
                        hand_landmark_result_t *hand_landmark, palm_t *palm)
{
    float mtxGlobal[16];
    float rotation = -RAD_TO_DEG (palm->rotation);  /* z rotation (from detection result) */
    float col_red []   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_yellow[] = {1.0f, 1.0f, 0.0f, 1.0f};
    float col_green [] = {0.0f, 1.0f, 0.0f, 1.0f};
    float col_cyan  [] = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_violet[] = {1.0f, 0.0f, 1.0f, 1.0f};
    float col_palm[]   = {0.8f, 0.8f, 0.8f, 0.8f};
    float col_gray[]   = {0.0f, 0.0f, 0.0f, 0.5f};
    float col_node[]   = {1.0f, 1.0f, 1.0f, 1.0f};

    /* transform to 3D coordinate */
    hand_landmark_result_t hand_draw;
    compute_3d_hand_pos (&hand_draw, texw, texh, hand_landmark, palm);


    for (int is_shadow = 0; is_shadow < 2; is_shadow ++)
    {
        float *colj;
        float *coln = col_node;
        float *colp = col_palm;

        matrix_identity (mtxGlobal);
        matrix_rotate   (mtxGlobal, rotation, 0.0f, 0.0f, 1.0f);

        if (is_shadow)
        {
            float mtxShadow[16];
            float light_dir[3]  = {1.0f, 2.0f, 1.0f};
            float ground_pos[3] = {0.0f, 0.0f, 0.0f};
            float ground_nrm[3] = {0.0f, 1.0f, 0.0f};

            matrix_translate (mtxGlobal, 0.0f, -100.0f, 0.0f);

            shadow_matrix (mtxShadow, light_dir, ground_pos, ground_nrm);
            matrix_translate (mtxShadow, -hand_draw.joint[0].x, -hand_draw.joint[0].y, -hand_draw.joint[0].z);
            mtxShadow[12] += hand_draw.joint[0].x;
            mtxShadow[13] += hand_draw.joint[0].y;
            mtxShadow[14] += hand_draw.joint[0].z;

            matrix_mult (mtxGlobal, mtxGlobal, mtxShadow);

            colj = col_gray;
            coln = col_gray;
            colp = col_gray;
        }

        /* joint point */
        for (int i = 0; i < HAND_JOINT_NUM; i ++)
        {
            float vec[3] = {hand_draw.joint[i].x, hand_draw.joint[i].y, hand_draw.joint[i].z};

            if (!is_shadow)
            {
                if      (i >= 17) colj = col_violet;
                else if (i >= 13) colj = col_cyan;
                else if (i >=  9) colj = col_green;
                else if (i >=  5) colj = col_yellow;
                else              colj = col_red;
            }

            draw_sphere (mtxGlobal, vec, 15.0f, colj);
        }

        /* joint node */
        render_node (mtxGlobal, &hand_draw, 0,  1, coln);
        render_node (mtxGlobal, &hand_draw, 0, 17, coln);

        render_node (mtxGlobal, &hand_draw,  1,  5, coln);
        render_node (mtxGlobal, &hand_draw,  5,  9, coln);
        render_node (mtxGlobal, &hand_draw,  9, 13, coln);
        render_node (mtxGlobal, &hand_draw, 13, 17, coln);

        for (int i = 0; i < 5; i ++)
        {
            int idx0 = 4 * i + 1;
            int idx1 = idx0 + 1;
            render_node (mtxGlobal, &hand_draw, idx0,  idx1  , coln);
            render_node (mtxGlobal, &hand_draw, idx0+1,idx1+1, coln);
            render_node (mtxGlobal, &hand_draw, idx0+2,idx1+2, coln);
        }

        /* palm region */
        if (!is_shadow)
        {
            render_palm_tri (mtxGlobal, &hand_draw, 0,  1,  5, colp);
            render_palm_tri (mtxGlobal, &hand_draw, 0,  5,  9, colp);
            render_palm_tri (mtxGlobal, &hand_draw, 0,  9, 13, colp);
            render_palm_tri (mtxGlobal, &hand_draw, 0, 13, 17, colp);
        }
    }
}

static void
render_3d_scene (int ofstx, int ofsty, int texw, int texh, 
                 hand_landmark_result_t  *landmark,
                 palm_detection_result_t *detection)
{
    float mtxGlobal[16];
    float floor_size_x = texw/2; //100.0f;
    float floor_size_y = texw/2; //100.0f;
    float floor_size_z = texw/2; //100.0f;

    /* background */
    matrix_identity (mtxGlobal);
    matrix_translate (mtxGlobal, 0, floor_size_y * 0.9f, 0);
    matrix_scale  (mtxGlobal, floor_size_x, floor_size_y, floor_size_z);
    draw_floor (mtxGlobal);

    for (int hand_id = 0; hand_id < detection->num; hand_id ++)
    {
        hand_landmark_result_t *hand_landmark = &landmark[hand_id];
        render_hand_landmark3d (ofstx, ofsty, texw, texh, hand_landmark, &detection->palms[hand_id]);
    }
}


static void
render_cropped_hand_image (int texid, int ofstx, int ofsty, int texw, int texh, palm_detection_result_t *detection, unsigned int hand_id)
{
    float texcoord[8];

    if (detection->num <= hand_id)
        return;

    palm_t *palm = &(detection->palms[hand_id]);
    float x0 = palm->hand_pos[0].x;
    float y0 = palm->hand_pos[0].y;
    float x1 = palm->hand_pos[1].x; //    0--------1
    float y1 = palm->hand_pos[1].y; //    |        |
    float x2 = palm->hand_pos[2].x; //    |        |
    float y2 = palm->hand_pos[2].y; //    3--------2
    float x3 = palm->hand_pos[3].x;
    float y3 = palm->hand_pos[3].y;
    texcoord[0] = x0;   texcoord[1] = y0;
    texcoord[2] = x3;   texcoord[3] = y3;
    texcoord[4] = x1;   texcoord[5] = y1;
    texcoord[6] = x2;   texcoord[7] = y2;

    draw_2d_texture_texcoord (texid, ofstx, ofsty, texw, texh, texcoord);
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
    char input_name_default[] = "pakutaso_vsign.jpg";
    char *input_name = input_name_default;
    int count;
    int win_w = 800;
    int win_h = 600;
    int texid;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    double ttime[10] = {0}, interval, invoke_ms0 = 0, invoke_ms1 = 0;
    int enable_palm_detect = 0;
    int enable_camera = 1;
    UNUSED (argc);
    UNUSED (*argv);

    {
        int c;
        const char *optstring = "mx";

        while ((c = getopt (argc, argv, optstring)) != -1) 
        {
            switch (c)
            {
            case 'm':
                enable_palm_detect = 1;
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

    egl_init_with_platform_window_surface (2, 8, 0, 0, win_w * 2, win_h);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);
    init_cube ((float)win_w / (float)win_h);

    init_tflite_hand_landmark ();

#if defined (USE_GL_DELEGATE) || defined (USE_GPU_DELEGATEV2)
    /* we need to recover framebuffer because GPU Delegate changes the context */
    glBindFramebuffer (GL_FRAMEBUFFER, 0);
    glViewport (0, 0, win_w, win_h);
#endif

#if defined (USE_INPUT_CAMERA_CAPTURE)
    /* initialize V4L2 capture function */
    if (enable_camera && init_capture () == 0)
    {
        /* allocate texture buffer for captured image */
        get_capture_dimension (&texw, &texh);
        texid = create_2d_texture (NULL, texw, texh);
        start_capture ();
    }
    else
#endif
    load_jpg_texture (input_name, &texid, &texw, &texh);
    adjust_texture (win_w, win_h, texw, texh, &draw_x, &draw_y, &draw_w, &draw_h);

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    for (count = 0; ; count ++)
    {
        palm_detection_result_t palm_ret = {0};
        hand_landmark_result_t  hand_ret[MAX_PALM_NUM] = {0};
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
            update_capture_texture (texid);
        }
#endif

        /* --------------------------------------- *
         *  palm detection
         * --------------------------------------- */
        if (enable_palm_detect)
        {
            feed_palm_detection_image (texid, win_w, win_h);

            ttime[2] = pmeter_get_time_ms ();
            invoke_palm_detection (&palm_ret, 0);
            ttime[3] = pmeter_get_time_ms ();
            invoke_ms0 = ttime[3] - ttime[2];
        }
        else
        {
            invoke_palm_detection (&palm_ret, 1);
        }

        /* --------------------------------------- *
         *  hand landmark
         * --------------------------------------- */
        invoke_ms1 = 0;
        for (int hand_id = 0; hand_id < palm_ret.num; hand_id ++)
        {
            feed_hand_landmark_image (texid, win_w, win_h, &palm_ret, hand_id);

            ttime[4] = pmeter_get_time_ms ();
            invoke_hand_landmark (&hand_ret[hand_id]);
            ttime[5] = pmeter_get_time_ms ();
            invoke_ms1 += ttime[5] - ttime[4];
        }

        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* visualize the hand pose estimation results. */
        draw_2d_texture (texid,  draw_x, draw_y, draw_w, draw_h, 0);
        render_palm_region (draw_x, draw_y, draw_w, draw_h, &palm_ret);
        //render_hand_landmark2d (draw_x, draw_y, draw_w, draw_h, &hand_ret[0]);

        /* draw cropped image of the hand area */
        for (int hand_id = 0; hand_id < palm_ret.num; hand_id ++)
        {
            float w = 100;
            float h = 100;
            float x = win_w - w - 10;
            float y = h * hand_id + 10;
            float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

            render_cropped_hand_image (texid, x, y, w, h, &palm_ret, hand_id);
            draw_2d_rect (x, y, w, h, col_white, 2.0f);
        }

        /* render 3D hand skelton */
        glViewport (win_w, 0, win_w, win_h);
        render_3d_scene (draw_x, draw_y, draw_w, draw_h, hand_ret, &palm_ret);


        glViewport (0, 0, win_w, win_h);
        draw_pmeter (0, 40);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite0 :%5.1f [ms]\nTFLite1 :%5.1f [ms]",
            interval, invoke_ms0, invoke_ms1);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

