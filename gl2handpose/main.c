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

/* resize image to DNN network input size and convert to fp32. */
void
feed_palm_detection_image(texture_2d_t *srctex, int win_w, int win_h)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_palm_detection_input_buf (&w, &h);
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
feed_hand_landmark_image(texture_2d_t *srctex, int win_w, int win_h, palm_detection_result_t *detection, unsigned int hand_id)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_hand_landmark_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

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
render_palm_region (int ofstx, int ofsty, int texw, int texh, palm_t *palm)
{
    float col_blue[]  = {0.0f, 0.0f, 1.0f, 1.0f};
    float col_red[]   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

    float x1 = palm->rect.topleft.x  * texw + ofstx;
    float y1 = palm->rect.topleft.y  * texh + ofsty;
    float x2 = palm->rect.btmright.x * texw + ofstx;
    float y2 = palm->rect.btmright.y * texh + ofsty;
    float score = palm->score;

    /* detect rectangle */
    draw_2d_rect (x1, y1, x2-x1, y2-y1, col_blue, 2.0f);

    /* detect score */
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

    /* ROI rectangle */
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

static void
render_2d_bone (int ofstx, int ofsty, int texw, int texh, hand_landmark_result_t *hand_landmark,
                int id0, int id1)
{
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float x0 = hand_landmark->joint[id0].x * texw + ofstx;
    float y0 = hand_landmark->joint[id0].y * texh + ofsty;
    float x1 = hand_landmark->joint[id1].x * texw + ofstx;
    float y1 = hand_landmark->joint[id1].y * texh + ofsty;

    draw_2d_line (x0, y0, x1, y1, col_white, 1.0f);
}

static void
compute_2d_skelton_pos (hand_landmark_result_t *dst_hand, hand_landmark_result_t *src_hand, palm_t *palm)
{
    float rotation = RAD_TO_DEG (palm->rotation);  /* z rotation (from detection result) */
    float ofset_x = palm->hand_cx;
    float ofset_y = palm->hand_cy;
    float scale_w = palm->hand_w;
    float scale_h = palm->hand_h;

    float mtx[16];
    matrix_identity (mtx);
    matrix_translate (mtx, ofset_x, ofset_y, 0.0f);
    matrix_rotate (mtx, rotation, 0.0f, 0.0f, 1.0f);
    matrix_scale (mtx, scale_w, scale_h, 1.0f);
    matrix_translate (mtx, -0.5f, -0.5f, 0.0f);

    /* multiply rotate matrix */
    for (int i = 0; i < HAND_JOINT_NUM; i ++)
    {
        float x = src_hand->joint[i].x;
        float y = src_hand->joint[i].y;

        float vec[2] = {x, y};
        matrix_multvec2 (mtx, vec, vec);

        dst_hand->joint[i].x = vec[0];
        dst_hand->joint[i].y = vec[1];
    }
}

static void
render_skelton_2d (int ofstx, int ofsty, int texw, int texh, palm_t *palm,
                   hand_landmark_result_t *hand_landmark)
{
    float col_red[]   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_cyan[]  = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

    /* transform to global coordinate */
    hand_landmark_result_t hand_draw;
    compute_2d_skelton_pos (&hand_draw, hand_landmark, palm);

    /* score of keypoint */
    {
        float x = hand_draw.joint[0].x * texw + ofstx;
        float y = hand_draw.joint[0].y * texh + ofsty;
        float score = hand_landmark->score;
        char buf[512];
        sprintf (buf, "key:%d", (int)(score * 100));
        draw_dbgstr_ex (buf, x, y, 1.0f, col_white, col_red);
    }

    /* keypoints */
    for (int i = 0; i < HAND_JOINT_NUM; i ++)
    {
        float x = hand_draw.joint[i].x  * texw + ofstx;
        float y = hand_draw.joint[i].y  * texh + ofsty;

        int r = 4;
        draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_cyan);
    }

    /* skeltons */
    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw,  0,  1);
    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw,  0, 17);

    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw,  1,  5);
    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw,  5,  9);
    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw,  9, 13);
    render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw, 13, 17);

    for (int i = 0; i < 5; i ++)
    {
        int idx0 = 4 * i + 1;
        int idx1 = idx0 + 1;
        render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw, idx0,   idx1);
        render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw, idx0+1, idx1+1);
        render_2d_bone (ofstx, ofsty, texw, texh, &hand_draw, idx0+2, idx1+2);
    }
}


static void
compute_3d_skelton_pos (hand_landmark_result_t *dst_hand, hand_landmark_result_t *src_hand, palm_t *palm)
{
    float xoffset = palm->hand_cx - 0.5f;
    float yoffset = palm->hand_cy - 0.5f;
    float zoffset = (1.0f - palm->hand_w) / (palm->hand_w) * 0.1f; /* (1/w - 1) = (1-w)/w */

    xoffset *= (1.0f + zoffset);
    yoffset *= (1.0f + zoffset);

    float rotation = -RAD_TO_DEG (palm->rotation);  /* z rotation (from detection result) */

    float mtx[16];
    matrix_identity (mtx);
    matrix_rotate (mtx, rotation, 0.0f, 0.0f, 1.0f);

    //fprintf (stderr, "hand_w = %f, zoffset = %f\n", palm->hand_w, zoffset);
    for (int i = 0; i < HAND_JOINT_NUM; i ++)
    {
        float x = src_hand->joint[i].x;
        float y = src_hand->joint[i].y;
        float z = src_hand->joint[i].z;

        x = (x + xoffset) - 0.5f;
        y = (y + yoffset) - 0.5f;
        z = (z + zoffset);
        x = x * s_gui_prop.pose_scale_x * 2;
        y = y * s_gui_prop.pose_scale_y * 2;
        z = z * s_gui_prop.pose_scale_z * 5;
        y = -y;
        z = -z;

        /* multiply rotate matrix */
        {
            float vec[2] = {x, y};
            matrix_multvec2 (mtx, vec, vec);
            x = vec[0];
            y = vec[1];
        }

        dst_hand->joint[i].x = x;
        dst_hand->joint[i].y = y;
        dst_hand->joint[i].z = z;
    }
}

static void
render_3d_bone (float *mtxGlobal, hand_landmark_result_t *pose, int idx0, int idx1, 
                float *color, float rad, int is_shadow)
{
    float *pos0 = (float *)&(pose->joint[idx0]);
    float *pos1 = (float *)&(pose->joint[idx1]);

    draw_bone (mtxGlobal, pos0, pos1, rad, color, is_shadow);
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
render_skelton_3d (int ofstx, int ofsty, hand_landmark_result_t *hand_landmark, palm_t *palm)
{
    float mtxGlobal[16], mtxTouch[16];
    float col_red []   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_yellow[] = {1.0f, 1.0f, 0.0f, 1.0f};
    float col_green [] = {0.0f, 1.0f, 0.0f, 1.0f};
    float col_cyan  [] = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_violet[] = {1.0f, 0.0f, 1.0f, 1.0f};
    float col_palm[]   = {0.8f, 0.8f, 0.8f, 0.8f};
    float col_gray[]   = {0.0f, 0.0f, 0.0f, 0.5f};
    float col_node[]   = {1.0f, 1.0f, 1.0f, 1.0f};

    get_touch_event_matrix (mtxTouch);

    /* transform to 3D coordinate */
    hand_landmark_result_t hand_draw;
    compute_3d_skelton_pos (&hand_draw, hand_landmark, palm);


    for (int is_shadow = 1; is_shadow >= 0; is_shadow --)
    {
        float *colj;
        float *coln = col_node;
        float *colp = col_palm;

        matrix_identity (mtxGlobal);
        matrix_translate (mtxGlobal, 0.0, 0.0, -s_gui_prop.camera_pos_z);
        matrix_mult (mtxGlobal, mtxGlobal, mtxTouch);

        if (is_shadow)
        {
            float mtxShadow[16];
            float light_dir[3]  = {1.0f, 2.0f, 1.0f};
            float ground_pos[3] = {0.0f, 0.0f, 0.0f};
            float ground_nrm[3] = {0.0f, 1.0f, 0.0f};

            shadow_matrix (mtxShadow, light_dir, ground_pos, ground_nrm);

            float shadow_y = - s_gui_prop.pose_scale_y;
            //shadow_y += pose->key3d[kNeck].y * 0.5f;
            matrix_translate (mtxGlobal, 0.0, shadow_y, 0);
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

            float rad = s_gui_prop.joint_radius;
            draw_sphere (mtxGlobal, vec, rad, colj, is_shadow);
        }

        /* joint node */
        float rad = s_gui_prop.bone_radius;
        render_3d_bone (mtxGlobal, &hand_draw, 0,  1, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, &hand_draw, 0, 17, coln, rad, is_shadow);

        render_3d_bone (mtxGlobal, &hand_draw,  1,  5, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, &hand_draw,  5,  9, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, &hand_draw,  9, 13, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, &hand_draw, 13, 17, coln, rad, is_shadow);

        for (int i = 0; i < 5; i ++)
        {
            int idx0 = 4 * i + 1;
            int idx1 = idx0 + 1;
            render_3d_bone (mtxGlobal, &hand_draw, idx0,  idx1  , coln, rad, is_shadow);
            render_3d_bone (mtxGlobal, &hand_draw, idx0+1,idx1+1, coln, rad, is_shadow);
            render_3d_bone (mtxGlobal, &hand_draw, idx0+2,idx1+2, coln, rad, is_shadow);
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
render_3d_scene (int ofstx, int ofsty,
                 hand_landmark_result_t  *landmark,
                 palm_detection_result_t *detection)
{
    float mtxGlobal[16], mtxTouch[16];
    float floor_size_x = 300.0f;
    float floor_size_y = 300.0f;
    float floor_size_z = 300.0f;

    get_touch_event_matrix (mtxTouch);

    /* background */
    matrix_identity (mtxGlobal);
    matrix_translate (mtxGlobal, 0, 0, -s_gui_prop.camera_pos_z);
    matrix_mult (mtxGlobal, mtxGlobal, mtxTouch);
    matrix_translate (mtxGlobal, 0, -s_gui_prop.pose_scale_y, 0);
    matrix_scale  (mtxGlobal, floor_size_x, floor_size_y, floor_size_z);
    matrix_translate (mtxGlobal, 0, 1.0, 0);
    draw_floor (mtxGlobal, floor_size_x/10, floor_size_y/10);

    for (int hand_id = 0; hand_id < detection->num; hand_id ++)
    {
        hand_landmark_result_t *hand_landmark = &landmark[hand_id];
        render_skelton_3d (ofstx, ofsty, hand_landmark, &detection->palms[hand_id]);
    }

    if (s_gui_prop.draw_axis)
    {
        /* (xyz)-AXIS */
        matrix_identity (mtxGlobal);
        matrix_translate (mtxGlobal, 0, 0, -s_gui_prop.camera_pos_z);
        matrix_mult (mtxGlobal, mtxGlobal, mtxTouch);
        for (int i = -1; i <= 1; i ++)
        {
            for (int j = -1; j <= 1; j ++)
            {
                float col_base[] = {0.1, 0.5, 0.5, 0.5};
                float dx = s_gui_prop.pose_scale_x;
                float dy = s_gui_prop.pose_scale_y;
                float dz = s_gui_prop.pose_scale_z;
                float rad = 1;

                {
                    float v0[3] = {-dx, i * dy, j * dz};
                    float v1[3] = { dx, i * dy, j * dz};
                    float col_red[] = {1.0, 0.0, 0.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_red : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                    draw_sphere (mtxGlobal, v1, rad, col, 0);
                }
                {
                    float v0[3] = {i * dx, -dy, j * dz};
                    float v1[3] = {i * dx,  dy, j * dz};
                    float col_green[] = {0.0, 1.0, 0.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_green : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                    draw_sphere (mtxGlobal, v1, rad, col, 0);
                }
                {
                    float v0[3] = {i * dx, j * dy, -dz};
                    float v1[3] = {i * dx, j * dy,  dz};
                    float col_blue[] = {0.0, 0.0, 1.0, 1.0};
                    float *col = (i == 0 && j == 0) ? col_blue : col_base;
                    draw_line (mtxGlobal, v0, v1, col);
                    draw_sphere (mtxGlobal, v1, rad, col, 0);
                }
            }
        }
    }
}


static void
render_cropped_hand_image (texture_2d_t *srctex, int ofstx, int ofsty, int texw, int texh, palm_detection_result_t *detection, unsigned int hand_id)
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

    s_gui_prop.frame_color[0] = 0.0f;
    s_gui_prop.frame_color[1] = 0.5f;
    s_gui_prop.frame_color[2] = 1.0f;
    s_gui_prop.frame_color[3] = 1.0f;
    s_gui_prop.pose_scale_x = 100.0f;
    s_gui_prop.pose_scale_y = 100.0f;
    s_gui_prop.pose_scale_z = 100.0f;
    s_gui_prop.camera_pos_z = 200.0f;
    s_gui_prop.joint_radius = 6.0f;
    s_gui_prop.bone_radius  = 2.0f;
    s_gui_prop.draw_axis    = 0;
    s_gui_prop.draw_pmeter  = 1;
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
    int win_w = 900;
    int win_h = 900;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    texture_2d_t captex = {0};
    double ttime[10] = {0}, interval, invoke_ms0 = 0, invoke_ms1 = 0;
    int use_quantized_tflite = 0;
    int enable_palm_detect = 0;
    int enable_camera = 1;
    UNUSED (argc);
    UNUSED (*argv);

    {
        int c;
        const char *optstring = "mqx";

        while ((c = getopt (argc, argv, optstring)) != -1)
        {
            switch (c)
            {
            case 'm':
                enable_palm_detect = 1;
                break;
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

    egl_init_with_platform_window_surface (2, 8, 0, 0, win_w * 2, win_h);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);
    init_cube ((float)win_w / (float)win_h);

    init_tflite_hand_landmark (use_quantized_tflite);
    setup_imgui (win_w * 2, win_h);

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


    /* --------------------------------------- *
     *  Render Loop
     * --------------------------------------- */
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
            update_capture_texture (&captex);
        }
#endif

        /* --------------------------------------- *
         *  palm detection
         * --------------------------------------- */
        if (enable_palm_detect)
        {
            feed_palm_detection_image (&captex, win_w, win_h);

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
            feed_hand_landmark_image (&captex, win_w, win_h, &palm_ret, hand_id);

            ttime[4] = pmeter_get_time_ms ();
            invoke_hand_landmark (&hand_ret[hand_id]);
            ttime[5] = pmeter_get_time_ms ();
            invoke_ms1 += ttime[5] - ttime[4];
        }

        /* --------------------------------------- *
         *  render scene (left half)
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* visualize the hand pose estimation results. */
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);

        for (int hand_id = 0; hand_id < palm_ret.num; hand_id ++)
        {
            palm_t *palm = &(palm_ret.palms[hand_id]);
            render_palm_region (draw_x, draw_y, draw_w, draw_h, palm);
            render_skelton_2d (draw_x, draw_y, draw_w, draw_h, palm, &hand_ret[hand_id]);
        }

        /* draw cropped image of the hand area */
        for (int hand_id = 0; hand_id < palm_ret.num; hand_id ++)
        {
            float w = 100;
            float h = 100;
            float x = win_w - w - 10;
            float y = h * hand_id + 10;
            float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

            render_cropped_hand_image (&captex, x, y, w, h, &palm_ret, hand_id);
            draw_2d_rect (x, y, w, h, col_white, 2.0f);
        }

        /* --------------------------------------- *
         *  render scene  (right half)
         * --------------------------------------- */
        glViewport (win_w, 0, win_w, win_h);
        render_3d_scene (draw_x, draw_y, hand_ret, &palm_ret);


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

