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
#include "tflite_pose3d.h"
#include "util_camera_capture.h"
#include "util_video_decode.h"
#include "render_pose3d.h"
#include "touch_event.h"
#include "render_imgui.h"

#define UNUSED(x) (void)(x)

typedef struct letterbox_tex_region_t
{
    float width;      /* full rect width  with margin */
    float height;     /* full rect height with margin */
    float tex_x;      /* start position of valid texture */
    float tex_y;      /* start position of valid texture */
    float tex_w;      /* width  of valid texture */
    float tex_h;      /* height of valid texture */
} letterbox_tex_region_t;

static letterbox_tex_region_t s_srctex_region;
static imgui_data_t s_gui_prop = {0};




/* resize image to DNN network input size and convert to fp32. */
void
feed_pose3d_image(texture_2d_t *srctex, int win_w, int win_h)
{
    int dst_w, dst_h;
    float *buf_fp32 = (float *)get_pose3d_input_buf (&dst_w, &dst_h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    float dst_aspect = (float)dst_w / (float)dst_h;
    float tex_aspect = (float)srctex->width / (float)srctex->height;
    float scaled_w, scaled_h;
    float offset_x, offset_y;

    if (dst_aspect > tex_aspect)
    {
        float scale = (float)dst_h / (float)srctex->height;
        scaled_w = scale * srctex->width;
        scaled_h = scale * srctex->height;
        offset_x = (dst_w - scaled_w) * 0.5;
        offset_y = 0;
    }
    else
    {
        float scale = (float)dst_w / (float)srctex->width;
        scaled_w = scale * srctex->width;
        scaled_h = scale * srctex->height;
        offset_x = 0;
        offset_y = (dst_h - scaled_h) * 0.5;
    }

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(dst_w * dst_h * 4);

    buf_ui8 = pui8;

    /* draw valid texture area */
    float dx = offset_x;
    float dy = win_h - dst_h + offset_y;
    draw_2d_texture_ex (srctex, dx, dy, scaled_w, scaled_h, 1);

    /* read full rect with margin */
    glPixelStorei (GL_PACK_ALIGNMENT, 4);
    glReadPixels (0, 0, dst_w, dst_h, GL_RGBA, GL_UNSIGNED_BYTE, buf_ui8);

    /* convert UI8 [0, 255] ==> FP32 [0, 1] */
    float mean =   0.0f;
    float std  = 255.0f;
    for (int y = 0; y < dst_h; y ++)
    {
        for (int x = 0; x < dst_w; x ++)
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

    s_srctex_region.width  = dst_w;     /* full rect width  with margin */
    s_srctex_region.height = dst_h;     /* full rect height with margin */
    s_srctex_region.tex_x  = offset_x;  /* start position of valid texture */
    s_srctex_region.tex_y  = offset_y;  /* start position of valid texture */
    s_srctex_region.tex_w  = scaled_w;  /* width  of valid texture */
    s_srctex_region.tex_h  = scaled_h;  /* height of valid texture */

    return;
}


/* render a bone of skelton. */
void
render_2d_bone (int ofstx, int ofsty, int drw_w, int drw_h, 
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
render_2d_scene (int x, int y, int w, int h, posenet_result_t *pose_ret)
{
    float col_red[]    = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_yellow[] = {1.0f, 1.0f, 0.0f, 1.0f};
    float col_green [] = {0.0f, 1.0f, 0.0f, 1.0f};
    float col_cyan[]   = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_violet[] = {1.0f, 0.0f, 1.0f, 1.0f};
    float col_blue[]   = {0.0f, 0.5f, 1.0f, 1.0f};

    {
        float ratio_w = s_srctex_region.width  / s_srctex_region.tex_w;
        float ratio_h = s_srctex_region.height / s_srctex_region.tex_h;
        float scale = (float)w / s_srctex_region.tex_w;
        w *= ratio_w;
        h *= ratio_h;
        x -= s_srctex_region.tex_x * scale;
        y -= s_srctex_region.tex_y * scale;
    }

    for (int i = 0; i < pose_ret->num; i ++)
    {
        /* right arm */
        render_2d_bone (x, y, w, h, pose_ret, i,  1,  2, col_red);
        render_2d_bone (x, y, w, h, pose_ret, i,  2,  3, col_red);
        render_2d_bone (x, y, w, h, pose_ret, i,  3,  4, col_red);

        /* left arm */
        render_2d_bone (x, y, w, h, pose_ret, i,  1,  5, col_violet);
        render_2d_bone (x, y, w, h, pose_ret, i,  5,  6, col_violet);
        render_2d_bone (x, y, w, h, pose_ret, i,  6,  7, col_violet);

        /* right leg */
        render_2d_bone (x, y, w, h, pose_ret, i,  1,  8, col_green);
        render_2d_bone (x, y, w, h, pose_ret, i,  8,  9, col_green);
        render_2d_bone (x, y, w, h, pose_ret, i,  9, 10, col_green);

        /* left leg */
        render_2d_bone (x, y, w, h, pose_ret, i,  1, 11, col_cyan);
        render_2d_bone (x, y, w, h, pose_ret, i, 11, 12, col_cyan);
        render_2d_bone (x, y, w, h, pose_ret, i, 12, 13, col_cyan);

        /* neck */
        render_2d_bone (x, y, w, h, pose_ret, i,  1,  0, col_yellow);

        /* eye */
        render_2d_bone (x, y, w, h, pose_ret, i,  0, 14, col_blue);
        render_2d_bone (x, y, w, h, pose_ret, i, 14, 16, col_blue);
        render_2d_bone (x, y, w, h, pose_ret, i,  0, 15, col_blue);
        render_2d_bone (x, y, w, h, pose_ret, i, 15, 17, col_blue);

        /* draw key points */
        for (int j = 0; j < kPoseKeyNum -1; j ++)
        {
            float *colj;
            if      (j >= 14) colj = col_blue;
            else if (j >= 11) colj = col_cyan;
            else if (j >=  8) colj = col_green;
            else if (j >=  5) colj = col_violet;
            else if (j >=  2) colj = col_red;
            else              colj = col_yellow;

            float keyx = pose_ret->pose[i].key[j].x * w + x;
            float keyy = pose_ret->pose[i].key[j].y * h + y;
            float score= pose_ret->pose[i].key[j].score;

            int r = 9;
            colj[3] = score;
            draw_2d_fillrect (keyx - (r/2), keyy - (r/2), r, r, colj);
            colj[3] = 1.0;
        }
    }
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
    int key_id = (s_count /1)% kPoseKeyNum;
    s_count ++;

    conf_min =  FLT_MAX;
    conf_max = -FLT_MAX;
    for (y = 0; y < heatmap_h; y ++)
    {
        for (x = 0; x < heatmap_w; x ++)
        {
            float confidence = heatmap[(y * heatmap_w * kPoseKeyNum)+ (x * kPoseKeyNum) + key_id];
            if (confidence < conf_min) conf_min = confidence;
            if (confidence > conf_max) conf_max = confidence;
        }
    }

    for (y = 0; y < heatmap_h; y ++)
    {
        for (x = 0; x < heatmap_w; x ++)
        {
            float confidence = heatmap[(y * heatmap_w * kPoseKeyNum)+ (x * kPoseKeyNum) + key_id];
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
        char strKey[][32] = {"Nose", "Neck", 
                             "RShoulder", "RElbow", "RWrist",
                             "LShoulder", "LElbow", "LWrist",
                             "RHip", "RKnee", "RAnkle",
                             "LHip", "LKnee", "LAnkle",
                             "LEye", "REye", "LEar", "REar", "C"};
        draw_dbgstr (strKey[key_id], ofstx + 5, 5);
    }
}


static void
compute_3d_skelton_pos (posenet_result_t *dst_pose, posenet_result_t *src_pose)
{
    /*
     *  because key3d[kNeck] is always zero,
     *  we need to add offsets (key2d[kNeck]) to translate it to the global world. 
     */
    float neck_x = src_pose->pose[0].key[kNeck].x;
    float neck_y = src_pose->pose[0].key[kNeck].y;
    float xoffset = (neck_x - 0.5f);
    float yoffset = (neck_y - 0.5f);

    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float x = src_pose->pose[0].key3d[i].x;
        float y = src_pose->pose[0].key3d[i].y;
        float z = src_pose->pose[0].key3d[i].z;
        float s = src_pose->pose[0].key3d[i].score;

        x = (x + xoffset) * s_gui_prop.pose_scale_x * 2;
        y = (y + yoffset) * s_gui_prop.pose_scale_y * 2;
        z = z * s_gui_prop.pose_scale_z;
        y = -y;
        z = -z;

        dst_pose->pose[0].key3d[i].x = x;
        dst_pose->pose[0].key3d[i].y = y;
        dst_pose->pose[0].key3d[i].z = z;
        dst_pose->pose[0].key3d[i].score = s;
    }
}

static void
render_3d_bone (float *mtxGlobal, pose_t *pose, int idx0, int idx1,
                float *color, float rad, int is_shadow)
{
    float *pos0 = (float *)&(pose->key3d[idx0]);
    float *pos1 = (float *)&(pose->key3d[idx1]);

    /* if the confidence score is low, draw more transparently. */
    float s0 = pose->key3d[idx0].score;
    float s1 = pose->key3d[idx1].score;
    float a  = color[3];

    color[3] = ((s0 > 0.1f) && (s1 > 0.1f)) ? a : 0.1f;
    draw_bone (mtxGlobal, pos0, pos1, rad, color, is_shadow);
    color[3] = a;
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
render_hand_landmark3d (int ofstx, int ofsty, posenet_result_t *pose_ret)
{
    float mtxGlobal[16], mtxTouch[16];
    float col_red []   = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_yellow[] = {1.0f, 1.0f, 0.0f, 1.0f};
    float col_green [] = {0.0f, 1.0f, 0.0f, 1.0f};
    float col_cyan  [] = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_violet[] = {1.0f, 0.0f, 1.0f, 1.0f};
    float col_blue[]   = {0.0f, 0.5f, 1.0f, 1.0f};
    float col_gray[]   = {0.0f, 0.0f, 0.0f, 0.2f};
    float col_node[]   = {1.0f, 1.0f, 1.0f, 1.0f};

    get_touch_event_matrix (mtxTouch);

    /* transform to 3D coordinate */
    posenet_result_t pose_draw;
    compute_3d_skelton_pos (&pose_draw, pose_ret);

    pose_t *pose = &pose_draw.pose[0];
    for (int is_shadow = 1; is_shadow >= 0; is_shadow --)
    {
        float *colj;
        float *coln = col_node;

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
        }

        /* joint point */
        for (int i = 0; i < kPoseKeyNum - 1; i ++)
        {
            float keyx = pose->key3d[i].x;
            float keyy = pose->key3d[i].y;
            float keyz = pose->key3d[i].z;
            float score= pose->key3d[i].score;

            float vec[3] = {keyx, keyy, keyz};

            if (!is_shadow)
            {
                if      (i >= 14) colj = col_blue;
                else if (i >= 11) colj = col_cyan;
                else if (i >=  8) colj = col_green;
                else if (i >=  5) colj = col_violet;
                else if (i >=  2) colj = col_red;
                else              colj = col_yellow;
            }

            float rad = (i < 14) ? s_gui_prop.joint_radius : s_gui_prop.joint_radius / 3;
            float alp = colj[3];
            colj[3] = (score > 0.1f) ? alp : 0.1f;
            draw_sphere (mtxGlobal, vec, rad, colj, is_shadow);
            colj[3] = alp;
        }

        /* right arm */
        float rad = s_gui_prop.bone_radius;
        render_3d_bone (mtxGlobal, pose,  1,  2, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, pose,  2,  3, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, pose,  3,  4, coln, rad, is_shadow);

        /* left arm */
        render_3d_bone (mtxGlobal, pose,  1,  5, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, pose,  5,  6, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, pose,  6,  7, coln, rad, is_shadow);

        /* right leg */
        render_3d_bone (mtxGlobal, pose,  1,  8, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, pose,  8,  9, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, pose,  9, 10, coln, rad, is_shadow);

        /* left leg */
        render_3d_bone (mtxGlobal, pose,  1, 11, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, pose, 11, 12, coln, rad, is_shadow);
        render_3d_bone (mtxGlobal, pose, 12, 13, coln, rad, is_shadow);

        /* neck */
        render_3d_bone (mtxGlobal, pose,  1,  0, coln, rad, is_shadow);

        /* eye */
        //render_3d_bone (mtxGlobal, pose,  0, 14, coln, 1.0f, is_shadow);
        //render_3d_bone (mtxGlobal, pose, 14, 16, coln, 1.0f, is_shadow);
        //render_3d_bone (mtxGlobal, pose,  0, 15, coln, 1.0f, is_shadow);
        //render_3d_bone (mtxGlobal, pose, 15, 17, coln, 1.0f, is_shadow);
    }
}

static void
render_3d_scene (int ofstx, int ofsty, posenet_result_t *pose_ret)
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

    render_hand_landmark3d (ofstx, ofsty, pose_ret);

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
    s_gui_prop.camera_pos_z = 300.0f;
    s_gui_prop.joint_radius = 8.0f;
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
    char input_name_default[] = "pakutaso_person.jpg";
    char *input_name = input_name_default;
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

    egl_init_with_platform_window_surface (2, 8, 0, 0, win_w * 2, win_h);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);
    init_cube ((float)win_w / (float)win_h);

    init_tflite_pose3d (use_quantized_tflite, &s_gui_prop.pose3d_config);

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
        create_video_texture (&captex, input_name);
        texw = captex.width;
        texh = captex.height;
        enable_camera = 0;
    }
    else
#endif
#if defined (USE_INPUT_CAMERA_CAPTURE)
    /* initialize V4L2 capture function */
    if (enable_camera && init_capture (CAPTURE_SQUARED_CROP) == 0)
    {
        create_capture_texture (&captex);
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
        enable_camera = 0;
    }
    adjust_texture (win_w, win_h, texw, texh, &draw_x, &draw_y, &draw_w, &draw_h);

    glClearColor (0.f, 0.f, 0.f, 1.0f);

    /* --------------------------------------- *
     *  Render Loop
     * --------------------------------------- */
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
         *  Pose estimation
         * --------------------------------------- */
        feed_pose3d_image (&captex, win_w, win_h);

        ttime[2] = pmeter_get_time_ms ();
        invoke_pose3d (&pose_ret);
        ttime[3] = pmeter_get_time_ms ();
        invoke_ms = ttime[3] - ttime[2];

        /* --------------------------------------- *
         *  render scene (left half)
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* visualize the object detection results. */
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);
        render_2d_scene (draw_x, draw_y, draw_w, draw_h, &pose_ret);

#if 0
        render_posenet_heatmap (draw_x, draw_y, draw_w, draw_h, &pose_ret);
#endif

        /* --------------------------------------- *
         *  render scene  (right half)
         * --------------------------------------- */
        glViewport (win_w, 0, win_w, win_h);
        render_3d_scene (draw_x, draw_y, &pose_ret);


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

