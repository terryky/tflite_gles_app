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
#include "camera_capture.h"
#include "video_decode.h"

#define UNUSED(x) (void)(x)


#if defined (USE_INPUT_CAMERA_CAPTURE)
static void
update_capture_texture (texture_2d_t *captex)
{
    int   cap_w, cap_h;
    uint32_t cap_fmt;
    void *cap_buf;

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

    /* convert UI8 [0, 255] ==> FP32 [0, 1] */
    float mean = 0.0f;
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
feed_iris_landmark_image(texture_2d_t *srctex, int win_w, int win_h, 
                         face_t *face, face_landmark_result_t *facemesh, int eye_id)
{
    int x, y, w, h;
    float *buf_fp32 = (float *)get_irismesh_landmark_input_buf (&w, &h);
    unsigned char *buf_ui8 = NULL;
    static unsigned char *pui8 = NULL;

    if (pui8 == NULL)
        pui8 = (unsigned char *)malloc(w * h * 4);

    buf_ui8 = pui8;

    float texcoord[8];

    float scale_x = face->face_w;
    float scale_y = face->face_h;
    float pivot_x = face->face_cx;
    float pivot_y = face->face_cy;
    float rotation= face->rotation;
    
    float x0 = facemesh->eye_pos[eye_id][0].x;
    float y0 = facemesh->eye_pos[eye_id][0].y;
    float x1 = facemesh->eye_pos[eye_id][1].x; //    0--------1
    float y1 = facemesh->eye_pos[eye_id][1].y; //    |        |
    float x2 = facemesh->eye_pos[eye_id][2].x; //    |        |
    float y2 = facemesh->eye_pos[eye_id][2].y; //    3--------2
    float x3 = facemesh->eye_pos[eye_id][3].x;
    float y3 = facemesh->eye_pos[eye_id][3].y;

    float mat[16];
    float vec[4][2] = {{x0, y0}, {x1, y1}, {x2, y2}, {x3, y3}};
    matrix_identity (mat);
    
    matrix_translate (mat, pivot_x, pivot_y, 0);
    matrix_rotate (mat, RAD_TO_DEG(rotation), 0, 0, 1);
    matrix_scale (mat, scale_x, scale_y, 1.0f);
    matrix_translate (mat, -0.5f, -0.5f, 0);

    matrix_multvec2 (mat, vec[0], vec[0]);
    matrix_multvec2 (mat, vec[1], vec[1]);
    matrix_multvec2 (mat, vec[2], vec[2]);
    matrix_multvec2 (mat, vec[3], vec[3]);

    x0 = vec[0][0];  y0 = vec[0][1];
    x1 = vec[1][0];  y1 = vec[1][1];
    x2 = vec[2][0];  y2 = vec[2][1];
    x3 = vec[3][0];  y3 = vec[3][1];

    /* Upside down */
    if (eye_id == 0)
    {
        texcoord[0] = x3;   texcoord[1] = y3;
        texcoord[2] = x0;   texcoord[3] = y0;
        texcoord[4] = x2;   texcoord[5] = y2;
        texcoord[6] = x1;   texcoord[7] = y1;
    }
    else /* need to horizontal flip for right eye */
    {
        texcoord[0] = x2;   texcoord[1] = y2;
        texcoord[2] = x1;   texcoord[3] = y1;
        texcoord[4] = x3;   texcoord[5] = y3;
        texcoord[6] = x0;   texcoord[7] = y0;
    }

    draw_2d_texture_ex_texcoord (srctex, 0, win_h - h, w, h, texcoord);

    glPixelStorei (GL_PACK_ALIGNMENT, 4);
    glReadPixels (0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf_ui8);

    /* convert UI8 [0, 255] ==> FP32 [-1, 1] */
    float mean = 0.0f;
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
render_detect_region (int ofstx, int ofsty, int texw, int texh, face_detect_result_t *detection)
{
    float col_red[]   = {1.0f, 0.0f, 0.0f, 1.0f};

    for (int i = 0; i < detection->num; i ++)
    {
        face_t *face = &(detection->faces[i]);
        float x1 = face->topleft.x  * texw + ofstx;
        float y1 = face->topleft.y  * texh + ofsty;
        float x2 = face->btmright.x * texw + ofstx;
        float y2 = face->btmright.y * texh + ofsty;

        /* rectangle region */
        draw_2d_rect (x1, y1, x2-x1, y2-y1, col_red, 2.0f);

#if 0
        float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float score = face->score;

        /* class name */
        char buf[512];
        sprintf (buf, "%d", (int)(score * 100));
        draw_dbgstr_ex (buf, x1, y1, 1.0f, col_white, col_red);

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

static void
render_cropped_eye_image (texture_2d_t *srctex, int ofstx, int ofsty, int texw, int texh,
                           face_t *face, face_landmark_result_t *facemesh, int eye_id)
{
    float texcoord[8];

    float scale_x = face->face_w;
    float scale_y = face->face_h;
    float pivot_x = face->face_cx;
    float pivot_y = face->face_cy;
    float rotation= face->rotation;
    
    float x0 = facemesh->eye_pos[eye_id][0].x;
    float y0 = facemesh->eye_pos[eye_id][0].y;
    float x1 = facemesh->eye_pos[eye_id][1].x; //    0--------1
    float y1 = facemesh->eye_pos[eye_id][1].y; //    |        |
    float x2 = facemesh->eye_pos[eye_id][2].x; //    |        |
    float y2 = facemesh->eye_pos[eye_id][2].y; //    3--------2
    float x3 = facemesh->eye_pos[eye_id][3].x;
    float y3 = facemesh->eye_pos[eye_id][3].y;

    float mat[16];
    float vec[4][2] = {{x0, y0}, {x1, y1}, {x2, y2}, {x3, y3}};
    matrix_identity (mat);
    
    matrix_translate (mat, pivot_x, pivot_y, 0);
    matrix_rotate (mat, RAD_TO_DEG(rotation), 0, 0, 1);
    matrix_scale (mat, scale_x, scale_y, 1.0f);
    matrix_translate (mat, -0.5f, -0.5f, 0);

    matrix_multvec2 (mat, vec[0], vec[0]);
    matrix_multvec2 (mat, vec[1], vec[1]);
    matrix_multvec2 (mat, vec[2], vec[2]);
    matrix_multvec2 (mat, vec[3], vec[3]);

    x0 = vec[0][0];  y0 = vec[0][1];
    x1 = vec[1][0];  y1 = vec[1][1];
    x2 = vec[2][0];  y2 = vec[2][1];
    x3 = vec[3][0];  y3 = vec[3][1];

    texcoord[0] = x0;   texcoord[1] = y0;
    texcoord[2] = x3;   texcoord[3] = y3;
    texcoord[4] = x1;   texcoord[5] = y1;
    texcoord[6] = x2;   texcoord[7] = y2;

    draw_2d_texture_ex_texcoord (srctex, ofstx, ofsty, texw, texh, texcoord);
}


static void
render_lines (int ofstx, int ofsty, int texw, int texh, float *mat, irismesh_result_t *irismesh, int *idx, int num)
{
    float col_red  [] = {1.0f, 0.0f, 0.0f, 1.0f};
    fvec3 *eye  = irismesh->eye_landmark;

    for (int i = 1; i < num; i ++)
    {
        float vec0[] = {eye[idx[i-1]].x, eye[idx[i-1]].y};
        float vec1[] = {eye[idx[i  ]].x, eye[idx[i  ]].y};
        matrix_multvec2 (mat, vec0, vec0);
        matrix_multvec2 (mat, vec1, vec1);
        float x0 = vec0[0] * texw + ofstx;   float y0 = vec0[1] * texh + ofsty;
        float x1 = vec1[0] * texw + ofstx;   float y1 = vec1[1] * texh + ofsty;

        draw_2d_line (x0, y0, x1, y1, col_red, 4.0f);
    }
}

static void
render_iris_landmark (int ofstx, int ofsty, int texw, int texh, irismesh_result_t *irismesh)
{
    float col_red  [] = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_green[] = {0.0f, 1.0f, 0.0f, 1.0f};
    fvec3 *eye  = irismesh->eye_landmark;
    fvec3 *iris = irismesh->iris_landmark;
    float mat[16];

    matrix_identity (mat);

    for (int i = 0; i < 71; i ++)
    {
        float x = eye[i].x * texw + ofstx;;
        float y = eye[i].y * texh + ofsty;;

        int r = 4;
        draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_red);
    }

    int eye_idx0[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int idx_num0 = sizeof(eye_idx0) / sizeof(int);
    render_lines (ofstx, ofsty, texw, texh, mat, irismesh, eye_idx0, idx_num0);

    int eye_idx1[] = {0, 9, 10, 11, 12, 13, 14, 15, 8};
    int idx_num1 = sizeof(eye_idx1) / sizeof(int);
    render_lines (ofstx, ofsty, texw, texh, mat, irismesh, eye_idx1, idx_num1);

    for (int i = 0; i < 5; i ++)
    {
        float x = iris[i].x * texw + ofstx;;
        float y = iris[i].y * texh + ofsty;;

        int r = 4;
        draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_green);
    }

    {
        float x0 = iris[0].x * texw + ofstx; float y0 = iris[0].y * texh + ofsty;
        float x1 = iris[1].x * texw + ofstx; float y1 = iris[1].y * texh + ofsty;
        float len = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        draw_2d_circle (x0, y0, len, col_green, 4);
    }
}

static void
render_iris_landmark_on_face (int ofstx, int ofsty, int texw, int texh, 
                              face_landmark_result_t *facemesh, irismesh_result_t *irismesh)
{
    float col_green[] = {0.0f, 1.0f, 0.0f, 1.0f};

    for (int eye_id = 0; eye_id < 2; eye_id ++)
    {
        fvec3 *iris = irismesh[eye_id].iris_landmark;

        eye_region_t *eye_rgn = &facemesh->eye_rgn[eye_id];
        float scale_x = eye_rgn->size.x;
        float scale_y = eye_rgn->size.y;
        float pivot_x = eye_rgn->center.x;
        float pivot_y = eye_rgn->center.y;
        float rotation= eye_rgn->rotation;

        float mat[16];
        matrix_identity (mat);
        matrix_translate (mat, pivot_x, pivot_y, 0);
        matrix_rotate (mat, RAD_TO_DEG(rotation), 0, 0, 1);
        matrix_scale (mat, scale_x, scale_y, 1.0f);
        matrix_translate (mat, -0.5f, -0.5f, 0);

        if (0)
        {
            float col_red  [] = {1.0f, 0.0f, 0.0f, 1.0f};
            fvec3 *eye  = irismesh[eye_id].eye_landmark;
            for (int i = 0; i < 71; i ++)
            {
                float vec[2] = {eye[i].x, eye[i].y};
                matrix_multvec2 (mat, vec, vec);

                float x = vec[0] * texw + ofstx;;
                float y = vec[1] * texh + ofsty;;

                int r = 2;
                draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_red);
            }
        }

        /* iris circle */
        for (int i = 0; i < 5; i ++)
        {
            float vec[2] = {iris[i].x, iris[i].y};
            matrix_multvec2 (mat, vec, vec);

            float x = vec[0] * texw + ofstx;
            float y = vec[1] * texh + ofsty;

            int r = 4;
            draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_green);
        }

        /* eye region boundary box */
        {
            float x0 = facemesh->eye_pos[eye_id][0].x * texw + ofstx;
            float y0 = facemesh->eye_pos[eye_id][0].y * texh + ofsty;
            float x1 = facemesh->eye_pos[eye_id][1].x * texw + ofstx; //    0--------1
            float y1 = facemesh->eye_pos[eye_id][1].y * texh + ofsty; //    |        |
            float x2 = facemesh->eye_pos[eye_id][2].x * texw + ofstx; //    |        |
            float y2 = facemesh->eye_pos[eye_id][2].y * texh + ofsty; //    3--------2
            float x3 = facemesh->eye_pos[eye_id][3].x * texw + ofstx;
            float y3 = facemesh->eye_pos[eye_id][3].y * texh + ofsty;

            float col_red[] = {1.0f, 0.0f, 0.0f, 1.0f};
            draw_2d_line (x0, y0, x1, y1, col_red, 1.0f);
            draw_2d_line (x1, y1, x2, y2, col_red, 1.0f);
            draw_2d_line (x2, y2, x3, y3, col_red, 1.0f);
            draw_2d_line (x3, y3, x0, y0, col_red, 1.0f);
        }
    }
}

static void
render_facemesh_keypoint (int ofstx, int ofsty, int texw, int texh, float *mat, fvec3 *joint, int idx)
{
    float col_cyan[] = {0.0f, 1.0f, 1.0f, 1.0f};

    float vec[2] = {joint[idx].x, joint[idx].y};
    matrix_multvec2 (mat, vec, vec);

    float x = vec[0] * texw + ofstx;
    float y = vec[1] * texh + ofsty;

    int r = 4;
    draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_cyan);
}

static void
render_iris_landmark_on_main (int ofstx, int ofsty, int texw, int texh, 
                              face_t *face, face_landmark_result_t *facemesh, irismesh_result_t *irismesh)
{
    float col_green[] = {0.0f, 1.0f, 0.0f, 1.0f};

    float mat_face[16];
    {
        float scale_x = face->face_w;
        float scale_y = face->face_h;
        float pivot_x = face->face_cx;
        float pivot_y = face->face_cy;
        float rotation= face->rotation;

        matrix_identity (mat_face);
        matrix_translate (mat_face, pivot_x, pivot_y, 0);
        matrix_rotate (mat_face, RAD_TO_DEG(rotation), 0, 0, 1);
        matrix_scale (mat_face, scale_x, scale_y, 1.0f);
        matrix_translate (mat_face, -0.5f, -0.5f, 0);
    }

    int key_idx[] = {1, 9, 10, 152, 78, 308, 234, 454};
    int key_num = sizeof(key_idx) / sizeof(int);
    for (int i = 0; i < key_num; i ++)
        render_facemesh_keypoint (ofstx, ofsty, texw, texh, mat_face, facemesh->joint, key_idx[i]);

    for (int eye_id = 0; eye_id < 2; eye_id ++)
    {
        fvec3 *iris = irismesh[eye_id].iris_landmark;

        float mat_eye[16];
        {
            eye_region_t *eye_rgn = &facemesh->eye_rgn[eye_id];
            float scale_x = eye_rgn->size.x;
            float scale_y = eye_rgn->size.y;
            float pivot_x = eye_rgn->center.x;
            float pivot_y = eye_rgn->center.y;
            float rotation= eye_rgn->rotation;

            matrix_identity (mat_eye);
            matrix_translate (mat_eye, pivot_x, pivot_y, 0);
            matrix_rotate (mat_eye, RAD_TO_DEG(rotation), 0, 0, 1);
            matrix_scale (mat_eye, scale_x, scale_y, 1.0f);
            matrix_translate (mat_eye, -0.5f, -0.5f, 0);
        }

        float mat[16];
        matrix_mult (mat, mat_face, mat_eye);

        int eye_idx0[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        int idx_num0 = sizeof(eye_idx0) / sizeof(int);
        render_lines (ofstx, ofsty, texw, texh, mat, &irismesh[eye_id], eye_idx0, idx_num0);

        int eye_idx1[] = {0, 9, 10, 11, 12, 13, 14, 15, 8};
        int idx_num1 = sizeof(eye_idx1) / sizeof(int);
        render_lines (ofstx, ofsty, texw, texh, mat, &irismesh[eye_id], eye_idx1, idx_num1);

        if (0)
        {
            float col_red  [] = {1.0f, 0.0f, 0.0f, 1.0f};
            fvec3 *eye  = irismesh[eye_id].eye_landmark;
            for (int i = 0; i < 71; i ++)
            {
                float vec[2] = {eye[i].x, eye[i].y};
                matrix_multvec2 (mat, vec, vec);

                float x = vec[0] * texw + ofstx;;
                float y = vec[1] * texh + ofsty;;

                int r = 4;
                draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_red);
            }
        }

        /* iris circle */
        for (int i = 0; i < 5; i ++)
        {
            float vec[2] = {iris[i].x, iris[i].y};
            matrix_multvec2 (mat, vec, vec);

            float x = vec[0] * texw + ofstx;
            float y = vec[1] * texh + ofsty;

            int r = 4;
            draw_2d_fillrect (x - (r/2), y - (r/2), r, r, col_green);
        }

        {
            float vec0[2] = {iris[0].x, iris[0].y};
            float vec1[2] = {iris[1].x, iris[1].y};
            matrix_multvec2 (mat, vec0, vec0);
            matrix_multvec2 (mat, vec1, vec1);

            float x0 = vec0[0] * texw + ofstx;
            float y0 = vec0[1] * texh + ofsty;
            float x1 = vec1[0] * texw + ofstx;
            float y1 = vec1[1] * texh + ofsty;

            float len = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
            draw_2d_circle (x0, y0, len, col_green, 4);
        }
    }
}

static void
flip_horizontal_iris_landmark (irismesh_result_t *irismesh)
{
    fvec3 *eye  = irismesh->eye_landmark;
    fvec3 *iris = irismesh->iris_landmark;

    for (int i = 0; i < 71; i ++)
    {
        eye[i].x = 1.0f - eye[i].x;
    }

    for (int i = 0; i < 5; i ++)
    {
        iris[i].x = 1.0f - iris[i].x;
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
    char input_name_default[] = "pakutaso.jpg";
    char *input_name = NULL;
    int count;
    int win_w = 900;
    int win_h = 900;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
    texture_2d_t captex = {0};
    double ttime[10] = {0}, interval, invoke_ms0 = 0, invoke_ms1 = 0, invoke_ms2 = 0;
    int use_quantized_tflite = 0;
    int enable_video = 0;
    int enable_camera = 1;
    UNUSED (argc);
    UNUSED (*argv);

    {
        int c;
        const char *optstring = "eqv:x";

        while ((c = getopt (argc, argv, optstring)) != -1) 
        {
            switch (c)
            {
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

    if (input_name == NULL)
        input_name = input_name_default;

    egl_init_with_platform_window_surface (2, 0, 0, 0, win_w * 2, win_h);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

    init_tflite_facemesh (use_quantized_tflite);

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


    glClearColor (0.5f, 0.5f, 0.5f, 1.0f);
    glClear (GL_COLOR_BUFFER_BIT);
    glViewport (0, 0, win_w, win_h);


    /* --------------------------------------- *
     *  Render Loop
     * --------------------------------------- */
    for (count = 0; ; count ++)
    {
        face_detect_result_t    face_detect_ret = {0};
        face_landmark_result_t  face_mesh_ret[MAX_FACE_NUM] = {0};
        irismesh_result_t       iris_mesh_ret[MAX_FACE_NUM][2] = {0};
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
         *  Iris landmark
         * --------------------------------------- */
        invoke_ms2 = 0;
        for (int face_id = 0; face_id < face_detect_ret.num; face_id ++)
        {
            for (int eye_id = 0; eye_id < 2; eye_id ++)
            {
                feed_iris_landmark_image (&captex, win_w, win_h, &face_detect_ret.faces[face_id], &face_mesh_ret[face_id], eye_id);

                ttime[6] = pmeter_get_time_ms ();
                invoke_irismesh_landmark (&iris_mesh_ret[face_id][eye_id]);
                ttime[7] = pmeter_get_time_ms ();
                invoke_ms2 += ttime[7] - ttime[6];
            }
            /* need to horizontal flip for right eye */
            flip_horizontal_iris_landmark (&iris_mesh_ret[face_id][1]);
        }


        /* --------------------------------------- *
         *  render scene (left half)
         * --------------------------------------- */
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* visualize the face pose estimation results. */
        draw_2d_texture_ex (&captex, draw_x, draw_y, draw_w, draw_h, 0);
        render_detect_region (draw_x, draw_y, draw_w, draw_h, &face_detect_ret);

        for (int face_id = 0; face_id < face_detect_ret.num; face_id ++)
        {
            render_iris_landmark_on_main (draw_x, draw_y, draw_w, draw_h, &face_detect_ret.faces[face_id],
                                          &face_mesh_ret[face_id], iris_mesh_ret[face_id]);
        }

        /* --------------------------------------- *
         *  render scene  (right half)
         * --------------------------------------- */
        glViewport (win_w, 0, win_w, win_h);

        /* draw cropped image of the face area */
        for (int face_id = 0; face_id < face_detect_ret.num; face_id ++)
        {
            float w = 300;
            float h = 300;
            float x = 0;
            float y = h * face_id;
            float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

            render_cropped_face_image (&captex, x, y, w, h, &face_detect_ret, face_id);
            render_iris_landmark_on_face (x, y, w, h, &face_mesh_ret[face_id], iris_mesh_ret[face_id]);
            draw_2d_rect (x, y, w, h, col_white, 2.0f);
        }

        
        /* draw cropped image of the eye area */
        for (int face_id = 0; face_id < face_detect_ret.num; face_id ++)
        {
            float w = 300;
            float h = 300;
            float x = 300;
            float y = h * face_id;
            float col_white[] = {1.0f, 1.0f, 1.0f, 1.0f};

            render_cropped_eye_image (&captex, x, y, w, h, &face_detect_ret.faces[face_id], &face_mesh_ret[face_id], 0);
            render_iris_landmark (x, y, w, h, &iris_mesh_ret[face_id][0]);
            draw_2d_rect (x, y, w, h, col_white, 2.0f);

            x += w;
            render_cropped_eye_image (&captex, x, y, w, h, &face_detect_ret.faces[face_id], &face_mesh_ret[face_id], 1);
            render_iris_landmark (x, y, w, h, &iris_mesh_ret[face_id][1]);
            draw_2d_rect (x, y, w, h, col_white, 2.0f);
        }


        /* --------------------------------------- *
         *  post process
         * --------------------------------------- */
        glViewport (0, 0, win_w, win_h);
        draw_pmeter (0, 40);

        sprintf (strbuf, "Interval:%5.1f [ms]\nTFLite0 :%5.1f [ms]\nTFLite1 :%5.1f [ms]\nTFLite2 :%5.1f [ms]",
            interval, invoke_ms0, invoke_ms1, invoke_ms2);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

