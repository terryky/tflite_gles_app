/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <GLES2/gl2.h>
#include "util_egl.h"
#include "assertgl.h"
#include "util_shader.h"
#include "util_matrix.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_debug.h"
#include "util_v4l2.h"

#define UNUSED(x) (void)(x)

static pthread_t    s_capture_thread;
static void         *s_capture_buf = NULL;
static unsigned int s_capture_fmt = 0;
static int          s_capture_w = 0;
static int          s_capture_h = 0;

static GLuint       s_texid_dummy;
static int          s_tex_w = 0;
static int          s_tex_h = 0;

static shader_obj_t s_sobj;
static float        s_matPrj[16];
static GLint        s_loc_mtx_mv;
static GLint        s_loc_mtx_pmv;
static GLint        s_loc_mtx_nrm;
static GLint        s_loc_color;
static GLint        s_loc_alpha;



static GLfloat s_vtx[] =
{
    -1.0f, 1.0f,  1.0f,
    -1.0f,-1.0f,  1.0f,
     1.0f, 1.0f,  1.0f,
     1.0f,-1.0f,  1.0f,

     1.0f, 1.0f, -1.0f,
     1.0f,-1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f,
    -1.0f,-1.0f, -1.0f,

     1.0f,  1.0f, 1.0f,
     1.0f, -1.0f, 1.0f,
     1.0f,  1.0f,-1.0f,
     1.0f, -1.0f,-1.0f,

    -1.0f,  1.0f,-1.0f,
    -1.0f, -1.0f,-1.0f,
    -1.0f,  1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f,
    
     1.0f,  1.0f, 1.0f,
     1.0f,  1.0f,-1.0f,
    -1.0f,  1.0f, 1.0f,
    -1.0f,  1.0f,-1.0f,
    
    -1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f,-1.0f,
     1.0f, -1.0f, 1.0f,
     1.0f, -1.0f,-1.0f,
};

static GLfloat s_nrm[] =
{
     0.0f,  0.0f,  1.0f,
     0.0f,  0.0f, -1.0f,
     1.0f,  0.0f,  0.0f,
    -1.0f,  0.0f,  0.0f,
     0.0f,  1.0f,  0.0f,
     0.0f, -1.0f,  0.0f,
};

static GLfloat s_uv [] =
{
#if 0
     0.0f, 1.0f,
     0.0f, 0.0f,
     1.0f, 1.0f,
     1.0f, 0.0f,
#else
     0.0f, 0.0f,
     0.0f, 1.0f,
     1.0f, 0.0f,
     1.0f, 1.0f,
#endif
};


static char s_strVS[] = "                                   \n\
                                                            \n\
attribute vec4  a_Vertex;                                   \n\
attribute vec3  a_Normal;                                   \n\
attribute vec2  a_TexCoord;                                 \n\
uniform   mat4  u_PMVMatrix;                                \n\
uniform   mat4  u_MVMatrix;                                 \n\
uniform   mat3  u_ModelViewIT;                              \n\
varying   vec3  v_diffuse;                                  \n\
varying   vec3  v_specular;                                 \n\
varying   vec2  v_texcoord;                                 \n\
const     float shiness = 16.0;                             \n\
const     vec3  LightPos = vec3(4.0, 4.0, 4.0);             \n\
const     vec3  LightCol = vec3(0.5, 0.5, 0.5);             \n\
                                                            \n\
void DirectionalLight (vec3 normal, vec3 eyePos)            \n\
{                                                           \n\
    vec3  lightDir = normalize (LightPos);                  \n\
    vec3  halfV    = normalize (LightPos - eyePos);         \n\
    float dVP      = max(dot(normal, lightDir), 0.0);       \n\
    float dHV      = max(dot(normal, halfV   ), 0.0);       \n\
                                                            \n\
    float pf = 0.0;                                         \n\
    if(dVP > 0.0)                                           \n\
        pf = pow(dHV, shiness);                             \n\
                                                            \n\
    v_diffuse += dVP * LightCol;                            \n\
    v_specular+= pf  * LightCol;                            \n\
}                                                           \n\
                                                            \n\
void main(void)                                             \n\
{                                                           \n\
    gl_Position = u_PMVMatrix * a_Vertex;                   \n\
    vec3 normal = normalize(u_ModelViewIT * a_Normal);      \n\
    vec3 eyePos = vec3(u_MVMatrix * a_Vertex);              \n\
                                                            \n\
    v_diffuse  = vec3(0.0);                                 \n\
    v_specular = vec3(0.0);                                 \n\
    DirectionalLight(normal, eyePos);                       \n\
                                                            \n\
    v_texcoord  = a_TexCoord;                               \n\
}                                                           ";

static char s_strFS[] = "                                   \n\
precision mediump float;                                    \n\
                                                            \n\
uniform vec3    u_color;                                    \n\
uniform float   u_alpha;                                    \n\
varying vec3    v_diffuse;                                  \n\
varying vec3    v_specular;                                 \n\
varying vec2    v_texcoord;                                 \n\
uniform sampler2D u_sampler;                                \n\
                                                            \n\
void main(void)                                             \n\
{                                                           \n\
    vec3 color;                                             \n\
    color = vec3(texture2D(u_sampler,  v_texcoord));        \n\
    color += (u_color * v_diffuse);                         \n\
    color += v_specular;                                    \n\
    gl_FragColor = vec4(color, u_alpha);                    \n\
}                                                           ";



static int
draw_cube (int count)
{
    int i;
    float matMV[16], matPMV[16], matMVI4x4[16], matMVI3x3[9];

    glEnable (GL_DEPTH_TEST);
    //glEnable (GL_CULL_FACE);
    
    glUseProgram( s_sobj.program );

    glEnableVertexAttribArray (s_sobj.loc_vtx);
    glEnableVertexAttribArray (s_sobj.loc_uv );
    glDisableVertexAttribArray(s_sobj.loc_nrm);
    glVertexAttribPointer (s_sobj.loc_vtx, 3, GL_FLOAT, GL_FALSE, 0, s_vtx);
    glVertexAttribPointer (s_sobj.loc_uv , 2, GL_FLOAT, GL_FALSE, 0, s_uv );

    matrix_identity (matMV);
    matrix_translate (matMV, 0.0f, 0.0f, -3.5f);
    matrix_rotate (matMV, 30.0f * sinf (count*0.01f), 1.0f, 0.0f, 0.0f);
    matrix_rotate (matMV, count*1.0f, 0.0f, 1.0f, 0.0f);

    matrix_copy (matMVI4x4, matMV);
    matrix_invert   (matMVI4x4);
    matrix_transpose(matMVI4x4);
    matMVI3x3[0] = matMVI4x4[0];
    matMVI3x3[1] = matMVI4x4[1];
    matMVI3x3[2] = matMVI4x4[2];
    matMVI3x3[3] = matMVI4x4[4];
    matMVI3x3[4] = matMVI4x4[5];
    matMVI3x3[5] = matMVI4x4[6];
    matMVI3x3[6] = matMVI4x4[8];
    matMVI3x3[7] = matMVI4x4[9];
    matMVI3x3[8] = matMVI4x4[10];

    matrix_mult (matPMV, s_matPrj, matMV);
    glUniformMatrix4fv (s_loc_mtx_mv,   1, GL_FALSE, matMV );
    glUniformMatrix4fv (s_loc_mtx_pmv,  1, GL_FALSE, matPMV);
    glUniformMatrix3fv (s_loc_mtx_nrm,  1, GL_FALSE, matMVI3x3);
    glUniform3f (s_loc_color, 0.5f, 0.5f, 0.5f);
    glUniform1f (s_loc_alpha, 0.9f);

    glEnable (GL_BLEND);

    for (i = 0; i < 6; i ++)
    {
        glBindTexture (GL_TEXTURE_2D, s_texid_dummy);

        glVertexAttribPointer (s_sobj.loc_vtx, 3, GL_FLOAT, GL_FALSE, 0, &s_vtx[4 * 3 * i]);
        glVertexAttrib4fv (s_sobj.loc_nrm, &s_nrm[3 * i]);
        glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);
    }

    glDisable (GL_BLEND);

    return 0;
}


#define _max(A, B)    ((A) > (B) ? (A) : (B))
#define _min(A, B)    ((A) < (B) ? (A) : (B))

static int
update_capture_texture ()
{
    static unsigned char *s_buf = NULL;
    int x, y;
    
    if (s_tex_w != s_capture_w)
    {
        s_tex_w = s_capture_w;
        s_tex_h = s_capture_h;

        glBindTexture (GL_TEXTURE_2D, s_texid_dummy);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, s_tex_w, s_tex_h, 
                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        s_buf = (unsigned char *)malloc (s_tex_w * s_tex_h * 4);
    }

    unsigned char *src8 = s_capture_buf;
    unsigned char *dst8 = s_buf;
    for (y = 0; y < s_tex_h; y ++)
    {
        for (x = 0; x < s_tex_w; x += 2)
        {
            int y0 = *src8 ++;
            int cb = *src8 ++;
            int y1 = *src8 ++;
            int cr = *src8 ++;

            y0 -= 16;
            y1 -= 16;
            cb -= 128;
            cr -= 128;
            int r, g, b;
            
            r = 1164 * y0 + 1596 * cr;
            g = 1164 * y0 -  392 * cb - 813 * cr;
            b = 1164 * y0 + 2017 * cb;
            
            r = _min (_max (r, 999) / 1000, 255);
            g = _min (_max (g, 999) / 1000, 255);
            b = _min (_max (b, 999) / 1000, 255);
            
            *dst8 ++ = r;
            *dst8 ++ = g;
            *dst8 ++ = b;
            *dst8 ++ = 255;

            r = 1164 * y1 + 1596 * cr;
            g = 1164 * y1 -  392 * cb - 813 * cr;
            b = 1164 * y1 + 2017 * cb;
            
            r = _min (_max (r, 999) / 1000, 255);
            g = _min (_max (g, 999) / 1000, 255);
            b = _min (_max (b, 999) / 1000, 255);
            
            *dst8 ++ = r;
            *dst8 ++ = g;
            *dst8 ++ = b;
            *dst8 ++ = 255;
        }
    }

    if (s_buf)
    {
        glBindTexture (GL_TEXTURE_2D, s_texid_dummy);
        glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, s_tex_w, s_tex_h, GL_RGBA, GL_UNSIGNED_BYTE, s_buf);
    }

    return 0;
}



void *
capture_thread_main ()
{
    capture_dev_t *cap_dev;
    int cap_devid = -1;
    int cap_w, cap_h;
    unsigned int cap_fmt;

    cap_dev = v4l2_open_capture_device (cap_devid);
    DBG_ASSERT (cap_dev, "failed to open V4L\n");

    v4l2_get_capture_wh (cap_dev, &cap_w, &cap_h);
    v4l2_get_capture_pixelformat (cap_dev, &cap_fmt);

    v4l2_show_current_capture_settings (cap_dev);

    v4l2_start_capture (cap_dev);

    while (1)
    {
        capture_frame_t *frame = v4l2_acquire_capture_frame (cap_dev);

        s_capture_buf = frame->vaddr;
        s_capture_fmt = cap_fmt;
        s_capture_w = cap_w;
        s_capture_h = cap_h;

        v4l2_release_capture_frame (cap_dev, frame);
    }
}


int
init_app (int win_w, int win_h)
{
    generate_shader (&s_sobj, s_strVS, s_strFS);
    s_loc_mtx_mv  = glGetUniformLocation(s_sobj.program, "u_MVMatrix" );
    s_loc_mtx_pmv = glGetUniformLocation(s_sobj.program, "u_PMVMatrix" );
    s_loc_mtx_nrm = glGetUniformLocation(s_sobj.program, "u_ModelViewIT" );
    s_loc_color   = glGetUniformLocation(s_sobj.program, "u_color" );
    s_loc_alpha   = glGetUniformLocation(s_sobj.program, "u_alpha" );

    matrix_proj_perspective (s_matPrj, 72.0f, (float)win_w/(float)win_h, 1.f, 1000.f);

    glGenTextures (1, &s_texid_dummy);

    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

    glClearColor (0.7f, 0.7f, 0.7f, 1.0f);

    pthread_create (&s_capture_thread, NULL, capture_thread_main, NULL);

    GLASSERT ();
    return 0;
}


int main(int argc, char *argv[])
{
    int win_w = 960;
    int win_h = 540;
    int count;
    double ttime0 = 0, ttime1 = 0, interval;
    char strbuf[512];
    UNUSED (argc);
    UNUSED (argv);

    if (egl_init_with_platform_window_surface (2, 24, 0, 0, win_w, win_h) < 0)
        exit (-1);

    init_app (win_w, win_h);

    for (count = 0; ; count ++)
    {
        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime1 = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime1 - ttime0 : 0;
        ttime0 = ttime1;

        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        update_capture_texture ();

        draw_cube (count);
        draw_pmeter (0, 40);

        sprintf (strbuf, "%.1f [ms]\n", interval);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

