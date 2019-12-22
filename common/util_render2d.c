/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include "assertgl.h"
#include "util_shader.h"
#include "util_matrix.h"
#include "util_render2d.h"

/* ------------------------------------------------------ *
 *  shader for FillColor
 * ------------------------------------------------------ */
static char vs_fill[] = "                             \n\
                                                      \n\
attribute    vec4    a_Vertex;                        \n\
uniform      mat4    u_PMVMatrix;                     \n\
void main (void)                                      \n\
{                                                     \n\
    gl_Position = u_PMVMatrix * a_Vertex;             \n\
}                                                     ";

static char fs_fill[] = "                             \n\
                                                      \n\
precision mediump float;                              \n\
uniform      vec4    u_Color;                         \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    gl_FragColor = u_Color;                           \n\
}                                                       ";

/* ------------------------------------------------------ *
 *  shader for Texture
 * ------------------------------------------------------ */
static char vs_tex[] = "                              \n\
attribute    vec4    a_Vertex;                        \n\
attribute    vec2    a_TexCoord;                      \n\
varying      vec2    v_TexCoord;                      \n\
uniform      mat4    u_PMVMatrix;                     \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    gl_Position = u_PMVMatrix * a_Vertex;             \n\
    v_TexCoord  = a_TexCoord;                         \n\
}                                                     \n";

static char fs_tex[] = "                              \n\
precision mediump float;                              \n\
varying     vec2      v_TexCoord;                     \n\
uniform     sampler2D u_sampler;                      \n\
uniform     vec4      u_Color;                        \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    gl_FragColor = texture2D (u_sampler, v_TexCoord); \n\
    gl_FragColor *= u_Color;                          \n\
}                                                     \n";

/* ------------------------------------------------------ *
 *  shader for External Texture
 * ------------------------------------------------------ */
static char fs_extex[] = "                            \n\
#extension GL_NV_EGL_stream_consumer_external: enable \n\
#extension GL_OES_EGL_image_external : enable         \n\
precision mediump float;                              \n\
varying     vec2     v_TexCoord;                      \n\
uniform samplerExternalOES u_sampler;                 \n\
uniform     vec4      u_Color;                        \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    gl_FragColor = texture2D (u_sampler, v_TexCoord); \n\
    gl_FragColor *= u_Color;                          \n\
}                                                     \n";

/* ------------------------------------------------------ *
 *  shader for MATLAB Jet colormap
 * ------------------------------------------------------ */
static char fs_cmap_jet[] ="                          \n\
precision mediump float;                              \n\
varying     vec2      v_TexCoord;                     \n\
uniform     sampler2D u_sampler;                      \n\
uniform     vec4      u_Color;                        \n\
                                                      \n\
float cmap_jet_red(float x) {                         \n\
    if (x < 0.7) {                                    \n\
        return 4.0 * x - 1.5;                         \n\
    } else {                                          \n\
        return -4.0 * x + 4.5;                        \n\
    }                                                 \n\
}                                                     \n\
                                                      \n\
float cmap_jet_green(float x) {                       \n\
    if (x < 0.5) {                                    \n\
        return 4.0 * x - 0.5;                         \n\
    } else {                                          \n\
        return -4.0 * x + 3.5;                        \n\
    }                                                 \n\
}                                                     \n\
                                                      \n\
float cmap_jet_blue(float x) {                        \n\
    if (x < 0.3) {                                    \n\
       return 4.0 * x + 0.5;                          \n\
    } else {                                          \n\
       return -4.0 * x + 2.5;                         \n\
    }                                                 \n\
}                                                     \n\
                                                      \n\
vec4 colormap_jet(float x) {                          \n\
    float r = clamp(cmap_jet_red(x),   0.0, 1.0);     \n\
    float g = clamp(cmap_jet_green(x), 0.0, 1.0);     \n\
    float b = clamp(cmap_jet_blue(x),  0.0, 1.0);     \n\
    return vec4(r, g, b, 1.0);                        \n\
}                                                     \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    vec4 src_col = texture2D (u_sampler, v_TexCoord); \n\
    gl_FragColor = colormap_jet (src_col.r);          \n\
    gl_FragColor *= u_Color;                          \n\
}                                                     \n";



#define SHADER_NUM 4
static char *s_shader[SHADER_NUM * 2] = 
{
    vs_fill,   fs_fill,
    vs_tex,    fs_tex,
    vs_tex,    fs_extex,
    vs_tex,    fs_cmap_jet,
};

static shader_obj_t s_sobj[SHADER_NUM];
static int s_loc_mtx[SHADER_NUM];
static int s_loc_color[SHADER_NUM];

static float varray[] =
{   0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0 };

static float tarray[] =
{   0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0 };

static float tarray2[] =
{   0.0, 1.0,
    0.0, 0.0,
    1.0, 1.0,
    1.0, 0.0 };


static float s_matprj[16];
static int
set_projection_matrix (int w, int h)
{
    float mat_proj[] =
    {
       0.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 0.0f,
      -1.0f, 1.0f, 0.0f, 1.0f};

    mat_proj[0] =  2.0f / (float)w;
    mat_proj[5] = -2.0f / (float)h;

    memcpy (s_matprj, mat_proj, 16*sizeof(float));

    GLASSERT ();
    return 0;
}



int
init_2d_renderer (int w, int h)
{
  int i;

    for (i = 0; i < SHADER_NUM; i ++)
    {
        if (generate_shader (&s_sobj[i], s_shader[2*i], s_shader[2*i + 1]) < 0)
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }

        s_loc_mtx[i]   = glGetUniformLocation(s_sobj[i].program, "u_PMVMatrix" );
        s_loc_color[i] = glGetUniformLocation(s_sobj[i].program, "u_Color" );
    }

    set_projection_matrix (w, h);

    return 0;
}

typedef struct _texparam
{
    int          textype;
    int          texid;
    int          x, y, w, h;
    int          upsidedown;
    float        color[4];
    float        rot;               /* degree */
    int          blendfunc_en;
    unsigned int blendfunc[4];      /* src_rgb, dst_rgb, src_alpha, dst_alpha */
} texparam_t;


static int
draw_2d_texture_in (texparam_t *tparam)
{
    int ttype = tparam->textype;  
    int texid = tparam->texid;
    float x   = tparam->x;
    float y   = tparam->y;
    float w   = tparam->w;
    float h   = tparam->h;
    float rot = tparam->rot;
    shader_obj_t *sobj = &s_sobj[ttype];
    float matrix[16];
    float *uv = tarray;

    glUseProgram (sobj->program);
    glUniform1i(sobj->loc_tex, 0);

    switch (ttype)
    {
    case 0:
        break;
    case 1:
        glBindTexture (GL_TEXTURE_2D, texid);
        uv = tparam->upsidedown ? tarray2 : tarray;
        break;
    case 2:
        glBindTexture (GL_TEXTURE_EXTERNAL_OES, texid);
        uv = tparam->upsidedown ? tarray : tarray2;
        break;
    default:
        break;
    }

    if (sobj->loc_uv >= 0)
    {
        glEnableVertexAttribArray (sobj->loc_uv);
        glVertexAttribPointer (sobj->loc_uv, 2, GL_FLOAT, GL_FALSE, 0, uv);
    }

    glEnable (GL_BLEND);

    if (tparam->blendfunc_en)
    {
        glBlendFuncSeparate (tparam->blendfunc[0], tparam->blendfunc[1],
                   tparam->blendfunc[2], tparam->blendfunc[3]);
    }
    else
    {
        glBlendFuncSeparate (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                   GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    }

    matrix_identity (matrix);
    matrix_translate (matrix, x, y, 0.0f);
    if (rot != 0)
    {
        matrix_translate (matrix, 0,  h * 0.5f, 0.0f);
        matrix_rotate (matrix, rot, 0.0f, 0.0f, 1.0f);
        matrix_translate (matrix, 0, -h * 0.5f, 0.0f);
    }
    matrix_scale (matrix, w, h, 1.0f);
    matrix_mult (matrix, s_matprj, matrix);

    glUniformMatrix4fv (s_loc_mtx[ttype], 1, GL_FALSE, matrix);
    glUniform4fv (s_loc_color[ttype], 1, tparam->color);

    if (sobj->loc_vtx >= 0)
    {
        glEnableVertexAttribArray (sobj->loc_vtx);
        glVertexAttribPointer (sobj->loc_vtx, 2, GL_FLOAT, GL_FALSE, 0, varray);
    }

    glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);

    glDisable (GL_BLEND);
    
    GLASSERT ();
    return 0;
    
}


int
draw_2d_texture (int texid, int x, int y, int w, int h, int upsidedown)
{
    texparam_t tparam = {0};
    tparam.x       = x;
    tparam.y       = y;
    tparam.w       = w;
    tparam.h       = h;
    tparam.texid   = texid;
    tparam.textype = 1;
    tparam.color[0]= 1.0f;
    tparam.color[1]= 1.0f;
    tparam.color[2]= 1.0f;
    tparam.color[3]= 1.0f;
    tparam.upsidedown = upsidedown;
    draw_2d_texture_in (&tparam);

    return 0;
}

int
draw_2d_texture_blendfunc (int texid, int x, int y, int w, int h,
                           int upsidedown, unsigned int *blendfunc)
{
    texparam_t tparam = {0};
    tparam.x       = x;
    tparam.y       = y;
    tparam.w       = w;
    tparam.h       = h;
    tparam.texid   = texid;
    tparam.textype = 1;
    tparam.color[0]= 1.0f;
    tparam.color[1]= 1.0f;
    tparam.color[2]= 1.0f;
    tparam.color[3]= 1.0f;
    tparam.upsidedown = upsidedown;
    tparam.blendfunc_en = 1;
    tparam.blendfunc[0] = blendfunc[0];
    tparam.blendfunc[1] = blendfunc[1];
    tparam.blendfunc[2] = blendfunc[2];
    tparam.blendfunc[3] = blendfunc[3];
    draw_2d_texture_in (&tparam);

    return 0;
}

int
draw_2d_texture_modulate (int texid, int x, int y, int w, int h,
                           int upsidedown, float *color, unsigned int *blendfunc)
{
    texparam_t tparam = {0};
    tparam.x       = x;
    tparam.y       = y;
    tparam.w       = w;
    tparam.h       = h;
    tparam.texid   = texid;
    tparam.textype = 1;
    tparam.color[0]= color[0];
    tparam.color[1]= color[1];
    tparam.color[2]= color[2];
    tparam.color[3]= color[3];
    tparam.upsidedown = upsidedown;
    tparam.blendfunc_en = 1;
    tparam.blendfunc[0] = blendfunc[0];
    tparam.blendfunc[1] = blendfunc[1];
    tparam.blendfunc[2] = blendfunc[2];
    tparam.blendfunc[3] = blendfunc[3];
    draw_2d_texture_in (&tparam);

    return 0;
}

int
draw_2d_colormap (int texid, int x, int y, int w, int h, float alpha, int upsidedown)
{
    texparam_t tparam = {0};
    tparam.x       = x;
    tparam.y       = y;
    tparam.w       = w;
    tparam.h       = h;
    tparam.texid   = texid;
    tparam.textype = 3;
    tparam.color[0]= 1.0f;
    tparam.color[1]= 1.0f;
    tparam.color[2]= 1.0f;
    tparam.color[3]= alpha;
    tparam.upsidedown = upsidedown;
    draw_2d_texture_in (&tparam);

    return 0;
}


int
draw_2d_fillrect (int x, int y, int w, int h, float *color)
{
    texparam_t tparam = {0};
    tparam.x       = x;
    tparam.y       = y;
    tparam.w       = w;
    tparam.h       = h;
    tparam.textype = 0;
    tparam.color[0]= color[0];
    tparam.color[1]= color[1];
    tparam.color[2]= color[2];
    tparam.color[3]= color[3];
    draw_2d_texture_in (&tparam);

    return 0;
}


int
draw_2d_rect (int x, int y, int w, int h, float *color, float line_width)
{
    int ttype = 0;
    shader_obj_t *sobj = &s_sobj[ttype];
    float matrix[16];
    
    glUseProgram (sobj->program);
    glUniform4fv (s_loc_color[ttype], 1, color);

    glEnable (GL_BLEND);
    glBlendFuncSeparate (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, 
               GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    matrix_identity (matrix);
    matrix_mult (matrix, s_matprj, matrix);
    glUniformMatrix4fv (s_loc_mtx[ttype], 1, GL_FALSE, matrix);

    glLineWidth (line_width);
    float x1 = x;
    float x2 = x + w;
    float y1 = y;
    float y2 = y + h;
    if (sobj->loc_vtx >= 0)
    {
        float vtx[] = {x1, y1,
                       x2, y1,
                       x2, y2,
                       x1, y2,
                       x1, y1};

        glEnableVertexAttribArray (sobj->loc_vtx);
        glVertexAttribPointer (sobj->loc_vtx, 2, GL_FLOAT, GL_FALSE, 0, vtx);
        glDrawArrays (GL_LINE_STRIP, 0, 5);
    }

    glDisable (GL_BLEND);

    GLASSERT ();
    return 0;
}


int
draw_2d_line (int x0, int y0, int x1, int y1, float *color, float line_width)
{
    float dx = x1 - x0;
    float dy = y1 - y0;
    float len = sqrtf (dx * dx + dy * dy);
    float theta = acosf (dx / len);

    if (dy < 0)
        theta = -theta;

    texparam_t tparam = {0};
    tparam.x       = x0;
    tparam.y       = y0 - 0.5f * line_width;
    tparam.w       = len;
    tparam.h       = line_width;
    tparam.rot     = RAD_TO_DEG (theta);
    tparam.textype = 0;
    tparam.color[0]= color[0];
    tparam.color[1]= color[1];
    tparam.color[2]= color[2];
    tparam.color[3]= color[3];
    draw_2d_texture_in (&tparam);

    GLASSERT ();
    return 0;
}

