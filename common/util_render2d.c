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
#include "util_texture.h"

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
#if USE_GLX
static char fs_extex[] = "                            \n\
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
#else
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
#endif

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


/* ------------------------------------------------------ *
 *  shader for YUYV Texture
 *      +--+--+--+--+
 *      | R| G| B| A|
 *      +--+--+--+--+
 *      |Y0| U|Y1| V|
 *      +--+--+--+--+
 * ------------------------------------------------------ */
static char vs_tex_yuyv[] = "                         \n\
attribute    vec4    a_Vertex;                        \n\
attribute    vec2    a_TexCoord;                      \n\
varying      vec2    v_TexCoord;                      \n\
varying      vec2    v_TexCoordPix;                   \n\
uniform      mat4    u_PMVMatrix;                     \n\
uniform      vec2    u_TexDim;                        \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    gl_Position   = u_PMVMatrix * a_Vertex;           \n\
    v_TexCoord    = a_TexCoord;                       \n\
    v_TexCoordPix = a_TexCoord * u_TexDim;            \n\
}                                                     \n";

static char fs_tex_yuyv[] = "                         \n\
precision mediump float;                              \n\
varying     vec2      v_TexCoord;                     \n\
varying     vec2      v_TexCoordPix;                  \n\
uniform     sampler2D u_sampler;                      \n\
uniform     vec4      u_Color;                        \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    vec2 evenodd = mod(v_TexCoordPix, 2.0);           \n\
    vec3 yuv, rgb;                                    \n\
    vec4 texcol = texture2D (u_sampler, v_TexCoord);  \n\
    if (evenodd.x < 1.0)                              \n\
    {                                                 \n\
        yuv.r = texcol.r;       /* Y */               \n\
        yuv.g = texcol.g - 0.5; /* U */               \n\
        yuv.b = texcol.a - 0.5; /* V */               \n\
    }                                                 \n\
    else                                              \n\
    {                                                 \n\
        yuv.r = texcol.b;       /* Y */               \n\
        yuv.g = texcol.g - 0.5; /* U */               \n\
        yuv.b = texcol.a - 0.5; /* V */               \n\
    }                                                 \n\
                                                      \n\
    rgb = mat3 (    1,        1,     1,               \n\
                    0, -0.34413, 1.772,               \n\
                1.402, -0.71414,     0) * yuv;        \n\
    gl_FragColor = vec4(rgb, 1.0);                    \n\
    gl_FragColor *= u_Color;                          \n\
}                                                     \n";


/* ------------------------------------------------------ *
 *  shader for YUYV Texture
 *      +--+--+--+--+
 *      | R| G| B| A|
 *      +--+--+--+--+
 *      | U|Y0| V|Y1|
 *      +--+--+--+--+
 * ------------------------------------------------------ */
static char vs_tex_uyvy[] = "                         \n\
attribute    vec4    a_Vertex;                        \n\
attribute    vec2    a_TexCoord;                      \n\
varying      vec2    v_TexCoord;                      \n\
varying      vec2    v_TexCoordPix;                   \n\
uniform      mat4    u_PMVMatrix;                     \n\
uniform      vec2    u_TexDim;                        \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    gl_Position   = u_PMVMatrix * a_Vertex;           \n\
    v_TexCoord    = a_TexCoord;                       \n\
    v_TexCoordPix = a_TexCoord * u_TexDim;            \n\
}                                                     \n";

static char fs_tex_uyvy[] = "                         \n\
precision mediump float;                              \n\
varying     vec2      v_TexCoord;                     \n\
varying     vec2      v_TexCoordPix;                  \n\
uniform     sampler2D u_sampler;                      \n\
uniform     vec4      u_Color;                        \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    vec2 evenodd = mod(v_TexCoordPix, 2.0);           \n\
    vec3 yuv, rgb;                                    \n\
    vec4 texcol = texture2D (u_sampler, v_TexCoord);  \n\
    if (evenodd.x < 1.0)                              \n\
    {                                                 \n\
        yuv.r = texcol.g;       /* Y */               \n\
        yuv.g = texcol.r - 0.5; /* U */               \n\
        yuv.b = texcol.b - 0.5; /* V */               \n\
    }                                                 \n\
    else                                              \n\
    {                                                 \n\
        yuv.r = texcol.a;       /* Y */               \n\
        yuv.g = texcol.r - 0.5; /* U */               \n\
        yuv.b = texcol.b - 0.5; /* V */               \n\
    }                                                 \n\
                                                      \n\
    rgb = mat3 (    1,        1,     1,               \n\
                    0, -0.34413, 1.772,               \n\
                1.402, -0.71414,     0) * yuv;        \n\
    gl_FragColor = vec4(rgb, 1.0);                    \n\
    gl_FragColor *= u_Color;                          \n\
}                                                     \n";

enum shader_type {
    SHADER_TYPE_FILL    = 0,    // 0
    SHADER_TYPE_TEX,            // 1
    SHADER_TYPE_EXTEX,          // 2
    SHADER_TYPE_CMAP_JET,       // 3
    SHADER_TYPE_TEX_YUYV,       // 4
    SHADER_TYPE_TEX_UYVY,       // 5

    SHADER_TYPE_MAX
};

#define SHADER_NUM SHADER_TYPE_MAX
static char *s_shader[SHADER_NUM * 2] = 
{
    vs_fill,   fs_fill,
    vs_tex,    fs_tex,
    vs_tex,    fs_extex,
    vs_tex,    fs_cmap_jet,
    vs_tex_yuyv, fs_tex_yuyv,
    vs_tex_uyvy, fs_tex_uyvy,
};

static shader_obj_t s_sobj[SHADER_NUM];
static int s_loc_mtx[SHADER_NUM];
static int s_loc_color[SHADER_NUM];
static int s_loc_texdim[SHADER_NUM];

static float varray[] =
{   0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0 };

static float s_matprj[16];
int
set_2d_projection_matrix (int w, int h)
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

        s_loc_mtx[i]    = glGetUniformLocation(s_sobj[i].program, "u_PMVMatrix");
        s_loc_color[i]  = glGetUniformLocation(s_sobj[i].program, "u_Color");
        s_loc_texdim[i] = glGetUniformLocation(s_sobj[i].program, "u_TexDim");
    }

    set_2d_projection_matrix (w, h);

    return 0;
}

typedef struct _texparam
{
    int          textype;
    int          texid;
    int          x, y, w, h;
    int          texw, texh;
    int          upsidedown;
    float        color[4];
    float        rot;               /* degree */
    float        px, py;            /* pivot */
    int          blendfunc_en;
    unsigned int blendfunc[4];      /* src_rgb, dst_rgb, src_alpha, dst_alpha */
    float        *user_texcoord;
} texparam_t;

static void
flip_texcoord (float *uv, unsigned int flip_mode)
{
    if (flip_mode & RENDER2D_FLIP_V)
    {
        uv[1] = 1.0f - uv[1];
        uv[3] = 1.0f - uv[3];
        uv[5] = 1.0f - uv[5];
        uv[7] = 1.0f - uv[7];
    }

    if (flip_mode & RENDER2D_FLIP_H)
    {
        uv[0] = 1.0f - uv[0];
        uv[2] = 1.0f - uv[2];
        uv[4] = 1.0f - uv[4];
        uv[6] = 1.0f - uv[6];
    }
}

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
    float tarray[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0 };
    float *uv = tarray;

    glBindBuffer (GL_ARRAY_BUFFER, 0);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

    glUseProgram (sobj->program);
    glActiveTexture (GL_TEXTURE0);
    glUniform1i(sobj->loc_tex, 0);

    switch (ttype)
    {
    case SHADER_TYPE_FILL:
        break;
    case SHADER_TYPE_TEX:
    case SHADER_TYPE_TEX_YUYV:
    case SHADER_TYPE_TEX_UYVY:
        glBindTexture (GL_TEXTURE_2D, texid);
        break;
    case SHADER_TYPE_EXTEX:
        glBindTexture (GL_TEXTURE_EXTERNAL_OES, texid);
        break;
    default:
        break;
    }

    flip_texcoord (uv, tparam->upsidedown);

    if (tparam->user_texcoord)
    {
        uv = tparam->user_texcoord;
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
        float px = tparam->px;
        float py = tparam->py;
        matrix_translate (matrix,  px,  py, 0.0f);
        matrix_rotate (matrix, rot, 0.0f, 0.0f, 1.0f);
        matrix_translate (matrix, -px, -py, 0.0f);
    }
    matrix_scale (matrix, w, h, 1.0f);
    matrix_mult (matrix, s_matprj, matrix);

    glUniformMatrix4fv (s_loc_mtx[ttype], 1, GL_FALSE, matrix);
    glUniform4fv (s_loc_color[ttype], 1, tparam->color);

    if (s_loc_texdim[ttype] >= 0)
    {
        float texdim[2];
        texdim[0] = tparam->texw;
        texdim[1] = tparam->texh;
        glUniform2fv (s_loc_texdim[ttype], 1, texdim);
    }

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
    tparam.textype = SHADER_TYPE_TEX;
    tparam.color[0]= 1.0f;
    tparam.color[1]= 1.0f;
    tparam.color[2]= 1.0f;
    tparam.color[3]= 1.0f;
    tparam.upsidedown = upsidedown;
    draw_2d_texture_in (&tparam);

    return 0;
}

int
draw_2d_texture_ex (texture_2d_t *tex, int x, int y, int w, int h, int upsidedown)
{
    texparam_t tparam = {0};
    tparam.x       = x;
    tparam.y       = y;
    tparam.w       = w;
    tparam.h       = h;
    tparam.texid   = tex->texid;
    tparam.textype = SHADER_TYPE_TEX;
    tparam.texw    = tex->width;
    tparam.texh    = tex->height;
    tparam.color[0]= 1.0f;
    tparam.color[1]= 1.0f;
    tparam.color[2]= 1.0f;
    tparam.color[3]= 1.0f;
    tparam.upsidedown = upsidedown;

    if (tex->format == pixfmt_fourcc('Y', 'U', 'Y', 'V'))
        tparam.textype = SHADER_TYPE_TEX_YUYV;
    else if (tex->format == pixfmt_fourcc('U', 'Y', 'V', 'Y'))
        tparam.textype = SHADER_TYPE_TEX_UYVY;
    else if (tex->format == pixfmt_fourcc('E', 'X', 'T', 'X'))
        tparam.textype = SHADER_TYPE_EXTEX;

    draw_2d_texture_in (&tparam);

    return 0;
}

int
draw_2d_texture_texcoord (int texid, int x, int y, int w, int h, float *user_texcoord)
{
    texparam_t tparam = {0};
    tparam.x       = x;
    tparam.y       = y;
    tparam.w       = w;
    tparam.h       = h;
    tparam.texid   = texid;
    tparam.textype = SHADER_TYPE_TEX;
    tparam.color[0]= 1.0f;
    tparam.color[1]= 1.0f;
    tparam.color[2]= 1.0f;
    tparam.color[3]= 1.0f;
    tparam.upsidedown = 0;
    tparam.user_texcoord = user_texcoord;
    draw_2d_texture_in (&tparam);

    return 0;
}

int
draw_2d_texture_ex_texcoord (texture_2d_t *tex, int x, int y, int w, int h, float *user_texcoord)
{
    texparam_t tparam = {0};
    tparam.x       = x;
    tparam.y       = y;
    tparam.w       = w;
    tparam.h       = h;
    tparam.texid   = tex->texid;
    tparam.textype = SHADER_TYPE_TEX;
    tparam.texw    = tex->width;
    tparam.texh    = tex->height;
    tparam.color[0]= 1.0f;
    tparam.color[1]= 1.0f;
    tparam.color[2]= 1.0f;
    tparam.color[3]= 1.0f;
    tparam.upsidedown = 0;
    tparam.user_texcoord = user_texcoord;

    if (tex->format == pixfmt_fourcc('Y', 'U', 'Y', 'V'))
        tparam.textype = SHADER_TYPE_TEX_YUYV;
    else if (tex->format == pixfmt_fourcc('U', 'Y', 'V', 'Y'))
        tparam.textype = SHADER_TYPE_TEX_UYVY;

    draw_2d_texture_in (&tparam);

    return 0;
}

int
draw_2d_texture_ex_texcoord_rot (texture_2d_t *tex, int x, int y, int w, int h, float *user_texcoord,
                                 float px, float py, float deg)
{
    texparam_t tparam = {0};
    tparam.x       = x;
    tparam.y       = y;
    tparam.w       = w;
    tparam.h       = h;
    tparam.texid   = tex->texid;
    tparam.textype = SHADER_TYPE_TEX;
    tparam.texw    = tex->width;
    tparam.texh    = tex->height;
    tparam.rot     = deg;
    tparam.px      = px * w;    /* relative pivot position (0 <= px <= 1) */
    tparam.py      = py * h;    /* relative pivot position (0 <= py <= 1) */
    tparam.color[0]= 1.0f;
    tparam.color[1]= 1.0f;
    tparam.color[2]= 1.0f;
    tparam.color[3]= 1.0f;
    tparam.upsidedown = 0;
    tparam.user_texcoord = user_texcoord;

    if (tex->format == pixfmt_fourcc('Y', 'U', 'Y', 'V'))
        tparam.textype = SHADER_TYPE_TEX_YUYV;
    else if (tex->format == pixfmt_fourcc('U', 'Y', 'V', 'Y'))
        tparam.textype = SHADER_TYPE_TEX_UYVY;

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
    tparam.textype = SHADER_TYPE_TEX;
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
    tparam.textype = SHADER_TYPE_TEX;
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
    tparam.textype = SHADER_TYPE_CMAP_JET;
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
    tparam.textype = SHADER_TYPE_FILL;
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
    int ttype = SHADER_TYPE_FILL;
    shader_obj_t *sobj = &s_sobj[ttype];
    float matrix[16];

    glBindBuffer (GL_ARRAY_BUFFER, 0);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

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
draw_2d_rect_rot (int x, int y, int w, int h, float *color, float line_width,
                  int px, int py, float rot_degree)
{
    int ttype = SHADER_TYPE_FILL;
    shader_obj_t *sobj = &s_sobj[ttype];
    float matrix[16];

    glBindBuffer (GL_ARRAY_BUFFER, 0);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

    glUseProgram (sobj->program);
    glUniform4fv (s_loc_color[ttype], 1, color);

    glEnable (GL_BLEND);
    glBlendFuncSeparate (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
               GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    matrix_identity (matrix);
    if (rot_degree != 0)
    {
        matrix_translate (matrix,  px,  py, 0);
        matrix_rotate (matrix, rot_degree, 0.0f, 0.0f, 1.0f);
        matrix_translate (matrix, -px, -py, 0);
    }

    matrix_mult (matrix, s_matprj, matrix);
    glUniformMatrix4fv (s_loc_mtx[ttype], 1, GL_FALSE, matrix);

    glLineWidth (line_width);
    float x1 = x;
    float x2 = x + w;
    float y1 = y;
    float y2 = y + h;
    if (sobj->loc_vtx >= 0)
    {
        float vtx[10] = {x1, y1,
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
    if (line_width == 1.0f)
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
        if (sobj->loc_vtx >= 0)
        {
            float vtx[] = {x0, y0, x1, y1};
            glEnableVertexAttribArray (sobj->loc_vtx);
            glVertexAttribPointer (sobj->loc_vtx, 2, GL_FLOAT, GL_FALSE, 0, vtx);
            glDrawArrays (GL_LINE_STRIP, 0, 2);
        }
        glDisable (GL_BLEND);
    }
    else
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
        tparam.px      = 0;
        tparam.py      = 0.5f * line_width;
        tparam.textype = SHADER_TYPE_FILL;
        tparam.color[0]= color[0];
        tparam.color[1]= color[1];
        tparam.color[2]= color[2];
        tparam.color[3]= color[3];
        draw_2d_texture_in (&tparam);
    }
    GLASSERT ();
    return 0;
}


#define CIRCLE_DIVNUM 15
int
draw_2d_fillcircle (int x, int y, int radius, float *color)
{
    int ttype = SHADER_TYPE_FILL;
    shader_obj_t *sobj = &s_sobj[ttype];
    float matrix[16];
    float vtx[(CIRCLE_DIVNUM+2) * 2];

    glBindBuffer (GL_ARRAY_BUFFER, 0);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

    glUseProgram (sobj->program);
    glUniform4fv (s_loc_color[ttype], 1, color);

    glEnable (GL_BLEND);
    glBlendFuncSeparate (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
               GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    matrix_identity (matrix);
    matrix_mult (matrix, s_matprj, matrix);
    glUniformMatrix4fv (s_loc_mtx[ttype], 1, GL_FALSE, matrix);

    if (sobj->loc_vtx >= 0)
    {
        vtx[0] = x;
        vtx[1] = y;
        for (int i = 0; i <= CIRCLE_DIVNUM; i ++)
        {
            float delta = 2 * M_PI / (float)CIRCLE_DIVNUM;
            float theta = delta * i;

            vtx[(i+1) * 2 + 0] = radius * cos (theta) + x;
            vtx[(i+1) * 2 + 1] = radius * sin (theta) + y;
        }

        glEnableVertexAttribArray (sobj->loc_vtx);
        glVertexAttribPointer (sobj->loc_vtx, 2, GL_FLOAT, GL_FALSE, 0, vtx);
        glDrawArrays (GL_TRIANGLE_FAN, 0, CIRCLE_DIVNUM + 2);
    }

    glDisable (GL_BLEND);

    GLASSERT ();
    return 0;
}

int
draw_2d_circle (int x, int y, int radius, float *color, float line_width)
{
    int ttype = SHADER_TYPE_FILL;
    shader_obj_t *sobj = &s_sobj[ttype];
    float matrix[16];
    float vtx[(CIRCLE_DIVNUM+2) * 2];

    glBindBuffer (GL_ARRAY_BUFFER, 0);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

    glUseProgram (sobj->program);
    glUniform4fv (s_loc_color[ttype], 1, color);

    glEnable (GL_BLEND);
    glBlendFuncSeparate (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
               GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    matrix_identity (matrix);
    matrix_mult (matrix, s_matprj, matrix);
    glUniformMatrix4fv (s_loc_mtx[ttype], 1, GL_FALSE, matrix);

    glLineWidth (line_width);
    if (sobj->loc_vtx >= 0)
    {
        for (int i = 0; i <= CIRCLE_DIVNUM; i ++)
        {
            float delta = 2 * M_PI / (float)CIRCLE_DIVNUM;
            float theta = delta * i;

            vtx[i * 2 + 0] = radius * cos (theta) + x;
            vtx[i * 2 + 1] = radius * sin (theta) + y;
        }

        glEnableVertexAttribArray (sobj->loc_vtx);
        glVertexAttribPointer (sobj->loc_vtx, 2, GL_FLOAT, GL_FALSE, 0, vtx);
        glDrawArrays (GL_LINE_STRIP, 0, CIRCLE_DIVNUM + 1);
    }
    glLineWidth (1);

    glDisable (GL_BLEND);

    GLASSERT ();
    return 0;
}

