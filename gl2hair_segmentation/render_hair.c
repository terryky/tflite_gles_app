/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
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
uniform     sampler2D u_sampler2;                     \n\
uniform     vec4      u_Color;                        \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    vec4 weight = texture2D (u_sampler2, v_TexCoord); \n\
    vec4 color1 = texture2D (u_sampler,  v_TexCoord); \n\
    vec4 color2 = u_Color;                            \n\
                                                      \n\
    float luminance = dot(color1.rgb, vec3(0.299, 0.587, 0.114));   \n\
    float mix_value = weight.a * luminance;           \n\
                                                      \n\
    gl_FragColor = mix(color1, color2, mix_value);    \n\
    gl_FragColor.a = 1.0; \n\
}                                                     \n";


static shader_obj_t s_sobj;
static int s_loc_mtx;
static int s_loc_color;
static int s_loc_texdim;
static int s_loc_tex0;
static int s_loc_tex1;


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
init_hair_renderer (int w, int h)
{
    if (generate_shader (&s_sobj, vs_tex, fs_tex) < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    s_loc_mtx    = glGetUniformLocation(s_sobj.program, "u_PMVMatrix");
    s_loc_color  = glGetUniformLocation(s_sobj.program, "u_Color");
    s_loc_texdim = glGetUniformLocation(s_sobj.program, "u_TexDim");

    s_loc_tex0   = glGetUniformLocation(s_sobj.program, "u_sampler");
    s_loc_tex1   = glGetUniformLocation(s_sobj.program, "u_sampler2");

    set_projection_matrix (w, h);

    return 0;
}



typedef struct _texparam
{
    int          textype;
    int          texid;
    int          texid1;
    int          x, y, w, h;
    int          texw, texh;
    int          upsidedown;
    float        color[4];
    float        rot;               /* degree */
    int          blendfunc_en;
    unsigned int blendfunc[4];      /* src_rgb, dst_rgb, src_alpha, dst_alpha */
    float        *user_texcoord;
} texparam_t;


static int
draw_2d_texture_in (texparam_t *tparam)
{
    int ttype = tparam->textype;  
    int texid = tparam->texid;
    int texid1= tparam->texid1;
    float x   = tparam->x;
    float y   = tparam->y;
    float w   = tparam->w;
    float h   = tparam->h;
    float rot = tparam->rot;
    shader_obj_t *sobj = &s_sobj;
    float matrix[16];
    float *uv = tarray;

    glUseProgram (sobj->program);
    glUniform1i(s_loc_tex0, 0);
    glUniform1i(s_loc_tex1, 1);

    switch (ttype)
    {
    case 0:     /* fill     */
        break;
    case 1:     /* tex      */
    case 4:     /* tex_yuyv */
	    glActiveTexture(GL_TEXTURE0);
        glBindTexture (GL_TEXTURE_2D, texid);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture (GL_TEXTURE_2D, texid1);

        uv = tparam->upsidedown ? tarray2 : tarray;
        break;
    case 2:     /* tex_extex */
        glBindTexture (GL_TEXTURE_EXTERNAL_OES, texid);
        uv = tparam->upsidedown ? tarray : tarray2;
        break;
    default:
        break;
    }

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
        matrix_translate (matrix, 0,  h * 0.5f, 0.0f);
        matrix_rotate (matrix, rot, 0.0f, 0.0f, 1.0f);
        matrix_translate (matrix, 0, -h * 0.5f, 0.0f);
    }
    matrix_scale (matrix, w, h, 1.0f);
    matrix_mult (matrix, s_matprj, matrix);

    glUniformMatrix4fv (s_loc_mtx, 1, GL_FALSE, matrix);
    glUniform4fv (s_loc_color, 1, tparam->color);

    if (s_loc_texdim >= 0)
    {
        float texdim[2];
        texdim[0] = tparam->texw;
        texdim[1] = tparam->texh;
        glUniform2fv (s_loc_texdim, 1, texdim);
    }

    if (sobj->loc_vtx >= 0)
    {
        glEnableVertexAttribArray (sobj->loc_vtx);
        glVertexAttribPointer (sobj->loc_vtx, 2, GL_FLOAT, GL_FALSE, 0, varray);
    }

    glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);

    glDisable (GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
    
    GLASSERT ();
    return 0;
    
}


int
draw_colored_hair (texture_2d_t *tex, int hair_texid, int x, int y, int w, int h, int upsidedown, float *color)
{
    texparam_t tparam = {0};
    tparam.x       = x;
    tparam.y       = y;
    tparam.w       = w;
    tparam.h       = h;
    tparam.texid   = tex->texid;
    tparam.texid1  = hair_texid;
    tparam.textype = 1;
    tparam.texw    = tex->width;
    tparam.texh    = tex->height;
    tparam.color[0]= color[0];
    tparam.color[1]= color[1];
    tparam.color[2]= color[2];
    tparam.color[3]= color[3];
    tparam.upsidedown = upsidedown;

    if (tex->format == pixfmt_fourcc('Y', 'U', 'Y', 'V'))
        tparam.textype = 4;

    draw_2d_texture_in (&tparam);

    return 0;
}

