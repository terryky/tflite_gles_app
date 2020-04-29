/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
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
#include "util_texture.h"

#define UNUSED(x) (void)(x)

static int          s_texid_dummy = 0;
static int          s_texid_floor;

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
const     vec3  LightCol = vec3(1.0, 1.0, 1.0);             \n\
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
    v_specular+= pf  * LightCol * 0.5;                      \n\
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



int
draw_cube (float *mtxGlobal)
{
    int i;
    float matMV[16], matPMV[16], matMVI4x4[16], matMVI3x3[9];

    glEnable (GL_DEPTH_TEST);
    glEnable (GL_CULL_FACE);
    
    glUseProgram( s_sobj.program );

    glEnableVertexAttribArray (s_sobj.loc_vtx);
    glEnableVertexAttribArray (s_sobj.loc_uv );
    glDisableVertexAttribArray(s_sobj.loc_nrm);
    glVertexAttribPointer (s_sobj.loc_vtx, 3, GL_FLOAT, GL_FALSE, 0, s_vtx);
    glVertexAttribPointer (s_sobj.loc_uv , 2, GL_FLOAT, GL_FALSE, 0, s_uv );

    matrix_identity (matMV);
    matrix_mult (matMV, matMV, mtxGlobal);

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
    glUniform3f (s_loc_color, 1.0f, 0.0f, 0.0f);
    glUniform1f (s_loc_alpha, 1.0f);

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


int
init_cube (float aspect)
{
    generate_shader (&s_sobj, s_strVS, s_strFS);
    s_loc_mtx_mv  = glGetUniformLocation(s_sobj.program, "u_MVMatrix" );
    s_loc_mtx_pmv = glGetUniformLocation(s_sobj.program, "u_PMVMatrix" );
    s_loc_mtx_nrm = glGetUniformLocation(s_sobj.program, "u_ModelViewIT" );
    s_loc_color   = glGetUniformLocation(s_sobj.program, "u_color" );
    s_loc_alpha   = glGetUniformLocation(s_sobj.program, "u_alpha" );

    matrix_proj_perspective (s_matPrj, 30.0f, aspect, 1.f, 1000.f);

    int texw, texh;
    load_png_texture ("floortile.png", &s_texid_floor, &texw, &texh);

    glBindTexture (GL_TEXTURE_2D, s_texid_floor);
    glGenerateMipmap (GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    GLASSERT ();
    return 0;
}



int
draw_floor (float *mtxGlobal)
{
    int i;
    float matMV[16], matPMV[16], matMVI4x4[16], matMVI3x3[9];
    GLfloat floor_uv [] =
    {
         0.0f, 0.0f,
         0.0f, 5.0f,
         5.0f, 0.0f,
         5.0f, 5.0f,
    };
    
    glDisable (GL_DEPTH_TEST);
    glEnable (GL_CULL_FACE);
    glFrontFace (GL_CW);
        
    glUseProgram( s_sobj.program );

    glEnableVertexAttribArray (s_sobj.loc_vtx);
    glEnableVertexAttribArray (s_sobj.loc_uv );
    glDisableVertexAttribArray(s_sobj.loc_nrm);
    glVertexAttribPointer (s_sobj.loc_vtx, 3, GL_FLOAT, GL_FALSE, 0, s_vtx);
    glVertexAttribPointer (s_sobj.loc_uv , 2, GL_FLOAT, GL_FALSE, 0, floor_uv );

    matrix_identity (matMV);
    matrix_mult (matMV, matMV, mtxGlobal);

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
        glBindTexture (GL_TEXTURE_2D, s_texid_floor);

        glVertexAttribPointer (s_sobj.loc_vtx, 3, GL_FLOAT, GL_FALSE, 0, &s_vtx[4 * 3 * i]);
        glVertexAttrib4fv (s_sobj.loc_nrm, &s_nrm[3 * i]);
        glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);
    }

    glDisable (GL_BLEND);
    glFrontFace (GL_CCW);

    return 0;
}



int
draw_line (float *mtxGlobal, float *p0, float *p1)
{
    float matMV[16], matPMV[16], matMVI4x4[16], matMVI3x3[9];
    GLfloat floor_vtx [6];

    floor_vtx[0] = p0[0];
    floor_vtx[1] = p0[1];
    floor_vtx[2] = p0[2];
    floor_vtx[3] = p1[0];
    floor_vtx[4] = p1[1];
    floor_vtx[5] = p1[2];


    glEnable (GL_DEPTH_TEST);
    glDisable (GL_CULL_FACE);
        
    glUseProgram( s_sobj.program );

    glEnableVertexAttribArray (s_sobj.loc_vtx);
    glEnableVertexAttribArray (s_sobj.loc_uv );
    glDisableVertexAttribArray(s_sobj.loc_nrm);
    glVertexAttribPointer (s_sobj.loc_vtx, 3, GL_FLOAT, GL_FALSE, 0, floor_vtx);
    glVertexAttribPointer (s_sobj.loc_uv , 2, GL_FLOAT, GL_FALSE, 0, s_uv );
    glVertexAttrib4fv (s_sobj.loc_nrm, s_nrm);

    matrix_identity (matMV);
    matrix_mult (matMV, matMV, mtxGlobal);

    matrix_identity (matMVI4x4);
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
    glUniform3f (s_loc_color, 0.0f, 0.0f, 1.0f);
    glUniform1f (s_loc_alpha, 1.0f);

    glDisable (GL_BLEND);

    glLineWidth(3.0f);

    glBindTexture (GL_TEXTURE_2D, s_texid_dummy);
    glDrawArrays (GL_LINES, 0, 2);

    glLineWidth(1.0f);
    return 0;
}

