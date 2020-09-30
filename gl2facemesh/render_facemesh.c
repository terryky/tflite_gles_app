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
#include "util_render2d.h"
#include "tflite_facemesh.h"

#define UNUSED(x) (void)(x)

static int          s_texid_floor;

static shader_obj_t s_sobj;
static float        s_matPrj[16];
static GLint        s_loc_mtx_mv;
static GLint        s_loc_mtx_pmv;
static GLint        s_loc_mtx_nrm;
static GLint        s_loc_color;
static GLint        s_loc_alpha;
static GLint        s_loc_lightpos;

static GLuint       s_vbo_vtxalpha[2];

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

static GLfloat s_nrm_inv[] =
{
     0.0f,  0.0f, -1.0f,
     0.0f,  0.0f,  1.0f,
    -1.0f,  0.0f,  0.0f,
     1.0f,  0.0f,  0.0f,
     0.0f, -1.0f,  0.0f,
     0.0f,  1.0f,  0.0f,
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
uniform   vec3  u_LightPos;                                 \n\
const     vec3  LightCol = vec3(1.0, 1.0, 1.0);             \n\
                                                            \n\
void DirectionalLight (vec3 normal, vec3 eyePos)            \n\
{                                                           \n\
    vec3  lightDir = normalize (u_LightPos);                \n\
    vec3  halfV    = normalize (u_LightPos - eyePos);       \n\
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
    v_diffuse  = vec3(0.5);                                 \n\
    v_specular = vec3(0.0);                                 \n\
    DirectionalLight(normal, eyePos);                       \n\
                                                            \n\
    v_diffuse = clamp(v_diffuse, 0.0, 1.0);                 \n\
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
    color = vec3(texture2D(u_sampler, v_texcoord));         \n\
    color *= (u_color * v_diffuse);                         \n\
    //color += v_specular;                                  \n\
    gl_FragColor = vec4(color, u_alpha);                    \n\
}                                                           ";


static void
compute_invmat3x3 (float *matMVI3x3, float *matMV)
{
    float matMVI4x4[16];

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
    s_loc_lightpos= glGetUniformLocation(s_sobj.program, "u_LightPos" );

    matrix_proj_perspective (s_matPrj, 30.0f, aspect, 10.f, 10000.f);

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
draw_floor (float *mtxGlobal, float div_u, float div_v)
{
    int i;
    float matMV[16], matPMV[16], matMVI3x3[9];
    GLfloat floor_uv [] =
    {
          0.0f,  0.0f,
          0.0f, div_v,
         div_u,  0.0f,
         div_u, div_v,
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
    compute_invmat3x3 (matMVI3x3, matMV);

    matrix_mult (matMV, mtxGlobal, matMV);
    matrix_mult (matPMV, s_matPrj, matMV);

    glUniformMatrix4fv (s_loc_mtx_mv,   1, GL_FALSE, matMV );
    glUniformMatrix4fv (s_loc_mtx_pmv,  1, GL_FALSE, matPMV);
    glUniformMatrix3fv (s_loc_mtx_nrm,  1, GL_FALSE, matMVI3x3);
    glUniform3f (s_loc_lightpos, 1.0f, 2.0f, 3.0f);
    glUniform3f (s_loc_color, 0.9f, 0.9f, 0.9f);
    glUniform1f (s_loc_alpha, 1.0f);

    glDisable (GL_BLEND);

    for (i = 0; i < 6; i ++)
    {
        glBindTexture (GL_TEXTURE_2D, s_texid_floor);

        glVertexAttribPointer (s_sobj.loc_vtx, 3, GL_FLOAT, GL_FALSE, 0, &s_vtx[4 * 3 * i]);
        glVertexAttrib4fv (s_sobj.loc_nrm, &s_nrm_inv[3 * i]);
        glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);
    }

    glDisable (GL_BLEND);
    glFrontFace (GL_CCW);

    return 0;
}











/* ------------------------------------------------------ *
 *  shader for Texture
 * ------------------------------------------------------ */
static char vs_tex[] = "                              \n\
attribute    vec4    a_Vertex;                        \n\
attribute    vec2    a_TexCoord;                      \n\
attribute    float   a_vtxalpha;                      \n\
varying      vec2    v_TexCoord;                      \n\
varying      float   v_vtxalpha;                      \n\
uniform      mat4    u_PMVMatrix;                     \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    gl_Position = u_PMVMatrix * a_Vertex;             \n\
    v_TexCoord  = a_TexCoord;                         \n\
    v_vtxalpha  = a_vtxalpha;                         \n\
}                                                     \n";

static char fs_tex[] = "                              \n\
precision mediump float;                              \n\
varying     vec2      v_TexCoord;                     \n\
varying     float     v_vtxalpha;                     \n\
uniform     sampler2D u_sampler;                      \n\
uniform     vec4      u_Color;                        \n\
                                                      \n\
void main (void)                                      \n\
{                                                     \n\
    gl_FragColor = texture2D (u_sampler, v_TexCoord); \n\
    gl_FragColor *= u_Color;                          \n\
    gl_FragColor.a *= v_vtxalpha;                     \n\
}                                                     \n";


static shader_obj_t s_sobj2;
static float s_matprj2[16];
static GLint        s_loc_mtx;
static GLint        s_loc_col;
static GLint        s_loc_vtxalpha;

static int
set_projection_matrix2 (int w, int h)
{
    float mat_proj[] =
    {
       0.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f, 0.0f,
      -1.0f, 1.0f, 0.0f, 1.0f};

    mat_proj[0] =  2.0f / (float)w;
    mat_proj[5] = -2.0f / (float)h;

    memcpy (s_matprj2, mat_proj, 16*sizeof(float));

    GLASSERT ();
    return 0;
}

static GLuint
create_vbo_alpha_array (int drill_eye_hole)
{
    /*
     *  Vertex indices are from:
     *      https://github.com/tensorflow/tfjs-models/blob/master/facemesh/src/keypoints.ts
     */
    const int face_contour_idx[] = {
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    };

    int num_idx;
    get_facemesh_tri_indicies (&num_idx, drill_eye_hole);
    float *alpha_array = (float *)malloc (num_idx * sizeof(float));

    for (int i = 0; i < num_idx; i ++)
    {
        float alpha = 1.0f;
        for (int j = 0; j < sizeof (face_contour_idx) / sizeof (int); j ++)
        {
            if (i == face_contour_idx[j])
            {
                alpha = 0;
                break;
            }
        }
        alpha_array[i] = alpha;
    }

    GLuint vboid;
    glGenBuffers (1, &vboid);

    glBindBuffer (GL_ARRAY_BUFFER, vboid);
    glBufferData (GL_ARRAY_BUFFER, num_idx * sizeof(float), alpha_array, GL_STATIC_DRAW);
    glBindBuffer (GL_ARRAY_BUFFER, 0);

    free (alpha_array);
    return vboid;
}


int
init_facemesh_renderer (int w, int h)
{
    if (generate_shader (&s_sobj2, vs_tex, fs_tex) < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    s_loc_mtx = glGetUniformLocation(s_sobj2.program, "u_PMVMatrix" );
    s_loc_col = glGetUniformLocation(s_sobj2.program, "u_Color" );
    s_loc_vtxalpha = glGetAttribLocation (s_sobj2.program, "a_vtxalpha");

    set_projection_matrix2 (w, h);

    s_vbo_vtxalpha[0] = create_vbo_alpha_array (0);
    s_vbo_vtxalpha[1] = create_vbo_alpha_array (1);

    return 0;
}


int
draw_facemesh_tri_tex (int texid, fvec3 *vtx, fvec3 *uv, float *color, int drill_eye_hole)
{
    shader_obj_t *sobj = &s_sobj2;
    float matrix[16];
    int num_idx;
    int *mesh_tris = get_facemesh_tri_indicies (&num_idx, drill_eye_hole);

    glBindBuffer (GL_ARRAY_BUFFER, 0);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

    glUseProgram (sobj->program);
    glUniform1i(sobj->loc_tex, 0);

    glBindTexture (GL_TEXTURE_2D, texid);

    if (sobj->loc_uv >= 0)
    {
        glEnableVertexAttribArray (sobj->loc_uv);
        glVertexAttribPointer (sobj->loc_uv, 3, GL_FLOAT, GL_FALSE, 0, uv);
    }

    glEnable (GL_BLEND);
    glEnable (GL_CULL_FACE);

    matrix_identity (matrix);
    matrix_mult (matrix, s_matprj2, matrix);

    glUniformMatrix4fv (s_loc_mtx, 1, GL_FALSE, matrix);
    glUniform4fv (s_loc_col, 1, color);

    glEnableVertexAttribArray (sobj->loc_vtx);
    glVertexAttribPointer (sobj->loc_vtx, 3, GL_FLOAT, GL_FALSE, 0, vtx);

    glBindBuffer (GL_ARRAY_BUFFER, s_vbo_vtxalpha[drill_eye_hole]);
    glEnableVertexAttribArray (s_loc_vtxalpha);
    glVertexAttribPointer (s_loc_vtxalpha, 1, GL_FLOAT, GL_FALSE, 0, 0);

    glDrawElements (GL_TRIANGLES, num_idx, GL_UNSIGNED_INT, mesh_tris);

    glDisable (GL_BLEND);
    glBindBuffer (GL_ARRAY_BUFFER, 0);

    GLASSERT ();
    return 0;
}

int
draw_facemesh_line (fvec3 *joint, float *color, int drill_eye_hole)
{
    int num_idx;
    int *mesh_tris = get_facemesh_tri_indicies (&num_idx, drill_eye_hole);

    glBindBuffer (GL_ARRAY_BUFFER, 0);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

    for (int i = 0; i < num_idx/3; i ++)
    {
        int idx0 = mesh_tris[3 * i + 0];
        int idx1 = mesh_tris[3 * i + 1];
        int idx2 = mesh_tris[3 * i + 2];
        float x1 = joint[idx0].x;
        float y1 = joint[idx0].y;
        float x2 = joint[idx1].x;
        float y2 = joint[idx1].y;
        float x3 = joint[idx2].x;
        float y3 = joint[idx2].y;

        draw_2d_line (x1, y1, x2, y2, color, 1.0f);
        draw_2d_line (x2, y2, x3, y3, color, 1.0f);
        draw_2d_line (x3, y3, x1, y1, color, 1.0f);
    }

    GLASSERT ();
    return 0;
}
