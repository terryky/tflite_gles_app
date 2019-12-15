/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GLES2/gl2.h>
#include "util_shader.h"
#include "util_render_target.h"
#include "assertgl.h"

static shader_obj_t s_sobj;

static char s_str_vs[] = "                         \n\
attribute     vec4   a_Vertex;                     \n\
attribute     vec2   a_TexCoord;                   \n\
varying       vec2   v_TexCoord;                   \n\
                                                   \n\
void main (void)                                   \n\
{                                                  \n\
  v_TexCoord  = a_TexCoord;                        \n\
  gl_Position = a_Vertex;                          \n\
}                                                  \n\
                                                   \n";

static char s_str_fs[] = "                         \n\
precision mediump float;                           \n\
varying vec2      v_TexCoord;                      \n\
uniform sampler2D u_sampler;                       \n\
                                                   \n\
void main (void)                                   \n\
{                                                  \n\
  gl_FragColor = texture2D(u_sampler, v_TexCoord); \n\
}                                                  \n";

static GLfloat s_Vertices[] =
{
  -1.0f,  1.0f,
  -1.0f, -1.0f,
   1.0f,  1.0f,
   1.0f, -1.0f,
};

static GLfloat s_TexCoords[] = 
{
  0.0f, 1.0f,
  0.0f, 0.0f,
  1.0f, 1.0f,
  1.0f, 0.0f,
};


int
init_normal_filter ()
{
    generate_shader (&s_sobj, s_str_vs, s_str_fs);
    GLASSERT ();

    return 0;
}


int
apply_normal_filter (render_target_t *dst_fbo, render_target_t *src_fbo)
{
  shader_obj_t *sobj = &s_sobj;

  set_render_target (dst_fbo);

  glUseProgram (sobj->program);

  glClearColor (0.1f, 0.5f, 0.9f, 0.6f);
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  glEnableVertexAttribArray (sobj->loc_vtx);
  glEnableVertexAttribArray (sobj->loc_uv );
  glVertexAttribPointer (sobj->loc_vtx, 2, GL_FLOAT, GL_FALSE, 0, s_Vertices);
  glVertexAttribPointer (sobj->loc_uv,  2, GL_FLOAT, GL_FALSE, 0, s_TexCoords);
  
  glBindTexture (GL_TEXTURE_2D, src_fbo->texid);
  glUniform1i (sobj->loc_tex, 0);

  glDrawArrays (GL_TRIANGLE_STRIP, 0, 4); 
      
  GLASSERT ();
  return 0;
}

