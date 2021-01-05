/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <GLES2/gl2.h>
#include "util_shader.h"
#include "assertgl.h"
#include "filter_gaussian.h"


typedef struct gauss_shader_obj_t
{
  GLuint program;
  GLint  loc_vtx;
  GLint  loc_clr;
  GLint  loc_uv;
  GLint  loc_tex;
  GLint  loc_woffset;
  GLint  loc_hoffset;
} gauss_shader_obj_t;

static gauss_shader_obj_t s_sobj;




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


/*
  radius = 2;

  +--+--+--+--+--+
  |  |  |  |  |  |
  +--+--+--+--+--+
  |  |  |  |  |  |
  +--+--+--+--+--+
  |  |  |##|  |  |
  +--+--+--+--+--+
  |  |  |  |  |  |
  +--+--+--+--+--+
  |  |  |  |  |  |
  +--+--+--+--+--+
*/

static float *
generate_gaussian_weight (int radius, float sigma)
{
  float *gweight = calloc (radius + 1, sizeof (float));
  float sum = 0;
  int   i;

  for (i = 0; i < radius + 1; i ++)
    {
      float w = (1.0f / sqrt(2.0f * M_PI * pow(sigma, 2.0f))) *
	exp(-pow(i, 2.0f) / (2.0f * pow(sigma, 2.0f)));

      if (i == 0)
	sum += w;
      else
	sum += w * 2.0f;

      gweight[i] = w;
    }
  
  /* normalize */
  for (i = 0; i < radius + 1; i ++)
    {
      gweight[i] /= sum;
    }

  return gweight;
}


static char *
generate_gaussian_vs (int radius)
{
  int i, fsize;
  char *str_vs, buf[1024];
  char vs_base[] = "                           \n\
attribute     vec4   a_Vertex;                 \n\
attribute     vec2   a_TexCoord;               \n\
uniform       float  u_texWOfst;               \n\
uniform       float  u_texHOfst;               \n\
varying       vec2   v_TexCoord[%d];           \n\
                                               \n\
void main (void)                               \n\
{                                              \n\
  gl_Position = a_Vertex;                      \n\
                                               \n\
  vec2 step = vec2 (u_texWOfst, u_texHOfst);   \n";

  
  str_vs = malloc(4096);

  fsize = radius * 2 + 1;
  sprintf (str_vs, vs_base, fsize);

  for (i = 0; i < fsize; i ++)
    {
      int  ofst = i - radius;

      if (ofst < 0)
	{
	  sprintf (buf, "v_TexCoord[%2d] = a_TexCoord - step * %f;\n", i, (float)(-ofst));
	}
      else if (ofst > 0)
	{
	  sprintf (buf, "v_TexCoord[%2d] = a_TexCoord + step * %f;\n", i, (float)( ofst));
	}
      else
	{
	  sprintf (buf, "v_TexCoord[%2d] = a_TexCoord;\n", i);
	}

      strcat (str_vs, buf);
    }

  sprintf (buf, "}\n");
  strcat (str_vs, buf);
  

  return str_vs;
}

static char 
*generate_gaussian_fs (int radius, float sigma)
{
  int   i, fsize;
  float *gweight = generate_gaussian_weight (radius, sigma);
  char *str_fs, buf[1024];
  char fs_base[] = "                           \n\
precision mediump float;                       \n\
uniform sampler2D u_sampler;                   \n\
varying vec2      v_TexCoord[%d];              \n\
                                               \n\
void main (void)                               \n\
{                                              \n\
  vec4 sum = vec4(0.0);                        \n";

  str_fs = malloc(4096);

  fsize = radius * 2 + 1;
  sprintf (str_fs, fs_base, fsize);

  for (i = 0; i < fsize; i ++)
    {
      int ofst = i - radius;

      if (ofst < 0)
	{
	  sprintf (buf, "sum += texture2D(u_sampler, v_TexCoord[%2d]) * %f;\n", i, gweight[-ofst]);
	}
      else
	{
	  sprintf (buf, "sum += texture2D(u_sampler, v_TexCoord[%2d]) * %f;\n", i, gweight[ ofst]);
	}
      
      strcat (str_fs, buf);
    }

  sprintf (buf, "gl_FragColor = sum;\n}\n");
  strcat (str_fs, buf);

  return str_fs;
}


static float
calc_filter_radius_by_sigma (float sigma)
{
  float cutoff_w = 1.0f / 256.0f;
  int   radius = 0;

  radius = sqrt(
		-2.0f * pow (sigma, 2.0f) * log(cutoff_w * sqrt(2.0f * M_PI * pow (sigma, 2.0f)))
		);
  radius = floor (radius);

  return radius;
}


int
init_gaussian_blur_filter (float sigma)
{
  int radius = 0;
  char *vs = NULL, *fs = NULL;
  shader_obj_t base_sobj;

  radius = calc_filter_radius_by_sigma (sigma);
  fprintf (stderr, "sigma: %f ==> radius: %3d (%3dx%3d)\n", 
	   sigma, radius, (radius * 2 + 1), (radius * 2 + 1));

  vs = generate_gaussian_vs (radius);
  fs = generate_gaussian_fs (radius, sigma);

  if (generate_shader (&base_sobj, vs, fs) < 0)
    {
      fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
      goto err_exit;
    }
  
  s_sobj.program = base_sobj.program;
  s_sobj.loc_vtx = base_sobj.loc_vtx;
  s_sobj.loc_clr = base_sobj.loc_clr;
  s_sobj.loc_uv  = base_sobj.loc_uv;
  s_sobj.loc_tex = base_sobj.loc_tex;
  s_sobj.loc_woffset = glGetUniformLocation (s_sobj.program, "u_texWOfst");
  s_sobj.loc_hoffset = glGetUniformLocation (s_sobj.program, "u_texHOfst");

  return 0;

 err_exit:
  if (vs) free (vs);
  if (fs) free (fs);

  return -1;
}




int
apply_gaussian_filter (render_target_t *dst_fbo, render_target_t *src_fbo)
{
  gauss_shader_obj_t *sobj = &s_sobj;
  render_target_t    fbo;

  create_render_target (&fbo, src_fbo->width, src_fbo->height, RTARGET_COLOR);

  glUseProgram (sobj->program);

  glDisable (GL_BLEND);
    
  glEnableVertexAttribArray (sobj->loc_vtx);
  glEnableVertexAttribArray (sobj->loc_uv );
  glVertexAttribPointer (sobj->loc_vtx, 2, GL_FLOAT, GL_FALSE, 0, s_Vertices);
  glVertexAttribPointer (sobj->loc_uv,  2, GL_FLOAT, GL_FALSE, 0, s_TexCoords);

  glUniform1i (sobj->loc_tex, 0);

  /* -------------------------- *
   * [PATH-1] Horizontal blur 
   * -------------------------- */
  set_render_target (&fbo);

  glClearColor (0.1f, 0.5f, 0.9f, 0.6f);
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  glBindTexture (GL_TEXTURE_2D, src_fbo->texc_id);

  glUniform1f (sobj->loc_woffset, 1.0f / src_fbo->width);
  glUniform1f (sobj->loc_hoffset, 0.0f);

  glDrawArrays (GL_TRIANGLE_STRIP, 0, 4); 

  /* -------------------------- *
   * [PATH-2] Vertical blur
   * -------------------------- */
  set_render_target (dst_fbo);

  glClearColor (0.1f, 0.5f, 0.9f, 0.6f);
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  glBindTexture (GL_TEXTURE_2D, fbo.texc_id);

  glUniform1f (sobj->loc_woffset, 0.0f);
  glUniform1f (sobj->loc_hoffset, 1.0f / src_fbo->height);
  
  glDrawArrays (GL_TRIANGLE_STRIP, 0, 4); 

  destroy_render_target (&fbo);

  GLASSERT ();
  return 0;
}
