/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#if defined (USE_GLES_31)
#include <GLES3/gl31.h>
#include <GLES3/gl3ext.h>
#endif
#include "util_shader.h"
#include "util_egl.h"


/* ----------------------------------------------------------- *
 *   create & compile shader
 * ----------------------------------------------------------- */
GLuint
compile_shader_text (GLenum shaderType, const char *text)
{
  GLuint shader;
  GLint	 stat;

  shader = glCreateShader (shaderType);
  glShaderSource  (shader, 1, (const char **)&text, NULL);
  glCompileShader (shader);

  glGetShaderiv	(shader, GL_COMPILE_STATUS, &stat);
  if (!stat) 
    {
      GLsizei len;
      char    *lpBuf;

      glGetShaderiv (shader, GL_INFO_LOG_LENGTH, &len);
      lpBuf = (char *)malloc (len);

      glGetShaderInfoLog (shader, len, &len, lpBuf);
      fprintf (stderr, "Error: problem compiling shader.\n");
      fprintf (stderr, "-----------------------------------\n");
      fprintf (stderr, "%s\n", lpBuf);
      fprintf (stderr, "-----------------------------------\n");

      free (lpBuf);

      return 0;
    }
	
  return shader;
}


GLuint
compile_shader_file (GLenum shaderType, const char *lpFName)
{
  FILE   *fp;
  char   *lpbuf = NULL;
  int    nFileSize = 0;
  int    nReadSize = 0;
  GLuint shader;


  fp = fopen (lpFName, "r");
  if (fp == NULL) 
    {
      fprintf (stderr, "can't open %s\n", lpFName);
      fprintf (stderr, "FATAL ERROR at %s(%d)\n", __FILE__, __LINE__);
      return 0;
    }
	
  fseek (fp, 0, SEEK_END);
  nFileSize = ftell (fp);
  fseek (fp, 0, SEEK_SET);

  lpbuf = (char *)malloc (nFileSize + 1); 
  if (lpbuf == NULL) 
    {
      fprintf (stderr, "FATAL ERROR at %s(%d)\n", __FILE__, __LINE__);
      fclose (fp);
      return 0;
    }

  nReadSize = fread (lpbuf, 1, nFileSize, fp);
  lpbuf[ nReadSize ] = '\0';
  
  shader = compile_shader_text (shaderType, lpbuf);

  fclose( fp );
  free (lpbuf);

  return shader;
}


/* ----------------------------------------------------------- *
 *    link shaders
 * ----------------------------------------------------------- */
GLuint
link_shaders (GLuint vertShader, GLuint fragShader)
{
  GLuint program = glCreateProgram();

  if (fragShader) glAttachShader (program, fragShader);
  if (vertShader) glAttachShader (program, vertShader);

  glLinkProgram (program);

  {
    GLint stat;
    glGetProgramiv (program, GL_LINK_STATUS, &stat);
    if (!stat) 
      {
	GLsizei len;
	char	*lpBuf;
	
	glGetProgramiv (program, GL_INFO_LOG_LENGTH, &len);
	lpBuf = (char *)malloc (len);

	glGetProgramInfoLog (program, len, &len, lpBuf);
	fprintf (stderr, "Error: problem linking shader.\n");
	fprintf (stderr, "-----------------------------------\n");
	fprintf (stderr, "%s\n", lpBuf);
	fprintf (stderr, "-----------------------------------\n");

	free (lpBuf);

	return 0;
      }
  }

  return program;
}

int
build_shader (const char *strVS, const char *strFS)
{
    GLuint vs, fs, prog;
    
    vs = compile_shader_text (GL_VERTEX_SHADER, strVS);
    fs = compile_shader_text (GL_FRAGMENT_SHADER, strFS);
    prog = link_shaders (vs, fs);

    return prog;
}

int
generate_shader (shader_obj_t *sobj, char *str_vs, char *str_fs)
{
  GLuint fs, vs, program;

  vs = compile_shader_text (GL_VERTEX_SHADER,   str_vs);
  fs = compile_shader_text (GL_FRAGMENT_SHADER, str_fs);
  if (vs == 0 || fs == 0)
    {
      fprintf (stderr, "Failed to compile shader.\n");
      return -1;
    }

  program = link_shaders (vs, fs);
  if (program == 0)
    {
      fprintf (stderr, "Failed to link shaders.\n");
      return -1;
    }

  glDeleteShader (vs);
  glDeleteShader (fs);

  sobj->program = program;
  sobj->loc_vtx = glGetAttribLocation (program, "a_Vertex"  );
  sobj->loc_nrm = glGetAttribLocation (program, "a_Normal"  );
  sobj->loc_clr = glGetAttribLocation (program, "a_Color"   );
  sobj->loc_uv  = glGetAttribLocation (program, "a_TexCoord");
  sobj->loc_tex = glGetUniformLocation(program, "u_sampler" );
  sobj->loc_mtx = glGetUniformLocation(program, "u_PMVMatrix" );
  sobj->loc_mtx_nrm = glGetUniformLocation(program, "u_NrmMatrix");

  return 0;
}

int
generate_shader_from_file (shader_obj_t *sobj, char *dir_name, char *vs_fname, char *fs_fname)
{
  GLuint fs, vs, program;
  char   vs_path[128], fs_path[128];

  snprintf (vs_path, sizeof (vs_path), "%s/%s", dir_name, vs_fname);
  snprintf (fs_path, sizeof (fs_path), "%s/%s", dir_name, fs_fname);

  vs = compile_shader_file (GL_VERTEX_SHADER,   vs_path);
  fs = compile_shader_file (GL_FRAGMENT_SHADER, fs_path);
  if (vs == 0 || fs == 0)
    {
      fprintf (stderr, "Failed to compile shader.\n");
      return -1;
    }

  program = link_shaders (vs, fs);
  if (program == 0)
    {
      fprintf (stderr, "Failed to link shaders.\n");
      return -1;
    }

  glDeleteShader (vs);
  glDeleteShader (fs);

  sobj->program = program;
  sobj->loc_vtx = glGetAttribLocation (program, "a_Vertex"  );
  sobj->loc_nrm = glGetAttribLocation (program, "a_Normal"  );
  sobj->loc_clr = glGetAttribLocation (program, "a_Color"   );
  sobj->loc_uv  = glGetAttribLocation (program, "a_TexCoord");
  sobj->loc_tex = glGetUniformLocation(program, "u_sampler" );
  sobj->loc_mtx = glGetUniformLocation(program, "u_PMVMatrix" );

  return 0;
}

#if defined (USE_GLES_31)
int
build_compute_shader (const char *strCS)
{
    GLuint cs, prog;

    cs = compile_shader_text (GL_COMPUTE_SHADER, strCS);
    prog = link_shaders (cs, 0);

    return prog;
}

int
build_compute_shader_from_file (char *dir_name, char *cs_fname)
{
    GLuint cs, prog;
    char cs_path[128];

    snprintf (cs_path, sizeof (cs_path), "%s/%s", dir_name, cs_fname);

    cs = compile_shader_file (GL_COMPUTE_SHADER, cs_path);
    if (cs == 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    prog = link_shaders (cs, 0);

    return prog;
}

#endif

int
delete_shader (shader_obj_t *sobj)
{
  glDeleteProgram (sobj->program);
  return 0;
}

#if defined (EXT_separate_shader_objects)
int
generate_separate_shader (separate_shader_obj_t *sobj, char *str_vs, char *str_fs)
{
    GLuint ppo;
    GLuint fs = 0, vs = 0;
    GLint  loc_vtx, loc_clr, loc_uv;

    PFNGLGENPROGRAMPIPELINESEXTPROC  glGenProgramPipelinesEXT  = NULL;
    PFNGLBINDPROGRAMPIPELINEEXTPROC  glBindProgramPipelineEXT  = NULL;
    PFNGLCREATESHADERPROGRAMVEXTPROC glCreateShaderProgramvEXT = NULL;
    PFNGLUSEPROGRAMSTAGESEXTPROC     glUseProgramStagesEXT     = NULL;
    
    EGL_GET_PROC_ADDR (glGenProgramPipelinesEXT);
    EGL_GET_PROC_ADDR (glBindProgramPipelineEXT);
    EGL_GET_PROC_ADDR (glCreateShaderProgramvEXT);
    EGL_GET_PROC_ADDR (glUseProgramStagesEXT);

    glGenProgramPipelinesEXT (1, &ppo);
    glBindProgramPipelineEXT(ppo);

    if (str_vs)
    {
        vs = glCreateShaderProgramvEXT (GL_VERTEX_SHADER, 1, (const GLchar **)&str_vs);
        if (vs == 0)
        {
          fprintf (stderr, "Failed to compile shader.\n");
          return -1;
        }

        glUseProgramStagesEXT(ppo, GL_VERTEX_SHADER_BIT_EXT, vs);
    }
    
    if (str_fs)
    {
        fs = glCreateShaderProgramvEXT (GL_FRAGMENT_SHADER, 1, (const GLchar **)&str_fs);
        if (fs == 0)
        {
          fprintf (stderr, "Failed to compile shader.\n");
          return -1;
        }

        glUseProgramStagesEXT(ppo, GL_FRAGMENT_SHADER_BIT_EXT, fs);
    }

    sobj->pipeline   = ppo;
    sobj->program_fs = fs;
    sobj->program_vs = vs;

    return 0;
}
#endif
