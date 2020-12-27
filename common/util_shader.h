/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef SHADER_UTIL_H
#define SHADER_UTIL_H

struct uniform_info
{
   const char *name;
   GLuint size;
   GLenum type;  /**< GL_FLOAT or GL_INT */
   GLfloat value[4];
   GLint location;  /**< filled in by InitUniforms() */
};

typedef struct shader_obj_t
{
  GLuint program;
  GLint  loc_vtx;
  GLint  loc_nrm;
  GLint  loc_clr;
  GLint  loc_uv;
  GLint  loc_tex;
  GLint  loc_mtx;
  GLint  loc_mtx_nrm;
} shader_obj_t;

typedef struct separate_shader_obj_t
{
  GLuint pipeline;
  GLuint program_fs;
  GLuint program_vs;
} separate_shader_obj_t;


#define END_OF_UNIFORMS   { NULL, 0, GL_NONE, { 0, 0, 0, 0 }, -1 }

#ifdef __cplusplus
extern "C" {
#endif

int build_shader (const char *strVS, const char *strFS);
int build_compute_shader (const char *strCS);
int build_compute_shader_from_file (char *dir_name, char *cs_fname);
int generate_shader (shader_obj_t *sobj, char *str_vs, char *str_fs);
int generate_shader_from_file (shader_obj_t *sobj, char *dir_name, char *vs_fname, char *fs_fname);
int generate_separate_shader (separate_shader_obj_t *sobj, char *str_vs, char *str_fs);

#ifdef __cplusplus
}
#endif

#endif /* SHADER_UTIL_H */
