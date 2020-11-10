/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef UTIL_GLX_H_
#define UTIL_GLX_H_

#define GL_GLEXT_PROTOTYPES  (1)
#define GLX_GLXEXT_PROTOTYPES (1)
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glxext.h>

int glx_initialize (int glx_version, int depth_size, int stencil_size, int sample_num,
                    int win_w, int win_h);
int glx_terminate ();
int glx_swap ();

#endif /* UTIL_GLX_H_ */
