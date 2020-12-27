/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_EGL_H_
#define _UTIL_EGL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <EGL/egl.h>
#include <EGL/eglext.h>

int egl_init_with_pbuffer_surface (int gles_version, int depth_size, int stencil_size, int sample_num, int win_w, int win_h);
int egl_init_with_window_surface (int gles_version, void *window, int depth_size, int stencil_size, int sample_num);
int egl_init_with_platform_window_surface (int gles_version, int depth_size, int stencil_size, int sample_num, int win_w, int win_h);
int egl_init_with_platform_device_surface (int gles_version, int depth_size, int stencil_size, int sample_num, int win_w, int win_h);
int egl_init_with_eglstream_surface       (int gles_version, int depth_size, int stencil_size, int sample_num, int win_w, int win_h, int stream_fd);
int egl_init_and_create_eglstream         (int *stream_fd);
int egl_create_eglstream_surface          (int gles_version, int depth_size, int stencil_size, int sample_num, int win_w, int win_h);
int egl_terminate ();
int egl_swap ();
int egl_set_swap_interval (int interval);

EGLImageKHR egl_create_eglimage (int width, int height);

int egl_get_current_surface_dimension (int *width, int *height);

int egl_show_current_context_attrib ();
int egl_show_current_config_attrib ();
int egl_show_current_surface_attrib ();
int egl_show_gl_info ();


void egl_set_motion_func (void (*func)(int x, int y));
void egl_set_button_func (void (*func)(int button, int state, int x, int y));
void egl_set_key_func (void (*func)(int key, int state, int x, int y));


EGLDisplay egl_get_display ();
EGLContext egl_get_context ();
EGLSurface egl_get_surface ();
EGLConfig  egl_get_config  ();

#define EGL_GET_PROC_ADDR(name)                                 \
do {                                                            \
    name = (void *)eglGetProcAddress (#name);                   \
    if (!name)                                                  \
    {                                                           \
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);  \
    }                                                           \
} while (0)


#ifdef __cplusplus
}
#endif
#endif
