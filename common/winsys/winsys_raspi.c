/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <signal.h>

#include "winsys_raspi.h"
#include "bcm_host.h"

#include "GLES2/gl2.h"
#include "EGL/egl.h"
#include "EGL/eglext.h"

#define UNUSED(x) (void)(x)

struct Display s_display;
struct Window  s_window;



void *
winsys_init_native_display (void)
{
    bcm_host_init();
    return EGL_DEFAULT_DISPLAY;
}


static EGL_DISPMANX_WINDOW_T nativewindow;
void *
winsys_init_native_window (void *dpy, int win_w, int win_h)
{
    DISPMANX_ELEMENT_HANDLE_T dispman_element;
    DISPMANX_DISPLAY_HANDLE_T dispman_display;
    DISPMANX_UPDATE_HANDLE_T dispman_update;
    VC_RECT_T dst_rect;
    VC_RECT_T src_rect;

    UNUSED (dpy);
    memset (&s_window, 0, sizeof (s_window));

    if (win_w == 0 || win_h == 0)
    {
        graphics_get_display_size(0 /* LCD */, (uint32_t *)&win_w, (uint32_t *)&win_h);
    }

    dst_rect.x = 0;
    dst_rect.y = 0;
    dst_rect.width  = win_w;
    dst_rect.height = win_h;
      
    src_rect.x = 0;
    src_rect.y = 0;
    src_rect.width  = win_w << 16;
    src_rect.height = win_h << 16;
    
    dispman_display = vc_dispmanx_display_open( 0 /* LCD */);
    dispman_update  = vc_dispmanx_update_start( 0 );
         
    dispman_element = vc_dispmanx_element_add ( dispman_update, dispman_display,
        0/*layer*/, &dst_rect, 0/*src*/,
        &src_rect, DISPMANX_PROTECTION_NONE, 0 /*alpha*/, 0/*clamp*/, 
        (DISPMANX_TRANSFORM_T) 0/*transform*/);
      
    nativewindow.element = dispman_element;
    nativewindow.width   = win_w;
    nativewindow.height  = win_h;
    vc_dispmanx_update_submit_sync( dispman_update );

    s_window.window_size.width  = win_w;
    s_window.window_size.height = win_h;
    s_window.display = &s_display;

    return &nativewindow;
}


int 
winsys_swap()
{
  return 0;
}

void *
winsys_create_native_pixmap (int width, int height)
{
  return NULL;
}

