/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */

/*
 * EGL window system dependent module for X11.
 * At first, you need to set up environment as below.
 *  > sudo apt install libgles2-mesa-dev libegl1-mesa-dev xorg-dev
 */
#include <stdio.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GLES2/gl2.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "winsys_x11.h"

#define UNUSED(x) (void)(x)

static Display *s_xdpy;
static Window  s_xwin;


void *
winsys_init_native_display (void)
{
    Display *xdpy = XOpenDisplay (NULL);
    if (xdpy == NULL)
    {
        fprintf (stderr, "Can't open XDisplay.\n");
    }

    s_xdpy = xdpy;

    return (void *)xdpy;
}


void *
winsys_init_native_window (void *dpy, int win_w, int win_h)
{
    UNUSED (dpy); /* We use XDisplay instead of EGLDisplay. */
    Display *xdpy = s_xdpy;

    unsigned long black = BlackPixel (xdpy, DefaultScreen (xdpy));
    unsigned long white = WhitePixel (xdpy, DefaultScreen (xdpy));

    Window xwin = XCreateSimpleWindow (xdpy,
                                       RootWindow (xdpy, DefaultScreen (xdpy)),
                                       0, 0, win_w, win_h, 
                                       1, black, white);
    XMapWindow (xdpy, xwin);
    XFlush (xdpy);
    
    s_xwin = xwin;

    return (void *)xwin;
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

