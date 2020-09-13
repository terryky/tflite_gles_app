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

static void (*s_motion_func)(int x, int y) = NULL;
static void (*s_button_func)(int button, int state, int x, int y) = NULL;
static void (*s_key_func)(int key, int state, int x, int y) = NULL;

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
    XSelectInput (xdpy, xwin, ButtonPressMask | ButtonReleaseMask | Button1MotionMask);
    XFlush (xdpy);

    s_xwin = xwin;

    return (void *)xwin;
}


int 
winsys_swap()
{
    XEvent event;
    while (XPending (s_xdpy))
    {
        XNextEvent (s_xdpy, &event);
        switch (event.type)
        {
        case ButtonPress:
            if (s_button_func)
            {
                s_button_func (0/*event.xbutton.button*/, 1, event.xbutton.x, event.xbutton.y);
            }
            break;
        case ButtonRelease:
            if (s_button_func)
            {
                s_button_func (0/*event.xbutton.button*/, 0, event.xbutton.x, event.xbutton.y);
            }
            break;
        case MotionNotify:
            if (s_motion_func)
            {
                s_motion_func (event.xmotion.x, event.xmotion.y);
            }
            break;
        default:
            /* Unknown event type, ignore it */
            break;
        }
    }
    return 0;
}

void *
winsys_create_native_pixmap (int width, int height)
{
    return NULL;
}




void egl_set_motion_func (void (*func)(int x, int y))
{
    s_motion_func = func;
}

void egl_set_button_func (void (*func)(int button, int state, int x, int y))
{
    s_button_func = func;
}

void egl_set_key_func (void (*func)(int key, int state, int x, int y))
{
    s_key_func = func;
}



