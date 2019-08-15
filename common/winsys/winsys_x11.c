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
    Display *dpy = XOpenDisplay (NULL);
    if (dpy == NULL)
    {
        fprintf (stderr, "Can't open XDisplay.\n");
    }

    s_xdpy = dpy;

    return (void *)dpy;
}


void *
winsys_init_native_window (void *dpy, int win_w, int win_h)
{
    Display *xdpy = s_xdpy;
    Window xwindow;

    unsigned long black,white;
    black=BlackPixel(xdpy,DefaultScreen(xdpy)); //色の取得
    white=WhitePixel(xdpy,DefaultScreen(xdpy));
    xwindow=XCreateSimpleWindow(xdpy,RootWindow(xdpy,DefaultScreen(xdpy)),100,50,800,530,1,black,white); //ウィンドウの生成

    XMapWindow(xdpy, xwindow);
    
    s_xwin = xwindow;

    return (void *)xwindow;
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

