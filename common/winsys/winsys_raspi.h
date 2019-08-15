/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef UTIL_WAYLAND_H_
#define UTIL_WAYLAND_H_

#define MOD_SHIFT_MASK    0x01
#define MOD_ALT_MASK      0x02
#define MOD_CONTROL_MASK  0x04

struct Window;
struct Display 
{
    void *wlDisplay;
	struct Window *window;
};

struct Geometry {
    int width, height;
};

struct Window {
	struct Display *display;
	int fullscreen, configured, opaque;
	struct Geometry geometry,window_size;
    void *wlEGLNativeWindow;
};


#endif

