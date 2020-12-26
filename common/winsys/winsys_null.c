/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>

#define UNUSED(x) (void)(x)

void *
winsys_init_native_display (void)
{
    return NULL;
}

void *
winsys_init_native_window (void *dpy, int win_w, int win_h)
{
    UNUSED (dpy);
    UNUSED (win_w);
    UNUSED (win_h);
    return NULL;
}

int 
winsys_swap()
{
    return 0;
}

void *
winsys_create_native_pixmap (int width, int height)
{
    UNUSED (width);
    UNUSED (height);
    return NULL;
}

