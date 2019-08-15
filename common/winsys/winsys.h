/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _WINSYS_H_
#define _WINSYS_H_

void *winsys_init_native_display (void);
void *winsys_init_native_window (void *dpy, int win_w, int win_h);
int   winsys_swap();
void *winsys_create_native_pixmap (int width, int height);
#endif /* _WINSYS_H_ */
