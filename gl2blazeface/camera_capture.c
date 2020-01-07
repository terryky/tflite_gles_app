/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "util_v4l2.h"
#include "util_debug.h"

static pthread_t    s_capture_thread;
static void         *s_capture_buf = NULL;
static capture_dev_t *s_cap_dev;
static int          s_capture_w, s_capture_h;
static unsigned int s_capture_fmt;

#define _max(A, B)    ((A) > (B) ? (A) : (B))
#define _min(A, B)    ((A) < (B) ? (A) : (B))

static int
convert_to_rgba8888 (void *buf, int cap_w, int cap_h, unsigned int fmt)
{
    int x, y;

    if (s_capture_buf == NULL)
    {
        s_capture_buf = (unsigned char *)malloc (cap_w * cap_h * 4);
    }

    if (fmt == v4l2_fourcc ('Y', 'U', 'Y', 'V'))
    {
        unsigned char *src8 = buf;
        unsigned char *dst8 = s_capture_buf;
        for (y = 0; y < cap_h; y ++)
        {
            for (x = 0; x < cap_w; x += 2)
            {
                int y0 = *src8 ++;
                int cb = *src8 ++;
                int y1 = *src8 ++;
                int cr = *src8 ++;

                y0 -= 16;
                y1 -= 16;
                cb -= 128;
                cr -= 128;
                int r, g, b;
                
                r = 1164 * y0 + 1596 * cr;
                g = 1164 * y0 -  392 * cb - 813 * cr;
                b = 1164 * y0 + 2017 * cb;
                
                r = _min (_max (r, 999) / 1000, 255);
                g = _min (_max (g, 999) / 1000, 255);
                b = _min (_max (b, 999) / 1000, 255);
                
                *dst8 ++ = r;
                *dst8 ++ = g;
                *dst8 ++ = b;
                *dst8 ++ = 255;

                r = 1164 * y1 + 1596 * cr;
                g = 1164 * y1 -  392 * cb - 813 * cr;
                b = 1164 * y1 + 2017 * cb;
                
                r = _min (_max (r, 999) / 1000, 255);
                g = _min (_max (g, 999) / 1000, 255);
                b = _min (_max (b, 999) / 1000, 255);
                
                *dst8 ++ = r;
                *dst8 ++ = g;
                *dst8 ++ = b;
                *dst8 ++ = 255;
            }
        }
    }
    else
    {
        fprintf (stderr, "ERR: %s(%d): not supported.\n", __FILE__, __LINE__);
        return -1;
    }
    return 0;
}


static void *
capture_thread_main ()
{
    v4l2_start_capture (s_cap_dev);

    while (1)
    {
        capture_frame_t *frame = v4l2_acquire_capture_frame (s_cap_dev);

        convert_to_rgba8888 (frame->vaddr, s_capture_w, s_capture_h, s_capture_fmt);

        v4l2_release_capture_frame (s_cap_dev, frame);
    }
    return 0;
}


int
init_capture ()
{
    int cap_devid = -1;
    capture_dev_t *cap_dev;
    int cap_w, cap_h;
    unsigned int cap_fmt;

    cap_dev = v4l2_open_capture_device (cap_devid);
    if (cap_dev == NULL)
    {
        fprintf (stderr, "can't open capture device.\n");
        return -1;
    }

    v4l2_get_capture_wh (cap_dev, &cap_w, &cap_h);
    v4l2_get_capture_pixelformat (cap_dev, &cap_fmt);

    v4l2_show_current_capture_settings (cap_dev);

    s_cap_dev     = cap_dev;
    s_capture_fmt = cap_fmt;
    s_capture_w = cap_w;
    s_capture_h = cap_h;

    return 0;
}

int 
get_capture_dimension (int *width, int *height)
{
    *width  = s_capture_w;
    *height = s_capture_h;

    return 0;
}

int 
get_capture_buffer (void ** buf)
{
    *buf = s_capture_buf;
    return 0;
}

int
start_capture ()
{
    pthread_create (&s_capture_thread, NULL, capture_thread_main, NULL);
    return 0;
}

