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
#include <stdlib.h>
#include <xcb/xcb.h>
#include <X11/Xlib-xcb.h>
#include <GLES2/gl2.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#define UNUSED(x) (void)(x)

static xcb_connection_t *s_xcb_conn;
static void (*s_motion_func)(int x, int y) = NULL;
static void (*s_button_func)(int button, int state, int x, int y) = NULL;
static void (*s_key_func)(int key, int state, int x, int y) = NULL;


uint32_t xcb_window_attrib_mask = XCB_CW_EVENT_MASK;
uint32_t xcb_window_attrib_list[] = {
    XCB_EVENT_MASK_BUTTON_PRESS   | XCB_EVENT_MASK_BUTTON_RELEASE |
    XCB_EVENT_MASK_POINTER_MOTION |
    XCB_EVENT_MASK_KEY_PRESS      | XCB_EVENT_MASK_KEY_RELEASE };


void *
winsys_init_native_display (void)
{
    Display *xdpy = XOpenDisplay (NULL);
    if (xdpy == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    xcb_connection_t *connection = XGetXCBConnection (xdpy);
    if (!connection)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    /* Use the XCB event-handling functions. (not the Xlib) */
    XSetEventQueueOwner (xdpy, XCBOwnsEventQueue);

    if (xcb_connection_has_error (connection))
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    s_xcb_conn = connection;

    return (void *)xdpy;
}


void *
winsys_init_native_window (void *dpy, int win_w, int win_h)
{
    UNUSED (dpy);

    const xcb_setup_t   *setup = xcb_get_setup (s_xcb_conn);
    xcb_screen_iterator_t iter = xcb_setup_roots_iterator (setup);

    xcb_screen_t *screen = iter.data;
    if (screen == 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    xcb_window_t window = xcb_generate_id (s_xcb_conn);
    if (window <= 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    xcb_void_cookie_t create_cookie = xcb_create_window_checked(
        s_xcb_conn,
        XCB_COPY_FROM_PARENT,           // depth
        window,
        screen->root,                   // parent window
        0, 0, win_w, win_h,             // x, y, w, h
        0,                              // border width
        XCB_WINDOW_CLASS_INPUT_OUTPUT,  // class
        screen->root_visual,            // visual
        xcb_window_attrib_mask,
        xcb_window_attrib_list);

    xcb_void_cookie_t map_cookie = xcb_map_window_checked (s_xcb_conn, window);

    /* Check errors. */
    if (xcb_request_check (s_xcb_conn, create_cookie))
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    if (xcb_request_check (s_xcb_conn, map_cookie))
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    xcb_flush (s_xcb_conn);

    return (void *)window;
}


int 
winsys_swap()
{
    xcb_generic_event_t *event;

    while ((event = xcb_poll_for_queued_event (s_xcb_conn)))
    {
        switch (event->response_type & ~0x80) 
        {
        case XCB_KEY_PRESS: {
            xcb_key_press_event_t *kp = (xcb_key_press_event_t *)event;
            //fprintf (stderr, "XCB_KEY_PRESS (%d)\n", kp->detail);

            if (s_key_func)
            {
                s_key_func (kp->detail, 1, kp->event_x, kp->event_y);
            }
            break;
        }
        case XCB_KEY_RELEASE: {
            xcb_key_release_event_t *kr = (xcb_key_release_event_t *)event;
            //fprintf (stderr, "XCB_KEY_RELEASE (%d)\n", kr->detail);

            if (s_key_func)
            {
                s_key_func (kr->detail, 0, kr->event_x, kr->event_y);
            }
            break;
        }
        case XCB_BUTTON_PRESS: {
            xcb_button_press_event_t *bp = (xcb_button_press_event_t *)event;
            //fprintf (stderr, "XCB_BUTTON_PRESS (%d,%d)\n", bp->event_x, bp->event_y);

            if (s_button_func)
            {
                s_button_func (0, 1, bp->event_x, bp->event_y);
            }
            break;
        }
        case XCB_BUTTON_RELEASE: {
            xcb_button_release_event_t *br = (xcb_button_release_event_t *)event;
            //fprintf (stderr, "XCB_BUTTON_RELEASE (%d,%d)\n", br->event_x, br->event_y);

            if (s_button_func)
            {
                s_button_func (0, 0, br->event_x, br->event_y);
            }
            break;
        }
        case XCB_MOTION_NOTIFY: {
            xcb_motion_notify_event_t *mn = (xcb_motion_notify_event_t *)event;
            //fprintf (stderr, "XCB_POINTER_MOTION: detail(%d) (%d, %d) stat(%d)\n",
            //            mn->detail, mn->event_x, mn->event_y, mn->state);

            if (s_motion_func)
            {
                s_motion_func (mn->event_x, mn->event_y);
            }
            break;
        }
        default:
            /* Unknown event type, ignore it */
            break;
        }

        free (event);
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



