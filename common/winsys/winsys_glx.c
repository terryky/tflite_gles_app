/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */

/*
    ## Prepare Host PC (Windows)
    - Install VcXsrv
    - Launch VcXsrv with parameters below:
    -- disable "Native opengl"
    -- enable  "Disable access control"

    ## On Headless Target device
    ```
        $ sudo apt install mesa-utils
        $ sudo apt install libgles2-mesa-dev libegl1-mesa-dev xorg-dev
        $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/mesa/libGL.so.1.2.0
        $ export DISPLAY=192.168.1.100:0.0   # specify the IP of your HostPC 
        $ make
        $ ./glxtri
    ```
*/

#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>

#define GL_GLEXT_PROTOTYPES  (1)
#define GLX_GLXEXT_PROTOTYPES (1)
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glxext.h>

#define UNUSED(x) (void)(x)

static Display *s_xdpy;
static Window  s_xwin;

int
glx_initialize (int glx_version, int depth_size, int stencil_size, int sample_num,
                int win_w, int win_h)
{
    UNUSED (glx_version);
    UNUSED (sample_num);
    int dblBuf[] = { GLX_RGBA,
        GLX_RED_SIZE,   1,
        GLX_GREEN_SIZE, 1,
        GLX_BLUE_SIZE,  1,
        GLX_DEPTH_SIZE, 1,
        GLX_DOUBLEBUFFER,
        None };
    
    /* Open a connection to the X server */
    Display *xdpy = XOpenDisplay (NULL);
    if (xdpy == NULL)
    {
        fprintf (stderr, "Can't open XDisplay.\n");
        return -1;
    }

    /* Make sure OpenGL's GLX extension is supported */
    int ebase;
    if (!glXQueryExtension (xdpy, &ebase, &ebase))
    {
        fprintf (stderr, "X server has no OpenGL GLX extension\n");
        return -1;
    }

    /* Find an appropriate OpenGL-capable visual. */
    XVisualInfo *visual = glXChooseVisual(xdpy, DefaultScreen(xdpy), dblBuf);
    if (visual == NULL)
    {
        fprintf (stderr, "X server has no RGB visual\n");
        return -1;
    }
    if (visual->class != TrueColor)
    {
        fprintf (stderr, "TrueColor visual required for this program\n");
        return -1;
    }

    /* Create an OpenGL rendering context. */
    GLXContext ctx = glXCreateContext (xdpy, visual, None, True);
    if (ctx == NULL)
    {
        fprintf (stderr, "could not create rendering context");
        return -1;
    }

    /* Create an X window with the selected visual. */
    Colormap cmap = XCreateColormap (xdpy, RootWindow (xdpy, visual->screen), visual->visual, AllocNone);

    XSetWindowAttributes swa;
    swa.colormap     = cmap;
    swa.border_pixel = 0;
    swa.event_mask   = ExposureMask | ButtonPressMask | StructureNotifyMask;

    Window xwin = XCreateWindow (xdpy, RootWindow (xdpy, visual->screen), 
        0, 0, win_w, win_h,
        0, visual->depth, InputOutput, visual->visual,
        CWBorderPixel | CWColormap | CWEventMask, &swa);

    glXMakeCurrent (xdpy, xwin, ctx);
    XMapWindow (xdpy, xwin);

    s_xdpy = xdpy;
    s_xwin = xwin;

    if (glGetString(GL_VERSION) == NULL)
    {
        fprintf (stderr, "\n");
        fprintf (stderr, "Failed to initialize GLX.\n");
        fprintf (stderr, "Please retry after setting the environment variables:\n");
        fprintf (stderr, "  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/mesa/libGL.so.1.2.0\n");
        fprintf (stderr, "  $ export DISPLAY=192.168.1.100:0.0   (specify your IP address)\n");
        fprintf (stderr, "\n");
        exit (-1);
    }

    printf ("---------------------------------------\n");
    printf ("GL_RENDERER   = %s\n", (char *) glGetString(GL_RENDERER));
    printf ("GL_VERSION    = %s\n", (char *) glGetString(GL_VERSION));
    printf ("GL_VENDOR     = %s\n", (char *) glGetString(GL_VENDOR));
    //printf ("GL_EXTENSIONS = %s\n", (char *) glGetString(GL_EXTENSIONS));
    printf ("---------------------------------------\n");

    return 0;
}

int
glx_terminate ()
{
    return 0;
}

int
glx_swap ()
{
    glXSwapBuffers (s_xdpy, s_xwin);
    return 0;
}


void egl_set_motion_func (void (*func)(int x, int y))
{
}

void egl_set_button_func (void (*func)(int button, int state, int x, int y))
{
}

void egl_set_key_func (void (*func)(int key, int state, int x, int y))
{
}


#if 1 /* work around for NVIDIA Jetson */
int
drmSyncobjTimelineSignal (int fd, const uint32_t *handles, uint64_t *points, uint32_t handle_count)
{
    return 0;
}

int
drmSyncobjTransfer (int fd, uint32_t dst_handle, uint64_t dst_point, uint32_t src_handle, uint64_t src_point, uint32_t flags)
{
    return 0;
}

int
drmSyncobjTimelineWait (int fd, uint32_t *handles, uint64_t *points, unsigned num_handles, int64_t timeout_nsec, unsigned flags, uint32_t *first_signaled)
{
    return 0;
}
int
drmSyncobjQuery(int fd, uint32_t *handles, uint64_t *points,  uint32_t handle_count)
{
    return 0;
}
int
drmSyncobjQuery2(int fd, uint32_t *handles, uint64_t *points, uint32_t handle_count, uint32_t flags)
{
    return 0;
}
#endif

