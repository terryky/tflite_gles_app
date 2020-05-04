/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <GLES2/gl2.h>
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "teapot.h"

#define UNUSED(x) (void)(x)

/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
int
main(int argc, char *argv[])
{
    int count;
    int win_w = 960;
    int win_h = 540;
    float col[3];
    double ttime0 = 0, ttime1 = 0, interval;
    char strbuf[512];
    UNUSED (argc);
    UNUSED (*argv);

    egl_init_with_platform_window_surface (2, 24, 0, 0, win_w, win_h);

    init_teapot ((float)win_w / (float)win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

    glClearColor (0.7f, 0.7f, 0.7f, 1.0f);

    srand (time (NULL));
    col[0] = (rand () % 255) / 255.0f;
    col[1] = (rand () % 255) / 255.0f;
    col[2] = (rand () % 255) / 255.0f;
    for (count = 0; ; count ++)
    {
        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime1 = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime1 - ttime0 : 0;
        ttime0 = ttime1;

        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        draw_teapot (count, col);
        draw_pmeter (0, 40);

        sprintf (strbuf, "%.1f [ms]\n", interval);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

