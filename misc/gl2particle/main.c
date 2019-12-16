/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <GLES2/gl2.h>
#include "util_egl.h"
#include "assertgl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_debug.h"
#include "util_render2d.h"
#include "util_texture.h"
#include "util_particle.h"

#define UNUSED(x) (void)(x)


particle_system_t   *s_particle[2];



int
init_app (int win_w, int win_h)
{
    particle_system_t *psys;

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

    float color1[][3] = {
        {0.25f, 0.5f, 1.0f},
        {0.25f, 0.6f, 1.0f},
        {0.30f, 0.6f, 1.0f},
        {0.40f, 0.6f, 1.0f},
        {0.70f, 0.8f, 1.0f}};

    float color2[][3] = {
        {1.0f, 0.5f, 0.25f},
        {1.0f, 0.6f, 0.25f},
        {1.0f, 0.6f, 0.30f},
        {1.0f, 0.6f, 0.40f},
        {1.0f, 0.8f, 0.70f}};

    psys = create_particle_system (5);
    add_particle_set (psys, 0, "particle_1.png", 100, color1[0]);
    add_particle_set (psys, 1, "particle_2.png", 100, color1[1]);
    add_particle_set (psys, 2, "particle_3.png", 100, color1[2]);
  add_particle_set (psys, 3, "particle_4.png", 100, color1[3]);
  add_particle_set (psys, 4, "particle_5.png", 100, color1[4]);
    s_particle[0] = psys;

    psys = create_particle_system (5);
    add_particle_set (psys, 0, "particle_1.png", 100, color2[0]);
    add_particle_set (psys, 1, "particle_2.png", 100, color2[1]);
    add_particle_set (psys, 2, "particle_3.png", 100, color2[2]);
  add_particle_set (psys, 3, "particle_4.png", 100, color2[3]);
  add_particle_set (psys, 4, "particle_5.png", 100, color2[4]);
    s_particle[1] = psys;

    return 0;
}


static int
render (int win_w, int win_h)
{
    float x, y;
    static float s_rad = 0.0f;

    s_rad += 0.02f;

    x = (cosf(s_rad * 1.0f) + 1.0f) * 0.5f * win_w;
    y = (sinf(s_rad * 1.0f) + 1.0f) * 0.5f * win_h;
    update_particle (s_particle[0], x, y);

    x = (cosf(s_rad * 1.0f) + 1.0f) * 0.5f * win_w;
    y = (sinf(s_rad * 2.0f) + 1.0f) * 0.5f * win_h;
    update_particle (s_particle[1], x, y);


    render_particle (s_particle[0]);
    render_particle (s_particle[1]);

    return 0;
}

int main(int argc, char *argv[])
{
    int win_w = 960;
    int win_h = 540;
    int count;
    double ttime0 = 0, ttime1 = 0, interval;
    char strbuf[512];
    UNUSED (argc);
    UNUSED (argv);

    if (egl_init_with_platform_window_surface (2, 0, 0, 0, win_w, win_h) < 0)
        exit (-1);

    init_app (win_w, win_h);

    for (count = 0; ; count ++)
    {
        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime1 = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime1 - ttime0 : 0;
        ttime0 = ttime1;

        glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
        glClear (GL_COLOR_BUFFER_BIT);

        render (win_w, win_h);

        draw_pmeter (0, 40);

        sprintf (strbuf, "%.1f [ms]\n", interval);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

