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
init_posenet_particle (int win_w, int win_h)
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

    int i;
    for (i = 0; i < 5; i ++)
    {
        //color1[i][0] *= 0.2f;
        //color1[i][1] *= 0.2f;
        //color1[i][2] *= 0.1f;
    }

    /* particle images for RightWrist */
    psys = create_particle_system (5);
    add_particle_set (psys, 0, "particle/particle_1.png", 100, color1[0]);
    add_particle_set (psys, 1, "particle/particle_2.png", 100, color1[1]);
    add_particle_set (psys, 2, "particle/particle_3.png", 100, color1[2]);
//  add_particle_set (psys, 3, "particle/particle_4.png", 100, color1[3]);
//  add_particle_set (psys, 4, "particle/particle_5.png", 100, color1[4]);
    s_particle[0] = psys;

    /* particle images for LeftWrist */
    psys = create_particle_system (5);
    add_particle_set (psys, 0, "particle/particle_1.png", 100, color2[0]);
    add_particle_set (psys, 1, "particle/particle_2.png", 100, color2[1]);
    add_particle_set (psys, 2, "particle/particle_3.png", 100, color2[2]);
//  add_particle_set (psys, 3, "particle/particle_4.png", 100, color2[3]);
//  add_particle_set (psys, 4, "particle/particle_5.png", 100, color2[4]);
    s_particle[1] = psys;

    return 0;
}


int
render_posenet_particle (float rx, float ry, float lx, float ly)
{
    update_particle (s_particle[0], rx, ry);    /* right wrist */
    update_particle (s_particle[1], lx, ly);    /* left  wrist */

    render_particle (s_particle[0]);
    render_particle (s_particle[1]);

    return 0;
}

