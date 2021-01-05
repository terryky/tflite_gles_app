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
#include "util_matrix.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_debug.h"
#include "util_render2d.h"
#include "util_render_target.h"
#include "filter_normal.h"
#include "filter_gaussian.h"
#include "cube.h"

#define UNUSED(x) (void)(x)

static render_target_t s_fb;
static render_target_t s_fbo;
static render_target_t s_fbo_scale;
static render_target_t s_fbo_blur;


int
init_app (int win_w, int win_h)
{
    int ret;

    init_cube ((float)win_w / (float)win_h);

    ret = create_render_target (&s_fb, win_w, win_h, 0);
    DBG_ASSERT (ret == 0, "failed to create fbo");

    ret = create_render_target (&s_fbo, win_w, win_h, RTARGET_COLOR);
    DBG_ASSERT (ret == 0, "failed to create fbo");

    ret = create_render_target (&s_fbo_scale, win_w/4, win_h/4, RTARGET_COLOR);
    DBG_ASSERT (ret == 0, "failed to create fbo");

    ret = create_render_target (&s_fbo_blur, win_w/4, win_h/4, RTARGET_COLOR);
    DBG_ASSERT (ret == 0, "failed to create fbo");

    /* non effect filter (just scale)*/
    ret = init_normal_filter ();
    DBG_ASSERT (ret == 0, "failed to create filter");

    /* gaussian blur filter */
    ret = init_gaussian_blur_filter (3.0f);
    DBG_ASSERT (ret == 0, "failed to create filter");

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

    return 0;
}


static int
render_to_fbo(int win_w, int win_h)
{
    static int s_x[2] = {100, 100};
    static int s_y[2] = {100, 100};
    static int s_dx[2] = {2, -2};
    static int s_dy[2] = {2,  2};
    float color0[] = {1.0, 1.0, 1.0, 1.0};
    float colorR[] = {1.0, 0.0, 0.0, 1.0};
    float colorB[] = {0.0, 0.0, 1.0, 1.0};
    static int s_isfirst = 1;
    int i;

    GLASSERT();    
    set_render_target (&s_fbo);

    if (s_isfirst)
    {
        s_x[1] = win_w - 100;

        glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
        glClear (GL_COLOR_BUFFER_BIT);
        s_isfirst = 0;
    }
    else
    {
        float col[] = {0.0, 0.0, 0.0, 0.1};
        draw_2d_fillrect (0, 0, win_w, win_h, col);
    }

    draw_2d_line (100,         100, s_x[0], s_y[0], colorB, 16.0f);
    draw_2d_line (win_w - 100, 100, s_x[1], s_y[1], colorR, 16.0f);

    draw_2d_line (100,         100, s_x[0], s_y[0], color0,  6.0f);
    draw_2d_line (win_w - 100, 100, s_x[1], s_y[1], color0,  6.0f);
    GLASSERT();

    for (i = 0; i < 2; i ++)
    {
        s_x[i] += s_dx[i];
        s_y[i] += s_dy[i];
        
        if (s_x[i] >= win_w) s_dx[i] *= -1;
        if (s_x[i]  <     0) s_dx[i] *= -1;
        if (s_y[i] >= win_h) s_dy[i] *= -1;
        if (s_y[i]  <     0) s_dy[i] *= -1;
    }

    apply_normal_filter (&s_fbo_scale, &s_fbo);
    apply_gaussian_filter (&s_fbo_blur, &s_fbo_scale);
    GLASSERT();

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

    if (egl_init_with_platform_window_surface (2, 24, 0, 0, win_w, win_h) < 0)
        exit (-1);

    init_app (win_w, win_h);

    for (count = 0; ; count ++)
    {
        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime1 = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime1 - ttime0 : 0;
        ttime0 = ttime1;

        render_to_fbo (win_w, win_h);

        set_render_target (&s_fb);

        glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        draw_cube (count);

        {
            //unsigned int blend_src  [] = {GL_ONE, GL_ZERO, GL_ONE, GL_ZERO};
            //unsigned int blend_blend[] = {GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA};
            unsigned int blend_add  [] = {GL_ONE, GL_ONE, GL_ZERO, GL_ONE};
            glDisable (GL_DEPTH_TEST);
            draw_2d_texture_blendfunc (s_fbo.texc_id,      0, 0, win_w, win_h, 0, blend_add);
            draw_2d_texture_blendfunc (s_fbo_blur.texc_id, 0, 0, win_w, win_h, 0, blend_add);
        }

        draw_pmeter (0, 40);

        sprintf (strbuf, "%.1f [ms]\n", interval);
        draw_dbgstr (strbuf, 10, 10);

        egl_swap();
    }

    return 0;
}

