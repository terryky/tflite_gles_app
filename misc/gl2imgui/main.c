/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <GLES2/gl2.h>
#include "util_egl.h"
#include "util_debugstr.h"
#include "util_pmeter.h"
#include "util_render2d.h"
#include "render_imgui.h"
#include "winsys.h"

#define UNUSED(x) (void)(x)

typedef struct _ivec2
{
    int x, y;
} ivec2;

static ivec2 s_mouse_pos  = {0};
static ivec2 s_mouse_pos0 = {0};
static int s_mouse_stat   = 0;
static int s_key_code     = 0;
static int s_key_stat     = 0;

#define MAX_TRAJECTORY_LENGTH   1024
static ivec2 s_mouse_trajectory[MAX_TRAJECTORY_LENGTH];
static int   s_mouse_trajectory_len = 0;

void
add_mouse_trajectory (int x, int y)
{
    s_mouse_trajectory[s_mouse_trajectory_len].x = x;
    s_mouse_trajectory[s_mouse_trajectory_len].y = y;
    s_mouse_trajectory_len ++;
    if (s_mouse_trajectory_len >= MAX_TRAJECTORY_LENGTH)
        s_mouse_trajectory_len = 0;
}

void
mousemove_cb (int x, int y)
{
    imgui_mousemove (x, y);

    s_mouse_pos.x = x;
    s_mouse_pos.y = y;

    if (s_mouse_stat)
        add_mouse_trajectory (x, y);
}

void
button_cb (int button, int state, int x, int y)
{
    imgui_mousebutton (button, state, x, y);

    if (state)
    {
        s_mouse_pos0.x = x;
        s_mouse_pos0.y = y;
        s_mouse_trajectory_len = 0;
        add_mouse_trajectory (x, y);
    }
    s_mouse_stat = state;
}

void
keyboard_cb (int key, int state, int x, int y)
{
    s_key_code = key;
    s_key_stat = state;
}

void
draw_trajectory ()
{
    float col_black[] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (int i = 1; i < s_mouse_trajectory_len; i ++)
    {
        float x0 = s_mouse_trajectory[i - 1].x;
        float y0 = s_mouse_trajectory[i - 1].y;
        float x1 = s_mouse_trajectory[i    ].x;
        float y1 = s_mouse_trajectory[i    ].y;
        draw_2d_line (x0, y0, x1, y1, col_black, 3.0f);
    }
}


void
draw_grid (int win_w, int win_h, float alpha)
{
    float col_gray[]   = {0.73f, 0.75f, 0.75f, alpha};
    float col_blue[]   = {0.00f, 0.00f, 1.00f, alpha};
    float col_red []   = {1.00f, 0.00f, 0.00f, alpha};
    float *col;

    col = col_gray;
    for (int y = 0; y < win_h; y += 10)
        draw_2d_line (0, y, win_w, y, col, 1.0f);

    for (int x = 0; x < win_w; x += 10)
        draw_2d_line (x, 0, x, win_h, col, 1.0f);

    col = col_blue;
    for (int y = 0; y < win_h; y += 100)
        draw_2d_line (0, y, win_w, y, col, 1.0f);

    for (int x = 0; x < win_w; x += 100)
        draw_2d_line (x, 0, x, win_h, col, 1.0f);
    
    col = col_red;
    for (int y = 0; y < win_h; y += 500)
        draw_2d_line (0, y, win_w, y, col, 1.0f);

    for (int x = 0; x < win_w; x += 500)
        draw_2d_line (x, 0, x, win_h, col, 1.0f);
}

void
draw_cross_hair (int win_w, int win_h)
{
    float col_lime[]   = {0.0f, 1.0f, 0.3f, 1.0f};
    float x0, y0, x1, y1;

    x0 = 0.0f; 
    y0 = s_mouse_pos.y;
    x1 = win_w;
    y1 = s_mouse_pos.y;
    draw_2d_line (x0, y0, x1, y1, col_lime, 1.0f);

    x0 = s_mouse_pos.x; 
    y0 = 0.0f;
    x1 = s_mouse_pos.x;
    y1 = win_h;
    draw_2d_line (x0, y0, x1, y1, col_lime, 1.0f);
}


/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
int
main(int argc, char *argv[])
{
    int count;
    int win_w = 960;
    int win_h = 960;
    double ttime0 = 0, ttime1 = 0, interval;
    char strbuf[512];
    imgui_data_t imgui_data = {0};
    UNUSED (argc);
    UNUSED (*argv);

    egl_init_with_platform_window_surface (2, 24, 0, 0, win_w, win_h);
    egl_set_motion_func (mousemove_cb);
    egl_set_button_func (button_cb);
    egl_set_key_func    (keyboard_cb);

    init_2d_renderer (win_w, win_h);
    init_pmeter (win_w, win_h, 500);
    init_dbgstr (win_w, win_h);

    imgui_data.clear_color[0] = 0.7f;
    imgui_data.clear_color[1] = 0.7f;
    imgui_data.clear_color[2] = 0.7f;
    imgui_data.clear_color[3] = 1.0f;
    imgui_data.grid_alpha = 1.0f;

    init_imgui (win_w, win_h);

    for (count = 0; ; count ++)
    {
        PMETER_RESET_LAP ();
        PMETER_SET_LAP ();

        ttime1 = pmeter_get_time_ms ();
        interval = (count > 0) ? ttime1 - ttime0 : 0;
        ttime0 = ttime1;

        float *col = imgui_data.clear_color;
        glClearColor (col[0], col[1], col[2], col[3]);
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        draw_grid (win_w, win_h, imgui_data.grid_alpha);
        draw_trajectory ();
        draw_cross_hair (win_w, win_h);

        draw_pmeter (0, 40);

        sprintf (strbuf, "[%d](%d, %d)", s_mouse_stat, s_mouse_pos.x, s_mouse_pos.y);
        draw_dbgstr (strbuf, s_mouse_pos.x, s_mouse_pos.y);

        sprintf (strbuf, "[%d](%d)", s_key_stat, s_key_code);
        draw_dbgstr (strbuf, 0, 0);

        sprintf (strbuf, "%.1f [ms]\n", interval);
        draw_dbgstr (strbuf, 10, 10);

        invoke_imgui (&imgui_data);

        egl_swap();
    }

    return 0;
}

