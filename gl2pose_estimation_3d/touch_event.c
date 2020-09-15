/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <string.h>
#include <math.h>
#include "util_matrix.h"

static int   s_mouse_down = 0;
static float s_mouse_pos [2] = {0};
static float s_mouse_pos0[2] = {0};
static float s_mdl_qtn [4];
static float s_mdl_qtn0[4];
static float s_mdl_mtx[16];

static int s_win_width  = 100;
static int s_win_height = 100;

void
touch_event_start (int id, int x, int y)
{
    s_mouse_down = 1;

    s_mouse_pos0[0] = s_mouse_pos[0] = x;
    s_mouse_pos0[1] = s_mouse_pos[1] = y;
    quaternion_copy (s_mdl_qtn0, s_mdl_qtn);
}

void
touch_event_end (int id)
{
    s_mouse_down = 0;
}

void
touch_event_move (int id, int x, int y)
{
    if (s_mouse_down)
    {
        float dx = x - s_mouse_pos0[0];
        float dy = y - s_mouse_pos0[1];

        float axis[3];
        axis[0] = 2 * M_PI * dy / s_win_height;
        axis[1] = 2 * M_PI * dx / s_win_width;
        axis[2] = 0;

        float rot = vec3_normalize (axis);

        float dqtn[4];
        quaternion_rotate (dqtn, rot, axis[0], axis[1], axis[2]);
        quaternion_mult (s_mdl_qtn, dqtn, s_mdl_qtn0);

        s_mouse_pos[0] = x;
        s_mouse_pos[1] = y;
    }
}


int
init_touch_event (int width, int height)
{
    quaternion_identity (s_mdl_qtn);
    quaternion_to_matrix (s_mdl_mtx, s_mdl_qtn);

    s_win_width  = width;
    s_win_height = height;
    return 0;
}

int
get_touch_event_matrix (float *mtx)
{
    quaternion_to_matrix (s_mdl_mtx, s_mdl_qtn);
    memcpy (mtx, s_mdl_mtx, sizeof (s_mdl_mtx));
    return 0;
}
