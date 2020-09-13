/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef UTIL_IMGUI_H_
#define UTIL_IMGUI_H_

#include "trt_pose3d.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _imgui_data_t
{
    pose3d_config_t pose3d_config;
    float frame_color[4];
    float pose_scale_x;
    float pose_scale_y;
    float pose_scale_z;
    float camera_pos_z;
    float joint_radius;
    float bone_radius;
    int   draw_axis;
    int   draw_pmeter;
} imgui_data_t;

int  init_imgui (int width, int height);

void imgui_mousebutton (int button, int state, int x, int y);
void imgui_mousemove (int x, int y);
int imgui_is_anywindow_hovered ();

int invoke_imgui (imgui_data_t *imgui_data);

#ifdef __cplusplus
}
#endif
#endif /* UTIL_IMGUI_H_ */
 