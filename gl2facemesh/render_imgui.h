/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef UTIL_IMGUI_H_
#define UTIL_IMGUI_H_

#include "tflite_facemesh.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _imgui_data_t
{
    float mask_alpha;
    int   mask_eye_hole;
    int   draw_mesh_line;
    int   draw_detect_rect;
    int   draw_pmeter;

    int   mask_num;
    int   cur_mask_id;
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
 