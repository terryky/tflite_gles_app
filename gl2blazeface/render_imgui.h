/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef UTIL_IMGUI_H_
#define UTIL_IMGUI_H_

#include "tflite_blazeface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _imgui_data_t
{
    blazeface_config_t blazeface_config;
    float frame_color[4];
} imgui_data_t;

int  init_imgui (int width, int height);
void imgui_mousebutton (int button, int state, int x, int y);
void imgui_mousemove (int x, int y);

int invoke_imgui (imgui_data_t *imgui_data);

#ifdef __cplusplus
}
#endif
#endif /* UTIL_IMGUI_H_ */
 