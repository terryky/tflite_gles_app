/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "render_imgui.h"

int
init_imgui (int win_w, int win_h)
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplOpenGL3_Init(NULL);

    io.DisplaySize = ImVec2 ((float)win_w, (float)win_h);

    return 0;
}

void
imgui_mousebutton (int button, int state, int x, int y)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);

    if (state)
        io.MouseDown[button] = true;
    else
        io.MouseDown[button] = false;
}

void
imgui_mousemove (int x, int y)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);
}

static void
render_gui (imgui_data_t *imgui_data)
{
    ImGui::Begin("Blazeface config");

    ImGui::SliderFloat("Score thresh", &imgui_data->blazeface_config.score_thresh, 0.0f, 1.0f);
    ImGui::SliderFloat("IOU   thresh", &imgui_data->blazeface_config.iou_thresh,   0.0f, 1.0f);

    ImVec4 frame_color;
    frame_color.x = imgui_data->frame_color[0];
    frame_color.y = imgui_data->frame_color[1];
    frame_color.z = imgui_data->frame_color[2];
    frame_color.w = imgui_data->frame_color[3];
    ImGui::ColorEdit3("Frame color", (float*)&frame_color);
    imgui_data->frame_color[0] = frame_color.x;
    imgui_data->frame_color[1] = frame_color.y;
    imgui_data->frame_color[2] = frame_color.z;
    imgui_data->frame_color[3] = frame_color.w;

    ImGui::End();
}

int
invoke_imgui (imgui_data_t *imgui_data)
{
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    render_gui (imgui_data);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return 0;
}
