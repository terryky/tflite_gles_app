/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "render_imgui.h"

static int  s_win_w;
static int  s_win_h;

static ImVec2 s_win_size[10];
static ImVec2 s_win_pos [10];
static int    s_win_num = 0;
static ImVec2 s_mouse_pos;


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

    s_win_w = win_w;
    s_win_h = win_h;

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

    s_mouse_pos.x = x;
    s_mouse_pos.y = y;
}

void
imgui_mousemove (int x, int y)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);

    s_mouse_pos.x = x;
    s_mouse_pos.y = y;
}

int
imgui_is_anywindow_hovered ()
{
#if 1
    int x = s_mouse_pos.x;
    int y = s_mouse_pos.y;
    for (int i = 0; i < s_win_num; i ++)
    {
        int x0 = s_win_pos[i].x;
        int y0 = s_win_pos[i].y;
        int x1 = x0 + s_win_size[i].x;
        int y1 = y0 + s_win_size[i].y;
        if ((x >= x0) && (x < x1) && (y >= y0) && (y < y1))
            return 1;
    }
    return 0;
#else
    return
#endif
}

static void
render_gui (imgui_data_t *imgui_data)
{
    int win_w = 300;
    int win_h = 220;
    int win_y = 10;
    s_win_num = 0;

    ImGui::SetNextWindowPos (ImVec2(s_win_w - win_w - 10, win_y), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(win_w,                win_h), ImGuiCond_FirstUseEver);

    ImGui::Begin("config");
    {
        ImGui::SliderFloat("pose_scale_x", &imgui_data->pose_scale_x,   0.0f, 1000.0f);
        ImGui::SliderFloat("pose_scale_y", &imgui_data->pose_scale_y,   0.0f, 1000.0f);
        ImGui::SliderFloat("pose_scale_z", &imgui_data->pose_scale_z,   0.0f, 1000.0f);
        ImGui::SliderFloat("camera_pos_z", &imgui_data->camera_pos_z,   0.0f, 1000.0f);
        ImGui::SliderFloat("joint_radius", &imgui_data->joint_radius,   0.0f, 20.0f);
        ImGui::SliderFloat("bone_radius",  &imgui_data->bone_radius,    0.0f, 20.0f);

        bool draw_axis   = imgui_data->draw_axis;
        bool draw_pmeter = imgui_data->draw_pmeter;
        ImGui::Checkbox("draw_axis",   &draw_axis);
        ImGui::Checkbox("draw_pmeter", &draw_pmeter);
        imgui_data->draw_axis   = draw_axis   ? 1 : 0;
        imgui_data->draw_pmeter = draw_pmeter ? 1 : 0;

        s_win_pos [s_win_num] = ImGui::GetWindowPos  ();
        s_win_size[s_win_num] = ImGui::GetWindowSize ();
        s_win_num ++;
    }
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
