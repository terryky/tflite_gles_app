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
    static bool show_demo_window = true;
    static bool show_another_window = false;

    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    if (show_demo_window)
        ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {
        static int counter = 0;

        // Create a window called "Hello, world!" and append into it.
        ImGui::Begin("Hello, world!");

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window",    &show_demo_window);   // Edit bools storing our window open/close state
        ImGui::Checkbox("Another Window", &show_another_window);

        // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::SliderFloat("grid alpha", &imgui_data->grid_alpha, 0.0f, 1.0f);

        // Edit 3 floats representing a color
        ImVec4 clear_color;
        clear_color.x = imgui_data->clear_color[0];
        clear_color.y = imgui_data->clear_color[1];
        clear_color.z = imgui_data->clear_color[2];
        clear_color.w = imgui_data->clear_color[3];
        ImGui::ColorEdit3("clear color", (float*)&clear_color);
        imgui_data->clear_color[0] = clear_color.x;
        imgui_data->clear_color[1] = clear_color.y;
        imgui_data->clear_color[2] = clear_color.z;
        imgui_data->clear_color[3] = clear_color.w;

        // Buttons return true when clicked (most widgets return true when edited/activated)
        if (ImGui::Button("Button"))
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGui::End();
    }

    // 3. Show another simple window.
    if (show_another_window)
    {
        ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
        ImGui::Text("Hello from another window!");
        if (ImGui::Button("Close Me"))
            show_another_window = false;
        ImGui::End();
    }
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
