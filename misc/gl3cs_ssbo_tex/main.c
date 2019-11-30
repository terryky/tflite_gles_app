/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <GLES3/gl31.h>
#include "util_shader.h"
#include "util_egl.h"
#include "util_texture.h"
#include "assertgl.h"

#define UNUSED(x) (void)(x)

static GLfloat s_vtx[] =
{
    -1.0f, 1.0f,        // +----+
    -1.0f,-1.0f,        // |   /|
     1.0f, 1.0f,        // | /  |
     1.0f,-1.0f,        // +----+
};


static char s_strVS[] =
    "#version 310 es                                    \n"
    "in vec4 a_Vertex;                                  \n"
    "                                                   \n"
    "void main (void)                                   \n"
    "{                                                  \n"
    "    gl_Position = a_Vertex;                        \n"
    "}                                                  \n";

static char s_strFS[] =
    "#version 310 es                                    \n"
    "precision mediump float;                           \n"
    "out vec4 oColor;                                   \n"
    "layout(location = 0) uniform ivec2 u_imgsize;      \n"
    "                                                   \n"
    "layout(std430) buffer;                             \n"
    "layout(binding = 1) buffer Input {                 \n"
    "    float elements[];                              \n"
    "} input_data;                                      \n"
    "                                                   \n"
    "void main()                                        \n"
    "{                                                  \n"
    "    int img_w = u_imgsize.x;                       \n"
    "    int img_h = u_imgsize.y;                       \n"
    "    ivec2 pos = ivec2(gl_FragCoord);               \n"
    "    if (pos.x >= img_w || pos.y >= img_h)          \n"
    "    {                                              \n"
    "        oColor = vec4(0.5, 0.5, 0.5, 1.0);         \n"
    "        return;                                    \n"
    "    }                                              \n"
    "                                                   \n"
    "    int idx = 3 * (pos.y * img_w + pos.x);         \n"
    "    float r = input_data.elements[idx + 0];        \n"
    "    float g = input_data.elements[idx + 1];        \n"
    "    float b = input_data.elements[idx + 2];        \n"
    "    oColor = vec4(r, g , b, 1.0);                  \n"
    "}                                                  \n";

/*
 * ComputeShader to convert GL Texture ==> SSBO
 */
static char s_strCS[] =
    "#version 310 es                                    \n"
    "                                                   \n"
    "layout(local_size_x = 16, local_size_y = 16) in;   \n"
    "layout(location = 0) uniform sampler2D u_sampler;  \n"
    "layout(location = 1) uniform ivec2 u_imgsize;      \n"
    "                                                   \n"
    "layout(std430) buffer;                             \n"
    "layout(binding = 1) buffer Output {                \n"
    "    float elements[];                              \n"
    "} output_data;                                     \n"
    "                                                   \n"
    "void main() {                                      \n"
    "    int img_w = u_imgsize.x;                       \n"
    "    int img_h = u_imgsize.y;                       \n"
    "    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);   \n"
    "    if (gid.x >= img_w || gid.y >= img_h)          \n"
    "        return;                                    \n"
    "                                                   \n"
    "    vec2 coord;                                    \n"
    "    coord.x = float(gid.x) / float(img_w);         \n"
    "    coord.y = float(gid.y) / float(img_h);         \n"
    "    coord.y = 1.0 - coord.y; /* upside down */     \n"
    "    vec3 pixel = texture(u_sampler, coord).xyz;    \n"
    "                                                   \n"
    "    int idx = 3 * (gid.y * img_w + gid.x);         \n"
    "    output_data.elements[idx + 0] = pixel.x;       \n"
    "    output_data.elements[idx + 1] = pixel.y;       \n"
    "    output_data.elements[idx + 2] = pixel.z;       \n"
    "}                                                  \n";



static int
init_ssbo (int ssbo_bufsize)
{
    GLuint ssboid;

    glGenBuffers (1, &ssboid);
    glBindBuffer (GL_SHADER_STORAGE_BUFFER, ssboid);
    glBufferData (GL_SHADER_STORAGE_BUFFER, ssbo_bufsize, NULL, GL_STREAM_COPY);
    glBindBuffer (GL_SHADER_STORAGE_BUFFER, 0);
    GLASSERT();

    return ssboid;
}

/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
int
main(int argc, char *argv[])
{
    char input_name_default[] = "monoscope.jpg";
    char *input_name = input_name_default;
    int win_w = 960;
    int win_h = 540;
    UNUSED (argc);
    UNUSED (*argv);

    egl_init_with_platform_window_surface (3, 0, 0, 0, win_w, win_h);

    /* build compute shader to resize texture. */
    int progCS = build_compute_shader (s_strCS);
    int loc_cs_tex = glGetUniformLocation(progCS, "u_sampler");
    int loc_cs_imgsize = glGetUniformLocation (progCS, "u_imgsize");

    /* build render shader to visualize resized texture in SSBO. */
    int progDraw = build_shader (s_strVS, s_strFS);
    int loc_vtx = glGetAttribLocation (progDraw, "a_Vertex");
    int loc_imgsize = glGetUniformLocation (progDraw, "u_imgsize");
    GLASSERT();

    /* load source texture image. */
    int texid, texw, texh;
    load_jpg_texture (input_name, &texid, &texw, &texh);

    /* allocate SSBO buffer. */
    int ssbo_bufsize = texw * texh * 3 * 4;
    int ssboid = init_ssbo (ssbo_bufsize);
    GLASSERT();


    float scale_x = 1.0f;
    float scale_y = 1.0f;
    float dx = -0.01;
    float dy = -0.02;
    for (int i = 0; ; i ++)
    {
        scale_x += dx;
        scale_y += dy;

        if (scale_x <= 0.0 || scale_x >= 1.0f)
            dx *= -1.0f;

        if (scale_y <= 0.0 || scale_y >= 1.0f)
            dy *= -1.0f;

        int resize_w = texw * scale_x;
        int resize_h = texh * scale_y;
        int ssbo_range = resize_w * resize_h * 3 * 4;

        /* ------------------------------------ *
         *  resize texture and write to SSBO.
         * ------------------------------------ */
        glUseProgram (progCS);

        glActiveTexture (GL_TEXTURE0);
        glBindTexture (GL_TEXTURE_2D, texid);
        glUniform1i (loc_cs_tex, 0);
        glUniform2i (loc_cs_imgsize, resize_w, resize_h);
        glBindBufferRange (GL_SHADER_STORAGE_BUFFER, 1, ssboid, 0, ssbo_range);

        int group_size = 16;
        int num_group_x = (int)ceil((float)resize_w / (float)group_size);
        int num_group_y = (int)ceil((float)resize_h / (float)group_size);
        glDispatchCompute (num_group_x, num_group_y, 1);

        glBindBuffer (GL_SHADER_STORAGE_BUFFER, 0);
        glBindTexture (GL_TEXTURE_2D, 0);


        /* ------------------------------------ *
         *  render the resized texture in SSBO.
         * ------------------------------------ */
        glClearColor (0.5f, 0.5f, 0.5f, 1.0f);
        glClear (GL_COLOR_BUFFER_BIT);

        glUseProgram (progDraw);
        glUniform2i (loc_imgsize, resize_w, resize_h);
        glBindBufferRange (GL_SHADER_STORAGE_BUFFER, 1, ssboid, 0, ssbo_range);

        glEnableVertexAttribArray (loc_vtx);
        glVertexAttribPointer (loc_vtx, 2, GL_FLOAT, GL_FALSE, 0, s_vtx);

        glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);

        glBindBuffer (GL_SHADER_STORAGE_BUFFER, 0);

        egl_swap();
    }

    return 0;
}

