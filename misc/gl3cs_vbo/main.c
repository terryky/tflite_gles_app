/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <GLES3/gl31.h>
#include "util_shader.h"
#include "util_egl.h"
#include "assertgl.h"

#define UNUSED(x) (void)(x)

static char s_strVS[] =
    "attribute    vec4    a_Vertex;                     \n"
    "attribute    vec4    a_Color;                      \n"
    "varying      vec4    v_color;                      \n"
    "                                                   \n"
    "void main (void)                                   \n"
    "{                                                  \n"
    "    gl_Position = a_Vertex;                        \n"
    "    v_color     = a_Color;                         \n"
    "}                                                  \n";

static char s_strFS[] =
    "precision mediump float;                           \n"
    "varying     vec4     v_color;                      \n"
    "                                                   \n"
    "void main (void)                                   \n"
    "{                                                  \n"
    "    gl_FragColor = v_color;                        \n"
    "}                                                  \n";


static char s_strCS[] =
    "#version 310 es                                    \n"
    "uniform float radius;                              \n"
    "                                                   \n"
    "struct AttribData                                  \n"
    "{                                                  \n"
    "   vec4 v;                                         \n"
    "   vec4 c;                                         \n"
    "};                                                 \n"
    "                                                   \n"
    "layout(std430, binding = 0) buffer destBuffer      \n"
    "{                                                  \n"
    "   AttribData data[];                              \n"
    "} outBuffer;                                       \n"
    "                                                   \n"
    "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; \n"
    "                                                   \n"
    "void main()                                        \n"
    "{                                                  \n"
    "   uint id = gl_GlobalInvocationID.x;              \n"
    "   uint gSize = gl_WorkGroupSize.x * gl_NumWorkGroups.x; \n"
    "   float p = float(id) / float(gSize);             \n"
    "   float rad = 2.0 * 3.14159265359 * p;            \n"
    "                                                   \n"
    "   outBuffer.data[id].v.x = sin(rad) * radius;     \n"
    "   outBuffer.data[id].v.y = cos(rad) * radius;     \n"
    "   outBuffer.data[id].v.z = 0.0;                   \n"
    "   outBuffer.data[id].v.w = 1.0;                   \n"
    "                                                   \n"
    "   outBuffer.data[id].c.r = p;                     \n"
    "   outBuffer.data[id].c.g = 0.0;                   \n"
    "   outBuffer.data[id].c.b = 1.0;                   \n"
    "   outBuffer.data[id].c.a = 1.0;                   \n"
    "}                                                  \n";




/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
#define NUM_VERTS 512
#define GROUP_SIZE_WIDTH 64

int
main(int argc, char *argv[])
{
    int i;
    int win_w = 960;
    int win_h = 540;
    GLuint progCS, progDraw;
    GLuint vbo;
    UNUSED (argc);
    UNUSED (*argv);

    egl_init_with_platform_window_surface (2, 0, 0, 0, win_w, win_h);

    /* build render shader */
    progDraw = build_shader (s_strVS, s_strFS);
    int loc_vtx = glGetAttribLocation (progDraw, "a_Vertex");
    int loc_clr = glGetAttribLocation (progDraw, "a_Color");
    GLASSERT();

    /* build compute shader */
    progCS = build_compute_shader (s_strCS);
    int loc_roll = glGetUniformLocation(progCS, "radius");
    GLASSERT();

    glGenBuffers (1, &vbo);
    glBindBuffer (GL_ARRAY_BUFFER, vbo);
    glBufferData (GL_ARRAY_BUFFER, NUM_VERTS * 2 * 4 * 4, NULL, GL_STATIC_DRAW);
    GLASSERT();

    for (i = 0; ; i ++)
    {
        float radius = (float)(i % 1000) * 0.001;

        /* Compute */
        glUseProgram (progCS);
        glUniform1f (loc_roll, radius);

        int gIndexBufferBinding = 0;
        glBindBufferBase (GL_SHADER_STORAGE_BUFFER, gIndexBufferBinding, vbo);

        glDispatchCompute (NUM_VERTS / GROUP_SIZE_WIDTH, 1, 1);

        glBindBufferBase (GL_SHADER_STORAGE_BUFFER, gIndexBufferBinding, 0);
        GLASSERT();

        /* Render */
        glClearColor (0.5f, 0.5f, 0.5f, 1.0f);
        glClear (GL_COLOR_BUFFER_BIT);

        glUseProgram (progDraw);

        glMemoryBarrier (GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
        glBindBuffer (GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray (loc_vtx);
        glEnableVertexAttribArray (loc_clr);
        glVertexAttribPointer (loc_vtx, 4, GL_FLOAT, GL_FALSE, 32, (void *)0);
        glVertexAttribPointer (loc_clr, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);

        glDrawArrays(GL_POINTS, 0, NUM_VERTS);

        egl_swap();
    }

    return 0;
}

