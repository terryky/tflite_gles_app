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
#include "ssbo_tensor.h"

#define UNUSED(x) (void)(x)

static int s_prog, s_prog_vis;
static int s_loc_tex;
static int s_loc_imgsize;
static int s_loc_vis_vtx;
static int s_loc_vis_imgsize;

/*
 *  Compute Shader to convert GL Texture to SSBO.
 *  [reference]
 *      https://stackoverflow.com/questions/55165114/android-opengl-shader-program-to-copy-image-from-camera-to-ssbo-for-tf-lite-gpu
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
    "    vec3 pixel = texture(u_sampler, coord).xyz;    \n"
    "                                                   \n"
    "    int idx = 3 * (gid.y * img_w + gid.x);         \n"
    "    output_data.elements[idx + 0] = pixel.x;       \n"
    "    output_data.elements[idx + 1] = pixel.y;       \n"
    "    output_data.elements[idx + 2] = pixel.z;       \n"
    "}                                                  \n";



/*
 *  Vertex & Fragment Shader to visualize the contents of SSBO.
 */
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
    "precision highp float;                             \n"
    "out vec4 oColor;                                   \n"
    "layout(location = 0) uniform ivec2 u_imgsize;      \n"
    "                                                   \n"
    "layout(std430) buffer;                             \n"
    "layout(binding = 0) buffer Input {                 \n"
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


static GLfloat s_vtx[] =
{
    -1.0f, 1.0f,        // +----+
    -1.0f,-1.0f,        // |   /|
     1.0f, 1.0f,        // | /  |
     1.0f,-1.0f,        // +----+
};




static int
create_ssbo (int ssbo_bufsize)
{
    GLuint ssboid;

    glGenBuffers (1, &ssboid);
    glBindBuffer (GL_SHADER_STORAGE_BUFFER, ssboid);
    glBufferData (GL_SHADER_STORAGE_BUFFER, ssbo_bufsize, NULL, GL_STREAM_COPY);
    glBindBuffer (GL_SHADER_STORAGE_BUFFER, 0);
    GLASSERT();

    return ssboid;
}

ssbo_t *
init_ssbo_tensor (int img_w, int img_h)
{
    ssbo_t *ssbo = (ssbo_t *)malloc(sizeof (ssbo_t));
    if (ssbo == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return NULL;
    }

    s_prog = build_compute_shader (s_strCS);
    s_loc_tex = glGetUniformLocation(s_prog, "u_sampler");
    s_loc_imgsize = glGetUniformLocation (s_prog, "u_imgsize");

    s_prog_vis = build_shader (s_strVS, s_strFS);
    s_loc_vis_vtx = glGetAttribLocation (s_prog_vis, "a_Vertex");
    s_loc_vis_imgsize = glGetUniformLocation (s_prog_vis, "u_imgsize");


    /* allocate SSBO buffer. */
    int ssbo_bufsize = img_w * img_h * 3 * sizeof(float);
    int ssboid = create_ssbo (ssbo_bufsize);

    ssbo->width   = img_w;
    ssbo->height  = img_h;
    ssbo->ssbo_id = ssboid;

    return ssbo;
}

int
resize_texture_to_ssbo (int texid, ssbo_t *ssbo)
{
    int resize_w = ssbo->active_width;
    int resize_h = ssbo->active_height;
    int ssboid   = ssbo->ssbo_id;
    int ssbo_range = resize_w * resize_h * 3 * sizeof(float);

    glUseProgram (s_prog);

    glActiveTexture (GL_TEXTURE0);
    glBindTexture (GL_TEXTURE_2D, texid);
    glUniform1i (s_loc_tex, 0);
    glUniform2i (s_loc_imgsize, resize_w, resize_h);
    glBindBufferRange (GL_SHADER_STORAGE_BUFFER, 1, ssboid, 0, ssbo_range);

    int group_size = 16;
    int num_group_x = (int)ceil((float)resize_w / (float)group_size);
    int num_group_y = (int)ceil((float)resize_h / (float)group_size);
    glDispatchCompute (num_group_x, num_group_y, 1);

    glBindBuffer (GL_SHADER_STORAGE_BUFFER, 0);
    glBindTexture (GL_TEXTURE_2D, 0);
    GLASSERT();

    return 0;
}


int
visualize_ssbo (ssbo_t *ssbo)
{
    int w = ssbo->active_width;
    int h = ssbo->active_height;
    int ssbo_id = ssbo->ssbo_id;

    glUseProgram (s_prog_vis);
    glUniform2i (s_loc_vis_imgsize, w, h);
    glBindBufferBase (GL_SHADER_STORAGE_BUFFER, 0, ssbo_id);

    glEnableVertexAttribArray (s_loc_vis_vtx);
    glVertexAttribPointer (s_loc_vis_vtx, 2, GL_FLOAT, GL_FALSE, 0, s_vtx);

    glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);

    glBindBuffer (GL_SHADER_STORAGE_BUFFER, 0);

    GLASSERT();
    return 0;
}

