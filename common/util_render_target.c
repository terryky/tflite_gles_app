/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GLES2/gl2.h>
#include "assertgl.h"
#include "util_shader.h"
#include "util_matrix.h"
#include "util_render_target.h"


int
create_render_target (render_target_t *rtarget, int w, int h, unsigned int flag)
{
    GLuint fbo_id = 0, tex_id = 0;

    if (flag)
    {
        glGenTextures (1, &tex_id);
        glBindTexture (GL_TEXTURE_2D, tex_id);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture (GL_TEXTURE_2D, 0);

        glGenFramebuffers (1, &fbo_id);
    }

    rtarget->texid  = tex_id;
    rtarget->fboid  = fbo_id;
    rtarget->width  = w;
    rtarget->height = h;

    GLASSERT ();
    return 0;
}


int
destroy_render_target (render_target_t *rtarget)
{
    GLuint fbotexid = rtarget->texid;
    GLuint fboid    = rtarget->fboid;

    glDeleteTextures (1, &fbotexid);
    glDeleteFramebuffers (1, &fboid);

    GLASSERT ();
    return 0;
}

int
set_render_target (render_target_t *rtarget)
{
    if (rtarget->fboid > 0)
    {
        glBindFramebuffer (GL_FRAMEBUFFER, rtarget->fboid);
        glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtarget->texid, 0);
    }
    else
    {
        glBindFramebuffer (GL_FRAMEBUFFER, 0);
    }

    glViewport (0, 0, rtarget->width, rtarget->height);
    glScissor  (0, 0, rtarget->width, rtarget->height);

    GLASSERT ();

  return 0;
}
