/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <GLES2/gl2.h>
#include "util_shader.h"
#include "util_egl.h"

#define UNUSED(x) (void)(x)

static GLfloat s_vtx[] =
{
    -0.5f, 0.5f,
    -0.5f,-0.5f,
     0.5f, 0.5f,
};

static GLfloat s_col[] =
{
    1.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 1.0f, 1.0f,
};

static char s_strVS[] = "                         \
                                                  \
attribute    vec4    a_Vertex;                    \
attribute    vec4    a_Color;                     \
varying      vec4    v_color;                     \
                                                  \
void main (void)                                  \
{                                                 \
    gl_Position = a_Vertex;                       \
    v_color     = a_Color;                        \
}                                                ";

static char s_strFS[] = "                         \
                                                  \
precision mediump float;                          \
varying     vec4     v_color;                     \
                                                  \
void main (void)                                  \
{                                                 \
    gl_FragColor = v_color;                       \
}                                                 ";


int
capture_to_img (char *lpFName, int nW, int nH)
{
    FILE *fp;
    unsigned char *lpBuf;
    char strFName[ 128 ];

    sprintf (strFName, "%s_RGBA8888_SIZE%dx%d.img", lpFName, nW, nH);

    fp = fopen (strFName, "wb");
    if (fp == NULL)
    {
        fprintf (stderr, "FATAL ERROR at %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    
    lpBuf = (unsigned char *)malloc (nW * nH * 4);
    if (lpBuf == NULL)
    {
        fprintf (stderr, "FATAL ERROR at %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    glReadPixels (0, 0, nW, nH, GL_RGBA, GL_UNSIGNED_BYTE, lpBuf);

    fwrite (lpBuf, 4, nW * nH, fp);

    free (lpBuf);
    fclose (fp);

    return 0;
}


/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
int
main(int argc, char *argv[])
{
    int i;
    shader_obj_t sobj;
    UNUSED (argc);
    UNUSED (*argv);

    egl_init_with_platform_window_surface (2, 0, 0, 0, 960, 540);

    generate_shader (&sobj, s_strVS, s_strFS);

    for (i = 0; i < 10; i ++)
    {
        glClearColor (0.5f, 0.5f, 0.5f, 1.0f);
        glClear (GL_COLOR_BUFFER_BIT);

        glUseProgram( sobj.program );

        glEnableVertexAttribArray (sobj.loc_vtx);
        glEnableVertexAttribArray (sobj.loc_clr);
        glVertexAttribPointer (sobj.loc_vtx, 2, GL_FLOAT, GL_FALSE, 0, s_vtx);
        glVertexAttribPointer (sobj.loc_clr, 4, GL_FLOAT, GL_FALSE, 0, s_col);

        glDrawArrays (GL_TRIANGLE_STRIP, 0, 3);

        egl_swap();
    }
    //capture_to_img ("out", 1920, 1080);
    sleep (10);

    return 0;
}

