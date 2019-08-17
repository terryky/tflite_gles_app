/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include "util_egl.h"

static int show_full_attrib = 1;


void
print_context_attrib (EGLDisplay dpy, EGLConfig cfg)
{
    EGLint ival;
    EGLint vals[8];

    fprintf (stderr, " %-32s: ", "EGL_CONFIG_ID"); ival = -1;
    eglGetConfigAttrib (dpy, cfg, EGL_CONFIG_ID, &ival);
    fprintf (stderr, "%d\n", ival);


    fprintf (stderr, " %-32s: ", "EGL_(BUFFER/R/G/B/A)_SIZE");
    vals[0] = -1; eglGetConfigAttrib (dpy, cfg, EGL_BUFFER_SIZE,    &vals[0]);
    vals[1] = -1; eglGetConfigAttrib (dpy, cfg, EGL_RED_SIZE,       &vals[1]);
    vals[2] = -1; eglGetConfigAttrib (dpy, cfg, EGL_GREEN_SIZE,     &vals[2]);
    vals[3] = -1; eglGetConfigAttrib (dpy, cfg, EGL_BLUE_SIZE,      &vals[3]);
    vals[4] = -1; eglGetConfigAttrib (dpy, cfg, EGL_ALPHA_SIZE,     &vals[4]);
    fprintf (stderr, "%d/%d/%d/%d/%d\n", 
        vals[0], vals[1], vals[2], vals[3], vals[4]);

    fprintf (stderr, " %-32s: ", "EGL_(DEPTH/STENCIL)_SIZE");
    vals[0] = -1; eglGetConfigAttrib (dpy, cfg, EGL_DEPTH_SIZE,   &vals[0]);
    vals[1] = -1; eglGetConfigAttrib (dpy, cfg, EGL_STENCIL_SIZE, &vals[1]);
    fprintf (stderr, "%d/%d\n", vals[0], vals[1]);

    if (show_full_attrib)
    {
        fprintf (stderr, " %-32s: ", "EGL_(ALPHA_MASK/LUMINANCE)_SIZE"); ival = -1;
        vals[0] = -1; eglGetConfigAttrib (dpy, cfg, EGL_ALPHA_MASK_SIZE, &vals[0]);
        vals[1] = -1; eglGetConfigAttrib (dpy, cfg, EGL_LUMINANCE_SIZE,  &vals[1]);
        fprintf (stderr, "%d/%d\n", vals[0], vals[1]);

        fprintf (stderr, " %-32s: ", "EGL_BIND_TO_TEXTURE_(RGB/RGBA)");
        vals[0] = -1; eglGetConfigAttrib (dpy, cfg, EGL_BIND_TO_TEXTURE_RGB , &vals[0]);
        vals[1] = -1; eglGetConfigAttrib (dpy, cfg, EGL_BIND_TO_TEXTURE_RGBA, &vals[1]);
        fprintf (stderr, "%d %d\n", vals[0], vals[1]);

        fprintf (stderr, " %-32s: ", "EGL_COLOR_BUFFER_TYPE"); ival = -1;
        eglGetConfigAttrib (dpy, cfg, EGL_COLOR_BUFFER_TYPE, &ival);
        switch (ival)
        {
        case EGL_RGB_BUFFER:
            fprintf (stderr, "EGL_RGB_BUFFER\n");
            break;
        case EGL_LUMINANCE_BUFFER:
            fprintf (stderr, "EGL_LUMINANCE_BUFFER\n");
            break;
        default:
            fprintf (stderr, "ERR\n");
        }

        fprintf (stderr, " %-32s: ", "EGL_CONFIG_CAVEAT"); ival = -1;
        eglGetConfigAttrib (dpy, cfg, EGL_CONFIG_CAVEAT, &ival);
        switch (ival)
        {
        case EGL_NONE:
            fprintf (stderr, "EGL_NONE\n");
            break;
        case EGL_SLOW_CONFIG:
            fprintf (stderr, "EGL_SLOW_CONFIG\n");
            break;
        case EGL_NON_CONFORMANT_CONFIG:
            fprintf (stderr, "EGL_NON_CONFORMANT_CONFIG\n");
            break;
        default:
            fprintf (stderr, "ERR\n");
        }
    }


    if (show_full_attrib)
    {
        fprintf (stderr, " %-32s: ", "EGL_CONFORMANT"); ival = -1;
        eglGetConfigAttrib (dpy, cfg, EGL_CONFORMANT, &ival);
      
        if (ival & EGL_OPENGL_ES_BIT )
            fprintf (stderr, "GLES1 ");
        if (ival & EGL_OPENGL_ES2_BIT )
            fprintf (stderr, "GLES2 ");
#if defined (EGL_OPENGL_ES3_BIT_KHR)
        if (ival & EGL_OPENGL_ES3_BIT_KHR )
            fprintf (stderr, "GLES3 ");
#endif
        fprintf (stderr, "\n");


        fprintf (stderr, " %-32s: ", "EGL_LEVEL"); ival = -1;
        eglGetConfigAttrib (dpy, cfg, EGL_LEVEL, &ival);
        fprintf (stderr, "%d\n", ival);

        fprintf (stderr, " %-32s: ", "EGL_MATCH_NATIVE_PIXMAP"); ival = -1;
        eglGetConfigAttrib (dpy, cfg, EGL_MATCH_NATIVE_PIXMAP, &ival);
        fprintf (stderr, "%d\n", ival);

        fprintf (stderr, " %-32s: ", "EGL_(MAX/MIN)_SWAP_INTERVAL");
        vals[0] = -1; eglGetConfigAttrib (dpy, cfg, EGL_MAX_SWAP_INTERVAL, &vals[0]);
        vals[1] = -1; eglGetConfigAttrib (dpy, cfg, EGL_MIN_SWAP_INTERVAL, &vals[1]);
        fprintf (stderr, "%d/%d\n", vals[0], vals[1]);

        fprintf (stderr, " %-32s: ", "EGL_NATIVE_RENDERABLE"); ival = -1;
        eglGetConfigAttrib (dpy, cfg, EGL_NATIVE_RENDERABLE, &ival);
        fprintf (stderr, "%d\n", ival);

        fprintf (stderr, " %-32s: ", "EGL_NATIVE_VISUAL_TYPE"); ival = -1;
        eglGetConfigAttrib (dpy, cfg, EGL_NATIVE_VISUAL_TYPE, &ival);
        switch (ival)
        {
        case EGL_NONE:
            fprintf (stderr, "EGL_NONE\n");
            break;
        default:
            fprintf (stderr, "UNKNOWN(%d)\n", ival);
        }

        fprintf (stderr, " %-32s: ", "EGL_RENDERABLE_TYPE"); ival = -1;
        eglGetConfigAttrib (dpy, cfg, EGL_RENDERABLE_TYPE, &ival);
        if (ival & EGL_OPENGL_ES_BIT )
            fprintf (stderr, "GLES1 ");
        if (ival & EGL_OPENGL_ES2_BIT )
            fprintf (stderr, "GLES2 ");
#if defined (EGL_OPENGL_ES3_BIT_KHR)
        if (ival & EGL_OPENGL_ES3_BIT_KHR )
            fprintf (stderr, "GLES3 ");
#endif
        fprintf (stderr, "\n");
    }

    fprintf (stderr, " %-32s: ", "EGL_SAMPLE_(BUFS/SMPS)");
    vals[0] = -1; eglGetConfigAttrib (dpy, cfg, EGL_SAMPLE_BUFFERS,  &vals[0]);
    vals[1] = -1; eglGetConfigAttrib (dpy, cfg, EGL_SAMPLES,         &vals[1]);
    fprintf (stderr, "%d/%d\n", vals[0], vals[1]);

    fprintf (stderr, " %-32s: ", "EGL_SURFACE_TYPE"); ival = -1;
    eglGetConfigAttrib (dpy, cfg, EGL_SURFACE_TYPE, &ival);
    if (ival & EGL_PBUFFER_BIT)
        fprintf (stderr, "PBUFFER/");
    if (ival & EGL_PIXMAP_BIT)
        fprintf (stderr, "PIXMAP/");
    if (ival & EGL_WINDOW_BIT)
        fprintf (stderr, "WINDOW/");
    if (ival & EGL_MULTISAMPLE_RESOLVE_BOX_BIT)
        fprintf (stderr, "MULTISAMPLE_RESOLVE_BOX/");
    if (ival & EGL_SWAP_BEHAVIOR_PRESERVED_BIT)
        fprintf (stderr, "SWAP_BEHAVIOR_PRESERVED");
    fprintf (stderr, "\n");


    if (show_full_attrib)
    {
        fprintf (stderr, " %-32s: ", "EGL_TRANSPARENT_TYPE"); ival = -1;
        eglGetConfigAttrib (dpy, cfg, EGL_TRANSPARENT_TYPE, &ival);
        switch (ival)
        {
        case EGL_NONE:
            fprintf (stderr, "EGL_NONE\n");
            break;
        case EGL_TRANSPARENT_RGB:
            fprintf (stderr, "EGL_TRANSPARENT_RGB\n");
            break;
        default:
            fprintf (stderr, "ERR\n");
        }

        fprintf (stderr, " %-32s: ", "EGL_TRANSPARENT_(R/G/B)_VALUE");
        vals[0] = -1; eglGetConfigAttrib (dpy, cfg, EGL_TRANSPARENT_RED_VALUE,  &vals[0]);
        vals[1] = -1; eglGetConfigAttrib (dpy, cfg, EGL_TRANSPARENT_GREEN_VALUE,&vals[1]);
        vals[2] = -1; eglGetConfigAttrib (dpy, cfg, EGL_TRANSPARENT_BLUE_VALUE, &vals[2]);
        fprintf (stderr, "%d/%d/%d\n", vals[0], vals[1], vals[2]);
    }
}

#define print_gl_ival(name)                                     \
do {                                                            \
    int ival[256];                                              \
    glGetIntegerv (name, ival);                                 \
    fprintf (stderr, "%-50s : %d\n", #name, ival[0]);           \
} while (0)

#define print_gl_nosupported(name)                              \
do {                                                            \
    fprintf (stderr, "%-50s ; ---\n", #name);                   \
} while (0)

int 
main (int argc, char *argv[])
{
    EGLDisplay dpy;
    EGLBoolean eglerr;
    const char *str;
    EGLint     num_configs;
    EGLConfig  *configs;
    int        i;

#if 1
    egl_init_with_platform_window_surface (2, 0, 0, 0, 960, 540);
#else
    egl_init_with_pbuffer_surface (2, 0, 0, 0, 960, 540);
#endif

    dpy = egl_get_display();
    if (dpy == EGL_NO_DISPLAY)
    {
        fprintf (stderr, "###ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }


    fprintf (stderr, "============================================\n");
    fprintf (stderr, "    EGL INFO\n");
    fprintf (stderr, "============================================\n");

    str = eglQueryString (dpy, EGL_CLIENT_APIS);
    fprintf (stderr, "EGL_CLIENT_APIS : %s\n", str);

    str = eglQueryString (dpy, EGL_VENDOR);
    fprintf (stderr, "EGL_VENDOR      : %s\n", str);

    str = eglQueryString (dpy, EGL_VERSION);
    fprintf (stderr, "EGL_VERSION     : %s\n", str);

    str = eglQueryString (dpy, EGL_EXTENSIONS);
    fprintf (stderr, "EGL_EXTENSIONS  :\n");

    if (str)
    {
        char *cpy, *p;

        /* on some platform, prohibited to overwrite the buffer. */
        cpy = (char *)malloc (strlen(str) + 1);
        strcpy (cpy, str);
        p = strtok ((char *)cpy, " ");
        while (p)
        {
            fprintf (stderr, "                  %s\n", p);
            p = strtok (NULL, " ");
        }
    }

    fprintf (stderr, "\n");

    eglerr = eglGetConfigs(dpy, NULL, 0, &num_configs);
    if (eglerr != EGL_TRUE)
    {
        fprintf (stderr, "###ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

  
    configs = (EGLConfig *)malloc (sizeof(EGLConfig) * num_configs);
    if (configs == NULL)
    {
        fprintf (stderr, "###ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    eglerr = eglGetConfigs (dpy, configs, num_configs, &num_configs);
    if (eglerr != EGL_TRUE)
    {
        fprintf (stderr, "###ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }


    for (i = 0; i < num_configs; i ++)
    {
        fprintf (stderr, "============================================ [%2d/%2d]\n",
                        i, num_configs);
        print_context_attrib (dpy, configs[i]);
    }


    free (configs);


    fprintf (stderr, "\n");
    fprintf (stderr, "============================================\n");
    fprintf (stderr, "    GL INFO\n");
    fprintf (stderr, "============================================\n");
    
    fprintf (stderr, "GL_VERSION      : %s\n", glGetString (GL_VERSION));
    fprintf (stderr, "GL_SL_VERSION   : %s\n", glGetString (GL_SHADING_LANGUAGE_VERSION));
    fprintf (stderr, "GL_VENDOR       : %s\n", glGetString (GL_VENDOR));
    fprintf (stderr, "GL_RENDERER     : %s\n", glGetString (GL_RENDERER));

    str = (const char *)glGetString (GL_EXTENSIONS);
    fprintf (stderr, "GL_EXTENSIONS   :\n");
    {
        char *cpy, *p;

        /* on some platform, prohibited to overwrite the buffer. */
        cpy = (char *)malloc (strlen(str) + 1);
        strcpy (cpy, str);
        p = strtok ((char *)cpy, " ");

        while (p)
        {
            fprintf (stderr, "                  %s\n", p);
            p = strtok (NULL, " ");
        }
    }

    print_gl_ival (GL_RED_BITS);
    print_gl_ival (GL_GREEN_BITS);
    print_gl_ival (GL_BLUE_BITS);
    print_gl_ival (GL_ALPHA_BITS);
    print_gl_ival (GL_DEPTH_BITS);
    print_gl_ival (GL_STENCIL_BITS);

#if defined (GL_MAX_TEXTURE_SIZE)
    print_gl_ival (GL_MAX_TEXTURE_SIZE);
#else
    print_gl_nosupported (GL_MAX_TEXTURE_SIZE)
#endif

#if defined (GL_NUM_COMPRESSED_TEXTURE_FORMATS)
    print_gl_ival (GL_NUM_COMPRESSED_TEXTURE_FORMATS);
#else
    print_gl_nosupported (GL_NUM_COMPRESSED_TEXTURE_FORMATS);
#endif

#if defined (GL_MAX_CUBE_MAP_TEXTURE_SIZE)
    print_gl_ival (GL_MAX_CUBE_MAP_TEXTURE_SIZE);
#else
    print_gl_nosupported (GL_MAX_CUBE_MAP_TEXTURE_SIZE);
#endif

#if defined (GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS)
    print_gl_ival (GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS);
#else
    print_gl_nosupported (GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS);
#endif

#if defined (GL_MAX_COMBINED_UNIFORM_BLOCKS)
    print_gl_ival (GL_MAX_COMBINED_UNIFORM_BLOCKS);
#else
    print_gl_nosupported (GL_MAX_COMBINED_UNIFORM_BLOCKS);
#endif

#if defined (GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)
    print_gl_ival (GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS);
#else
    print_gl_nosupported (GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS);
#endif

#if defined (GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS)
    print_gl_ival (GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS);
#else
    print_gl_nosupported (GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS);
#endif

#if defined (GL_MAX_COLOR_ATTACHMENTS)
    print_gl_ival (GL_MAX_COLOR_ATTACHMENTS);
#else
    print_gl_nosupported (GL_MAX_COLOR_ATTACHMENTS);
#endif

#if defined (GL_MIN_PROGRAM_TEXEL_OFFSET)
    print_gl_ival (GL_MIN_PROGRAM_TEXEL_OFFSET);
#else
    print_gl_nosupported (GL_MIN_PROGRAM_TEXEL_OFFSET);
#endif

#if defined (GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS)
    print_gl_ival (GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS);
#else
    print_gl_nosupported (GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS);
#endif

#if defined (GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS)
    print_gl_ival (GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS);
#else
    print_gl_nosupported (GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS);
#endif

#if defined (GL_MAX_DRAW_BUFFERS)
    print_gl_ival (GL_MAX_DRAW_BUFFERS);
#else
    print_gl_nosupported (GL_MAX_DRAW_BUFFERS);
#endif

#if defined (GL_MAX_ARRAY_TEXTURE_LAYERS)
    print_gl_ival (GL_MAX_ARRAY_TEXTURE_LAYERS);
#else
    print_gl_nosupported (GL_MAX_ARRAY_TEXTURE_LAYERS);
#endif

#if defined (GL_MAX_3D_TEXTURE_SIZE)
    print_gl_ival (GL_MAX_3D_TEXTURE_SIZE);
#else
    print_gl_nosupported (GL_MAX_3D_TEXTURE_SIZE);
#endif

#if defined (GL_MAX_ELEMENTS_VERTICES)
    print_gl_ival (GL_MAX_ELEMENTS_VERTICES);
#else
    print_gl_nosupported (GL_MAX_ELEMENTS_VERTICES);
#endif

#if defined (GL_MAX_ELEMENT_INDEX)
    print_gl_ival (GL_MAX_ELEMENT_INDEX);
#else
    print_gl_nosupported (GL_MAX_ELEMENT_INDEX);
#endif

#if defined (GL_MAX_ELEMENTS_INDICES)
    print_gl_ival (GL_MAX_ELEMENTS_INDICES);
#else
    print_gl_nosupported (GL_MAX_ELEMENTS_INDICES);
#endif

#if defined (GL_MAX_FRAGMENT_INPUT_COMPONENTS)
    print_gl_ival (GL_MAX_FRAGMENT_INPUT_COMPONENTS);
#else
    print_gl_nosupported (GL_MAX_FRAGMENT_INPUT_COMPONENTS);
#endif

#if defined (GL_MAX_FRAGMENT_UNIFORM_BLOCKS)
    print_gl_ival (GL_MAX_FRAGMENT_UNIFORM_BLOCKS);
#else
    print_gl_nosupported (GL_MAX_FRAGMENT_UNIFORM_BLOCKS);
#endif

#if defined (GL_MAX_FRAGMENT_UNIFORM_VECTORS)
    print_gl_ival (GL_MAX_FRAGMENT_UNIFORM_VECTORS);
#else
    print_gl_nosupported (GL_MAX_FRAGMENT_UNIFORM_VECTORS);
#endif

#if defined (GL_MAX_FRAGMENT_UNIFORM_COMPONENTS)
    print_gl_ival (GL_MAX_FRAGMENT_UNIFORM_COMPONENTS);
#else
    print_gl_nosupported (GL_MAX_FRAGMENT_UNIFORM_COMPONENTS);
#endif

#if defined (GL_MAX_PROGRAM_TEXEL_OFFSET)
    print_gl_ival (GL_MAX_PROGRAM_TEXEL_OFFSET);
#else
    print_gl_nosupported (GL_MAX_PROGRAM_TEXEL_OFFSET);
#endif
    
#if defined (GL_MAX_RENDERBUFFER_SIZE)
    print_gl_ival (GL_MAX_RENDERBUFFER_SIZE);
#else
    print_gl_nosupported (GL_MAX_RENDERBUFFER_SIZE);
#endif

#if defined (GL_MAX_SAMPLES)
    print_gl_ival (GL_MAX_SAMPLES);
#else
    print_gl_nosupported (GL_MAX_SAMPLES);
#endif

#if defined (GL_MAX_SERVER_WAIT_TIMEOUT)
    print_gl_ival (GL_MAX_SERVER_WAIT_TIMEOUT);
#else
    print_gl_nosupported (GL_MAX_SERVER_WAIT_TIMEOUT);
#endif
    
#if defined (GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS)
    print_gl_ival (GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS);
#else
    print_gl_nosupported (GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS);
#endif

#if defined (GL_MAX_UNIFORM_BLOCK_SIZE)
    print_gl_ival (GL_MAX_UNIFORM_BLOCK_SIZE);
#else
    print_gl_nosupported (GL_MAX_UNIFORM_BLOCK_SIZE);
#endif

#if defined (GL_MAX_UNIFORM_BUFFER_BINDINGS)
    print_gl_ival (GL_MAX_UNIFORM_BUFFER_BINDINGS);
#else
    print_gl_nosupported (GL_MAX_UNIFORM_BUFFER_BINDINGS);
#endif

#if defined (GL_MAX_VARYING_COMPONENTS)
    print_gl_ival (GL_MAX_VARYING_COMPONENTS);
#else
    print_gl_nosupported (GL_MAX_VARYING_COMPONENTS);
#endif

#if defined (GL_MAX_VARYING_VECTORS)
    print_gl_ival (GL_MAX_VARYING_VECTORS);
#else
    print_gl_nosupported (GL_MAX_VARYING_VECTORS);
#endif

#if defined (GL_MAX_VERTEX_ATTRIBS)
    print_gl_ival (GL_MAX_VERTEX_ATTRIBS);
#else
    print_gl_nosupported (GL_MAX_VERTEX_ATTRIBS);
#endif

#if defined (GL_MAX_VERTEX_OUTPUT_COMPONENTS)
    print_gl_ival (GL_MAX_VERTEX_OUTPUT_COMPONENTS);
#else
    print_gl_nosupported (GL_MAX_VERTEX_OUTPUT_COMPONENTS);
#endif

#if defined (GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS)
    print_gl_ival (GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS);
#else
    print_gl_nosupported (GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS);
#endif

#if defined (GL_MAX_VERTEX_UNIFORM_BLOCKS)
    print_gl_ival (GL_MAX_VERTEX_UNIFORM_BLOCKS);
#else
    print_gl_nosupported (GL_MAX_VERTEX_UNIFORM_BLOCKS);
#endif

#if defined (GL_MAX_VERTEX_UNIFORM_VECTORS)
    print_gl_ival (GL_MAX_VERTEX_UNIFORM_VECTORS);
#else
    print_gl_nosupported (GL_MAX_VERTEX_UNIFORM_VECTORS);
#endif

#if defined (GL_MAX_VERTEX_UNIFORM_COMPONENTS)
    print_gl_ival (GL_MAX_VERTEX_UNIFORM_COMPONENTS);
#else
    print_gl_nosupported (GL_MAX_VERTEX_UNIFORM_COMPONENTS);
#endif

#if defined (GL_MAX_TEXTURE_IMAGE_UNITS)
    print_gl_ival (GL_MAX_TEXTURE_IMAGE_UNITS);
#else
    print_gl_nosupported (GL_MAX_TEXTURE_IMAGE_UNITS);
#endif

    egl_terminate ();

    return 0;
}
