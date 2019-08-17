/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <GLES2/gl2.h>
#include "assertgl.h"

#if defined (USE_PNG_TEXTURE)
#include "util_image_png.h"
#endif

#if defined (USE_JPEG_TEXTURE)
#include "util_image_jpg.h"
#endif


static GLuint
create_2d_texture (void *imgbuf, int width, int height)
{
    GLuint texid;

    glGenTextures (1, &texid );
    glBindTexture (GL_TEXTURE_2D, texid);

    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#ifdef WIN32
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
#else
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#endif

    glPixelStorei (GL_UNPACK_ALIGNMENT, 4);

    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA,
        width, height, 0, GL_RGBA,
        GL_UNSIGNED_BYTE, imgbuf);

    return texid;
}


int
load_png_texture (char *name, int *lpTexID, int *lpWidth, int *lpHeight)
{
#if defined (USE_PNG_TEXTURE)
    unsigned int  width, height;
    int           ctype, mem_size;
    void          *imgbuf;
    GLuint        texid;

    open_png_from_file (name, &width, &height, &ctype);

    mem_size = width * height * 4;
    imgbuf = malloc (mem_size);
    if (imgbuf == NULL)
    {
        fprintf (stderr, "Failed to load PNG: %s\n", name);
        return -1;
    }

    decode_png_from_file (name, imgbuf);

    texid = create_2d_texture (imgbuf, width, height);

    if (lpTexID)  *lpTexID  = texid;
    if (lpWidth)  *lpWidth  = width;
    if (lpHeight) *lpHeight = height;
    free (imgbuf);

    GLASSERT();
    return 0;
#else
    fprintf (stderr, "please enable compile option: \"USE_PNG_TEXTURE\"\n");
    return -1;
#endif
}

int
load_jpg_texture (char *name, int *lpTexID, int *lpWidth, int *lpHeight)
{
#if defined (USE_JPEG_TEXTURE)
    unsigned int width, height;
    int          mem_size;
    void         *imgbuf;
    GLuint       texid;

    open_jpeg_from_file (name, &width, &height);

    mem_size = width * height * 4;
    imgbuf = malloc (mem_size);
    if (imgbuf == NULL)
    {
        fprintf (stderr, "Failed to load JPG: %s\n", name);
        return -1;
    }

    decode_jpeg_from_file (name, imgbuf);

    texid = create_2d_texture (imgbuf, width, height);

    if (lpTexID)  *lpTexID  = texid;
    if (lpWidth)  *lpWidth  = width;
    if (lpHeight) *lpHeight = height;
    free (imgbuf);

    GLASSERT();
    return 0;
#else
    fprintf (stderr, "please enable compile option: \"USE_JPEG_TEXTURE\"\n");
    return -1;
#endif
}


int
load_png_cube_texture (char *name[], int *lpTexID)
{
#if defined (USE_PNG_TEXTURE)
    unsigned int  width, height;
    int           ctype, mem_size;
    unsigned char *lpBuf;
    unsigned int  texID;
    int i;
    GLenum target[] = { 
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
        GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    };

    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texID);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    for(i = 0; i < 6; i++)
    {
        open_png_from_file (name[i], &width, &height, &ctype);
        mem_size = width * height * 4;

        lpBuf = malloc (mem_size);
        if (lpBuf == NULL)
        {
            fprintf (stderr, "Failed to load PNG: %s\n", name[i]);
            return -1;
        }
        
        decode_png_from_file (name[i], lpBuf);

        if (lpBuf == NULL)
        {
            fprintf(stderr, "%s is not found!!\n", name[i]);
            return -1;
        }

        glTexImage2D (target[i], 0, GL_RGBA,
            width, height, 0, GL_RGBA,
            GL_UNSIGNED_BYTE,
            lpBuf);

        free( lpBuf );
        GLASSERT();
    }

    glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#if 0
    glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
#endif
    *lpTexID = texID;

    GLASSERT();
    return 0;
#else
    fprintf (stderr, "please enable compile option: \"USE_PNG_TEXTURE\"\n");
    return -1;
#endif
}
