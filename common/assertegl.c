/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <EGL/egl.h>
#include "assertegl.h"


static char *
GetEGLErrMsg( int nCode )
{
    switch( nCode )
    {
    case EGL_SUCCESS:             return "EGL_SUCCESS";            break;
    case EGL_NOT_INITIALIZED:     return "EGL_NOT_INITIALIZED";    break;
    case EGL_BAD_ACCESS:          return "EGL_BAD_ACCESS";         break;
    case EGL_BAD_ALLOC:           return "EGL_BAD_ALLOC";          break;
    case EGL_BAD_ATTRIBUTE:       return "EGL_BAD_ATTRIBUTE";      break;
    case EGL_BAD_CONTEXT:         return "EGL_BAD_CONTEXT";        break;
    case EGL_BAD_CONFIG:          return "EGL_BAD_CONFIG";         break;
    case EGL_BAD_CURRENT_SURFACE: return "EGL_BAD_CURRENT_SURFACE";break;
    case EGL_BAD_DISPLAY:         return "EGL_BAD_DISPLAY";        break;
    case EGL_BAD_SURFACE:         return "EGL_BAD_SURFACE";        break;
    case EGL_BAD_PARAMETER:       return "EGL_BAD_PARAMETER";      break;
    case EGL_BAD_NATIVE_PIXMAP:   return "EGL_BAD_NATIVE_PIXMAP";  break;
    case EGL_BAD_NATIVE_WINDOW:   return "EGL_BAD_NATIVE_WINDOW";  break;
    case EGL_CONTEXT_LOST:        return "EGL_CONTEXT_LOST";       break;
    default:                      return "UNKNOWN EGL ERROR";      break;
    }
}


void
AssertEGLError( char *lpFile, int nLine )
{
    int error;

    while (( error = eglGetError()) != EGL_SUCCESS )
    {
        printf( "[EGL ASSERT ERR] \"%s\"(%d):0x%04x(%s)\n",
                    lpFile, nLine, error, GetEGLErrMsg( error ) );
    }
}

