/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GLES2/gl2.h>
#include "assertgl.h"
#include "util_debug.h"

static char *
GetGLErrMsg( int nCode )
{
    switch( nCode )
    {
    case 0x0000: return "GL_NO_ERROR";          break;
    case 0x0500: return "GL_INVALID_ENUM";      break;
    case 0x0501: return "GL_INVALID_VALUE";     break;
    case 0x0502: return "GL_INVALID_OPERATION"; break;
    case 0x0503: return "GL_STACK_OVERFLOW";    break;
    case 0x0504: return "GL_STACK_UNDERFLOW";   break;
    case 0x0505: return "GL_OUT_OF_MEMORY";     break;
    default:     return "UNKNOWN ERROR";        break;
    }
}


void
AssertGLError( const char *lpFile, int nLine )
{
    int error;

    while (( error = glGetError()) != GL_NO_ERROR )
    {
        DBG_LOGE( "[GL ASSERT ERR] \"%s\"(%d):0x%04x(%s)\n",
                    lpFile, nLine, error, GetGLErrMsg( error ) );
    }
}

