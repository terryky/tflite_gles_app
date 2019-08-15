/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef __ASSERTGL_H__
#define __ASSERTGL_H__

void 
AssertGLError( char *lpFile, int nLine );

#if 1
    #define GLASSERT()      AssertGLError( __FILE__, __LINE__ )
#else
    #define GLASSERT()      ((void)0)
#endif

#endif /* __ASSERTGL_H__ */

