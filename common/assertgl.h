/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef __ASSERTGL_H__
#define __ASSERTGL_H__

#ifdef __cplusplus
extern "C" {
#endif


void 
AssertGLError( const char *lpFile, int nLine );

#if 1
    #define GLASSERT()      AssertGLError( __FILE__, __LINE__ )
#else
    #define GLASSERT()      ((void)0)
#endif

#ifdef __cplusplus
}
#endif
#endif /* __ASSERTGL_H__ */

