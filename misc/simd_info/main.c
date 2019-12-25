#include <stdio.h>
#include <stdlib.h>

#define PRINT_ENABLE(func)  fprintf (stderr, "%-16s: [ENABLED]\n", func)
#define PRINT_DISABLE(func) fprintf (stderr, "%-16s: [-------]\n", func)



int
main (int argc, char *argv[])
{
#if defined(__MMX__)
    PRINT_ENABLE ("__MMX__");
#else
    PRINT_DISABLE("__MMX__");
#endif




#if defined(__SSE__)
    PRINT_ENABLE ("__SSE__");
#else
    PRINT_DISABLE("__SSE__");
#endif

#if defined(__SSE2__)
    PRINT_ENABLE ("__SSE2__");
#else
    PRINT_DISABLE("__SSE2__");
#endif

#if defined(__SSE3__)
    PRINT_ENABLE ("__SSE3__");
#else
    PRINT_DISABLE("__SSE3__");
#endif

#if defined(__SSE4_1__)
    PRINT_ENABLE ("__SSE4_1__");
#else
    PRINT_DISABLE("__SSE4_1__");
#endif

#if defined(__SSE4_2__)
    PRINT_ENABLE ("__SSE4_2__");
#else
    PRINT_DISABLE("__SSE4_2__");
#endif




#if defined(__AES__)
    PRINT_ENABLE ("__AES__");
#else
    PRINT_DISABLE("__AES__");
#endif

#if defined(__AVX__)
    PRINT_ENABLE ("__AVX__");
#else
    PRINT_DISABLE("__AVX__");
#endif
    
#if defined(__AVX2__)
    PRINT_ENABLE ("__AVX2__");
#else
    PRINT_DISABLE("__AVX2__");
#endif




#if defined(__AVX512BW__)
    PRINT_ENABLE ("__AVX512BW__");
#else
    PRINT_DISABLE("__AVX512BW__");
#endif

#if defined(__AVX512CD__)
    PRINT_ENABLE ("__AVX512CD__");
#else
    PRINT_DISABLE("__AVX512CD__");
#endif

#if defined(__AVX512DQ__)
    PRINT_ENABLE ("__AVX512DQ__");
#else
    PRINT_DISABLE("__AVX512DQ__");
#endif

#if defined(__AVX512ER__)
    PRINT_ENABLE ("__AVX512ER__");
#else
    PRINT_DISABLE("__AVX512ER__");
#endif

#if defined(__AVX512F__)
    PRINT_ENABLE ("__AVX512F__");
#else
    PRINT_DISABLE("__AVX512F__");
#endif

#if defined(__AVX512PF__)
    PRINT_ENABLE ("__AVX512PF__");
#else
    PRINT_DISABLE("__AVX512PF__");
#endif

#if defined(__AVX512VL__)
    PRINT_ENABLE ("__AVX512VL__");
#else
    PRINT_DISABLE("__AVX512VL__");
#endif

#if defined(__AVX512IFMA__)
    PRINT_ENABLE ("__AVX512IFMA__");
#else
    PRINT_DISABLE("__AVX512IFMA__");
#endif

#if defined(__AVX512VBMI__)
    PRINT_ENABLE ("__AVX512VBMI__");
#else
    PRINT_DISABLE("__AVX512VBMI__");
#endif




#if defined(__FMA__)
    PRINT_ENABLE ("__FMA__");
#else
    PRINT_DISABLE("__FMA__");
#endif




#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    PRINT_ENABLE ("__ARM_NEON__");
#else
    PRINT_DISABLE("__ARM_NEON__");
#endif

    return 0;
}
