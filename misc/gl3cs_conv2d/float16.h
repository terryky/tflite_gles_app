#ifndef _FLOAT16_H_
#define _FLOAT16_H_

typedef unsigned short fp16_t;

typedef union unifloat
{
    unsigned int n;
    float        f;
} unifloat_t;


/*
 *        +--------+--------+--------+--------+
 *  fp32: |Seeeeeee|efffffff|ffffffff|ffffffff|
 *        +--------+--------+--------+--------+
 *
 *                          +--------+--------+
 *  fp16:                   |Seeeeeff|ffffffff|
 *                          +--------+--------+
 */


static inline float
half2float (unsigned short half_)
{
    unsigned int half = half_;
    unsigned int sign = (half & 0x8000);
    unsigned int expn = ((((half >> 10) & 0x1f) - 15 + 127) & 0xFF);
    unsigned int frac = (half & 0x3FF);
    unsigned int fp32 = (sign << 16) | (expn << 23) | (frac << 13);

    unifloat_t u;
    u.n = fp32;
    return u.f;
}



static inline fp16_t
float2half (float fval_)
{
    unifloat_t u;
    u.f = fval_;

    unsigned int fval = u.n;
    unsigned int sign = (fval >> 16) & 0x8000;
    unsigned int expn = ((fval >> 23) - 127 + 15) & 0x1f;
    unsigned int frac = (fval >> 13) & 0x3ff;
    unsigned int fp32 = (sign) | (expn << 10) | (frac);

    return (fp16_t)fp32;
}


#endif /* _FLOAT16_H_ */
    