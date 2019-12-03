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


#endif /* _FLOAT16_H_ */
    