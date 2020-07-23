#include <cstdint>
#include <cstdio>
#include <cmath>
#include <time.h>

#if defined(__ARM_NEON__) || defined(__aarch64__)
#include <arm_neon.h>
#else
#error "This app requires NEON"
#endif

struct FP32
{
    union {
        float    f;
        uint32_t ui;
    };
};

void
show_FPCR ()
{
    unsigned int fpcr = __builtin_aarch64_get_fpcr();
    printf ("FPCR: %08x\n", fpcr);
}

/* 
 *  set flush-to-zero (bit 24) 
 *  |.......Z|........|........|........|
 */
void
set_FPCR_FTZ (int mode)
{
    unsigned int fpcr = __builtin_aarch64_get_fpcr() & ~(1u << 24);
    fpcr &= ~(1u << 24);
    fpcr |= mode << 24;
    __builtin_aarch64_set_fpcr(fpcr);
}

void
disable_FTZ ()
{
    set_FPCR_FTZ (0);
}

void
enable_FTZ ()
{
    set_FPCR_FTZ (1);
}


static double
pmeter_get_time_ms ()
{
    struct timespec tv;
    clock_gettime (CLOCK_MONOTONIC, &tv);
    return  (tv.tv_sec*1000 + (float)tv.tv_nsec/1000000.0);
} 

static const char *
get_fpclassify_str (float val)
{
    switch (std::fpclassify(val))
    {
    case FP_INFINITE : return "FP_INFINITE";
    case FP_NAN      : return "FP_NAN";
    case FP_NORMAL   : return "FP_NORMAL";
    case FP_SUBNORMAL: return "FP_SUBNORMAL";
    case FP_ZERO     : return "FP_ZERO";
    default          : return "???";
    }
}


#define ARRAYSIZE (1000)
#define LOOPNUM   (1000*1000*1000)
float32x4_t *vec_a;
float32x4_t *vec_b;
float32x4_t *vec_c;
float       *val_a;
float       *val_b;
float       *val_c;

void
calculate_neon ()
{
    vec_a = new float32x4_t[ARRAYSIZE];
    vec_b = new float32x4_t[ARRAYSIZE];
    vec_c = new float32x4_t[ARRAYSIZE];

    for (int i = 0; i < ARRAYSIZE; i++)
    {
        vec_a[i] = vdupq_n_f32(0.1f);
        vec_b[i] = vdupq_n_f32(1.2e-38f);
        vec_c[i] = vdupq_n_f32(0.0f);
    }
    
    double ttime[2];
    ttime[0] = pmeter_get_time_ms ();
    for (int i = 0; i < LOOPNUM; i ++)
    {
        int j = i % ARRAYSIZE;
        vec_c[j] += vmulq_f32(vec_a[j], vec_b[j]);
    }
    ttime[1] = pmeter_get_time_ms ();

    __attribute__((aligned(16))) float buf[4];
    vst1q_f32(buf, vec_c[0]);

    FP32 fp;
    fp.f = buf[0];

    printf("--------------------------------------\n");
    printf("calc_result  = 0x%08x (%g)\n", fp.ui, fp.f);
    printf("fpclassify() = %s\n", get_fpclassify_str(fp.f));
    printf("elapsed time = %f\n", ttime[1] - ttime[0]);
    printf("--------------------------------------\n");

    delete vec_a;
    delete vec_b;
    delete vec_c;
}

void
calculate_float ()
{
    val_a = new float[ARRAYSIZE];
    val_b = new float[ARRAYSIZE];
    val_c = new float[ARRAYSIZE];

    for (int i = 0; i < ARRAYSIZE; i++)
    {
        val_a[i] = 0.1f;
        val_b[i] = 1.2e-38f;
        val_c[i] = 0.0f;
    }

    double ttime[2];
    ttime[0] = pmeter_get_time_ms ();
    for (int i = 0; i < LOOPNUM; i ++)
    {
        int j = i % ARRAYSIZE;
        val_c[j] += val_a[j] * val_b[j];
    }
    ttime[1] = pmeter_get_time_ms ();

    FP32 fp;
    fp.f = val_c[0];

    printf("--------------------------------------\n");
    printf("calc_result  = 0x%08x (%g)\n", fp.ui, fp.f);
    printf("fpclassify() = %s\n", get_fpclassify_str(fp.f));
    printf("elapsed time = %f\n", ttime[1] - ttime[0]);
    printf("--------------------------------------\n");

    delete val_a;
    delete val_b;
    delete val_c;
}

int main(int argc, char **argv)
{
    printf ("\n---------------------- DEFAULT -------------------\n");
    show_FPCR ();
    calculate_neon ();
    calculate_float ();

    printf ("\n---------------------- Enable FTZ -------------------\n");
    enable_FTZ ();
    show_FPCR ();
    calculate_neon ();
    calculate_float ();

    printf ("\n---------------------- Disable FTZ -------------------\n");
    disable_FTZ ();
    show_FPCR ();
    calculate_neon ();
    calculate_float ();

    return 0;
}
