/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <GLES3/gl31.h>
#include "util_shader.h"
#include "util_egl.h"
#include "assertgl.h"
#include "float16.h"

#define UNUSED(x) (void)(x)
#define ALIGN4(x) (((unsigned int)(x) + 0x003) & ~0x003)

typedef struct _ssbo_t
{
    int n, h, w, c;
    int bufsize;
    GLuint ssbo_id;
} ssbo_t;



double
pmeter_get_time_ms ()
{
    struct timespec tv;
    clock_gettime (CLOCK_MONOTONIC, &tv);
    return  (tv.tv_sec*1000 + (float)tv.tv_nsec/1000000.0);
}


static int
generate_shader_source (char *dst_fname, char *src_fname, 
                    int local_size_x, int local_size_y, int local_size_z)
{
    FILE *fpsrc, *fpdst;
    char strbuf[1024];
    int replaced = 0;

    fpsrc = fopen (src_fname, "r");
    fpdst = fopen (dst_fname, "w");
    if (fpsrc == NULL || fpdst == NULL)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    while (fgets(strbuf, 1024, fpsrc) != NULL)
    {
        if (replaced == 0)
        {
            char *lptmp = strstr (strbuf, "local_size_x");
            if (lptmp)
            {
                snprintf (strbuf, 1024, 
                    "layout(local_size_x = %d, local_size_y = %d, local_size_z = %d) in;\n",
                    local_size_x, local_size_y, local_size_z);
                replaced = 1;
            }
        }
        fputs (strbuf, fpdst);
    }
    
    fclose (fpsrc);
    fclose (fpdst);
    
    return 0;
}

static int
create_ssbo (ssbo_t *ssbo, int use_mediump)
{
    int numitem = ssbo->n * ssbo->h * ssbo->w * ssbo->c;
    int bufsize;
    void *ssbo_buf;

    if (use_mediump)
    {
        bufsize = numitem * sizeof(fp16_t);
        fp16_t *buf = (fp16_t *)malloc(bufsize);
        if (buf == NULL)
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }

        for (int i = 0; i < numitem; i ++)
        {
            buf[i] = float2half ((float)rand() / (float)RAND_MAX);
        }
        ssbo_buf = buf;
    }
    else
    {
        bufsize = numitem * sizeof(float);
        float *buf = (float *)malloc(bufsize);
        if (buf == NULL)
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }

        for (int i = 0; i < numitem; i ++)
        {
            buf[i] = (float)rand() / (float)RAND_MAX;
        }
        ssbo_buf = buf;
    }

    GLuint ssbo_id;
    glGenBuffers (1, &ssbo_id);
    glBindBuffer (GL_SHADER_STORAGE_BUFFER, ssbo_id);
    glBufferData (GL_SHADER_STORAGE_BUFFER, bufsize, ssbo_buf, GL_STREAM_COPY);
    glBindBuffer (GL_SHADER_STORAGE_BUFFER, 0);
    GLASSERT();

    ssbo->ssbo_id = ssbo_id;
    ssbo->bufsize = bufsize;

    return 0;
}

static void
bind_ssbo (int binding, ssbo_t *ssbo)
{
    glBindBufferRange (GL_SHADER_STORAGE_BUFFER, binding, 
        ssbo->ssbo_id, 0, ssbo->bufsize);
}

/*--------------------------------------------------------------------------- *
 *      M A I N    F U N C T I O N
 *--------------------------------------------------------------------------- */
int
main(int argc, char *argv[])
{
    char input_cs_name[2][64] = {"kernel/conv2d_highp.cs", "kernel/conv2d_mediump.cs"};
    char gen_cs_name[2][64];
    char *input_cs, *gen_cs;
    int use_mediump = 0;
    int workload_x = 9;
    int workload_y = 9;
    int workload_z = 1024;
    int workload_z4= workload_z / 4;
    int local_size_x = 8;
    int local_size_y = 4;
    int local_size_z = 8;
    UNUSED (argc);
    UNUSED (*argv);

    const struct option long_options[] = {
        {"workload_x",   required_argument, NULL, 'X'},
        {"workload_y",   required_argument, NULL, 'Y'},
        {"workload_z",   required_argument, NULL, 'Z'},
        {"local_size_x", required_argument, NULL, 'x'},
        {"local_size_y", required_argument, NULL, 'y'},
        {"local_size_z", required_argument, NULL, 'z'},
        {"use_mediump", no_argument, NULL, 'm'},
        {0, 0, 0, 0},
    };

    int c, option_index;
    while ((c = getopt_long (argc, argv, "X:Y:Z:x:y:z:m",
                             long_options, &option_index)) != -1)
    {
        switch (c)
        {
        case 'X': workload_x   = atoi (optarg); break;
        case 'Y': workload_y   = atoi (optarg); break;
        case 'Z': workload_z   = atoi (optarg); break;
        case 'x': local_size_x = atoi (optarg); break;
        case 'y': local_size_y = atoi (optarg); break;
        case 'z': local_size_z = atoi (optarg); break;
        case 'm': use_mediump  = 1; break;
        case '?':
            return -1;
        }
    }

    snprintf (gen_cs_name[0], 64, "kernel/conv2d_highp_%dx%dx%d.cs",   local_size_x, local_size_y, local_size_z);
    snprintf (gen_cs_name[1], 64, "kernel/conv2d_mediump_%dx%dx%d.cs", local_size_x, local_size_y, local_size_z);
    input_cs = input_cs_name[use_mediump];
    gen_cs   =gen_cs_name[use_mediump];

    generate_shader_source (gen_cs, input_cs, local_size_x, local_size_y, local_size_z);

    workload_z  = ALIGN4(workload_z);
    workload_z4 = workload_z / 4;

    fprintf (stderr, "-----------------------------------\n");
    fprintf (stderr, "SHADER FILENAME: %s\n", gen_cs);
    fprintf (stderr, "GLOBAL_WORKLOAD: (%d, %d, %d)\n", 
                            workload_x, workload_y, workload_z);
    fprintf (stderr, "LOCAL_WORK_SIZE: (%d, %d, %d)\n", 
                            local_size_x, local_size_y, local_size_z);
    fprintf (stderr, "-----------------------------------\n");

    int win_w = 100;
    int win_h = 100;
    egl_init_with_platform_window_surface (3, 0, 0, 0, win_w, win_h);

    /* initialize compute shader */
    int progCS = build_compute_shader_from_file ("./", gen_cs);
    int loc_clip       = glGetUniformLocation(progCS, "clip");
    int loc_input_h    = glGetUniformLocation(progCS, "input_data_0_h");
    int loc_input_w    = glGetUniformLocation(progCS, "input_data_0_w");
    int loc_output_h   = glGetUniformLocation(progCS, "output_data_0_h");
    int loc_output_w   = glGetUniformLocation(progCS, "output_data_0_w");
    int loc_src_depth  = glGetUniformLocation(progCS, "src_depth");
    int loc_weight_h   = glGetUniformLocation(progCS, "weights_h");
    int loc_weight_w   = glGetUniformLocation(progCS, "weights_w");
    int loc_workload_x = glGetUniformLocation(progCS, "workload_x");
    int loc_workload_y = glGetUniformLocation(progCS, "workload_y");
    int loc_workload_z = glGetUniformLocation(progCS, "workload_z");

    /* allocate SSBO buffer. */
    ssbo_t ssbo_in     = {         1, workload_x, workload_y, workload_z, 0, 0}; //    1x9x9x1024
    ssbo_t ssbo_out    = {         1, workload_x, workload_y, workload_z, 0, 0}; //    1x9x9x1024
    ssbo_t ssbo_weight = {workload_z,          1,          1, workload_z, 0, 0}; // 1024x1x1x1024
    ssbo_t ssbo_bias   = {workload_z,          1,          1,          1, 0, 0}; // 1024x1x1x   1

    create_ssbo (&ssbo_in, use_mediump);
    create_ssbo (&ssbo_out, use_mediump);
    create_ssbo (&ssbo_weight, use_mediump);
    create_ssbo (&ssbo_bias, use_mediump);

    float ttime0, ttime1;
    float ttime_sum = 0;
    for (int i = 0; i < 1000; i ++)
    {
        ttime0 = pmeter_get_time_ms ();

        glUseProgram (progCS);

        bind_ssbo (0, &ssbo_in);
        bind_ssbo (1, &ssbo_out);
        bind_ssbo (2, &ssbo_weight);
        bind_ssbo (3, &ssbo_bias);

        glProgramUniform1f (progCS, loc_clip,       6.0f);
        glProgramUniform1i (progCS, loc_input_h,    workload_y);    // 9
        glProgramUniform1i (progCS, loc_input_w,    workload_x);    // 9
        glProgramUniform1i (progCS, loc_output_h,   workload_y);    // 9
        glProgramUniform1i (progCS, loc_output_w,   workload_x);    // 9
        glProgramUniform1i (progCS, loc_src_depth,  workload_z4);   // 256
        glProgramUniform1i (progCS, loc_weight_h,   workload_z4);   // 256
        glProgramUniform1i (progCS, loc_weight_w,   4);             // 4
        glProgramUniform1i (progCS, loc_workload_x, workload_x);    // 9
        glProgramUniform1i (progCS, loc_workload_y, workload_y);    // 9
        glProgramUniform1i (progCS, loc_workload_z, workload_z4);   // 256

        int num_group_x = (int)ceil((float)workload_x  / (float)local_size_x);
        int num_group_y = (int)ceil((float)workload_y  / (float)local_size_y);
        int num_group_z = (int)ceil((float)workload_z4 / (float)local_size_z);
        glDispatchCompute (num_group_x, num_group_y, num_group_z);

        glMemoryBarrier (GL_ALL_BARRIER_BITS);
        glFinish();
        GLASSERT();

        ttime1 = pmeter_get_time_ms ();
        ttime_sum += ttime1 - ttime0;

        int n = i + 1;
        if ((n % 100) == 0)
        {
            fprintf (stderr, "[%4d] Dispatch Time: %8.2f[ms]\n", n, ttime_sum / (float)n);
        }
#if 0
        glClearColor (0.5f, 0.5f, 0.5f, 1.0f);
        glClear (GL_COLOR_BUFFER_BIT);
        egl_swap();
#endif
    }

    return 0;
}

