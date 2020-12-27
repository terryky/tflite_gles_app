/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <GLES2/gl2.h>
#include "util_pmeter.h"
#include "util_shader.h"

static int    s_laptime_idx[10] = {0};
static int    s_laptime_num[10] = {0};
static float  s_laptime_stack[10][PMETER_MAX_LAP_NUM];
static double s_last_laptime[10] = {0};

double
pmeter_get_time_ms ()
{
    struct timespec tv;
    clock_gettime (CLOCK_MONOTONIC, &tv);
    return  (tv.tv_sec*1000 + (float)tv.tv_nsec/1000000.0);
} 

void
pmeter_reset_lap (int id)
{
    s_laptime_idx[id] = 0;
    s_laptime_num[id] = 0;
}

void
pmeter_set_lap (int id)
{
    if (s_laptime_idx[id] >= PMETER_MAX_LAP_NUM)
        return;

    double laptime = pmeter_get_time_ms ();
    s_laptime_stack[id][s_laptime_idx[id]] = laptime - s_last_laptime[id];
    s_laptime_idx[id] ++;
    s_laptime_num[id] ++;

    s_last_laptime[id] = laptime;
}

static void
pmeter_get_laptime (int id, int *num, float **laptime)
{
    *num = s_laptime_num[id];
    *laptime = s_laptime_stack[id];

    s_laptime_num[id] = 0;
}


static char vs_pmeter[] = "                  \n\
attribute vec4 a_Vertex;                     \n\
uniform   vec4 u_Translate;                  \n\
uniform   vec4 u_PrjMul, u_PrjAdd;           \n\
                                             \n\
void main()                                  \n\
{                                            \n\
   vec4 pos;                                 \n\
   pos         = a_Vertex + u_Translate;     \n\
   pos         = pos * u_PrjMul;             \n\
   gl_Position = pos + u_PrjAdd;             \n\
}                                            \n";

static char fs_pmeter[] = "                  \n\
precision mediump float;                     \n\
uniform vec4  u_Color;                       \n\
void main()                                  \n\
{                                            \n\
   gl_FragColor = u_Color;                   \n\
}                                            \n";

#define PMETER_DPY_NUM  10
#define PMETER_NUM      4
#define PMETER_DATA_NUM 1000

static int      s_wndW, s_wndH;
static int      s_data_num;
static int      s_pm_idx[PMETER_DPY_NUM];
static GLuint   s_pm_prg;
static int      s_locVtxPM, s_locColPM, s_locTransPM;
static int      s_locPrjMulPM, s_locPrjAddPM;

static GLfloat s_vertPM[PMETER_DPY_NUM][PMETER_NUM][PMETER_DATA_NUM*2];

void init_pmeter (int win_w, int win_h, int data_num)
{
    int i, j, k;
    s_pm_prg = build_shader(vs_pmeter, fs_pmeter);

    s_locVtxPM    = glGetAttribLocation  (s_pm_prg, "a_Vertex" );
    s_locTransPM  = glGetUniformLocation (s_pm_prg, "u_Translate" );
    s_locPrjMulPM = glGetUniformLocation (s_pm_prg, "u_PrjMul" );
    s_locPrjAddPM = glGetUniformLocation (s_pm_prg, "u_PrjAdd" );
    s_locColPM    = glGetUniformLocation (s_pm_prg, "u_Color"  );

    if (data_num > PMETER_DATA_NUM)
        data_num = PMETER_DATA_NUM;

    for ( k = 0; k < PMETER_DPY_NUM; k ++)
    {
        s_pm_idx[k] = 0;

        for ( i = 0; i < PMETER_NUM; i ++ )
        {
            for ( j = 0; j < PMETER_DATA_NUM; j ++ )
            {
                s_vertPM[k][i][ 2 * j    ] = 0.0f;
                s_vertPM[k][i][ 2 * j + 1] = (float)j;
            }
        }
    }
    s_wndW = win_w;
    s_wndH = win_h;
    s_data_num = data_num;
}

static int set_pmeter_val (int dpy_id, int id, float val)
{
    if ( id >= PMETER_NUM )
        return -1;

    if ( dpy_id >= PMETER_DPY_NUM )
        return -1;

    s_vertPM[dpy_id][id][2 * s_pm_idx[dpy_id] + 0] = val;
    return 0;
}

int draw_pmeter_ex (int dpy_id, int x, int y, float scale)
{
    int i, num_time;
    float vert1[] = { 0.0f, 0.0f, 0.0f, (float)PMETER_DATA_NUM };
    float vert2[] = { 0.0f, 0.0f, 100.0f, 0.0f  };
    float *laptime,sumval;
    static int   s_ncnt[PMETER_DPY_NUM] = {0};
    static float s_lap[PMETER_DPY_NUM][10] = {{0}};

    pmeter_get_laptime (dpy_id, &num_time, &laptime);

    sumval = 0;
    for (i = 0; i < num_time; i ++)
    {
        s_lap[dpy_id][i] += laptime[i];
        sumval += laptime[i];
    }
    s_lap[dpy_id][i] += sumval;
    s_ncnt[dpy_id] ++;

    if (laptime[0] > 100.0f) laptime[0] = 100.0f;
    if (laptime[1] > 100.0f) laptime[1] = 100.0f;
    if (laptime[2] > 100.0f) laptime[2] = 100.0f;
    if (sumval     > 100.0f) sumval     = 100.0f;
    set_pmeter_val (dpy_id, 0, laptime[0]); /* BLUE:    render */
    set_pmeter_val (dpy_id, 1, laptime[1]); /* SKYBLUE: render */
    set_pmeter_val (dpy_id, 2, laptime[2]); /* SKYBLUE: render */
    set_pmeter_val (dpy_id, 3, sumval);     /* RED:     total  */

    s_pm_idx[dpy_id] ++;
    if (s_pm_idx[dpy_id] >= s_data_num)
        s_pm_idx[dpy_id] = 0;

    glUseProgram (s_pm_prg);

    glBindBuffer (GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray (s_locVtxPM);
    glUniform4f (s_locPrjMulPM, scale * 2.0f / (float)s_wndW, -2.0f / (float)s_wndH, 0.0f, 0.0f);
    glUniform4f (s_locPrjAddPM, -1.0f, 1.0f, 1.0f, 1.0f);

    glDisable (GL_DEPTH_TEST);
    glDisable (GL_CULL_FACE );
    glLineWidth (1.0f);

    /* AXIS */
    glVertexAttribPointer (s_locVtxPM, 2, GL_FLOAT, GL_FALSE, 0, vert1);
    glUniform4f (s_locColPM, 0.5f, 0.5f, 0.5f, 1.0f);
    for ( i = 1; i <= 10; i ++ )
    {
        glUniform4f (s_locTransPM, x + i * 10, y, 0.0f, 0.0f); 
        glDrawArrays (GL_LINES, 0, 2);
    }
    //glUniform4f (s_locColPM, 0.0f, 1.0f, 0.0f, 1.0f);
    //glUniform4f (s_locTransPM, x + 34, y, 0.0f, 0.0f); 
    //glDrawArrays (GL_LINES, 0, 2);

    /* GRAPH */
    glUniform4f (s_locTransPM, x, y, 0.0f, 0.0f );
    glVertexAttribPointer (s_locVtxPM, 2, GL_FLOAT, GL_FALSE, 0, s_vertPM[dpy_id][0]);
    glUniform4f (s_locColPM,   0.0f, 0.0f, 1.0f, 1.0f);
    glDrawArrays (GL_LINE_STRIP, 0, s_data_num);

    glVertexAttribPointer (s_locVtxPM, 2, GL_FLOAT, GL_FALSE, 0, s_vertPM[dpy_id][1]);
    glUniform4f (s_locColPM,   0.0f, 1.0f, 1.0f, 1.0f);
    glDrawArrays (GL_LINE_STRIP, 0, s_data_num);

    glVertexAttribPointer (s_locVtxPM, 2, GL_FLOAT, GL_FALSE, 0, s_vertPM[dpy_id][2]);
    glUniform4f (s_locColPM, 1.0f, 0.5f, 0.2f, 1.0f);
    glDrawArrays (GL_LINE_STRIP, 0, s_data_num);

    glVertexAttribPointer (s_locVtxPM, 2, GL_FLOAT, GL_FALSE, 0, s_vertPM[dpy_id][3]);
    glUniform4f (s_locColPM, 1.0f, 0.0f, 0.0f, 1.0f);
    glDrawArrays (GL_LINE_STRIP, 0, s_data_num);

    /* CURSOR */
    glLineWidth (3.0f);
    glVertexAttribPointer (s_locVtxPM, 2, GL_FLOAT, GL_FALSE, 0, vert2);
    glUniform4f (s_locColPM, 0.0f, 1.0f, 0.0f, 1.0f);
    glUniform4f (s_locTransPM, x, y + s_pm_idx[dpy_id], 0.0f, 0.0f); 
    glDrawArrays (GL_LINES, 0, 2);

    return 0;
}

int draw_pmeter (int x, int y)
{
    return draw_pmeter_ex (0, x, y, 1.0f);
}

