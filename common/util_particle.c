/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <GLES2/gl2.h>
#include <math.h>
#include "assertgl.h"
#include "util_particle.h"
#include "util_texture.h"
#include "util_render2d.h"
#include "util_debug.h"


int
add_particle_set (particle_system_t *psys, int i, char *png_fname, int num, float *color)
{
    int            texid, texw, texh;
    particle_set_t *pset = &(psys->pset[i]);
    particle_t     *p;

    load_png_texture (png_fname, &texid, &texw, &texh);

    p = (particle_t *)calloc (sizeof (particle_t), num);
    DBG_ASSERT (p, "alloc error");

    pset->texid        = texid;
    pset->texw         = texw;
    pset->texh         = texh;
    pset->color[0]     = color[0];
    pset->color[1]     = color[1];
    pset->color[2]     = color[2];
    pset->num_particle = num;
    pset->p            = p;

    return 0;
}


particle_system_t *
create_particle_system (int num_pset)
{
    particle_system_t   *psys;
    particle_set_t      *pset;

    psys = (particle_system_t *)calloc (sizeof (particle_system_t), 1);
    pset = (particle_set_t *)calloc (sizeof (particle_set_t), num_pset);
    DBG_ASSERT (psys, "alloc error");
    DBG_ASSERT (pset, "alloc error");

    psys->num_pset = num_pset;
    psys->pset     = pset;

    return psys;
}



static void
random_vec2d (float *x, float *y)
{
    float dx = (float)rand () / (float)RAND_MAX;    /* [ 0.0, 1.0] */
    float dy = (float)rand () / (float)RAND_MAX;
    dx = 2.0f * dx - 1.0f;                          /* [-1.0, 1.0] */
    dy = 2.0f * dy - 1.0f;
    float len = sqrtf (dx * dx + dy * dy);
    *x = dx / len;
    *y = dy / len;
}

int
emit_particle (particle_set_t *pset, float sx, float sy)
{
    int i;
    for (i = 0; i < pset->num_particle; i ++)
    {
        particle_t *p = &pset->p[i];
        if (p->alpha <= 0.0f)
        {
            float x, y;
            random_vec2d (&x, &y);

            p->pos[0]   = sx;
            p->pos[1]   = sy;
            p->dir[0]   = x;
            p->dir[1]   = y;
            p->velocity = 3.0f;
            p->alpha    = 1.0f;

            break;
        }
    }
    return 0;
}

int
update_particle (particle_system_t *psys, float x0, float y0)
{
    int i, iset;
    for (iset = 0; iset < psys->num_pset; iset ++)
    {
        particle_set_t *pset = &psys->pset[iset];
        if (pset == NULL)
            continue;

        emit_particle (pset, x0, y0);
        for (i = 0; i < pset->num_particle; i ++)
        {
            particle_t *p = &pset->p[i];
            p->pos[0] += (p->dir[0] * p->velocity);
            p->pos[1] += (p->dir[1] * p->velocity);
            p->alpha -= 0.02f;
        }
    }
    return 0;
}

int
render_particle (particle_system_t *psys)
{
    int i, iset;

    unsigned int blend_add  [] = {GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE};
    for (iset = 0; iset < psys->num_pset; iset ++)
    {
        particle_set_t *pset = &(psys->pset[iset]);
        if (pset == NULL)
            continue;

        float color[4];
        color[0] = pset->color[0];
        color[1] = pset->color[1];
        color[2] = pset->color[2];

        for (i = 0; i < pset->num_particle; i ++)
        {
            particle_t *p = &pset->p[i];
            if (p->alpha > 0.0f)
            {
                color[3] = p->alpha;
                float x = p->pos[0] - (pset->texw * 0.5);
                float y = p->pos[1] - (pset->texh * 0.5);
                draw_2d_texture_modulate (pset->texid, x, y, pset->texw, pset->texh, 0, color, blend_add);
            }
        }
    }
    
    return 0;
}
