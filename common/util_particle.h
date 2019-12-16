/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef _UTIL_PARTICLE_H_
#define _UTIL_PARTICLE_H_



typedef struct _particle_t
{
    float   pos[2];
    float   dir[2];
    float   velocity;
    float   alpha;
} particle_t;

typedef struct _particle_set_t
{
    int     texid;
    int     texw;
    int     texh;
    float   color[3];
    int     num_particle;
    particle_t *p;
} particle_set_t;

typedef struct _particle_system_t
{
    int     num_pset;
    particle_set_t *pset;

} particle_system_t;



particle_system_t *create_particle_system (int num_pset);
int add_particle_set (particle_system_t *psys, 
                      int i, char *png_fname, int num, float *color);
int update_particle (particle_system_t *psys, float x0, float y0);
int render_particle (particle_system_t *psys);

#endif /* _UTIL_PARTICLE_H_ */
