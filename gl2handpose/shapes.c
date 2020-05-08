/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <math.h>
#include "shapes.h"
#include "assertgl.h"


#ifdef WIN32 
#define SQRT sqrt
#define COS  cos
#define SIN  sin
#else
#define SQRT sqrtf
#define COS  cosf
#define SIN  sinf
#endif

typedef void (*PFUNCTION)(float u, float v, float *x, float *y, float *z);

typedef struct shape_param_t
{
    int   nDivU, nDivV;
    float min_u, max_u;
    float min_v, max_v;
} shape_param_t;

static void
cross (float *vec1, float *vec2, float *dst)
{
    dst[0] = (vec1[1] * vec2[2]) - (vec1[2] * vec2[1]);
    dst[1] = (vec1[2] * vec2[0]) - (vec1[0] * vec2[2]);
    dst[2] = (vec1[0] * vec2[1]) - (vec1[1] * vec2[0]);
}

static float
length (float *vec)
{
    float x2 = vec[0] * vec[0];
    float y2 = vec[1] * vec[1];
    float z2 = vec[2] * vec[2];
    return (float)SQRT(x2 + y2 + z2);
}

static void
normalize( float *vec )
{
    float len = length(vec);
    vec[0] /= len;
    vec[1] /= len;
    vec[2] /= len;
}

static int 
get_num_faces (int nDivU, int nDivV )
{
    return (nDivU - 1) * (nDivV - 1) * 2;
}

static int
gen_shape_buffers (int nDivU, int nDivV, shape_obj_t *pshape)
{
    int bufSize = sizeof(unsigned short) * get_num_faces (nDivU, nDivV) * 3;
    unsigned short *pIndex = (unsigned short *)malloc (bufSize);
    if (pIndex == NULL)
        return -1;

    glGenBuffers (1, &pshape->vbo_vtx);
    glGenBuffers (1, &pshape->vbo_col);
    glGenBuffers (1, &pshape->vbo_nrm);
    glGenBuffers (1, &pshape->vbo_tng);
    glGenBuffers (1, &pshape->vbo_uv );
    glGenBuffers (1, &pshape->vbo_idx);

    for (int i = 0; i < nDivU - 1; i ++)
    {
        for (int j = 0; j < nDivV - 1; j ++)
        {
            int idx = (j * (nDivU - 1) + i) * 6;

            pIndex[idx + 0] = ( j ) * nDivU + ( i );
            pIndex[idx + 1] = ( j ) * nDivU + (i+1);
            pIndex[idx + 2] = (j+1) * nDivU + (i+1);
            pIndex[idx + 3] = ( j ) * nDivU + ( i );
            pIndex[idx + 4] = (j+1) * nDivU + (i+1);
            pIndex[idx + 5] = (j+1) * nDivU + ( i );
        }
    }

    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, pshape->vbo_idx);
    glBufferData (GL_ELEMENT_ARRAY_BUFFER, bufSize, pIndex, GL_STATIC_DRAW);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

    return 0;
}

static void
generate_shape (PFUNCTION function, shape_param_t *sparam, shape_obj_t *shape)
{
    int   i, j;
    float *pVertex, *pColor, *pUV, *pNormal, *pTangent;
    int   nSampleU = sparam->nDivU;
    int   nSampleV = sparam->nDivV;
    int   nVertex = nSampleU * nSampleV;
    float fMinU = sparam->min_u;
    float fMinV = sparam->min_v;
    float fMaxU = sparam->max_u;
    float fMaxV = sparam->max_v;

    gen_shape_buffers (nSampleU, nSampleV, shape);

    pVertex = malloc (sizeof(float) * nVertex * 3);
    pColor  = malloc (sizeof(float) * nVertex * 3);
    pNormal = malloc (sizeof(float) * nVertex * 3);
    pUV     = malloc (sizeof(float) * nVertex * 2);
    pTangent= malloc (sizeof(float) * nVertex * 3);

    for (i = 0; i < nSampleU; i ++)
    {
        for (j = 0; j < nSampleV; j ++)
        {
            float u = fMinU + i * (fMaxU-fMinU) / (float)(nSampleU-1);
            float v = fMinV + j * (fMaxV-fMinV) / (float)(nSampleV-1);
            float x,y,z;

            function (u, v, &x, &y, &z);

            pVertex[(j*nSampleU+i)*3 + 0] = x;
            pVertex[(j*nSampleU+i)*3 + 1] = y;
            pVertex[(j*nSampleU+i)*3 + 2] = z;
        }
    }

    for (i = 0; i < nSampleU; i ++)
    {
        for (j = 0; j < nSampleV; j ++)
        {
            pUV[(j*nSampleU+i)*2 + 0] = (float)i / (float)(nSampleU-1);
            pUV[(j*nSampleU+i)*2 + 1] = (float)j / (float)(nSampleV-1);
        }
    }

    for (i = 0; i < nSampleU; i ++)
    {
        for (j = 0; j < nSampleV; j ++)
        {
            pColor[(j*nSampleU+i)*3 + 0] =        (float)i / (float)(nSampleU-1);
            pColor[(j*nSampleU+i)*3 + 1] = 1.0f - (float)i / (float)(nSampleU-1);
            pColor[(j*nSampleU+i)*3 + 2] =        (float)j / (float)(nSampleV-1);
        }
    }

    for ( i = 0; i < nSampleU-1; i ++ )
    {
        for ( j = 0; j < nSampleV-1; j ++ )
        {
            float ptA[3], ptB[3], ptC[3], AB[3], AC[3], normal[3];

            ptA[0] = pVertex[(  j  *nSampleU+i  )*3+0];
            ptA[1] = pVertex[(  j  *nSampleU+i  )*3+1];
            ptA[2] = pVertex[(  j  *nSampleU+i  )*3+2];
            ptB[0] = pVertex[(  j  *nSampleU+i+1)*3+0];
            ptB[1] = pVertex[(  j  *nSampleU+i+1)*3+1];
            ptB[2] = pVertex[(  j  *nSampleU+i+1)*3+2];
            ptC[0] = pVertex[((j+1)*nSampleU+i  )*3+0];
            ptC[1] = pVertex[((j+1)*nSampleU+i  )*3+1];
            ptC[2] = pVertex[((j+1)*nSampleU+i  )*3+2];

            AB [0] = ptB[0] - ptA[0];
            AB [1] = ptB[1] - ptA[1];
            AB [2] = ptB[2] - ptA[2];
            AC [0] = ptC[0] - ptA[0];
            AC [1] = ptC[1] - ptA[1];
            AC [2] = ptC[2] - ptA[2];

            cross (AB, AC, normal);
            normalize (normal);

            pNormal[(j*nSampleU+i)*3 + 0] = -normal[0];
            pNormal[(j*nSampleU+i)*3 + 1] = -normal[1];
            pNormal[(j*nSampleU+i)*3 + 2] = -normal[2];

            normalize (AB);
            pTangent[(j*nSampleU+i)*3 + 0] = -AB[0];
            pTangent[(j*nSampleU+i)*3 + 1] = -AB[1];
            pTangent[(j*nSampleU+i)*3 + 2] = -AB[2];
        }
    }

    for (i = 0; i < nSampleU - 1; i ++)
    {
        pNormal[((nSampleV-1)*nSampleU+i)*3+0] = pNormal[(i)*3+0];
        pNormal[((nSampleV-1)*nSampleU+i)*3+1] = pNormal[(i)*3+1];
        pNormal[((nSampleV-1)*nSampleU+i)*3+2] = pNormal[(i)*3+2];

        pTangent[((nSampleV-1)*nSampleU+i)*3+0] = pTangent[(i)*3+0];
        pTangent[((nSampleV-1)*nSampleU+i)*3+1] = pTangent[(i)*3+1];
        pTangent[((nSampleV-1)*nSampleU+i)*3+2] = pTangent[(i)*3+2];
    }

    for (j = 0; j < nSampleV - 1; j ++)
    {
        pNormal[(j*nSampleU+nSampleU-1)*3+0] = pNormal[(j*nSampleU)*3+0];
        pNormal[(j*nSampleU+nSampleU-1)*3+1] = pNormal[(j*nSampleU)*3+1];
        pNormal[(j*nSampleU+nSampleU-1)*3+2] = pNormal[(j*nSampleU)*3+2];

        pTangent[(j*nSampleU+nSampleU-1)*3+0] = pTangent[(j*nSampleU)*3+0];
        pTangent[(j*nSampleU+nSampleU-1)*3+1] = pTangent[(j*nSampleU)*3+1];
        pTangent[(j*nSampleU+nSampleU-1)*3+2] = pTangent[(j*nSampleU)*3+2];
    }

    pNormal[((nSampleV-1)*nSampleU + (nSampleU-1))*3+0] = pNormal[((nSampleV-2)*nSampleU + (nSampleU-2))*3+0];
    pNormal[((nSampleV-1)*nSampleU + (nSampleU-1))*3+1] = pNormal[((nSampleV-2)*nSampleU + (nSampleU-2))*3+1];
    pNormal[((nSampleV-1)*nSampleU + (nSampleU-1))*3+2] = pNormal[((nSampleV-2)*nSampleU + (nSampleU-2))*3+2];

    pTangent[((nSampleV-1)*nSampleU + (nSampleU-1))*3+0]= pTangent[((nSampleV-2)*nSampleU + (nSampleU-2))*3+0];
    pTangent[((nSampleV-1)*nSampleU + (nSampleU-1))*3+1]= pTangent[((nSampleV-2)*nSampleU + (nSampleU-2))*3+1];
    pTangent[((nSampleV-1)*nSampleU + (nSampleU-1))*3+2]= pTangent[((nSampleV-2)*nSampleU + (nSampleU-2))*3+2];

    glBindBuffer (GL_ARRAY_BUFFER, shape->vbo_vtx );
    glBufferData (GL_ARRAY_BUFFER, nVertex * 3 * sizeof(float), pVertex, GL_STATIC_DRAW);

    glBindBuffer (GL_ARRAY_BUFFER, shape->vbo_col );
    glBufferData (GL_ARRAY_BUFFER, nVertex * 3 * sizeof(float), pColor,  GL_STATIC_DRAW);

    glBindBuffer (GL_ARRAY_BUFFER, shape->vbo_uv );
    glBufferData (GL_ARRAY_BUFFER, nVertex * 2 * sizeof(float), pUV,     GL_STATIC_DRAW);

    glBindBuffer (GL_ARRAY_BUFFER, shape->vbo_nrm );
    glBufferData (GL_ARRAY_BUFFER, nVertex * 3 * sizeof(float), pNormal, GL_STATIC_DRAW);

    glBindBuffer (GL_ARRAY_BUFFER, shape->vbo_tng );
    glBufferData (GL_ARRAY_BUFFER, nVertex * 3 * sizeof(float), pTangent, GL_STATIC_DRAW);

    glBindBuffer (GL_ARRAY_BUFFER, 0 );

    shape->num_faces = get_num_faces(nSampleU, nSampleV);

    free (pVertex);
    free (pColor );
    free (pNormal);
    free (pUV    );
}

static void func_Plan(float u,float v, float* x,float* y,float* z)
{
    *x = u;
    *y = 0;
    *z = v;
}

static void func_Moebius(float u,float v, float* x,float* y,float* z)
{
    float R = 9;
    *x = R * ((float) cos(v) + u * (float) COS(v / 0.5f) * (float) cos(v));
    *y = R * ((float) sin(v) + u * (float) COS(v / 0.5f) * (float) sin(v));
    *z = R * u * (float) sin(v / 0.5f);
}

static void func_Torus(float u,float v, float* x,float* y,float* z)
{
    float R=1.0f, r=2.0f;
    *x = R * (float) cos(v) * (r + (float) cos(u));
    *y = R * (float) sin(v) * (r + (float) cos(u));
    *z = R * (float) sin(u);
}

static void func_KleinBottle(float u,float v, float* x,float* y,float* z)
{
    float botx = (6-2)  * (float) COS(u) * (1 + (float) sin(u));
    float boty = (16-4) * (float) SIN(u);
    float rad  = (4-1)  * (1 - (float) cos(u)/2);

    if (u > 1.7 * M_PI)
    {
        *x = botx + rad * (float) cos(u) * (float) cos(v);
        *y = boty + rad * (float) sin(u) * (float) cos(v);
    }
    else if (u > M_PI)
    {
        *x = botx + rad * (float) cos(v+M_PI);
        *y = boty;
    }
    else
    {
        *x = botx + rad * (float) cos(u) * (float) cos(v);
        *y = boty + rad * (float) sin(u) * (float) cos(v);
    }

    *z = rad * (float) -sin(v);
    *y -= 2;
}

static void func_BoySurface(float u,float v, float* x,float* y,float* z)
{
    float a = (float) cos(u*0.5f) * (float) SIN(v);
    float b = (float) sin(u*0.5f) * (float) SIN(v);
    float c = (float) cos(v);
    *x = ((2*a*a-b*b-c*c) + 2*b*c*(b*b-c*c) + c*a*(a*a-c*c) + a*b*(b*b-a*a)) / 2;
    *y = ((b*b-c*c) + c*a*(c*c-a*a) + a*b*(b*b-a*a)) * (float) sqrt(3.0f) / 2;
    *z = (a+b+c) * ((a+b+c)*(a+b+c)*(a+b+c) + 4*(b-a)*(c-b)*(a-c))/8;
    *x*=10;
    *y*=10;
    *z*=10;
}

static void func_DiniSurface(float u,float v, float* x,float* y,float* z)
{
    *x = (float)  cos(u) * (float) sin(v);
    *y = (float) -cos(v) - (float) log((float) tan(v/2)) - .2f*u;
    *z = (float) -sin(u) * (float) sin(v);
    *x*=5;
    *y*=4;
    *z*=5;
    *y-=3;
}

static void func_Sphere(float u,float v, float* x,float* y,float* z)
{
    float R = 1;
    *x = R * SIN((0.5-v)*M_PI);
    *y = R * COS((0.5-v)*M_PI) * COS(u * 2 * M_PI);
    *z = R * COS((0.5-v)*M_PI) * SIN(u * 2 * M_PI);
}

static void func_Cylinder(float u,float v, float* x,float* y,float* z)
{
    float R = 1;
    *x = R * COS(u * 2 * M_PI);
    *y = R * SIN(u * 2 * M_PI);
    *z = R * (0.5-v) * 2;
}

/* -------------------------------------------------------------------------- *
 *  generate parametric shapes.
 * -------------------------------------------------------------------------- */
int
shape_create (int type, int nDivU, int nDivV, shape_obj_t *pshape)
{
    PFUNCTION func;
    shape_param_t sparam = {};
    sparam.nDivU = nDivU;
    sparam.nDivV = nDivV;
    sparam.min_u = 0.0f;
    sparam.max_u = 1.0f;
    sparam.min_v = 0.0f;
    sparam.max_v = 1.0f;

    switch( type )
    {
    case SHAPE_TORUS:
        func = func_Torus;
        sparam.max_u = 2 * M_PI;
        sparam.max_v = 2 * M_PI;
        break;
    case SHAPE_MOEBIUS:
        func = func_Moebius;
        sparam.min_u = -M_PI/6.0f;
        sparam.max_u =  M_PI/6.0f;
        sparam.min_v = 0;
        sparam.max_v = 2 * M_PI;
        break;
    case SHAPE_KLEINBOTTLE:
        func = func_KleinBottle; 
        sparam.max_u = 2 * M_PI;
        sparam.max_v = 2 * M_PI;
        break;
    case SHAPE_BOYSURFACE:
        func = func_BoySurface;
        sparam.min_u = 0.001f;
        sparam.max_u = M_PI;
        sparam.min_v = 0.001f;
        sparam.max_v = M_PI;
        break;
    case SHAPE_DINISURFACE:
        func = func_DiniSurface;
        sparam.min_u = 0;
        sparam.max_u = 4 * M_PI;
        sparam.min_v = 0.01f;
        sparam.max_v = 0.5 * M_PI;
        break;
    case SHAPE_SPHERE:
        func = func_Sphere;
        break;
    case SHAPE_CYLINDER:
        func = func_Cylinder;
        break;
    default:
        func = func_Plan;
    }

    generate_shape (func, &sparam, pshape);

    return 0;
}


