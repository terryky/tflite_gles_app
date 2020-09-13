/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define M_PId180f     (3.1415926f / 180.0f)
#include "util_matrix.h"

float
vec3_length (float *v)
{
    return (float) sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static float
vec2_length (float *v)
{
    return (float) sqrt(v[0] * v[0] + v[1] * v[1]);
}

float
vec3_normalize (float *v)
{
    float len, invLen;

    len = vec3_length (v);
    if (len == 0.0f)
    {
        return 0.0f;
    }    
    invLen = 1.0f / len;

    v[0] *= invLen;
    v[1] *= invLen;
    v[2] *= invLen;

    return len;
}

static float
vec2_normalize (float *v)
{
    float len, invLen;

    len = vec2_length (v);
    if (len == 0.0f)
    {
        return len;
    }
    invLen = 1.0f / len;

    v[0] *= invLen;
    v[1] *= invLen;

    return len;
}

static void
turn_x (float *m, float cosA, float sinA)
{
    float m01, m02;
    float m11, m12;
    float m21, m22;
    float m31, m32;
    float mx01, mx02;
    float mx11, mx12;
    float mx21, mx22;
    float mx31, mx32;

    m01 = m[4];  //m->m01;
    m02 = m[8];  //m->m02;
    m11 = m[5];  //m->m11;
    m12 = m[9];  //m->m12;
    m21 = m[6];  //m->m21;
    m22 = m[10]; //m->m22;
    m31 = m[7];  //m->m31;
    m32 = m[11]; //m->m32;

    mx01 = cosA * m01;
    mx02 = sinA * m01;

    mx11 = cosA * m11;
    mx12 = sinA * m11;

    mx21 = cosA * m21;
    mx22 = sinA * m21;

    mx31 = cosA * m31;
    mx32 = sinA * m31;

    mx01 = sinA * m02 + mx01;
    mx02 = cosA * m02 - mx02;

    mx11 = sinA * m12 + mx11;
    mx12 = cosA * m12 - mx12;

    mx21 = sinA * m22 + mx21;
    mx22 = cosA * m22 - mx22;

    mx31 = sinA * m32 + mx31;
    mx32 = cosA * m32 - mx32;

    m[4] = mx01;
    m[8] = mx02;

    m[5] = mx11;
    m[9] = mx12;

    m[6] = mx21;
    m[10] = mx22;

    m[7] = mx31;
    m[11] = mx32;
}

/*
 * void turn_y(float *m, float cosA, float cosB) 
 * local rotation around Y-axis
 * M = M * Ry
 *
 * | m00 m01 m02 m03 |   | m00 m01 m02 m03 |   |  cosA 0 sinA 0 |
 * | m10 m11 m12 m13 | = | m10 m11 m12 m13 | * |   0   1   0  0 |
 * | m20 m21 m22 m23 |   | m20 m21 m22 m23 |   | -sinA 0 cosA 0 |
 * | m30 m31 m32 m33 |   | m30 m31 m32 m33 |   |   0     0  0 1 |
 */ 
static void
turn_y (float *m, float cosA, float sinA)
{
    float m00, m02;
    float m10, m12;
    float m20, m22;
    float m30, m32;
    float mx00, mx02;
    float mx10, mx12;
    float mx20, mx22;
    float mx30, mx32;

    m00 = m[0];  //m->m00;
    m02 = m[8];  //m->m02;
    m10 = m[1];  //m->m10;
    m12 = m[9];  //m->m12;
    m20 = m[2];  //m->m20;
    m22 = m[10]; //m->m22;
    m30 = m[3];  //m->m30;
    m32 = m[11]; //m->m32;

    mx00 = cosA * m00;
    mx02 = sinA * m00;

    mx10 = cosA * m10;
    mx12 = sinA * m10;

    mx20 = cosA * m20;
    mx22 = sinA * m20;

    mx30 = cosA * m30;
    mx32 = sinA * m30;

    mx00 = -sinA * m02 + mx00;
    mx02 =  cosA * m02 + mx02;

    mx10 = -sinA * m12 + mx10;
    mx12 =  cosA * m12 + mx12;

    mx20 = -sinA * m22 + mx20;
    mx22 =  cosA * m22 + mx22;

    mx30 = -sinA * m32 + mx30;
    mx32 =  cosA * m32 + mx32;

    m[0] = mx00;
    m[8] = mx02;

    m[1] = mx10;
    m[9] = mx12;

    m[2] = mx20;
    m[10] = mx22;

    m[3] = mx30;
    m[11] = mx32;
}

/*
 * void turn_z(float *m, float cosA, float sinA) 
 * local rotation around Z-axis
 * M = M * Rz
 *
 * | m00 m01 m02 m03 |   | m00 m01 m02 m03 |   | cosA -sinA 0 0 |
 * | m10 m11 m12 m13 | = | m10 m11 m12 m13 | * | sinA  cosA 0 0 |
 * | m20 m21 m22 m23 |   | m20 m21 m22 m23 |   |   0     0  1 0 |
 * | m30 m31 m32 m33 |   | m30 m31 m32 m33 |   |   0     0  0 1 |
 *
 */ 
static void
turn_z (float *m, float cosA, float sinA)
{
    float m00, m01;
    float m10, m11;
    float m20, m21;
    float m30, m31;
    float mx00, mx01;
    float mx10, mx11;
    float mx20, mx21;
    float mx30, mx31;

    m00 = m[0];  //m->m00;
    m01 = m[4];  //m->m01;
    m10 = m[1];  //m->m10;
    m11 = m[5];  //m->m11;
    m20 = m[2];  //m->m20;
    m21 = m[6];  //m->m21;
    m30 = m[3];  //m->m30;
    m31 = m[7];  //m->m31;

    mx00 = cosA * m00;
    mx01 = sinA * m00;

    mx10 = cosA * m10;
    mx11 = sinA * m10;

    mx20 = cosA * m20;
    mx21 = sinA * m20;

    mx30 = cosA * m30;
    mx31 = sinA * m30;

    mx00 = sinA * m01 + mx00;
    mx01 = cosA * m01 - mx01;

    mx10 = sinA * m11 + mx10;
    mx11 = cosA * m11 - mx11;

    mx20 = sinA * m21 + mx20;
    mx21 = cosA * m21 - mx21;

    mx30 = sinA * m31 + mx30;
    mx31 = cosA * m31 - mx31;

    m[0] = mx00;
    m[4] = mx01;
    m[1] = mx10;
    m[5] = mx11;
    m[2] = mx20;
    m[6] = mx21;
    m[3] = mx30;
    m[7] = mx31;
}

/************************************************************
   Translate Matrix
   M = M * T
  
    | m00  m04  m08  m12 |   | 1  0  0  x |   | m00  m04  m08  (m00*x + m04*y + m08*z + m12) |
    | m01  m05  m09  m13 | * | 0  1  0  y | = | m01  m05  m09  (m01*x + m05*y + m09*z + m13) |
    | m02  m06  m10  m14 |   | 0  0  1  z |   | m02  m06  m10  (m02*x + m06*y + m10*z + m14) |
    | m03  m07  m11  m15 |   | 0  0  0  1 |   | m03  m07  m11  (m03*x + m07*y + m11*z + m15) |
***********************************************************/
void
matrix_translate (float *m, float x, float y, float z)
{
    float m00, m01, m02, m03;
    float m04, m05, m06, m07;
    float m08, m09, m10, m11;
    float m12, m13, m14, m15;

    m00 = m[ 0]; m04 = m[ 4]; m08 = m[ 8]; /* m12 = m[12]; */
    m01 = m[ 1]; m05 = m[ 5]; m09 = m[ 9]; /* m13 = m[13]; */
    m02 = m[ 2]; m06 = m[ 6]; m10 = m[10]; /* m14 = m[14]; */
    m03 = m[ 3]; m07 = m[ 7]; m11 = m[11]; /* m15 = m[15]; */

    m12 = m[12];
    m13 = m[13];
    m14 = m[14];
    m15 = m[15];

    m12 += m08 * z;
    m13 += m09 * z;
    m14 += m10 * z;
    m15 += m11 * z;

    m12 += m04 * y;
    m13 += m05 * y;
    m14 += m06 * y;
    m15 += m07 * y;

    m12 += m00 * x;
    m13 += m01 * x;
    m14 += m02 * x;
    m15 += m03 * x;

    m[12] = m12;
    m[13] = m13;
    m[14] = m14;
    m[15] = m15;
}

/************************************************************
  Rotate Matrix
    | m00  m04  m08  m12 |   | r00  r04  r08   0 |
    | m01  m05  m09  m13 | * | r01  r05  r09   0 |
    | m02  m06  m10  m14 |   | r02  r06  r10   0 |
    | m03  m07  m11  m15 |   |   0    0    0   1 |
    
    m00 = m00*r00 + m04*r01 + m08*r02
    m01 = m01*r00 + m05*r01 + m09*r02
    m02 = m02*r00 + m06*r01 + m10*r02
    m03 = m03*r00 + m07*r01 + m11*r02
    
    m04 = m00*r04 + m04*r05 + m08*r06
    m05 = m01*r04 + m05*r05 + m09*r06
    m06 = m02*r04 + m06*r05 + m10*r06
    m07 = m03*r04 + m07*r05 + m11*r06
    
    m08 = m00*r08 + m04*r09 + m08*r10
    m09 = m01*r08 + m05*r09 + m09*r10
    m10 = m02*r08 + m06*r09 + m10*r10
    m11 = m03*r08 + m07*r09 + m11*r10
    
    m12 = m12
    m13 = m13
    m14 = m14
    m15 = m15
***********************************************************/
void
matrix_rotate (float *m, float angle, float x, float y, float z)
{
    float v[3], angleRadian;
    float sinA, cosA, cosA2;
    float xcosA2, ycosA2, zcosA2;
    float xsinA, ysinA, zsinA;

    angleRadian = angle * M_PId180f;
    sinA   = (float)sin(angleRadian);
    cosA   = (float)cos(angleRadian);

    /* for fast rotation around X-Axis/Y-Axis,and Z-Axis */
    if (x == 0.0f && y == 0.0f && z != 0.0f)
    {
        if (z < 0.0f)
        {
            /* If the Axis of the Rotation is minus, Rotate Backwords */
            sinA = -sinA;
        }
        /* Z Axis Rotateion */
        turn_z (m, cosA, sinA);
        return;
    }
    else if (x == 0.0f && y != 0.0f && z == 0.0f)
    {
        if (y < 0.0f)
        {
            /* If the Axis of the Rotation is minus, Rotate Backwords */
            sinA = -sinA;
        }
        /* Y Axis Rotation */
        turn_y (m, cosA, sinA);
        return;
    }
    else if (x != 0.0f && y == 0.0f && z == 0.0f)
    {
        if (x < 0.0f)
        {
            /* If the Axis of the Rotation is minus, Rotate Backwords */
            sinA = -sinA;
        }
        /* X Axis Rotation */
        turn_x (m, cosA, sinA);
        return;
    }

    {
        float r00, r01, r02;
        float r10, r11, r12;
        float r20, r21, r22;

        /* normalization of 3D-vector */
        v[0] = x;
        v[1] = y;
        v[2] = z;
        vec3_normalize (v);

        x = v[0];
        y = v[1];
        z = v[2];

        /* making rotation matrix */
        cosA2  = 1.0f - cosA;
        xsinA  = x * sinA;
        ysinA  = y * sinA;
        zsinA  = z * sinA;
        xcosA2 = x * cosA2;
        ycosA2 = y * cosA2;
        zcosA2 = z * cosA2;

        r00  = x * xcosA2 + cosA;
        r10  = y * xcosA2 + zsinA;
        r20  = z * xcosA2 - ysinA;

        r01  = x * ycosA2 - zsinA;
        r11  = y * ycosA2 + cosA;
        r21  = z * ycosA2 + xsinA;

        r02  = x * zcosA2 + ysinA;
        r12  = y * zcosA2 - xsinA;
        r22  = z * zcosA2 + cosA;

        /* multing with 3x3 rotating matrix. */
        {
            float fm0, fm1, fm2;
            float mx, my, mz;

            /* load 0th low of "m" */
            fm0 = m[ 0]; fm1 = m[ 4]; fm2 = m[ 8]; /* fm3 = m[12]; */

            mx = fm0 * r00;
            my = fm0 * r01;
            mz = fm0 * r02;

            mx += fm1 * r10;
            my += fm1 * r11;
            mz += fm1 * r12;

            mx += fm2 * r20;
            my += fm2 * r21;
            mz += fm2 * r22;

            fm0 = m[ 1]; fm1 = m[ 5]; fm2 = m[ 9]; /* fm3 = m[13]; */

            m[0] = mx; m[4] = my; m[8] = mz;

            /* *************************** */
            mx = fm0 * r00;
            my = fm0 * r01;
            mz = fm0 * r02;

            mx += fm1 * r10;
            my += fm1 * r11;
            mz += fm1 * r12;

            mx += fm2 * r20;
            my += fm2 * r21;
            mz += fm2 * r22;

            fm0 = m[ 2]; fm1 = m[ 6]; fm2 = m[10]; /* m23 = m[14]; */

            m[1] = mx; m[5] = my; m[9] = mz;

            /* *************************** */
            mx = fm0 * r00;
            my = fm0 * r01;
            mz = fm0 * r02;

            mx += fm1 * r10;
            my += fm1 * r11;
            mz += fm1 * r12;

            mx += fm2 * r20;
            my += fm2 * r21;
            mz += fm2 * r22;

            fm0 = m[ 3]; fm1 = m[ 7]; fm2 = m[11]; /* m33 = m[15]; */

            m[2] = mx; m[6] = my; m[10] = mz;

            /* *************************** */
            mx = fm0 * r00;
            my = fm0 * r01;
            mz = fm0 * r02;

            mx += fm1 * r10;
            my += fm1 * r11;
            mz += fm1 * r12;

            mx += fm2 * r20;
            my += fm2 * r21;
            mz += fm2 * r22;

            m[3] = mx; m[7] = my; m[11] = mz;
        }
    }
}

/******************************************
   Scale Matrix
    | m00  m04  m08  m12 |   | x  0  0  0 |
    | m01  m05  m09  m13 | * | 0  y  0  0 |
    | m02  m06  m10  m14 |   | 0  0  z  0 |
    | m03  m07  m11  m15 |   | 0  0  0  1 |
*******************************************/
void
matrix_scale (float *m, float x, float y, float z)
{
    float m00, m01, m02, m03;
    float m04, m05, m06, m07;
    float m08, m09, m10, m11;
    /* float m12, m13, m14, m15; */

    m00 = m[ 0]; m04 = m[ 4]; m08 = m[ 8]; /* m12 = m[12]; */
    m01 = m[ 1]; m05 = m[ 5]; m09 = m[ 9]; /* m13 = m[13]; */
    m02 = m[ 2]; m06 = m[ 6]; m10 = m[10]; /* m14 = m[14]; */
    m03 = m[ 3]; m07 = m[ 7]; m11 = m[11]; /* m15 = m[15]; */

    m00 = m00 * x;
    m04 = m04 * y;
    m08 = m08 * z;

    m01 = m01 * x;
    m05 = m05 * y;
    m09 = m09 * z;

    m02 = m02 * x;
    m06 = m06 * y;
    m10 = m10 * z;

    m03 = m03 * x;
    m07 = m07 * y;
    m11 = m11 * z;

    m[ 0] = m00;
    m[ 4] = m04;
    m[ 8] = m08;

    m[ 1] = m01;
    m[ 5] = m05;
    m[ 9] = m09;

    m[ 2] = m02;
    m[ 6] = m06;
    m[10] = m10;

    m[ 3] = m03;
    m[ 7] = m07;
    m[11] = m11;
}

/******************************************
   Scale Skew
    | m00  m04  m08  m12 |   | 1          tan(skewX)  0  0 |
    | m01  m05  m09  m13 | * | tan(skewY) 1           0  0 |
    | m02  m06  m10  m14 |   | 0          0           1  0 |
    | m03  m07  m11  m15 |   | 0          0           0  1 |
*******************************************/
void
matrix_skew (float *m, float x, float y)
{
    float angleRadianX, angleRadianY;
    float tanX, tanY;
    float m00, m01, m02, m03;
    float m04, m05, m06, m07;
    //float m08, m09, m10, m11;
    //float m12, m13, m14, m15;

    angleRadianX = x * M_PId180f;
    angleRadianY = y * M_PId180f;
    tanX = (float)tan(angleRadianX);
    tanY = (float)tan(angleRadianY);

    m00 = m[ 0]; m04 = m[ 4]; /* m08 = m[ 8]; m12 = m[12]; */
    m01 = m[ 1]; m05 = m[ 5]; /* m09 = m[ 9]; m13 = m[13]; */
    m02 = m[ 2]; m06 = m[ 6]; /* m10 = m[10]; m14 = m[14]; */
    m03 = m[ 3]; m07 = m[ 7]; /* m11 = m[11]; m15 = m[15]; */

    m00 += m04 * tanY;
    m01 += m05 * tanY;
    m02 += m06 * tanY;
    m03 += m07 * tanY;

    m04 += m00 * tanX;
    m05 += m01 * tanX;
    m06 += m02 * tanX;
    m07 += m03 * tanX;

    m[ 0] = m00;
    m[ 1] = m01;
    m[ 2] = m02;
    m[ 3] = m03;

    m[ 4] = m04;
    m[ 5] = m05;
    m[ 6] = m06;
    m[ 7] = m07;
}

/******************************************
   Model Lookat Matrix
*******************************************/
void
matrix_modellookat (float *m, float *src_pos, float *tgt_pos, float twist)
{
    float deltaV[2];
    float cosA, sinA, cosB, sinB;

    deltaV[0] = tgt_pos[0] - src_pos[0];    // dx
    deltaV[1] = tgt_pos[2] - src_pos[2];    // dz

    if (deltaV[0] != 0.0f || deltaV[1] != 0.0f)
    {
        float lenXZ = vec2_normalize (deltaV);
        cosA = deltaV[1];
        sinA = deltaV[0];

        deltaV[0] = lenXZ;
        deltaV[1] = tgt_pos[1] - src_pos[1];
        vec2_normalize (deltaV);
        cosB = deltaV[0];
        sinB = deltaV[1];
    }
    else /* if ((dx == 0) && (dz == 0)) */
    {
        cosA = -1.0f;
        sinA =  0.0f;

        cosB =  0.0f;
        sinB = (tgt_pos[1] < 0.0f) ? -1.0f : 1.0f;
    }

    if (twist == 0.0f)
    {
        m[0] = cosA;            // m00
        m[4] = sinA * (-sinB);  // m01
        m[8] = sinA *   cosB;   // m02

        m[1] = 0.0f;            // m10
        m[5] = cosB;            // m11
        m[9] = sinB;            // m12

        m[2] = -sinA;           // m20
        m[6] = cosA * (-sinB);  // m21
        m[10]= cosA *   cosB;   // m22
    }
    else /* if (twist != 0) */
    {
        float cosC, sinC, mSinAsinB, mCosAsinB;

        cosC = (float)cos(twist);
        sinC = (float)sin(twist);

        mSinAsinB = sinA * (-sinB);
        mCosAsinB = cosA * (-sinB);

        m[0] =  cosA * cosC + mSinAsinB * sinC;
        m[4] = -cosA * sinC + mSinAsinB * cosC;
        m[8] =  sinA * cosB;

        m[1] =  cosB * sinC;
        m[5] =  cosB * cosC;
        m[9] =  sinB;

        m[2] = -sinA * cosC + mCosAsinB * sinC;
        m[6] =  sinA * sinC + mCosAsinB * cosC;
        m[10]=  cosA * cosB;
    }

    m[12] = src_pos[0]; // m03
    m[13] = src_pos[1]; // m13
    m[14] = src_pos[2]; // m23

    m[ 3] = 0.0f;       // m30
    m[ 7] = 0.0f;       // m31
    m[11] = 0.0f;       // m32
    m[15] = 1.0f;       // m33
}



/******************************************
   Multiply Matrix
     M = M1 * M2
*******************************************/
void
matrix_mult (float *m, float *m1, float *m2)
{
    float fm0, fm1, fm2, fm3;
    float fpm00, fpm01, fpm02, fpm03;
    float fpm10, fpm11, fpm12, fpm13;
    float fpm20, fpm21, fpm22, fpm23;
    float fpm30, fpm31, fpm32, fpm33;
    float x, y, z, w;

    /* load pMb */
    fpm00 = m2[0];
    fpm01 = m2[4];
    fpm02 = m2[8];
    fpm03 = m2[12];

    fpm10 = m2[1];
    fpm11 = m2[5];
    fpm12 = m2[9];
    fpm13 = m2[13];

    fpm20 = m2[2];
    fpm21 = m2[6];
    fpm22 = m2[10];
    fpm23 = m2[14];

    fpm30 = m2[3];
    fpm31 = m2[7];
    fpm32 = m2[11];
    fpm33 = m2[15];

    /*  process 0-line of "m1" */
    fm0 = m1[0];
    fm1 = m1[4];
    fm2 = m1[8];
    fm3 = m1[12];

    x = fm0 * fpm00;
    y = fm0 * fpm01;
    z = fm0 * fpm02;
    w = fm0 * fpm03;

    x += fm1 * fpm10;
    y += fm1 * fpm11;
    z += fm1 * fpm12;
    w += fm1 * fpm13;

    x += fm2 * fpm20;
    y += fm2 * fpm21;
    z += fm2 * fpm22;
    w += fm2 * fpm23;

    x += fm3 * fpm30;
    y += fm3 * fpm31;
    z += fm3 * fpm32;
    w += fm3 * fpm33;

    fm0 = m1[1];
    fm1 = m1[5];
    fm2 = m1[9];
    fm3 = m1[13];

    m[0] = x;
    m[4] = y;
    m[8] = z;
    m[12] = w;

    /* *************************** */
    x = fm0 * fpm00;
    y = fm0 * fpm01;
    z = fm0 * fpm02;
    w = fm0 * fpm03;

    x += fm1 * fpm10;
    y += fm1 * fpm11;
    z += fm1 * fpm12;
    w += fm1 * fpm13;

    x += fm2 * fpm20;
    y += fm2 * fpm21;
    z += fm2 * fpm22;
    w += fm2 * fpm23;

    x += fm3 * fpm30;
    y += fm3 * fpm31;
    z += fm3 * fpm32;
    w += fm3 * fpm33;

    fm0 = m1[2];
    fm1 = m1[6];
    fm2 = m1[10];
    fm3 = m1[14];

    m[1] = x;
    m[5] = y;
    m[9] = z;
    m[13] = w;

    /* *************************** */
    x = fm0 * fpm00;
    y = fm0 * fpm01;
    z = fm0 * fpm02;
    w = fm0 * fpm03;

    x += fm1 * fpm10;
    y += fm1 * fpm11;
    z += fm1 * fpm12;
    w += fm1 * fpm13;

    x += fm2 * fpm20;
    y += fm2 * fpm21;
    z += fm2 * fpm22;
    w += fm2 * fpm23;

    x += fm3 * fpm30;
    y += fm3 * fpm31;
    z += fm3 * fpm32;
    w += fm3 * fpm33;

    fm0 = m1[3];
    fm1 = m1[7];
    fm2 = m1[11];
    fm3 = m1[15];

    m[2] = x;
    m[6] = y;
    m[10] = z;
    m[14] = w;

    /* *************************** */
    x = fm0 * fpm00;
    y = fm0 * fpm01;
    z = fm0 * fpm02;
    w = fm0 * fpm03;

    x += fm1 * fpm10;
    y += fm1 * fpm11;
    z += fm1 * fpm12;
    w += fm1 * fpm13;

    x += fm2 * fpm20;
    y += fm2 * fpm21;
    z += fm2 * fpm22;
    w += fm2 * fpm23;

    x += fm3 * fpm30;
    y += fm3 * fpm31;
    z += fm3 * fpm32;
    w += fm3 * fpm33;

    m[3] = x;
    m[7] = y;
    m[11] = z;
    m[15] = w;
}



void
matrix_identity (float *m)
{
    m[ 0] = 1.0f; m[ 4] = 0.0f; m[ 8] = 0.0f; m[12] = 0.0f;
    m[ 1] = 0.0f; m[ 5] = 1.0f; m[ 9] = 0.0f; m[13] = 0.0f;
    m[ 2] = 0.0f; m[ 6] = 0.0f; m[10] = 1.0f; m[14] = 0.0f;
    m[ 3] = 0.0f; m[ 7] = 0.0f; m[11] = 0.0f; m[15] = 1.0f;
}



void
matrix_perspective (float *m, float depth)
{
    m[8]  += -1.0f/depth * m[12];
    m[9]  += -1.0f/depth * m[13];
    m[10] += -1.0f/depth * m[14];
    m[11] += -1.0f/depth * m[15];
}

void
matrix_projectto2d (float *m)
{
    m[ 2] = 0.0f;
    m[ 6] = 0.0f;
    m[ 8] = 0.0f;
    m[ 9] = 0.0f;
    m[10] = 1.0f;
    m[11] = 0.0f;
    m[14] = 0.0f;
}


int
matrix_isidentity (float *m)
{
    if (m[ 0] != 1.0f || m[ 4] != 0.0f || m[ 8] != 0.0f || m[12] != 0.0f ||
        m[ 1] != 0.0f || m[ 5] != 1.0f || m[ 9] != 0.0f || m[13] != 0.0f ||
        m[ 2] != 0.0f || m[ 6] != 0.0f || m[10] != 1.0f || m[14] != 0.0f ||
        m[ 3] != 0.0f || m[ 7] != 0.0f || m[11] != 0.0f || m[15] != 1.0f )
    {
        return 0;
    }
    return 1;
}


int
matrix_is2d (float *m)
{
    /*
         [m00]  [m04]  m08  [m12]
         [m01]  [m05]  m09  [m13]
          m02    m06   m10   m14
          m03    m07   m11   m15
    */
    if (m[ 2] != 0.0f || m[ 3] != 0.0f ||
        m[ 6] != 0.0f || m[ 7] != 0.0f ||
        m[ 8] != 0.0f || m[ 9] != 0.0f ||
        m[10] != 1.0f || m[11] != 0.0f ||
        m[14] != 0.0f || m[15] != 1.0f)
    {
        return 0;
    }
    return 1;
}

int
matrix_iszero (float *m)
{
  /*
      m00   m04   m08  [m12]
      m01   m05   m09  [m13]
      m02   m06   m10  [m14]
      m03   m07   m11  [m15]
  */
    int i;
    for (i = 0; i < 12; i ++)
    {
        if (m[i] != 0.0f)
        return 0;
    }
    return 1;
}


/* does this matrix keep rectangle shape ?
 *   - only has 2d scale or/and translation .
 *   - doesn't have rotation or/and skew.
 */
int
matrix_is2d_scale_trans (float *m)
{
    if (matrix_iszero (m))
        return 1;

    if (!matrix_is2d (m) ||
        m[1] != 0.0f || 
        m[4] != 0.0f)
    {
        return 0;
    }
    return 1;
}


/* 
 *  (dst_vec) = (matrix)*(src_vec)
 *    - accept (dst_vec) == (src_vec)
 */
void
matrix_multvec2 (float *m, float *svec, float *dvec)
{
    float v0 = svec[0];
    float v1 = svec[1];
    float _d0, _d1, m00, m01, m04, m05, m12, m13;

    m00 = m[ 0];
    m01 = m[ 1];
    m04 = m[ 4];
    m05 = m[ 5];
    m12 = m[12];
    m13 = m[13];

    _d0 = m12;
    _d1 = m13;
    _d0 += m04 * v1;
    _d1 += m05 * v1;
    _d0 += m00 * v0;
    _d1 += m01 * v0;
    dvec[0] = _d0;
    dvec[1] = _d1;
}

void
matrix_multvec4 (float *m, float *svec, float *dvec)
{
    float v0 = svec[0];
    float v1 = svec[1];
    float v2 = svec[2];
    float v3 = svec[3];
    float m00, m01, m02, m03;
    float m04, m05, m06, m07;
    float m08, m09, m10, m11;
    float m12, m13, m14, m15;
    float _d0, _d1, _d2, _d3;

    m00 = m[ 0]; m04 = m[ 4]; m08 = m[ 8]; m12 = m[12];
    m01 = m[ 1]; m05 = m[ 5]; m09 = m[ 9]; m13 = m[13];
    m02 = m[ 2]; m06 = m[ 6]; m10 = m[10]; m14 = m[14];
    m03 = m[ 3]; m07 = m[ 7]; m11 = m[11]; m15 = m[15];

    _d0 = m12 * v3;
    _d1 = m13 * v3;
    _d2 = m14 * v3;
    _d3 = m15 * v3;
    _d0 += m08 * v2;
    _d1 += m09 * v2;
    _d2 += m10 * v2;
    _d3 += m11 * v2;
    _d0 += m04 * v1;
    _d1 += m05 * v1;
    _d2 += m06 * v1;
    _d3 += m07 * v1;
    _d0 += m00 * v0;
    _d1 += m01 * v0;
    _d2 += m02 * v0;
    _d3 += m03 * v0;
    dvec[0] = _d0;
    dvec[1] = _d1;
    dvec[2] = _d2;
    dvec[3] = _d3;
}


void
matrix_print (float *m)
{
    printf ("%10.5f %10.5f %10.5f %10.5f\n", m[ 0], m[ 4], m[ 8], m[12]);
    printf ("%10.5f %10.5f %10.5f %10.5f\n", m[ 1], m[ 5], m[ 9], m[13]);
    printf ("%10.5f %10.5f %10.5f %10.5f\n", m[ 2], m[ 6], m[10], m[14]);
    printf ("%10.5f %10.5f %10.5f %10.5f\n", m[ 3], m[ 7], m[11], m[15]);
}

void
matrix_copy (float *d, float *s)
{
    memcpy( d, s, sizeof(float) * 16 );
}

void
matrix_proj_frustum (float *mat, float left, float right, float bottom, float top, float znear, float zfar)
{
    memset( mat, 0x0, sizeof(float) * 16 );

    mat[ 0] =    2.0f  * znear    / ( right - left   );
    mat[ 5] =    2.0f  * znear    / ( top   - bottom );
    mat[ 8] =  ( right + left   ) / ( right - left   );
    mat[ 9] =  ( top   + bottom ) / ( top   - bottom );
    mat[10] = -( zfar  + znear  ) / ( zfar  - znear  );
    mat[11] = -1.0f;     
    mat[14] = -(2.0f * zfar * znear) / (zfar - znear);
}

void
matrix_proj_perspective (float *mat, float fovy, float aspect, float znear, float zfar)
{
    float a,b,c,d,f;

    memset( mat, 0x0, sizeof(float) * 16 );

    f = (float)(cos(2.0f*3.14f*(fovy/2.0f)/360.0f)/sin(2.0f*3.14f*(fovy/2.0f)/360.0f));
    a = f/aspect;                        /* a = f/aspect                */
    b = f;                               /* b = f                       */
    c = (zfar+znear)/(znear-zfar);       /* c = (far+near)/(near-far)   */
    d = (2.0f*zfar*znear)/(znear-zfar);  /* d = (2*far*near)/(near-far) */

    mat[ 0] = a;
    mat[ 5] = b;
    mat[10] = c;
    mat[11] = -1.0f;
    mat[14] = d;
}

void
matrix_proj_ortho (float *mat, float left, float right, float bottom, float top, float znear, float zfar)
{
	memset (mat, 0x0, sizeof(float) * 16);

    mat[ 0] =  2.0f / (right - left  );
    mat[ 5] =  2.0f / (top   - bottom);
    mat[10] = -2.0f / (zfar  - znear );
    mat[12] = -(right + left  ) / (right - left  );
    mat[13] = -(top   + bottom) / (top   - bottom);
    mat[14] = -(zfar  + znear ) / (zfar  - znear );
    mat[15] = 1.0f;
}

void
matrix_transpose (float *m)
{
    float      m01, m02, m03;
    float m04,      m06, m07;
    float m08, m09,      m11;
    float m12, m13, m14;

               ; m04 = m[ 4]; m08 = m[ 8]; m12 = m[12];
    m01 = m[ 1];              m09 = m[ 9]; m13 = m[13];
    m02 = m[ 2]; m06 = m[ 6];              m14 = m[14];
    m03 = m[ 3]; m07 = m[ 7]; m11 = m[11];

    /*m[ 0] = m00;*/
    m[ 1] = m04;
    m[ 2] = m08;
    m[ 3] = m12;
    m[ 4] = m01;
    /*m[ 5] = m05;*/
    m[ 6] = m09;
    m[ 7] = m13;
    m[ 8] = m02;
    m[ 9] = m06;
    /*m[10] = m10;*/
    m[11] = m14;
    m[12] = m03;
    m[13] = m07;
    m[14] = m11;
    /*m[15] = m15;*/
}


void
matrix_invert (float *m)
{
    float m00, m01, m02, m03;
    float m04, m05, m06, m07;
    float m08, m09, m10, m11;
    float m12, m13, m14, m15;
    float W00, W04, W08, W12;
    float W01, W05, W09, W13;
    float W02, W06, W10, W14;
    float W03, W07, W11, W15;
    float det, invdet;

    m00 = m[ 0]; m04 = m[ 4]; m08 = m[ 8]; m12 = m[12];
    m01 = m[ 1]; m05 = m[ 5]; m09 = m[ 9]; m13 = m[13];
    m02 = m[ 2]; m06 = m[ 6]; m10 = m[10]; m14 = m[14];
    m03 = m[ 3]; m07 = m[ 7]; m11 = m[11]; m15 = m[15];

    if (m03 == 0.0f && m07 == 0.0f && m11 == 0.0f && m15 == 1.0f)
    {
        W00 =   m05 * m10 - m09 * m06;
        W04 = -(m01 * m10 - m09 * m02);
        W08 =   m01 * m06 - m05 * m02;

        det = m00 * W00 + m04 * W04 + m08 * W08;

        if ( det == 0.0f )
        {
            return;
        }
        invdet = 1.0f / det;

        W01 = -(m04 * m10 - m08 * m06);
        W05 =   m00 * m10 - m08 * m02;
        W09 = -(m00 * m06 - m04 * m02);

        W02 =   m04 * m09  - m08 * m05;
        W06 = -(m00 * m09  - m08 * m01);
        W10 =   m00 * m05  - m04 * m01;

        W03 = -(W00 * m12 + W01 * m13 + W02 * m14);
        W07 = -(W04 * m12 + W05 * m13 + W06 * m14);
        W11 = -(W08 * m12 + W09 * m13 + W10 * m14);

        /* M^-1[ij] = invdet * M[ji] */
        m[ 0]  = W00 * invdet;
        m[ 4]  = W01 * invdet;
        m[ 8]  = W02 * invdet;
        m[12]  = W03 * invdet;

        m[ 1]  = W04 * invdet;
        m[ 5]  = W05 * invdet;
        m[ 9]  = W06 * invdet;
        m[13]  = W07 * invdet;

        m[ 2]  = W08 * invdet;
        m[ 6]  = W09 * invdet;
        m[10]  = W10 * invdet;
        m[14]  = W11 * invdet;

        m[ 3]  = 0.0f;
        m[ 7]  = 0.0f;
        m[11]  = 0.0f;
        m[15]  = 1.0f;
    }
    else
    {
        W00 = (m05 * (m10 * m15 - m14 * m11))
            + (m09 * (m14 * m07 - m06 * m15))
            + (m13 * (m06 * m11 - m10 * m07));
        W01 = (m09 * (m14 * m03 - m02 * m15))
            + (m13 * (m02 * m11 - m10 * m03))
            + (m01 * (m10 * m15 - m14 * m11));
        W02 = (m13 * (m02 * m07 - m06 * m03))
            + (m01 * (m06 * m15 - m14 * m07))
            + (m05 * (m14 * m03 - m02 * m15));
        W03 = (m01 * (m06 * m11 - m10 * m07))
            + (m05 * (m10 * m03 - m02 * m11))
            + (m09 * (m02 * m07 - m06 * m03));
        W04 = (m06 * (m11 * m12 - m15 * m08))
            + (m10 * (m15 * m04 - m07 * m12))
            + (m14 * (m07 * m08 - m11 * m04));
        W05 = (m10 * (m15 * m00 - m03 * m12))
            + (m14 * (m03 * m08 - m11 * m00))
            + (m02 * (m11 * m12 - m15 * m08));
        W06 = (m14 * (m03 * m04 - m07 * m00))
            + (m02 * (m07 * m12 - m15 * m04))
            + (m06 * (m15 * m00 - m03 * m12));
        W07 = (m02 * (m07 * m08 - m11 * m04))
            + (m06 * (m11 * m00 - m03 * m08))
            + (m10 * (m03 * m04 - m07 * m00));
        W08 = (m07 * (m08 * m13 - m12 * m09))
            + (m11 * (m12 * m05 - m04 * m13))
            + (m15 * (m04 * m09 - m08 * m05));
        W09 = (m11 * (m12 * m01 - m00 * m13))
            + (m15 * (m00 * m09 - m08 * m01))
            + (m03 * (m08 * m13 - m12 * m09));
        W10 = (m15 * (m00 * m05 - m04 * m01))
            + (m03 * (m04 * m13 - m12 * m05))
            + (m07 * (m12 * m01 - m00 * m13));
        W11 = (m03 * (m04 * m09 - m08 * m05))
            + (m07 * (m08 * m01 - m00 * m09))
            + (m11 * (m00 * m05 - m04 * m01));
        W12 = (m04 * (m09 * m14 - m13 * m10))
            + (m08 * (m13 * m06 - m05 * m14))
            + (m12 * (m05 * m10 - m09 * m06));
        W13 = (m08 * (m13 * m02 - m01 * m14))
            + (m12 * (m01 * m10 - m09 * m02))
            + (m00 * (m09 * m14 - m13 * m10));
        W14 = (m12 * (m01 * m06 - m05 * m02))
            + (m00 * (m05 * m14 - m13 * m06))
            + (m04 * (m13 * m02 - m01 * m14));
        W15 = (m00 * (m05 * m10 - m09 * m06))
            + (m04 * (m09 * m02 - m01 * m10))
            + (m08 * (m01 * m06 - m05 * m02));

        det = W00 * m00 - W01 * m04 + W02 * m08 - W03 * m12;

        if ( det == 0.0f )
        {
            return;
        }
        invdet = 1.0f / det;

        m[ 0] =  W00 * invdet;
        m[ 1] = -W01 * invdet;
        m[ 2] =  W02 * invdet;
        m[ 3] = -W03 * invdet;
        m[ 4] = -W04 * invdet;
        m[ 5] =  W05 * invdet;
        m[ 6] = -W06 * invdet;
        m[ 7] =  W07 * invdet;
        m[ 8] =  W08 * invdet;
        m[ 9] = -W09 * invdet;
        m[10] =  W10 * invdet;
        m[11] = -W11 * invdet;
        m[12] = -W12 * invdet;
        m[13] =  W13 * invdet;
        m[14] = -W14 * invdet;
        m[15] =  W15 * invdet;
    }
}


float vector_normalize (float *lpV)
{
    float len2, length, invLength;

    len2 = lpV[0] * lpV[0] + lpV[1] * lpV[1] + lpV[2] * lpV[2];
    length = sqrtf(len2);
    if(length == 0.0f)
    {
        return 0.0f;
    }

    invLength = 1.0f / length;
    lpV[0] *= invLength;
    lpV[1] *= invLength;
    lpV[2] *= invLength;

    return length;
}


/* lpR = lpP * lpQ */
void quaternion_mult (float *lpR, float *lpP, float *lpQ)
{
    float pw, px, py, pz;
    float qw, qx, qy, qz;

    pw = lpP[0]; px = lpP[1]; py = lpP[2]; pz = lpP[3];
    qw = lpQ[0]; qx = lpQ[1]; qy = lpQ[2]; qz = lpQ[3];

    lpR[0] = pw * qw - px * qx - py * qy - pz * qz;
    lpR[1] = pw * qx + px * qw + py * qz - pz * qy;
    lpR[2] = pw * qy - px * qz + py * qw + pz * qx;
    lpR[3] = pw * qz + px * qy - py * qx + pz * qw;
}


void quaternion_to_matrix (float *lpM, float *lpQ)
{
    float qw, qx, qy, qz;
    float x2, y2, z2;
    float xy, yz, zx;
    float wx, wy, wz;

    qw = lpQ[0]; qx = lpQ[1]; qy = lpQ[2]; qz = lpQ[3];

    x2 = 2.0f * qx * qx;
    y2 = 2.0f * qy * qy;
    z2 = 2.0f * qz * qz;

    xy = 2.0f * qx * qy;
    yz = 2.0f * qy * qz;
    zx = 2.0f * qz * qx;

    wx = 2.0f * qw * qx;
    wy = 2.0f * qw * qy;
    wz = 2.0f * qw * qz;

    lpM[ 0] = 1.0f - y2 - z2;
    lpM[ 4] = xy - wz;
    lpM[ 8] = zx + wy;
    lpM[12] = 0.0f;

    lpM[ 1] = xy + wz;
    lpM[ 5] = 1.0f - z2 - x2;
    lpM[ 9] = yz - wx;
    lpM[13] = 0.0f;

    lpM[ 2] = zx - wy;
    lpM[ 6] = yz + wx;
    lpM[10] = 1.0f - x2 - y2;
    lpM[14] = 0.0f;

    lpM[ 3] = lpM[ 7] = lpM[11] = 0.0f;
    lpM[15] = 1.0f;
    }


void quaternion_rotate (float *lpQ, float rad, float ax, float ay, float az)
{
    float hrad = 0.5f * rad;
    float s = sinf(hrad);

    lpQ[0] = cosf(hrad);    /* w */
    lpQ[1] = s * ax;        /* x */
    lpQ[2] = s * ay;        /* y */
    lpQ[3] = s * az;        /* z */
}

void quaternion_identity (float *lpQ)
{
    lpQ[0] = 1.0f;          /* w */
    lpQ[1] = 0.0f;          /* x */
    lpQ[2] = 0.0f;          /* y */
    lpQ[3] = 0.0f;          /* z */
}

void quaternion_copy (float *lpTo, float *lpFrom)
{
    lpTo[0] = lpFrom[0];    /* w */
    lpTo[1] = lpFrom[1];    /* x */
    lpTo[2] = lpFrom[2];    /* y */
    lpTo[3] = lpFrom[3];    /* z */
}
