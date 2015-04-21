#include <stdlib.h>
#include <stdio.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include <math.h>
#include "cvc_complex.h"
#include "global.h"
#include "cvc_utils.h"
#include "cvc_geometry.h"

int *cvc_iup=NULL, *cvc_idn=NULL, *cvc_ipt=NULL , **cvc_ipt_=NULL, ***cvc_ipt__=NULL;

unsigned long int get_index(const int t, const int x, const int y, const int z)
{

  unsigned long int tt, xx, yy, zz, ix;

  tt = (t + T) % T;
  xx = (x + LX) % LX;
  yy = (y + LY) % LY;
  zz = (z + LZ) % LZ;
  ix = ((tt*LX+xx)*LY+yy)*LZ+zz;

#ifdef MPI
  if(t==T) {
    ix = VOLUME +              (xx*LY+yy)*LZ+zz;
  }
  if(t==-1) {
    ix = VOLUME +   LX*LY*LZ + (xx*LY+yy)*LZ+zz;
  }
#  if defined PARALLELTX || defined PARALLELTXY
  if(x==LX) {
    ix = VOLUME + 2*LX*LY*LZ +           (tt*LY+yy)*LZ+zz;
  }
  if(x==-1) {
    ix = VOLUME + 2*LX*LY*LZ + T*LY*LZ + (tt*LY+yy)*LZ+zz;
  }
#  endif
#  if defined PARALLELTXY
  if(y==LY) {
    ix = VOLUME + 2*(LX*LY*LZ + T*LY*LZ) + (tt*LX+xx)*LZ+zz;
  }
  if(y==-1) {
    ix = VOLUME + 2*(LX*LY*LZ + T*LY*LZ) + T*LX*LZ + (tt*LX+xx)*LZ+zz;
  }
#  endif

#  if defined PARALLELTX || defined PARALLELTXY

  /* x-t-edges */
  if(x==LX) {
    if(t==T) {
      ix = VOLUME + RAND           + yy*LZ+zz;
    }
    if(t==-1) {
      ix = VOLUME + RAND + 2*LY*LZ + yy*LZ+zz;
    }
  }
  if(x==-1) {
    if(t==T) {
      ix = VOLUME + RAND +   LY*LZ + yy*LZ+zz;
    }
    if(t==-1) {
      ix = VOLUME + RAND + 3*LY*LZ + yy*LZ+zz;
    }
  }
#  if defined PARALLELTXY
  /* y-t-edges */
  if(t==T) {
    if(y==LY) {
      ix = VOLUME + RAND + 4*LY*LZ           + xx*LZ+zz;
    }
    if(y==-1) {
      ix = VOLUME + RAND + 4*LY*LZ +   LX*LZ + xx*LZ+zz;
    }
  }
  if(t==-1) {
    if(y==LY) {
      ix = VOLUME + RAND + 4*LY*LZ + 2*LX*LZ + xx*LZ+zz;
    }
    if(y==-1) {
      ix = VOLUME + RAND + 4*LY*LZ + 3*LX*LZ + xx*LZ+zz;
    }
  }

  /* y-x-edges */
  if(x==LX) {
    if(y==LY) {
      ix = VOLUME + RAND + 4*(LY*LZ + LX*LZ)          + tt*LZ+zz;
    }
    if(y==-1) {
      ix = VOLUME + RAND + 4*(LY*LZ + LX*LZ) +   T*LZ + tt*LZ+zz;
    }
  }
  if(x==-1) {
    if(y==LY) {
      ix = VOLUME + RAND + 4*(LY*LZ + LX*LZ) + 2*T*LZ + tt*LZ+zz;
    }
    if(y==-1) {
      ix = VOLUME + RAND + 4*(LY*LZ + LX*LZ) + 3*T*LZ + tt*LZ+zz;
    }
  }
#  endif  /* of if defined PARALLELTXY */
#  endif  /* of if defined PARALLELTX || defined PARALLELTXY */
#endif
  return(ix);
}

void cvc_geometry() {

  int x0, x1, x2, x3;
  int y0, y1, y2, y3, ix;
  int isboundary;

#ifdef MPI
  int start_valuet = 1;

#  if defined PARALLELTX || defined PARALLELTXY
  int start_valuex = 1;
#  else
  int start_valuex = 0;
#  endif

#  if defined PARALLELTXY
  int start_valuey = 1;
#  else
  int start_valuey = 0;
#  endif

#else
  int start_valuet = 0;
  int start_valuex = 0;
  int start_valuey = 0;
#endif

  for(x0=-start_valuet; x0<T +start_valuet; x0++) {
  for(x1=-start_valuex; x1<LX+start_valuex; x1++) {
  for(x2=-start_valuey; x2<LY+start_valuey; x2++) {

  for(x3=0; x3<LZ; x3++) {

    isboundary = 0;
    if(x0==-1 || x0== T) isboundary++;
    if(x1==-1 || x1==LX) isboundary++;
    if(x2==-1 || x2==LY) isboundary++;

    y0=x0; y1=x1; y2=x2; y3=x3;
    if(x0==-1) y0=T +1;
    if(x1==-1) y1=LX+1;
    if(x2==-1) y2=LY+1;

    if(isboundary > 2) {
      g_ipt[y0][y1][y2][y3] = -1;
      continue;
    }

    ix = get_index(x0, x1, x2, x3);

    g_ipt[y0][y1][y2][y3] = ix;

    g_iup[ix][0] = get_index(x0+1, x1, x2, x3);
    g_iup[ix][1] = get_index(x0, x1+1, x2, x3);
    g_iup[ix][2] = get_index(x0, x1, x2+1, x3);
    g_iup[ix][3] = get_index(x0, x1, x2, x3+1);

    g_idn[ix][0] = get_index(x0-1, x1, x2, x3);
    g_idn[ix][1] = get_index(x0, x1-1, x2, x3);
    g_idn[ix][2] = get_index(x0, x1, x2-1, x3);
    g_idn[ix][3] = get_index(x0, x1, x2, x3-1);

  }
  }
  }
  }
/*
  if(g_cart_id==0) {

    for(x0=-start_valuet; x0< T+start_valuet; x0++) {
    for(x1=-start_valuex; x1<LX+start_valuex; x1++) {
    for(x2=-start_valuey; x2<LY+start_valuey; x2++) {
    for(x3=0; x3<LZ; x3++) {
      y0=x0; y1=x1; y2=x2; y3=x3;
      if(x0==-1) y0 = T+1;
      if(x1==-1) y1 =LX+1;
      if(x2==-1) y2 =LY+1;
      fprintf(stdout, "[%2d cvc_geometry] %3d%3d%3d%3d%6d\n", g_cart_id, x0, x1, x2, x3, g_ipt[y0][y1][y2][y3]);
    }}}}

  }
*/
/*
  if(g_cart_id==0) {
    for(x0=-start_valuet; x0<T+start_valuet; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      y0=x0; y1=x1; y2=x2; y3=x3;
      if(x0==-1) y0=T+1;
      ix = g_ipt[y0][y1][y2][y3];      
      fprintf(stdout, "%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d\n", x0, x1, x2, x3, 
        g_iup[ix][0], g_idn[ix][0], g_iup[ix][1], g_idn[ix][1],
	g_iup[ix][2], g_idn[ix][2], g_iup[ix][3], g_idn[ix][3]);
    }
    }
    }
    }
  }
*/
}

int init_geometry(void) {

  int ix = 0, V;
  int dx = 0, dy = 0;

  VOLUME         = T*LX*LY*LZ;
  VOLUMEPLUSRAND = VOLUME;
  RAND           = 0;
  EDGES          = 0;

#ifdef MPI
  RAND           += 2*LX*LY*LZ;
  VOLUMEPLUSRAND += 2*LX*LY*LZ;

#if defined PARALLELTX || defined PARALLELTXY
  RAND           += 2*T*LY*LZ;
  EDGES          +=             4*LY*LZ;
  VOLUMEPLUSRAND += 2*T*LY*LZ + 4*LY*LZ;
  dx = 2;
#endif

#if defined PARALLELTXY
  RAND           += 2*T*LX*LZ;
  EDGES          +=             4*LX*LZ + 4*T*LZ;
  VOLUMEPLUSRAND += 2*T*LX*LZ + 4*LX*LZ + 4*T*LZ;
  dy = 2;
#endif

#endif  /* of ifdef MPI */

  if(g_cart_id==0) fprintf(stdout, "# VOLUME = %d\n# RAND   = %d\n# EDGES  = %d\n# VOLUMEPLUSRAND = %d\n",
    VOLUME, RAND, EDGES, VOLUMEPLUSRAND);

  V = VOLUMEPLUSRAND;

  g_idn = (int**)calloc(V, sizeof(int*));
  if((void*)g_idn == NULL) return(1);

  cvc_idn = (int*)calloc(4*V, sizeof(int));
  if((void*)cvc_idn == NULL) return(2);

  g_iup = (int**)calloc(V, sizeof(int*));
  if((void*)g_iup==NULL) return(3);

  cvc_iup = (int*)calloc(4*V, sizeof(int));
  if((void*)cvc_iup==NULL) return(4);

  g_ipt = (int****)calloc(T+2, sizeof(int*));
  if((void*)g_ipt == NULL) return(5);

  cvc_ipt__ = (int***)calloc((T+2)*(LX+dx), sizeof(int*));
  if((void*)cvc_ipt__ == NULL) return(6);

  cvc_ipt_ =  (int**)calloc((T+2)*(LX+dx)*(LY+dy), sizeof(int*));
  if((void*)cvc_ipt_ == NULL) return(7);

  cvc_ipt =   (int*)calloc((T+2)*(LX+dx)*(LY+dy)*LZ, sizeof(int));
  if((void*)cvc_ipt == NULL) return(8);

 
  g_iup[0] = cvc_iup;
  g_idn[0] = cvc_idn;
  for(ix=1; ix<V; ix++) {
    g_iup[ix] = g_iup[ix-1] + 4;
    g_idn[ix] = g_idn[ix-1] + 4;
  }

  cvc_ipt_[0]  = cvc_ipt;
  cvc_ipt__[0] = cvc_ipt_;
  g_ipt[0] = cvc_ipt__;
  for(ix=1; ix<(T+2)*(LX+dx)*(LY+dy); ix++) cvc_ipt_[ix]  = cvc_ipt_[ix-1]  + LZ;

  for(ix=1; ix<(T+2)*(LX+dx);         ix++) cvc_ipt__[ix] = cvc_ipt__[ix-1] + (LY+dy);

  for(ix=1; ix<(T+2);                 ix++) g_ipt[ix] = g_ipt[ix-1] + (LX+dx);

  /* initialize the boundary condition */
  co_phase_up[0].re = cos(BCangle[0]*M_PI / (double)T_global);
  co_phase_up[0].im = sin(BCangle[0]*M_PI / (double)T_global);
  co_phase_up[1].re = cos(BCangle[1]*M_PI / (double)(LX*g_nproc_x));
  co_phase_up[1].im = sin(BCangle[1]*M_PI / (double)(LX*g_nproc_x));
  co_phase_up[2].re = cos(BCangle[2]*M_PI / (double)(LY*g_nproc_y));
  co_phase_up[2].im = sin(BCangle[2]*M_PI / (double)(LY*g_nproc_y));
  co_phase_up[3].re = cos(BCangle[3]*M_PI / (double)LZ);
  co_phase_up[3].im = sin(BCangle[3]*M_PI / (double)LZ);

  /* initialize the gamma matrices */
  init_gamma();

  return(0);
}

void free_geometry() {

  free(cvc_idn);
  free(cvc_iup);
  free(cvc_ipt);
  free(cvc_ipt_);
  free(cvc_ipt__);
  free(g_ipt);
  free(g_idn);
  free(g_iup);
}
