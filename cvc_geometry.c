#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#include <math.h>
#include "cvc_complex.h"
#include "global.h"
#include "cvc_utils.h"
#include "cvc_geometry.h"

namespace cvc {

int *iup=NULL, *idn=NULL, *ipt=NULL , **ipt_=NULL, ***ipt__=NULL;

unsigned long int get_index(const int t, const int x, const int y, const int z)
{

  unsigned long int tt, xx, yy, zz, ix;

  tt = (t + T) % T;
  xx = (x + LX) % LX;
  yy = (y + LY) % LY;
  zz = (z + LZ) % LZ;
  ix = ((tt*LX+xx)*LY+yy)*LZ+zz;

#ifdef HAVE_MPI
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

void geometry() {

  int x0, x1, x2, x3;
  int y0, y1, y2, y3, ix;
  int isboundary;
  int i_even, i_odd;
  int itzyx;

#ifdef HAVE_MPI
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

    // is even / odd
    if(isboundary == 0) {
      g_iseven[ix] = ( x0 + T *g_proc_coords[0] + x1 + LX*g_proc_coords[1] \
                     + x2 + LY*g_proc_coords[2] + x3 + LZ*g_proc_coords[3] ) % 2 == 0;

      // replace this by indext function
      itzyx = ( ( x0*LZ + x3 ) * LY + x2 ) * LX + x1;
      g_isevent[itzyx] = g_iseven[ix];
    }

  }}}} // of x3, x2, x1, x0


  i_even = 0; i_odd = 0;
  for(ix=0; ix<VOLUME;ix++) {
    // this will have to be changed if to be used with MPI
    if(g_iseven[ix]) {
      g_lexic2eo[ix] = i_even;
      g_eo2lexic[i_even] = ix;
      i_even++;
    } else {
      g_lexic2eo[ix] = i_odd + VOLUME/2;
      g_eo2lexic[i_odd+VOLUME/2] = ix;
      i_odd++;
    }
  }

  // this will have to be changed if to be used with MPI
  itzyx = 0;
  i_even = 0; i_odd = 0;
  for(x0=0;x0<T; x0++) {
  for(x3=0;x3<LZ;x3++) {
  for(x2=0;x2<LY;x2++) {
  for(x1=0;x1<LX;x1++) {
    ix = g_ipt[x0][x1][x2][x3];
    if(g_isevent[itzyx]) {
      g_lexic2eot[ix] = i_even;
      g_eot2lexic[i_even] = ix;
      i_even++;
    } else {
      g_lexic2eot[ix] = i_odd + VOLUME/2;
      g_eot2lexic[i_odd+VOLUME/2] = ix;
      i_odd++;
    }
    itzyx++;
  }}}}



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
      fprintf(stdout, "[%2d geometry] %3d%3d%3d%3d%6d\n", g_cart_id, x0, x1, x2, x3, g_ipt[y0][y1][y2][y3]);
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
/*
  if(g_cart_id==0) {
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];      
      fprintf(stdout, "%6d%3d%3d%3d%3d%6d%6d\n", ix, x0, x1, x2, x3, 
        g_lexic2eo[ix], g_lexic2eot[ix] );      
    }}}}
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

#ifdef HAVE_MPI
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

#endif  /* of ifdef HAVE_MPI */

  if(g_cart_id==0) {
    fprintf(stdout, "# [init_geometry] (T, LX, LY, LZ) = (%d, %d, %d, %d)\n", T, LX, LY, LZ);
    fprintf(stdout, "# [init_geometry] VOLUME = %d\n", VOLUME);
    fprintf(stdout, "# [init_geometry] RAND   = %d\n", RAND);
    fprintf(stdout, "# [init_geometry] EDGES  = %d\n", EDGES);
    fprintf(stdout, "# [init_geometry] VOLUMEPLUSRAND = %d\n", VOLUMEPLUSRAND);
  }

  V = VOLUMEPLUSRAND;

  g_idn = (int**)calloc(V, sizeof(int*));
  if((void*)g_idn == NULL) return(1);

  idn = (int*)calloc(4*V, sizeof(int));
  if((void*)idn == NULL) return(2);

  g_iup = (int**)calloc(V, sizeof(int*));
  if((void*)g_iup==NULL) return(3);

  iup = (int*)calloc(4*V, sizeof(int));
  if((void*)iup==NULL) return(4);

  g_ipt = (int****)calloc(T+2, sizeof(int*));
  if((void*)g_ipt == NULL) return(5);

  ipt__ = (int***)calloc((T+2)*(LX+dx), sizeof(int*));
  if((void*)ipt__ == NULL) return(6);

  ipt_ =  (int**)calloc((T+2)*(LX+dx)*(LY+dy), sizeof(int*));
  if((void*)ipt_ == NULL) return(7);

  ipt =   (int*)calloc((T+2)*(LX+dx)*(LY+dy)*LZ, sizeof(int));
  if((void*)ipt == NULL) return(8);

 
  g_iup[0] = iup;
  g_idn[0] = idn;
  for(ix=1; ix<V; ix++) {
    g_iup[ix] = g_iup[ix-1] + 4;
    g_idn[ix] = g_idn[ix-1] + 4;
  }

  ipt_[0]  = ipt;
  ipt__[0] = ipt_;
  g_ipt[0] = ipt__;
  for(ix=1; ix<(T+2)*(LX+dx)*(LY+dy); ix++) ipt_[ix]  = ipt_[ix-1]  + LZ;

  for(ix=1; ix<(T+2)*(LX+dx);         ix++) ipt__[ix] = ipt__[ix-1] + (LY+dy);

  for(ix=1; ix<(T+2);                 ix++) g_ipt[ix] = g_ipt[ix-1] + (LX+dx);


  g_lexic2eo = (int*)calloc(V, sizeof(int));
  if(g_lexic2eo == NULL) return(9);

  g_lexic2eot = (int*)calloc(V, sizeof(int));
  if(g_lexic2eot == NULL) return(10);

  g_eo2lexic = (int*)calloc(V, sizeof(int));
  if(g_eo2lexic == NULL) return(11);

  g_eot2lexic = (int*)calloc(V, sizeof(int));
  if(g_eot2lexic == NULL) return(11);

  g_iseven = (int*)calloc(V, sizeof(int));
  if(g_iseven == NULL) return(12);

  g_isevent = (int*)calloc(V, sizeof(int));
  if(g_isevent == NULL) return(12);


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

  free(idn);
  free(iup);
  free(ipt);
  free(ipt_);
  free(ipt__);
  free(g_ipt);
  free(g_idn);
  free(g_iup);
  free(g_lexic2eo);
  free(g_lexic2eot);
  free(g_eo2lexic);
  free(g_eot2lexic);
  free(g_iseven);
  free(g_isevent);
}


int init_multigrid_decompositon(int degree, int**lexic2sub, int***sub2lexic, int**insub) {

  unsigned int nsublat = 0, lvol = 0;
  unsigned int length=0;
  unsigned int ix, isub, isub2, i, latnum;
  unsigned int *eocounter[2] = {NULL, NULL};
  int gx[4], rx[4], lx[4];
  int x0, x1, x2, x3, lsize[4];
  int isodd = degree % 2;

  // TEST
  FILE *ofs=NULL;
  char filename[200];

  if(*lexic2sub != NULL) {
    free(*lexic2sub);
    *lexic2sub = NULL;
  }

  if(*sub2lexic != NULL) {
    free(*sub2lexic);
    *sub2lexic = NULL;
  }

  if(*insub != NULL) {
    free(*insub);
    *insub = NULL;
  }
  
  nsublat = 1<<((degree/2)*4 + isodd);
  if(VOLUME % nsublat != 0) {
    if(g_cart_id == 0) fprintf(stderr, "[] Error, nsublat !| VOLUME\n");
    return(4);
  }
  lvol = VOLUME / nsublat;
  length = 1<<(degree/2);

  lsize[0] = T / length;
  lsize[1] = LX / length;
  lsize[2] = LY / length;
  lsize[3] = LZ / length;

  if(g_cart_id == 0) fprintf(stdout, "# number of sublattices = %u; sites per sublattice = %u, length = %u\n",
      nsublat, lvol, length);

  if( (*insub = (int*)malloc(VOLUME * sizeof(int))) == NULL ) {
    return(1);
  }

  if( (*lexic2sub = (int*)malloc(VOLUME * sizeof(int))) == NULL ) {
    return(2);
  }

  if( (*sub2lexic = (int**)malloc(nsublat * sizeof(int*))) == NULL ) return(3);
  if( ( (*sub2lexic)[0] = (int*)malloc(nsublat*lvol * sizeof(int))) == NULL ) return(4);
  for(i=1;i<nsublat;i++) (*sub2lexic)[i]  = (*sub2lexic)[i-1] + lvol;
  
  if(isodd) {
    eocounter[0] = (unsigned int*)malloc(nsublat/2*sizeof(int));
    eocounter[1] = (unsigned int*)malloc(nsublat/2*sizeof(int));
    memset(eocounter[0],0,nsublat/2*sizeof(int));
    memset(eocounter[1],0,nsublat/2*sizeof(int));
  } else {
    eocounter[0] = (unsigned int*)malloc(nsublat*sizeof(int));
    memset(eocounter[0],0,nsublat*sizeof(int));
  }

  for(x0=0; x0<T;  x0++) {
    gx[0] = x0 + g_proc_coords[0] * T;
    lx[0] = gx[0] / length;
    rx[0] = gx[0] % length;
  for(x1=0; x1<LX; x1++) {
    gx[1] = x1 + g_proc_coords[1] * LX;
    lx[1] = gx[1] / length;
    rx[1] = gx[1] % length;
  for(x2=0; x2<LY; x2++) {
    gx[2] = x2 + g_proc_coords[2] * LY;
    lx[2] = gx[2] / length;
    rx[2] = gx[2] % length;
  for(x3=0; x3<LZ; x3++) {
    gx[3] = x3 + g_proc_coords[3] * LZ;
    lx[3] = gx[3] / length;
    rx[3] = gx[3] % length;

    ix = g_ipt[x0][x1][x2][x3];

    latnum = ( ( rx[0] * length + rx[1] ) * length + rx[2] ) * length + rx[3];
    //fprintf(stdout, "# inital latnum = %d\n", latnum);

    // check for odd degree
    if(isodd) {
      i = ( lx[0] + lx[1] + lx[2] + lx[3] ) % 2;
      isub = eocounter[i][latnum];
      eocounter[i][latnum]++;
      latnum = 2 * latnum + i;
    } else {
      // isub = ( ( lx[0] * lsize[1] + lx[1] ) * lsize[2] + lx[2] ) * lsize[3] + lx[3];
      isub = eocounter[0][latnum];
      eocounter[0][latnum]++;
    }
    //fprintf(stdout, "# final isub = %u\n", isub);
    //fprintf(stdout, "# final latnum = %d\n", latnum);

    (*lexic2sub)[ix]           = isub;
    (*insub)[ix]               = latnum;
    (*sub2lexic)[latnum][isub] = ix;

  }}}}

  // TEST

  sprintf(filename, "geom.%.2d.%.2d", g_nproc, g_cart_id);
  ofs = fopen(filename, "w");

  fprintf(ofs, "# lexic2sub:\n");
  for(x0=0; x0<T;  x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[x0][x1][x2][x3];
    fprintf(ofs, "\t%6d |%3d%3d%3d%3d | %4d%6d\n", ix, x0, x1, x2, x3, (*insub)[ix], (*lexic2sub)[ix]);
  }}}}

  fprintf(ofs, "\n# sub2lexic:\n");
  for(i=0;i<nsublat;i++) {
      fprintf(ofs, "# sub lattice no. %d\n", i);
    for(ix=0;ix<lvol;ix++) {
      fprintf(ofs, "\t%3d%6d%6d\n", i, ix, (*sub2lexic)[i][ix]);
    }
  }
  fclose(ofs);

  if(eocounter[0] != NULL) free(eocounter[0]);
  if(eocounter[1] != NULL) free(eocounter[1]);
  return(0);
}


void fini_multigrid_decompositon(int**lexic2sub, int***sub2lexic, int**insub) {
  if(*lexic2sub != NULL) {
    free(*lexic2sub);
    *lexic2sub = NULL;
  }
  if(*sub2lexic != NULL) {
    if((*sub2lexic)[0] != NULL) {
      free((*sub2lexic)[0]);
    }
    free(*sub2lexic);
    *sub2lexic = NULL;
  }
  if(*insub != NULL) {
    free(*insub);
    *insub = NULL;
  }
  return;
}

}
