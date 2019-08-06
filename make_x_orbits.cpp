#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "global.h"

namespace cvc {

/********************************************************************************
 * 4d x^2 orbits
 ********************************************************************************/
int init_x_orbits_4d(unsigned int **xid, unsigned int **xid_count, double **xid_val, unsigned int *xid_nc, unsigned int ***xid_member, int gcoords[4]) {

  unsigned int i;
  long int k;
  int x0, x1, x2, x3;
  int y0, y1, y2, y3;
  int z0, z1, z2, z3;
  unsigned int ix, Nclasses;
  unsigned int *radii = NULL;
  unsigned int Vhalf = VOLUME / 2;
  FILE *ofs=NULL;
  char filename[200];

  radii = (unsigned int*)malloc(Vhalf*sizeof(unsigned int));
  if(radii == NULL) {
    return(1);
  }
  
  for(x0=0; x0<T; x0++) {
    y0 = x0+g_proc_coords[0]*T  - gcoords[0];
    z0 = y0 > T_global/2 ? y0 -T_global : (y0 <= -T_global/2 ? y0+T_global : y0);
  for(x1=0; x1<LX; x1++) {
    y1 = x1+g_proc_coords[1]*LX - gcoords[1];
    z1 = y1 > LX_global/2 ? y1 -LX_global : (y1 <= -LX_global/2 ? y1+LX_global : y1);
  for(x2=0; x2<LY; x2++) {
    y2 = x2+g_proc_coords[2]*LY - gcoords[2];
    z2 = y2 > LY_global/2 ? y2 -LY_global : (y2 <= -LY_global/2 ? y2+LY_global : y2);
  for(x3=0; x3<LZ; x3++) {
    y3 = x3+g_proc_coords[3]*LZ - gcoords[3];
    z3 = y3 > LZ_global/2 ? y3 -LZ_global : (y3 <= -LZ_global/2 ? y3+LZ_global : y3);
    ix = g_ipt[x0][x1][x2][x3];
    if(g_iseven[ix]) continue;
    ix = g_lexic2eosub[ix];
    radii[ix] = z0*z0 + z1*z1 + z2*z2 + z3*z3;
    /* fprintf(stdout, "\t%6u (%3d, %3d, %3d, %3d) --->  (%3d, %3d, %3d, %3d) = %8u\n", ix, x0, x1, x2, x3, z0, z1, z2, z3, radii[ix]); */
  }}}}


  Nclasses = 1;
  for(i=1; i<Vhalf; i++) {
    int new_class = 1;
    for(k=i-1; k>=0; k--) {
      /* fprintf(stdout, "\t%6u %8u %6ld %8u\t %4u\n", i, radii[i], k, radii[k], Nclasses); */
      if(radii[i] == radii[k]) {
        new_class = 0;
        break;
      }
    }
    if(new_class) {
      Nclasses++;
      /* fprintf(stdout, "\t%6u %8u new class\n", i, radii[i]); */
    }
  }

  printf("# [make_x_orbits_4d] proc%.4d Nclasses = %u\n", g_cart_id, Nclasses);

  *xid       = (unsigned int*)malloc(Vhalf*sizeof(unsigned int));
  *xid_count = (unsigned int*)malloc(Nclasses*sizeof(unsigned int));
  *xid_val   = (double*)malloc(Nclasses * sizeof(double));
  if( *xid == NULL || *xid_count == NULL || *xid_val == NULL) {
    return(2);
  }

  *xid_nc = Nclasses;

  for(i=0; i<Nclasses; i++) (*xid_val)[i] = -1.;
  memset( *xid_count, 0, Nclasses*sizeof(unsigned int));

  for(x0=0; x0<T; x0++) {
    y0 = x0+g_proc_coords[0]*T  - gcoords[0];
    z0 = y0 > T_global/2 ? y0 -T_global : (y0 <= -T_global/2 ? y0+T_global : y0);
  for(x1=0; x1<LX; x1++) {
    y1 = x1+g_proc_coords[1]*LX - gcoords[1];
    z1 = y1 > LX_global/2 ? y1 -LX_global : (y1 <= -LX_global/2 ? y1+LX_global : y1);
  for(x2=0; x2<LY; x2++) {
    y2 = x2+g_proc_coords[2]*LY - gcoords[2];
    z2 = y2 > LY_global/2 ? y2 -LY_global : (y2 <= -LY_global/2 ? y2+LY_global : y2);
  for(x3=0; x3<LZ; x3++) {
    y3 = x3+g_proc_coords[3]*LZ - gcoords[3];
    z3 = y3 > LZ_global/2 ? y3 -LZ_global : (y3 <= -LZ_global/2 ? y3+LZ_global : y3);
    ix = g_ipt[x0][x1][x2][x3];
    if(g_iseven[ix]) continue;
    ix = g_lexic2eosub[ix];

    /* fprintf(stdout, "# [] processing %6u = (%3d, %3d, %3d, %3d) ---> (%3d, %3d, %3d, %3d)\n", ix, x0, x1, x2, x3, z0, z1, z2, z3); */

    for(i=0; i<Nclasses; i++) {
      /* fprintf(stdout, "\t%4u %6u %6u %16.7e\n", i,ix, radii[ix], (*xid_val)[i]); */

      if( (double)(radii[ix]) == (*xid_val)[i]) {
        /* add to this classe */
        (*xid_count)[i]++;
        (*xid)[ix] = i;
        /* fprintf(stdout, " add new member %6u %6u %16.7e\n", i, ix, (*xid_val)[i]); */
        break;

      } else if ((*xid_val)[i] == -1. ) {
        /* add new class */
        (*xid_val)[i] = (double)(radii[ix]);
        (*xid_count)[i]++;
        (*xid)[ix] = i;
        /* fprintf(stdout, " add new class %6u %6u %16.7e\n", i, ix, (*xid_val)[i]); */
        break;

      }
    }
    if(i == Nclasses) {
      fprintf(stderr, "[make_x_orbits_4d] Error, could not find matching radius for %u within %u classes\n", radii[ix], Nclasses);
      return(4);
    }
  }}}}



  free(radii);

  *xid_member = (unsigned int**)malloc(Nclasses * sizeof(unsigned int*));
  if( *xid_member == NULL) {
    return(3);
  }
  (*xid_member)[0] = (unsigned int*)malloc(Vhalf * sizeof(unsigned int));
  if( (*xid_member)[0] == NULL) {
    return(4);
  }
  for(i=1; i<Nclasses; i++) (*xid_member)[i] = (*xid_member)[i-1] + (*xid_count)[i-1];

  radii = (unsigned int*)malloc(Nclasses * sizeof(unsigned int));
  if( radii == NULL ) {
    return(5);
  }
  memset(radii, 0, Nclasses * sizeof(unsigned int));
  for(ix=0; ix<Vhalf; ix++) {
    i = (*xid)[ix];  /* class id for ix */
    (*xid_member)[i][radii[i]] = ix;
    radii[i]++;
  }
  free(radii);

#if 0
  for(ix=0; ix<Vhalf; ix++) {
    fprintf(stdout, "xid \t%8u %8u\n", ix, (*xid)[ix]);
  }


  ix = 0;
  for(i=0; i<Nclasses; i++) ix += (*xid_count)[i];
  fprintf(stdout, "# [make_x_orbits_4d] total number of members in %u classes  %u\n", Nclasses, ix);


  sprintf(filename, "init_x_orbits_4d.%.4d", g_cart_id);
  if( ( ofs = fopen(filename, "w") ) == NULL ) { return(1); }
  /* ofs = stdout; */
  for(i = 0; i < Nclasses; i++) {
    fprintf(ofs, "# class %6u val %25.16e members %6u\n", i, (*xid_val)[i], (*xid_count)[i]);

    for(ix=0; ix < (*xid_count)[i]; ix++) {
      fprintf(ofs, "\t %6d %6d %6u\n", i, ix, (*xid_member)[i][ix]);
    }

  }
  fclose(ofs);
#endif
  return(0);
}  /* init_x_orbits_4d */

int fini_x_orbits_4d(unsigned int **xid, unsigned int **xid_count, double **xid_val, unsigned int ***xid_member) {
  if(*xid != NULL) free(*xid);
  if(*xid_count != NULL) free(*xid_count);
  if(*xid_val != NULL) free(*xid_val);
  if(*xid_member != NULL) {
  if( (*xid_member)[0] != NULL) free( (*xid_member)[0]);
    free(*xid_member);
  }
  return(0);
}  /* end of fini_x_orbits_4d */

}  /* end of namespace cvc */
