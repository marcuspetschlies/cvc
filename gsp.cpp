/***************************************************
 * cvc_utils.c                                     *
 ***************************************************/
 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif

#include "cvc_complex.h"
#include "global.h"
#include "ilinalg.h"
#include "cvc_geometry.h"
#include "read_input_parser.h"
#include "gsp.h"

namespace cvc {

/***********************************************
 * Np - number of momenta
 * Ng - number of gamma matrices
 * Nt - number of timeslices
 * Nv - number of eigenvectors
 ***********************************************/

int gsp_init (double ******gsp_out, int Np, int Ng, int Nt, int Nv) {

  double *****gsp;
  int i, k, l, c, x0;

  /***********************************************
   * allocate gsp space
   ***********************************************/
  gsp = (double*****)malloc(Np * sizeof(double****));
  if(gsp == NULL) {
    fprintf(stderr, "[gsp_init] Error from malloc\n");
    return(1);
  }

  gsp[0] = (double****)malloc(Np * Ng * sizeof(double***));
  if(gsp[0] == NULL) {
    fprintf(stderr, "[gsp_init] Error from malloc\n");
    return(2);
  }
  for(i=1; i<Np; i++) gsp[i] = gsp[i-1] + Ng;

  gsp[0][0] = (double***)malloc(Nt * Np * Ng * sizeof(double**));
  if(gsp[0][0] == NULL) {
    fprintf(stderr, "[gsp_init] Error from malloc\n");
    return(3);
  }

  c = 0;
  for(i=0; i<Np; i++) {
    for(k=0; k<Ng; k++) {
      if (c == 0) {
        c++;
        continue;
      }
      gsp[i][k] = gsp[0][0] + c * Nt;
      c++;
    }
  }

  gsp[0][0][0] = (double**)malloc(Nv * Nt * Np * Ng * sizeof(double*));
  if(gsp[0][0][0] == NULL) {
    fprintf(stderr, "[gsp_init] Error from malloc\n");
    return(4);
  }

  c = 0;
  for(i=0; i<Np; i++) {
    for(k=0; k<Ng; k++) {
      for(x0=0; x0<Nt; x0++) {
        if (c == 0) {
          c++;
          continue;
        }
        gsp[i][k][x0] = gsp[0][0][0] + c * Nv;
        c++;
      }
    }
  }

  gsp[0][0][0][0] = (double*)malloc(2 * Nv * Nv * Nt * Np * Ng * sizeof(double));
  if(gsp[0][0][0][0] == NULL) {
    fprintf(stderr, "[gsp_init] Error from malloc\n");
    return(5);
  }
  c = 0;
  for(i=0; i<Np; i++) {
    for(k=0; k<Ng; k++) {
      for(x0=0; x0<Nt; x0++) {
        for(l=0; l<Nv; l++) {
          if (c == 0) {
            c++;
            continue;
          }
          gsp[i][k][x0][l] = gsp[0][0][0][0] + c * 2*Nv;
          c++;
        }
      }
    }
  }
  *gsp_out = gsp;
  return(0);
} /* end of gsp_init */

/***********************************************
 * free gsp field
 ***********************************************/
int gsp_fini(double******gsp) {

  if( (*gsp) != NULL ) {
    if((*gsp)[0] != NULL ) {
      if((*gsp)[0][0] != NULL ) {
        if((*gsp)[0][0][0] != NULL ) {
          if((*gsp)[0][0][0][0] != NULL ) {
            free((*gsp)[0][0][0][0]);
          }
          free((*gsp)[0][0][0]);
        }
        free((*gsp)[0][0]);
      }
      free((*gsp)[0]);
    }
    free((*gsp));
  }

  return(0);
}  /* end of gsp_fini */

}  /* end of namespace cvc */

