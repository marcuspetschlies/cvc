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
#ifdef HAVE_OPENMP
#include <omp.h>
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

void gsp_make_eo_phase_field (double*phase_e, double*phase_o, int *momentum) {

  const int nthreads = g_num_threads;

  int ix, iix;
  int x0, x1, x2, x3;
  int threadid = 0;
  double ratime, retime;
  double dtmp;


  if(g_cart_id == 0) {
    fprintf(stdout, "# [gsp_make_eo_phase_field] using phase momentum = (%d, %d, %d)\n", momentum[0], momentum[1], momentum[2]);
  }

  ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) private(ix,iix,x0,x1,x2,x3,dtmp,threadid) firstprivate(nthreads,T,LX,LY,LZ) shared(phase_e, phase_o, momentum)
{
  threadid = omp_get_thread_num();
#endif
  /* make phase field in eo ordering */
  for(x0 = threadid; x0<T; x0 += nthreads) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix  = g_ipt[x0][x1][x2][x3];
      iix = g_lexic2eosub[ix];
      dtmp = 2. * M_PI * (
          (x1 + g_proc_coords[1]*LX) * momentum[0] / (double)LX_global +
          (x2 + g_proc_coords[2]*LY) * momentum[1] / (double)LY_global +
          (x3 + g_proc_coords[3]*LZ) * momentum[2] / (double)LZ_global );
      if(g_iseven[ix]) {
        phase_e[2*iix  ] = cos(dtmp);
        phase_e[2*iix+1] = sin(dtmp);
      } else {
        phase_o[2*iix  ] = cos(dtmp);
        phase_o[2*iix+1] = sin(dtmp);
      }
    }}}
  }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif


  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [] time for making eo phase field = %e seconds\n", retime-ratime);
}  /* end of gsp_make_eo_phase_field */

}  /* end of namespace cvc */

