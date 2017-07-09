/****************************************************
 * contract_diagrams.c
 * 
 * Mon Jun  5 16:00:53 CDT 2017
 *
 * PURPOSE:
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "matrix_init.h"
#include "gamma.h"
#include "contract_diagrams.h"

namespace cvc {

#if 0
/****************************************************
 * we always sum in the following way
 * v2[alpha_p[0], alpha_p[1], alpha_p[2], m] g[alpha_2, beta]  v3[beta,m]
 ****************************************************/
int contract_diagram_v2_gamma_v3 ( double _Complex **vdiag, double _Complex **v2, double _Complex **v3, gamma_matrix_type g, int perm[3], unsigned int N, int init ) {

  if ( init ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [] initializing output field to zero\n");
    memset( vdiag[0], 0, 16*T_global*sizeof(double _Complex ) );
  }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int it = 0; it < N; it++ ) {

    for ( int alpha = 0; alpha < 4; alpha++ ) {
      for ( int beta = 0; beta < 4; beta++ ) {

        int vdiag_index = 4 * alpha + beta;
        /* vdiag[it][vdiag_index] = 0.; */

        /****************************************************/
        /****************************************************/

        for ( int gamma = 0; gamma < 4; gamma++ ) {

          int idx[3] = { alpha, beta, gamma };

          int pidx[3] = { idx[perm[0]], idx[perm[1]], idx[perm[2]] };

          for ( int delta = 0; delta < 4; delta++ ) {
            for ( int m = 0; m < 3; m++ ) {

              /* use the permutation */
              int v2_pindex = 3 * ( 4 * ( 4 * pidx[0] + pidx[1] ) + pidx[2] ) + m;
 
              int v3_index  = 3 * delta + m;

              vdiag[it][vdiag_index] -=  v2[it][v2_pindex] * v3[it][v3_index] * g.m[gamma][delta];
            }  /* end of loop on color index m */
          }  /* end of loop on spin index delta */
        }  /* end of loop on spin index gamma */

        /****************************************************/
        /****************************************************/

      }  /* end of loop on spin index beta */
    }  /* end of loop on spin index alpha */

  }  /* end of loop on N */

}  /* end of function contract_diagram_v2_gamma_v3 */
#endif  /* of if 0*/

/****************************************************
 * we always sum in the following way
 * v2[alpha_p[0], alpha_p[1], alpha_p[2], m] g[alpha_2, alpha_3]  v3[ alpha_p[3], m]
 ****************************************************/
int contract_diagram_v2_gamma_v3 ( double _Complex **vdiag, double _Complex **v2, double _Complex **v3, gamma_matrix_type g, int perm[4], unsigned int N, int init ) {

  if ( init ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [contract_diagram_v2_gamma_v3] initializing output field to zero\n");
    memset( vdiag[0], 0, 16*T_global*sizeof(double _Complex ) );
  }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int it = 0; it < N; it++ ) {

    for ( int alpha = 0; alpha < 4; alpha++ ) {
      for ( int beta = 0; beta < 4; beta++ ) {

        int vdiag_index = 4 * alpha + beta;
        /* vdiag[it][vdiag_index] = 0.; */

        /****************************************************/
        /****************************************************/

        for ( int gamma = 0; gamma < 4; gamma++ ) {
          for ( int delta = 0; delta < 4; delta++ ) {

            int idx[4]  = { alpha, beta, gamma, delta };

            int pidx[4] = { idx[perm[0]], idx[perm[1]], idx[perm[2]], idx[perm[3]] };

            for ( int m = 0; m < 3; m++ ) {

              /* use the permutation */
              int v2_pindex = 3 * ( 4 * ( 4 * pidx[0] + pidx[1] ) + pidx[2] ) + m;
 
              int v3_index  = 3 * pidx[3] + m;

              vdiag[it][vdiag_index] -=  v2[it][v2_pindex] * v3[it][v3_index] * g.m[gamma][delta];
            }  /* end of loop on color index m */
          }  /* end of loop on spin index delta */
        }  /* end of loop on spin index gamma */

        /****************************************************/
        /****************************************************/

      }  /* end of loop on spin index beta */
    }  /* end of loop on spin index alpha */

  }  /* end of loop on N */

  return(0);
}  /* end of function contract_diagram_v2_gamma_v3 */

/****************************************************
 * we always sum in the following way
 * goet[b_oet][a_oet]  v2[a_oet][alpha_p[0], alpha_p[1], alpha_p[2], m] g[alpha_2, alpha_3]  v3[b_oet][ alpha_p[3], m]
 ****************************************************/
int contract_diagram_oet_v2_gamma_v3 ( double _Complex **vdiag, double _Complex ***v2, double _Complex ***v3, gamma_matrix_type goet, gamma_matrix_type g, int perm[4], unsigned int N, int init ) {

  if ( init ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [contract_diagram_oet_v2_amma_v3] initializing output field to zero\n");
    memset( vdiag[0], 0, 16*T_global*sizeof(double _Complex ) );
  }

  for ( int sigma_oet = 0; sigma_oet < 4; sigma_oet++ ) {
  for ( int tau_oet   = 0; tau_oet   < 4; tau_oet++ ) {

    double _Complex c = goet.m[tau_oet][sigma_oet];
    if ( c == 0 ) continue;

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int it = 0; it < N; it++ ) {

      for ( int alpha = 0; alpha < 4; alpha++ ) {
        for ( int beta = 0; beta < 4; beta++ ) {

          int vdiag_index = 4 * alpha + beta;
          /* vdiag[it][vdiag_index] = 0.; */

          /****************************************************/
          /****************************************************/

          for ( int gamma = 0; gamma < 4; gamma++ ) {
            for ( int delta = 0; delta < 4; delta++ ) {

              int idx[4]  = { alpha, beta, gamma, delta };

              int pidx[4] = { idx[perm[0]], idx[perm[1]], idx[perm[2]], idx[perm[3]] };

              for ( int m = 0; m < 3; m++ ) {

                /* use the permutation */
                int v2_pindex = 3 * ( 4 * ( 4 * pidx[0] + pidx[1] ) + pidx[2] ) + m;
 
                int v3_index  = 3 * pidx[3] + m;

                vdiag[it][vdiag_index] -=  c * v2[sigma_oet][it][v2_pindex] * v3[tau_oet][it][v3_index] * g.m[gamma][delta];
              }  /* end of loop on color index m */
            }  /* end of loop on spin index delta */
          }  /* end of loop on spin index gamma */

          /****************************************************/
          /****************************************************/

        }  /* end of loop on spin index beta */
      }  /* end of loop on spin index alpha */

    }  /* end of loop on N */

  }  /* end of loop on tau   oet */
  }  /* end of loop on sigma oet */
  return(0);
}  /* end of function contract_diagram_oet_v2_gamma_v3 */

#if 0
/****************************************************
 *
 ****************************************************/
void contract_b1 (double _Complex ***b1, double _Complex **v3, **double v2, gamma_matrix_type g) {

  for( int it = 0; it < T; it++ ) {
    for(int alpha = 0; alpha < 4; alpha++) {
    for(int beta = 0; beta < 4; beta++) {
      double _Complex z;
      for(int m = 0; m < 3; m++) {
        for(int gamma = 0; gamma < 4; gamma++) {
        for(int delta = 0; delta < 4; delta++) {
          int i3 = 3*gamma + m;
          int i2 = 4 * ( 4 * ( 4*m + beta ) + alpha ) + delta;
          z += -v3[it][i3] * v2[it][i2] * g.m[gamma][delta];
        }}
      }
      b1[it][alpha][beta] = z;
    }}
  }  /* loop on timeslices */
}  /* end of contract_b1 */

void contract_b2 (double _Complex ***b2, double _Complex **v3, **double v2, gamma_matrix_type g) {

  for( int it = 0; it < T; it++ ) {
    for(int alpha = 0; alpha < 4; alpha++) {
    for(int beta = 0; beta < 4; beta++) {
      double _Complex z;
      for(int m = 0; m < 3; m++) {
        for(int gamma = 0; gamma < 4; gamma++) {
        for(int delta = 0; delta < 4; delta++) {
          int i3 = 3*gamma + m;
          int i2 = 4 * ( 4 * ( 4*m + delta ) + alpha ) + beta;
          z += -v3[it][i3] * v2[it][i2] * g.m[gamma][delta];
        }}
      }
      b2[it][alpha][beta] = z;
    }}
  }  /* loop on timeslices */
}  /* end of contract_b2 */
#endif  /* end of if 0 */

/****************************************************
 * search for m1 in m2
 ****************************************************/
int match_momentum_id ( int **pid, int **m1, int **m2, int N1, int N2 ) {
#if 0 
  fprintf(stdout, "# [match_momentum_id] N1 = %d N2 = %d m2 == NULL ? %d\n", N1, N2 , m2 == NULL);
  for ( int i = 0; i < N1; i++ ) {
    fprintf(stdout, "# [match_momentum_id] m1 %d  %3d %3d %3d\n", i, m1[i][0], m1[i][1], m1[i][2]);
  }

  for ( int i = 0; i < N2; i++ ) {
    fprintf(stdout, "# [match_momentum_id] m2 %d  %3d %3d %3d\n", i, m2[i][0], m2[i][1], m2[i][2]);
  }
  return(1);
#endif

  if ( N1 > N2 ) {
    fprintf(stderr, "[match_momentum_id] Error, N1 > N2\n");
    return(1);
  }

  if ( *pid == NULL ) {
    *pid = (int*)malloc (N1 * sizeof(int) );
  }

  for ( int i = 0; i < N1; i++ ) {
    int found = 0;
    int p[3] = { m1[i][0], m1[i][1], m1[i][2] };

    for ( int k = 0; k < N2; k++ ) {
      if ( p[0] == m2[k][0] && p[1] == m2[k][1] && p[2] == m2[k][2] ) {
        (*pid)[i] = k;
        found = 1;
        break;
      }
    }
    if ( found == 0 ) {
      fprintf(stderr, "[match_momentum_id] Warning, could not find momentum no %d = %3d %3d %3d\n",
          i, p[0], p[1], p[2]);
      (*pid)[i] = -1;
      /* return(2); */
    }
  }

  /* TEST */
  if ( g_verbose > 2 ) {
    for ( int i = 0; i < N1; i++ ) {
      fprintf(stdout, "# [match_momentum_id] m1[%2d] = %3d %3d %3d matches m2[%2d] = %3d %3d %3d\n",
          i, m1[i][0], m1[i][1], m1[i][2],
          (*pid)[i], m2[(*pid)[i]][0], m2[(*pid)[i]][1], m2[(*pid)[i]][2]);
    }
  }

  return(0);
}  /* end of match_momentum_id */

}  /* end of namespace cvc */
