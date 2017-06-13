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

namespace cvc {

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

}  /* end of namespace cvc */
