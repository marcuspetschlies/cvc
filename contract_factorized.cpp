/****************************************************
 * contract_factorized.c
 * 
 * Mon May 22 17:12:00 CEST 2017
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
#include "project.h"

namespace cvc {

/******************************************************
 *
 ******************************************************/
int contract_v3  (double **v3, double*phi, fermion_propagator_type*prop, unsigned int N ) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for(unsigned int ix=0; ix < N; ix++) {
    _v3_eq_fv_dot_fp( v3[ix], phi+_GSI(ix), prop[ix]);
  }
}  /* contract_v3 */

/******************************************************
 *
 ******************************************************/
int contract_v2 (double **v2, double *phi, fermion_propagator_type *prop1, fermion_propagator_type *prop2, unsigned int N ) {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  double v1[72];
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(unsigned int ix=0; ix < N; ix++) {
    _v1_eq_fv_eps_fp( v1, phi+_GSI(ix), prop1[ix] );

    _v2_eq_v1_eps_fp( v2[ix], v1, prop2[ix] );
  }

#ifdef HAVE_OPENMP
}
#endif
}  /* end of contract_v2 */

/******************************************************
 *
 ******************************************************/
int contract_v1 (double **v1, double *phi, fermion_propagator_type *prop1, unsigned int N ) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for(unsigned int ix=0; ix < N; ix++) {
    _v1_eq_fv_eps_fp( v1[ix], phi+_GSI(ix), prop1[ix] );
  }
}  /* end of contract_v1 */

}  /* end of namespace cvc */
