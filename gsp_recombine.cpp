/***************************************************
 * gsp_recombine.cpp
 *
 * Di 8. Mai 14:00:46 CEST 2018
 *
 ***************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
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
#include "global.h"
#include "ilinalg.h"
#include "cvc_geometry.h"
#include "io_utils.h"
#include "read_input_parser.h"
#include "cvc_utils.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "scalar_products.h"
#include "iblas.h"
#include "table_init_z.h"
#include "rotations.h"
#include "gsp_recombine.h"


namespace cvc {


/***********************************************************************************************
 * input
 *   s    = T x numV x numV double complex
 *   N    = number of matrices
 *   numV = matrix size for each i = 0,...,N-1
 *   w    = spectral weight
 *
 * output
 *  r
 ***********************************************************************************************/
void gsp_tr_mat_weight (double _Complex * const r , double _Complex *** const s , double * const w, int const numV, int const N ) {

  for ( int i = 0; i < N; i++ ) {
    r[i] = rot_mat_trace_weight_re ( s[i], w, numV );
  }
  return;
}  // end of gsp_tr_mat_weight 


/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 * input
 *   s    = T x numV x numV double complex
 *   N    = number of matrices
 *   numV = matrix size for each i = 0,...,N-1
 *   w    = spectral weight
 *
 * output
 *  r = T double complex
 ***********************************************************************************************/
void gsp_tr_mat_weight_mat_weight ( double _Complex * const r , 
    double _Complex *** const s1 ,
    double * const w1, 
    double _Complex *** const s2 ,
    double * const w2, 
    int const numV, int const N 
  ) {

  memset ( r, 0, N*sizeof(double _Complex) );
  if ( ( w1 == NULL ) || ( w2 == NULL ) ) {
    fprintf ( stderr, "[gsp_tr_mat_weight_mat_weight] Error, a weight is NULL\n" );
    return;
  }

  for ( int i = 0; i < N; i++ ) {
  for ( int k = 0; k < N; k++ ) {
    int const ik = ( i - k + N ) % N;
    r[ik] += co_eq_trace_mat_ti_weight_ti_mat_ti_weight_re ( s1[i], w1, s2[k], w2, numV );
  }}
  return;
}  // end of gsp_tr_mat_weight_mat_weight

/***********************************************************************************************/
/***********************************************************************************************/

}  // end of namespace cvc
