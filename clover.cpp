/****************************************************
 * clover.cpp
 *
 * Fri Apr 21 13:44:01 CEST 2017
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

#ifdef __cplusplus
}
#endif

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "Q_clover_phi.h"
#include "cvc_timer.h"

namespace cvc {

int init_clover ( double *** clover_term, double **(*mzz)[2], double **(*mzzinv)[2], double*gauge_field, double const mass, double const csw ) {
  
  struct timeval ta, tb;


  /***********************************************
   * check, that mzz, mzzinv and clover_term
   * are not yet initialized
   ***********************************************/
  if ( (*mzz)[0] != NULL || (*mzz)[1] != NULL || (*mzzinv)[0] != NULL || (*mzzinv)[1] != NULL  || *clover_term != NULL ) {
    fprintf ( stderr, "[init_clover] Error input fields not NULL %s %d\n", __FILE__, __LINE__ );
    return ( 1 );
  }


  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  clover_term_init ( clover_term, 6);
  clover_term_init ( &((*mzz)[0]), 6);
  clover_term_init ( &((*mzz)[1]), 6);

  gettimeofday ( &ta, (struct timezone *)NULL );

  clover_term_eo  ( *clover_term, gauge_field );

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "init_clover", "clover_term_eo", g_cart_id == 0 );

  gettimeofday ( &ta, (struct timezone *)NULL );

  clover_mzz_matrix ( (*mzz)[0], *clover_term,  mass, csw);

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "init_clover", "clover_mzz_matrix", g_cart_id == 0 );

  gettimeofday ( &ta, (struct timezone *)NULL );

  clover_mzz_matrix ( (*mzz)[1], *clover_term, -mass, csw);

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "init_clover", "clover_mzz_matrix", g_cart_id == 0 );

  clover_term_fini ( clover_term );
  clover_term_init ( &((*mzzinv)[0]), 8);
  clover_term_init ( &((*mzzinv)[1]), 8);

  gettimeofday ( &ta, (struct timezone *)NULL );

  clover_mzz_inv_matrix ( (*mzzinv)[0], (*mzz)[0] );

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "init_clover", "clover_mzz_inv_matrix", g_cart_id == 0 );

  gettimeofday ( &ta, (struct timezone *)NULL );

  clover_mzz_inv_matrix ( (*mzzinv)[1], (*mzz)[1] );

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "init_clover", "clover_mzz_inv_matrix", g_cart_id == 0 );

  /*
  (*mzz)[0]    = g_mzz_up;
  (*mzz)[1]    = g_mzz_dn;
  (*mzzinv)[0] = g_mzzinv_up;
  (*mzzinv)[1] = g_mzzinv_dn;
  */
  return(0);

}  /* end of init_clover */


void fini_clover ( double **(*mzz)[2], double **(*mzzinv)[2] ) {
  clover_term_fini( &((*mzz)[0])    );
  clover_term_fini( &((*mzz)[1])    );
  clover_term_fini( &((*mzzinv)[0]) );
  clover_term_fini( &((*mzzinv)[1]) );
}  /* end of fini_clover */


}
