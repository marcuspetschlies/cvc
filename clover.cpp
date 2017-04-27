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

namespace cvc {

int init_clover ( double **(*mzz)[2], double **(*mzzinv)[2], double*gauge_field ) {
  
  double ratime, retime;

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  clover_term_init(&g_clover, 6);
  clover_term_init(&g_mzz_up, 6);
  clover_term_init(&g_mzz_dn, 6);


  ratime = _GET_TIME;
  clover_term_eo (g_clover, gauge_field );
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [init_clover] time for clover_term_eo = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (g_mzz_up, g_clover, g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [init_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (g_mzz_dn, g_clover, -g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [init_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  clover_term_fini( &g_clover );
  clover_term_init(&g_mzzinv_up, 8);
  clover_term_init(&g_mzzinv_dn, 8);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (g_mzzinv_up, g_mzz_up);
  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [init_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (g_mzzinv_dn, g_mzz_dn);
  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [init_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  (*mzz)[0]    = g_mzz_up;
  (*mzz)[1]    = g_mzz_dn;
  (*mzzinv)[0] = g_mzzinv_up;
  (*mzzinv)[1] = g_mzzinv_dn;

  return(0);

}  /* end of init_clover */


void fini_clover (void) {
  clover_term_fini( &g_mzz_up    );
  clover_term_fini( &g_mzz_dn    );
  clover_term_fini( &g_mzzinv_up );
  clover_term_fini( &g_mzzinv_dn );
}  /* end of fini_clover */

}
