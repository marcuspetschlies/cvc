#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_MPI
#include <mpi.h>  
#endif
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "mpi_init.h"
#include "cvc_geometry.h"
#include "table_init_d.h"
#include "cvc_utils.h"
#include "smearing_techniques.h"
#include "gradient_flow.h"

namespace cvc {

/******************************************************************
 * calculate \bar{Z}_X
 * = a Z(g) + b z
 ******************************************************************/
void ZX ( double * const z, double * const g, double const a, double const b ) {
#ifdef HAVE_MPI
    xchange_gauge_field ( g );
#endif
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  double tmp[18], tmp2[18];
  double * _z = NULL;
  double * _g = NULL;
  unsigned int iix;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for ( unsigned int x = 0; x < VOLUME; ++x) {
    for ( int mu = 0; mu < 4; ++mu)
    {
        iix      = _GGI( x, mu);
        _z       = z + iix;
        _g       = g + iix;

        /* calculate new staples */
        generic_staples( tmp, x, mu, g );

        /* multiply by new weight a */
        _cm_ti_eq_re( tmp, a );

        /* complete plaquette by multiplication with U_mu,x^+ */
        _cm_eq_cm_ti_cm_dag ( tmp2, tmp, _g );

        /* antihermitean part */
        _cm_eq_antiherm_trless_cm ( tmp, tmp2 );

        /* z_old <- z_old * b */
        _cm_ti_eq_re ( _z , b );

        /* z_new <- z_old + tmp */
        _cm_pl_eq_cm ( _z , tmp );
    }
  }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  return;
}  /* end of ZX */

/******************************************************************
 * calculate exp ( \bar{Z}_X dt ) g
 ******************************************************************/
void apply_ZX ( double * const g, double * const z, double const dt ) {
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  double tmp[18], tmp2[18];
  double * _z = NULL;
  double * _g = NULL;
  unsigned int iix;

#ifdef HAVE_OPENMP
#pragma omp barrier
#pragma omp for
#endif
    for ( unsigned int x = 0; x < VOLUME; ++x) {
      for(int mu = 0 ; mu < 4; ++mu)
      {
        iix = _GGI( x, mu);
        _z  = z + iix;
        _g  = g + iix;

        /* multiply z by epsilon
         * tmp <- z * dt */
        _cm_eq_cm_ti_re ( tmp, _z, dt );

        /* exponentiate
         * tmp2 <- exp( tmp ) */
        exposu3 ( tmp2, tmp );

        /* tmp <- tmp2 x g = exp( dt z ) g */
        _cm_eq_cm_ti_cm( tmp, tmp2, _g );

        /* g <- tmp = tmp2 g = exp ( dt z ) g */
        _cm_eq_cm ( _g, tmp );
      }
    }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

/******************************************************************
 * apply Laplace to spinor field
 ******************************************************************/
int spinor_field_eq_cov_displ_spinor_field ( double * const s, double * const r_in, int const mu, int const fbwd, double * const gauge_field ) {

  const unsigned int N = VOLUME;

  double * r = (double*) malloc ( _GSI( (VOLUME+RAND) ) * sizeof(double) );
  if ( r == NULL ) {
    fprintf ( stderr, "[spinor_field_eq_cov_displ_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__ );
    return ( 1 );
  }
  memcpy ( r, r_in, _GSI(VOLUME)*sizeof(double) );

#ifdef HAVE_MPI
  xchange_field( r );
#endif

  if ( fbwd == 0 ) {
    /**************************************************************
     * FORWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int ix = 0; ix < N; ix++ ) {

      double * const s_ = s + _GSI(ix);

      /* ============================================ */
      /* =============== direction mu =============== */
      double * const U_fwd = gauge_field+_GGI(ix,mu);

      /* ix_fwd */
      unsigned int const ix_fwd = g_iup[ix][mu];

      double * const r_fwd_ = r + _GSI(ix_fwd);

      /* s_ = U_fwd r_fwd_ */
      _fv_eq_cm_ti_fv( s_, U_fwd, r_fwd_ );

  }  /* end of loop on ix over VOLUME */

} else if ( fbwd == 1 ) {
    /**************************************************************
     * BACKWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int ix = 0; ix < N; ix++ ) {

      double * const s_ = s + _GSI(ix);

      /* ============================================ */
      /* =============== direction mu =============== */
      double * const U_bwd = gauge_field+_GGI( g_idn[ix][mu], mu);

      /* ix_bwd */
      unsigned int const ix_bwd = g_idn[ix][mu];

      double * const r_bwd_ = r + _GSI(ix_bwd);

      /* s_ = U_bwd^+ r_bwd_ */
      _fv_eq_cm_dag_ti_fv( s_, U_bwd, r_bwd_ );

  }  /* end of loop on ix over VOLUME */

 }  /* end of if fbwd = 0 or 1 */

  free ( r );
  return ( 0 );
}  /* end of spinor_field_eq_cov_displ_spinor_field */


/******************************************************************
 * using 4-dimensional isotropic stout smearing as kernel
 ******************************************************************/
void flow_fwd_gauge_spinor_field ( double * const g, double * const phi, unsigned int const niter, double const dt ) {

  size_t sizeof_gauge_field = 72 * VOLUME * sizeof ( double ) ;

  double * w = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
  double * z = init_1level_dtable ( 72 * VOLUME );

  memcpy ( w, g, sizeof_gauge_field );

  /******************************************************************
   * STEP 0
   *
   * W0 = g
   ******************************************************************/
  xchange_gauge_field ( w );

  /* loop on time steps */
  for ( unsigned int iter = 0; iter < niter; iter++ ) {

    /******************************************************************
     * STEP 1
     *
     * W1 = exp( 1/4 Z0 ) W0
     *
     ******************************************************************/

    ZX ( z, w, 1./4., 0. );

    apply_ZX ( w, z, dt );

    /******************************************************************
     * STEP 2
     *
     * W2 = exp( 8/9 Z1 - 17/36 Z0 ) W1
     *
     ******************************************************************/

    ZX ( z, w, 8./9., -17./9. );

    apply_ZX ( w, z, dt );

    /******************************************************************
     * STEP 3
     *
     * W3 = exp( 3/4 Z2 -8/9 Z1 + 17/36 Z0 ) W2
     *
     ******************************************************************/

    ZX ( z, w, 3./4., -1. );

    apply_ZX ( w, z, dt );

  }  /* end of loop on iterations */


  fini_1level_dtable ( &w );
  fini_1level_dtable ( &z );

}  /* end of flow_fwd_gauge_field */



}  /* end of namespace cvc */

