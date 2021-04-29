/*********************************************
 * scalar_products.cpp
 * 
 * Sun Feb  5 13:48:24 CET 2017a
 *
 * PURPOSE:
 * - scalar product functions for spinor fields
 *********************************************/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include "global.h"
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "mpi_init.h"
#include "Q_phi.h"
#include "cvc_utils.h"
#include "scalar_products.h"

namespace cvc {


/*********************************************/
/*********************************************/

/*********************************************
* complex-valued 4-dim scalar product of two
* spinor fields
*********************************************/
void spinor_scalar_product_co ( complex * const w, double * const xi, double * const phi, unsigned int const V) {

  complex paccum;
  
#ifdef HAVE_MPI
 
#endif
#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif
  paccum.re = 0.;
  paccum.im = 0.;

#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel default(shared)
{
#endif
  complex p2;
  p2.re = 0.;
  p2.im = 0.;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int ix = 0; ix < V; ix ++ ) {
    unsigned int const iix = _GSI( ix );
    _co_pl_eq_fv_dag_ti_fv(&p2, xi+iix, phi+iix);
  }
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
#endif

  paccum.re += p2.re;
  paccum.im += p2.im;

#ifdef HAVE_OPENMP
  omp_unset_lock(&writelock);
}  /* end of parallel region */
  omp_destroy_lock(&writelock);
#endif

  /* fprintf(stdout, "# [spinor_scalar_product_co] %d local: %e %e\n", g_cart_id, paccum.re, paccum.im); */

#ifdef HAVE_MPI
  complex pall;
  pall.re=0.; pall.im=0.;
  if ( MPI_Allreduce(&paccum, &pall, 2, MPI_DOUBLE, MPI_SUM, g_cart_grid) != MPI_SUCCESS ) {
    if ( g_cart_id == 0 ) fprintf ( stderr, "[] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
    w->re = sqrt( -1. );
    w->im = sqrt( -1. );
  } else {
    w->re = pall.re;
    w->im = pall.im;
  }
#else
  w->re = paccum.re;
  w->im = paccum.im;
#endif
}  /* end of spinor_scalar_product_co */

/*********************************************/
/*********************************************/

/*********************************************
 * real-valued 4-dim scalar product of two
 * spinor fields
 *********************************************/
void spinor_scalar_product_re(double * const r, double * const xi, double * const phi, unsigned int const V) {

  double w = 0.;

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif
  
#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel default(shared) 
{
#endif
   
  double w2 = 0.;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int ix = 0; ix < V; ix++ ) {
    unsigned int const iix = _GSI(ix);
    _re_pl_eq_fv_dag_ti_fv(w2, xi+iix, phi+iix);
  }
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
#endif

  w += w2;

#ifdef HAVE_OPENMP
  omp_unset_lock(&writelock);
}  /* end of parallel region */
  omp_destroy_lock(&writelock);
#endif

  /* fprintf(stdout, "# [spinor_scalar_product_re] %d local: %e\n", g_cart_id, w); */
#ifdef HAVE_MPI
  double wall = 0.;
  if ( MPI_Allreduce(&w, &wall, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid) != MPI_SUCCESS ) {
    if ( g_cart_id == 0 ) fprintf ( stderr, "[spinor_scalar_product_re] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
    *r = sqrt ( -1. );
  } else {
    *r = wall;
  }
#else
  *r = w;
#endif
}  /* end of spinor_scalar_product_re */

/*************************************************
 * eo = 0 --- even subfield
 * eo = 1 --- odd subfield
 *************************************************/
void eo_spinor_spatial_scalar_product_co( double _Complex * w, double * const xi, double * const phi, int const eo) {
 
  unsigned int const N = VOLUME / 2;

  memset( w, 0, T*sizeof(double _Complex) );

#ifdef HAVE_OPENMP
  omp_lock_t writelock;

  omp_init_lock(&writelock);
#pragma omp parallel default(shared)
{
#endif
  complex p2[T];
  memset(p2, 0, T*sizeof(complex));

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int ix = 0; ix < N; ix++ ) {
    unsigned int const iix = _GSI(ix);
    unsigned int const it  = g_eosub2t[eo][ix];
    _co_pl_eq_fv_dag_ti_fv( (p2+it), xi+iix, phi+iix);
  }
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
#endif
  for( unsigned int it = 0; it < T; it++) {
    w[it] += p2[it].re + p2[it].im * I;
    // TEST
    // fprintf(stdout, "# [eo_spinor_spatial_scalar_product_co] proc%.4d thread%.4d %3d %25.16e %25.16e\n", g_cart_id, threadid, it, p2[it].re, p2[it].im );
  }

#ifdef HAVE_OPENMP
  omp_unset_lock(&writelock);
}  // end of parallel region
  omp_destroy_lock(&writelock);
#endif
  // fprintf(stdout, "# [spinor_scalar_product_co] %d local: %e %e\n", g_cart_id, creal(p2), cimag(p2));

#ifdef HAVE_MPI
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  double px[2*T];
  memcpy( px, w, 2*T*sizeof(double) );
  if ( MPI_Allreduce(px, w, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm) != MPI_SUCCESS ) {
    fprintf ( stderr, "[eo_spinor_spatial_scalar_product_co] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }
#  endif
#endif
}  // eo_spinor_spatial_scalar_product_co

/*********************************************/
/*********************************************/

/*********************************************
 * complex-valued 4-dim generalized scalar product
 * of two spinor fields, including Dirac gamma
 * matrix
 *********************************************/
void eo_spinor_dag_gamma_spinor(complex * const gsp, double * const xi, int const gid, double * const phi) {

  const unsigned int N = VOLUME / 2;

#ifdef HAVE_OPENMP
#pragma omp parallel default(shared)
{
#endif
  double spinor1[24];
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(unsigned int ix=0; ix<N; ix++) {
    unsigned int const iix = _GSI(ix);
    double * const phi_  = phi + iix;
    double * const xi_   = xi  + iix;
    complex * const gsp_ = gsp + ix;

    _fv_eq_gamma_ti_fv( spinor1, gid, phi_ );
    _co_eq_fv_dag_ti_fv( gsp_, xi_, spinor1 );
  }
#ifdef HAVE_OPENMP
}
#endif

}  /* end of eo_spinor_dag_gamma_spinor */

/*************************************************************
 * Note, that the phase field must be the even phase field for eo = 0
 * and the odd phase field for eo = 1
 *************************************************************/
void eo_gsp_momentum_projection (complex * const gsp_p, complex * const gsp_x, complex * const phase, int const eo) {
  
  const unsigned int N = VOLUME / 2;
  const int *index_ptr = g_eosub2t[eo];

  complex p[T];
#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#else
  complex *p2 = p;
#endif

  memset(p, 0, T*sizeof(complex));
#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel default(shared)
{
  complex p2[T];

  memset(p2, 0, T*sizeof(complex));
#pragma omp for
#endif
  for( unsigned int ix=0; ix<N; ix++) {
    int const it = index_ptr[ix];
    complex * const phase_ = phase + ix;
    complex * const gsp_x_ = gsp_x + ix;
    complex * const p2_ = p2 + it;
    _co_pl_eq_co_ti_co(p2_, gsp_x_ , phase_  );
  }
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
  for(int it=0; it<T; it++) {
    _co_pl_eq_co(p+it, p2+it);
  }
  omp_unset_lock(&writelock);
}  /* end of parallel region */

  omp_destroy_lock(&writelock);

#endif


#ifdef HAVE_MPI
  MPI_Allreduce(p, gsp_p, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
#else
  memcpy(gsp_p, p, 2*T*sizeof(double));
#endif
}  /* end of eo_gsp_momentum_projection */


/*************************************************************
 * double-2-valued 4-dim scalar product of two
 * spinor fields
 *************************************************************/
void spinor_scalar_product_d2 ( double * const w, double * const xi, double * const phi, unsigned int const V) {

  double paccum[2] = { 0., 0. };
  
#ifdef HAVE_OPENMP
  omp_lock_t writelock;

  omp_init_lock(&writelock);
#pragma omp parallel default(shared)
{
#endif
  double p2[2] = { 0., 0. };

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int ix = 0; ix < V; ix ++ ) {

    unsigned int const iix = _GSI( ix );

    double * const _xi  =  xi + iix;
    double * const _phi = phi + iix;

    _d2_pl_eq_fv_dag_ti_fv ( p2, _xi, _phi );
  }
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
#endif

  paccum[0] += p2[0];
  paccum[1] += p2[1];

#ifdef HAVE_OPENMP
  omp_unset_lock(&writelock);
}  /* end of parallel region */
  omp_destroy_lock(&writelock);
#endif

#ifdef HAVE_MPI
  double pall[2] = { 0., 0. };

  if ( MPI_Allreduce ( paccum, pall, 2, MPI_DOUBLE, MPI_SUM, g_cart_grid) != MPI_SUCCESS ) {
    if ( g_cart_id == 0 ) fprintf ( stderr, "[spinor_scalar_product_d2] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
    w[0] = sqrt( -1. );
    w[1] = sqrt( -1. );
  } else {
    w[0] = pall[0];
    w[1] = pall[1];
  }
#else
  w[0] = paccum[0];
  w[1] = paccum[1];
#endif
}  /* end of spinor_scalar_product_d2 */

/*************************************************************/
/*************************************************************/

}  /* end of namespace cvc */
