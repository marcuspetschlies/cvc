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
void spinor_scalar_product_co(complex *w, double *xi, double *phi, int V) {

  complex paccum;

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif
  paccum.re = 0.;
  paccum.im = 0.;

#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel default(shared) firstprivate(V) shared(xi,phi,paccum)
{
#endif
  int iix;
  complex p2 =  {0., 0.};

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int ix = 0; ix < V; ix++ ) {
    iix = _GSI( ix );
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
  if ( MPI_Allreduce(&paccum, w, 2, MPI_DOUBLE, MPI_SUM, g_cart_grid) != MPI_SUCCESS ) {
    fprintf(stderr, "[spinor_scalar_product_co] Error from MPI_Allreduce\n");
    EXIT(1);
  }
#else
  w->re = paccum.re;
  w->im = paccum.im;
#endif
}  /* end of spinor_scalar_product_co */

/*************************************************/
/*************************************************/

void spinor_scalar_product_re(double *r, double *xi, double *phi, int V) {

  double w = 0.;

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif
  
#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel default(shared) shared(w,xi,phi,V)
{
#endif
  unsigned int iix;
  double w2 = 0.;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int ix = 0; ix < V; ix++ ) {
    iix = _GSI (ix);
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
  if ( MPI_Allreduce(&w, r, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid) !=  MPI_SUCCESS ) {
    fprintf(stderr, "[spinor_scalar_product_re] Error from MPI_Allreduce\n");
    EXIT(1);
  }
#else
  *r = w;
#endif
  return;
}  /* end of spinor_scalar_product_re */

/*************************************************/
/*************************************************/

/*************************************************
 * eo = 0 --- even subfield
 * eo = 1 --- odd subfield
 *************************************************/
void eo_spinor_spatial_scalar_product_co(complex *w, double *xi, double *phi, int eo) {
 
  const int nthreads = g_num_threads;
  const int sincr = _GSI(nthreads);
  const unsigned int N = VOLUME / 2;

  int ix, iix, it;
  int threadid = 0;
  complex p[T];
#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#else
  complex *p2 = p;
#endif

  memset(p, 0, T*sizeof(complex));

#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel default(shared) private(ix,iix,it,threadid) shared(xi,phi,eo)
{
  complex p2[T];
  threadid = omp_get_thread_num();
#endif
  memset(p2, 0, T*sizeof(complex));
  iix = _GSI(threadid);
  for(ix = threadid; ix < N; ix += nthreads) {
    it  = g_eosub2t[eo][ix];
    _co_pl_eq_fv_dag_ti_fv( (p2+it), xi+iix, phi+iix);
    iix += sincr;
  }
#ifdef HAVE_OPENMP

  omp_set_lock(&writelock);
  for(it=0; it<T; it++) {
    _co_pl_eq_co(p+it, p2+it);
    /* TEST */
    /* fprintf(stdout, "# [eo_spinor_spatial_scalar_product_co] proc%.4d thread%.4d %3d %25.16e %25.16e\n", g_cart_id, threadid, it, p2[it].re, p2[it].im); */
  }
  omp_unset_lock(&writelock);

}  /* end of parallel region */
  omp_destroy_lock(&writelock);
#endif
  /* fprintf(stdout, "# [spinor_scalar_product_co] %d local: %e %e\n", g_cart_id, p2.re, p2.im); */

#ifdef HAVE_MPI
  memset(w, 0, T*sizeof(complex));
  MPI_Allreduce(p, w, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
#else
  memcpy(w, p, T*sizeof(complex));
#endif
}  /* eo_spinor_spatial_scalar_product_co */

void eo_spinor_dag_gamma_spinor(complex*gsp, double*xi, int gid, double*phi) {

  const int nthreads = g_num_threads;
  const unsigned int N = VOLUME / 2;

  unsigned int ix, iix;
  double spinor1[24];

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(spinor1,ix,iix) shared(xi,gid,phi)
#endif
  for(ix=0; ix<N; ix++) {
    iix = _GSI(ix);
    _fv_eq_gamma_ti_fv(spinor1, gid, phi+iix);
    _co_eq_fv_dag_ti_fv(gsp+ix, xi+iix, spinor1);
  }

}  /* end of eo_spinor_dag_gamma_spinor */

/*************************************************************
 * Note, that the phase field must be the even phase field for eo = 0
 * and the odd phase field for eo = 1
 *************************************************************/
void eo_gsp_momentum_projection (complex *gsp_p, complex *gsp_x, complex *phase, int eo) {
  
  const int nthreads = g_num_threads;
  const unsigned int N = VOLUME / 2;
  const int *index_ptr = g_eosub2t[eo];

  unsigned int ix, it;
  complex p[T];
#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#else
  complex *p2 = p;
#endif

  memset(p, 0, T*sizeof(complex));
#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel default(shared) private(ix,it) shared(phase,gsp_x)
{
  complex p2[T];

  memset(p2, 0, T*sizeof(complex));
#pragma omp for
#endif
  for(ix=0; ix<N; ix++) {
    it = index_ptr[ix];
    _co_pl_eq_co_ti_co(p2+it, gsp_x+ix, phase+ix );
  }
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
  for(it=0; it<T; it++) {
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

}  /* end of namespace cvc */
