/************************************************
 * prepare_source.cpp
 *
 * Di 12. Jul 17:02:38 CEST 2016
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * CHANGES:
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <getopt.h>

#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "ranlxd.h"

namespace cvc {

#ifdef F_
#define _F(s) s##_
#else
#define _F(s) s
#endif

extern "C" void _F(zgemv) ( char*TRANS, int *M, int *N, double _Complex *ALPHA, double _Complex *A, int *LDA, double _Complex *X,
  int *INCX, double _Complex *BETA, double _Complex * Y, int *INCY, int len_TRANS);

int prepare_volume_source(double *s, unsigned int V) {

  int status = 0;

  switch(g_noise_type) {
    case 1:
      status = rangauss(s, 24*V);
      break;
    case 2:
      status = ranz2(s, 24*V);
      break;
  }

  return(status);
}  /* end of prepare_volume_source */


int project_spinor_field(double *s, double * r, int parallel, double *V, int num, unsigned int N) {

  unsigned int k, status;
  double _Complex *r_ptr = (double _Complex *)r;
  double _Complex *s_ptr = (double _Complex *)s;
  double _Complex *V_ptr = (double _Complex *)V;
  double _Complex *p = NULL, *p_buffer = NULL;
  int BLAS_M = 12*N;
  int BLAS_N = num;
  int BLAS_LDA = BLAS_M;
  int BLAS_INCX = 1;
  int BLAS_INCY = 1;
  double _Complex BLAS_ALPHA=1., BLAS_BETA=0.;
  char BLAS_TRANS = 'T';

  double _Complex *BLAS_A = (double _Complex *)V;
  double _Complex *BLAS_X = s_ptr;
  double _Complex *BLAS_Y = NULL;
 
  if (s == NULL || r == NULL || V == NULL || num <= 0 ) {
    fprintf(stderr, "[project_spinor_field] Error, wrong parameter values\n");
    return(4);
  }

  if( (p = (double _Complex*)malloc(num * sizeof(double _Complex))) == NULL ) {
    fprintf(stderr, "[project_spinor_field] Error from malloc\n");
    return(1);
  }
  BLAS_Y = p;

  if( (p_buffer = (double _Complex*)malloc(num * sizeof(double _Complex))) == NULL ) {
    fprintf(stderr, "[project_spinor_field] Error from malloc\n");
    return(2);
  }

  /* complex conjugate r */
#ifdef HAVE_OPENMP
#pragma omp parallel for shared(s_ptr,r_ptr)
#endif
  for(k=0; k<12*N; k++) { 
    s_ptr[k] = conj(r_ptr[k]); 
  } 

/*
  for(k=0; k<12*N; k++) { 
    fprintf(stdout, "sc_ptr[%d,%d] = %25.16e + %25.16e*1.i\n", g_cart_id+1,k+1,creal(s_ptr[k]), cimag(s_ptr[k]));
  }
  for(k=0; k<12*N; k++) { 
    fprintf(stdout, "V[%d,%d] = %25.16e + %25.16e*1.i\n", g_cart_id+1,k+1,creal(V_ptr[k]), cimag(V_ptr[k]));
  }
*/

  /* multiply p = (V^+ r)^* */
  _F(zgemv)(&BLAS_TRANS, &BLAS_M, &BLAS_N, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_X, &BLAS_INCX, &BLAS_BETA, BLAS_Y, &BLAS_INCY,1);

  /* for(k=0; k<num; k++) { fprintf(stdout, "# p %3d %3d %25.16e %25.16e\n", g_cart_id, k, creal(p[k]), cimag(p[k])); } */

  /* complex conjugate p */
#ifdef HAVE_OPENMP
#pragma omp parallel for shared(p_buffer,p)
#endif
  for(k=0; k<num; k++) { p_buffer[k] = conj(p[k]); } 

#ifdef HAVE_MPI
  status = MPI_Allreduce(p_buffer, p, 2*num, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  if(status != MPI_SUCCESS) {
    fprintf(stderr, "[project_spinor_field] Error from MPI_Allreduce, status was %d\n", status);
    return(1);
  }
#else
  memcpy(p, p_buffer, num*sizeof(double _Complex));
#endif

/*
  if(g_cart_id == 1) {
    for(k=0; k<num; k++) { 
      fprintf(stdout, "# [project_spinor_field] %3d %25.16e %25.16e\n", k, creal(p[k]), cimag(p[k])); 
    } 
  }
*/
  /* s = V r */
  BLAS_TRANS = 'N';
  BLAS_M = 12*N;
  BLAS_N = num;
  BLAS_LDA = BLAS_M;
  BLAS_A = (double _Complex*)V;
  BLAS_X = p;
  BLAS_Y = s_ptr;
  BLAS_INCX =1.;
  BLAS_INCY =1.;

  if(parallel) {
    BLAS_ALPHA = 1.;
    BLAS_BETA  = 0.;
  } else {
    memcpy(s,r,24*N*sizeof(double));
    BLAS_ALPHA = -1.;
    BLAS_BETA  =  1.;
  }

  _F(zgemv)(&BLAS_TRANS, &BLAS_M, &BLAS_N, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_X, &BLAS_INCX, &BLAS_BETA, BLAS_Y, &BLAS_INCY,1);


  if(p != NULL) free(p);
  if(p_buffer != NULL) free(p_buffer);

  return(0);
}  /* end of project_spinor_field */

/*********************************************************
 * out: s_even, s_odd, even and odd part of source field
 * in: source_coords, global source coordinates
 *     have_source == 1 if process has source, otherwise 0
 *     work0, work1,... auxilliary eo work fields
 *
 * IMPORTANT: s_even and s_odd must have halo sites 
 *********************************************************/
int init_eo_spincolor_pointsource_propagator(double *s_even, double *s_odd, int global_source_coords[4], int isc, int sign, int have_source, double *work0) {
 
  unsigned int Vhalf = VOLUME/2;

  int local_source_coords[4] = { global_source_coords[0]%T, global_source_coords[1]%LX, global_source_coords[2]%LY, global_source_coords[3]%LZ };

  int source_location_iseven = have_source ? g_iseven[ g_ipt[local_source_coords[0]][local_source_coords[1]][local_source_coords[2]][local_source_coords[3]] ] : -1;

  unsigned int eo_source_location = g_lexic2eosub[ g_ipt[local_source_coords[0]][local_source_coords[1]][local_source_coords[2]][local_source_coords[3]] ];

  double spinor1[24];
  size_t bytes = 24*Vhalf*sizeof(double);

  /* all procs: initialize to zero */
  memset(s_even, 0, bytes);
  memset(s_odd,  0, bytes);

  /* source node: set source */

  if(source_location_iseven) {

    if(have_source) {
      work0[_GSI(eo_source_location) + 2*isc] = 1.0;
    }

    /* all procs:
     *  g5 X_oe = g5 (g5 Xbar_eo^+ g5) = Xbar_eo^+ g5
     */
    X_oe (s_odd, work0, sign*g_mu, g_gauge_field);
    g5_phi(s_odd, Vhalf);

    /* M_oo^-1 even */
    M_zz_inv (s_even, work0, sign*g_mu);
  } else {
    if(have_source) {
      spinor1[2*isc] = 1.0;
      _fv_eq_gamma_ti_fv( s_odd+_GSI( eo_source_location ), 5, spinor1 );
    }
  }

  /* all procs: xchange even field */
  xchange_eo_field(s_even, 0);
  /* all procs: xchange odd field */
  xchange_eo_field(s_odd,  1);

  /* done */

  return(0);
}  /* end of prepare_eo_spincolor_point_source */

int fini_eo_spincolor_pointsource_propagator(double *p_even, double *p_odd, double *r_even, double *r_odd , int sign, double *work0) {
 
  unsigned int Vhalf = VOLUME/2;

  size_t bytes = 24*Vhalf*sizeof(double);

  memcpy( p_odd, r_odd, bytes);
  spinor_field_ti_eq_re (p_odd, 2.*g_kappa, Vhalf);

  X_eo (p_even, p_odd, sign*g_mu, g_gauge_field);
  spinor_field_pl_eq_spinor_field(p_even, r_even, Vhalf);

  /* done */

  return(0);
}  /* end of fini_eo_spincolor_pointsource_propagator */


}  /* end of namespace cvc */
