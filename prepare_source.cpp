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

}  /* end of namespace cvc */
