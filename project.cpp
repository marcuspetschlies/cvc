/************************************************
 * project.cpp
 *
 * Sat Oct 22 12:00:47 CEST 2016
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
#include "iblas.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "project.h"

namespace cvc {

int project_spinor_field(double *s, double * r, int parallel, double *V, int num, unsigned int N) {

  unsigned int k, status;
  double _Complex *r_ptr = (double _Complex *)r;
  double _Complex *s_ptr = (double _Complex *)s;
/*  double _Complex *V_ptr = (double _Complex *)V; */
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

/******************************************************************************************************
 * project propagator field (num1 spinor fields)
 *
 * B = s is num1 x [12N] (C) = [12N] x num1 (F)
 * A = V is num2 x [12N] (C) = [12N] x num2 (F)
 *
 * zgemm calculates p = V^H x s, which is num2  x num1 (F) = num1 x num2  (C)
 *
 * zgemm calculates V x p, which is [12N] x num1 (F) = num1 x [12N] (C)
 *
 * r and s can be identical
 *
 ******************************************************************************************************/
int project_propagator_field(double *s, double * r, int parallel, double *V, int num1, int num2, unsigned int N) {

  const int items_p = num1 * num2;  /* number of double _Complex items */
  const size_t bytes_p = (size_t)items_p * sizeof(double _Complex);
  unsigned int status;
  int BLAS_M, BLAS_N, BLAS_K; 
  int BLAS_LDA, BLAS_LDB ,BLAS_LDC;
  char BLAS_TRANSA, BLAS_TRANSB;
  double _Complex *p = NULL, *p_buffer = NULL;
  double _Complex BLAS_ALPHA, BLAS_BETA;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double ratime, retime;
 
  if (s == NULL || r == NULL || V == NULL || num1 <= 0 || num2 <= 0) {
    fprintf(stderr, "[project_propagator_field] Error, wrong parameter values\n");
    return(4);
  }
 
  ratime = _GET_TIME;

  if( (p = (double _Complex*)malloc(bytes_p)) == NULL ) {
    fprintf(stderr, "[project_propagator_field] Error from malloc\n");
    return(1);
  }

  /* projection on V-basis */
  BLAS_ALPHA  = 1.;
  BLAS_BETA   = 0.;
  BLAS_TRANSA = 'C';
  BLAS_TRANSB = 'N';
  BLAS_M      = num2;
  BLAS_K      = 12*N;
  BLAS_N      = num1;
  BLAS_A      = (double _Complex*)V;
  BLAS_B      = (double _Complex*)r;
  BLAS_C      = p;
  BLAS_LDA    = BLAS_K;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
  /* allreduce across all processes */
  if( (p_buffer = (double _Complex*)malloc( bytes_p )) == NULL ) {
    fprintf(stderr, "[project_propagator_field] Error from malloc\n");
    return(2);
  }

  memcpy(p_buffer, p, bytes_p);
  status = MPI_Allreduce(p_buffer, p, 2*items_p, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  if(status != MPI_SUCCESS) {
    fprintf(stderr, "[project_propagator_field] Error from MPI_Allreduce, status was %d\n", status);
    return(1);
  }
  free(p_buffer); p_buffer = NULL;
#endif

  /* expand in V-basis or subtract expansion in V-basis  */
  if(parallel) {
    BLAS_ALPHA  =  1.;
    BLAS_BETA   =  0.;
  } else {
    if(r != s) {
      /* copy r to s */
      size_t bytes = 24 * (size_t)num1 * (size_t)N * sizeof(double);
      memcpy(s, r, bytes ) ;
    }
    BLAS_ALPHA  = -1.;
    BLAS_BETA   =  1.;
  }
  BLAS_TRANSA = 'N';
  BLAS_TRANSB = 'N';
  BLAS_M      = 12*N;
  BLAS_K      = num2;
  BLAS_N      = num1;
  BLAS_A      = (double _Complex*)V;
  BLAS_B      = p;
  BLAS_C      = (double _Complex*)s;
  BLAS_LDA    = BLAS_M;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  free(p);

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [project_propagator_field] time for projection = %e seconds\n", retime-ratime);

  return(0);
}  /* end of project_propagator_field */

/******************************************************************************************************
 * project coefficents
 *
 * B = s is num1 x [12N] (C) = [12N] x num1 (F)
 * A = V is num2 x [12N] (C) = [12N] x num2 (F)
 *
 * zgemm calculates p = V^H x r, which is num2  x num1 (F) = num1 x num2  (C)
 *
 ******************************************************************************************************/
int project_reduce_from_propagator_field (double *p, double * r, double *V, int num1, int num2, unsigned int N) {

  const int items_p = num1 * num2;  /* number of double _Complex items */
  const size_t bytes_p = (size_t)items_p * sizeof(double _Complex);
  unsigned int status;
  int BLAS_M, BLAS_N, BLAS_K; 
  int BLAS_LDA, BLAS_LDB ,BLAS_LDC;
  char BLAS_TRANSA, BLAS_TRANSB;
  double _Complex *p_buffer = NULL;
  double _Complex BLAS_ALPHA, BLAS_BETA;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double ratime, retime;
 
  if (p == NULL || r == NULL || V == NULL || num1 <= 0 || num2 <= 0) {
    fprintf(stderr, "[project_reduce_from_propagator_field] Error, wrong parameter values\n");
    return(4);
  }
 
  ratime = _GET_TIME;

  /* projection on V-basis */
  BLAS_ALPHA  = 1.;
  BLAS_BETA   = 0.;
  BLAS_TRANSA = 'C';
  BLAS_TRANSB = 'N';
  BLAS_M      = num2;
  BLAS_K      = 12*N;
  BLAS_N      = num1;
  BLAS_A      = (double _Complex*)V;
  BLAS_B      = (double _Complex*)r;
  BLAS_C      = (double _Complex*)p;
  BLAS_LDA    = BLAS_K;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
  /* allreduce across all processes */
  if( (p_buffer = (double _Complex*)malloc( bytes_p )) == NULL ) {
    fprintf(stderr, "[project_reduce_from_propagator_field] Error from malloc\n");
    return(2);
  }

  memcpy(p_buffer, p, bytes_p);
  status = MPI_Allreduce(p_buffer, p, 2*items_p, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  if(status != MPI_SUCCESS) {
    fprintf(stderr, "[project_reduce_from_propagator_field] Error from MPI_Allreduce, status was %d\n", status);
    return(1);
  }
  free(p_buffer); p_buffer = NULL;
#endif

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [project_reduce_from_propagator_field] time for projection = %e seconds\n", retime-ratime);

  return(0);
}  /* end of project_reduce_from_propagator_field */


/******************************************************************************************************
 * project propagator field (num1 spinor fields)
 *
 * B = p is num1 x num2  (C) = num2  x num1 (F)
 * A = V is num2 x [12N] (C) = [12N] x num2 (F)
 *
 * C = s is num1 x [12N] (C) = [12N] x num1 (F)
 *
 * zgemm calculates V x p, which is [12N] x num1 (F) = num1 x [12N] (C)
 *
 ******************************************************************************************************/
int project_expand_to_propagator_field(double *s, double *p, double *V, int num1, int num2, unsigned int N) {

  int BLAS_M, BLAS_N, BLAS_K; 
  int BLAS_LDA, BLAS_LDB ,BLAS_LDC;
  char BLAS_TRANSA, BLAS_TRANSB;
  double _Complex BLAS_ALPHA, BLAS_BETA;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double ratime, retime;
 
  if (s == NULL || p == NULL || V == NULL || num1 <= 0 || num2 <= 0) {
    fprintf(stderr, "[project_propagator_field] Error, wrong parameter values\n");
    return(4);
  }
 
  ratime = _GET_TIME;

  /* expand in V-basis */
  BLAS_ALPHA  =  1.;
  BLAS_BETA   =  0.;
  BLAS_TRANSA = 'N';
  BLAS_TRANSB = 'N';
  BLAS_M      = 12*N;
  BLAS_K      = num2;
  BLAS_N      = num1;
  BLAS_A      = (double _Complex*)V;
  BLAS_B      = (double _Complex*)p;
  BLAS_C      = (double _Complex*)s;
  BLAS_LDA    = BLAS_M;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [project_expand_to_propagator_field] time for projection = %e seconds\n", retime-ratime);

  return(0);
}  /* end of project_expand_to_propagator_field */

}  /* end of namespace cvc */
