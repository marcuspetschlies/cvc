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

#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif


#include "cvc_linalg.h"
#include "iblas.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "matrix_init.h"
#include "table_init_z.h"
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

  if ( (s == r ) && !parallel ) {
    fprintf ( stderr, "[project_spinor_field] Error, s and r must not coincide in memory for orthogonal projection\n");
    return(41);
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
#pragma omp parallel for private(k) shared(s_ptr,r_ptr)
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
  F_GLOBAL(zgemv, ZGEMV)(&BLAS_TRANS, &BLAS_M, &BLAS_N, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_X, &BLAS_INCX, &BLAS_BETA, BLAS_Y, &BLAS_INCY,1);

  /* for(k=0; k<num; k++) { fprintf(stdout, "# p %3d %3d %25.16e %25.16e\n", g_cart_id, k, creal(p[k]), cimag(p[k])); } */

  /* complex conjugate p */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(k) shared(p_buffer,p)
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

  F_GLOBAL(zgemv, ZGEMV)(&BLAS_TRANS, &BLAS_M, &BLAS_N, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_X, &BLAS_INCX, &BLAS_BETA, BLAS_Y, &BLAS_INCY,1);


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

  F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

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

  F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  free(p);

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [project_propagator_field] time for projection = %e seconds\n", retime-ratime);

  return(0);
}  /* end of project_propagator_field */


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
int project_propagator_field_weighted(double *s, double * r, int parallel, double *V, double *weights, int num1, int num2, unsigned int N) {

  const int items_p = num1 * num2;  /* number of double _Complex items */
  const size_t bytes_p = (size_t)items_p * sizeof(double _Complex);
  unsigned int status;
  int BLAS_M, BLAS_N, BLAS_K; 
  int BLAS_LDA, BLAS_LDB ,BLAS_LDC;
  char BLAS_TRANSA, BLAS_TRANSB;
  double _Complex **p = NULL, *p_buffer = NULL;
  double _Complex BLAS_ALPHA, BLAS_BETA;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double ratime, retime;
 
  if (s == NULL || r == NULL || V == NULL || num1 <= 0 || num2 <= 0) {
    fprintf(stderr, "[project_propagator_field_weighted] Error, wrong parameter values\n");
    return(4);
  }
 
  ratime = _GET_TIME;

  if ( init_2level_zbuffer ( &p, num1, num2) != 0 ) {
    fprintf(stderr, "[project_propagator_field_weighted] Error from init_2level_zbuffer %s %d\n", __FILE__, __LINE__);
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
  BLAS_C      = p[0];
  BLAS_LDA    = BLAS_K;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;

  F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
  /* allreduce across all processes */
  if( (p_buffer = (double _Complex*)malloc( bytes_p )) == NULL ) {
    fprintf(stderr, "[project_propagator_field_weighted] Error from malloc\n");
    return(2);
  }

  memcpy(p_buffer, p[0], bytes_p);
  status = MPI_Allreduce(p_buffer, p[0], 2*items_p, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  if(status != MPI_SUCCESS) {
    fprintf(stderr, "[project_propagator_field_weighted] Error from MPI_Allreduce, status was %d\n", status);
    return(1);
  }
  free(p_buffer); p_buffer = NULL;
#endif

  /* weight the projection coefficients */
  for( int i = 0; i < num1; i++ ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for shared(i)
#endif
    for( int k = 0; k < num2; k++ ) {
      p[i][k] *= weights[k];
    }
  }

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
  BLAS_B      = p[0];
  BLAS_C      = (double _Complex*)s;
  BLAS_LDA    = BLAS_M;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;

  F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  fini_2level_zbuffer ( &p );

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [project_propagator_field_weighted] time for projection = %e seconds\n", retime-ratime);

  return(0);
}  /* end of project_propagator_field_weighted */

/******************************************************************************************************
 * project coefficents
 *
 * B = s is num1 x [12N] (C) = [12N] x num1 (F)
 * A = V is num2 x [12N] (C) = [12N] x num2 (F)
 *
 * zgemm calculates p = V^H x r, which is num2  x num1 (F) = num1 x num2  (C)
 *
 ******************************************************************************************************/
int project_reduce_from_propagator_field (double *p, double * r, double *V, int num1, int num2, unsigned int N, int xchange) {

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

  F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
  if ( xchange == 1 ) {
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
  }  /* end of if xchange */
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

  F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [project_expand_to_propagator_field] time for projection = %e seconds\n", retime-ratime);

  return(0);
}  /* end of project_expand_to_propagator_field */


/**************************************************************************************************************
 * V     is nv              x VOL3 (C) = VOL3 x nv (F)
 * phase is momentum_number x VOL3 (C) = VOL3 x momentum_number (F)
 *
 * zgemm calculates t(V) x phase, which is  nv x momentum_number (F) = momentum_number x nv (C)
 **************************************************************************************************************/
int momentum_projection (double*V, double *W, unsigned int nv, int momentum_number, int (*momentum_list)[3]) {

  typedef struct {
    int x[3];
  } point;

  const double MPI2 = M_PI * 2.;
  const unsigned int VOL3 = LX*LY*LZ;

  int x1, x2, x3;
  unsigned int i, ix;
  double _Complex **zphase = NULL;

  char BLAS_TRANSA, BLAS_TRANSB;
  int BLAS_M, BLAS_K, BLAS_N, BLAS_LDA, BLAS_LDB, BLAS_LDC;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  init_2level_buffer( (double***)(&zphase), momentum_number, 2*VOL3 );

  point *lexic_coords = (point*)malloc(VOL3*sizeof(point));
  if(lexic_coords == NULL) {
    fprintf(stderr, "[momentum_projection] Error from malloc\n");
    EXIT(1);
  }
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[0][x1][x2][x3];
    lexic_coords[ix].x[0] = x1;
    lexic_coords[ix].x[1] = x2;
    lexic_coords[ix].x[2] = x3;
  }}}

  /* loop on sink momenta */
  for(i=0; i < momentum_number; i++) {
    /* phase field */
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    const double q[3] = { MPI2 * momentum_list[i][0] / LX_global,
                          MPI2 * momentum_list[i][1] / LY_global,
                          MPI2 * momentum_list[i][2] / LZ_global };
    const double q_offset = g_proc_coords[1]*LX * q[0] + g_proc_coords[2]*LY * q[1] + g_proc_coords[3]*LZ * q[2];
    double q_phase;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix=0; ix<VOL3; ix++) {
      q_phase = q_offset +
                lexic_coords[ix].x[0] * q[0] +
                lexic_coords[ix].x[1] * q[1] +
                lexic_coords[ix].x[2] * q[2];
      zphase[i][ix] = cos(q_phase) + I*sin(q_phase);
    }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }  /* end of loop on sink momenta */

  free( lexic_coords );

  BLAS_TRANSA = 'T';
  BLAS_TRANSB = 'N';
  BLAS_M     = nv;
  BLAS_K     = VOL3;
  BLAS_N     = momentum_number;
  BLAS_A     = (double _Complex*)V;
  BLAS_B     = zphase[0];
  BLAS_C     = (double _Complex*)W;
  BLAS_LDA   = BLAS_K;
  BLAS_LDB   = BLAS_K;
  BLAS_LDC   = BLAS_M;

  F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  fini_2level_buffer((double***)(&zphase));

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
  unsigned int count = 2 * nv * momentum_number;
  void *buffer = malloc(count * sizeof(double));
  if(buffer == NULL) {
    return(1);
  }
  memcpy(buffer, W, count*sizeof(double));
  int status = MPI_Allreduce(buffer, (void*)W, count, MPI_DOUBLE, MPI_SUM, g_ts_comm);
  if(status != MPI_SUCCESS) {
    fprintf(stderr, "[momentum_projection] Error from MPI_Allreduce, status was %d\n", status);
    return(2);
  }
  free(buffer);
#  endif
#endif

  return(0);
}  /* end of momentum_projection */

/**************************************************************************************************************
 * V     is VOL3            x nv   (C) = nv   x VOL3            (F)
 * phase is momentum_number x VOL3 (C) = VOL3 x momentum_number (F)
 *
 * zgemm calculates V x phase, which is  nv x momentum_number (F) = momentum_number x nv (C)
 **************************************************************************************************************/
int momentum_projection2 ( double * const V, double * const W, unsigned int const nv, int const momentum_number, int (* const momentum_list)[3], int const gshift[3] ) {

  /*
  typedef struct {
    int x[3];
  } point; */

  double const MPI2 = M_PI * 2.;
  unsigned int const VOL3 = LX*LY*LZ;

  char BLAS_TRANSA, BLAS_TRANSB;
  int BLAS_M, BLAS_K, BLAS_N, BLAS_LDA, BLAS_LDB, BLAS_LDC;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  /**************************************************************************************************************
   * prepare the Fourier phase matrix
   **************************************************************************************************************/
  /*
  point *lexic_coords = (point*)malloc(VOL3*sizeof(point));
  if(lexic_coords == NULL) {
    fprintf(stderr, "[momentum_projection] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(1);
  }

  for ( int x1=0; x1<LX; x1++) {
  for ( int x2=0; x2<LY; x2++) {
  for ( int x3=0; x3<LZ; x3++) {
    ix = g_ipt[0][x1][x2][x3];
    lexic_coords[ix].x[0] = x1;
    lexic_coords[ix].x[1] = x2;
    lexic_coords[ix].x[2] = x3;
  }}} */

  int shift[3] = {0,0,0};
  if(gshift != NULL) {
    memcpy( shift, gshift, 3*sizeof(int) );
  }

  if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [momentum_projection2] using shift vector (%d, %d, %d)\n", shift[0], shift[1], shift[2]);

  double _Complex ** zphase = init_2level_ztable ( momentum_number, 2*VOL3 );
  if ( zphase == NULL ) {
    fprintf ( stderr, "# [] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  for ( int i = 0; i < momentum_number; i++ ) {
    /* phase field */
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    double const q[3] = { MPI2 * momentum_list[i][0] / LX_global,
                          MPI2 * momentum_list[i][1] / LY_global,
                          MPI2 * momentum_list[i][2] / LZ_global };
    double const q_offset = ( g_proc_coords[1]*LX - shift[0] ) * q[0] + ( g_proc_coords[2]*LY - shift[1] ) * q[1] + ( g_proc_coords[3]*LZ - shift[2] ) * q[2];
    double q_phase;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for ( unsigned int ix=0; ix < VOL3; ix++) {
      q_phase = q_offset \
         + g_lexic2coords[ix][1] * q[0] \
         + g_lexic2coords[ix][2] * q[1] \
         + g_lexic2coords[ix][3] * q[2];

         /* 
         + lexic_coords[ix].x[0]*q[0] \
         + lexic_coords[ix].x[1]*q[1] \
         + lexic_coords[ix].x[2]*q[2]; */

      zphase[i][ix] = cos(q_phase) + I*sin(q_phase);
    }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }  /* end of loop on sink momenta */

  /* free( lexic_coords ); */

  /**************************************************************************************************************
   * now the blas call
   **************************************************************************************************************/
  BLAS_TRANSA = 'N';
  BLAS_TRANSB = 'N';
  BLAS_M     = nv;
  BLAS_K     = VOL3;
  BLAS_N     = momentum_number;
  BLAS_A     = (double _Complex*)V;
  BLAS_B     = zphase[0];
  BLAS_C     = (double _Complex*)W;
  BLAS_LDA   = BLAS_M;
  BLAS_LDB   = BLAS_K;
  BLAS_LDC   = BLAS_M;

  F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  fini_2level_ztable( &zphase );

#ifdef HAVE_MPI
#if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ ) 
  /**************************************************************************************************************
   * MPI_Allreduce on g_cart_grid timeslice level
   **************************************************************************************************************/
  int nelem = 2 * nv * momentum_number;
  void *buffer = malloc ( nelem * sizeof(double) );
  if(buffer == NULL) {
    fprintf(stderr, "[momentum_projection2] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  memcpy ( buffer, W, nelem * sizeof(double) );
  int status = MPI_Allreduce(buffer, (void*)W, nelem, MPI_DOUBLE, MPI_SUM, g_ts_comm);
  if(status != MPI_SUCCESS) {
    fprintf(stderr, "[momentum_projection2] Error from MPI_Allreduce, status was %d\n", status);
    return(2);
  }
  free(buffer);
#endif
#endif  /* of if def HAVE_MPI */

  return(0);
}  /* end of momentum_projection2 */


/*************************************************************************
 * 3d momentum phase field in lexicographic ordering
 *************************************************************************/
void make_lexic_phase_field_3d (double*phase, int *momentum) {

  const double TWO_MPI = 2. * M_PI;
  const double p[3] = { TWO_MPI * momentum[0] / (double)LX_global, TWO_MPI * momentum[1] / (double)LY_global, TWO_MPI * momentum[2] / (double)LZ_global };
  const double phase_part = (g_proc_coords[1]*LX) * p[0] + (g_proc_coords[2]*LY) * p[1] + (g_proc_coords[3]*LZ) * p[2];

  double ratime, retime;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [make_lexic_phase_field_3d] using phase momentum = (%d, %d, %d)\n", momentum[0], momentum[1], momentum[2]);
  }

  ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) shared(phase)
{
#endif
  unsigned int ix;
  int x1, x2, x3;
  double dtmp;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  /* make phase field in lexic ordering */

  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix  = g_ipt[0][x1][x2][x3];
    dtmp = phase_part + x1 * p[0] + x2 * p[1] + x3 * p[2];
    phase[2*ix  ] = cos(dtmp);
    phase[2*ix+1] = sin(dtmp);
  }}}
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [make_lexic_phase_field_3d] time for making lexic phase field = %e seconds\n", retime-ratime);
}  /* end of make_lexic_phase_field_3d */

/*************************************************************************
 * momentum phase field separated in even and odd part
 *************************************************************************/
void make_eo_phase_field (double*phase_e, double*phase_o, int *momentum) {

  const double TWO_MPI = 2. * M_PI;
  const double p[3] = { TWO_MPI * momentum[0] / (double)LX_global, TWO_MPI * momentum[1] / (double)LY_global, TWO_MPI * momentum[2] / (double)LZ_global };
  const double phase_part = (g_proc_coords[1]*LX) * p[0] + (g_proc_coords[2]*LY) * p[1] + (g_proc_coords[3]*LZ) * p[2];

  double ratime, retime;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [make_eo_phase_field] using phase momentum = (%d, %d, %d)\n", momentum[0], momentum[1], momentum[2]);
  }

  ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) shared(phase_e, phase_o)
{
#endif
  unsigned int ix, iix;
  int x0, x1, x2, x3;
  double dtmp;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  /* make phase field in eo ordering */
  for(x0 = 0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix  = g_ipt[x0][x1][x2][x3];
      iix = g_lexic2eosub[ix];
      dtmp = phase_part + x1 * p[0] + x2 * p[1] + x3 * p[2];
      if(g_iseven[ix]) {
        phase_e[2*iix  ] = cos(dtmp);
        phase_e[2*iix+1] = sin(dtmp);
      } else {
        phase_o[2*iix  ] = cos(dtmp);
        phase_o[2*iix+1] = sin(dtmp);
      }
    }}}
  }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif


  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [make_eo_phase_field] time for making eo phase field = %e seconds\n", retime-ratime);
}  /* end of make_eo_phase_field */


/***********************************************
 * phase field on odd sublattice in sliced 3d
 * ordering (which I think is the same as odd
 * ordering)
 ***********************************************/
void make_o_phase_field_sliced3d (double _Complex**phase, int *momentum) {

  const double TWO_MPI = 2. * M_PI;
  const double p[3] = { TWO_MPI * momentum[0] / (double)LX_global, TWO_MPI * momentum[1] / (double)LY_global, TWO_MPI * momentum[2] / (double)LZ_global };
  const double phase_part = (g_proc_coords[1]*LX) * p[0] + (g_proc_coords[2]*LY) * p[1] + (g_proc_coords[3]*LZ) * p[2];

  double ratime, retime;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [make_o_phase_field_sliced3d] using phase momentum = (%d, %d, %d)\n", momentum[0], momentum[1], momentum[2]);
  }

  ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) shared(phase, momentum)
{
#endif

  unsigned int ix, iix;
  int x0, x1, x2, x3;
  double _Complex dtmp;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  /* make phase field in o ordering */
  for(x0 = 0; x0<T; x0 ++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix  = g_ipt[x0][x1][x2][x3];
      iix = g_eosub2sliced3d[1][g_lexic2eosub[ix] ];
      dtmp = ( phase_part + x1*p[0] + x2*p[1] + x3*p[2] ) * I; 
      if(!g_iseven[ix]) {
        phase[x0][iix] = cexp(dtmp);
      }
    }}}
  }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [make_o_phase_field_sliced3d] time for making eo phase field = %e seconds\n", retime-ratime);
}  /* end of make_o_phase_field_sliced3d */

/***********************************************
 * phase field on even/odd sublattice in sliced 3d
 * ordering (which I think is the same as odd
 * ordering)
 * eo - even 0 / odd 1
 ***********************************************/
void make_eo_phase_field_sliced3d (double _Complex**phase, int *momentum, int eo) {

  const double TWO_MPI = 2. * M_PI;
  const int eo_iseven = (int)(eo == 0);
  const double p[3] = { TWO_MPI * momentum[0] / (double)LX_global, TWO_MPI * momentum[1] / (double)LY_global, TWO_MPI * momentum[2] / (double)LZ_global };
  const double phase_part = (g_proc_coords[1]*LX) * p[0] + (g_proc_coords[2]*LY) * p[1] + (g_proc_coords[3]*LZ) * p[2];

  double ratime, retime;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [make_o_phase_field_sliced3d] using phase momentum = (%d, %d, %d)\n", momentum[0], momentum[1], momentum[2]);
  }

  ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) shared(phase, momentum)
{
#endif

  unsigned int ix, iix;
  int x0, x1, x2, x3;
  double _Complex dtmp;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  /* make phase field in o ordering */
  for(x0 = 0; x0<T; x0 ++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix  = g_ipt[x0][x1][x2][x3];
      iix = g_eosub2sliced3d[eo][g_lexic2eosub[ix] ];
      dtmp = ( phase_part + x1*p[0] + x2*p[1] + x3*p[2] ) * I; 
      if(g_iseven[ix] == eo_iseven) {
        phase[x0][iix] = cexp(dtmp);
      }
    }}}
  }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [make_o_phase_field_sliced3d] time for making eo phase field = %e seconds\n", retime-ratime);
}  /* end of make_o_phase_field_sliced3d */

/****************************************************************
 * phase field on a timeslice of even/odd sublattice
 * input:
 *   momentum_number - number of momenta in momentum_list
 *   momentum_list - list of integer 3-momentum vectors
 *   timeslice - timeslice for which phase field is constructed
 *   eo - even 0 / odd 
 * output:
 *   phase 2-dim array of phases momentum_number x VOL3/2
 ****************************************************************/
void make_eo_phase_field_timeslice (double _Complex**phase, int momentum_number, int (*momentum_list)[3], int timeslice, int eo) {

  const double TWO_MPI = 2. * M_PI;
  const unsigned int VOL3half = LX*LY*LZ/2;

  int imom;
  double ratime, retime;

  ratime = _GET_TIME;
  for( imom = 0; imom < momentum_number; imom++ ) {
    const double p[3] = { TWO_MPI * momentum_list[imom][0] / (double)LX_global, TWO_MPI * momentum_list[imom][1] / (double)LY_global, TWO_MPI * momentum_list[imom][2] / (double)LZ_global };
    const double phase_part = (g_proc_coords[1]*LX) * p[0] + (g_proc_coords[2]*LY) * p[1] + (g_proc_coords[3]*LZ) * p[2];

    if(g_verbose > 2 && g_cart_id == 0) {
      fprintf(stdout, "# [make_eo_phase_field_timeslice] using phase momentum = (%d, %d, %d)\n", momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2] );
    }

#ifdef HAVE_OPENMP
#pragma omp parallel shared(phase,timeslice,eo)
{
#endif
    unsigned int ix;
    double _Complex ztmp;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    /* make phase field in eo ordering */
    for(ix=0; ix < VOL3half; ix++) {
      /* ztmp = ( phase_part + g_eot2xyz[eo][timeslice][ix][0] * p[0] + g_eot2xyz[eo][timeslice][ix][1] * p[1] + g_eot2xyz[eo][timeslice][ix][2] * p[2] ) * I; */
      ztmp = ( phase_part +   g_eosubt2coords[eo][timeslice][ix][0] * p[0] +   g_eosubt2coords[eo][timeslice][ix][1] * p[1] +   g_eosubt2coords[eo][timeslice][ix][2] * p[2] ) * I;
      phase[imom][ix] = cexp(ztmp);
    }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  }  /* end of loop on momenta */
  retime = _GET_TIME;
  if(g_verbose > 1 && g_cart_id == 0) fprintf(stdout, "# [make_eo_phase_field_timeslice] time for make_eo_phase_field_timeslice = %e seconds\n", retime-ratime);

}  /* end of make_eo_phase_field_timeslice */


/**************************************************************************************************************/
/**************************************************************************************************************/

/**************************************************************************************************************
 * V     is nv              x VOL3half (C) = VOL3half x nv (F)
 * phase is momentum_number x VOL3half (C) = VOL3half x momentum_number (F)
 *
 * zgemm calculates t(V) x phase, which is  nv x momentum_number (F) = momentum_number x nv (C)
 **************************************************************************************************************/
int momentum_projection_eo_timeslice (
    double * const V, 
    double * const W, 
    unsigned int const nv, 
    int const momentum_number, 
    const int (*momentum_list)[3], 
    int const t, 
    int const ieo, 
    double const momentum_shift[3], 
    int const add, 
    int const ts_reduce
) {

  const double MPI2 = M_PI * 2.;
  const unsigned int VOL3half = LX*LY*LZ/2;

  double _Complex **zphase = NULL;
  double ratime, retime;
  double shift[3] = {0.,0.,0.}; 
  char BLAS_TRANSA, BLAS_TRANSB;
  int BLAS_M, BLAS_K, BLAS_N, BLAS_LDA, BLAS_LDB, BLAS_LDC;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = add == 0 ? 0. : 1.;

  // if ( g_cart_id == 0 ) fprintf ( stdout, "# [momentum_projection_eo_timeslice] alpha = %25.16e %25.16e beta %25.16e %25.16e add %d\n",
  //     creal( BLAS_ALPHA ), cimag( BLAS_ALPHA ), creal( BLAS_BETA ), cimag( BLAS_BETA ), add );

  ratime = _GET_TIME;

  if ( init_2level_zbuffer( &zphase, momentum_number, VOL3half ) != 0 ) {
    fprintf(stderr, "[momentum_projection_eo_timeslice] Error from init_2level_zbuffer %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  if ( momentum_shift != NULL ) {
    shift[0] = momentum_shift[0];
    shift[1] = momentum_shift[1];
    shift[2] = momentum_shift[2];
  }

  /* loop on sink momenta */
  for( int i = 0; i < momentum_number; i++) {
    /* phase field */
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    const double q[3] = { MPI2 * momentum_list[i][0] / LX_global,
                          MPI2 * momentum_list[i][1] / LY_global,
                          MPI2 * momentum_list[i][2] / LZ_global };
    const double q_offset = g_proc_coords[1]*LX * q[0] + g_proc_coords[2]*LY * q[1] + g_proc_coords[3]*LZ * q[2] \
                            + ( shift[0] * q[0] + shift[1] * q[1] + shift[2] * q[2] );
    double q_phase;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for( unsigned int ix = 0; ix < VOL3half; ix++ ) {
      q_phase = q_offset \
                + g_eosubt2coords[ieo][t][ix][0] * q[0] \
                + g_eosubt2coords[ieo][t][ix][1] * q[1] \
                + g_eosubt2coords[ieo][t][ix][2] * q[2];

      // zphase[i][ix] = cos(q_phase) + I*sin(q_phase);
      zphase[i][ix] = cexp ( I * q_phase);
    }  /* end of loop on VOL3half */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }  /* end of loop on sink momenta */

  BLAS_TRANSA = 'T';
  BLAS_TRANSB = 'N';
  BLAS_M     = nv;
  BLAS_K     = VOL3half;
  BLAS_N     = momentum_number;
  BLAS_A     = (double _Complex*)V;
  BLAS_B     = zphase[0];
  BLAS_C     = (double _Complex*)W;
  BLAS_LDA   = BLAS_K;
  BLAS_LDB   = BLAS_K;
  BLAS_LDC   = BLAS_M;

  F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  fini_2level_zbuffer( &zphase );

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
  if ( ts_reduce ) {
    unsigned int items = 2 * nv * momentum_number;
    void *buffer = malloc( items * sizeof(double));
    if( buffer == NULL) {
      fprintf(stderr, "[momentum_projection_eo_timeslice] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(1);
    }
    int status = MPI_Allreduce ( W, buffer, items, MPI_DOUBLE, MPI_SUM, g_ts_comm );
    if(status != MPI_SUCCESS) {
      fprintf(stderr, "[momentum_projection_eo_timeslice] Error from MPI_Allreduce, status was %d %s %d\n", status, __FILE__, __LINE__);
      return(2);
    }
    memcpy( W, buffer, items * sizeof(double));
    free ( buffer );
  }  // of if ts_reduce
#  endif
#endif

  retime = _GET_TIME;
  if ( g_cart_id == 0 && g_verbose > 0 ) {
    fprintf(stdout, "# [momentum_projection_eo_timeslice] time for momentum_projection_eo_timeslice = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }
  return(0);
}  // end of momentum_projection_eo_timeslice


/******************************************************************************************************
 * projection coefficents, per timeslice
 *
 * B = s is num1 x [12N] (C) = [12N] x num1 (F)
 * A = V is num2 x [12N] (C) = [12N] x num2 (F)
 *
 * zgemm calculates p = V^H x r, which is num2  x num1 (F) = num1 x num2  (C)
 *
 * - expects fields of space-time size nt x N
 *   e.g. N = VOL3half, nt = T
 ******************************************************************************************************/
int project_reduce_from_propagator_field_per_timeslice (double *p, double * r, double *V, int num1, int num2, int nt, unsigned int N) {

  const unsigned int offset_field           = nt * _GSI( N );
  const unsigned int offset_field_timeslice =      _GSI( N );
  const size_t sizeof_field                 = offset_field * sizeof(double);
  const unsigned int offset_p               = (unsigned int)num1 * (unsigned int)num2 * 2;
  const int items_p                         = nt * num1 * num2;  /* number of double _Complex items */
  const size_t bytes_p                      = (size_t)items_p * sizeof(double _Complex);

  int exitstatus;
  int BLAS_M, BLAS_N, BLAS_K; 
  int BLAS_LDA, BLAS_LDB ,BLAS_LDC;
  char BLAS_TRANSA, BLAS_TRANSB;
  double _Complex *p_buffer = NULL;
  double _Complex BLAS_ALPHA, BLAS_BETA;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double *V_aux = NULL, *r_aux = NULL;
  double ratime, retime;
 
  if (p == NULL || r == NULL || V == NULL || num1 <= 0 || num2 <= 0) {
    fprintf(stderr, "[project_reduce_from_propagator_field_per_timeslice] Error, wrong parameter values\n");
    return(4);
  }
 
  ratime = _GET_TIME;

  alloc_spinor_field ( &V_aux, num2 * N );

  alloc_spinor_field ( &r_aux, num1 * N );

  /* projection on V-basis */
  BLAS_ALPHA  = 1.;
  BLAS_BETA   = 0.;
  BLAS_TRANSA = 'C';
  BLAS_TRANSB = 'N';
  BLAS_M      = num2;
  BLAS_K      = 12*N;
  BLAS_N      = num1;
  BLAS_LDA    = BLAS_K;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;
  BLAS_A      = (double _Complex*)( V_aux );
  BLAS_B      = (double _Complex*)( r_aux );


  for ( int it = 0; it < nt; it++ ) {
    unsigned int offset_time = it * offset_field_timeslice;

    for ( int k = 0; k < num1; k++ ) {
      unsigned int offset_total = k * offset_field + offset_time;
      memcpy( r_aux + k*offset_field_timeslice, r + offset_total, sizeof_field );
    }

    for ( int k = 0; k < num2; k++ ) {
      unsigned int offset_total = k * offset_field + offset_time;
      memcpy( V_aux + k*offset_field_timeslice, V + offset_total, sizeof_field );
    }

    BLAS_C = (double _Complex*)( p + it * offset_p );

    F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  }  /* of loop on it */

  free ( V_aux );
  free ( r_aux );

#ifdef HAVE_MPI
    /* allreduce across all processes */
    if( (p_buffer = (double _Complex*)malloc( bytes_p )) == NULL ) {
      fprintf(stderr, "[project_reduce_from_propagator_field_per_timeslice] Error from malloc\n");
      return(2);
    }

    memcpy(p_buffer, p, bytes_p);
    exitstatus = MPI_Allreduce(p_buffer, p, 2*items_p, MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[project_reduce_from_propagator_field_per_timeslice] Error from MPI_Allreduce, exitstatus was %d\n", exitstatus);
      return(1);
    }
    free(p_buffer); p_buffer = NULL;
#endif

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [project_reduce_from_propagator_field] time for projection = %e seconds\n", retime-ratime);

  return(0);
}  /* end of project_reduce_from_propagator_field_per_timeslice */


}  /* end of namespace cvc */
