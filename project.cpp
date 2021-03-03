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
#include "matrix_init.h"
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

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

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

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

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
  double ratime, retime;

  char BLAS_TRANSA, BLAS_TRANSB;
  int BLAS_M, BLAS_K, BLAS_N, BLAS_LDA, BLAS_LDB, BLAS_LDC;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  ratime = _GET_TIME;

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
      q_phase = q_offset \
        + lexic_coords[ix].x[0] * q[0] \
        + lexic_coords[ix].x[1] * q[1] \
        + lexic_coords[ix].x[2] * q[2];
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

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  fini_2level_buffer((double***)(&zphase));

#ifdef HAVE_MPI
  i = 2 * nv * momentum_number;
  void *buffer = malloc(i * sizeof(double));
  if(buffer == NULL) {
    return(1);
  }
  memcpy(buffer, W, i*sizeof(double));
  int status = MPI_Allreduce(buffer, (void*)W, i, MPI_DOUBLE, MPI_SUM, g_ts_comm);
  if(status != MPI_SUCCESS) {
    fprintf(stderr, "[momentum_projection] Error from MPI_Allreduce, status was %d\n", status);
    return(2);
  }
  free(buffer);
#endif

  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [momentum_projection] time for momentum_projection = %e seconds\n", retime-ratime);
  return(0);
}  /* end of momentum_projection */

/**************************************************************************************************************
 * V     is VOL3            x nv   (C) = nv   x VOL3            (F)
 * phase is momentum_number x VOL3 (C) = VOL3 x momentum_number (F)
 *
 * zgemm calculates V x phase, which is  nv x momentum_number (F) = momentum_number x nv (C)
 **************************************************************************************************************/
int momentum_projection2 (double*V, double *W, unsigned int nv, int momentum_number, const int (* const momentum_list)[3], int gshift[3]) {

  typedef struct {
    int x[3];
  } point;

  const double MPI2 = M_PI * 2.;
  const unsigned int VOL3 = LX*LY*LZ;

  int x1, x2, x3;
  unsigned int i, ix;
  double _Complex **zphase = NULL;
  double ratime, retime;

  char BLAS_TRANSA, BLAS_TRANSB;
  int BLAS_M, BLAS_K, BLAS_N, BLAS_LDA, BLAS_LDB, BLAS_LDC;
  int shift[3];
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  ratime = _GET_TIME;

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

  if(gshift == NULL) {
    memset( shift, 0, 3*sizeof(int) );
  } else {
    memcpy( shift, gshift, 3*sizeof(int) );
  }
  if(g_cart_id == 0 && g_verbose > 4 ) fprintf(stdout, "# [momentum_projection2] using shift vector (%d, %d, %d)\n", shift[0], shift[1], shift[2]);

  init_2level_buffer( (double***)(&zphase), momentum_number, 2*VOL3 );

  for(i=0; i < momentum_number; i++) {
    /* phase field */
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    const double  q[3] = { MPI2 * momentum_list[i][0] / LX_global,
                           MPI2 * momentum_list[i][1] / LY_global,
                           MPI2 * momentum_list[i][2] / LZ_global };
    const double q_offset = ( g_proc_coords[1]*LX - shift[0] ) * q[0] + ( g_proc_coords[2]*LY - shift[1] ) * q[1] + ( g_proc_coords[3]*LZ - shift[2] ) * q[2];
    double q_phase;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix=0; ix < VOL3; ix++) {
      q_phase = q_offset \
         + lexic_coords[ix].x[0]*q[0] \
         + lexic_coords[ix].x[1]*q[1] \
         + lexic_coords[ix].x[2]*q[2];

      zphase[i][ix] = cos(q_phase) + I*sin(q_phase);
    }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }  /* end of loop on sink momenta */

  free( lexic_coords );

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

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  fini_2level_buffer((double***)(&zphase));

#ifdef HAVE_MPI
  i = 2 * nv * momentum_number;
  void *buffer = malloc(i * sizeof(double));
  if(buffer == NULL) {
    return(1);
  }
  memcpy(buffer, W, i*sizeof(double));
  int status = MPI_Allreduce(buffer, (void*)W, i, MPI_DOUBLE, MPI_SUM, g_ts_comm);
  if(status != MPI_SUCCESS) {
    fprintf(stderr, "[momentum_projection] Error from MPI_Allreduce, status was %d\n", status);
    return(2);
  }
  free(buffer);
#endif

  retime = _GET_TIME;
  if( g_cart_id == 0 && g_verbose > 2) fprintf(stdout, "# [momentum_projection2] time for momentum_projection2 = %e seconds\n", retime-ratime);
  return(0);
}  /* end of momentum_projection2 */


/**************************************************************************************************************
 * momentum_projection3 
 *
 * V     is nv              x VOL3 (C) = VOL3 x nv (F)
 * phase is momentum_number x VOL3 (C) = VOL3 x momentum_number (F)
 *
 * zgemm calculates t(phase) x V, which is  momentum_number x nv (F) = nv x momentum_number (C)
 *
 * This version is like momentum_projection, except, that the final result here is nv x momentum_number and
 * not momentum_number x nv as in momentum_projection. This is advantageous if nv = T and we need to do
 * an mpi gather operation afterwards.
 **************************************************************************************************************/
int momentum_projection3 (double*V, double *W, unsigned int nv, int momentum_number, int (*momentum_list)[3]) {

  typedef struct {
    int x[3];
  } point;

  const double MPI2 = M_PI * 2.;
  const unsigned int VOL3 = LX*LY*LZ;

  int x1, x2, x3;
  unsigned int i, ix;
  double _Complex **zphase = NULL;
  double ratime, retime;

  char BLAS_TRANSA, BLAS_TRANSB;
  int BLAS_M, BLAS_K, BLAS_N, BLAS_LDA, BLAS_LDB, BLAS_LDC;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  ratime = _GET_TIME;

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
      q_phase = q_offset \
        + lexic_coords[ix].x[0] * q[0] \
        + lexic_coords[ix].x[1] * q[1] \
        + lexic_coords[ix].x[2] * q[2];
      zphase[i][ix] = cos(q_phase) + I*sin(q_phase);
    }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }  /* end of loop on sink momenta */

  free( lexic_coords );

  BLAS_TRANSA = 'T';
  BLAS_TRANSB = 'N';
  BLAS_M     = momentum_number;
  BLAS_K     = VOL3;
  BLAS_N     = nv;
  BLAS_A     = zphase[0];
  BLAS_B     = (double _Complex*)V;
  BLAS_C     = (double _Complex*)W;
  BLAS_LDA   = BLAS_K;
  BLAS_LDB   = BLAS_K;
  BLAS_LDC   = BLAS_M;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  fini_2level_buffer((double***)(&zphase));

#ifdef HAVE_MPI
  i = 2 * nv * momentum_number;
  void *buffer = malloc(i * sizeof(double));
  if(buffer == NULL) {
    return(1);
  }
  memcpy(buffer, W, i*sizeof(double));
  int status = MPI_Allreduce(buffer, (void*)W, i, MPI_DOUBLE, MPI_SUM, g_ts_comm);
  if(status != MPI_SUCCESS) {
    fprintf(stderr, "[momentum_projection] Error from MPI_Allreduce, status was %d\n", status);
    return(2);
  }
  free(buffer);
#endif

  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [momentum_projection3] time for momentum_projection3 = %e seconds\n", retime-ratime);
  return(0);
}  /* end of momentum_projection3 */



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
      iix = g_eosub2sliced3d[1][g_lexic2eosub[ix] ];
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

/*******************************************************************/
/*******************************************************************/

/****************************************************************
 * 3-dim phase field
 * input:
 *   momentum_number - number of momenta in momentum_list
 *   momentum_list - list of integer 3-momentum vectors
 * output:
 *   phase 2-dim array of phases momentum_number x VOL3
 ****************************************************************/
void make_phase_field_timeslice (double _Complex ** const phase, int const momentum_number, int (* const momentum_list)[3] ) {

  double const TWO_MPI = 2. * M_PI;
  unsigned int const VOL3 = LX*LY*LZ;
  double _Complex const IMAGU = 1.0 * I;


  for( int imom = 0; imom < momentum_number; imom++ ) {
    /* 3-vector p for current momentum */
    double const p[3] = { TWO_MPI * momentum_list[imom][0] / (double)LX_global,
                          TWO_MPI * momentum_list[imom][1] / (double)LY_global,
                          TWO_MPI * momentum_list[imom][2] / (double)LZ_global };

    /* phase part due to MPI cart grid offset */
    double const phase_part = (g_proc_coords[1]*LX) * p[0] + (g_proc_coords[2]*LY) * p[1] + (g_proc_coords[3]*LZ) * p[2];

    if(g_verbose > 2 && g_cart_id == 0) {
      fprintf(stdout, "# [make_phase_field_timeslice] using phase momentum = (%d, %d, %d)\n", momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2] );
    }

    /* make phase field in lexic ordering */
#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for( unsigned int ix = 0; ix < VOL3; ix++) {
      unsigned int const x1 =   ix                            / ( LY * LZ );
      unsigned int const x2 = ( ix - x1 * LY * LZ           ) /        LZ;
      unsigned int const x3 = ( ix - x1 * LY * LZ - x2 * LZ );
      double const dtmp = phase_part + x1 * p[0] + x2 * p[1] + x3 * p[2];
      /* exp( i phase ) */
      phase[imom][ix] = cexp( dtmp * IMAGU );
    }

  }  /* end of loop on momenta */

}  /* end of make_phase_field_timeslice */

}  /* end of namespace cvc */
