/****************************************************
 * understand_zgemm.cpp 
 *
 * Do 30. Jun 14:37:17 CEST 2016
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <complex.h>


#define MAIN_PROGRAM

#include "ranlxd.h"


#ifdef F_
#define _F(s) s##_
#else
#define _F(s) s
#endif

using namespace cvc;

int rangauss (double * y1, unsigned int NRAND) {

  const double TWO_MPI = 2. * M_PI;
  const unsigned int nrandh = NRAND/2;
  unsigned int k, k2, k2p1;
  double x1;

  if(NRAND%2 != 0) {
    fprintf(stderr, "Error, NRAND must be an even number\n");
    return(1);
  }

  /* fill the complete field y1 */
  ranlxd(y1,NRAND);

  for(k=0; k<nrandh; k++) {
    k2   = 2*k;
    k2p1 = k2+1;

    x1       = sqrt( -2. * log(y1[k2]) );
    y1[k2]   = x1 * cos( TWO_MPI * y1[k2p1] );
    y1[k2p1] = x1 * sin( TWO_MPI * y1[k2p1] );
  }  /* end of loop on nrandh */
  return(0);
}


void print_Rmatrix (double _Complex *a, int m, int n, char *name) {
  int i, k;
  fprintf(stdout, "%s <- array(dim=c(%d,%d))\n", name, m, n);
  for(i=0; i<m; i++) {
    for(k=0; k<n; k++) {
    fprintf(stdout, "%s[%d.%d] <- %25.16e + %25.16e*1.i\n", name, i+1, k+1, creal(a[i*n+k]), cimag(a[i*n+k]));
    }
  }
  fprintf(stdout, "\n\n");
}

void make_random_matrix (double _Complex **a, int m, int n) {
  if(*a != NULL) free(*a);
  *a = (double _Complex*)malloc(n*m*sizeof(double _Complex));
  rangauss ((double*)(*a), 2*n*m);
}
void make_zero_matrix (double _Complex **a, int m, int n) {
  if(*a != NULL) free(*a);
  *a = (double _Complex*)calloc(n*m, sizeof(double _Complex));
}

extern "C" void _F(zgemm) ( char*TRANSA, char*TRANSB, int *M, int *N, int *K, double _Complex *ALPHA, double _Complex *A, int *LDA, double _Complex *B,
        int *LDB, double _Complex *BETA, double _Complex * C, int *LDC, int len_TRANSA, int len_TRANSB);

extern "C" void _F(zgemv) ( char*TRANS, int *M, int *N, double _Complex *ALPHA, double _Complex *A, int *LDA, double _Complex *X,
      int *INCX, double _Complex *BETA, double _Complex * Y, int *INCY, int len_TRANS);

int main(int argc, char **argv) {
  
  int c;
  int i, k, l, n;
  int verbose = 0;
  int dim1 = 3, dim2 = 7, dim3 = 11;

  double _Complex *A = NULL;
  double _Complex *B = NULL;
  double _Complex *C = NULL;
/*
  double _Complex TA[DIMM * DIMK];
  double _Complex TB[DIMK * DIMN];
  double _Complex TC[DIMM * DIMN];

  double _Complex X[DIMM];
  double _Complex Y[DIMK];
*/
  char BLAS_TRANSA, BLAS_TRANSB;
/*  char BLAS_TRANS; */
  int BLAS_M,  BLAS_N, BLAS_K;
  double _Complex BLAS_ALPHA;
  double _Complex BLAS_BETA;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
/*  double _Complex *BLAS_X=NULL, *BLAS_Y=NULL; */
  int BLAS_LDA;
  int BLAS_LDB;
  int BLAS_LDC;
/*  int BLAS_INCX, BLAS_INCY; */


  while ((c = getopt(argc, argv, "h?v:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'h':
    case '?':
    default:
      exit(0);
      break;
    }
  }

  make_random_matrix (&A, dim1, dim2); /* fortran; dim2 x dim1 */
  make_random_matrix (&B, dim1, dim3); /* fortran: dim3 x dim1 */

  print_Rmatrix (A, dim1, dim2, "A");
  print_Rmatrix (B, dim1, dim3, "B");

  make_zero_matrix (&C, dim2, dim3);

  BLAS_TRANSA = 'N';
  BLAS_TRANSB = 'T';
  BLAS_M     = dim2;
  BLAS_K     = dim1;
  BLAS_N     = dim3;
  BLAS_ALPHA = 1.;
  BLAS_BETA  = 0.;
  BLAS_A     = A;
  BLAS_B     = B;
  BLAS_LDA   = BLAS_M;
  BLAS_LDB   = BLAS_K;
  BLAS_LDC   = BLAS_M;

  /* _F(zgemv) ( &BLAS_TRANS, &BLAS_M, &BLAS_N, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_X, &BLAS_INCX, &BLAS_BETA, BLAS_Y, &BLAS_INCY, 1); */

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC, 1, 1);

  print_Rmatrix (C, dim1, dim3, "C");

  return(0);
}
