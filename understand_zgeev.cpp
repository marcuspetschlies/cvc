/****************************************************
 * understand_zgeev.cpp 
 *
 * Mo 29. Jan 17:39:36 CET 2018
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

/**********************************************************************************/
/**********************************************************************************/

void print_Rmatrix (double _Complex *a, int m, int n, char *name) {
  int i, k;
  fprintf(stdout, "%s <- array(dim=c(%d,%d))\n", name, m, n);
  for(i=0; i<m; i++) {
    for(k=0; k<n; k++) {
    fprintf(stdout, "%s[%d,%d] <- %25.16e + %25.16e*1.i\n", name, i+1, k+1, creal(a[i*n+k]), cimag(a[i*n+k]));
    }
  }
  fprintf(stdout, "\n\n");
}

/**********************************************************************************/
/**********************************************************************************/

void make_random_matrix (double _Complex **a, int m, int n) {
  if(*a != NULL) free(*a);
  *a = (double _Complex*)malloc(n*m*sizeof(double _Complex));
  rangauss ((double*)(*a), 2*n*m);
}

/**********************************************************************************/
/**********************************************************************************/

void make_zero_matrix (double _Complex **a, int m, int n) {
  if(*a != NULL) free(*a);
  *a = (double _Complex*)calloc(n*m, sizeof(double _Complex));
}

/**********************************************************************************/
/**********************************************************************************/


extern "C" void _F( zgeev ) ( char *JOBVL, char *JOBVR, int *N, double _Complex *A, int *LDA, double _Complex *W, double _Complex *VL, int *LDVL, double _Complex *VR, int *LDVR, double _Complex *WORK,
                              int*LWORK, double *RWORK, int *INFO, int len_JOBVL, int len_JOBVR );

extern "C" void _F(zgemv) ( char*TRANS, int *M, int *N, double _Complex *ALPHA, double _Complex *A, int *LDA, double _Complex *X,
          int *INCX, double _Complex *BETA, double _Complex * Y, int *INCY, int len_TRANS);



/**********************************************************************************/
/**********************************************************************************/

int main(int argc, char **argv) {
  
  int c;
  int dim = 11;
  // int INT_0 = 0;
  int INT_1 = 1;

  double _Complex *A = NULL, *A2 = NULL;
  double _Complex *work = NULL;
  double _Complex *evecs = NULL, *evals = NULL, *levecs = NULL;
  double _Complex Z_1 = 1., Z_0 = 0.;
  double _Complex *X = NULL, *Y = NULL;
  int lwork = 3*dim+1;
  double *rwork = NULL;
  int info;

  char CHAR_N = 'N', CHAR_V = 'V', CHAR_T = 'T', CHAR_C = 'C';

  while ((c = getopt(argc, argv, "h?")) != -1) {
    switch (c) {
    case 'h':
    case '?':
    default:
      exit(0);
      break;
    }
  }

  make_random_matrix (&A, dim, dim);

  print_Rmatrix (A, dim, dim, "A");

  make_zero_matrix ( &A2, dim, dim );
    
  memcpy ( A2, A, dim*dim*sizeof(double _Complex ) );

  evecs = (double _Complex *)malloc ( dim * dim * sizeof(double _Complex) );

  levecs = (double _Complex *)malloc ( dim * dim * sizeof(double _Complex) );

  evals = (double _Complex *)malloc ( dim * sizeof(double _Complex) );

  work = (double _Complex *)malloc ( lwork * sizeof(double _Complex) );

  rwork = (double *)malloc ( 2 * dim * sizeof(double) );


  _F( zgeev ) ( &CHAR_V, &CHAR_V, &dim, A, &dim, evals, levecs, &dim, evecs, &dim, work, &lwork, rwork, &info, 1, 1 );


  fprintf ( stdout, "# [understand_zgeev] info = %d\n", info );
  if ( info == 0 ) {
    fprintf ( stdout, "# [understand_zgeev] optimal lwork = %d \n", (int)(creal( work[0])) );
  }
  for ( int i = 0; i < dim; i++  ) {
    fprintf ( stdout, "# [understand_zgeev] eval %2d = %25.16e %25.16e\n", i, creal(evals[i]), cimag(evals[i]) );
  }

  X = (double _Complex*) malloc ( dim * sizeof (double _Complex ) );
  Y = (double _Complex*) malloc ( dim * sizeof (double _Complex ) );

  fprintf ( stdout, "  X <- list()\n" );
  fprintf ( stdout, "  Y <- list()\n" );

#if 0
  for ( int i = 0; i < dim; i++ ) {


    for ( int k = 0; k < dim; k++ ) {
      X[k] = evecs[i * dim + k];
      // X[k] = evecs[k * dim + i];
    }

    fprintf ( stdout, "  X[[%2d]] <- numeric()\n", i+1 );
    for ( int k = 0; k < dim; k++ ) {
      fprintf ( stdout, "  X[[%2d]][%2d] = %25.16e + %25.16e*1.i\n", i+1, k+1, creal(X[k]), cimag(X[k]) );
    }

    memset ( Y, 0, dim * sizeof(double _Complex ) );
    
    _F( zgemv) ( &CHAR_N, &dim, &dim, &Z_1, A2, &dim, X, &INT_1, &Z_0, Y, &INT_1, 1 );

    fprintf ( stdout, "  Y[[%2d]] <- numeric()\n", i+1 );
    for ( int k = 0; k < dim; k++ ) {
      fprintf ( stdout, "  Y[[%2d]][%2d] = %25.16e + %25.16e*1.i\n", i+1, k+1, creal(Y[k]), cimag(Y[k]) );
    }

    double norm = 0.;
    for ( int k = 0; k < dim; k++ ) {
      double _Complex z = X[k] * evals[i] - Y[k];
      norm += creal ( z * conj ( z ) );
    }
    
    norm = sqrt( norm );
    fprintf ( stdout, "# [understand_zgeev] ev %2d norm diff %25.16e\n", i, norm );
  }
#endif

  for ( int i = 0; i < dim; i++ ) {

    for ( int k = 0; k < dim; k++ ) {
      X[k] = levecs[i * dim + k];
    }

    fprintf ( stdout, "  X[[%2d]] <- numeric()\n", i+1 );
    for ( int k = 0; k < dim; k++ ) {
      fprintf ( stdout, "  X[[%2d]][%2d] = %25.16e + %25.16e*1.i\n", i+1, k+1, creal(X[k]), cimag(X[k]) );
    }

    memset ( Y, 0, dim * sizeof(double _Complex ) );
    
    _F( zgemv) ( &CHAR_C, &dim, &dim, &Z_1, A2, &dim, X, &INT_1, &Z_0, Y, &INT_1, 1 );

    fprintf ( stdout, "  Y[[%2d]] <- numeric()\n", i+1 );
    for ( int k = 0; k < dim; k++ ) {
      fprintf ( stdout, "  Y[[%2d]][%2d] = %25.16e + %25.16e*1.i\n", i+1, k+1, creal(Y[k]), cimag(Y[k]) );
    }

    double norm = 0.;
    for ( int k = 0; k < dim; k++ ) {
      double _Complex z = X[k] * conj( evals[i] ) - Y[k];
      norm += creal ( z * conj ( z ) );
    }
    
    norm = sqrt( norm );
    fprintf ( stdout, "# [understand_zgeev] lev %2d norm diff %25.16e\n", i, norm );
  }

  free ( A );
  free ( A2 );
  free ( evecs );
  free ( levecs );
  free ( evals );
  free ( work );
  free ( rwork );
  free ( X );
  free ( Y );

  return(0);
}
