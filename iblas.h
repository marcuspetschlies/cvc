#ifndef _IBLAS_H
#define _IBLAS_H
/***************************************************
 * iblas.h
 ***************************************************/
#include <complex.h>

#ifdef F_
#define _F(s) s##_
#else
#define _F(s) s
#endif

/***********************************************************************
          subroutine zgemm  (   character   TRANSA,
            character   TRANSB,
            integer   M,
            integer   N,
            integer   K,
            complex*16    ALPHA,
            complex*16, dimension(lda,*)    A,
            integer   LDA,
            complex*16, dimension(ldb,*)    B,
            integer   LDB,
            complex*16    BETA,
            complex*16, dimension(ldc,*)    C,
            integer   LDC 
          )
          
 ***********************************************************************/
extern "C" void _F(zgemm) ( char*TRANSA, char*TRANSB, int *M, int *N, int *K, double _Complex *ALPHA, double _Complex *A, int *LDA, double _Complex *B,
    int *LDB, double _Complex *BETA, double _Complex * C, int *LDC, int len_TRANSA, int len_TRANSB);



/***********************************************************************
   subroutine zgemv   (   character   TRANS,
                          integer   M,
                          integer   N,
                          complex*16    ALPHA,
                          complex*16, dimension(lda,*)    A,
                          integer   LDA,
                          complex*16, dimension(*)    X,
                          integer   INCX,
                          complex*16    BETA,
                          complex*16, dimension(*)    Y,
                          integer   INCY 
                      )   
 ***********************************************************************/
extern "C" void _F(zgemv) ( char*TRANS, int *M, int *N, double _Complex *ALPHA, double _Complex *A, int *LDA, double _Complex *X,
      int *INCX, double _Complex *BETA, double _Complex * Y, int *INCY, int len_TRANS);



/***********************************************************************
  complex*16 function zdotc   (   integer   N,
                                  complex*16, dimension(*)    ZX,
                                  integer   INCX,
                                  complex*16, dimension(*)    ZY,
                                  integer   INCY 
                              )     
 ***********************************************************************/
extern void _F(zdotc)(int* n, _Complex double x[], int* incx, _Complex double y[], int* incy);

/***********************************************************************
ZGEQRF computes a QR factorization of a complex M-by-N matrix A:

    A = Q * ( R ),
            ( 0 )

 where:

    Q is a M-by-M orthogonal matrix;
    R is an upper-triangular N-by-N matrix;
    0 is a (M-N)-by-N zero matrix, if M > N.
 ***********************************************************************/

extern "C" void _F(zgeqrf) (int* M, int*N, _Complex double *A, int*LDA, _Complex double *TAU, _Complex double *WORK, int*LWORK, int*INFO );

extern "C" void _F(zungqr) ( int * M, int * N, int * K, double _Complex * A, int * LDA, double _Complex * TAU, double _Complex * WORK, int * LWORK, int * INFO);

extern "C" void _F(zgeqr)( int * M, int * N, double _Complex * A, int * LDA, double _Complex * T, int * TSIZE, double _Complex * WORK, int * LWORK, int * INFO );

extern "C" void _F(zgemqr)( char * SIDE, char *  TRANS, int * M, int * N, int * K, double _Complex * A, int * LDA, double _Complex* T, int * TSIZE, double _Complex * C, int * LDC, double _Complex * WORK, int * LWORK, int * INFO, int len_SIDE, int len_TRANS ); 


#endif
