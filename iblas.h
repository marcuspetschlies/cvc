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

#endif
