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
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif


#define DIMM 10
#define DIMK 20
#define DIMN 30


#define MAIN_PROGRAM

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "read_input_parser.h"
#include "ranlxd.h"


using namespace cvc;

#ifdef F_
#define _F(s) s##_
#else
#define _F(s) s
#endif


extern "C" void _F(zgemm) ( char*TRANSA, char*TRANSB, int *M, int *N, int *K, double _Complex *ALPHA, double _Complex *A, int *LDA, double _Complex *B,
        int *LDB, double _Complex *BETA, double _Complex * C, int *LDC, int len_TRANSA, int len_TRANSB);

extern "C" void _F(zgemv) ( char*TRANS, int *M, int *N, double _Complex *ALPHA, double _Complex *A, int *LDA, double _Complex *X,
      int *INCX, double _Complex *BETA, double _Complex * Y, int *INCY, int len_TRANS);

int main(int argc, char **argv) {
  
  int c, exitstatus;
  int i, ik, im, in;
  int filename_set = 0;
  int x0, x1;
  double plaq=0;
  double spinor1[24], spinor2[24];
  complex w, w2;
  int verbose = 0;
  char filename[200];
  unsigned int Vhalf, VOL3;
  double **eo_spinor_field = NULL;

  double _Complex A[DIMM * DIMK];
  double _Complex TA[DIMM * DIMK];
  double _Complex B[DIMK * DIMN];
  double _Complex TB[DIMK * DIMN];
  double _Complex C[DIMM * DIMN];
  double _Complex TC[DIMM * DIMN];

  double _Complex X[DIMM];
  double _Complex Y[DIMK];

  char BLAS_TRANSA, BLAS_TRANSB, BLAS_TRANS;
  int BLAS_M,  BLAS_N, BLAS_K;
  double _Complex BLAS_ALPHA;
  double _Complex BLAS_BETA;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL, *BLAS_X=NULL, *BLAS_Y=NULL;
  int BLAS_LDA;
  int BLAS_LDB;
  int BLAS_LDC;
  int BLAS_INCX, BLAS_INCY;



#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      exit(0);
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "apply_Dtm.input");
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);


  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);


  /* initialize T etc. */
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
		  "# [%2d] LXstart      = %3d\n"\
                  "# [%2d] LY_global    = %3d\n"\
                  "# [%2d] LY           = %3d\n"\
		  "# [%2d] LYstart      = %3d\n",\
		  g_cart_id, g_cart_id, T_global, g_cart_id, T, g_cart_id, Tstart,
		             g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart,
		             g_cart_id, LY_global, g_cart_id, LY, g_cart_id, LYstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;
  VOL3 = LX * LY * LZ;

  rangauss ((double*)A, 2*DIMM*DIMK);
  rangauss ((double*)B, 2*DIMK*DIMN);

  rangauss ((double*)X, 2*DIMK);

  for(im=0; im<DIMM; im++) {
    for(ik=0; ik<DIMK; ik++) {
      TA[ik*DIMM+im] = A[im*DIMK+ik];
    }
  }
#if 0
  for(ik=0; ik<DIMK; ik++) {
    for(in=0; in<DIMN; in++) {
      TB[in*DIMK+ik] = B[ik*DIMN+in];
    }
  }

  BLAS_ALPHA = 1.;
  BLAS_BETA  = 0.;
/*
  BLAS_M = DIMM;
  BLAS_N = DIMN;
  BLAS_K = DIMK;
  BLAS_TRANSA = 'N';
  BLAS_TRANSB = 'N';
  BLAS_A = TA;
  BLAS_B = TB;
  BLAS_C = TC;
  BLAS_LDA = DIMM;
  BLAS_LDB = DIMK;
  BLAS_LDC = DIMM;
*/

  BLAS_M = DIMM;
  BLAS_N = DIMN;
  BLAS_K = DIMK;
  BLAS_TRANSA = 'N';
  BLAS_TRANSB = 'N';
  BLAS_A = TA;
  BLAS_B = TB;
  BLAS_C = TC;
  BLAS_LDA = DIMM;
  BLAS_LDB = DIMK;
  BLAS_LDC = DIMM;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);
#endif
  fprintf(stdout, "A <- array(dim=c(%d,%d))\n", DIMM, DIMK);
  for(im=0; im<DIMM; im++) {
    for(ik=0; ik<DIMK; ik++) {
      fprintf(stdout, "A[%d,%d] <- %25.16e + %25.16e*1.i\n", im+1,ik+1, creal(A[im*DIMK+ik]), cimag(A[im*DIMK+ik]));
    }
  }
  fprintf(stdout, "TA <- array(dim=c(%d,%d))\n", DIMK, DIMM);
  for(ik=0; ik<DIMK; ik++) {
    for(im=0; im<DIMM; im++) {
      fprintf(stdout, "TA[%d,%d] <- %25.16e + %25.16e*1.i\n", ik+1,im+1, creal(TA[ik*DIMM+im]), cimag(TA[ik*DIMM+im]));
    }
  }
#if 0
  fprintf(stdout, "B <- array(dim=c(%d,%d))\n", DIMK, DIMN);
  for(ik=0; ik<DIMK; ik++) {
    for(in=0; in<DIMN; in++) {
      fprintf(stdout, "B[%d,%d] <- %25.16e + %25.16e*1.i\n", ik+1,in+1, creal(B[ik*DIMN+in]), cimag(B[ik*DIMN+in]));
    }
  } 
  fprintf(stdout, "TB <- array(dim=c(%d,%d))\n", DIMN, DIMK);
  for(in=0; in<DIMN; in++) {
    for(ik=0; ik<DIMK; ik++) {
      fprintf(stdout, "TB[%d,%d] <- %25.16e + %25.16e*1.i\n", in+1,ik+1, creal(TB[in*DIMK+ik]), cimag(TB[in*DIMK+ik]));
    }
  } 

  for(im=0; im<DIMM; im++) {
    for(in=0; in<DIMN; in++) {
      C[im*DIMN+in] = TC[in*DIMM+im];
    }
  }
  fprintf(stdout, "C <- array(dim=c(%d,%d))\n", DIMM, DIMN);
  for(im=0; im<DIMM; im++) {
    for(in=0; in<DIMN; in++) {
      fprintf(stdout, "C[%d,%d] <- %25.16e + %25.16e*1.i\n", im+1,in+1, creal(C[im*DIMN+in]), cimag(C[im*DIMN+in]));
    }
  } 
  fprintf(stdout, "TC <- array(dim=c(%d,%d))\n", DIMN, DIMM);
  for(in=0; in<DIMN; in++) {
    for(im=0; im<DIMM; im++) {
      fprintf(stdout, "TC[%d,%d] <- %25.16e + %25.16e*1.i\n", in+1,im+1, creal(TC[in*DIMM+im]), cimag(TC[in*DIMM+im]));
    }
  } 
#endif

  fprintf(stdout, "X <- numeric(%d)\n", DIMM);
  for(ik=0; ik<DIMM; ik++) {
    fprintf(stdout, "X[%d] <- %25.16e + %25.16e*1.i\n", ik+1, creal(X[ik]), cimag(X[ik]));
  }
  for(ik=0; ik<DIMM; ik++) {
    X[ik] = conj(X[ik]);
  }


  BLAS_M     = DIMK;
  BLAS_N     = DIMM;
  BLAS_TRANS = 'N';
  BLAS_ALPHA = 1.;
  BLAS_BETA  = 0.;
  BLAS_A     = A;
  BLAS_LDA   = BLAS_M;
  BLAS_X     = X;
  BLAS_Y     = Y;
  BLAS_INCX  = 1;
  BLAS_INCY  = 1;

  _F(zgemv) ( &BLAS_TRANS, &BLAS_M, &BLAS_N, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_X, &BLAS_INCX, &BLAS_BETA, BLAS_Y, &BLAS_INCY, 1);

  for(im=0; im<DIMK; im++) {
    Y[im] = conj(Y[im]);
  }

  fprintf(stdout, "Y <- numeric(%d)\n", DIMK);
  for(im=0; im<DIMK; im++) {
    fprintf(stdout, "Y[%d] <- %25.16e + %25.16e*1.i\n", im+1, creal(Y[im]), cimag(Y[im]));
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  free_geometry();

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_invert] %s# [test_invert] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_invert] %s# [test_invert] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}
