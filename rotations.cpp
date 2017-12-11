/****************************************************
 * rotations.cpp
 *
 * Sat May 13 21:06:31 CEST 2017
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif

#include "cvc_complex.h"
#include "iblas.h"
#include "ilinalg.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "mpi_init.h"
#include "matrix_init.h"
#include "gamma.h"
#include "rotations.h"
#include "cvc_utils.h"

namespace cvc {

static int n_block[3], block_L, nn_block, block_LL[3], rot_block_params_set = 0;

rotation_type cubic_group_double_cover_rotations[48];
rotation_type cubic_group_rotations[24];

void rot_init_rotation_table () {
#include "set_cubic_group_double_cover_elements.h"

#include "set_cubic_group_elements.h"
  return;
}  /* end of rot_init_rotation_table */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 * check lattice geometry
 *
 * get the minimal local lattice size, which gives the block
 * size block_L;
 *
 * get the number of blocks in each direction
 *
 * set block_LL[3], which gives the block length in each
 * of the 3 directions
 *
 ***********************************************************/
void rot_init_block_params (void) {
  block_L = _MIN( _MIN(LX,LY) , LZ );
  if ( g_cart_id == 0 )
    fprintf(stdout, "# [rot_set_block_params] block length = %d\n", block_L);
  block_LL[0] = block_L;
  block_LL[1] = block_L;
  block_LL[2] = block_L;

  if ( LX % block_L != 0 || LY % block_L != 0 || LZ % block_L != 0 ) {
    if (g_cart_id == 0) {
      fprintf(stderr, "[rot_set_block_params] Error, LX, LY, LZ must be divisible by block length\n");
      fflush(stderr);
    }
    EXIT(4);
  }
  n_block[0] = LX / block_L;
  n_block[1] = LY / block_L;
  n_block[2] = LZ / block_L;
  if ( g_cart_id == 0 )
    fprintf(stdout, "# [rot_set_block_params] number of blocks = %d %d %d\n", n_block[0], n_block[1], n_block[2] );

  nn_block = LX_global / block_L;
  if (g_cart_id == 0)
    fprintf(stdout, "# [rot_set_block_params] number of blocks globally = %d\n", nn_block );

  rot_block_params_set = 1;
}  /* rot_init_block_params */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
void rot_printf_matrix (double _Complex **R, int N, char *A, FILE*ofs ) {
  if ( g_cart_id == 0 ) {
    fprintf(ofs, "%s <- array(dim = c(%d , %d))\n", A, N, N);
    for( int ik = 0; ik < N; ik++ ) {
    for( int il = 0; il < N; il++ ) {
      fprintf(ofs, "%s[%d,%d] <- %25.16e + %25.16e*1.i\n", A, ik+1, il+1, creal( R[ik][il] ), cimag( R[ik][il] ));
    }}
    fflush(ofs);
  }
}  /* end of rot_printf_matrix */
 
/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
void rot_printf_rint_matrix (double _Complex **R, int N, char *A, FILE*ofs ) {
  if ( g_cart_id == 0 ) {
    fprintf(ofs, "# [rot_printf_rint_matrix] matrix %s\n", A);
    for( int ik = 0; ik < N; ik++ ) {
      fprintf(ofs, "    (%2d  ", (int)( creal( R[ik][0] ) ) );
      for( int il = 1; il < N; il++ ) {
        fprintf(ofs, "%2d  ", (int)( creal( R[ik][il] ) ) );
      }
      fprintf(ofs, ")\n");
    }
    fprintf(ofs, "\n");
    fflush(ofs);
  }
}  /* end of rot_printf_matrix */
 
/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
double _Complex determinant_dxd (double _Complex **R, int d) {
  double _Complex res = 0.;

  switch(d) {
    case 1:
      res = R[0][0];
      break;
    case 2:
      res = R[0][0] * R[1][1] - R[0][1] * R[1][0];
      break;
    case 3:
      res = R[0][0] * ( R[1][1] * R[2][2]  - R[1][2] * R[2][1] )
          + R[0][1] * ( R[1][2] * R[2][0]  - R[1][0] * R[2][2] )
          + R[0][2] * ( R[1][0] * R[2][1]  - R[1][1] * R[2][0] );
      break;
    case 4:
      res =
      + R[0][0] * R[0][1] * R[0][2] * R[0][3]
      - R[0][0] * R[0][1] * R[0][3] * R[0][2]
      + R[0][0] * R[0][3] * R[0][1] * R[0][2]
      - R[0][0] * R[0][3] * R[0][2] * R[0][1]
      + R[0][0] * R[0][2] * R[0][3] * R[0][1]
      - R[0][0] * R[0][2] * R[0][1] * R[0][3]
      - R[0][3] * R[0][0] * R[0][1] * R[0][2]
      + R[0][3] * R[0][0] * R[0][2] * R[0][1]
      - R[0][3] * R[0][2] * R[0][0] * R[0][1]
      + R[0][3] * R[0][2] * R[0][1] * R[0][0]
      - R[0][3] * R[0][1] * R[0][2] * R[0][0]
      + R[0][3] * R[0][1] * R[0][0] * R[0][2]
      + R[0][2] * R[0][0] * R[0][1] * R[0][3]
      - R[0][2] * R[0][0] * R[0][3] * R[0][1]
      + R[0][2] * R[0][3] * R[0][0] * R[0][1]
      - R[0][2] * R[0][3] * R[0][1] * R[0][0]
      + R[0][2] * R[0][1] * R[0][3] * R[0][0]
      - R[0][2] * R[0][1] * R[0][0] * R[0][3]
      + R[0][1] * R[0][2] * R[0][0] * R[0][3]
      - R[0][1] * R[0][2] * R[0][3] * R[0][0]
      + R[0][1] * R[0][3] * R[0][2] * R[0][0]
      - R[0][1] * R[0][3] * R[0][0] * R[0][2]
      + R[0][1] * R[0][0] * R[0][3] * R[0][2]
      - R[0][1] * R[0][0] * R[0][2] * R[0][3];
      break;
    default:
      fprintf(stdout, "[determinant_dxd] Error, dim = %d not implemented\n", d);
      return(sqrt(-1.));
      break;
  }

  return(res);

} /* end of determinant_dxd */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * safe, if C = R
 ***********************************************************/
void rot_mat_adj (double _Complex **C, double _Complex **R, int N) {

  double _Complex **S = rot_init_rotation_matrix ( N );
  for(int i=0; i<N; i++) {
  for(int k=0; k<N; k++) {
    S[i][k] = conj( R[k][i] );
  }}
  memcpy(C[0], S[0], N*N*sizeof(double _Complex));
  rot_fini_rotation_matrix ( &S );
}  /* end of rot_mat_adj */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
void rot_spherical2cartesian_3x3 (double _Complex **C, double _Complex **S) {

  const double _Complex r = 1. / sqrt(2.);

  char CHAR_N = 'N', CHAR_C = 'C';
  int INT_3 = 3;
  int exitstatus;
  double _Complex Z_1 = 1., Z_0 = 0.;

  double _Complex **U=NULL, **A=NULL;
  exitstatus = init_2level_buffer ( (double***)(&U), 3, 6);
  if( exitstatus != 0 ) {
    fprintf(stderr, "[spherical2cartesian_3x3] Error from init_2level_buffer\n");
    EXIT(1);
  }
  exitstatus = init_2level_buffer ( (double***)(&A), 3, 6);
  if( exitstatus != 0 ) {
    fprintf(stderr, "[spherical2cartesian_3x3] Error from init_2level_buffer\n");
    EXIT(2);
  }
  
  U[0][0] = -r;
  U[0][1] =  I*r;
  U[0][2] =  0.;
  U[1][0] =  0.;
  U[1][1] =  0.;
  U[1][2] =  1.;
  U[2][0] =  r;
  U[2][1] =  I*r;
  U[2][2] =  0.;

  /* _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1); */

  /* A (C) = U^+ (C) x S (C) = S (F) x U^+ (F) */
  _F(zgemm) ( &CHAR_N, &CHAR_C, &INT_3, &INT_3, &INT_3, &Z_1, S[0], &INT_3, U[0], &INT_3, &Z_0, A[0], &INT_3, 1, 1);

  /* R (C) = A (C) x U (C) = U (F) x A (F) */
  _F(zgemm) ( &CHAR_N, &CHAR_N, &INT_3, &INT_3, &INT_3, &Z_1, U[0], &INT_3, A[0], &INT_3, &Z_0, C[0], &INT_3, 1, 1);

  fini_2level_buffer ( (double***)(&U));
  fini_2level_buffer ( (double***)(&A));

  return;
}  /* end of rot_spherical2cartesian_3x3 */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
void rot_mat_ti_mat (double _Complex **C, double _Complex **A, double _Complex **B, int N) {

  char CHAR_N = 'N';
  int INT_N = N;
  double _Complex Z_1 = 1., Z_0 = 0.;

  /* _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1); */

  /* */
  _F(zgemm) ( &CHAR_N, &CHAR_N, &INT_N, &INT_N, &INT_N, &Z_1, B[0], &INT_N, A[0], &INT_N, &Z_0, C[0], &INT_N, 1, 1);

  return;
}  /* end of rot_mat_ti_mat */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
void rot_mat_ti_mat_adj (double _Complex **C, double _Complex **A, double _Complex **B, int N) {

  char CHAR_N = 'N', CHAR_C = 'C';
  int INT_N = N;
  double _Complex Z_1 = 1., Z_0 = 0.;

  /* _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1); */

  /* */
  _F(zgemm) ( &CHAR_C, &CHAR_N, &INT_N, &INT_N, &INT_N, &Z_1, B[0], &INT_N, A[0], &INT_N, &Z_0, C[0], &INT_N, 1, 1);

  return;
}  /* end of rot_mat_ti_mat_adj */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
void rot_mat_adj_ti_mat (double _Complex **C, double _Complex **A, double _Complex **B, int N) {

  char CHAR_N = 'N', CHAR_C = 'C';
  int INT_N = N;
  double _Complex Z_1 = 1., Z_0 = 0.;

  /* _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1); */

  /* */
  _F(zgemm) ( &CHAR_N, &CHAR_C, &INT_N, &INT_N, &INT_N, &Z_1, B[0], &INT_N, A[0], &INT_N, &Z_0, C[0], &INT_N, 1, 1);

  return;
}  /* end of rot_mat_adj_ti_mat */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
double rot_mat_norm2 (double _Complex **R, int N) {
  double res = 0.;
  for(int i=0; i<N; i++) {
  for(int k=0; k<N; k++) {
    res += creal( R[i][k] * conj(R[i][k]) );
  }}
  return(res);
}  /* end of rot_mat_norm2 */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
int rot_mat_check_is_sun (double _Complex **R, int N) {
  const double eps = 5.e-15;
  double _Complex **A = NULL;
  double _Complex z = determinant_dxd (R, N);

  if( init_2level_buffer ( (double***)(&A), N, 2*N ) != 0 ) {
    EXIT (1);
  }

  rot_mat_ti_mat_adj (A, R, R, N);

  double d = rot_mat_norm2 (A, N);

  fini_2level_buffer( (double***)(&A));

  return(  (int)( ( fabs(d - N) < eps ) && ( fabs( creal(z) - 1 ) < eps ) && ( fabs( cimag(z) ) < eps ) ) );

}  /* end of rot_mat_check_is_sun */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
int rot_mat_check_is_real_int (double _Complex **R, int N ) {

  const double eps = 2.e-15;
  int res = 1;
  for(int i=0; i<N; i++) {
  for(int k=0; k<N; k++) {
    double re = creal ( R[i][k] );
    double im = cimag ( R[i][k] );

    if ( fabs( re - round( re ) ) < eps && fabs( im ) < eps ) {
      res &= 1;
      R[i][k] = round( re ) + 0*I;
    } else {
      res &= 0;
    }
  }}
  return(res);
}  /* rot_mat_check_is_int */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
void rot_mat_ti_eq_re (double _Complex **R, double c, int N) {
  for(int i=0; i<N; i++) {
  for(int k=0; k<N; k++) {
    R[i][k] *= c;
  }}
  return;
}  /* end of rot_mat_ti_eq_re */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
long unsigned int factorial (int n)
{
  if (n >= 1)
    return  (long unsigned int)n * factorial(n-1);
  else
    return(1);
}  /* end of factorial */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
void axis2polar ( double*theta, double*phi, int n[3] ) {

  double r = sqrt( n[0]*n[0] + n[1]*n[1] + n[2]*n[2] );

  if ( r == 0. ) {
    *theta = 0.;
    *phi   = 0.;
  } else {
    *theta = acos( n[2] / r );
    *phi   = atan2( n[1], n[0] );
  }

  if (g_cart_id == 0 ) {
    fprintf(stdout, "# [axis2polar] n %2d %2d %2d   phi %25.16e pi   theta  %25.16e pi\n", n[0], n[1], n[2], *phi/M_PI, *theta/M_PI);
    fflush(stdout);
  }

  return;

}  /* end of axis2polar */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * input
 * J2 = 2 x J (J = 0, 1/2, 1, 3/2, ... )
 * n  = direction of rotation axis
 * w  = rotation angle
 *
 * output
 * R  = rotation matrix in spherical basis
 ***********************************************************/
void rot_rotation_matrix_spherical_basis ( double _Complex**R, int J2, int n[3], double w) {

  double theta, phi;
  double v, vsqr_mi_one;
  double _Complex u;

  axis2polar ( &theta, &phi, n );

  v = sin ( w / 2. ) * sin ( theta );
  vsqr_mi_one = v*v - 1;

  u = cos ( w / 2. ) - I * sin( w / 2.) * cos( theta );

  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [rotation_matrix_spherical_basis] v = %25.16e\n"\
                    "# [rotation_matrix_spherical_basis] u = %25.16e + %25.16e I\n",
                    v, creal(u), cimag(u));
  }
  for( int ik = 0; ik <= J2; ik++ ) {
    int k2 = J2 - 2*ik;

    int J_mi_m1 = ( J2 - k2 ) / 2;
    int J_pl_m1 = ( J2 + k2 ) / 2;

    long unsigned int J_mi_m1_fac = factorial (J_mi_m1);
    long unsigned int J_pl_m1_fac = factorial (J_pl_m1);

    for( int il = 0; il <= J2; il++ ) {
      int l2 = J2 - 2 * il;

      int J_mi_m2 = ( J2 - l2 ) / 2;
      int J_pl_m2 = ( J2 + l2 ) / 2;

      long unsigned int J_mi_m2_fac = factorial (J_mi_m2);
      long unsigned int J_pl_m2_fac = factorial (J_pl_m2);

      double norm = sqrt( J_pl_m1_fac * J_mi_m1_fac * J_pl_m2_fac * J_mi_m2_fac );

      int m1_pl_m2 = (k2 + l2 ) / 2;
      int m1_mi_m2 = (k2 - l2 ) / 2;

      if ( m1_pl_m2 >= 0 ) {

        int smax = _MIN( J_mi_m1, J_mi_m2 );
        if (g_cart_id == 0 ) {
          fprintf(stdout, "# [rotation_matrix_spherical_basis] 2 J = %d, 2 m1 = %d, 2 m2 = %d, smax = %d\n", J2, k2, l2, smax);
        }

        double _Complex ssum = 0.;
        for( int s = 0; s <= smax; s++ ) {
          ssum += pow ( v, J2 - m1_pl_m2 - 2*s ) * pow( vsqr_mi_one, s ) / ( factorial(s) * factorial(s + m1_pl_m2) * factorial(J_mi_m1 - s) * factorial( J_mi_m2 - s) );
        }

        R[ik][il] = ( cos( m1_mi_m2 * phi) - sin( m1_mi_m2*phi) * I ) * ssum * cpow( u, m1_pl_m2)        * cpow( -I, J2 - m1_pl_m2) * norm;


      } else {
        int smax = _MIN( J_pl_m1, J_pl_m2 );
        if (g_cart_id == 0 ) {
          fprintf(stdout, "# [rotation_matrix_spherical_basis] 2 J = %d, 2 m1 = %d, 2 m2 = %d, smax = %d\n", J2, k2, l2, smax);
        }

        double _Complex ssum = 0.;
        for( int s = 0; s <= smax; s++ ) {
          ssum += pow ( v , J2 + m1_pl_m2 - 2*s ) * pow( vsqr_mi_one, s ) / ( factorial(s) * factorial(s - m1_pl_m2) * factorial(J_pl_m1 - s) * factorial( J_pl_m2 - s) );
        }

        R[ik][il] = ( cos( m1_mi_m2 * phi) - sin( m1_mi_m2*phi) * I ) * ssum * cpow( conj(u), -m1_pl_m2) * cpow( -I, J2 + m1_pl_m2) * norm;

      }

    }  /* end of loop on m2 */
  }  /* end of loop on m1 */

  /* TEST */
  /*
  if (g_cart_id == 0 ) {
    fprintf(stdout, "R <- array(dim=(%d , %d))\n", J2+1, J2+1);
    for( int ik = 0; ik <= J2; ik++ ) {
    for( int il = 0; il <= J2; il++ ) {
      fprintf(stdout, "R[%d,%d] <- %25.16e + %25.16e*1.i\n", ik+1, il+1, creal( R[ik][il] ), cimag( R[ik][il] ));
    }}
  }
  */

  return;
}  /* end of rot_rotation_matrix_spherical_basis */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * bi-spinor rotation matrix as (1/2, 0) + (0, 1/2)
 ***********************************************************/
int rot_bispinor_rotation_matrix_spherical_basis ( double _Complex**ASpin, int n[3], double w ) {

  double _Complex **SSpin = NULL;
  int exitstatus = init_2level_buffer( (double***)&SSpin, 2, 4 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[rot_bispinor_rotation_matrix_spherical_basis] Error from init_2level_buffer, status was %d\n", exitstatus);
    return(1);
  }
  if ( ASpin == NULL ) {
    /* exitstatus = init_2level_buffer( (double***)&ASpin, 4, 8 ); */
    /* if ( exitstatus != 0 ) { */
    /* fprintf(stderr, "[rot_bispinor_rotation_matrix_spherical_basis] Error from init_2level_buffer, status was %d\n", exitstatus); */
    fprintf(stderr, "[rot_bispinor_rotation_matrix_spherical_basis] Error, ASpin is NULL\n");
    return(1);
  }

  rot_rotation_matrix_spherical_basis ( SSpin, 1, n, w);

  ASpin[0][0] = SSpin[0][0];
  ASpin[0][1] = SSpin[0][1];
  ASpin[1][0] = SSpin[1][0];
  ASpin[1][1] = SSpin[1][1];
  ASpin[2][2] = SSpin[0][0];
  ASpin[2][3] = SSpin[0][1];
  ASpin[3][2] = SSpin[1][0];
  ASpin[3][3] = SSpin[1][1];

  fini_2level_buffer( (double***)&SSpin );
  return( 0 );
}  /* end of rot_bispinor_rotation_matrix_spherical_basis */ 

/***********************************************************/
/***********************************************************/

/***********************************************************
 * rotate a 3-dim integer vector
 ***********************************************************/
void rot_point ( int nrot[3], int n[3], double _Complex **R) {
  nrot[0] = (int)creal(R[0][0] * (double _Complex)n[0] + R[0][1] * (double _Complex)n[1] + R[0][2] * (double _Complex)n[2]);
  nrot[1] = (int)creal(R[1][0] * (double _Complex)n[0] + R[1][1] * (double _Complex)n[1] + R[1][2] * (double _Complex)n[2]);
  nrot[2] = (int)creal(R[2][0] * (double _Complex)n[0] + R[2][1] * (double _Complex)n[1] + R[2][2] * (double _Complex)n[2]);
}  /* end of rot_point */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * inverse rotate a 3-dim integer vector
 ***********************************************************/
void rot_point_inv ( int nrot[3], int n[3], double _Complex **R) {
  nrot[0] = (int)creal( conj(R[0][0]) * (double _Complex)n[0] + conj(R[1][0]) * (double _Complex)n[1] + conj(R[2][0]) * (double _Complex)n[2]);
  nrot[1] = (int)creal( conj(R[0][1]) * (double _Complex)n[0] + conj(R[1][1]) * (double _Complex)n[1] + conj(R[2][1]) * (double _Complex)n[2]);
  nrot[2] = (int)creal( conj(R[0][2]) * (double _Complex)n[0] + conj(R[1][2]) * (double _Complex)n[1] + conj(R[2][2]) * (double _Complex)n[2]);
}  /* end of rot_point_inv */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * rotate a global 3-dim integer vector module L
 ***********************************************************/
void rot_global_point_mod ( int nrot[3], int n[3], double _Complex **R) {
  nrot[0] = ( (int)creal(R[0][0] * (double _Complex)n[0] + R[0][1] * (double _Complex)n[1] + R[0][2] * (double _Complex)n[2]) + LX_global ) % LX_global;
  nrot[1] = ( (int)creal(R[1][0] * (double _Complex)n[0] + R[1][1] * (double _Complex)n[1] + R[1][2] * (double _Complex)n[2]) + LY_global ) % LY_global;
  nrot[2] = ( (int)creal(R[2][0] * (double _Complex)n[0] + R[2][1] * (double _Complex)n[1] + R[2][2] * (double _Complex)n[2]) + LZ_global ) % LZ_global;
}  /* end of rot_point */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * rotate a 3-dim integer vector
 * center of the rotation is the center of the lattice
 * cube, (LX/2, LY/2, LZ/2)
 ***********************************************************/
void rot_center_global_point ( int nrot[3], int n[3], double _Complex **R) {
  int nshift[3] = { n[0] - LX_global/2, n[1] - LY_global/2, n[2] - LZ_global/2 };
  nrot[0] = (int)creal( R[0][0] * (double _Complex)nshift[0] + R[0][1] * (double _Complex)nshift[1] + R[0][2] * (double _Complex)nshift[2]) + LX_global/2;
  nrot[1] = (int)creal( R[1][0] * (double _Complex)nshift[0] + R[1][1] * (double _Complex)nshift[1] + R[1][2] * (double _Complex)nshift[2]) + LY_global/2;
  nrot[2] = (int)creal( R[2][0] * (double _Complex)nshift[0] + R[2][1] * (double _Complex)nshift[1] + R[2][2] * (double _Complex)nshift[2]) + LZ_global/2;
}  /* end of rot_point */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * inverse rotate a 3-dim integer vector, center of
 * rotation is the center of the lattice cube
 ***********************************************************/
void rot_center_global_point_inv ( int nrot[3], int n[3], double _Complex **R) {
  int nshift[3] = { n[0] - LX_global/2, n[1] - LY_global/2, n[2] - LZ_global/2 };
  nrot[0] = (int)creal( conj(R[0][0]) * (double _Complex)nshift[0] + conj(R[1][0]) * (double _Complex)nshift[1] + conj(R[2][0]) * (double _Complex)nshift[2]) + LX_global/2;
  nrot[1] = (int)creal( conj(R[0][1]) * (double _Complex)nshift[0] + conj(R[1][1]) * (double _Complex)nshift[1] + conj(R[2][1]) * (double _Complex)nshift[2]) + LY_global/2;
  nrot[2] = (int)creal( conj(R[0][2]) * (double _Complex)nshift[0] + conj(R[1][2]) * (double _Complex)nshift[1] + conj(R[2][2]) * (double _Complex)nshift[2]) + LZ_global/2;
}  /* end of rot_point_inv */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * rotate a 3-dim local integer vector
 * center of the rotation is the center of the local lattice
 * cube, (l/2, l/2, l/2)
 ***********************************************************/
void rot_center_local_point ( int nrot[3], int n[3], double _Complex **R, int l[3]) {
  int nshift[3] = { 2*n[0] - l[0], 2*n[1] - l[1], 2*n[2] - l[2] };
  nrot[0] = (int)creal( R[0][0] * (double _Complex)nshift[0] + R[0][1] * (double _Complex)nshift[1] + R[0][2] * (double _Complex)nshift[2]) + l[0];
  nrot[1] = (int)creal( R[1][0] * (double _Complex)nshift[0] + R[1][1] * (double _Complex)nshift[1] + R[1][2] * (double _Complex)nshift[2]) + l[1];
  nrot[2] = (int)creal( R[2][0] * (double _Complex)nshift[0] + R[2][1] * (double _Complex)nshift[1] + R[2][2] * (double _Complex)nshift[2]) + l[2];

  if( nrot[0] % 2 != 0 || nrot[1] % 2 != 0 || nrot[2] % 2 != 0  ) {
    fprintf(stderr, "[rot_center_local_point] Error, nrot not even vector\n");
    EXIT(1);
  } else {
    nrot[0] /= 2;
    nrot[1] /= 2;
    nrot[2] /= 2;
  }
}  /* end of rot_point */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * rotate a 3-dim local integer vector
 * center of the rotation is the center of the local lattice
 * cube, (l/2, l/2, l/2)
 ***********************************************************/
void rot_center_local_point_inv ( int nrot[3], int n[3], double _Complex **R, int l[3]) {
  int nshift[3] = { 2*n[0] - l[0], 2*n[1] - l[1], 2*n[2] - l[2] };
  nrot[0] = (int)creal( conj(R[0][0]) * (double _Complex)nshift[0] + conj(R[1][0]) * (double _Complex)nshift[1] + conj(R[2][0]) * (double _Complex)nshift[2]) + l[0];
  nrot[1] = (int)creal( conj(R[0][1]) * (double _Complex)nshift[0] + conj(R[1][1]) * (double _Complex)nshift[1] + conj(R[2][1]) * (double _Complex)nshift[2]) + l[1];
  nrot[2] = (int)creal( conj(R[0][2]) * (double _Complex)nshift[0] + conj(R[1][2]) * (double _Complex)nshift[1] + conj(R[2][2]) * (double _Complex)nshift[2]) + l[2];

  if( nrot[0] % 2 != 0 || nrot[1] % 2 != 0 || nrot[2] % 2 != 0  ) {
    fprintf(stderr, "[rot_center_local_point] Error, nrot not even vector\n");
    EXIT(1);
  } else {
    nrot[0] /= 2;
    nrot[1] /= 2;
    nrot[2] /= 2;
  }
}  /* end of rot_point */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * check equality of two rotation matrices, i.e.
 *
 * tr [ (A-B)^+ (A-B) ]
 *
 ***********************************************************/
double rot_mat_diff_norm2 (double _Complex **R, double _Complex **S , int N ) {

  double _Complex z;
  double norm2 = 0;

  for ( int i = 0; i < N; i++ ) {
  for ( int k = 0; k < N; k++ ) {
    z = R[i][k] - S[i][k];
    norm2 += creal( z * conj(z) );
  }}
  return(norm2);
}  /* end of rot_mat_diff_norm2 */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * rotate the gauge field
 ***********************************************************/
int rot_gauge_field ( double*gf_rot, double *gf, double _Complex **R) {

  int dperm[3] = {0,0,0};
  int exitstatus;
  int n_block[3], block_L, nn_block, block_LL[3];
  double **gf_buffer = NULL;

#if (defined HAVE_MPI) && (defined PARALLELTXYZ)
  /* bnd =3 bounary */
  double ***gf_bnd3_buffer = NULL;
#endif

  /* check lattice geometry */
  block_L = _MIN( _MIN(LX,LY) , LZ );
  if ( g_cart_id == 0 )
    fprintf(stdout, "# [rot_gauge_field] block length = %d\n", block_L);
  block_LL[0] = block_L;
  block_LL[1] = block_L;
  block_LL[2] = block_L;
 
  if ( LX % block_L != 0 || LY % block_L != 0 || LZ % block_L != 0 ) {
    if (g_cart_id == 0) {
      fprintf(stderr, "[rot_gauge_field] Error, LX, LY, LZ must be divisible by block length\n");
      fflush(stderr);
    }
    EXIT(4);
  }
  n_block[0] = LX / block_L;
  n_block[1] = LY / block_L;
  n_block[2] = LZ / block_L;
  if ( g_cart_id == 0 )
    fprintf(stdout, "# [rot_gauge_field] number of blocks = %d %d %d\n", n_block[0], n_block[1], n_block[2] );

  nn_block = LX_global / block_L;
  if (g_cart_id == 0)
    fprintf(stdout, "# [rot_gauge_field] number of blocks globally = %d\n", nn_block );

  /* check rotation matrix is real- and integer-valued */
  if( !rot_mat_check_is_real_int (R, 3 ) ) {
    if ( g_cart_id == 0 ) {
      fprintf(stderr, "[rot_gauge_field] Error, rotation matrix must be real-integer-valued\n");
      fflush(stderr);
    }
    return(3);
  }
  
  /* rotation of directions */
  for( int i=0; i<3; i++ ) {
    int d[3] = {0,0,0};
    int drot[3] = {0,0,0};
    d[i] = 1;
    rot_point ( drot, d, R);
    /* drot[k] is +/- 1 if non-zero */
    dperm[i] = drot[0] != 0 ? drot[0] : ( drot[1] != 0 ? 2*drot[1] : 3*drot[2] );
  }
  if (g_cart_id == 0 )
    fprintf(stdout, "# [rot_gauge_field] permutation of directions = ( 1  2  3) / (%2d %2d %2d)\n", dperm[0], dperm[1], dperm[2] );

  /* buffer for gauge field */
  gf_buffer = NULL;
  exitstatus = init_2level_buffer( &gf_buffer, T, 72*block_L*block_L*block_L );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[rot_gauge_field] Error from init_2level_buffer, status was %d\n", exitstatus);
    return(3);
  }


#if (defined HAVE_MPI) && (defined PARALLELTXYZ)
  double **gf_mbnd3_buffer = NULL;
  exitstatus = init_3level_buffer( &gf_bnd3_buffer, g_ts_nproc, T, 72 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[rot_gauge_field] Error from init_2level_buffer, status was %d\n", exitstatus);
    return(3);
  }
  exitstatus = init_2level_buffer( &gf_mbnd3_buffer, T, 72 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[rot_gauge_field] Error from init_2level_buffer, status was %d\n", exitstatus);
    return(3);
  }
  for( int it = 0; it < T; it++ ) {
    memcpy( gf_mbnd3_buffer[it], gf + _GGI(g_ipt[it][0][0][0],0), 72*sizeof(double) );
  }
  exitstatus = MPI_Allgather( gf_mbnd3_buffer[0], T*72, MPI_DOUBLE, gf_bnd3_buffer[0][0], T*72, MPI_DOUBLE, g_ts_comm);
  if ( exitstatus != MPI_SUCCESS ) {
    fprintf(stderr, "[rot_gauge_field] Error from MPI_Allgather, status was %d\n", exitstatus);
    return(3);
  }
  fini_2level_buffer( &gf_mbnd3_buffer );
#endif

  /* loop on blocks */

  for( int ibx = 0; ibx < n_block[0]; ibx++ ) {
  for( int iby = 0; iby < n_block[1]; iby++ ) {
  for( int ibz = 0; ibz < n_block[2]; ibz++ ) {

    /******************************************************
     * rotate a local block of the lattice
     *
     *  store in gf_buffer
     *
     * 
     *  loop on target points
     ******************************************************/
    for( int x = 0; x < block_L; x++ ) {
    for( int y = 0; y < block_L; y++ ) {
    for( int z = 0; z < block_L; z++ ) {

      /* target 3-dim. point in gf_buffer */
      unsigned int ix3d = block_L * ( block_L * x + y ) + z;
      unsigned int iy;
      int n[3] = {x,y,z}, nrot[3];
      int point_coords_rot[3], point_coords_shift[3];
      
      /* source point for (x,y,z) by rotation */
      rot_center_local_point( nrot, n, R, block_LL );

      /* point coords in the local lattice */
      point_coords_rot[0] = nrot[0] + ibx * block_L;
      point_coords_rot[1] = nrot[1] + iby * block_L;
      point_coords_rot[2] = nrot[2] + ibz * block_L;

      rot_reduce_point_bnd ( point_coords_shift, point_coords_rot );
      if ( rot_check_point_bnd ( point_coords_shift ) ) {

#if (defined HAVE_MPI) && (defined PARALLELTXYZ)
        if ( point_coords_shift[0] == LX && point_coords_shift[1] == LY && point_coords_shift[2] == LZ ) {
          int proc_coords[3], proc_id;
          if( MPI_Cart_coords(g_ts_comm, g_ts_id, 3, proc_coords) != MPI_SUCCESS ) {
            fprintf(stderr, "[rot_gauge_field] Error from MPI_Cart_coords\n");
            return(34);
          }
          /* get proc id of triple shifted process */
          proc_coords[0] = ( proc_coords[0] + 1 ) % g_nproc_x;
          proc_coords[1] = ( proc_coords[1] + 1 ) % g_nproc_y;
          proc_coords[2] = ( proc_coords[2] + 1 ) % g_nproc_z;
          if ( MPI_Cart_rank( g_ts_comm, proc_coords, &proc_id) != MPI_SUCCESS ) {
            fprintf(stderr, "[rot_gauge_field] Error from MPI_Cart_rank\n");
            return(35);
          }
          /* set direction 0 */
          for( int it=0; it<T; it++ ) {
            _cm_eq_cm ( gf_buffer[it] + _GGI(ix3d, 0), gf_bnd3_buffer[proc_id][it] + _GGI(0,0) );
          }
        } else {
#endif
          fprintf(stderr, "[rot_gauge_field] Error, point (%3d, %3d, %3d) ---> (%3d, %3d, %3d) exceeds boundary halo %s %d\n",
              n[0], n[1], n[2], point_coords_shift[0], point_coords_shift[1], point_coords_shift[2], __FILE__, __LINE__);
          fflush(stderr);
          EXIT(4);
#if (defined HAVE_MPI) && (defined PARALLELTXYZ)
        }
#endif
      } else {
        /* TEST */
        /* fprintf(stdout, "# [rot_gauge_field] proc %.4d block  %2d %2d %2d  n %2d %2d %2d --- %2d %2d %2d --- %2d %2d %2d\n", g_cart_id,
            ibx, iby,ibz,
            x,y, z, 
            point_coords_rot[0], point_coords_rot[1], point_coords_rot[2],
            point_coords_shift[0], point_coords_shift[1], point_coords_shift[2]); */

        /* direction 0 */
        for( int it=0; it<T; it++ ) {
          iy = g_ipt[it][point_coords_shift[0]][point_coords_shift[1]][point_coords_shift[2]];
          /* TEST */
          /* fprintf(stdout, "# [rot_gauge_field] proc %.4d iy = %u = %2d %2d %2d %2d\n", g_cart_id, iy, it, point_coords_shift[0], point_coords_shift[1], point_coords_shift[2]);
          fflush(stdout);
#ifdef HAVE_MPI
          MPI_Barrier(g_cart_grid );
#endif
           */
          /* copy color matrix for ix, 0 <- iy, 0 */
          _cm_eq_cm ( gf_buffer[it] + _GGI(ix3d, 0), gf + _GGI(iy,0) );
        }
#if 0
#endif  /* of if 0 */
      }  /* end of else of if rot_check_point_bnd > 2 */

      /* loop on directions 1, 2, 3 */
      for( int i=0; i<3; i++) {
        int shift = dperm[i] < 0;
        int dir = abs( dperm[i] );

        point_coords_shift[0] = point_coords_rot[0];
        point_coords_shift[1] = point_coords_rot[1];
        point_coords_shift[2] = point_coords_rot[2];

        if (shift) { point_coords_shift[dir-1]--; }

        rot_reduce_point_bnd ( point_coords_shift, point_coords_shift );

        if ( rot_check_point_bnd ( point_coords_shift ) ) {
          fprintf(stderr, "[rot_gauge_field] Error, point (%3d, %3d, %3d) ---> (%3d, %3d, %3d) exceeds boundary halo %s %d\n",
              n[0], n[1], n[2], point_coords_shift[0], point_coords_shift[1], point_coords_shift[2], __FILE__, __LINE__);
          fflush(stderr);
          EXIT(4);
        }

        if( shift ) {
          for( int it=0; it<T; it++ ) {
            iy = g_ipt[it][point_coords_shift[0]][point_coords_shift[1]][point_coords_shift[2]];

            /* TEST */
            /* if (it == 0) {
              fprintf(stdout, "# [rot_gauge_field] n (%2d, %2d, %2d) to (%2d, %2d, %2d) to %4u = (%2d, %2d, %2d) (%2d)\n", n[0], n[1], n[2], nrot[0], nrot[1], nrot[2], iy ,
                 point_coords_shift[0], point_coords_shift[1], point_coords_shift[2], dperm[i]);
            } */

            _cm_eq_cm_dag ( gf_buffer[it] + _GGI(ix3d, i+1 ), gf + _GGI(iy, dir) );
          }  /* end of loop on timeslices */
        } else {
          for( int it=0; it<T; it++ ) {
            iy = g_ipt[it][point_coords_shift[0]][point_coords_shift[1]][point_coords_shift[2]];

            /* TEST */
            /* if (it == 0) {
              fprintf(stdout, "# [rot_gauge_field] n (%2d, %2d, %2d) to (%2d, %2d, %2d) to %4u = (%2d, %2d, %2d) (%2d)\n", n[0], n[1], n[2], nrot[0], nrot[1], nrot[2], iy ,
                               point_coords_shift[0], point_coords_shift[1], point_coords_shift[2], dperm[i]);
            } */
            _cm_eq_cm ( gf_buffer[it] + _GGI(ix3d, i+1) , gf + _GGI(iy, dir ) );
          }  /* end of loop on timeslices */
        }
      }
#if 0
#endif  /* of if 0 */
    }}}



#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||(defined PARALLELTXY) || (defined PARALLELTXYZ) )
    /******************************************************
     * rotate the local block itself
     *
     *   inside the global lattice
     *
     *   may need MPI send
     ******************************************************/

    int block_coords[3]; 
    int block_coords_rot[3];
    int proc_coords[3], proc_id_send, proc_id_recv;
    int block_dim[3] = {nn_block-1, nn_block-1, nn_block-1};

    /* communicate corresponding gauge field time blocks back and forth  */
    int items = T * block_L * block_L * block_L * 72;
    size_t bytes = items * sizeof(double);
    int cntr=0, recv_counter, **recv_block_coords = NULL;
    double ***gf_mbuffer = NULL;
    MPI_Request request[g_nproc];
    MPI_Status status[g_nproc];

    gf_mbuffer = (double***)malloc( n_block[0]*n_block[1]*n_block[2] * sizeof(double**));
    if ( gf_mbuffer == NULL ) {
      fprintf(stderr, "[rot_gauge_field] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(3);
    }

    recv_block_coords = (int**)malloc( n_block[0]*n_block[1]*n_block[2] * sizeof(int*) );
    recv_block_coords[0] = (int*)malloc( 3 * n_block[0]*n_block[1]*n_block[2] * sizeof(int) ); 
    for( int i=1; i < n_block[0]*n_block[1]*n_block[2]; i++ ) {
      recv_block_coords[i] = recv_block_coords[i-1] + 3;
    }

    /******************************************************
     * where do I send my data to?
     *
     * the receiving process needs R recv = me, so
     * I look for Rinv me = recv
     ******************************************************/
    block_coords[0] = ibx + g_proc_coords[1] * n_block[0];
    block_coords[1] = iby + g_proc_coords[2] * n_block[1];
    block_coords[2] = ibz + g_proc_coords[3] * n_block[2];
    /* rotate global block coordinates */
    rot_center_local_point_inv ( block_coords_rot, block_coords, R, block_dim );

    /* TEST */
    /* fprintf(stdout, "# [rot_gauge_field] proc%.4d block %2d %2d %2d block coordinates = %2d %2d %2d --->  %2d %2d %2d\n", g_cart_id,
        ibx, iby,ibz,
        block_coords[0], block_coords[1], block_coords[2],
        block_coords_rot[0], block_coords_rot[1], block_coords_rot[2]);
    fflush(stdout); */

    /* MPI-process to receive from  */
    proc_coords[0] = block_coords_rot[0] / n_block[0];
    proc_coords[1] = block_coords_rot[1] / n_block[1];
    proc_coords[2] = block_coords_rot[2] / n_block[2];

    exitstatus = MPI_Cart_rank( g_ts_comm, proc_coords, &proc_id_send);
    if ( exitstatus != MPI_SUCCESS ) {
      return(5);
    }
    /* TEST */
    /* fprintf(stdout, "# [rot_gauge_field] proc%.4d block %2d %2d %2d   send-to process coords =  %2d %2d %2d = %2d\n",
        g_cart_id, 
        ibx, iby, ibz,
        proc_coords[0], proc_coords[1], proc_coords[2], proc_id_send );
    fflush(stdout); */

    if ( proc_id_send != g_ts_id ) {   /* I don't send to myself */
      fprintf(stdout, "# [rot_gauge_field] proc%.4d / %2d  block %2d %2d %2d starting send to process %2d\n", g_cart_id, g_ts_id, ibx,iby,ibz, proc_id_send);
      MPI_Isend ( gf_buffer[0], items, MPI_DOUBLE, proc_id_send, 100+g_ts_id, g_ts_comm, &request[cntr]);
      cntr++;
    }

    /******************************************************
     * from which process(es)? do I get data?
     *
     * is any of my blocks in the works by some other process,
     * i.e. is R (me) = some other process' current block
     * x,y,z ? 
     *
     ******************************************************/
    recv_counter = 0;
    for( int i1=0; i1 < n_block[0]; i1++ ) {
    for( int i2=0; i2 < n_block[1]; i2++ ) {
    for( int i3=0; i3 < n_block[2]; i3++ ) {

      /* one of my blocks */
      int block_coords_aux[3] = {
        i1 + g_proc_coords[1] * n_block[0],
        i2 + g_proc_coords[2] * n_block[1],
        i3 + g_proc_coords[3] * n_block[2] };

      /* find process, which works on this block */
      rot_center_local_point ( block_coords_rot, block_coords_aux, R, block_dim );


      /* TEST */
      /* if (g_cart_id == 0 ) {
        fprintf(stdout, "# [rot_gauge_field] proc%.4d block %2d %2d %2d  aux %2d %2d %2d ---> %2d %2d %2d --->  %2d %2d %2d\n",
            g_cart_id, 
            // block_coords[0], block_coords[1], block_coords[2],
            ibx, iby, ibz,
            block_coords_aux[0], block_coords_aux[1], block_coords_aux[2],
            block_coords_rot[0], block_coords_rot[1], block_coords_rot[2],
            block_coords_rot[0] % n_block[0],
            block_coords_rot[1] % n_block[1],
            block_coords_rot[2] % n_block[2] );
      } */

      /* MPI-process to receive from  */
      if ( block_coords_rot[0] % n_block[0] == ibx &&
           block_coords_rot[1] % n_block[1] == iby &&
           block_coords_rot[2] % n_block[2] == ibz ) {

        int proc_coords[3] = { block_coords_rot[0] / n_block[0], block_coords_rot[1] / n_block[1], block_coords_rot[2] / n_block[2] };
        exitstatus = MPI_Cart_rank( g_ts_comm, proc_coords, &proc_id_recv);
        if ( exitstatus != MPI_SUCCESS ) {
          return(4);
        }

        /* TEST */
        /* fprintf(stdout, "# [rot_gauge_field] proc%.4d block %2d %2d %2d process %2d has data for me to replace block %2d %2d %2d\n",
            g_cart_id, 
            // block_coords[0], block_coords[1], block_coords[2],
            ibx, iby, ibz,
            proc_id_recv,
          block_coords_aux[0], block_coords_aux[1], block_coords_aux[2] ); */

        /* initialize new mbuffer */
        gf_mbuffer[recv_counter] = NULL;
        exitstatus = init_2level_buffer( &(gf_mbuffer[recv_counter]), T, 72*block_L * block_L * block_L );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[rot_gauge_field] Error from init_2level_buffer, status was %d\n", exitstatus);
          return(3);
        }

        if ( proc_id_recv != g_ts_id ) {   /* recv from different process */
          fprintf(stdout, "# [rot_gauge_field] proc%.4d / %2d  block %2d %2d %2d starting recv from process %2d\n", g_cart_id, g_ts_id, ibx,iby,ibz, proc_id_recv);
          MPI_Irecv( gf_mbuffer[recv_counter][0], items, MPI_DOUBLE, proc_id_recv, 100+proc_id_recv, g_ts_comm, &request[cntr]);
          cntr++;
        } else { /* this is myself, just memcpy */
          memcpy( gf_mbuffer[recv_counter][0], gf_buffer[0], bytes ); 
        }
        recv_block_coords[recv_counter][0] = i1;
        recv_block_coords[recv_counter][1] = i2;
        recv_block_coords[recv_counter][2] = i3;
        /* count up recv_buffer */
        recv_counter++;
      }
    }}}
        
    fprintf(stdout, "# [rot_gauge_field] proc%.4d / %2d  waiting %d send/recv to finish\n", g_cart_id, g_ts_id, cntr);
    fflush(stdout);
    MPI_Waitall(cntr, request, status);

    fprintf(stdout, "# [rot_gauge_field] proc%.4d / %2d  waiting done\n", g_cart_id, g_ts_id);
    fflush(stdout);
    /* loop on receives */
    for ( int irecv = 0; irecv < recv_counter; irecv++ ) {
        
      memcpy( gf_buffer[0], gf_mbuffer[irecv][0], bytes ); 

#else
      int irecv = 0;
      int recv_block_coords[1][3] = { { ibx, iby, ibz} };  /* 0,0,0 in this case */
#endif  /* of if HAVE_MPI and need distribution */


      /* TEST */
      /* fprintf(stdout, "# [rot_gauge_field] proc%.4d block %2d %2d %2d replacing block %2d %2d %2d\n", g_cart_id,
          ibx, iby, ibz, 
          recv_block_coords[irecv][0],
          recv_block_coords[irecv][1],
          recv_block_coords[irecv][2] ); */

    /* set the corresponding block gf_rot */
    for( int it = 0; it < T; it++ ) {
      for( int x = 0; x < block_L; x++ ) {
        int xloc  = x + recv_block_coords[irecv][0] * block_L;
      for( int y = 0; y < block_L; y++ ) {
        int yloc  = y + recv_block_coords[irecv][1] * block_L;
      for( int z = 0; z < block_L; z++ ) {
        int zloc  = z + recv_block_coords[irecv][2] * block_L;

        unsigned int ix3d = block_L * ( block_L * x + y ) + z;
        unsigned int iy = g_ipt[it][xloc][yloc][zloc];

        /* TEST */
        /* fprintf(stdout, "# [rot_gauge_field] proc%.4d block %2d %2d %2d replacing block %2d %2d %2d coords %3d%3d%3d%3d\n", g_cart_id,
            ibx, iby, ibz,
            recv_block_coords[irecv][0],
            recv_block_coords[irecv][1],
            recv_block_coords[irecv][2],
            it, xloc, yloc,zloc );*/


        for( int i=0; i<4; i++ ) {
          _cm_eq_cm( gf_rot + _GGI(iy, i), gf_buffer[it] + _GGI(ix3d,i) );
        }
      }}}
    }


#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||(defined PARALLELTXY) || (defined PARALLELTXYZ) )

      fini_2level_buffer( &(gf_mbuffer[irecv]) );

    }  /* end of loop on receives */

    free ( gf_mbuffer );
    free( recv_block_coords[0] );
    free( recv_block_coords );

#endif  /* of if HAVE_MPI and need distribution */

#if 0
#endif  /* of if 0 */

  }}}  /* end of loop on blocks */

  fini_2level_buffer ( &gf_buffer );
#if (defined HAVE_MPI) && (defined PARALLELTXYZ)
  fini_3level_buffer ( &gf_bnd3_buffer );
#endif


  return(0);

}  /* end of rot_gauge_field */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 * rotate a spinor field
 *
 ***********************************************************/
int rot_spinor_field ( double*sf_rot, double *sf, double _Complex **R) {

  int dperm[3] = {0,0,0};
  int exitstatus;
  double **sf_buffer = NULL;
  double *sf_aux = NULL;

#if (defined HAVE_MPI) && (defined PARALLELTXYZ)
  /* bnd =3 bounary */
  double ***sf_bnd3_buffer = NULL;
#endif

  /* spinor field sf in sf_aux 
   *
   * sf_aux has bnd2, if needed
   * */
#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||(defined PARALLELTXY) || (defined PARALLELTXYZ) )
  sf_aux = (double*)malloc( _GSI(VOLUMEPLUSRAND) * sizeof(double) );
  memset( sf_aux,  0,  _GSI(VOLUMEPLUSRAND) * sizeof(double) );
  if ( sf_aux == NULL )  {
    fprintf(stderr, "[rot_spinor_field] Error from malloc\n");
    fflush(stderr);
    EXIT(4);
  }
  memcpy( sf_aux, sf, _GSI(VOLUME)*sizeof(double) );
  xchange_spinor_field_bnd2( sf_aux );
#else
  sf_aux = sf;
#endif

  if ( ! rot_block_params_set ) rot_init_block_params();

  /* check rotation matrix is real- and integer-valued */
  if( !rot_mat_check_is_real_int (R, 3 ) ) {
    if ( g_cart_id == 0 ) {
      fprintf(stderr, "[rot_spinor_field] Error, rotation matrix must be real-integer-valued\n");
      fflush(stderr);
    }
    return(3);
  }
  
  /* rotation of directions */
  for( int i=0; i<3; i++ ) {
    int d[3] = {0,0,0};
    int drot[3] = {0,0,0};
    d[i] = 1;
    rot_point ( drot, d, R);
    /* drot[k] is +/- 1 if non-zero */
    dperm[i] = drot[0] != 0 ? drot[0] : ( drot[1] != 0 ? 2*drot[1] : 3*drot[2] );
  }
  if (g_cart_id == 0 )
    fprintf(stdout, "# [rot_spinor_field] permutation of directions = ( 1  2  3) / (%2d %2d %2d)\n", dperm[0], dperm[1], dperm[2] );

  /* buffer for gauge field */
  sf_buffer = NULL;
  exitstatus = init_2level_buffer( &sf_buffer, T, 24*block_L*block_L*block_L );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[rot_spinor_field] Error from init_2level_buffer, status was %d\n", exitstatus);
    return(3);
  }


#if (defined HAVE_MPI) && (defined PARALLELTXYZ)
  double **sf_mbnd3_buffer = NULL;
  exitstatus = init_3level_buffer( &sf_bnd3_buffer, g_ts_nproc, T, 24 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[rot_spinor_field] Error from init_2level_buffer, status was %d\n", exitstatus);
    return(3);
  }
  exitstatus = init_2level_buffer( &sf_mbnd3_buffer, T, 24 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[rot_spinor_field] Error from init_2level_buffer, status was %d\n", exitstatus);
    return(3);
  }
  for( int it = 0; it < T; it++ ) {
    memcpy( sf_mbnd3_buffer[it], sf_aux + _GSI(g_ipt[it][0][0][0]), 24*sizeof(double) );
  }
  exitstatus = MPI_Allgather( sf_mbnd3_buffer[0], T*24, MPI_DOUBLE, sf_bnd3_buffer[0][0], T*24, MPI_DOUBLE, g_ts_comm);
  if ( exitstatus != MPI_SUCCESS ) {
    fprintf(stderr, "[rot_spinor_field] Error from MPI_Allgather, status was %d\n", exitstatus);
    return(3);
  }
  fini_2level_buffer( &sf_mbnd3_buffer );
#endif

  /* loop on blocks */

  for( int ibx = 0; ibx < n_block[0]; ibx++ ) {
  for( int iby = 0; iby < n_block[1]; iby++ ) {
  for( int ibz = 0; ibz < n_block[2]; ibz++ ) {

    /******************************************************
     * rotate a local block of the lattice
     *
     *  store in sf_buffer
     *
     * 
     *  loop on target points
     ******************************************************/
    for( int x = 0; x < block_L; x++ ) {
    for( int y = 0; y < block_L; y++ ) {
    for( int z = 0; z < block_L; z++ ) {

      /* target 3-dim. point in sf_buffer */
      unsigned int ix3d = block_L * ( block_L * x + y ) + z;
      unsigned int iy;
      int n[3] = {x,y,z}, nrot[3];
      int point_coords_rot[3], point_coords_shift[3];
      
      /* source point for (x,y,z) by rotation */
      rot_center_local_point( nrot, n, R, block_LL );

      /* point coords in the local lattice */
      point_coords_rot[0] = nrot[0] + ibx * block_L;
      point_coords_rot[1] = nrot[1] + iby * block_L;
      point_coords_rot[2] = nrot[2] + ibz * block_L;

      rot_reduce_point_bnd ( point_coords_shift, point_coords_rot );
      if ( rot_check_point_bnd ( point_coords_shift ) ) {

#if (defined HAVE_MPI) && (defined PARALLELTXYZ)
        if ( point_coords_shift[0] == LX && point_coords_shift[1] == LY && point_coords_shift[2] == LZ ) {
          int proc_coords[3], proc_id;
          if( MPI_Cart_coords(g_ts_comm, g_ts_id, 3, proc_coords) != MPI_SUCCESS ) {
            fprintf(stderr, "[rot_spinor_field] Error from MPI_Cart_coords\n");
            return(34);
          }
          /* get proc id of triple shifted process */
          proc_coords[0] = ( proc_coords[0] + 1 ) % g_nproc_x;
          proc_coords[1] = ( proc_coords[1] + 1 ) % g_nproc_y;
          proc_coords[2] = ( proc_coords[2] + 1 ) % g_nproc_z;
          if ( MPI_Cart_rank( g_ts_comm, proc_coords, &proc_id) != MPI_SUCCESS ) {
            fprintf(stderr, "[rot_spinor_field] Error from MPI_Cart_rank\n");
            return(35);
          }
          for( int it=0; it<T; it++ ) {
            _fv_eq_fv ( sf_buffer[it] + _GSI(ix3d), sf_bnd3_buffer[proc_id][it] );
          }
        } else {
#endif
          fprintf(stderr, "[rot_spinor_field] Error, point (%3d, %3d, %3d) ---> (%3d, %3d, %3d) exceeds boundary halo %s %d\n",
              n[0], n[1], n[2], point_coords_shift[0], point_coords_shift[1], point_coords_shift[2], __FILE__, __LINE__);
          fflush(stderr);
          EXIT(4);
#if (defined HAVE_MPI) && (defined PARALLELTXYZ)
        }
#endif
      } else {
        /* TEST */
        /* fprintf(stdout, "# [rot_spinor_field] proc %.4d block  %2d %2d %2d  n %2d %2d %2d --- %2d %2d %2d --- %2d %2d %2d\n", g_cart_id,
            ibx, iby,ibz,
            x,y, z, 
            point_coords_rot[0], point_coords_rot[1], point_coords_rot[2],
            point_coords_shift[0], point_coords_shift[1], point_coords_shift[2]);*/

        /* set the spinor field */
        for( int it=0; it<T; it++ ) {
          iy = g_ipt[it][point_coords_shift[0]][point_coords_shift[1]][point_coords_shift[2]];
          /* TEST */
          /* fprintf(stdout, "# [rot_spinor_field] proc %.4d iy = %u = %2d %2d %2d %2d\n", g_cart_id, iy, it, point_coords_shift[0], point_coords_shift[1], point_coords_shift[2]);
          fflush(stdout);
#ifdef HAVE_MPI
          MPI_Barrier(g_cart_grid );
#endif
           */
          /* copy color matrix for ix, 0 <- iy, 0 */
          _fv_eq_fv ( sf_buffer[it] + _GSI(ix3d), sf_aux + _GSI(iy) );
        }
      }  /* end of else of if rot_check_point_bnd > 2 */
    }}}
#if 0
#endif  /* of if 0 */

#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||(defined PARALLELTXY) || (defined PARALLELTXYZ) )
    /******************************************************
     * rotate the local block itself
     *
     *   inside the global lattice
     *
     *   may need MPI send
     ******************************************************/

    int block_coords[3]; 
    int block_coords_rot[3];
    int proc_coords[3], proc_id_send, proc_id_recv;
    int block_dim[3] = {nn_block-1, nn_block-1, nn_block-1};

    /* communicate corresponding gauge field time blocks back and forth  */
    int items = T * block_L * block_L * block_L * 24;
    size_t bytes = items * sizeof(double);
    int cntr=0, recv_counter, **recv_block_coords = NULL;
    double ***sf_mbuffer = NULL;
    MPI_Request request[g_nproc];
    MPI_Status status[g_nproc];

    sf_mbuffer = (double***)malloc( n_block[0]*n_block[1]*n_block[2] * sizeof(double**));
    if ( sf_mbuffer == NULL ) {
      fprintf(stderr, "[rot_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(3);
    }

    recv_block_coords = (int**)malloc( n_block[0]*n_block[1]*n_block[2] * sizeof(int*) );
    recv_block_coords[0] = (int*)malloc( 3 * n_block[0]*n_block[1]*n_block[2] * sizeof(int) ); 
    for( int i=1; i < n_block[0]*n_block[1]*n_block[2]; i++ ) {
      recv_block_coords[i] = recv_block_coords[i-1] + 3;
    }

    /******************************************************
     * where do I send my data to?
     *
     * the receiving process needs R recv = me, so
     * I look for Rinv me = recv
     ******************************************************/
    block_coords[0] = ibx + g_proc_coords[1] * n_block[0];
    block_coords[1] = iby + g_proc_coords[2] * n_block[1];
    block_coords[2] = ibz + g_proc_coords[3] * n_block[2];
    /* rotate global block coordinates */
    rot_center_local_point_inv ( block_coords_rot, block_coords, R, block_dim );

    /* TEST */
    /* fprintf(stdout, "# [rot_spinor_field] proc%.4d block %2d %2d %2d block coordinates = %2d %2d %2d --->  %2d %2d %2d\n", g_cart_id,
        ibx, iby,ibz,
        block_coords[0], block_coords[1], block_coords[2],
        block_coords_rot[0], block_coords_rot[1], block_coords_rot[2]);
    fflush(stdout); */

    /* MPI-process to receive from  */
    proc_coords[0] = block_coords_rot[0] / n_block[0];
    proc_coords[1] = block_coords_rot[1] / n_block[1];
    proc_coords[2] = block_coords_rot[2] / n_block[2];

    exitstatus = MPI_Cart_rank( g_ts_comm, proc_coords, &proc_id_send);
    if ( exitstatus != MPI_SUCCESS ) {
      return(5);
    }
    /* TEST */
    /* fprintf(stdout, "# [rot_spinor_field] proc%.4d block %2d %2d %2d   send-to process coords =  %2d %2d %2d = %2d\n",
        g_cart_id, 
        ibx, iby, ibz,
        proc_coords[0], proc_coords[1], proc_coords[2], proc_id_send );
    fflush(stdout); */

    if ( proc_id_send != g_ts_id ) {   /* I don't send to myself */
      fprintf(stdout, "# [rot_spinor_field] proc%.4d / %2d  block %2d %2d %2d starting send to process %2d\n", g_cart_id, g_ts_id, ibx,iby,ibz, proc_id_send);
      MPI_Isend ( sf_buffer[0], items, MPI_DOUBLE, proc_id_send, 100+g_ts_id, g_ts_comm, &request[cntr]);
      cntr++;
    }

    /******************************************************
     * from which process(es)? do I get data?
     *
     * is any of my blocks in the works by some other process,
     * i.e. is R (me) = some other process' current block
     * x,y,z ? 
     *
     ******************************************************/
    recv_counter = 0;
    for( int i1=0; i1 < n_block[0]; i1++ ) {
    for( int i2=0; i2 < n_block[1]; i2++ ) {
    for( int i3=0; i3 < n_block[2]; i3++ ) {

      /* one of my blocks */
      int block_coords_aux[3] = {
        i1 + g_proc_coords[1] * n_block[0],
        i2 + g_proc_coords[2] * n_block[1],
        i3 + g_proc_coords[3] * n_block[2] };

      /* find process, which works on this block */
      rot_center_local_point ( block_coords_rot, block_coords_aux, R, block_dim );


      /* TEST */
      /* if (g_cart_id == 0 ) {
        fprintf(stdout, "# [rot_spinor_field] proc%.4d block %2d %2d %2d  aux %2d %2d %2d ---> %2d %2d %2d --->  %2d %2d %2d\n",
            g_cart_id, 
            // block_coords[0], block_coords[1], block_coords[2],
            ibx, iby, ibz,
            block_coords_aux[0], block_coords_aux[1], block_coords_aux[2],
            block_coords_rot[0], block_coords_rot[1], block_coords_rot[2],
            block_coords_rot[0] % n_block[0],
            block_coords_rot[1] % n_block[1],
            block_coords_rot[2] % n_block[2] );
      } */

      /* MPI-process to receive from  */
      if ( block_coords_rot[0] % n_block[0] == ibx &&
           block_coords_rot[1] % n_block[1] == iby &&
           block_coords_rot[2] % n_block[2] == ibz ) {

        int proc_coords[3] = { block_coords_rot[0] / n_block[0], block_coords_rot[1] / n_block[1], block_coords_rot[2] / n_block[2] };
        exitstatus = MPI_Cart_rank( g_ts_comm, proc_coords, &proc_id_recv);
        if ( exitstatus != MPI_SUCCESS ) {
          return(4);
        }

        /* TEST */
        /* fprintf(stdout, "# [rot_spinor_field] proc%.4d block %2d %2d %2d process %2d has data for me to replace block %2d %2d %2d\n",
            g_cart_id, 
            // block_coords[0], block_coords[1], block_coords[2],
            ibx, iby, ibz,
            proc_id_recv,
          block_coords_aux[0], block_coords_aux[1], block_coords_aux[2] ); */

        /* initialize new mbuffer */
        sf_mbuffer[recv_counter] = NULL;
        exitstatus = init_2level_buffer( &(sf_mbuffer[recv_counter]), T, 24*block_L * block_L * block_L );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[rot_spinor_field] Error from init_2level_buffer, status was %d\n", exitstatus);
          return(3);
        }

        if ( proc_id_recv != g_ts_id ) {   /* recv from different process */
          fprintf(stdout, "# [rot_spinor_field] proc%.4d / %2d  block %2d %2d %2d starting recv from process %2d\n", g_cart_id, g_ts_id, ibx,iby,ibz, proc_id_recv);
          MPI_Irecv( sf_mbuffer[recv_counter][0], items, MPI_DOUBLE, proc_id_recv, 100+proc_id_recv, g_ts_comm, &request[cntr]);
          cntr++;
        } else { /* this is myself, just memcpy */
          memcpy( sf_mbuffer[recv_counter][0], sf_buffer[0], bytes ); 
        }
        recv_block_coords[recv_counter][0] = i1;
        recv_block_coords[recv_counter][1] = i2;
        recv_block_coords[recv_counter][2] = i3;
        /* count up recv_buffer */
        recv_counter++;
      }
    }}}
        
    fprintf(stdout, "# [rot_spinor_field] proc%.4d / %2d  waiting %d send/recv to finish\n", g_cart_id, g_ts_id, cntr);
    fflush(stdout);
    MPI_Waitall(cntr, request, status);

    fprintf(stdout, "# [rot_spinor_field] proc%.4d / %2d  waiting done\n", g_cart_id, g_ts_id);
    fflush(stdout);
    /* loop on receives */
    for ( int irecv = 0; irecv < recv_counter; irecv++ ) {
        
      memcpy( sf_buffer[0], sf_mbuffer[irecv][0], bytes ); 
#else
      int irecv = 0;
      int recv_block_coords[1][3] = { { ibx, iby, ibz} };  /* 0,0,0 in this case */
#endif  /* of if HAVE_MPI and need distribution */

      /* TEST */
      /* fprintf(stdout, "# [rot_spinor_field] proc%.4d block %2d %2d %2d replacing block %2d %2d %2d\n", g_cart_id,
          ibx, iby, ibz, 
          recv_block_coords[irecv][0],
          recv_block_coords[irecv][1],
          recv_block_coords[irecv][2] ); */

    /* set the corresponding block sf_rot */
    for( int it = 0; it < T; it++ ) {
      for( int x = 0; x < block_L; x++ ) {
        int xloc  = x + recv_block_coords[irecv][0] * block_L;
      for( int y = 0; y < block_L; y++ ) {
        int yloc  = y + recv_block_coords[irecv][1] * block_L;
      for( int z = 0; z < block_L; z++ ) {
        int zloc  = z + recv_block_coords[irecv][2] * block_L;

        unsigned int ix3d = block_L * ( block_L * x + y ) + z;
        unsigned int iy = g_ipt[it][xloc][yloc][zloc];

        /* TEST */
        /* fprintf(stdout, "# [rot_spinor_field] proc%.4d block %2d %2d %2d replacing block %2d %2d %2d coords %3d%3d%3d%3d\n", g_cart_id,
            ibx, iby, ibz,
            recv_block_coords[irecv][0],
            recv_block_coords[irecv][1],
            recv_block_coords[irecv][2],
            it, xloc, yloc,zloc );*/

        _fv_eq_fv( sf_rot + _GSI(iy), sf_buffer[it] + _GSI(ix3d) );
      }}}
    }  /* end of loop on timeslices */
#if 0
#endif  /* of if 0 */

#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||(defined PARALLELTXY) || (defined PARALLELTXYZ) )

      fini_2level_buffer( &(sf_mbuffer[irecv]) );

    }  /* end of loop on receives */

    free ( sf_mbuffer );
    free( recv_block_coords[0] );
    free( recv_block_coords );

#endif  /* of if HAVE_MPI and need distribution */

  }}}  /* end of loop on blocks */

  fini_2level_buffer ( &sf_buffer );
#if (defined HAVE_MPI) && (defined PARALLELTXYZ)
  fini_3level_buffer ( &sf_bnd3_buffer );
#endif

#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||(defined PARALLELTXY) || (defined PARALLELTXYZ) )
  free ( sf_aux );
#endif

  return(0);

}  /* end of rot_spinor_field */

/***********************************************************/
/***********************************************************/

void rot_bispinor_mat_ti_spinor_field (double *sf_rot, double _Complex **R, double *sf, unsigned int N) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix = 0; ix < N; ix++ ) { 
    unsigned int offset = _GSI(ix);
    rot_bispinor_mat_ti_fv( sf_rot+offset, R, sf+offset );
  }
}  /* end of rot_bispinor_mat_ti_spinor_field */

/***********************************************************/
/***********************************************************/

void rot_bispinor_mat_ti_fp_field( fermion_propagator_type *fp_rot, double _Complex ** R, fermion_propagator_type *fp, unsigned int N ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix = 0; ix < N; ix++ ) { 
    rot_bispinor_mat_ti_fp( fp_rot[ix], R, fp[ix] );
  }
}  /* end of rot_bispinor_mat_ti_fp_field */

/***********************************************************/
/***********************************************************/

void rot_fp_field_ti_bispinor_mat ( fermion_propagator_type *fp_rot, double _Complex ** R, fermion_propagator_type *fp, unsigned int N) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix = 0; ix < N; ix++ ) { 
    rot_fp_ti_bispinor_mat ( fp_rot[ix], R, fp[ix] );
  }
}  /* end of rot_fp_field_ti_bispinor_mat */

/***********************************************************/
/***********************************************************/

void rot_spinor_field_ti_bispinor_mat ( double**sf_rot, double _Complex ** R, double**sf, unsigned int N ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix = 0; ix < N; ix++ ) { 
    rot_fv_ti_bispinor_mat ( sf_rot, R, sf, ix);
  }
}  /* end of rot_spinor_field_ti_bispinor_mat */

/***********************************************************/
/***********************************************************/

void rot_bispinor_mat_ti_sp_field ( spinor_propagator_type *sp_rot, double _Complex ** R, spinor_propagator_type *sp, unsigned int N ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix = 0; ix < N; ix++ ) { 
    rot_bispinor_mat_ti_sp ( sp_rot[ix], R, sp[ix] );
  }
}  /* end of rot_bispinor_mat_ti_sp_field */

/***********************************************************/
/***********************************************************/

void rot_sp_field_ti_bispinor_mat ( spinor_propagator_type *sp_rot, double _Complex ** R, spinor_propagator_type *sp, unsigned int N ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix = 0; ix < N; ix++ ) { 
    rot_sp_ti_bispinor_mat ( sp_rot[ix], R, sp[ix] );
  }
}  /* end of rot_sp_field_bispinor_mat */

/***********************************************************/
/***********************************************************/

void rot_inversion_matrix_spherical_basis ( double _Complex**R, int J2, int bispinor ) {
  memset ( R[0], 0, (1+bispinor)*(J2+1) * (1+bispinor)*(J2+1) * sizeof(double _Complex) );

  if ( J2 == 0 ) { 
    R[0][0] = -1.; 
  
  } else if ( J2 == 2 ) { 
    R[0][0] = -1.; 
    R[1][1] = -1.; 
    R[2][2] = -1.; 
  } else if ( J2 == 1 && bispinor ) {
    gamma_matrix_type g;
    gamma_matrix_set ( &g, 0, 1 );

    memcpy ( R[0], g.v, 16*sizeof(double _Complex) );
  } else {
    fprintf( stderr, "[rot_inversion_matrix_spherical_basis] Error, unknown combination of J and bispinor\n");
    return;
  }

  return;
}  /* end of rot_inversion_matrix_spherical_basis */


/***********************************************************/
/***********************************************************/
}  /* end of namespace cvc */
