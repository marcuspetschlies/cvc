/************************************************
 * fft.cpp
 *
 * Tue Sep 26 10:53:36 CEST 2017
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
#include "matrix_init.h"
#include "fft.h"

namespace cvc {

/**************************************************************************************************************
 * reorder
 * r[ i ] = s [ permutation[i] ]
 **************************************************************************************************************/
void complex_field_reorder ( double _Complex *r, double _Complex *s, unsigned int *p, unsigned int N) {

  if ( r == s ) {
    fprintf(stderr, "[complex_field_reorder] Error, r == s %s %d\n", __FILE__, __LINE__);
    EXIT(1);
  }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int ix = 0; ix < N; ix++ ) {
    r[ix] = s[ p[ix] ];
  }
}  /* end of complex_field_reorder */

/**************************************************************************************************************
 * fft function
 **************************************************************************************************************/

int ft_4dim ( double *r, double *s, int sign, int init ) {

  const double TWO_MPI = (double)sign * 2. * M_PI;

  double ratime, retime;
  int exitstatus;
  double _Complex **sx = NULL;
 
  double _Complex *s3 = NULL;

  static double _Complex **phase_matrix[4] = { NULL, NULL, NULL, NULL };
  static unsigned int *index_reorder[4];

  /***********************************************************
   * set ZGEMM parameters
   ***********************************************************/
  char BLAS_TRANSA  = 'N', BLAS_TRANSB = 'N';
  int BLAS_M, BLAS_K, BLAS_N, BLAS_LDA, BLAS_LDB, BLAS_LDC;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  ratime = _GET_TIME;
  
  /***********************************************************
   * auxilliary fields
   ***********************************************************/
  s3 = ( double _Complex *) malloc ( VOLUME * sizeof(double _Complex ) );
  if ( s3 == NULL) { 
    fprintf(stderr, "[ft_4dim] Error from malloc, status was %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

  /***********************************************************
   * index reordering fields
   ***********************************************************/

  if ( init == 1 || init == 3 ) {

    if ( g_cart_id == 0 && g_verbose > 2 ) fprintf( stdout, "# [ft_4dim] initialize index reordering\n");

    if ( index_reorder[0] != NULL ) { free ( index_reorder[0] ); }

    index_reorder[0] = (unsigned int*) malloc ( 4 * VOLUME * sizeof(unsigned int) );
    if ( index_reorder[0] == NULL ) {
      fprintf(stderr, "[ft_4dim] Error from malloc %s %d\n", __FILE__, __LINE__ );
      return(1);
    }
    index_reorder[1] = index_reorder[0] + VOLUME;
    index_reorder[2] = index_reorder[1] + VOLUME;
    index_reorder[3] = index_reorder[2] + VOLUME;

    /***********************************************************
     * make the index reorder fields
     *
     *   index_reorder [ target index ] = source index
     ***********************************************************/
    for ( int x0 = 0; x0 < T;  x0++ ) {
    for ( int x1 = 0; x1 < LX; x1++ ) {
    for ( int x2 = 0; x2 < LY; x2++ ) {
    for ( int x3 = 0; x3 < LZ; x3++ ) {
      unsigned int i_txyz = ( ( (unsigned int)x0 * LX + x1 ) * LY + x2 ) * LZ + x3;

      unsigned int i_xyzt = ( ( (unsigned int)x1 * LY + x2 ) * LZ + x3 ) * T  + x0;
      
      unsigned int i_yztx = ( ( (unsigned int)x2 * LZ + x3 ) * T  + x0 ) * LX + x1;

      unsigned int i_ztxy = ( ( (unsigned int)x3 * T  + x0 ) * LX + x1 ) * LY + x2;
        
      /* x y z t <- t x y z */

      index_reorder[0][ i_xyzt ] = i_txyz; 

      /* y z t x <- x y z t */
      index_reorder[1][ i_yztx ] = i_xyzt;

      /* z t x y <- y z t x */
      index_reorder[2][ i_ztxy ] = i_yztx;

      /* t x y z <- z t x y */
      index_reorder[3][ i_txyz ] = i_ztxy;
    }}}}

  }

  /***********************************************************
   * build the 1-dim FT phase matrix, t-direction
   ***********************************************************/
  if ( init == 1 || init == 3 ) {

    if ( g_cart_id == 0 && g_verbose > 2 ) fprintf( stdout, "# [ft_4dim] initialize phase matrices\n");

    if (  phase_matrix[0] != 0 ) { fini_2level_zbuffer ( &(phase_matrix[0]) ); }
    if (  phase_matrix[1] != 0 ) { fini_2level_zbuffer ( &(phase_matrix[1]) ); }
    if (  phase_matrix[2] != 0 ) { fini_2level_zbuffer ( &(phase_matrix[2]) ); }
    if (  phase_matrix[3] != 0 ) { fini_2level_zbuffer ( &(phase_matrix[3]) ); }

    if ( ( exitstatus = init_2level_zbuffer ( &(phase_matrix[0]), T_global, T_global ) ) != 0 ) {
      fprintf(stderr, "[ft_4dim] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(1);
    }
    if ( ( exitstatus = init_2level_zbuffer ( &(phase_matrix[1]), LX_global, LX_global ) ) != 0 ) {
      fprintf(stderr, "[ft_4dim] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(1);
    }
    if ( ( exitstatus = init_2level_zbuffer ( &(phase_matrix[2]), LY_global, LY_global ) ) != 0 ) {
      fprintf(stderr, "[ft_4dim] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(1);
    }
    if ( ( exitstatus = init_2level_zbuffer ( &(phase_matrix[3]), LZ_global, LZ_global ) ) != 0 ) {
      fprintf(stderr, "[ft_4dim] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(1);
    }

    for ( int ip = 0; ip < T_global; ip++ ) {
      double p = TWO_MPI * ip  / (double)T_global;
      for ( int ix = 0; ix < T_global; ix++ ) {
        phase_matrix[0][ip][ix] = cexp ( I * p * ix );
      }
    }

    for ( int ip = 0; ip < LX_global; ip++ ) {
      double p = TWO_MPI * ip  / (double)LX_global;
      for ( int ix = 0; ix < LX_global; ix++ ) {
        phase_matrix[1][ip][ix] = cexp ( I * p * ix );
      }
    }

    for ( int ip = 0; ip < LY_global; ip++ ) {
      double p = TWO_MPI * ip  / (double)LY_global;
      for ( int ix = 0; ix < LY_global; ix++ ) {
        phase_matrix[2][ip][ix] = cexp ( I * p * ix );
      }
    }

    for ( int ip = 0; ip < LZ_global; ip++ ) {
      double p = TWO_MPI * ip  / (double)LZ_global;
      for ( int ix = 0; ix < LZ_global; ix++ ) {
        phase_matrix[3][ip][ix] = cexp ( I * p * ix );
      }
    }
  }  /* end of if init == 1 */

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * Fourier transform t-direction
   ***********************************************************/

  if ( ( exitstatus = init_2level_zbuffer ( &sx, 2, T_global*LX*LY*LZ ) ) != 0 ) {
    fprintf(stderr, "[ft_4dim] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

  memcpy ( s3, s, VOLUME * sizeof(double _Complex ) );

#ifdef HAVE_MPI
  if ( ( exitstatus = MPI_Allgather ( s3, 2*VOLUME, MPI_DOUBLE, sx[0], 2*VOLUME, MPI_DOUBLE, g_tr_comm ) ) != MPI_SUCCESS ) {
    fprintf(stderr, "[ft_4dim] Error from MPI_Allgather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
#else
  memcpy ( sx[0], s3, VOLUME * sizeof(double _Complex ) );
#endif

  /***********************************************************
   * 1-dim. Fourer transform
   ***********************************************************/
  BLAS_M      = LX*LY*LZ ;
  BLAS_K      = T_global;
  BLAS_N      = T_global;
  BLAS_A      = sx[0];
  BLAS_B      = phase_matrix[0][0];
  BLAS_C      = sx[1];
  BLAS_LDA    = BLAS_M;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  /***********************************************************
   * reorder, t-direction becomes innermost
   ***********************************************************/
  complex_field_reorder ( s3, sx[1]+g_proc_coords[0]*VOLUME, index_reorder[0], VOLUME );

  fini_2level_zbuffer ( &sx );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * Fourier transform, x-direction
   ***********************************************************/
  if ( ( exitstatus = init_2level_zbuffer ( &sx, 2, LX_global * T*LY*LZ ) ) != 0 ) {
    fprintf(stderr, "[ft_4dim] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

#if ( defined HAVE_MPI ) && ( ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ ) )
  if ( ( exitstatus = MPI_Allgather ( s3, 2*VOLUME, MPI_DOUBLE, sx[0], 2*VOLUME, MPI_DOUBLE, g_xr_comm ) ) != MPI_SUCCESS ) {
    fprintf(stderr, "[ft_4dim] Error from MPI_Allgather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
#else
  memcpy ( sx[0], s3, VOLUME * sizeof(double _Complex ) );
#endif

  /***********************************************************
   * 1-dim. Fourer transform
   ***********************************************************/
  BLAS_M      = T*LY*LZ ;
  BLAS_K      = LX_global;
  BLAS_N      = LX_global;
  BLAS_A      = sx[0];
  BLAS_B      = phase_matrix[1][0];
  BLAS_C      = sx[1];
  BLAS_LDA    = BLAS_M;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  /***********************************************************
   * reorder, x-direction becomes outermost
   ***********************************************************/
  complex_field_reorder ( s3, sx[1]+g_proc_coords[1]*VOLUME, index_reorder[1], VOLUME );

  fini_2level_zbuffer ( &sx );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * Fourier transform, y-direction
   ***********************************************************/
  if ( ( exitstatus = init_2level_zbuffer ( &sx, 2, LY_global * T*LX*LZ ) ) != 0 ) {
    fprintf(stderr, "[ft_4dim] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

#if ( defined HAVE_MPI ) && ( ( defined PARALLELTXY ) || ( defined PARALLELTXYZ ) )
  if ( ( exitstatus = MPI_Allgather ( s3, 2*VOLUME, MPI_DOUBLE, sx[0], 2*VOLUME, MPI_DOUBLE, g_yr_comm ) ) != MPI_SUCCESS ) {
    fprintf(stderr, "[ft_4dim] Error from MPI_Allgather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
#else
  memcpy ( sx[0], s3, VOLUME * sizeof(double _Complex ) );
#endif

  /***********************************************************
   * 1-dim. Fourer transform
   ***********************************************************/
  BLAS_M      = T*LX*LZ ;
  BLAS_K      = LY_global;
  BLAS_N      = LY_global;
  BLAS_A      = sx[0];
  BLAS_B      = phase_matrix[2][0];
  BLAS_C      = sx[1];
  BLAS_LDA    = BLAS_M;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  /***********************************************************
   * reorder, z-direction becomes outermost
   ***********************************************************/
  complex_field_reorder ( s3, sx[1]+g_proc_coords[2]*VOLUME, index_reorder[2], VOLUME );

  fini_2level_zbuffer ( &sx );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * Fourier transform, z-direction
   ***********************************************************/
  if ( ( exitstatus = init_2level_zbuffer ( &sx, 2, LZ_global * T*LX*LY ) ) != 0 ) {
    fprintf(stderr, "[ft_4dim] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

#if ( defined HAVE_MPI ) && ( defined PARALLELTXYZ )
  if ( ( exitstatus = MPI_Allgather ( s3, 2*VOLUME, MPI_DOUBLE, sx[0], 2*VOLUME, MPI_DOUBLE, g_zr_comm ) ) != MPI_SUCCESS ) {
    fprintf(stderr, "[ft_4dim] Error from MPI_Allgather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
#else
  memcpy ( sx[0], s3, VOLUME * sizeof(double _Complex ) );
#endif

  /***********************************************************
   * 1-dim. Fourer transform
   ***********************************************************/
  BLAS_M      = T*LX*LY ;
  BLAS_K      = LZ_global;
  BLAS_N      = LZ_global;
  BLAS_A      = sx[0];
  BLAS_B      = phase_matrix[3][0];
  BLAS_C      = sx[1];
  BLAS_LDA    = BLAS_M;
  BLAS_LDB    = BLAS_K;
  BLAS_LDC    = BLAS_M;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

  /***********************************************************
   * reorder, t-direction becomes outermost
   ***********************************************************/
  complex_field_reorder ( s3, sx[1]+g_proc_coords[3]*VOLUME, index_reorder[3], VOLUME );

  fini_2level_zbuffer ( &sx );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * copy result to r
   ***********************************************************/
  memcpy ( r, s3, VOLUME*sizeof( double _Complex ) );

  /***********************************************************/
  /***********************************************************/

  if ( init == 2 || init == 3 ) {
    fini_2level_zbuffer ( &(phase_matrix[0]) );
    fini_2level_zbuffer ( &(phase_matrix[1]) );
    fini_2level_zbuffer ( &(phase_matrix[2]) );
    fini_2level_zbuffer ( &(phase_matrix[3]) );
    free ( index_reorder[0] ); 
    index_reorder[0] = NULL;
    index_reorder[1] = NULL;
    index_reorder[2] = NULL;
    index_reorder[3] = NULL;
  }
  free ( s3 ); s3 = NULL;

  retime = _GET_TIME;
  if ( g_cart_id == 0 ) {
    fprintf( stdout, "# [ft_4dim] time for ft_4dim = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__ );
  }

  return(0);
}  /* end of ft_4dim */


}  /* end of namespace cvc */
