/****************************************************
 * test_rot_Cgamma
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>
#include "ranlxd.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "read_input_parser.h"
#include "matrix_init.h"
#include "rotations.h"
#include "group_projection.h"
#include "gamma.h"

using namespace cvc;

/***********************************************************/
/***********************************************************/

int main(int argc, char **argv) {

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  // FILE *ofs = NULL;


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
      case 'f':
        strcpy(filename, optarg);
        filename_set=1;
        break;
      case 'h':
      case '?':
      default:
        fprintf(stdout, "# [test_rot_Cgamma] exit\n");
        exit(1);
      break;
    }
  }

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_rot_Cgamma] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
   mpi_init(argc, argv);

  /***********************************************************
   * set number of openmp threads
   ***********************************************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_rot_Cgamma] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_rot_Cgamma] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_rot_Cgamma] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************/
  /***********************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_rot_Cgamma] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /***********************************************************
   *
   ***********************************************************/
  init_gamma_matrix ();

  /***********************************************************
   *
   ***********************************************************/
  rot_init_rotation_table();

  rot_mat_table_type Spin_1, Spin_12b;

  init_rot_mat_table ( &Spin_1 );
  init_rot_mat_table ( &Spin_12b );


  if ( ( exitstatus = set_rot_mat_table_spin ( &Spin_1, 2, 0 ) ) != 0 ) {
    fprintf (stderr, "[test_rot_Cgamma] Error from set_rot_mat_table_spin, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( ( exitstatus = set_rot_mat_table_spin ( &Spin_12b, 1, 1 ) ) != 0 ) {
    fprintf (stderr, "[test_rot_Cgamma] Error from set_rot_mat_table_spin, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

#if 0
  gamma_matrix_type g;
  /***********************************************************
   * set gamma matrix to id
   ***********************************************************/
  gamma_matrix_set ( &g, 0, 1. );

  /***********************************************************
   * print gamma matrix
   ***********************************************************/
  gamma_matrix_printf ( &g, "gamma", stdout );
#endif

  gamma_matrix_type g[16], C, Cg[3];
  /***********************************************************
   * set gamma matrix to id
   ***********************************************************/
  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set ( &(g[i]), i, 1. );

    /***********************************************************
     * print gamma matrix
     ***********************************************************/
    /* if ( g_verbose > 1 ) {
      char name[20];
      sprintf ( name, "g%.2d", i );
      gamma_matrix_printf ( g+i, name, stdout );
    } */
  }

  /***********************************************************
   * charge conjugation matrix
   ***********************************************************/
  gamma_matrix_init ( &C );

  gamma_matrix_mult ( &C, g, g+2 );

  gamma_matrix_printf ( &C, "C", stdout );

  for ( int i = 0; i < 3; i++ ) {
    gamma_matrix_init ( Cg+i );

    gamma_matrix_mult ( Cg+i, &C, g+i+1 );

    if ( g_verbose > 1 ) {
      char name[20];
      sprintf ( name, "Cg%.2d", i+1 );
      gamma_matrix_printf ( Cg+i, name, stdout );
    }

  }


  /***********************************************************
   * loop on rotations
   ***********************************************************/
  for(int irot=0; irot < 48; irot++ )
  {

    fprintf(stdout, "\n\n# [test_rot_Cgamma] rot no %2d\n", irot );

    /***********************************************************
     * init helper matrices
     ***********************************************************/
    double _Complex **A = init_2level_ztable ( 4, 4 );

    double _Complex *** B = init_3level_ztable ( 3, 4, 4);

    gamma_matrix_type gout[3], grot;

    gamma_matrix_init ( &grot );

    /* grot <- Spin_12b */
    memcpy ( grot.v , Spin_12b.R[irot][0], 16*sizeof(double _Complex) );
   
    /***********************************************************
     * induced spin-1/2 + 1/2 rotation
     ***********************************************************/
    for ( int i = 0; i < 3; i++ ) {

      gamma_matrix_init ( gout+i );

      /* gout_i <- grot x Cg_i grot^T */
      gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( gout+i, &grot, 'N', Cg+i, &grot, 'T' );

      if ( g_verbose > 1 ) {
        char name[20];
        sprintf ( name, "SN_Cg%.2d_ST", i+1 );
        gamma_matrix_printf ( gout+i, name, stdout );
      }

    }

    /* R <- cartesian <- Spin_1 */
    double _Complex **R = init_2level_ztable ( 3, 3 );
    rot_spherical2cartesian_3x3 ( R, Spin_1.R[irot] );

    if ( rot_mat_check_is_real_int ( R, 3 ) == 0 ) {
      fprintf ( stderr, "[test_rot_Cgamma] Error from rot_mat_check_is_real_int\n" );
      EXIT(2);
    }

    rot_printf_matrix ( R, 3, "R", stdout );

    /***********************************************************
     * spin-1 rotation in cartesian basis,
     * left-multiplication with C
     ***********************************************************/
    for ( int i = 0; i < 3; i++ ) {

      /* A <- g_k R_ki */
      rot_mat_zero ( A, 4 );
      rot_mat_zero ( B[i], 4 );

      for ( int k = 0; k < 3; k++ ) {

        double _Complex const zcoeff = R[k][i] * g_gamma_transposed_sign[k+1] * g_gamma_transposed_sign[i+1];

        fprintf ( stdout, "# [test_rot_Cgamma] zcoeff = %25.16e %25.16e\n", creal(zcoeff), cimag(zcoeff) );

        rot_mat_pl_eq_mat_ti_co ( A, g[k+1].m, zcoeff, 4 );
      }
      rot_printf_matrix ( A, 4, "A", stdout );


      /* B_i <- C A = C g_k R_ki */
      rot_mat_ti_mat ( B[i], C.m, A, 4 );

    }

    /***********************************************************
     * difference and norm
     ***********************************************************/
    for ( int i = 0; i < 3; i++ ) {
      /* diff <- || gout_i - B_i || */
      double diff = rot_mat_norm_diff ( gout[i].m, B[i], 4 );

      /* norm <- || gout_i || */
      double norm = sqrt ( rot_mat_norm2  ( gout[i].m, 4 ) );
 
      rot_printf_matrix_comp ( gout[i].m, B[i], 4, "gout-B", stdout );


      fprintf ( stdout, "# [test_rot_Cgamma] rot %2d comp %d diff %16.7e   norm %16.7e    ok %d\n", irot, i, diff, norm , (diff/norm)<9.e-15);
    }
    
    // g_gamma_transposed_sign[]

    fini_2level_ztable ( &A );
    fini_2level_ztable ( &R );
    fini_3level_ztable( &B );

  }  /* end of loop on rotations */


  fini_rot_mat_table ( &Spin_1 );
  fini_rot_mat_table ( &Spin_12b );
#if 0
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_rot_Cgamma] %s# [test_rot_Cgamma] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_rot_Cgamma] %s# [test_rot_Cgamma] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
