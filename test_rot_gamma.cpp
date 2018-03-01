/****************************************************
 * test_rot_gamma.cpp
 *
 * Mi 24. Jan 12:41:04 CET 2018
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
        fprintf(stdout, "# [test_rot_gamma] exit\n");
        exit(1);
      break;
    }
  }


  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_rot_gamma] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
   mpi_init(argc, argv);

  /***********************************************************
   * set number of openmp threads
   ***********************************************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [p2gg] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************/
  /***********************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_rot_gamma] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   *
   ***********************************************************/
  rot_init_rotation_table();

  rot_mat_table_type Spin_1, Spin_12b;

  init_rot_mat_table ( &Spin_1 );
  init_rot_mat_table ( &Spin_12b );


  if ( ( exitstatus = set_rot_mat_table_spin ( &Spin_1, 2, 0 ) ) != 0 ) {
    fprintf (stderr, "[test_rot_gamma] Error from set_rot_mat_table_spin, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( ( exitstatus = set_rot_mat_table_spin ( &Spin_12b, 1, 1 ) ) != 0 ) {
    fprintf (stderr, "[test_rot_gamma] Error from set_rot_mat_table_spin, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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

  gamma_matrix_type g[3];
  /***********************************************************
   * set gamma matrix to id
   ***********************************************************/
  gamma_matrix_set ( g,   1, 1. );
  gamma_matrix_set ( g+1, 2, 1. );
  gamma_matrix_set ( g+2, 3, 1. );

  /***********************************************************
   * print gamma matrix
   ***********************************************************/
  gamma_matrix_printf ( g,   "gamma1", stdout );
  gamma_matrix_printf ( g+1, "gamma2", stdout );
  gamma_matrix_printf ( g+2, "gamma3", stdout );


  /***********************************************************
   * loop on rotations
   ***********************************************************/
  for(int irot=0; irot < 48; irot++ )
  /* for(int irot=0; irot < 1; irot++ ) */
  {

    fprintf(stdout, "\n\n# [test_rot_gamma] rot no %2d\n", irot );


    /***********************************************************
     * init 2 helper matrices
     ***********************************************************/
    double _Complex **A = rot_init_rotation_matrix ( 4 );
    double _Complex **R = rot_init_rotation_matrix ( 3 );

    double _Complex **B[3], **C[3]; 
   
    B[0] = rot_init_rotation_matrix ( 4 );
    B[1] = rot_init_rotation_matrix ( 4 );
    B[2] = rot_init_rotation_matrix ( 4 );


    for ( int i = 0; i < 3; i++ ) {
      /***********************************************************
       * A <- Spin_12b g.m
       * B <- A Spin_12b^+
       ***********************************************************/
      rot_mat_ti_mat ( A, Spin_12b.R[irot], g[i].m, 4);

      rot_mat_ti_mat_adj ( B[i], A, Spin_12b.R[irot], 4 );

    }

    // rot_mat_adj ( R, Spin_1.R[irot], 3);
    // rot_spherical2cartesian_3x3 ( R, R );

    rot_mat_spin1_cartesian ( R, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );
    rot_mat_adj ( R, R, 3);
    /* rot_printf_matrix ( R, 3, "R", stdout ); */

    rot_mat_check_is_real_int ( R, 3 );
    rot_printf_rint_matrix ( R, 3, "R", stdout );
    
    for ( int i = 0; i < 3; i++ ) {

      C[i] = rot_init_rotation_matrix ( 4 );

      for ( int k = 0; k < 3; k++ ) {

        rot_mat_pl_eq_mat_ti_co ( C[i], g[k].m, R[i][k], 4);
        // rot_mat_pl_eq_mat_ti_co ( C[i], g[k].m, R[k][i], 4);
      }


      fprintf(stdout, "\n# [test_rot_gamma] rot no %2d comp %d\n", irot, i );
      char name[20];
      /*
      sprintf ( name, "B[[%2d]]", i );
      rot_printf_matrix ( B[i], 4, name, stdout );
      
      fprintf(stdout, "\n");

      sprintf ( name, "C[[%2d]]", i );
      rot_printf_matrix ( C[i], 4, name, stdout );
*/
      sprintf ( name, "Gamma[[%d]]", i );
      rot_printf_matrix_comp ( C[i], B[i], 4, name, stdout );

      /***********************************************************
       * || g.m - Spin_12b g.m Spin_12b^+ ||
       ***********************************************************/
      double norm =  rot_mat_norm_diff ( C[i], B[i], 4 );
 
      fprintf ( stdout, "# [test_rot_gamma] rot %2d comp %d norm diff %16.7e    ok %d\n", irot, i, norm , norm<9.e-15);

    }  /* end of loop on 3-vector components */


    rot_fini_rotation_matrix( &A );
    rot_fini_rotation_matrix( &R );
    rot_fini_rotation_matrix( B );
    rot_fini_rotation_matrix( B+1 );
    rot_fini_rotation_matrix( B+2 );
    rot_fini_rotation_matrix( C );
    rot_fini_rotation_matrix( C+1 );
    rot_fini_rotation_matrix( C+2 );

  }  /* end of loop on rotations */


  fini_rot_mat_table ( &Spin_1 );
  fini_rot_mat_table ( &Spin_12b );


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_rot_gamma] %s# [test_rot_gamma] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_rot_gamma] %s# [test_rot_gamma] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
