/****************************************************
 * test_rot_gamma_spherical.cpp
 *
 * Do 1. Mär 17:29:46 CET 2018
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
        fprintf(stdout, "# [test_rot_gamma_spherical] exit\n");
        exit(1);
      break;
    }
  }


  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_rot_gamma_spherical] Reading input from file %s\n", filename);
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
    fprintf(stderr, "[test_rot_gamma_spherical] Error from init_geometry\n");
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
    fprintf (stderr, "[test_rot_gamma_spherical] Error from set_rot_mat_table_spin, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( ( exitstatus = set_rot_mat_table_spin ( &Spin_12b, 1, 1 ) ) != 0 ) {
    fprintf (stderr, "[test_rot_gamma_spherical] Error from set_rot_mat_table_spin, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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


  double _Complex **GammaCart[3], **GammaSpher[3];
   
  for ( int i = 0; i < 3; i++ ) {
    GammaCart[i] = rot_init_rotation_matrix ( 4 );
    memcpy ( GammaCart[i][0], g[i].v, 16*sizeof(double _Complex ) );

    GammaSpher[i]    = rot_init_rotation_matrix ( 4 );
  }

  rot_cartesian_to_spherical_contravariant_mat ( GammaSpher, GammaCart, 4, 4 );

  /***********************************************************
   * loop on rotations
   ***********************************************************/
  for(int irot=0; irot < 48; irot++ )
  /* for(int irot=0; irot < 1; irot++ ) */
  {

    fprintf(stdout, "\n\n# [test_rot_gamma_spherical] rot no %2d\n", irot );


    /***********************************************************
     * init 2 helper matrices
     ***********************************************************/
    double _Complex **R = rot_init_rotation_matrix ( 3 );

    rot_mat_assign ( R, Spin_1.R[irot], 3);
    rot_mat_adj ( R, R, 3);


    for ( int i = 0; i < 3; i++ ) {

      /***********************************************************
       * A <- Spin_12b g.m
       * B <- A Spin_12b^+
       ***********************************************************/

      double _Complex **A = rot_init_rotation_matrix ( 4 );
      double _Complex **B = rot_init_rotation_matrix ( 4 );

      rot_mat_ti_mat ( A, Spin_12b.R[irot], GammaSpher[i], 4);

      rot_mat_ti_mat_adj ( B, A, Spin_12b.R[irot], 4 );

      rot_fini_rotation_matrix( &A );


      /***********************************************************
       * A <- Spin_12b g.m
       * B <- A Spin_12b^+ = Spin_12b g.m Spin_12b^+
       ***********************************************************/
      double _Complex **C = rot_init_rotation_matrix ( 4 );

      for ( int k = 0; k < 3; k++ ) {
        rot_mat_pl_eq_mat_ti_co ( C, GammaSpher[k], R[i][k], 4);
      }


      fprintf(stdout, "\n# [test_rot_gamma_spherical] rot no %2d comp %d\n", irot, i );
      char name[20];

      //sprintf ( name, "B[[%2d]]", i );
      //rot_printf_matrix ( B[i], 4, name, stdout );
      
      //fprintf(stdout, "\n");

      //sprintf ( name, "C[[%2d]]", i );
      //rot_printf_matrix ( C[i], 4, name, stdout );

      //sprintf ( name, "Gamma[[%d]]", i );
      //rot_printf_matrix_comp ( C[i], B[i], 4, name, stdout );

      /***********************************************************
       * || g.m - Spin_12b g.m Spin_12b^+ ||
       ***********************************************************/
      double norm =  rot_mat_norm_diff ( C, B, 4 );
      double norm2 = sqrt ( rot_mat_norm2 ( B, 3) );

 
      fprintf ( stdout, "# [test_rot_gamma_spherical] rot %2d comp %d norm diff %16.7e / %16.7e   ok %d\n", irot, i, norm , norm2, norm<9.e-15);

      rot_fini_rotation_matrix( &B );
      rot_fini_rotation_matrix( &C );
    }  /* end of loop on 3-vector components */


    rot_fini_rotation_matrix( &R );

  }  /* end of loop on rotations */

  for ( int i = 0; i < 3; i++ ) {
    rot_fini_rotation_matrix ( GammaCart+i );
    rot_fini_rotation_matrix ( GammaSpher+i );
  }

  fini_rot_mat_table ( &Spin_1 );
  fini_rot_mat_table ( &Spin_12b );


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_rot_gamma_spherical] %s# [test_rot_gamma_spherical] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_rot_gamma_spherical] %s# [test_rot_gamma_spherical] end of run\n", ctime(&g_the_time));
  }


  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
