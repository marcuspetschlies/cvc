/****************************************************
 * test_rot_mat_direct.cpp
 *
 * Fr 26. Jan 09:53:27 CET 2018
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
#include "ranlxd.h"

using namespace cvc;

/***********************************************************/
/***********************************************************/

int main(int argc, char **argv) {

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  int Ndim = 0;


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:d:")) != -1) {
    switch (c) {
      case 'f':
        strcpy(filename, optarg);
        filename_set=1;
        break;
      case 'd':
        Ndim = atoi ( optarg );
        fprintf ( stdout, "# [test_rot_mat_direct] Ndim set to %d\n", Ndim );
        break;
      case 'h':
      case '?':
      default:
        fprintf(stdout, "# [test_rot_mat_direct] exit\n");
        exit(1);
      break;
    }
  }


  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_rot_mat_direct] Reading input from file %s\n", filename);
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
    fprintf(stderr, "[test_rot_mat_direct] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * initialize rotation matrix tables
   ***********************************************************/
  rot_init_rotation_table();

  /***********************************************************
   * chekc spin and dimension
   * Ndim = 2 J + 1
   ***********************************************************/
  if ( Ndim <= 0 ) {
    fprintf(stderr, "[test_rot_mat_direct] Error, Ndim non-positive\n");
    EXIT(2);
  }

  /***********************************************************
   * loop on rotations
   ***********************************************************/
  for(int irot=0; irot < 48; irot++ )
  {
    char name[20];

    fprintf(stdout, "\n# [test_rot_mat_direct] rotation no. %2d n = (%2d, %2d, %2d) w = %16.7e pi\n", irot+1,
        cubic_group_double_cover_rotations[irot].n[0], cubic_group_double_cover_rotations[irot].n[1], cubic_group_double_cover_rotations[irot].n[2],
        cubic_group_double_cover_rotations[irot].w / M_PI );

    double _Complex **Us = rot_init_rotation_matrix (Ndim);

    rot_rotation_matrix_spherical_basis ( Us, Ndim-1, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );

    if ( rot_mat_check_is_sun ( Us, Ndim) ) {
      fprintf(stdout, "# [test_rot_mat_direct] rot_mat_check_is_sun rot %2d ok\n", irot+1);
    } else {
      fprintf( stdout, "[test_rot_mat_direct] Error rot_mat_check_is_sun rot %2d failed\n", irot+1);
    }


    if ( Ndim == 3 ) {

      double _Complex **Uc = rot_init_rotation_matrix (Ndim);

      rot_spherical2cartesian_3x3 ( Uc, Us );

      double _Complex **Ud = rot_init_rotation_matrix (Ndim);

      exitstatus = rot_mat_spin1_cartesian ( Ud, cubic_group_double_cover_rotations[irot].n,  cubic_group_double_cover_rotations[irot].w );
   
      double norm  = rot_mat_norm_diff ( Uc, Ud, Ndim );
      double norm2 = rot_mat_norm2 ( Uc, Ndim );

  
      fprintf( stdout, "# [test_rot_mat_direct] spin 1 rot %2d s2c - direct norm diff = %16.7e norm = %16.7e\n", irot+1, norm, sqrt(norm2) );

      rot_fini_rotation_matrix ( &Uc );
      rot_fini_rotation_matrix ( &Ud );

    } else if ( Ndim = 2 ) {

      double _Complex **Ud = rot_init_rotation_matrix (Ndim);

      exitstatus = rot_mat_spin1_2_spherical ( Ud, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );

      double norm  = rot_mat_norm_diff ( Us, Ud, Ndim );
      double norm2 = rot_mat_norm2 ( Us, Ndim );
      fprintf( stdout, "# [test_rot_mat_direct] spin 1_2 rot %2d s - direct norm diff = %16.7e norm = %16.7e\n", irot+1, norm , norm2 );


      rot_printf_matrix_comp ( Us, Ud, Ndim, "spin12", stdout );


      rot_fini_rotation_matrix ( &Ud );


    }  /* end of if Ndim == 3 */



    rot_fini_rotation_matrix ( &Us );
  

  }  /* end of loop on rotations */

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_rot_mat_direct] %s# [test_rot_mat_direct] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_rot_mat_direct] %s# [test_rot_mat_direct] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
