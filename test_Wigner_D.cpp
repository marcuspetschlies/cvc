/****************************************************
 * test_Wigner_D.cpp
 *
 * Di 23. Jan 09:58:25 CET 2018
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

  const double ONE_OVER_SQRT2 = 1. / sqrt(2.);

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  int Ndim = 0;
  double norm, norm2;
  FILE *ofs = NULL;
  double _Complex **R = NULL;
  double _Complex **A = NULL;
  double _Complex **W = NULL;


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
        fprintf(stdout, "# [test_Wigner_D] exit\n");
        exit(1);
      break;
    }
  }


  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_Wigner_D] Reading input from file %s\n", filename);
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
    fprintf(stderr, "[test_Wigner_D] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   *
   ***********************************************************/
  rot_init_rotation_table();

  Ndim = 3;

  R = rot_init_rotation_matrix (Ndim);
  A = rot_init_rotation_matrix (Ndim);
  W = rot_init_rotation_matrix (Ndim);

  /***********************************************************
   * loop on rotations
   ***********************************************************/
  /* ofs = stdout; */

  ofs = fopen("spin1_rotation_matrices", "w");
  if ( ofs == NULL ) {
    fprintf(stderr, "[test_Wigner_D] Error from fopen\n");
    EXIT(2);
  }


  for(int irot=0; irot < 24; irot++ )
  // for(int irot = 18; irot <= 18; irot++ )
  {
    char name[20];

    if (g_cart_id == 0 ) {
      fprintf( ofs, "\n# [test_Wigner_D] rotation no. %2d n = (%2d, %2d, %2d) w = %16.7e pi    alpha = %16.7e pi, beta  = %16.7e pi, gamma = %16.7e pi\n", irot+1,
          cubic_group_rotations[irot].n[0], cubic_group_rotations[irot].n[1], cubic_group_rotations[irot].n[2],
          cubic_group_rotations[irot].w / M_PI,
          cubic_group_rotations[irot].a[0] / M_PI, cubic_group_rotations[irot].a[1] / M_PI, cubic_group_rotations[irot].a[2] / M_PI );
    }

    double euler_angles[3];

    rot_mat_get_euler_angles ( euler_angles, cubic_group_rotations[irot].n, cubic_group_rotations[irot].w );


    fprintf( ofs, "\n# [test_Wigner_D] rot %2d paper angles  alpha = %16.7e pi, beta  = %16.7e pi, gamma = %16.7e pi\n", irot+1,
        cubic_group_rotations[irot].a[0] / M_PI, cubic_group_rotations[irot].a[1] / M_PI, cubic_group_rotations[irot].a[2] / M_PI );
          
    fprintf( ofs, "\n# [test_Wigner_D] rot %2d Euler angles  alpha = %16.7e pi, beta  = %16.7e pi, gamma = %16.7e pi\n", irot+1,
        euler_angles[0] / M_PI, euler_angles[1] / M_PI, euler_angles[2] / M_PI ); 

    fprintf( ofs, "\n# [test_Wigner_D] rot %2d paper angles  sin_b_h = %16.7e, tan_a+g_h  = %16.7e, a-g_h = %16.7e pi \n", irot+1,
      sin ( cubic_group_rotations[irot].a[1] / 2. ), 
      tan ( (cubic_group_rotations[irot].a[0] + cubic_group_rotations[irot].a[2] ) / 2. ),
      ( cubic_group_rotations[irot].a[0] - cubic_group_rotations[irot].a[2] ) / 2. / M_PI );

    fprintf( ofs, "\n# [test_Wigner_D] rot %2d Euler angles  sin_b_h = %16.7e, tan_a+g_h  = %16.7e, a-g_h = %16.7e pi \n", irot+1,
      sin ( euler_angles[1] / 2. ), 
      tan ( ( euler_angles[0] + euler_angles[2] ) / 2. ),
      ( euler_angles[0] - euler_angles[2] ) / 2. / M_PI );

    rot_rotation_matrix_spherical_basis ( R, Ndim-1, cubic_group_rotations[irot].n, cubic_group_rotations[irot].w );
    rot_rotation_matrix_spherical_basis_Wigner_D ( W, Ndim-1, cubic_group_rotations[irot].a );

    if ( rot_mat_check_is_sun ( W, Ndim) ) {
      fprintf(ofs, "# [test_Wigner_D] rot_mat_check_is_sun rot %2d ok\n", irot+1);
    } else {
      fprintf(ofs, "[test_Wigner_D] Error rot_mat_check_is_sun rot %2d failed\n", irot+1);
    }


    sprintf(name, "W_spherical[[%2d]]", irot+1);
    rot_printf_matrix ( W, Ndim, name, ofs );
//    sprintf(name, "R_spherical[[%2d]]", irot+1);
//    rot_printf_matrix ( R, Ndim, name, ofs );
//    rot_printf_matrix_comp ( W, R, Ndim, name, ofs );

    double norm = rot_mat_norm_diff ( W, R, Ndim );
    fprintf( ofs, "# [test_Wigner_D] rot %2d norm diff = %16.7e\n", irot+1, norm );


    if ( Ndim == 3 ) {
      rot_spherical2cartesian_3x3 (A, W );
      if ( rot_mat_check_is_real_int ( A, Ndim ) ) {
        if (g_cart_id == 0 )
          fprintf(ofs, "# [test_Wigner_D] rot_mat_check_is_real_int rot %2d ok\n", irot+1);
      } else {
        fprintf(ofs, "[test_Wigner_D] Error, rot_mat_check_is_real_int rot %2d failed\n", irot+1);
      }

      sprintf(name, "A_cartesian[[%2d]]", irot+1);
      rot_printf_rint_matrix (A, Ndim, name, ofs );
    }

  }  /* end of loop on rotations */

  /* fclose ( ofs ); */

  rot_fini_rotation_matrix( &R );
  rot_fini_rotation_matrix( &A );
  rot_fini_rotation_matrix( &W );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_Wigner_D] %s# [test_Wigner_D] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_Wigner_D] %s# [test_Wigner_D] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
