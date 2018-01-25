/****************************************************
 * test_euler_angles_D.cpp
 *
 * Do 25. Jan 09:38:06 CET 2018
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

#define _SQR(_a) ((_a)*(_a))

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

  const double eps = 5.e-15;

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  double norm, norm2;
  FILE *ofs = NULL;


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
        fprintf(stdout, "# [test_euler_angles_D] exit\n");
        exit(1);
      break;
    }
  }


  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_euler_angles_D] Reading input from file %s\n", filename);
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
    fprintf(stderr, "[test_euler_angles_D] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   *
   ***********************************************************/
  rot_init_rotation_table();

  /***********************************************************
   * loop on rotations
   ***********************************************************/
  ofs = stdout;


  for ( int i = 0; i < 24; i++ )
  {

    char name[20];

    /* fprintf( ofs, "\n# [test_euler_angles_D] rotation no. %2d n = (%2d, %2d, %2d) w = %16.7e pi    alpha = %16.7e pi, beta  = %16.7e pi, gamma = %16.7e pi\n", irot+1,
        cubic_group_rotations[irot].n[0], cubic_group_rotations[irot].n[1], cubic_group_rotations[irot].n[2],
        cubic_group_rotations[irot].w / M_PI,
        cubic_group_rotations[irot].a[0] / M_PI, cubic_group_rotations[irot].a[1] / M_PI, cubic_group_rotations[irot].a[2] / M_PI );*/

    double euler_angles[2][3];

    int irot = cubic_group_double_cover_identification_table[i][0];
    rot_mat_get_euler_angles ( euler_angles[0], cubic_group_rotations[irot].n, cubic_group_rotations[irot].w );

    fprintf( ofs, "\n# [test_euler_angles_D] rot %2d Euler angles  alpha = %16.7e pi, beta  = %16.7e pi, gamma = %16.7e pi\n", irot+1,
        euler_angles[0][0] / M_PI, euler_angles[0][1] / M_PI, euler_angles[0][2] / M_PI ); 

    irot = cubic_group_double_cover_identification_table[i][1];
    rot_mat_get_euler_angles ( euler_angles[1], cubic_group_rotations[irot].n, cubic_group_rotations[irot].w );

    fprintf( ofs, "\n# [test_euler_angles_D] rot %2d Euler angles  alpha = %16.7e pi, beta  = %16.7e pi, gamma = %16.7e pi\n\n", irot+1,
        euler_angles[1][0] / M_PI, euler_angles[1][1] / M_PI, euler_angles[1][2] / M_PI ); 

    if ( sqrt( _SQR( euler_angles[0][0] - euler_angles[1][0] ) + _SQR( euler_angles[0][1] - euler_angles[1][1] ) + _SQR( euler_angles[0][2] - euler_angles[1][2] ) ) < eps ) {
      fprintf( stdout, "# [test_euler_angles] rot pair no %2d Euler angles match\n", i);
    }

  }  /* end of loop on rotations */


  /* fclose ( ofs ); */

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_euler_angles_D] %s# [test_euler_angles_D] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_euler_angles_D] %s# [test_euler_angles_D] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
