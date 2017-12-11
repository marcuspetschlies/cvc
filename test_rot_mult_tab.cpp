/****************************************************
 * test_rot_mult_tab.cpp
 *
 * Mo 11. Dez 16:25:05 CET 2017
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

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

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
#include "prepare_source.h"
#include "Q_phi.h"
#include "scalar_products.h"
#include "group_projection.h"

using namespace cvc;

void usage() {
  EXIT(0);
}

int main(int argc, char **argv) {

  const int Ndim = 3;

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  /* FILE *ofs = NULL; */
  double _Complex **R = NULL;
  double _Complex **A = NULL;
  double _Complex **B = NULL;


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
        fprintf(stdout, "# [test_rot_mult_tab] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_rot_mult_tab] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_rot_mult_tab] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_rot_mult_tab] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_rot_mult_tab] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /*********************************
   * set up geometry fields
   *********************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_rot_mult_tab] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();


  /*********************************
   * initialize rotation matrix
   * tables for cubic group and
   * its double cover
   *********************************/
  rot_init_rotation_table();

  /*********************************
   * initialize random number 
   * generator
   *********************************/
  rlxd_init(2, g_seed);


  R = rot_init_rotation_matrix (Ndim);
  A = rot_init_rotation_matrix (Ndim);
  B = rot_init_rotation_matrix (Ndim);

  /***********************************************************
   * loop on rotations
   ***********************************************************/
  for(int irot=0; irot < 48; irot++ )
  {
    char name[12];

    if (g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rot_mult_tab] rotation no. %2d n = (%2d, %2d, %2d) w = %16.7e pi\n", irot,
          cubic_group_double_cover_rotations[irot].n[0], cubic_group_double_cover_rotations[irot].n[1], cubic_group_double_cover_rotations[irot].n[2],
          cubic_group_double_cover_rotations[irot].w);
    }

    rot_rotation_matrix_spherical_basis ( R, Ndim-1, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );

    sprintf(name, "Ashpe[%.2d]", irot);
    rot_printf_matrix ( R, Ndim, name, stdout );


    rot_spherical2cartesian_3x3 (A, R);
    if ( rot_mat_check_is_real_int ( A, Ndim ) ) {
      if (g_cart_id == 0 )
        fprintf(stdout, "# [test_rot_mult_tab] rot_mat_check_is_real_int matrix A rot %2d ok\n", irot);
    } else {
      EXIT(6);
    }

    sprintf(name, "Akart[%.2d]", irot);
    rot_printf_rint_matrix (A, Ndim, name, stdout );

  }  /* end of loop on rotations */

  rot_mat_table_type rtab;

  init_rot_mat_table ( &rtab );

  if ( ( exitstatus = set_rot_mat_table_spin ( &rtab, 2, 0 ) ) != 0 ) {
    fprintf(stderr, "[test_rot_mult_tab] Error from set_rot_mat_table_spin; status was %d\n", exitstatus );
    exit(1);
  }

  fini_rot_mat_table ( &rtab );

rot_spherical2cartesian_3x3 (A, R);
    if ( rot_mat_check_is_real_int ( A, Ndim ) ) {
            if (g_cart_id == 0 )
                      fprintf(stdout, "# [test_rot_mult_tab] rot_mat_check_is_real_int matrix A rot %2d ok\n", irot);
                } else {
                        EXIT(6);
                            }

    sprintf(name, "Akart[%.2d]", irot);
        rot_printf_rint_matrix (A, Ndim, name, stdout );



  rot_fini_rotation_matrix( &R );
  rot_fini_rotation_matrix( &A );
  rot_fini_rotation_matrix( &B );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_rot_mult_tab] %s# [test_rot_mult_tab] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_rot_mult_tab] %s# [test_rot_mult_tab] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
