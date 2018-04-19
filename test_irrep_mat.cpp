/****************************************************
 * test_irrep_mat.cpp
 *
 * Do 19. Apr 16:54:45 CEST 2018
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
#include "scalar_products.h"

#define _SQR(_a) ((_a)*(_a))

using namespace cvc;

int main(int argc, char **argv) {

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;


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
        fprintf(stdout, "# [test_irrep_mat] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_irrep_mat] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

 /*********************************
  * set number of openmp threads
  *********************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_irrep_mat] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_irrep_mat] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  rot_init_rotation_table();

  rlxd_init(2, g_seed);

  /***********************************************************
   * momentum lists
   ***********************************************************/

  int mom[3] = {0,0,1};

  int rot_count = 0;
  for(int irot=0; irot < 48; irot++ ) {

    double _Complex ** S = rot_init_rotation_matrix (3);
    double _Complex ** R = rot_init_rotation_matrix (3);

    // rot_rotation_matrix_spherical_basis_Wigner_D ( S, 2, cubic_group_rotations_v2[irot].a );
    rot_rotation_matrix_spherical_basis ( S, 2, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );
    
    rot_spherical2cartesian_3x3 ( R, S );

    if ( ! rot_mat_check_is_real_int ( R, 3 ) ) {
      fprintf(stderr, "[test_irrep_mat] rot_mat_check_is_real_int false for matrix R rot %2d\n", irot);
      EXIT(6);
    }

    int prot[3] = {0,0,0};
    rot_point ( prot, mom, R );

    double norm1 = sqrt(
      _SQR(prot[0] - mom[0]) + 
      _SQR(prot[1] - mom[1]) + 
      _SQR(prot[2] - mom[2]) );

    double norm2 = 1.; // = sqrt(
      //_SQR(prot[0] + mom[0]) + 
      //_SQR(prot[1] + mom[1]) + 
      //_SQR(prot[2] + mom[2]) );
      
    if ( norm1 == 0 || norm2 == 0) {
      rot_count++;
      fprintf( stdout, "# [test_irrep_mat] rot %2d n = %3d %3d %3d%16.7e w = %16.7e prot %3d %3d %3d\n", irot+1,
          cubic_group_double_cover_rotations[irot].n[0],
          cubic_group_double_cover_rotations[irot].n[1],
          cubic_group_double_cover_rotations[irot].n[2],
          cubic_group_double_cover_rotations[irot].w / M_PI,
          prot[0], prot[1], prot[2] );

      char name[40];
      sprintf(name, "S[[%2d]]", irot+1);
      // rot_printf_rint_matrix (R, 3, name, stdout );
      rot_printf_matrix (S, 3, name, stdout );
    }

    rot_fini_rotation_matrix ( &S );
    rot_fini_rotation_matrix ( &R );

  }  // end of loop on rotations

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_irrep_mat] %s# [test_irrep_mat] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_irrep_mat] %s# [test_irrep_mat] end of run\n", ctime(&g_the_time));
  }

  free_geometry ();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
