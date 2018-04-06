/****************************************************
 * test_reference_rotations.cpp
 *
 * Sat May 13 22:36:53 CEST 2017
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
        fprintf(stdout, "# [reference_rotations] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [reference_rotations] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

 /*********************************
  * set number of openmp threads
  *********************************/
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

  if(init_geometry() != 0) {
    fprintf(stderr, "[reference_rotations] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  rot_init_rotation_table();

  rlxd_init(2, g_seed);

  /***********************************************************
   * momentum lists
   ***********************************************************/

  int const momentum_ref[3][3] = { {0,0,1}, {1,1,0}, {1,1,1} };

  int const momentum_num[3] = {6, 12, 8};

  int momentum_class_c4v[ 6][3] = { {0,0,1}, {0,0,-1}, {0,1,0}, {0,-1,0}, {1,0,0}, {-1,0,0} };

  int const momentum_class_c2v[12][3] = { {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0}, {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1}, {0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1} };

  int const momentum_class_c3v[ 8][3] = { {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1}, {-1,1,1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1} };

  char const  *momentum_class_name[3] = { "C4v", "C2v", "C3v" };

  /***********************************************************
   * loop on momentum classes
   ***********************************************************/
  for ( int ic = 0; ic < 3; ic++ ) 
  {

    fprintf ( stdout, "# [reference_rotations] ##################################################################################\n\n");
    fprintf ( stdout, "# [reference_rotations] momentum class %d pref %3d %3d %3d\n", ic, momentum_ref[ic][0], momentum_ref[ic][1], momentum_ref[ic][2] );


    int **momentum_class = NULL;
    init_2level_ibuffer ( &momentum_class, momentum_num[ic], 3 );

    switch(ic) {
      case 0:
         for ( int imom=0; imom<momentum_num[ic]; imom++ ) memcpy ( momentum_class[imom], momentum_class_c4v[imom], 3*sizeof(int) );
         break;
      case 1:
         for ( int imom=0; imom<momentum_num[ic]; imom++ ) memcpy ( momentum_class[imom], momentum_class_c2v[imom], 3*sizeof(int) );
         break;
      case 2:
         for ( int imom=0; imom<momentum_num[ic]; imom++ ) memcpy ( momentum_class[imom], momentum_class_c3v[imom], 3*sizeof(int) );
         break;
    }


    /***********************************************************
     * loop on momenta insice class
     ***********************************************************/
    for ( int imom = 0; imom < momentum_num[ic]; imom++ ) {
      fprintf ( stdout, "# [reference_rotations] ==================================================================================\n\n");
      fprintf ( stdout, "# [reference_rotations] momentum class %d p %3d %3d %3d\n\n", ic, momentum_class[imom][0], momentum_class[imom][1], momentum_class[imom][2] );

      unsigned int rot_count = 0;

      /***********************************************************
       * loop on rotations
       ***********************************************************/
#if 0
      for(int irot=0; irot < 48; irot++ ) {

        double _Complex ** S = rot_init_rotation_matrix (3);
        double _Complex ** R = rot_init_rotation_matrix (3);

        rot_rotation_matrix_spherical_basis ( S, 2, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );

        rot_spherical2cartesian_3x3 ( R, S );

        if ( ! rot_mat_check_is_real_int ( R, 3 ) ) {
          fprintf(stderr, "[reference_rotations] rot_mat_check_is_real_int false for matrix R rot %2d\n", irot);
          EXIT(6);
        }

        int prot[3] = {0,0,0};
        rot_point ( prot, (int*)momentum_ref[ic], R );

        double norm = sqrt(
          _SQR(prot[0] - momentum_class[imom][0]) + 
          _SQR(prot[1] - momentum_class[imom][1]) + 
          _SQR(prot[2] - momentum_class[imom][2]) );

      
        if ( norm == 0 ) {
          rot_count++;
          fprintf( stdout, "# [reference_rotations] rot %2d n = ( %d,  %d,  %d) w = %16.6e pi\n", irot+1,
              cubic_group_double_cover_rotations[irot].n[0], cubic_group_double_cover_rotations[irot].n[1], cubic_group_double_cover_rotations[irot].n[2],
              cubic_group_double_cover_rotations[irot].w / M_PI );

          char name[40];
          sprintf(name, "Rref[[%d,%2d,%2u]]", ic, imom, rot_count);
          rot_printf_rint_matrix (R, 3, name, stdout );
        }

        rot_fini_rotation_matrix ( &S );
        rot_fini_rotation_matrix ( &R );

      }  // end of loop on rotations
#endif  // of if 0

      for(int irot=0; irot < 24; irot++ ) {

        double _Complex ** S = rot_init_rotation_matrix (3);
        double _Complex ** R = rot_init_rotation_matrix (3);

        rot_rotation_matrix_spherical_basis_Wigner_D ( S, 2, cubic_group_rotations_v2[irot].a );

        rot_spherical2cartesian_3x3 ( R, S );

        if ( ! rot_mat_check_is_real_int ( R, 3 ) ) {
          fprintf(stderr, "[reference_rotations] rot_mat_check_is_real_int false for matrix R rot %2d\n", irot);
          EXIT(6);
        }

        int prot[3] = {0,0,0};
        rot_point ( prot, (int*)momentum_ref[ic], R );

        double norm = sqrt(
          _SQR(prot[0] - momentum_class[imom][0]) + 
          _SQR(prot[1] - momentum_class[imom][1]) + 
          _SQR(prot[2] - momentum_class[imom][2]) );

      
        if ( norm == 0 ) {
          rot_count++;
          fprintf( stdout, "# [reference_rotations] rot %2d %5s a = %16.7e pi %16.7e pi %16.7e pi\n", irot+1,
              cubic_group_rotations_v2[irot].name,
              cubic_group_rotations_v2[irot].a[0] / M_PI,
              cubic_group_rotations_v2[irot].a[1] / M_PI,
              cubic_group_rotations_v2[irot].a[2] / M_PI );

          char name[40];
          sprintf(name, "Rref[[%d,%2d,%2u]]", ic, imom, rot_count);
          rot_printf_rint_matrix (R, 3, name, stdout );
        }

        rot_fini_rotation_matrix ( &S );
        rot_fini_rotation_matrix ( &R );

      }  // end of loop on rotations


    }  // end of loop on momenta inside class

    fini_2level_ibuffer ( &momentum_class );
  }  // end of loop on classes

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [reference_rotations] %s# [reference_rotations] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [reference_rotations] %s# [reference_rotations] end of run\n", ctime(&g_the_time));
  }

  free_geometry ();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
