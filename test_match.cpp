/****************************************************
 * test_match.cpp
 *
 * Do 8. Mär 11:34:48 CET 2018
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
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>
#include "ranlxd.h"


#define MAIN_PROGRAM

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
#include "scalar_products.h"
#include "group_projection.h"

#define _SQR(_a) ((_a)*(_a))

using namespace cvc;


int main(int argc, char **argv) {

  double const eps = 1.e-14;

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
        fprintf(stdout, "# [test_match] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_match] Reading input from file %s\n", filename);
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
    fprintf(stderr, "[test_match] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  rot_init_rotation_table();

  rlxd_init(2, g_seed);

#if 0
  rot_mat_table_type rspin, sspin;

  init_rot_mat_table ( &rspin );
  init_rot_mat_table ( &sspin );

  set_rot_mat_table_spin_single_cover ( &rspin, 2, 0, 1 );

  set_rot_mat_table_spin_single_cover ( &sspin, 2, 1, 1 );

  /***********************************************************
   * loop on rotations
   ***********************************************************/

  for(int irot=0; irot < 24; irot++ )
  {
    int have_match = 0;
    for ( int k = 0; k < 24; k++ ) {

      double norm = rot_mat_norm_diff ( sspin.R[irot], rspin.R[k], 3 );

      if ( norm <= eps ) {
        fprintf ( stdout, "# [test_match] sspin %2d matches rspin %2d\n", irot, k );
        have_match++;
      }

    }

    if ( have_match != 1 )  {
      fprintf(stderr, "[test_match] Error matching rotation %2d\n", irot );
      EXIT(2);
    }
  }
#endif

  int nlg = 0;
  little_group_type *lg;

  nlg = little_group_read_list ( &lg, "little_groups_Oh.tab" );

  little_group_show ( lg, stdout, nlg );

  for ( int ilg = 0; ilg < nlg; ilg++ ) {

    fprintf ( stdout, "\n\n# [test_match] little group %s\n", lg[ilg].name );

    for ( int irot = 0; irot < lg[ilg].nr; irot++ ) {

      double _Complex ** S = rot_init_rotation_matrix ( 3 );
      double _Complex ** R = rot_init_rotation_matrix ( 3 );

      rot_rotation_matrix_spherical_basis_Wigner_D ( S, 2, cubic_group_rotations_v2[lg[ilg].r[irot]].a );

      if ( ! rot_mat_check_is_sun ( S , 3) ) {
        fprintf ( stderr, "[test_match] rot %2d is not SU(3)\n", irot+1 );
        EXIT(2);
      } else {
        fprintf ( stdout, "# [test_match] rot %2d is SU(3)\n", irot+1 );
      }
      rot_spherical2cartesian_3x3 ( R, S);
      if ( ! rot_mat_check_is_real_int ( R, 3 ) ) {
        fprintf ( stderr, "[test_match] rot %2d is not real int\n", irot+1 );
        EXIT(3);
      } else {
        fprintf ( stdout, "# [test_match] rot %2d is real int\n", irot+1 );
      }

      int drot[3] = {0,0,0};

      rot_point ( drot, lg[ilg].d, R );

      double norm = sqrt (
        _SQR( lg[ilg].d[0] - drot[0] ) + 
        _SQR( lg[ilg].d[1] - drot[1] ) + 
        _SQR( lg[ilg].d[2] - drot[2] ) );
      double norm2 =  sqrt ( _SQR( lg[ilg].d[0]) + _SQR( lg[ilg].d[1]) + _SQR( lg[ilg].d[2]) );
      fprintf ( stdout, "# [test_match] rot %2d norm diff  %e %e\n", irot+1, norm, norm2 );

      rot_fini_rotation_matrix ( &S );
      rot_fini_rotation_matrix ( &R );
    }
    fprintf ( stdout, "\n" );

    for ( int irot = 0; irot < lg[ilg].nrm; irot++ ) {

      double _Complex ** S = rot_init_rotation_matrix ( 3 );
      double _Complex ** R = rot_init_rotation_matrix ( 3 );

      rot_rotation_matrix_spherical_basis_Wigner_D ( S, 2, cubic_group_rotations_v2[lg[ilg].rm[irot]].a );

      if ( ! rot_mat_check_is_sun ( S , 3) ) {
        fprintf ( stderr, "[test_match] Irot %2d is not SU(3)\n", irot+1 );
        EXIT(2);
      } else {
        fprintf ( stdout, "# [test_match] Irot %2d is SU(3)\n", irot+1 );
      }
      rot_spherical2cartesian_3x3 ( R, S);
      if ( ! rot_mat_check_is_real_int ( R, 3 ) ) {
        fprintf ( stderr, "[test_match] Irot %2d is not real int\n", irot+1 );
        EXIT(3);
      } else {
        fprintf ( stdout, "# [test_match] Irot %2d is real int\n", irot+1 );
      }

      int drot[3] = {0,0,0};

      rot_point ( drot, lg[ilg].d, R );

      double norm = sqrt (
        _SQR( lg[ilg].d[0] + drot[0] ) + 
        _SQR( lg[ilg].d[1] + drot[1] ) + 
        _SQR( lg[ilg].d[2] + drot[2] ) );
      double norm2 =  sqrt ( _SQR( lg[ilg].d[0]) + _SQR( lg[ilg].d[1]) + _SQR( lg[ilg].d[2]) );
      fprintf ( stdout, "# [test_match] Irot %2d norm diff  %e %e\n", irot+1, norm, norm2 );

      rot_fini_rotation_matrix ( &S );
      rot_fini_rotation_matrix ( &R );
    }
    fprintf ( stdout, "\n" );
  }

  free_geometry ();

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_match] %s# [test_match] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_match] %s# [test_match] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
