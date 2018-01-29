/****************************************************
 * test_dinv.cpp
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


#define _SQR(_a) ((_a)*(_a))

using namespace cvc;

void usage() {
  EXIT(0);
}

int main(int argc, char **argv) {

  int c;
  int filename_set = 0;
  char filename[100];


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
        fprintf(stdout, "# [test_dinv] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_dinv] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_dinv] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_dinv] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_dinv] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /*********************************
   * set up geometry fields
   *********************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_dinv] Error from init_geometry\n");
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

  little_group_type *lg = NULL;
  int nlg = 0;

  if ( ( nlg = little_group_read_list ( &lg, "little_groups_2Oh.tab") ) <= 0 )
  {
    fprintf(stderr, "[test_dinv] Error from little_group_read_list, status was %d\n", nlg);
    EXIT(2);
  }
  fprintf(stdout, "# [test_dinv] number of little groups = %d\n", nlg);

  little_group_show ( lg, stdout, nlg );

  
  /******************************************************************/
  /******************************************************************/

  /******************************************************************
   * loop on little groups
   ******************************************************************/

  for ( int ilg = 0; ilg < nlg; ilg++ )
  {


    fprintf( stdout, "\n\n# [test_dinv] ===========================================================================================\n");
    fprintf( stdout, "# [test_dinv] lg %8s d %2d %2d %2d\n", lg[ilg].name , lg[ilg].d[0], lg[ilg].d[1], lg[ilg].d[2] );

    for ( int irot = 0; irot < lg[ilg].nr; irot++ )
    {

      fprintf( stdout, "\n# [test_dinv] lg %8s rot %2d\n", lg[ilg].name , lg[ilg].r[irot]+1 );

      double _Complex **U = rot_init_rotation_matrix ( 3 );
      rot_rotation_matrix_spherical_basis ( U, 2, cubic_group_double_cover_rotations[lg[ilg].r[irot]].n, cubic_group_double_cover_rotations[lg[ilg].r[irot]].w );


      double _Complex **R = rot_init_rotation_matrix ( 3 );
      rot_spherical2cartesian_3x3 ( R, U );

      fprintf ( stdout, "# [test_dinv] R is real int = %d\n", rot_mat_check_is_real_int ( R, 3 ) );


      int p[3];

      rot_point ( p, lg[ilg].d, R );

      double norm = sqrt( 
          _SQR( p[0] - lg[ilg].d[0] ) + 
          _SQR( p[1] - lg[ilg].d[1] ) + 
          _SQR( p[2] - lg[ilg].d[2] ) );

      rot_printf_matrix ( R, 3, "R", stdout );

      fprintf( stdout, "# [test_dinv] lg %8s Rd %2d %2d %2d\n", lg[ilg].name , p[0], p[1], p[2] );

      fprintf( stdout, "# [test_dinv] lg %8s r %2d norm diff %16.7e\n", lg[ilg].name , lg[ilg].r[irot]+1, norm );

      rot_fini_rotation_matrix ( &U );
      rot_fini_rotation_matrix ( &R );

    }  /* end of loop on rotations */

    for ( int irot = 0; irot < lg[ilg].nrm; irot++ )
    {

      fprintf( stdout, "\n# [test_dinv] lg %8s Irot %2d\n", lg[ilg].name , lg[ilg].rm[irot]+1 );

      double _Complex **U = rot_init_rotation_matrix ( 3 );
      rot_rotation_matrix_spherical_basis ( U, 2, cubic_group_double_cover_rotations[lg[ilg].rm[irot]].n, cubic_group_double_cover_rotations[lg[ilg].rm[irot]].w );


      double _Complex **R = rot_init_rotation_matrix ( 3 );
      rot_spherical2cartesian_3x3 ( R, U );

      fprintf ( stdout, "# [test_dinv] IR is real int = %d\n", rot_mat_check_is_real_int ( R, 3 ) );


      int p[3];

      rot_point ( p, lg[ilg].d, R );

      double norm = sqrt( 
          _SQR( p[0] + lg[ilg].d[0] ) + 
          _SQR( p[1] + lg[ilg].d[1] ) + 
          _SQR( p[2] + lg[ilg].d[2] ) );

      rot_printf_matrix ( R, 3, "R", stdout );

      fprintf( stdout, "# [test_dinv] lg %8s IRd %2d %2d %2d\n", lg[ilg].name , p[0], p[1], p[2] );

      fprintf( stdout, "# [test_dinv] lg %8s r %2d norm diff %16.7e\n", lg[ilg].name , lg[ilg].rm[irot]+1, norm );

      rot_fini_rotation_matrix ( &U );
      rot_fini_rotation_matrix ( &R );

    }  /* end of loop on rotations */
  }  /* end of loop on little groups */


  /***********************************************************
   * finalize
   ***********************************************************/

  little_group_fini ( &lg, nlg );


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_dinv] %s# [test_dinv] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_dinv] %s# [test_dinv] end of run\n", ctime(&g_the_time));
  }

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
