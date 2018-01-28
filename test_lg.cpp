/****************************************************
 * test_lg.cpp
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
#include "group_projection.h"


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
        fprintf(stdout, "# [test_lg] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_lg] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /****************************************************/
  /****************************************************/

  /****************************************************
   * initialize MPI parameters for cvc
   ****************************************************/
   mpi_init(argc, argv);

  /****************************************************/
  /****************************************************/

  /****************************************************
   * set number of openmp threads
   ****************************************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_lg] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_lg] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_lg] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /****************************************************/
  /****************************************************/

  /****************************************************
   *
   ****************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_lg] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /****************************************************/
  /****************************************************/

  /****************************************************
   * initialize RANLUX random number generator
   ****************************************************/
  rlxd_init( 2, g_seed );

  /****************************************************/
  /****************************************************/

  rot_init_rotation_table();

  /****************************************************/
  /****************************************************/

  little_group_type *lg = NULL;
  int nlg = 0;

  if ( ( nlg = little_group_read_list ( &lg, "little_groups_2Oh.tab") ) <= 0 )
  {
    fprintf(stderr, "[test_lg] Error from little_group_read_list, status was %d\n", nlg);
    EXIT(2);
  }
  fprintf(stdout, "# [test_lg] number of little groups = %d\n", nlg);

  little_group_show ( lg, stdout, nlg );

  /****************************************************/
  /****************************************************/

  /****************************************************
   * test subduction
   ****************************************************/
  little_group_projector_type p;

  if ( ( exitstatus = init_little_group_projector ( &p ) ) != 0 ) {
    fprintf ( stderr, "# [test_lg] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  int **interpolator_momentum_list = NULL;
  int interpolator_number   = 1;               /* one (for now imaginary) interpolator */
  int interpolator_J2       = 3;               /* 2 x J */
  int interpolator_bispinor = 0;               /* no need for bispinor now */
  char correlator_name[]    = "basis_vector";  /* we don't want a correlator here, just a basis vector*/

  exitstatus = init_2level_ibuffer ( &interpolator_momentum_list, interpolator_number, 3 );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "# [test_lg] Error from init_2level_ibuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  interpolator_momentum_list[0][0] = 0;
  interpolator_momentum_list[0][1] = 0;
  interpolator_momentum_list[0][2] = 0;


  /****************************************************/
  /****************************************************/

  /****************************************************
   * loop on little groups
   ****************************************************/
  for ( int ilg = 0; ilg <= 0; ilg++ ) {

    int n_irrep = lg[ilg].nirrep;

    for ( int i_irrep = 7; i_irrep <= 7; i_irrep++ ) {

      /****************************************************
       * rotation matrix for current irrep
       ****************************************************/
      rot_mat_table_type r_irrep;
      init_rot_mat_table ( &r_irrep );
      exitstatus = set_rot_mat_table_cubic_group_double_cover ( &r_irrep, lg[ilg].name, lg[ilg].lirrep[i_irrep] );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "# [test_lg] Error from set_rot_mat_table_cubic_group_double_cover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }


      /****************************************************
       * loop on reference rows of spin matrix
       ****************************************************/
      /* for ( int ref_row_spin = 0; ref_row_spin <= interpolator_J2; ref_row_spin++ ) { */
      int ref_row_spin = -1;

        int dim_irrep = r_irrep.dim;
  
        /****************************************************
         * loop on reference rows of irrep matrix
         ****************************************************/
        for ( int ref_row_target = 0; ref_row_target < dim_irrep; ref_row_target++ ) {

          /****************************************************
           * loop on irrep multiplet
           ****************************************************/
          /* for ( int row_target = 0; row_target < dim_irrep; row_target++ ) { */
          int row_target = -1;

            exitstatus = little_group_projector_set ( &p, &(lg[0]), lg[ilg].lirrep[i_irrep], row_target, interpolator_number,
                &interpolator_J2, interpolator_momentum_list, &interpolator_bispinor,
                ref_row_target , &ref_row_spin, correlator_name );

            if ( exitstatus != 0 ) {
              fprintf ( stderr, "# [test_lg] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }

            /****************************************************/
            /****************************************************/
 
            exitstatus = little_group_projector_show ( &p, stdout , 0);
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "# [test_lg] Error from little_group_projector_show, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }

            /****************************************************/
            /****************************************************/
            exitstatus =  little_group_projector_apply ( &p );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "# [test_lg] Error from little_group_projector_apply, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }

            /****************************************************/
            /****************************************************/

            fini_little_group_projector ( &p );

          /* } */  /* end of loop on row_target */

        }  /* end of loop on ref_row_target */

      /* }  */ /* end of loop on ref_row_spin */


      fini_rot_mat_table ( &r_irrep );

    }  /* end of loop on irreps */

  }  /* end of loop on little groups */


  /****************************************************/
  /****************************************************/

  little_group_fini ( &lg, nlg );

  /****************************************************/
  /****************************************************/

  /****************************************************
   * finalize
   ****************************************************/
  free_geometry();

#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_lg] %s# [test_lg] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_lg] %s# [test_lg] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);
}
