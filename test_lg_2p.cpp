/****************************************************
 * test_lg_2p.cpp
 *
 * Do 1. Feb 16:55:25 CET 2018
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
#include "table_init.h"
#include "rotations.h"
#include "ranlxd.h"
#include "group_projection.h"


using namespace cvc;

int main(int argc, char **argv) {

#if defined CUBIC_GROUP_DOUBLE_COVER
  char const little_group_list_filename[] = "little_groups_2Oh.tab";
  int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_double_cover;
#elif defined CUBIC_GROUP_SINGLE_COVER
  const char little_group_list_filename[] = "little_groups_Oh.tab";
  int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_single_cover;
#endif


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
        fprintf(stdout, "# [test_lg_2p] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_lg_2p] Reading input from file %s\n", filename);
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
 if(g_cart_id == 0) fprintf(stdout, "# [test_lg_2p] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_lg_2p] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_lg_2p] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /****************************************************/
  /****************************************************/

  /****************************************************
   *
   ****************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0 ) {
    fprintf(stderr, "[test_lg_2p] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
  const int nlg = little_group_read_list ( &lg, little_groups_list_filename );

  if ( nlg <= 0 ) {
    fprintf(stderr, "[test_lg_2p] Error from little_group_read_list, status was %d %s %d\n", nlg, __FILE__, __LINE__);
    EXIT(2);
  }
  fprintf(stdout, "# [test_lg_2p] number of little groups = %d\n", nlg);

  little_group_show ( lg, stdout, nlg );

  /****************************************************/
  /****************************************************/

  /****************************************************
   * test subduction
   ****************************************************/
  little_group_projector_type p;

  if ( ( exitstatus = init_little_group_projector ( &p ) ) != 0 ) {
    fprintf ( stderr, "# [test_lg_2p] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  int interpolator_number = 2;
  char correlator_name[]  = "basis_vector";

  int **interpolator_momentum_list = init_2level_itable ( interpolator_number, 3 );
  if ( interpolator_momentum_list == NULL ) {
    fprintf ( stderr, "# [test_lg_2p] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(2);
  }

  int *interpolator_bispinor = init_1level_itable ( interpolator_number  );
  if ( interpolator_bispinor == NULL ) {
    fprintf ( stderr, "# [test_lg_2p] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(2);
  }

  int * interpolator_parity = init_1level_itable ( interpolator_number  );
  if ( interpolator_parity == NULL ) {
    fprintf ( stderr, "# [test_lg_2p] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(2);
  }

  int * interpolator_J2 = init_1level_itable ( interpolator_number  );
  if ( interpolator_J2 == NULL ) {
    fprintf ( stderr, "# [test_lg_2p] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(2);
  }

  int * ref_row_spin = init_1level_itable ( interpolator_number  );
  if ( ref_row_spin == NULL ) {
    fprintf ( stderr, "# [test_lg_2p] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(2);
  }

  /****************************************************/
  /****************************************************/

  /****************************************************
   * set some values for testing
   ****************************************************/
  interpolator_J2[0] = 2;
  interpolator_J2[1] = 1;

  interpolator_bispinor[0] = 0;
  interpolator_bispinor[1] = 1;

  for ( int i = 0; i < interpolator_number; i++ ) {
    interpolator_momentum_list[i][0] = 0;
    interpolator_momentum_list[i][1] = 0;
    interpolator_momentum_list[i][2] = 0;

    interpolator_parity[i]   = 0;
  }

  /****************************************************/
  /****************************************************/

  /****************************************************
   * loop on little groups
   ****************************************************/
  for ( int ilg = 0; ilg < nlg; ilg++ )
  // for ( int ilg = 3; ilg <= 3; ilg++ )
  {

    int n_irrep = lg[ilg].nirrep;

    /****************************************************
     * loop on irreps
     ****************************************************/
    for ( int i_irrep = 0; i_irrep < n_irrep; i_irrep++ )
    // for ( int i_irrep = 5; i_irrep <= 5; i_irrep++ )
    {

      /****************************************************
       * rotation matrix for current irrep
       ****************************************************/
      rot_mat_table_type r_irrep;
      init_rot_mat_table ( &r_irrep );
      exitstatus = set_rot_mat_table ( &r_irrep, lg[ilg].name, lg[ilg].lirrep[i_irrep] );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "# [test_lg_2p] Error from set_rot_mat_table_cubic_group_double_cover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      /****************************************************
       * loop on reference rows of irrep matrix
       ****************************************************/
      for ( int ref_row_target = 0; ref_row_target < r_irrep.dim; ref_row_target++ )
      // for ( int ref_row_target = 0; ref_row_target <= 0; ref_row_target++ )
      {
  
        /****************************************************
         * loop on reference rows of spin matrix
         ****************************************************/
        // for ( ref_row_spin[0] = 0; ref_row_spin[0] <= interpolator_J2[0]; ref_row_spin[0]++ )
        // {
        // for ( ref_row_spin[1] = 0; ref_row_spin[1] <= interpolator_J2[1]; ref_row_spin[1]++ )

        ref_row_spin[0] = -3;
        for ( ref_row_spin[1] = 0; ref_row_spin[1] >= ref_row_spin[0]; ref_row_spin[1]-- )
        {

          /****************************************************
           * loop on spin quantum numbers
           ****************************************************/
          sprintf ( filename, "lg_%s_irrep_%s_J2_%d_%d_sref_%d_%d_tref_%d.sbd", lg[ilg].name, lg[ilg].lirrep[i_irrep],
              interpolator_J2[0], interpolator_J2[1], -ref_row_spin[0], -ref_row_spin[0] + 2*ref_row_spin[1], ref_row_target );

          FILE*ofs = fopen ( filename, "w" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "# [test_lg_2p] Error from fopen %s %d\n", __FILE__, __LINE__);
            EXIT(2);
          }

  
          exitstatus = little_group_projector_set ( &p, &(lg[ilg]), lg[ilg].lirrep[i_irrep], -1, interpolator_number,
              interpolator_J2, interpolator_momentum_list, interpolator_bispinor, interpolator_parity, ref_row_target, ref_row_spin, correlator_name );
  
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "# [test_lg_2p] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
  
          /****************************************************/
          /****************************************************/
   
          exitstatus = little_group_projector_show ( &p, ofs , 1 );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "# [test_lg_2p] Error from little_group_projector_show, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
  
          /****************************************************/
          /****************************************************/
          exitstatus =  
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "# [test_lg_2p] Error from little_group_projector_apply, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
  
          little_group_projector_applicator_type **app = little_group_projector_apply_product ( &p, ofs );
          if ( app == NULL ) {
            fprintf ( stderr, "# [test_lg_2p] Error from little_group_projector_apply_product %s %d\n", __FILE__, __LINE__);
            EXIT(2);
          }

          /****************************************************/
          /****************************************************/
  
          /****************************************************
           * finalize applicators
           ****************************************************/

          for ( int irow = 0; irow < dim_irrep; irow++ ) {
            free ( fini_little_group_projector_applicator ( app[irow] ) );
          }
          free ( app );

          /****************************************************/
          /****************************************************/


          fini_little_group_projector ( &p );

          fclose ( ofs );
  
        }  /* end of loop on ref_row_spin 1 */
        // }  /* end of loop on ref_row_spin 0 */
  

      }  // end of loop on ref_row_target

      fini_rot_mat_table ( &r_irrep );

    }  // end of loop on irreps

  }  // end of loop on little groups


  /****************************************************/
  /****************************************************/


  fini_2level_itable ( &interpolator_momentum_list );

  fini_1level_itable ( &interpolator_bispinor );

  fini_1level_itable ( &interpolator_parity );

  fini_1level_itable ( &interpolator_J2 );

  fini_1level_itable ( &ref_row_spin );

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
    fprintf(stdout, "# [test_lg_2p] %s# [test_lg_2p] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_lg_2p] %s# [test_lg_2p] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);
}
