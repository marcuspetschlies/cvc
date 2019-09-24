/****************************************************
 * test_lg
 *
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
#include "table_init_i.h"
#include "rotations.h"
#include "ranlxd.h"
#include "group_projection.h"
#include "little_group_projector_set.h"

#define _NORM_SQR_3D(_a) ( (_a)[0] * (_a)[0] + (_a)[1] * (_a)[1] + (_a)[2] * (_a)[2] )


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
  int refframerot = -1;  // no reference frame rotation


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:r:")) != -1) {
    switch (c) {
      case 'f':
        strcpy(filename, optarg);
        filename_set=1;
        break;
      case 'r':
        refframerot = atoi ( optarg );
        fprintf ( stdout, "# [test_lg] using Reference frame rotation no. %d\n", refframerot );
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
   * initialize and set geometry fields
   ****************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0) {
    fprintf(stderr, "[test_lg] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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

  /****************************************************
   * set cubic group single/double cover
   * rotation tables
   ****************************************************/
  rot_init_rotation_table();

  /****************************************************/
  /****************************************************/

  /****************************************************
   * read relevant little group lists with their
   * rotation lists and irreps from file
   ****************************************************/
  little_group_type *lg = NULL;
  int const nlg = little_group_read_list ( &lg, little_group_list_filename );
  if ( nlg <= 0 ) {
    fprintf(stderr, "[test_lg] Error from little_group_read_list, status was %d %s %d\n", nlg, __FILE__, __LINE__ );
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

#if 0
  int const interpolator_number       = 1;           // one (for now imaginary) interpolator
  int const interpolator_bispinor[1]  = {0};         // bispinor no 0 / yes 1
  int const interpolator_parity[1]    = {1};         // intrinsic operator parity, value 1 = intrinsic parity +1, -1 = intrinsic parity -1,
                                                     // value 0 = opposite parity not taken into account
  int const interpolator_cartesian[1] = {0};         // spherical basis (0) or cartesian basis (1) ? cartesian basis only meaningful for J = 1, J2 = 2, i.e. 3-dim. representation
  int const interpolator_J2[1]        = {3};
  char const correlator_name[]    = "basis_vector";  // just some arbitrary name for now

#endif  /* of if 0 */

  int const interpolator_number       = 2;               // one (for now imaginary) interpolator
  int const interpolator_bispinor[2]  = {0,0};           // no need for bispinor now
  int const interpolator_parity[2]    = {1,1};           // intrinsic operator parity
  int const interpolator_cartesian[2] = {1,1};           // spherical basis (0) or cartesian basis (1) ? cartesian basis only meaningful for J = 1, J2 = 2, i.e. 3-dim. representation
  char const correlator_name[]        = "basis_vector";  // just some arbitrary name for now
  int const interpolator_J2[2]        = {2,2};

  int ** interpolator_momentum_list = init_2level_itable ( interpolator_number, 3 );
  if ( interpolator_momentum_list == NULL ) {
    fprintf ( stderr, "# [test_lg] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(2);
  }

  /****************************************************
   * set the first momentum, the second follows
   * form momentum conservation
   ****************************************************/
/*
  interpolator_momentum_list[0][0] = 0;
  interpolator_momentum_list[0][1] = 0;
  interpolator_momentum_list[0][2] = 1;
*/

  /****************************************************/
  /****************************************************/

  /****************************************************
   * loop on little groups
   ****************************************************/
  /* for ( int ilg = 0; ilg < nlg; ilg++ ) */
  for ( int ilg = 0; ilg <= 0; ilg++ )
  {

    int const n_irrep = lg[ilg].nirrep;

    /* if ( ( strcmp( lg[ilg].name, "Oh" ) == 0 ) || ( strcmp( lg[ilg].name, "2Oh" ) == 0 ) )  {
      interpolator_parity = 0;
    } else {
      interpolator_parity = 1;
    }
     */

#if 0
    /****************************************************
     * complete the interpolator momentum list
     * by using momentum conservation and 
     * total momentum stored in d
     ****************************************************/
    if ( interpolator_number == 1 ) {

      interpolator_momentum_list[0][0] = lg[ilg].d[0];
      interpolator_momentum_list[0][1] = lg[ilg].d[1];
      interpolator_momentum_list[0][2] = lg[ilg].d[2];

    } else if ( interpolator_number == 2 ) {

      interpolator_momentum_list[1][0] = lg[ilg].d[0] - interpolator_momentum_list[0][0]; 
      interpolator_momentum_list[1][1] = lg[ilg].d[1] - interpolator_momentum_list[0][1];
      interpolator_momentum_list[1][2] = lg[ilg].d[2] - interpolator_momentum_list[0][2];

    }
#endif  /* end of if 0 */

    /****************************************************
     * get the total momentum given
     * d-vector and reference rotation
     ****************************************************/
    int Ptot[3] = { lg[ilg].d[0], lg[ilg].d[1], lg[ilg].d[2] };

    if ( refframerot > -1 ) {
      double _Complex ** refframerot_p = rot_init_rotation_matrix ( 3 );
      if ( refframerot_p == NULL ) {
        fprintf(stderr, "[test_lg] Error rot_init_rotation_matrix %s %d\n", __FILE__, __LINE__);
        EXIT(10);
      }

#if defined CUBIC_GROUP_DOUBLE_COVER
      rot_mat_spin1_cartesian ( refframerot_p, cubic_group_double_cover_rotations[refframerot].n, cubic_group_double_cover_rotations[refframerot].w );
#elif defined CUBIC_GROUP_SINGLE_COVER
      rot_rotation_matrix_spherical_basis_Wigner_D ( refframerot_p, 2, cubic_group_rotations_v2[refframerot].a );
      rot_spherical2cartesian_3x3 ( refframerot_p, refframerot_p );
#endif
      if ( ! ( rot_mat_check_is_real_int ( refframerot_p, 3) ) ) {
        fprintf(stderr, "[test_lg] Error rot_mat_check_is_real_int refframerot_p %s %d\n", __FILE__, __LINE__);
        EXIT(72);
      }
      rot_point ( Ptot, Ptot, refframerot_p );
      rot_fini_rotation_matrix ( &refframerot_p );
      if ( g_verbose > 2 ) fprintf ( stdout, "# [test_lg] Ptot = %3d %3d %3d   R[%2d] ---> Ptot = %3d %3d %3d\n",
          lg[ilg].d[0], lg[ilg].d[1], lg[ilg].d[2],
          refframerot, Ptot[0], Ptot[1], Ptot[2] );
    }

    /****************************************************
     * loop on irreps
     *   within little group
     ****************************************************/
    for ( int i_irrep = 0; i_irrep < n_irrep; i_irrep++ )
    /* for ( int i_irrep = 8; i_irrep <= 8; i_irrep++ ) */
    {

      /****************************************************
       * loop on reference rows of spin matrix
       ****************************************************/
      for ( int r1 = 0; r1 <= interpolator_J2[0]; r1++ ) {
        /* int const ref_row_spin[1] = { r1 }; */

      for ( int r2 = 0; r2 <= interpolator_J2[1]; r2++ ) {

        int const ref_row_spin[2] = { r1, r2 }; 

        /****************************************************
         * rotation matrix for current irrep
         ****************************************************/
        rot_mat_table_type r_irrep;
        init_rot_mat_table ( &r_irrep );

        exitstatus = set_rot_mat_table ( &r_irrep, lg[ilg].name, lg[ilg].lirrep[i_irrep] );

        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[test_lg] Error from set_rot_mat_table_cubic_group, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }

        int const dim_irrep = r_irrep.dim;
  
        /****************************************************
         * loop on reference rows of irrep matrix
         ****************************************************/
        for ( int ref_row_target = 0; ref_row_target < dim_irrep; ref_row_target++ ) {

          /****************************************************
           * loop on momentum configurations
           ****************************************************/

          for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
              
            if ( interpolator_number == 1 ) {
              interpolator_momentum_list[0][0] = Ptot[0];
              interpolator_momentum_list[0][1] = Ptot[1];
              interpolator_momentum_list[0][2] = Ptot[2];

            } else if ( interpolator_number == 2 ) {

              interpolator_momentum_list[0][0] = g_sink_momentum_list[imom][0];
              interpolator_momentum_list[0][1] = g_sink_momentum_list[imom][1];
              interpolator_momentum_list[0][2] = g_sink_momentum_list[imom][2];

              /* interpolator_momentum_list[1][0] = Ptot[0] - interpolator_momentum_list[0][0]; 
              interpolator_momentum_list[1][1] = Ptot[1] - interpolator_momentum_list[0][1];
              interpolator_momentum_list[1][2] = Ptot[2] - interpolator_momentum_list[0][2];
              */

              /* TEST */
              interpolator_momentum_list[1][0] = 0;
              interpolator_momentum_list[1][1] = 0;
              interpolator_momentum_list[1][2] = 0;
            }

            /****************************************************
             * cut for single interpolator momenta with norm
             * square > 3
             ****************************************************/
            if ( _NORM_SQR_3D( interpolator_momentum_list[0] ) > 3 ) {
              fprintf ( stdout, "# [test_lg] skipping p1 = (%d, %d, %d)\n",
                  interpolator_momentum_list[0][0], interpolator_momentum_list[0][1], interpolator_momentum_list[0][2] );
              continue;
            }

            if ( interpolator_number > 1 ) {
              if ( _NORM_SQR_3D( interpolator_momentum_list[1] ) > 3 ) {
                fprintf ( stdout, "# [test_lg] skipping p2 = (%d, %d, %d)\n",
                    interpolator_momentum_list[1][0], interpolator_momentum_list[1][1], interpolator_momentum_list[1][2] );
                continue;
              }
            }

            /****************************************************
             * momentum tag
             ****************************************************/
            char momentum_str[100];
            sprintf( momentum_str, ".p1x%d_p1y%d_p1z%d", interpolator_momentum_list[0][0], interpolator_momentum_list[0][1], interpolator_momentum_list[0][2] );
            if ( interpolator_number == 2 ) {
              sprintf( momentum_str, "%s.p2x%d_p2y%d_p2z%d", momentum_str, interpolator_momentum_list[1][0], interpolator_momentum_list[1][1], interpolator_momentum_list[1][2] );
            }

            /****************************************************
             * output file
             ****************************************************/
            if ( interpolator_number == 2 ) {
              sprintf ( filename, "lg_%s_irrep_%s_J2_%d_%d_spinref%d_%d_irrepref%d%s_Rref%.2d.sbd", 
                  lg[ilg].name, lg[ilg].lirrep[i_irrep], interpolator_J2[0], interpolator_J2[1], ref_row_spin[0], ref_row_spin[1], ref_row_target, momentum_str,
                  refframerot );
            } else if ( interpolator_number == 1 ) {
              sprintf ( filename, "lg_%s_irrep_%s_J2_%d_spinref%d_irrepref%d%s_Rref%.2d.sbd",
                  lg[ilg].name, lg[ilg].lirrep[i_irrep], interpolator_J2[0], ref_row_spin[0], ref_row_target , momentum_str,
                  refframerot );
            }

            FILE*ofs = fopen ( filename, "w" );
            if ( ofs == NULL ) {
              fprintf ( stderr, "# [test_lg] Error from fopen %s %d\n", __FILE__, __LINE__);
              EXIT(2);
            }

            /****************************************************
             * loop on irrep multiplet
             ****************************************************/
            /* for ( int row_target = 0; row_target < dim_irrep; row_target++ ) { */
            int const row_target = -1;
  
            /* exitstatus = little_group_projector_set ( &p, &(lg[ilg]), lg[ilg].lirrep[i_irrep], row_target, interpolator_number,
                  &interpolator_J2, (const int**)interpolator_momentum_list, &interpolator_bispinor, &interpolator_parity, &interpolator_cartesian,
                  ref_row_target , &ref_row_spin, correlator_name );
             */
  
              exitstatus = little_group_projector_set ( &p, &(lg[ilg]), lg[ilg].lirrep[i_irrep], row_target, interpolator_number,
                  interpolator_J2, (const int**)interpolator_momentum_list, interpolator_bispinor, interpolator_parity, interpolator_cartesian,
                  ref_row_target , ref_row_spin, correlator_name, refframerot );

              if ( exitstatus != 0 ) {
                fprintf ( stderr, "# [test_lg] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(2);
              }
  
              /****************************************************/
              /****************************************************/
   
              exitstatus = little_group_projector_show ( &p, ofs , 1 );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "# [test_lg] Error from little_group_projector_show, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(2);
              }
  

              /****************************************************
               * apply the projector
               ****************************************************/
              little_group_projector_applicator_type **app = little_group_projector_apply ( &p, ofs );
              if ( app == NULL ) {
                fprintf ( stderr, "# [test_lg] Error from little_group_projector_apply %s %d\n", __FILE__, __LINE__);
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
  
            /* } */  // end of loop on row_target
  
            /****************************************************/
            /****************************************************/
 
            /****************************************************
             * close output file
             ****************************************************/
            fclose ( ofs );

          }  /* end of loop on sink momenta */

        }  /* end of loop on ref_row_target */


        fini_rot_mat_table ( &r_irrep );

      }  /* end of loop on ref_row_spin2 */
      }  /* end of loop on ref_row_spin1 */

    }  /* end of loop on irreps */

  }  /* end of loop on little groups */


  /****************************************************/
  /****************************************************/

  for ( int i = 0; i < nlg; i++ ) {
    little_group_fini ( &(lg[i]) );
  }
  free ( lg );

  /****************************************************/
  /****************************************************/

  /****************************************************
   * finalize
   ****************************************************/
  fini_2level_itable ( &interpolator_momentum_list );
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
