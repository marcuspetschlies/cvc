/****************************************************
 * projection_matrix_piN
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

#include "iblas.h"
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

  double const eps = 1.e-14;

  char const operator_side [2][16] = { "annihilation", "creation" };

  char const cartesian_vector_name[3][2] = {"x", "y", "z"};
  char const bispinor_name[4][2] = { "0", "1", "2", "3" };


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
        fprintf ( stdout, "# [projection_matrix_piN] using Reference frame rotation no. %d\n", refframerot );
        break;
      case 'h':
      case '?':
      default:
        fprintf(stdout, "# [projection_matrix_piN] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [projection_matrix_piN] Reading input from file %s\n", filename);
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
 if(g_cart_id == 0) fprintf(stdout, "# [projection_matrix_piN] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [projection_matrix_piN] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[projection_matrix_piN] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /****************************************************/
  /****************************************************/

  /****************************************************
   * initialize and set geometry fields
   ****************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0) {
    fprintf(stderr, "[projection_matrix_piN] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
    fprintf(stderr, "[projection_matrix_piN] Error from little_group_read_list, status was %d %s %d\n", nlg, __FILE__, __LINE__ );
    EXIT(2);
  }
  fprintf(stdout, "# [projection_matrix_piN] number of little groups = %d\n", nlg);

  little_group_show ( lg, stdout, nlg );

  /****************************************************/
  /****************************************************/

  /****************************************************
   * test subduction
   ****************************************************/
  little_group_projector_type p;

  if ( ( exitstatus = init_little_group_projector ( &p ) ) != 0 ) {
    fprintf ( stderr, "# [projection_matrix_piN] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  int const interpolator_number       = 2;           // one (for now imaginary) interpolator
  int const interpolator_bispinor[2]  = {0,1};         // bispinor no 0 / yes 1
  int const interpolator_parity[2]    = {-1,1};         // intrinsic operator parity, value 1 = intrinsic parity +1, -1 = intrinsic parity -1,
                                                     // value 0 = opposite parity not taken into account
  int const interpolator_cartesian[2] = {0,0};         // spherical basis (0) or cartesian basis (1) ? cartesian basis only meaningful for J = 1, J2 = 2, i.e. 3-dim. representation
  int const interpolator_J2[2]        = {0,1};
  char const correlator_name[]    = "pixN";

  int ** interpolator_momentum_list = init_2level_itable ( interpolator_number, 3 );
  if ( interpolator_momentum_list == NULL ) {
    fprintf ( stderr, "# [projection_matrix_piN] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(2);
  }

  /****************************************************/
  /****************************************************/

  /****************************************************
   * loop on little groups
   ****************************************************/
  // for ( int ilg = 0; ilg < nlg; ilg++ )
  for ( int ilg = 0; ilg < 1; ilg++ )
  {

    int const n_irrep = lg[ilg].nirrep;

    /****************************************************
     * get the total momentum given
     * d-vector and reference rotation
     ****************************************************/
    int Ptot[3] = { lg[ilg].d[0], lg[ilg].d[1], lg[ilg].d[2] };

    int momentum_number = 0;

    int * sink_momentum_id = get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, lg[ilg].d, g_sink_momentum_list, g_sink_momentum_number );

    for ( int i = 0; i < g_seq2_source_momentum_number; i++ ) {
      if ( sink_momentum_id[i] != -1 ) momentum_number++;
    }
    int ** momentum_list = init_2level_itable ( momentum_number, 3 );



    /****************************************************
     * complete the interpolator momentum list
     * by using momentum conservation and 
     * total momentum stored in d
     ****************************************************/

    interpolator_momentum_list[0][0] = lg[ilg].d[0];
    interpolator_momentum_list[0][1] = lg[ilg].d[1];
    interpolator_momentum_list[0][2] = lg[ilg].d[2];

    interpolator_momentum_list[1][0] = lg[ilg].d[0];
    interpolator_momentum_list[1][1] = lg[ilg].d[1];
    interpolator_momentum_list[1][2] = lg[ilg].d[2];

    if ( refframerot > -1 ) {
      double _Complex ** refframerot_p = rot_init_rotation_matrix ( 3 );
      if ( refframerot_p == NULL ) {
        fprintf(stderr, "[projection_matrix_piN] Error rot_init_rotation_matrix %s %d\n", __FILE__, __LINE__);
        EXIT(10);
      }

#if defined CUBIC_GROUP_DOUBLE_COVER
      rot_mat_spin1_cartesian ( refframerot_p, cubic_group_double_cover_rotations[refframerot].n, cubic_group_double_cover_rotations[refframerot].w );
#elif defined CUBIC_GROUP_SINGLE_COVER
      rot_rotation_matrix_spherical_basis_Wigner_D ( refframerot_p, 2, cubic_group_rotations_v2[refframerot].a );
      rot_spherical2cartesian_3x3 ( refframerot_p, refframerot_p );
#endif
      if ( ! ( rot_mat_check_is_real_int ( refframerot_p, 3) ) ) {
        fprintf(stderr, "[projection_matrix_piN] Error rot_mat_check_is_real_int refframerot_p %s %d\n", __FILE__, __LINE__);
        EXIT(72);
      }
      rot_point ( Ptot, Ptot, refframerot_p );
      rot_fini_rotation_matrix ( &refframerot_p );
      if ( g_verbose > 2 ) fprintf ( stdout, "# [projection_matrix_piN] Ptot = %3d %3d %3d   R[%2d] ---> Ptot = %3d %3d %3d\n",
          lg[ilg].d[0], lg[ilg].d[1], lg[ilg].d[2],
          refframerot, Ptot[0], Ptot[1], Ptot[2] );
    }

    /****************************************************
     * loop on irreps
     *   within little group
     ****************************************************/
    for ( int i_irrep = 0; i_irrep < n_irrep; i_irrep++ )
    /* for ( int i_irrep = 5; i_irrep < 6; i_irrep++ ) */
    {

      /****************************************************
       * rotation matrix for current irrep
       ****************************************************/
      rot_mat_table_type r_irrep;
      init_rot_mat_table ( &r_irrep );

      exitstatus = set_rot_mat_table ( &r_irrep, lg[ilg].name, lg[ilg].lirrep[i_irrep] );

      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[projection_matrix_piN] Error from set_rot_mat_table_cubic_group, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      int const irrep_dim  = r_irrep.dim;
      int const spinor_dim = ( 1 + interpolator_bispinor[0] ) * ( interpolator_J2[0] + 1 ) * ( 1 + interpolator_bispinor[1] ) * ( interpolator_J2[1] + 1 ); 

      double _Complex **** projection_matrix = init_4level_ztable ( momentum_number, spinor_dim, momentum_number, spinor_dim );  /* annihilation and creation operator */
      if ( projection_matrix == NULL ) {
        fprintf ( stderr, "[projection_matrix_piN] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
        EXIT(2);
      }

      /****************************************************
       * output file
       ****************************************************/
      sprintf ( filename, "lg_%s_irrep_%s_J2_%d_%d_Rref%.2d.sbd",
      lg[ilg].name, lg[ilg].lirrep[i_irrep], interpolator_J2[0], interpolator_J2[1], refframerot );

      FILE*ofs = fopen ( filename, "w" );
      if ( ofs == NULL ) {
        fprintf ( stderr, "# [projection_matrix_piN] Error from fopen %s %d\n", __FILE__, __LINE__);
        EXIT(2);
      }

      /****************************************************
       * row of target irrep
       ****************************************************/
      int const row_target = -1;
      const int ref_row_target = -1;
      const int * ref_row_spin = NULL;
  
      exitstatus = little_group_projector_set ( &p, &(lg[ilg]), lg[ilg].lirrep[i_irrep], row_target, interpolator_number,
          interpolator_J2, (const int**)interpolator_momentum_list, interpolator_bispinor, interpolator_parity, interpolator_cartesian,
          ref_row_target , ref_row_spin, correlator_name, refframerot );

      if ( exitstatus != 0 ) {
        fprintf ( stderr, "# [projection_matrix_piN] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }
  
      /****************************************************/
      /****************************************************/
   
      exitstatus = little_group_projector_show ( &p, ofs , 1 );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "# [projection_matrix_piN] Error from little_group_projector_show, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      fprintf ( stdout, "# [projection_matrix_piN] spinor_dime = %d\n", spinor_dim  );
      fprintf ( stdout, "# [projection_matrix_piN] irrep_dim   = %d\n", irrep_dim );

      /* for ( int iac = 0; iac <= 1; iac++ )  */
      for ( int iac = 0; iac <= 0; iac++ ) 
      {

        /* for ( int ibeta = 0; ibeta < r_irrep.dim; ibeta++ ) */
        for ( int ibeta = 0; ibeta < 1; ibeta++ )
        {

          /* for ( int imu = 0; imu < r_irrep.dim; imu++ ) */
          for ( int imu = 0; imu < 1; imu++ ) 
          {

            for ( int ir = 0; ir < p.rtarget->n ; ir++ ) {

              /****************************************************
               * creation / annihilation operator
               ****************************************************/
              for ( int ip1 = 0; ip1 < momentum_number; ip1++ ) {
                int p1[3] = {
                  momentum_list[ip1]
                }
              for ( int k1 = 0; k1 < p.rspin[0].dim; k1++ ) {
              for ( int l1 = 0; l1  < p.rspin[1].dim; l1++ ) {

                int const kl1 = p.rspin[1].dim * k1 + l1;

                for ( int k2 = 0; k2 < p.rspin[0].dim; k2++ ) {
                for ( int l2 = 0; l2  < p.rspin[1].dim; l2++ ) {

                    int const kl2 = p.rspin[1].dim * k2 + l2;

                    /* add the proper rotation */
                    projection_matrix[iac][0][imu][ibeta][kl1][kl2] += iac == 0 ? \
                        /* annihilation */ \
                        conj( p.rspin[0].R[ir][k2][k1]  * p.rspin[1].R[ir][l2][l1] )  *        p.rtarget->R[ir][imu][ibeta] : \
                        /* creation  */ \
                              p.rspin[0].R[ir][k1][k2]  * p.rspin[1].R[ir][l1][l2]    * conj ( p.rtarget->R[ir][imu][ibeta]   );

                    /* add the rotation-inversion */
                    projection_matrix[iac][0][imu][ibeta][kl1][kl2] += iac == 0 ? \
                        /* annihilation */ \
                        p.parity[0] * p.parity[1] * conj( p.rspin[0].IR[ir][k2][k1] * p.rspin[1].IR[ir][l2][l1] ) *        p.rtarget->IR[ir][imu][ibeta] : \
                        /* creation */ \
                        p.parity[0] * p.parity[1] *       p.rspin[0].IR[ir][k1][k2] * p.rspin[1].IR[ir][l1][l2]   * conj ( p.rtarget->IR[ir][imu][ibeta]   );
                }}
              }}
          }  /* end of loop on rotations / rotation-inversions */

          /* normalize */
          rot_mat_ti_eq_re ( projection_matrix[iac][0][imu][ibeta], (double)p.rtarget->dim/(2.*p.rtarget->n), spinor_dim );

          int const rank = gs_onb_mat ( projection_matrix[iac][1][imu][ibeta], projection_matrix[iac][2][imu][ibeta], projection_matrix[iac][0][imu][ibeta], spinor_dim, spinor_dim );

          fprintf ( stdout, "\n\n# [projection_matrix_piN] lg %20s irrep %20s beta %2d mu %2d rank %d\n", lg[ilg].name, lg[ilg].lirrep[i_irrep], ibeta, imu, rank );

          fprintf( stdout, "\n" );
          for ( int ir = 0; ir < spinor_dim; ir++ ) {
          for ( int is = 0; is < spinor_dim; is++ ) {
            // if ( cabs( projection_matrix[iac][0][imu][ibeta][ir][is] ) > eps  ) {
              fprintf (stdout, "%s P %2d %2d    %2d %2d    %2d %2d    %25.16e %25.16e\n", operator_side[iac], ir/p.rspin[1].dim, ir%p.rspin[1].dim, is/p.rspin[1].dim, is%p.rspin[1].dim, ir, is,
                  __dgeps( creal( projection_matrix[iac][0][imu][ibeta][ir][is] ), eps  ), __dgeps( cimag( projection_matrix[iac][0][imu][ibeta][ir][is] ), eps ) );
            //}
          }}

          fprintf( stdout, "\n" );
          for ( int ir = 0; ir < rank; ir++ ) {
          for ( int is = 0; is < spinor_dim; is++ ) {
            // if ( cabs( projection_matrix[iac][1][imu][ibeta][ir][is] ) > eps  ) {
              fprintf (stdout, "%s R %2d %2d    %2d %2d    %2d %2d    %25.16e %25.16e\n", operator_side[iac], ir/p.rspin[1].dim, ir%p.rspin[1].dim, is/p.rspin[1].dim, is%p.rspin[1].dim, ir, is,
                  __dgeps( creal( projection_matrix[iac][1][imu][ibeta][ir][is] ), eps  ), __dgeps( cimag( projection_matrix[iac][1][imu][ibeta][ir][is] ), eps ) );
            //}
          }}

          fprintf( stdout, "\n" );
          for ( int ir = 0; ir < rank; ir++ ) {
          for ( int is = 0; is < spinor_dim; is++ ) {
            // if ( cabs( projection_matrix[iac][2][imu][ibeta][ir][is] ) > eps  ) {
              fprintf (stdout, "%s Q %2d %2d    %2d %2d    %2d %2d    %25.16e %25.16e\n", operator_side[iac], ir/p.rspin[1].dim, ir%p.rspin[1].dim, is/p.rspin[1].dim, is%p.rspin[1].dim, ir, is,
                  __dgeps( creal( projection_matrix[iac][2][imu][ibeta][ir][is] ), eps  ), __dgeps( cimag( projection_matrix[iac][2][imu][ibeta][ir][is] ), eps ) );
            //}
          }}

          fprintf (  stdout, "\n" );
          for ( int ir = 0; ir < rank; ir++ ) {
            fprintf( stdout, "operator %d = ", ir );
            for ( int is = 0; is < spinor_dim; is++ ) {
              double _Complex z = projection_matrix[iac][2][imu][ibeta][ir][is];
              if ( cabs ( z ) > eps ) {
                fprintf( stdout, "%s(%s,%s)[%e,%e] + ", correlator_name, cartesian_vector_name[is/p.rspin[1].dim], bispinor_name[is%p.rspin[1].dim], 
                    __dgeps ( creal( projection_matrix[iac][2][imu][ibeta][ir][is]), eps ) ,
                    __dgeps (  cimag( projection_matrix[iac][2][imu][ibeta][ir][is] ), eps ) );
              }
            }
            fprintf( stdout, "\n" );
          }

          fprintf( stdout, "\n\n" );

        }  /* end of loop on target irrep rows */
      }  /* end of loop on irrep matrix ref. rows */

    }  /* end of source / sink side */

      /****************************************************/
      /****************************************************/
  
      fini_little_group_projector ( &p );
  
      /****************************************************/
      /****************************************************/
 
      /****************************************************
       * close output file
       ****************************************************/
      fclose ( ofs );


      fini_rot_mat_table ( &r_irrep );

      fini_4level_ztable ( &projection_matrix );

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
    fprintf(stdout, "# [projection_matrix_piN] %s# [projection_matrix_piN] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [projection_matrix_piN] %s# [projection_matrix_piN] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif
  return(0);
}
