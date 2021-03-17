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
#include "contract_diagrams.h"
#include "contractions_io.h"

#define _NORM_SQR_3D(_a) ( (_a)[0] * (_a)[0] + (_a)[1] * (_a)[1] + (_a)[2] * (_a)[2] )

#define _P3_EQ_P3(_p,_q) ( ( (_p)[0]==(_q)[0] ) && ( (_p)[1]==(_q)[1] ) && ( (_p)[2]==(_q)[2] ) )

using namespace cvc;

inline int get_momentum_id (int const p[3] , int ** const q, int const nq  ) {
  for ( int i = 0; i < nq; i++ ) {
    if ( _P3_EQ_P3(p, q[i] ) ) return ( i );
  }
  return ( -1 );
}  /* end of get_momentum_id */


int main(int argc, char **argv) {

  double const eps = 1.e-14;

  char const operator_side [2][16] = { "annihilation", "creation" };

  // char const cartesian_vector_name[3][2] = {"x", "y", "z"};
  // char const bispinor_name[4][2] = { "0", "1", "2", "3" };


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
  int write_projector = 1;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ph?f:r:")) != -1) {
    switch (c) {
      case 'f':
        strcpy(filename, optarg);
        filename_set=1;
        break;
      case 'r':
        refframerot = atoi ( optarg );
        fprintf ( stdout, "# [projection_matrix_piN] using Reference frame rotation no. %d\n", refframerot );
        break;
      case 'p':
        write_projector = 1;
        fprintf ( stdout, "# [projection_matrix_piN] write_projector set to %d\n", write_projector );
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

  int const interpolator_number       = 2;              // one (for now imaginary) interpolator
  int const interpolator_bispinor[2]  = {   0,   1 };   // bispinor no 0 / yes 1
  int const interpolator_parity[2]    = {  -1,   1 };   // intrinsic operator parity, value 1 = intrinsic parity +1, -1 = intrinsic parity -1,
                                                        // value 0 = opposite parity not taken into account
  int const interpolator_cartesian[2] = {   0,   0 };   // spherical basis (0) or cartesian basis (1) ? cartesian basis only meaningful for J = 1, J2 = 2, i.e. 3-dim. representation
  int const interpolator_J2[2]        = {   0 ,  1 };   // 2 * spin

  char const interpolator_name[2][12]  = { "pi", "N"};  // name used in operator listing
  char const interpolator_tex_name[2][2][12]  = { { "\\pi", "\\pi^\\dagger"}, {"N", "\\bar{N}" } };  // name used in operator listing

  char const correlator_name[]    = "pixN";

  int ** interpolator_momentum_list = init_2level_itable ( interpolator_number, 3 );
  if ( interpolator_momentum_list == NULL ) {
    fprintf ( stderr, "[projection_matrix_piN] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(2);
  }

  /****************************************************
   * output filename for information on subduction
   ****************************************************/

  char subduction_output_filename[200];
  sprintf( subduction_output_filename, "subduction.%s-%s.h5", interpolator_name[0], interpolator_name[1] );

  /****************************************************
   * loop on little groups
   ****************************************************/
  for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ )
  /* for ( int ilg = 0; ilg < nlg; ilg++ ) */
  {

    int const Ptot[3] = {
      g_total_momentum_list[iptot][0],
      g_total_momentum_list[iptot][1],
      g_total_momentum_list[iptot][2] };

    /****************************************************
     * get reference momentum vector for Ptot and
     * the reference frame rotation number
     ****************************************************/
    int Pref[3];

    exitstatus = get_reference_rotation ( Pref, &refframerot, Ptot );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[projection_matrix_D] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(10);
    }

    /****************************************************
     * no rotation = -1 ---> 0 = identity
     ****************************************************/
    if ( refframerot == -1 ) refframerot = 0;

    int ilg;
    for ( ilg=0; ilg <nlg; ilg++ ) {
      if (
          ( lg[ilg].d[0] == Pref[0] ) &&
          ( lg[ilg].d[1] == Pref[1] ) &&
          ( lg[ilg].d[2] == Pref[2] )  ) {
        break;
      }
    }
    if ( ilg == nlg ) {
      fprintf(stderr, "[projection_matrix_D] Error, could not match Pref to lg %s %d\n", __FILE__, __LINE__);
      EXIT(10);
    } else {
      fprintf ( stdout, "# [projection_matrix_D] Ptot %3d %3d %3d   Pref %3d %3d %3d R %2d  lg %s   %s %d\n",
         Ptot[0], Ptot[1], Ptot[2],
         Pref[0], Pref[1], Pref[2],
         refframerot+1, lg[ilg].name, __FILE__, __LINE__);
    }

    /****************************************************
     * complete the interpolator momentum list
     * by using momentum conservation and 
     * total momentum stored in Ptot
     *
     * HERE THIS IS JUST TO FILL IN SOMETHING FOR
     * interpolator_momentum_list; momentum_list itself
     * is used later on
     ****************************************************/
    interpolator_momentum_list[0][0] = momentum_list[0][0];
    interpolator_momentum_list[0][1] = momentum_list[0][1];
    interpolator_momentum_list[0][2] = momentum_list[0][2];

    interpolator_momentum_list[1][0] = -momentum_list[0][0] + Ptot[0];
    interpolator_momentum_list[1][1] = -momentum_list[0][0] + Ptot[1];
    interpolator_momentum_list[1][2] = -momentum_list[0][0] + Ptot[2];

    /****************************************************
     * loop on irreps
     *   within little group
     ****************************************************/
    for ( int i_irrep = 0; i_irrep < lg[ilg].nirrep; i_irrep++ )
    /* for ( int i_irrep = 5; i_irrep <= 5; i_irrep++ ) */
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

      int const irrep_dim    = r_irrep.dim;
      int const spinor_dim   = ( 1 + interpolator_bispinor[0] ) * ( interpolator_J2[0] + 1 ) * ( 1 + interpolator_bispinor[1] ) * ( interpolator_J2[1] + 1 ); 

      fprintf ( stdout, "# [projection_matrix_piN] spinor_dime = %d\n", spinor_dim  );
      fprintf ( stdout, "# [projection_matrix_piN] irrep_dim   = %d\n", irrep_dim );

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
   
      if ( write_projector ) {

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

        exitstatus = little_group_projector_show ( &p, ofs , 1 );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "# [projection_matrix_piN] Error from little_group_projector_show, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }

        /****************************************************
         * close output file
         ****************************************************/
        fclose ( ofs );

      }

      /****************************************************
       * loop on creation / annihilation operator
       ****************************************************/
      for ( int iac = 0; iac <= 1; iac++ )
      {

        /****************************************************
         * determine list and number of sink momenta 
         * (for N operator)
         * the list seq2_source_momentum is given and 
         * we list all from sink_momentum with 
         * sink_momentum + seq_source_momentum = d 
         * in momentum_list
         ***************************************************/
        int momentum_number = 0;

        int * sink_momentum_id = get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, Ptot, g_sink_momentum_list, g_sink_momentum_number );
        if ( sink_momentum_id == NULL ) {
          fprintf ( stderr, "[projection_matrix_piN] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__);
          EXIT(2);
        }

        for ( int i = 0; i < g_seq2_source_momentum_number; i++ ) {
          if ( sink_momentum_id[i] != -1 ) momentum_number++;
        }
        int ** momentum_list = init_2level_itable ( momentum_number, 3 );
        for ( int k = 0, i = 0; i < g_seq2_source_momentum_number; i++ ) {
          if ( sink_momentum_id[i] != -1 ) {
              memcpy ( momentum_list[k], g_sink_momentum_list[ sink_momentum_id[i] ], 3*sizeof( int ) );
              k++;
          }
        }
        if ( g_verbose > 2 ) {
          fprintf( stdout, "# [projection_matrix_piN] momentum_number = %d\n", momentum_number );
          for ( int i = 0; i < momentum_number; i++ ) {
            fprintf( stdout, "  %2d    %3d %3d %3d\n", i, momentum_list[i][0], momentum_list[i][1], momentum_list[i][2] );
         } 
        }

        int const mom_ncdim = 2;
        int const mom_cdim[2] = { momentum_number, 3 };
        char mom_tag[400];
        sprintf ( mom_tag, "pi1" );

        sprintf( tag_prefix, "/%s/%s/PX%d_PY%d_PZ%d/%s/pi1",
            operator_side[iac],
            lg[ilg].name, Ptot[0], Ptot[1], Ptot[2], lg[ilg].lirrep[i_irrep] );

        exitstatus = write_h5_contraction ( momentum_list[0], NULL,  subduction_output_filename, mom_tag, "int", mom_ncdim, mom_cdim );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[projection_matrix_piN] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }

        /****************************************************
         * build the momentum id lists for
         * proper rotation and rotation-inversions
         ****************************************************/
  
        int *** rotated_momentum_id = init_3level_itable ( momentum_number, p.rtarget->n, 2 );
        if ( rotated_momentum_id== NULL ) {
          fprintf ( stderr, "# [projection_matrix_piN] Error from init_Xlevel_itable %s %d\n", __FILE__, __LINE__);
          EXIT(2);
        }
  
        /****************************************************
         * loop on proper- / inversion- rotations
         ****************************************************/
        for ( int ir = 0; ir < p.rtarget->n ; ir++ ) {
  
          /****************************************************
           * loop on particle momenta
           ****************************************************/
          for ( int imom = 0; imom < momentum_number; imom++ ) {
  
            /****************************************************
             * set the particle momentum vectors
             ****************************************************/
            int p1[3] = { momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2] };
  
            /****************************************************
             * determine the proper-rotated and inversion-rotated
             * particle momentum vectors
             ****************************************************/
            int rp1[3], irp1[3];
  
            rot_point ( rp1, p1, p.rp->R[ir] );
            rot_point ( irp1, p1, p.rp->IR[ir] );
  
            /****************************************************
             * find them in the list momenta to calculate
             * the correct matrix index
             ****************************************************/
  
            int const imom_r  =  get_momentum_id ( rp1 , momentum_list, momentum_number );
            int const imom_ir =  get_momentum_id ( irp1 , momentum_list, momentum_number );
  
            if ( imom_r == -1 || imom_ir == -1 ) {
              fprintf ( stderr, "[projection_matrix_piN] Error from get_momentum_id mom idx %d %d %s %d\n", imom_r, imom_ir, __FILE__, __LINE__ );
              EXIT(12);
            } else if ( g_verbose > 2 ) {
  
              fprintf( stdout, "# [projection_matrix_piN]  mom %d (%3d,%3d,%3d) rotated %d  (%3d,%3d,%3d) and %d (%3d,%3d,%3d) %s %d\n",
                     imom,    momentum_list[imom   ][0], momentum_list[imom   ][1], momentum_list[imom   ][2],
                     imom_r,  momentum_list[imom_r ][0], momentum_list[imom_r ][1], momentum_list[imom_r ][2],
                     imom_ir, momentum_list[imom_ir][0], momentum_list[imom_ir][1], momentum_list[imom_ir][2] , __FILE__, __LINE__ );
  
              rotated_momentum_id[imom][ir][0] = imom_r;
              rotated_momentum_id[imom][ir][1] = imom_ir;
            }
          }
        }

        /****************************************************
         * loop on irrep matrix ref. row
         ****************************************************/
        // for ( int ibeta = 0; ibeta < r_irrep.dim; ibeta++ )
        for ( int ibeta = 0; ibeta < 1; ibeta++ )
        {

          int const matrix_dim = spinor_dim * momentum_number;
          if ( g_verbose > 2 ) fprintf ( stdout, "# [projection_matrix_piN] matrix_dim = %d %s %d\n", matrix_dim, __FILE__, __LINE__);

          double _Complex *** projection_matrix_v = init_3level_ztable ( r_irrep.dim, matrix_dim, matrix_dim );  /* annihilation and creation operator */
          if ( projection_matrix_v == NULL ) {
            fprintf ( stderr, "[projection_matrix_piN] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
            EXIT(2);
          }

          double _Complex *** projection_matrix_s = init_3level_ztable ( r_irrep.dim, matrix_dim, matrix_dim );  /* for Gram-Schmidt decomposition, rotation matrix */
          if ( projection_matrix_s == NULL ) {
            fprintf ( stderr, "[projection_matrix_piN] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
            EXIT(2);
          }

          double _Complex *** projection_matrix_u = init_3level_ztable ( r_irrep.dim, matrix_dim, matrix_dim );  /* for Gram-Schmidt decomposition, operators coefficients */
          if ( projection_matrix_u == NULL ) {
            fprintf ( stderr, "[projection_matrix_piN] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
            EXIT(2);
          }

          int rank = -1;

          /****************************************************
           * loop on irrep row
           ****************************************************/
          for ( int imu = 0; imu < r_irrep.dim; imu++ )
          {

            /****************************************************
             * loop on proper- / inversion- rotations 
             ****************************************************/
            for ( int ir = 0; ir < p.rtarget->n ; ir++ ) {

              /****************************************************
               * loop on particle momenta
               ****************************************************/
              for ( int imom = 0; imom < momentum_number; imom++ ) {
              
                /****************************************************
                 * loop on particle-spin rotation matrix elements
                 ****************************************************/

                /****************************************************
                 * this is the projected side index
                 * k1,l1,p
                 ****************************************************/
                for ( int k1 = 0; k1 < p.rspin[0].dim; k1++ ) {
                for ( int l1 = 0; l1 < p.rspin[1].dim; l1++ ) {

                  int const kl1  = p.rspin[1].dim * k1 + l1;
                  int const idx_projected = imom * spinor_dim + kl1;

                  for ( int k2 = 0; k2 < p.rspin[0].dim; k2++ ) {
                  for ( int l2 = 0; l2 < p.rspin[1].dim; l2++ ) {

                    int const kl2 = p.rspin[1].dim * k2 + l2;
                    int const idx_bare_pr = rotated_momentum_id[imom][ir][0] * spinor_dim + kl2;
                    int const idx_bare_ir = rotated_momentum_id[imom][ir][1] * spinor_dim + kl2;

                    /* add the proper rotation */
                    projection_matrix_v[imu][idx_projected][idx_bare_pr]  += ( iac == 0 ) ? \
                        /* annihilation; means S(R^-1)_MM' = S(R)_M'M^* */ \
                                                    conj( p.rspin[0].R[ir][k2][k1]  * p.rspin[1].R[ir][l2][l1] )  *        p.rtarget->R[ir][imu][ibeta] : \
                        /* creation; means S(R)_M'M  */ \
                                                          p.rspin[0].R[ir][k2][k1]  * p.rspin[1].R[ir][l2][l1]    * conj ( p.rtarget->R[ir][imu][ibeta]   );

                    /* add the rotation-inversion */
                    projection_matrix_v[imu][idx_projected][idx_bare_ir] += ( iac == 0 ) ? \
                        /* annihilation */ \
                        p.parity[0] * p.parity[1] * conj( p.rspin[0].IR[ir][k2][k1] * p.rspin[1].IR[ir][l2][l1] ) *        p.rtarget->IR[ir][imu][ibeta] : \
                        /* creation */ \
                        p.parity[0] * p.parity[1] *       p.rspin[0].IR[ir][k2][k1] * p.rspin[1].IR[ir][l2][l1]   * conj ( p.rtarget->IR[ir][imu][ibeta]   );

                  }}
                }}
              }  /* end of loop on momenta */
            }  /* end of loop on rotations / rotation-inversions */

            /* normalize */
            rot_mat_ti_eq_re ( projection_matrix_v[imu], (double)p.rtarget->dim/(2.*p.rtarget->n), matrix_dim );


            char tag_prefix[400];

            sprintf( tag_prefix, "/%s/%s/PX%d_PY%d_PZ%d/%s/row%d/J2_%d/bispinor_%d/J2_%d/bispinor_%d/refrow%d",
                operator_side[iac],
                lg[ilg].name, Ptot[0], Ptot[1], Ptot[2], lg[ilg].lirrep[i_irrep], imu,
                interpolator_J2[0], interpolator_bispinor[0], interpolator_J2[1], interpolator_bispinor[1], ibeta );

           if ( g_verbose > 2 ) fprintf ( stdout, "# [projection_matrix_piN] processing %s/row%d %s %d\n", tag_prefix, imu, __FILE__, __LINE__ );
           

            int const new_rank = gs_onb_mat ( projection_matrix_s[imu], projection_matrix_u[imu], projection_matrix_v[imu], matrix_dim, matrix_dim );

            if ( rank == -1 ) rank = new_rank;

            if ( rank != new_rank ) {
              fprintf( stderr, "[projection_matrix_piN] Error, %s row %d has rank %d different from %d %s %d\n", tag_prefix, imu, rank, new_rank,  __FILE__, __LINE__ );
              EXIT(14);
            }

            if ( rank == 0 ) {
              fprintf( stdout, "# [projection_matrix_piN] %s rank is zero; continue %s %d\n", tag_prefix, __FILE__, __LINE__ );
              continue;
            }

            
            /****************************************************
             * write coefficient matrices to hdf5 file
             * 
             ****************************************************/
            int const dim[2] = { matrix_dim, matrix_dim };
            char tag[500];
            
            sprintf( tag, "%s/v", tag_prefix );

            exitstatus = write_h5_contraction ( projection_matrix_v[imu][0], NULL, subduction_output_filename, tag, "double", 2, dim );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[projection_matrix_piN] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }

            sprintf( tag, "%s/s", tag_prefix );
            
            exitstatus = write_h5_contraction ( projection_matrix_s[imu][0], NULL, subduction_output_filename, tag, "double", 2, dim );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[projection_matrix_piN] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }

            sprintf( tag, "%s/u", tag_prefix );
            
            exitstatus = write_h5_contraction ( projection_matrix_u[imu][0], NULL, subduction_output_filename, tag, "double", 2, dim );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[projection_matrix_piN] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }

            /****************************************************
             * print the operator in mixed 
             * text + coefficient form
             ****************************************************/
       
            sprintf ( filename, "lg_%s.rref%d.irrep_%s.row_%d.refrow_%d.j2_%d_%d.ac_%d.opr.tex",
                lg[ilg].name, refframerot,  lg[ilg].lirrep[i_irrep], imu, ibeta,
                interpolator_J2[0], interpolator_J2[1], iac );

            FILE * ofs2 = fopen ( filename, "w" );
            if ( ofs2 == NULL ) {
              fprintf ( stderr, "[projection_matrix_piN] Error from fopen %s %d\n", __FILE__, __LINE__);
              EXIT(2);
            }

            fprintf( ofs2, " $LG = %s$, $\\vec{P}_{\\mathrm{tot}} = (%d,\\,%d,\\,%d)$, $\\Lambda = %s$, $\\lambda = %d$, $J = %d/2(%d) \\oplus %d/2(%d)$, $\\beta_{\\mathrm{ref}} = %d$\n",
                lg[ilg].name, Ptot[0], Ptot[1], Ptot[2], lg[ilg].lirrep[i_irrep], imu,
                interpolator_J2[0], interpolator_bispinor[0], interpolator_J2[1], interpolator_bispinor[1], ibeta );


            for ( int ir = 0; ir < rank; ir++ )
            {
            
              fprintf ( ofs2, "\n\\begin{align}\n" );

              if ( iac == 0 ) {
                fprintf( ofs2, "O_{%d} &= \n%%\\label{}\n\\\\\n", ir+1);
              } else {
                fprintf( ofs2, "\\bar{O}_{%d} &= \n%%\\label{}\n\\\\\n", ir+1);
              }
              for ( int is = 0; is < matrix_dim; is++ ) {

                int const imom = is / spinor_dim;
                int const p1[3] = {
                    momentum_list[imom][0],
                    momentum_list[imom][1],
                    momentum_list[imom][2] };

                int const p2[3] = {
                    Ptot[0] - momentum_list[imom][0],
                    Ptot[1] - momentum_list[imom][1],
                    Ptot[2] - momentum_list[imom][2] };

                int const ispin[2] = {
                    ( is % spinor_dim ) / p.rspin[1].dim,
                    ( is % spinor_dim ) % p.rspin[1].dim };

                double _Complex z = projection_matrix_u[imu][ir][is];
                if ( cabs ( z ) > eps ) {

                  /* fprintf( ofs2, "  %s(%d,%d,%d) %d %s(%d,%d,%d) %d [%e,%e]  +  ", 
                      interpolator_name[0], p1[0], p1[1], p1[2], ispin[0],
                      interpolator_name[1], p2[0], p2[1], p2[2], ispin[1],
                      __dgeps ( creal(z), eps ), __dgeps ( cimag(z), eps ) ); */

                  fprintf( ofs2, " &\\quad + %s_{%d}\\left(%d,%d,%d \\right) \\, %s_{%d}\\left(%d,%d,%d\\right)\\, \\left[%+8.7f  %+8.7f\\,i\\right]  \\nonumber \\\\\n", 
                      interpolator_tex_name[0][iac], ispin[0],
                      p1[0], p1[1], p1[2],
                      interpolator_tex_name[1][iac], ispin[1],
                      p2[0], p2[1], p2[2],
                      __dgeps ( creal(z), eps ), __dgeps ( cimag(z), eps ) );
                }
              }  /* end of loop on matrix dimension */
            
              fprintf ( ofs2, "& \\nonumber\n\\end{align}\n\n" );

            }  /* end of loop on rank = loop on operators */

            fclose ( ofs2 );
#if 0
#endif

          }  /* end of loop on target irrep rows */

          /****************************************************
           * check irrep multiplett rotation
           ****************************************************/

          if ( rank == 0 ) {
            fprintf( stdout, "# [projection_matrix_piN] rank is zero, no test %s %d\n", __FILE__, __LINE__ );
            continue;
          }
#if 0
          exitstatus = check_subduction_matrix_multiplett_rotation ( projection_matrix_v, matrix_dim, p, operator_side[iac], momentum_number, rotated_momentum_id );
#endif
          exitstatus = check_subduction_matrix_multiplett_rotation ( projection_matrix_u, rank, p, operator_side[iac], momentum_number, rotated_momentum_id );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[projection_matrix_piN] Error from check_subduction_matrix_multiplett_rotation, status was %d %s %d\n",
               exitstatus,  __FILE__, __LINE__ );
            EXIT(290);
          }

           
          fini_3level_ztable ( &projection_matrix_v );
          fini_3level_ztable ( &projection_matrix_s );
          fini_3level_ztable ( &projection_matrix_u );

        }  /* end of loop on irrep matrix ref. rows */

        fini_2level_itable ( &momentum_list );
        if ( sink_momentum_id != NULL ) free( sink_momentum_id );

        fini_3level_itable ( &rotated_momentum_id );

      }  /* end of source / sink side */

      /****************************************************/
      /****************************************************/


      fini_little_group_projector ( &p );
  
      /****************************************************/
      /****************************************************/
 
      fini_rot_mat_table ( &r_irrep );

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
