/****************************************************
 * projection_matrix_D
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
#include "contractions_io.h"

#define _NORM_SQR_3D(_a) ( (_a)[0] * (_a)[0] + (_a)[1] * (_a)[1] + (_a)[2] * (_a)[2] )


using namespace cvc;

char const operator_side[2][16] = { "annihilation", "creation" };

#if 0
/****************************************************
 * check irrep multiplett-rotation property
 * for subduction matrix v
 ****************************************************/
int check_multiplett_rotation ( double _Complex *** const v , int const rank, little_group_projector_type const p , int const ac) {

  double const eps = 1.e-12;

  int const matrix_dim = p.rspin[0].dim * p.rspin[1].dim;

  int any_wrong = 0;

  char tag[800];
  sprintf( tag, "/%s/%s/refframerot%d/%s/%s/%s/refrow%d", operator_side[ac], p.rtarget->group, p.refframerot, p.rtarget->irrep, p.rspin[0].irrep, p.rspin[1].irrep, p.ref_row_target );
  
  /****************************************************
   * matrices for rotated v
   ****************************************************/
  double _Complex ** A = init_2level_ztable ( rank, matrix_dim );
  double _Complex ** B = init_2level_ztable ( rank, matrix_dim );

  /****************************************************
   * loop on multiplet members
   ****************************************************/
  for ( int imu = 0; imu < p.rtarget->dim; imu++ ) {

    for ( int irot = 0; irot < 2*p.rtarget->n; irot++ )  {
  
      double _Complex ** const r0 = irot < p.rtarget->n ? p.rspin[0].R[irot] : p.rspin[0].IR[irot % p.rtarget->n ];
      double _Complex ** const r1 = irot < p.rtarget->n ? p.rspin[1].R[irot] : p.rspin[1].IR[irot % p.rtarget->n ];
      double _Complex ** const tt = irot < p.rtarget->n ? p.rtarget->R[irot] : p.rtarget->IR[irot % p.rtarget->n ];

      double const parity = irot < p.rtarget->n ? 1. : p.parity[0] * p.parity[1];

      if ( ac == 1 ) {
  
        /****************************************************
         * multiplication A_ik <- V_il S(R)_kl
         ****************************************************/
        for (int i = 0; i < rank; i++ ) {
        for (int k = 0; k < matrix_dim; k++ ) {
          double _Complex z = 0.;
          for ( int l = 0; l < matrix_dim; l++ ) {
            // z += v[imu][i][l] * p.rspin[0].R[irot][k/p.rspin[1].dim][l/p.rspin[1].dim] * p.rspin[1].R[irot][k%p.rspin[1].dim][l%p.rspin[1].dim];
            z += v[imu][i][l] * r0[k/p.rspin[1].dim][l/p.rspin[1].dim] * r1[k%p.rspin[1].dim][l%p.rspin[1].dim];
          }
          A[i][k] = z * parity;
        }}
  
        /****************************************************
         * multiplication B_ik <- V^gamma'_ik  T(R)^gamma',gamma
         ****************************************************/
        for (int i = 0; i < rank; i++ ) {
        for (int k = 0; k < matrix_dim; k++ ) {
          double _Complex z = 0.;
  
          for ( int igamma = 0; igamma < p.rtarget->dim; igamma++ ) {
            // z += p.rtarget->R[irot][igamma][imu] * v[igamma][i][k];
            z += tt[igamma][imu] * v[igamma][i][k];
          }
          B[i][k] = z;
        }}
      } else if ( ac == 0 ) {

        /****************************************************
         * multiplication A_ik <- V_il S(R)_kl
         ****************************************************/
        for (int i = 0; i < rank; i++ ) {
        for (int k = 0; k < matrix_dim; k++ ) {
          double _Complex z = 0.;
          for ( int l = 0; l < matrix_dim; l++ ) {
            z += v[imu][i][l] * conj( r0[k/p.rspin[1].dim][l/p.rspin[1].dim] * r1[k%p.rspin[1].dim][l%p.rspin[1].dim] );
          }
          A[i][k] = z * parity;
        }}
  
        /****************************************************
         * multiplication B_ik <- V^gamma'_ik  T(R)^gamma',gamma
         ****************************************************/
        for (int i = 0; i < rank; i++ ) {
        for (int k = 0; k < matrix_dim; k++ ) {
          double _Complex z = 0.;
  
          for ( int igamma = 0; igamma < p.rtarget->dim; igamma++ ) {
            z += conj ( tt[igamma][imu] ) * v[igamma][i][k];
          }
          B[i][k] = z;
        }}

      }

      /****************************************************
       * compare
       ****************************************************/
      double gnorm = 0., gdiff = 0.;
      for (int i = 0; i < rank; i++ ) {
      for (int k = 0; k < matrix_dim; k++ ) {
        double const diff = cabs( A[i][k] - B[i][k] );
        double const avrg = cabs( A[i][k] + B[i][k] ) * 0.5;
        if ( g_verbose > 5 ) fprintf( stdout, "multiplett %d   %2d   %2d %2d       %25.16e %25.16e      %25.16e %25.16e      %e   %e \n", 
            imu, irot, i, k, creal( A[i][k] ), cimag( A[i][k] ), creal( B[i][k] ), cimag( B[i][k] ), diff , avrg );
            gnorm += avrg;
            gdiff += diff;
      }}
      char const * const state = ( gnorm > eps && gdiff/gnorm > eps  ) ? any_wrong=1, "wrong" : "okay";
      fprintf( stdout, "# [check_multiplett_rotation] %s/row%d/rot%d normdiff %e   %e  %s\n", tag, imu, irot, gdiff , gnorm, state );

  
    }  /* end of loop on rotations */

  }  /* end of loop on multiplet members */

  /****************************************************
   * deallocate temporary matrices
   ****************************************************/
  fini_2level_ztable ( &A );
  fini_2level_ztable ( &B );

  return ( any_wrong );

}  /* end of check_multiplett_rotation  */
#endif

/****************************************************/
/****************************************************/


int main(int argc, char **argv) {

  double const eps = 1.e-12;

  char const cartesian_vector_name[3][2] = {"x", "y", "z"};
  char const bispinor_name[4][2] = { "0", "1", "2", "3" };


#if defined CUBIC_GROUP_DOUBLE_COVER
  char const little_group_list_filename[] = "little_groups_2Oh.tab";
  int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_double_cover;
#else
#warning "[projection_matrix_D] need CUBIC_GROUP_DOUBLE_COVER"
   EXIT(1);
#endif


  int c;
  int filename_set = 0;
  char filename[500];
  int exitstatus;
  int refframerot = 0;  // no reference frame rotation = identity, i.e. not any reference frame rotation


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
        fprintf ( stdout, "# [projection_matrix_D] using Reference frame rotation no. %d\n", refframerot );
        break;
      case 'h':
      case '?':
      default:
        fprintf(stdout, "# [projection_matrix_D] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [projection_matrix_D] Reading input from file %s\n", filename);
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
 if(g_cart_id == 0) fprintf(stdout, "# [projection_matrix_D] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [projection_matrix_D] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[projection_matrix_D] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /****************************************************/
  /****************************************************/

  /****************************************************
   * initialize and set geometry fields
   ****************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0) {
    fprintf(stderr, "[projection_matrix_D] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
    fprintf(stderr, "[projection_matrix_D] Error from little_group_read_list, status was %d %s %d\n", nlg, __FILE__, __LINE__ );
    EXIT(2);
  }
  fprintf(stdout, "# [projection_matrix_D] number of little groups = %d\n", nlg);

  little_group_show ( lg, stdout, nlg );

  /****************************************************/
  /****************************************************/

  /****************************************************
   * test subduction
   ****************************************************/
  little_group_projector_type p;

  if ( ( exitstatus = init_little_group_projector ( &p ) ) != 0 ) {
    fprintf ( stderr, "# [projection_matrix_D] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  int const interpolator_number       = 2;           // one (for now imaginary) interpolator
  int const interpolator_bispinor[2]  = {0,1};         // bispinor no 0 / yes 1
  int const interpolator_parity[2]    = {-1,1};         // intrinsic operator parity, value 1 = intrinsic parity +1, -1 = intrinsic parity -1,
                                                     // value 0 = opposite parity not taken into account
  int const interpolator_cartesian[2] = {1,0};         // spherical basis (0) or cartesian basis (1) ? cartesian basis only meaningful for J = 1, J2 = 2, i.e. 3-dim. representation
  int const interpolator_J2[2]        = {2,1};
  
  char const interpolator_name[2][12]  = { "Delta", ""};  // name used in operator listing
  char const interpolator_tex_name[2][2][20]  = { { "\\Delta", "\\bar{\\Delta}" }, { "", "" } };  // name used in operator listing

  char const correlator_name[]    = "D-D";

  int ** interpolator_momentum_list = init_2level_itable ( interpolator_number, 3 );
  if ( interpolator_momentum_list == NULL ) {
    fprintf ( stderr, "# [projection_matrix_D] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(2);
  }

  /****************************************************/
  /****************************************************/

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
     * total momentum stored in d
     ****************************************************/

    interpolator_momentum_list[0][0] = lg[ilg].d[0];
    interpolator_momentum_list[0][1] = lg[ilg].d[1];
    interpolator_momentum_list[0][2] = lg[ilg].d[2];

    interpolator_momentum_list[1][0] = lg[ilg].d[0];
    interpolator_momentum_list[1][1] = lg[ilg].d[1];
    interpolator_momentum_list[1][2] = lg[ilg].d[2];

#if 0
    /****************************************************
     * get the total momentum given
     * d-vector and reference rotation
     ****************************************************/
    int Ptot[3] = { lg[ilg].d[0], lg[ilg].d[1], lg[ilg].d[2] };

    if ( refframerot > -1 ) {
      double _Complex ** refframerot_p = rot_init_rotation_matrix ( 3 );
      if ( refframerot_p == NULL ) {
        fprintf(stderr, "[projection_matrix_D] Error rot_init_rotation_matrix %s %d\n", __FILE__, __LINE__);
        EXIT(10);
      }

#if defined CUBIC_GROUP_DOUBLE_COVER
      rot_mat_spin1_cartesian ( refframerot_p, cubic_group_double_cover_rotations[refframerot].n, cubic_group_double_cover_rotations[refframerot].w );
#endif
      if ( ! ( rot_mat_check_is_real_int ( refframerot_p, 3) ) ) {
        fprintf(stderr, "[projection_matrix_D] Error rot_mat_check_is_real_int refframerot_p %s %d\n", __FILE__, __LINE__);
        EXIT(72);
      }
      rot_point ( Ptot, Ptot, refframerot_p );
      rot_fini_rotation_matrix ( &refframerot_p );
      if ( g_verbose > 2 ) fprintf ( stdout, "# [projection_matrix_D] Ptot = %3d %3d %3d   R[%2d] ---> Ptot = %3d %3d %3d\n",
          lg[ilg].d[0], lg[ilg].d[1], lg[ilg].d[2],
          refframerot, Ptot[0], Ptot[1], Ptot[2] );
    }
#endif

    /****************************************************
     * loop on irreps
     *   within little group
     ****************************************************/
    for ( int i_irrep = 0; i_irrep < lg[ilg].nirrep; i_irrep++ )
    // for ( int i_irrep = 7; i_irrep < 8; i_irrep++ )
    {

      /****************************************************
       * rotation matrix for current irrep
       ****************************************************/
      rot_mat_table_type r_irrep;
      init_rot_mat_table ( &r_irrep );

      exitstatus = set_rot_mat_table ( &r_irrep, lg[ilg].name, lg[ilg].lirrep[i_irrep] );

      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[projection_matrix_D] Error from set_rot_mat_table_cubic_group, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      int const irrep_dim  = r_irrep.dim;
      int const spinor_dim = ( 1 + interpolator_bispinor[0] ) * ( interpolator_J2[0] + 1 ) * ( 1 + interpolator_bispinor[1] ) * ( interpolator_J2[1] + 1 ); 

      int const matrix_dim = spinor_dim;

      /****************************************************
       * momentum tag
       ****************************************************/
      char momentum_str[100];
      sprintf( momentum_str, ".px%d_py%d_pz%d", Ptot[0], Ptot[1], Ptot[2] );

      /****************************************************
       * output file
       ****************************************************/
      sprintf ( filename, "lg_%s_irrep_%s_J2_%d_%d_%s_Rref%.2d.sbd",
      lg[ilg].name, lg[ilg].lirrep[i_irrep], interpolator_J2[0], interpolator_J2[1], momentum_str, refframerot );

      FILE*ofs = fopen ( filename, "w" );
      if ( ofs == NULL ) {
        fprintf ( stderr, "# [projection_matrix_D] Error from fopen %s %d\n", __FILE__, __LINE__);
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
        fprintf ( stderr, "# [projection_matrix_D] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }
  
      /****************************************************/
      /****************************************************/
   
      exitstatus = little_group_projector_show ( &p, ofs , 1 );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "# [projection_matrix_D] Error from little_group_projector_show, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      fprintf ( stdout, "# [projection_matrix_D] spinor_dime = %d\n", spinor_dim  );
      fprintf ( stdout, "# [projection_matrix_D] irrep_dim   = %d\n", irrep_dim );

      for ( int iac = 0; iac <= 1; iac++  )
      {

        for ( int ibeta = 0; ibeta < r_irrep.dim; ibeta++ )
        {

          p.ref_row_target = ibeta;

          /****************************************************
           * allocate subduction matrices / 
           * projection coefficient matrices
           *
           * with GS we will have
           *
           * s v = u
           *
           * v = original matrix
           * s = (Gs) lower triangle transformation matrix, rank = # operators
           * u = (GS) basis matrix, rank = # operators
           ****************************************************/

          double _Complex *** projection_matrix_v = init_3level_ztable ( irrep_dim, matrix_dim, matrix_dim );  /* annihilation and creation operator */
          if ( projection_matrix_v == NULL ) {
            fprintf ( stderr, "[projection_matrix_D] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
            EXIT(2);
          }

          double _Complex *** projection_matrix_s = init_3level_ztable ( irrep_dim, matrix_dim, matrix_dim );  /* for Gram-Schmidt decomposition */
          if ( projection_matrix_s == NULL ) {
            fprintf ( stderr, "[projection_matrix_D] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
            EXIT(2);
          }
          double _Complex *** projection_matrix_u = init_3level_ztable ( irrep_dim, matrix_dim, matrix_dim );  /* for Gram-Schmidt decomposition */
          if ( projection_matrix_u == NULL ) {
            fprintf ( stderr, "[projection_matrix_D] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
            EXIT(2);
          }

          int rank = -1;

          /****************************************************
           * loop on irrep multiplet members
           ****************************************************/
          for ( int imu = 0; imu < r_irrep.dim; imu++ )
          {

            for ( int ir = 0; ir < p.rtarget->n ; ir++ ) {

              /****************************************************
               * creation / annihilation operator
               ****************************************************/
              for ( int k1 = 0; k1 < p.rspin[0].dim; k1++ ) {
              for ( int l1 = 0; l1  < p.rspin[1].dim; l1++ ) {

                int const kl1 = p.rspin[1].dim * k1 + l1;

                for ( int k2 = 0; k2 < p.rspin[0].dim; k2++ ) {
                for ( int l2 = 0; l2  < p.rspin[1].dim; l2++ ) {

                    int const kl2 = p.rspin[1].dim * k2 + l2;

                    /* add the proper rotation */
                    projection_matrix_v[imu][kl1][kl2] += iac == 0 ? \
                        /* annihilation */ \
                        conj( p.rspin[0].R[ir][k2][k1]  * p.rspin[1].R[ir][l2][l1] )  *        p.rtarget->R[ir][imu][ibeta] : \
                        /* creation  */ \
                              p.rspin[0].R[ir][k2][k1]  * p.rspin[1].R[ir][l2][l1]    * conj ( p.rtarget->R[ir][imu][ibeta]   );

                    /* add the rotation-inversion */
                    projection_matrix_v[imu][kl1][kl2] += iac == 0 ? \
                        /* annihilation */ \
                        p.parity[0] * p.parity[1] * conj( p.rspin[0].IR[ir][k2][k1] * p.rspin[1].IR[ir][l2][l1] ) *        p.rtarget->IR[ir][imu][ibeta] : \
                        /* creation */ \
                        p.parity[0] * p.parity[1] *       p.rspin[0].IR[ir][k2][k1] * p.rspin[1].IR[ir][l2][l1]   * conj ( p.rtarget->IR[ir][imu][ibeta]   );
                }}
              }}
            }  /* end of loop on rotations / rotation-inversions */

            /* normalize */
            rot_mat_ti_eq_re ( projection_matrix_v[imu], (double)p.rtarget->dim/(2.*p.rtarget->n), matrix_dim );

            /****************************************************
             * output tag prefix
             ****************************************************/
            char tag_prefix[400];
  
            sprintf( tag_prefix, "/%s/%s/PX%d_PY%d_PZ%d/%s/row%d/J2_%d/bispinor_%d/J2_%d/bispinor_%d/refrow%d",
                operator_side[iac],
                lg[ilg].name, Ptot[0], Ptot[1], Ptot[2], lg[ilg].lirrep[i_irrep], imu,
                interpolator_J2[0], interpolator_bispinor[0], interpolator_J2[1], interpolator_bispinor[1], ibeta );

            /****************************************************
             * coefficients coefficients for operator onb from GS
             ****************************************************/
            if ( rank == -1 ) {
              rank = gs_onb_mat ( projection_matrix_s[imu], projection_matrix_u[imu], projection_matrix_v[imu], matrix_dim, matrix_dim );
            } else {
              int const new_rank = gs_onb_mat ( projection_matrix_s[imu], projection_matrix_u[imu], projection_matrix_v[imu], matrix_dim, matrix_dim );
              if ( rank != new_rank ) {
                fprintf( stderr, "[projection_matrix_D] Error, %s row %d has rank %d different from %d %s %d\n", tag_prefix, imu, rank, new_rank,  __FILE__, __LINE__ );
                EXIT(14);
              }
            }

            if ( rank == 0 ) {
              fprintf( stdout, "# [projection_matrix_D] %s rank is zero; continue %s %d\n", tag_prefix, __FILE__, __LINE__ );
              continue;
            }

            /****************************************************/
            /****************************************************/

            /****************************************************
             * write coefficient matrices to hdf5 file
             *
             ****************************************************/
            int dim[2] = { matrix_dim, 2 * matrix_dim };
            char tag[500];
  
            sprintf( filename, "subduction.%s.h5", interpolator_name[0] );
  
            sprintf( tag, "%s/v", tag_prefix );
  
            exitstatus = write_h5_contraction ( (double*)(projection_matrix_v[imu][0]), NULL, filename, tag, "double", 2, dim );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[projection_matrix_D] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }
  
            sprintf( tag, "%s/s", tag_prefix );
  
            exitstatus = write_h5_contraction ( (double*)(projection_matrix_s[imu][0]), NULL, filename, tag, "double", 2, dim );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[projection_matrix_D] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }
  
            sprintf( tag, "%s/u", tag_prefix );
  
            dim[0] = rank;
            exitstatus = write_h5_contraction ( (double*)(projection_matrix_u[imu][0]), NULL, filename, tag, "double", 2, dim );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[projection_matrix_D] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }
  
            /****************************************************/
            /****************************************************/
  
            /****************************************************
             * print the operator in mixed
             * text + coefficient form
             ****************************************************/
  
            sprintf ( filename, "lg_%s.rref%d.irrep_%s.row_%d.refrow_%d.j2_%d_%d.ac_%d.opr.tex",
                lg[ilg].name, refframerot,  lg[ilg].lirrep[i_irrep], imu, ibeta,
                interpolator_J2[0], interpolator_J2[1], iac );
  
            FILE * ofs2 = fopen ( filename, "w" );
            if ( ofs2 == NULL ) {
              fprintf ( stderr, "[projection_matrix_D] Error from fopen %s %d\n", __FILE__, __LINE__);
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
  
                int const ispin[2] = { is / p.rspin[1].dim, is % p.rspin[1].dim };
  
                double _Complex z = projection_matrix_u[imu][ir][is];
                if ( cabs ( z ) > eps ) {
  
                  fprintf( ofs2, " &\\quad + %s_{%s,\\,%s}\\left(%d,%d,%d \\right) \\, \\left[%+8.7f  %+8.7f\\,i\\right]  \\nonumber \\\\\n",
                      interpolator_tex_name[0][iac], cartesian_vector_name[ispin[0]], bispinor_name[ispin[1]],
                      Ptot[0], Ptot[1], Ptot[2],
                      __dgeps ( creal(z), eps ), __dgeps ( cimag(z), eps ) );
                }
              }  /* end of loop on matrix dimension */
  
              fprintf ( ofs2, "& \\nonumber\n\\end{align}\n\n" );
  
            }  /* end of loop on rank = loop on operators */
  
            fclose ( ofs2 );

          }  /* end of loop on target irrep rows */

          /****************************************************
           * test irrep multiplet rotation property
           ****************************************************/
          if ( rank > 0 ) {

            /* exitstatus = check_subduction_matrix_multiplett_rotation ( projection_matrix_v , matrix_dim, p , operator_side[iac], 1, NULL );
            if ( exitstatus != 0 ) {
              fprintf( stderr, "[projection_matrix_D] Error from check_subduction_matrix_multiplett_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(12);
            } */

            exitstatus = check_subduction_matrix_multiplett_rotation ( projection_matrix_u , rank, p , operator_side[iac], 1, NULL );
            if ( exitstatus != 0 ) {
              fprintf( stderr, "[projection_matrix_D] Error from check_subduction_matrix_multiplett_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(12);
            }

          }


          /****************************************************
           * deallocate matrices
           ****************************************************/
          fini_3level_ztable ( &projection_matrix_v );
          fini_3level_ztable ( &projection_matrix_s );
          fini_3level_ztable ( &projection_matrix_u );

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

    }  /* end of loop on irreps */

  }  /* end of loop on total momenta */


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
    fprintf(stdout, "# [projection_matrix_D] %s# [projection_matrix_D] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [projection_matrix_D] %s# [projection_matrix_D] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif
  return(0);
}
