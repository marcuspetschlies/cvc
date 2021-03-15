/****************************************************
 * projection_matrix_basis_vectors
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
#include "table_init_d.h"
#include "rotations.h"
#include "ranlxd.h"
#include "group_projection.h"
#include "little_group_projector_set.h"
#include "contractions_io.h"

#include "clebsch_gordan.h"

#define _NORM_SQR_3D(_a) ( (_a)[0] * (_a)[0] + (_a)[1] * (_a)[1] + (_a)[2] * (_a)[2] )

using namespace cvc;

char * spin_tag ( char * const r, int const j2 ) {
  if ( j2 % 2 ) {
    sprintf ( r, "%d/2", j2 );
  } else {
    sprintf ( r, "%d", j2/2 );
  }
  return( r );
}  /* spin_tag */

char * spin_irrep_name ( char * const r, int const j2 ) {
  if ( j2 % 2 ) {
    sprintf ( r, "spin%d_2", j2 );
  } else {
    sprintf ( r, "spin%d", j2/2 );
  }
  printf ( "# [spin_irrep_name] j2 = %d  r = %s\n", j2 , r);
  return( r );
}  /* spin_irrep_name */


char const operator_side[2][16] = { "annihilation", "creation" };

/****************************************************/
/****************************************************/


int main(int argc, char **argv) {

  double const eps = 1.e-12;

  // char const cartesian_vector_name[3][2] = {"x", "y", "z"};
  // char const bispinor_name[4][2] = { "0", "1", "2", "3" };


#if defined CUBIC_GROUP_DOUBLE_COVER
  char const little_group_list_filename[] = "little_groups_2Oh.tab";
  int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_double_cover;
#else
#warning "[projection_matrix_basis_vectors] need CUBIC_GROUP_DOUBLE_COVER"
   EXIT(1);
#endif


  int c;
  int filename_set = 0;
  char filename[500];
  int exitstatus;
  int refframerot = 0;  // no reference frame rotation = identity, i.e. not any reference frame rotation
  int write_projector = 0;

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
        fprintf ( stdout, "# [projection_matrix_basis_vectors] using Reference frame rotation no. %d\n", refframerot );
        break;
      case 'p':
        write_projector = 1;
        fprintf ( stdout, "# [projection_matrix_basis_vectors] write_projector set to %d\n", write_projector );
        break;
      case 'h':
      case '?':
      default:
        fprintf(stdout, "# [projection_matrix_basis_vectors] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [projection_matrix_basis_vectors] Reading input from file %s\n", filename);
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
  if(g_cart_id == 0) fprintf(stdout, "# [projection_matrix_basis_vectors] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [projection_matrix_basis_vectors] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[projection_matrix_basis_vectors] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /****************************************************/
  /****************************************************/

  /****************************************************
   * initialize and set geometry fields
   ****************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0) {
    fprintf(stderr, "[projection_matrix_basis_vectors] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
    fprintf(stderr, "[projection_matrix_basis_vectors] Error from little_group_read_list, status was %d %s %d\n", nlg, __FILE__, __LINE__ );
    EXIT(2);
  }
  fprintf(stdout, "# [projection_matrix_basis_vectors] number of little groups = %d\n", nlg);

  little_group_show ( lg, stdout, nlg );

  /****************************************************/
  /****************************************************/

  /****************************************************
   * test subduction
   ****************************************************/
  little_group_projector_type p;

  if ( ( exitstatus = init_little_group_projector ( &p ) ) != 0 ) {
    fprintf ( stderr, "# [projection_matrix_basis_vectors] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  int const interpolator_number       = 2;           // one (for now imaginary) interpolator
  int const interpolator_bispinor[2]  = { 0, 0 };         // bispinor no 0 / yes 1
  int const interpolator_parity[2]    = { 1, 1 };         // intrinsic operator parity, value 1 = intrinsic parity +1, -1 = intrinsic parity -1,
                                                     // value 0 = opposite parity not taken into account
  int const interpolator_cartesian[2] = { 0, 0 };         // spherical basis (0) or cartesian basis (1) ? cartesian basis only meaningful for J = 1, J2 = 2, i.e. 3-dim. representation
  
  // char const interpolator_name[2][12]  = { "", ""};  // name used in operator listing
  // char const interpolator_tex_name[2][2][20]  = { { "", "" }, { "", "" } };  // name used in operator listing

  char const correlator_name[]    = "basi vector";

  int ** interpolator_momentum_list = init_2level_itable ( interpolator_number, 3 );
  if ( interpolator_momentum_list == NULL ) {
    fprintf ( stderr, "# [projection_matrix_basis_vectors] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(2);
  }

  /****************************************************/
  /****************************************************/

  /****************************************************
   * loop on little groups
   ****************************************************/
  for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ )
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
      fprintf(stderr, "[projection_matrix_basis_vectors] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
      fprintf(stderr, "[projection_matrix_basis_vectors] Error, could not match Pref to lg %s %d\n", __FILE__, __LINE__);
      EXIT(10);
    } else {
      fprintf ( stdout, "# [projection_matrix_basis_vectors] Ptot %3d %3d %3d   Pref %3d %3d %3d R %2d  lg %s   %s %d\n",
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
        fprintf ( stderr, "[projection_matrix_basis_vectors] Error from set_rot_mat_table_cubic_group, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      /****************************************************
       * momentum tag
       ****************************************************/
      char momentum_str[100];
      sprintf( momentum_str, ".px%d_py%d_pz%d", Ptot[0], Ptot[1], Ptot[2] );

      /****************************************************
       * loop on orbital angular momentum
       ****************************************************/
      for ( int J2_orbital = 0; J2_orbital <= 4; J2_orbital += 2 ) {
  
          int interpolator_J2[2] = { J2_orbital, 1 };
  
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
          fprintf ( stderr, "[projection_matrix_basis_vectors] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
    
        /****************************************************/
        /****************************************************/
     
        /****************************************************
         * write projector to file
         ****************************************************/
        if ( write_projector ) {
  
          sprintf ( filename, "lg_%s_irrep_%s_J2_%d_%d_%s_Rref%.2d.sbd",
          lg[ilg].name, lg[ilg].lirrep[i_irrep], interpolator_J2[0], interpolator_J2[1], momentum_str, refframerot );
  
          FILE*ofs = fopen ( filename, "w" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "# [projection_matrix_basis_vectors] Error from fopen %s %d\n", __FILE__, __LINE__);
            EXIT(2);
          }
  
          exitstatus = little_group_projector_show ( &p, ofs , 1 );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[projection_matrix_basis_vectors] Error from little_group_projector_show, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
  
          /****************************************************
           * close output file
           ****************************************************/
          fclose ( ofs );
        }
  
        /****************************************************
         * loop on operator sides
         * ( sink / source = annihilation / creation )
         ****************************************************/
        for ( int iac = 1; iac <= 1; iac++  )
        {
  
          for ( int ibeta = 0; ibeta < r_irrep.dim; ibeta++ )
          {
  
            p.ref_row_target = ibeta;
  
            /****************************************************
             * loop on target J2
             ****************************************************/
            for ( int J2_target = abs( interpolator_J2[0] - interpolator_J2[1] ); J2_target <= interpolator_J2[0] + interpolator_J2[1]; J2_target += 2 ) {
  
              if ( g_verbose > 2 ) fprintf ( stdout, "# [projection_matrix_basis_vectors] target J2 = %d %s %d\n", J2_target, __FILE__, __LINE__);
  
              /****************************************************
               * reduce to target J2
               ****************************************************/
              double *** cg = init_3level_dtable ( J2_target + 1, interpolator_J2[0] + 1, interpolator_J2[1] + 1 );
              if ( cg == NULL ) {
                fprintf ( stderr, "[projection_matrix_basis_vectors] Error from linit_Xlevel_Ytable %s %d\n",  __FILE__, __LINE__);
                EXIT(2);
              }
  
  
              for ( int k = 0; k <= J2_target; k++ ) {
                for ( int m0 = 0; m0 <= interpolator_J2[0]; m0++ ) {
                for ( int m1 = 0; m1 <= interpolator_J2[1]; m1++ ) {
                  cg[k][m0][m1] = clebsch_gordan_coeff ( J2_target, J2_target-2*k, interpolator_J2[0], interpolator_J2[0]-2*m0, interpolator_J2[1], interpolator_J2[1]-2*m1 );
                  fprintf( stdout, "cg %3d %3d     %3d %3d     %3d %3d     %10.7f\n",
                      J2_target, J2_target-2*k, interpolator_J2[0], interpolator_J2[0]-2*m0, interpolator_J2[1], interpolator_J2[1]-2*m1 ,  cg[k][m0][m1] );
                }}
              }
  
              int const irrep_dim  = r_irrep.dim;
              int const spinor_dim = ( interpolator_J2[0] + 1 ) * ( interpolator_J2[1] + 1 ); 
  
              int const matrix_dim = J2_target + 1;
  
              fprintf ( stdout, "# [projection_matrix_basis_vectors] spinor_dime = %d\n", matrix_dim  );
              fprintf ( stdout, "# [projection_matrix_basis_vectors] spinor_dime = %d\n", spinor_dim  );
              fprintf ( stdout, "# [projection_matrix_basis_vectors] irrep_dim   = %d\n", irrep_dim );
  
  
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
              double _Complex *** projection_matrix_aux = init_3level_ztable ( irrep_dim, spinor_dim, spinor_dim );  /* annihilation and creation operator */
              if ( projection_matrix_aux == NULL ) {
                fprintf ( stderr, "[projection_matrix_basis_vectors] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
                EXIT(2);
              }
  
              double _Complex *** projection_matrix_v = init_3level_ztable ( irrep_dim, spinor_dim, matrix_dim );  /* annihilation and creation operator */
              if ( projection_matrix_v == NULL ) {
                fprintf ( stderr, "[projection_matrix_basis_vectors] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
                EXIT(2);
              }
  
              double _Complex *** projection_matrix_s = init_3level_ztable ( irrep_dim, spinor_dim, spinor_dim );  /* for Gram-Schmidt decomposition */
              if ( projection_matrix_s == NULL ) {
                fprintf ( stderr, "[projection_matrix_basis_vectors] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
                EXIT(2);
              }
              double _Complex *** projection_matrix_u = init_3level_ztable ( irrep_dim, spinor_dim, matrix_dim );  /* for Gram-Schmidt decomposition */
              if ( projection_matrix_u == NULL ) {
                fprintf ( stderr, "[projection_matrix_basis_vectors] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__);
                EXIT(2);
              }
  
              int rank = -1;
  
              /****************************************************
               * output tag prefix
               ****************************************************/
              char tag_prefix[500], J2_target_tag[20];
              if ( J2_target % 2 == 0 ) {
                sprintf ( J2_target_tag, "spin%d", J2_target/2 );
              } else {
                sprintf ( J2_target_tag, "spin%d_2", J2_target );
              }
    
              sprintf( tag_prefix, "/%s/%s/%s/%s/%s/%s/refrow%d", operator_side[iac], p.rtarget->group, p.rtarget->irrep, J2_target_tag,  p.rspin[0].irrep, p.rspin[1].irrep, p.ref_row_target );
  
              /****************************************************
               * loop on irrep multiplet members
               ****************************************************/
              for ( int imu = 0; imu < r_irrep.dim; imu++ ) {
  
                memset ( projection_matrix_aux[imu][0], 0, spinor_dim*spinor_dim*sizeof(double _Complex ) );
  
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
                      projection_matrix_aux[imu][kl1][kl2] += iac == 0 ? \
                          /* annihilation */ \
                          conj( p.rspin[0].R[ir][k2][k1]  * p.rspin[1].R[ir][l2][l1] )  *        p.rtarget->R[ir][imu][ibeta] : \
                          /* creation  */ \
                                p.rspin[0].R[ir][k2][k1]  * p.rspin[1].R[ir][l2][l1]    * conj ( p.rtarget->R[ir][imu][ibeta]   );
  
                      /* add the rotation-inversion */
                      projection_matrix_aux[imu][kl1][kl2] += iac == 0 ? \
                          /* annihilation */ \
                          p.parity[0] * p.parity[1] * conj( p.rspin[0].IR[ir][k2][k1] * p.rspin[1].IR[ir][l2][l1] ) *        p.rtarget->IR[ir][imu][ibeta] : \
                          /* creation */ \
                          p.parity[0] * p.parity[1] *       p.rspin[0].IR[ir][k2][k1] * p.rspin[1].IR[ir][l2][l1]   * conj ( p.rtarget->IR[ir][imu][ibeta]   );
                    }}
                  }}
                }  /* end of loop on rotations / rotation-inversions */
  
                /* normalize */
                rot_mat_ti_eq_re ( projection_matrix_aux[imu], (double)p.rtarget->dim/(2.*p.rtarget->n), spinor_dim );
  
                /****************************************************
                 * reduction to target J2 with CG coefficients
                 ****************************************************/
                for ( int k = 0; k < spinor_dim; k++ ) {
                  for ( int j = 0; j < matrix_dim; j++ ) {
                    projection_matrix_v[imu][k][j] = 0;
  
                    for ( int m0 = 0; m0 < p.rspin[0].dim; m0++ ) {
                    for ( int m1 = 0; m1 < p.rspin[1].dim; m1++ ) {
                      int const m01 = p.rspin[1].dim * m0 + m1;
  
                      projection_matrix_v[imu][k][j] += cg[j][m0][m1] * projection_matrix_aux[imu][k][m01];
                    }}
                  }
                }
  
                /*
                for ( int i = 0; i < spinor_dim; i++ ) {
                for ( int k = 0; k < spinor_dim; k++ ) {
                  fprintf( stdout, "projection_matrix_aux %d %d %d    %25.16e %25.16e\n", imu, i, k, 
                      creal ( projection_matrix_aux[imu][i][k] ),
                      cimag ( projection_matrix_aux[imu][i][k] ) );
                }}
                for ( int i = 0; i < spinor_dim; i++ ) {
                for ( int k = 0; k < matrix_dim; k++ ) {
                  fprintf( stdout, "projection_matrix_v   %d %d %d    %25.16e %25.16e\n", imu, i, k, 
                      creal ( projection_matrix_v[imu][i][k] ),
                      cimag ( projection_matrix_v[imu][i][k] ) );
                }}
                */
  
                /****************************************************
                 * coefficients coefficients for operator onb from GS
                 ****************************************************/
  
                memset ( projection_matrix_s[imu][0], 0, spinor_dim * spinor_dim * sizeof(double _Complex ) );
                memset ( projection_matrix_u[imu][0], 0, spinor_dim * matrix_dim * sizeof(double _Complex ) );
  
                int const new_rank = gs_onb_mat ( projection_matrix_s[imu], projection_matrix_u[imu], projection_matrix_v[imu], spinor_dim, matrix_dim );
                fprintf( stdout, "# [projection_matrix_basis_vectors] %s/row%d new rank is %d %s %d\n", tag_prefix, imu, new_rank, __FILE__, __LINE__ );
                if ( rank == -1 ) {
                  rank = new_rank;
                } else {
                  if ( rank != new_rank ) {
                    fprintf( stderr, "[projection_matrix_basis_vectors] Error, %s/row%d has rank %d different from %d %s %d\n", tag_prefix, imu, rank, new_rank,  __FILE__, __LINE__ );
                    EXIT(14);
                  }
                }
  
#if 0
                if ( g_verbose > 5 ) {
  
                  fprintf( stdout, "\n\n" ); 
                  for ( int i = 0; i < spinor_dim; i++ ) {
                  for ( int k = 0; k < spinor_dim; k++ ) {
                    fprintf( stdout, "projection_matrix_s   %d %d %d    %25.16e %25.16e\n", imu, i, k, 
                        creal ( projection_matrix_s[imu][i][k] ),
                        cimag ( projection_matrix_s[imu][i][k] ) );
                  }}
                  fprintf( stdout, "\n\n" ); 
  
                  for ( int i = 0; i < spinor_dim; i++ ) {
                  for ( int k = 0; k < matrix_dim; k++ ) {
                    fprintf( stdout, "projection_matrix_u   %d %d %d    %25.16e %25.16e\n", imu, i, k, 
                        creal ( projection_matrix_u[imu][i][k] ),
                        cimag ( projection_matrix_u[imu][i][k] ) );
                  }}
                  fprintf( stdout, "\n\n" ); 
                }
#endif
              
              }  /* end of loop on target irrep rows */
  
              /****************************************************/
              /****************************************************/
              
              if ( rank == 0 ) {
                fprintf( stdout, "# [projection_matrix_basis_vectors] %s rank is zero; continue %s %d\n", tag_prefix, __FILE__, __LINE__ );
                continue;
              }
  
              /****************************************************
               * write coefficient matrices to hdf5 file
               *
               ****************************************************/
              sprintf( filename, "subduction.basis_vectors.h5" );
  
              for ( int imu = 0; imu < r_irrep.dim; imu++ ) {
                int const dim[2] = { spinor_dim, matrix_dim };
                char tag[600];
    
                sprintf( tag, "%s/row%d/v", tag_prefix, imu );
    
                exitstatus = write_h5_contraction ( projection_matrix_v[imu][0], NULL, filename, tag, "double", 2, dim );
                if ( exitstatus != 0 ) {
                  fprintf ( stderr, "[projection_matrix_basis_vectors] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(2);
                }
    
                sprintf( tag, "%s/row%d/s", tag_prefix, imu );
    
                exitstatus = write_h5_contraction ( projection_matrix_s[imu][0], NULL, filename, tag, "double", 2, dim );
                if ( exitstatus != 0 ) {
                  fprintf ( stderr, "[projection_matrix_basis_vectors] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(2);
                }
    
                sprintf( tag, "%s/row%d/u", tag_prefix, imu );
    
                exitstatus = write_h5_contraction ( projection_matrix_u[imu][0], NULL, filename, tag, "double", 2, dim );
                if ( exitstatus != 0 ) {
                  fprintf ( stderr, "[projection_matrix_basis_vectors] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(2);
                }
    
              }
  
              /****************************************************/
              /****************************************************/
    
              /****************************************************
               * print the operator in mixed
               * text + coefficient form
               ****************************************************/
              char j_tag[12] = "NA", j0_tag[12] = "NA", j1_tag[12] = "NA", m_tag[12] = "NA";
              sprintf ( filename, "lg_%s.irrep_%s.refrow_%d.%s.%s.%s.%s.tex",
                  lg[ilg].name, lg[ilg].lirrep[i_irrep], ibeta,
                  spin_irrep_name(j_tag, J2_target), spin_irrep_name( j0_tag, interpolator_J2[0] ), spin_irrep_name ( j1_tag, interpolator_J2[1] ), operator_side[iac] );
  
              FILE * ofs2 = fopen ( filename, "w" );
              if ( ofs2 == NULL ) {
                fprintf ( stderr, "[projection_matrix_basis_vectors] Error from fopen %s %d\n", __FILE__, __LINE__);
                EXIT(2);
              } else {
                fprintf ( stdout, "# [projection_matrix_basis_vectors] Writing to file %s %s %d\n", filename, __FILE__, __LINE__);
              }
    
              for ( int ir = 0; ir < rank; ir++ ) {
              
                for ( int imu = 0; imu < r_irrep.dim; imu++ ) {
  
                  for ( int is = 0; is < matrix_dim; is++ ) {
                    double _Complex z = projection_matrix_u[imu][ir][is];
                    if ( cabs ( z ) > eps ) {
  
                      fprintf ( ofs2, "%6s & %6s & $%4s$ & $%4s$ & $%4s$ & $%d$ & $%d$ & $%4s$  & $%+8.7f  %+8.7f\\,i$ \\\\\n",
                          p.rtarget->group, p.rtarget->irrep, 
                          spin_tag(j_tag, J2_target), spin_tag( j0_tag, interpolator_J2[0] ),  spin_tag( j1_tag, interpolator_J2[1] ),
                          ir, imu, spin_tag( m_tag, J2_target-2*is ), 
                          __dgeps ( creal(z), eps ), __dgeps ( cimag(z), eps ) );
                    }
                  }  /* end of loop on matrix dimension */
    
                }  /* end of loop on rank = loop on operators / rank */
    
              }  /* end of loop on target irrep rows */
              fprintf ( ofs2, "\\hline\n" );
              fclose ( ofs2 );
  
              /****************************************************
               * test irrep multiplet rotation property
               * use a p_target projector
               ****************************************************/
#if 0 
              if ( rank > 0 ) {

                exitstatus = check_subduction_matrix_multiplett_rotation ( projection_matrix_v, rank, p, operator_side[iac], 1, NULL );
                if ( exitstatus != 0 ) {
                  fprintf( stderr, "[projection_matrix_basis_vectors] Error from check_subduction_matrix_multiplett_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(12);
                } */

                exitstatus = check_subduction_matrix_multiplett_rotation ( projection_matrix_u, rank, p, operator_side[iac], 1, NULL );

                if ( exitstatus != 0 ) {
                  fprintf( stderr, "[projection_matrix_basis_vectors] Error from check_subduction_matrix_multiplett_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(12);
                }
  
              }  /* end of if rank > 0 */
#endif 
              /****************************************************
               * deallocate matrices
               ****************************************************/
              fini_3level_ztable ( &projection_matrix_v );
              fini_3level_ztable ( &projection_matrix_s );
              fini_3level_ztable ( &projection_matrix_u );
  
              fini_3level_dtable ( &cg );

            }  /* end of loop on target J2 values */
  
          }  /* end of loop on irrep matrix ref. rows */
  
        }  /* end of source / sink side */
  
        /****************************************************/
        /****************************************************/
    
        fini_little_group_projector ( &p );
    

      }  /* end of loop on orbital angular momentum */

      /****************************************************/
      /****************************************************/
 
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

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [projection_matrix_basis_vectors] %s# [projection_matrix_basis_vectors] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [projection_matrix_basis_vectors] %s# [projection_matrix_basis_vectors] end of run\n", ctime(&g_the_time));
  }

  return(0);
}
