/****************************************************
 * test_ref_rotation2
 * 
 * PURPOSE:
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "read_input_parser.h"
#include "matrix_init.h"
#include "table_init_z.h"
#include "table_init_2pt.h"
#include "contract_diagrams.h"
#include "aff_key_conversion.h"
#include "zm4x4.h"
#include "gamma.h"
#include "twopoint_function_utils.h"
#include "rotations.h"
#include "group_projection.h"
#include "little_group_projector_set.h"
#include "ranlxd.h"

#define MAX_UDLI_NUM 1000


using namespace cvc;

/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
 
  double const deps = 5.e-14;

#if defined CUBIC_GROUP_DOUBLE_COVER
  char const little_group_list_filename[] = "little_groups_2Oh.tab";
  /* int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_double_cover; */
#elif defined CUBIC_GROUP_SINGLE_COVER
  const char little_group_list_filename[] = "little_groups_Oh.tab";
  /* int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_single_cover; */
#endif

  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[200];
  double ratime, retime;
  FILE *ofs = NULL;

  /***********************************************************
   * set Cg basis projection coefficients
   ***********************************************************/
  double const Cgamma_basis_matching_coeff[16] = {
   -1.00,  /*  0 =  Cgy        */
   -1.00,  /*  1 =  Cgzg5      */
   +1.00,  /*  2 =  Cg0        */
   +1.00,  /*  3 =  Cgxg5      */
   -1.00,  /*  4 =  Cgyg0      */
   +1.00,  /*  5 =  Cgyg5g0    */
   -1.00,  /*  6 =  Cgyg5      */
   -1.00,  /*  7 =  Cgz        */
   -1.00,  /*  8 =  Cg5g0      */
   +1.00,  /*  9 =  Cgx        */
   +1.00,  /* 10 =  Cgzg5g0    */
   -1.00,  /* 11 =  C          */
   -1.00,  /* 12 =  Cgxg5g0    */
   +1.00,  /* 13 =  Cgxg0      */
   +1.00,  /* 14 =  Cg5        */
   -1.00   /* 15 =  Cgzg0      */
  };

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
      exit(1);
      break;
    }
  }

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  read_input_parser(filename);

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[test_ref_rotation2] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /***********************************************************
   * report git version
   ***********************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_ref_rotation2] git version = %s\n", g_gitversion);
  }

  /***********************************************************
   * set geometry
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0 ) {
    fprintf(stderr, "[test_ref_rotation2] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
  geometry();

  /****************************************************
   * set cubic group single/double cover
   * rotation tables
   ****************************************************/
  rot_init_rotation_table();

  /***********************************************************
   * initialize gamma matrix algebra and several
   * gamma basis matrices
   ***********************************************************/
  init_gamma_matrix ("cvc");

  /******************************************************
   * set gamma matrices
   *   tmLQCD counting
   ******************************************************/
  gamma_matrix_type gamma[16];
  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set ( &(gamma[i]), i, 1. );
  }

  gamma_matrix_type gdelta[3];
  gamma_matrix_set ( gdelta,   9, Cgamma_basis_matching_coeff[9] );
  gamma_matrix_set ( gdelta+1, 0, Cgamma_basis_matching_coeff[0] );
  gamma_matrix_set ( gdelta+2, 7, Cgamma_basis_matching_coeff[7] );
   
  /******************************************************
   * loop on 2-point functions
   ******************************************************/
  rlxd_init( 2, g_seed);

  /******************************************************
   * loop on 2-point functions
   ******************************************************/
  for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

    /******************************************************
     * print the 2-point function parameters
     ******************************************************/
    sprintf ( filename, "twopoint_function_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[test_ref_rotation2] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    char name[] = "TWPT";
    twopoint_function_print ( &(g_twopoint_function_list[i2pt]), name, ofs );
    fclose ( ofs );

    /****************************************************
     * read little group parameters
     ****************************************************/
    little_group_type little_group;
    if ( ( exitstatus = little_group_read ( &little_group, g_twopoint_function_list[i2pt].group, little_group_list_filename ) ) != 0 ) {
      fprintf ( stderr, "[test_ref_rotation2] Error from little_group_read, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(2);
    }
    
    sprintf ( filename, "little_group_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[test_ref_rotation2] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    little_group_show ( &little_group, ofs, 1 );
    fclose ( ofs );

    /****************************************************
     * initialize and set projector 
     * for current little group and irrep
     ****************************************************/
    little_group_projector_type projector;
    if ( ( exitstatus = init_little_group_projector ( &projector ) ) != 0 ) {
      fprintf ( stderr, "# [test_ref_rotation2] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    /****************************************************
     * dimension parameter governs spin representation
     ****************************************************/
    int const vector_dim = 3;
    int const spinor_dim = g_twopoint_function_list[i2pt].d;

    if ( g_verbose > 2 ) fprintf ( stdout, "# [test_ref_rotation2] vector_dim = %d spinor_dim = %d\n", vector_dim, spinor_dim );

    /****************************************************
     * parameters for setting the projector
     ****************************************************/
    int n_spin                  = 2;         /* number of spin fields to be rotated */ 
    int ref_row_target          = -1;        /* no reference row for target irrep */
    int * ref_row_spin          = NULL;      /* no reference row for spin matrices */
    int refframerot             = -1;        /* reference frame rotation added below */
    int row_target              = -1;        /* no target row */
    int cartesian_list[2]       = { 0, 1 };  /* not cartesian */
    int parity_list[2]          = { 1, -1 };  /* intrinsic parity is +1 */
    const int ** momentum_list  = NULL;      /* no momentum list given */
    int bispinor_list[2]        = { 1, 0 };  /* bispinor yes / no */
    int J2_list[2]              = { 1, 2 };  /* 2 x spin + 1 = dim ( / 2 for bispinor ) */

    int const Ptot[3] = {
      g_twopoint_function_list[i2pt].pf1[0] + g_twopoint_function_list[i2pt].pf2[0],
      g_twopoint_function_list[i2pt].pf1[1] + g_twopoint_function_list[i2pt].pf2[1],
      g_twopoint_function_list[i2pt].pf1[2] + g_twopoint_function_list[i2pt].pf2[2] };

    /* if ( g_verbose > 1 ) fprintf ( stdout, "# [test_ref_rotation2] twopoint_function %3d Ptot = %3d %3d %3d\n", i2pt, 
        Ptot[0], Ptot[1], Ptot[2] ); */

#if 0
    /****************************************************
     * do we need a reference frame rotation ?
     ****************************************************/
    int Pref[3] = {-1,-1,-1};
    exitstatus = get_reference_rotation ( Pref, &refframerot, Ptot );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_ref_rotation2] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(4);
    } else if ( g_verbose > 1 ) {
      fprintf ( stdout, "# [test_ref_rotation2] twopoint_function %3d Ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n", i2pt, 
          Ptot[0], Ptot[1], Ptot[2], refframerot, Pref[0], Pref[1], Pref[2]);
    }
#endif  /* of if 0 */

    /****************************************************
     * set the projector with the info we have
     ****************************************************/
    exitstatus = little_group_projector_set (
        &projector,
        &little_group,
        g_twopoint_function_list[i2pt].irrep ,
        row_target,
        n_spin,
        J2_list,
        momentum_list,
        bispinor_list,
        parity_list,
        cartesian_list,
        ref_row_target,
        ref_row_spin,
        g_twopoint_function_list[i2pt].type,
        refframerot );

    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_ref_rotation2] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(3);
    }

    sprintf ( filename, "little_group_projector_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[test_ref_rotation2] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    exitstatus = little_group_projector_show ( &projector, ofs, 1 );
    fclose ( ofs );

    /****************************************************
     * check, that projector has correct d-vector
     ****************************************************/
    if ( ( projector.P[0] != Ptot[0] ) || ( projector.P[1] != Ptot[1] ) || ( projector.P[2] != Ptot[2] ) ) {
      fprintf ( stderr, "[test_ref_rotation2] Error, projector P != Ptot\n" );
      EXIT(12);
    } else {
      if ( g_verbose > 2 ) fprintf ( stdout, "# [test_ref_rotation2] projector P == Ptot\n" );
    }

    /****************************************************
     *
     ****************************************************/
    int const irrep_dim = projector.rtarget->dim;
    int const nrot      = projector.rtarget->n;

    /******************************************************
     * example data field
     ******************************************************/
    double _Complex **** C = init_4level_ztable ( vector_dim, vector_dim, spinor_dim, spinor_dim );
    if ( C == NULL ) {
      fprintf ( stderr, "[test_ref_rotation2] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    twopoint_function_type ****** Cproj = init_6level_2pttable ( irrep_dim, irrep_dim, irrep_dim, irrep_dim, vector_dim, vector_dim );
    if ( Cproj == NULL ) {
      fprintf ( stderr, "[test_ref_rotation2] Error from init_6level_2pttable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    int n2pt = irrep_dim * irrep_dim * irrep_dim * irrep_dim * vector_dim * vector_dim;
    for ( int i = 0; i < n2pt; i++ ) {
      twopoint_function_init ( &( Cproj[0][0][0][0][0][i] ) );
      twopoint_function_copy ( &( Cproj[0][0][0][0][0][i] ), &(g_twopoint_function_list[i2pt]) , 0 );
      twopoint_function_allocate ( &( Cproj[0][0][0][0][0][i] ) );
    }


    printf("%d\n",vector_dim);
    ranlxd ( (double*)(C[0][0][0]), 2 * vector_dim * vector_dim * spinor_dim * spinor_dim );

    for ( int i = 0; i < vector_dim; i++ ) {
    for ( int k = 0; k < vector_dim; k++ ) {
      char Cname[30];
      sprintf ( Cname, "C%d_%d", i, k );
      rot_printf_matrix ( C[i][k], spinor_dim, Cname, stdout );
    }}

    /******************************************************
     * loop on little group elements
     *  - rotations and rotation-reflections
     *  - at sink, left-applied element
     ******************************************************/
    for ( int irotl = 0; irotl < 2*nrot; irotl ++ )
    {

      if ( g_verbose > 3 ) fprintf ( stdout, "# [test_ref_rotation2] left rotref %2d - %2d\n", irotl / nrot, irotl % nrot );

      /******************************************************
       * set the spinor ( = spin-1/2 bispinor ) rotation matrix
       *
       *                                          proper rotation               rotation-reflection
       ******************************************************/
      double _Complex ** Sl = ( irotl < nrot ) ? projector.rspin[0].R[irotl] : projector.rspin[0].IR[irotl-nrot];
      double _Complex ** Rl = ( irotl < nrot ) ? projector.rspin[1].R[irotl] : projector.rspin[1].IR[irotl-nrot];

      /******************************************************
       * loop on little group elements
       *  - rotations and rotation-reflections
       *  - at source, right-applied element
       *
       *  --- initial ---
       ******************************************************/
      for ( int irotr = 0; irotr < 2*nrot; irotr ++ )
      {

        if ( g_verbose > 3 ) fprintf ( stdout, "# [test_ref_rotation2] right rotref %2d - %2d\n", irotr / nrot, irotr % nrot );

        double _Complex ** Sr = ( irotr < nrot ) ? projector.rspin[0].R[irotr] : projector.rspin[0].IR[irotr-nrot];
        double _Complex ** Rr = ( irotr < nrot ) ? projector.rspin[1].R[irotr] : projector.rspin[1].IR[irotr-nrot];

        /******************************************************
         * residual rotation of complete 4x4 correlator
         * at sink ( left ) and source ( right )
         *
         * use Sl, Sr above
         *
         * tp.c[0] <- Sl^H x tp.c[0] x Sr
         ******************************************************/
        double _Complex ** R1 = init_2level_ztable ( spinor_dim, spinor_dim );
        double _Complex ** R2 = init_2level_ztable ( spinor_dim, spinor_dim );

        double _Complex **** RCR = init_4level_ztable ( vector_dim, vector_dim, spinor_dim, spinor_dim );
        if ( C == NULL ) {
          fprintf ( stderr, "[test_ref_rotation2] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        for ( int ivl = 0; ivl < vector_dim; ivl++ ) {

#if 0
          /* find rotated vector index  */
          double _Complex * vl_vec  = init_1level_ztable ( vector_dim );
          double _Complex * vl_vec2 = init_1level_ztable ( vector_dim );

          int iRvl = -1;
          double _Complex zRvl = 0.;
          vl_vec[ivl] = 1.;
          rot_mat_ti_vec ( vl_vec2, Rl, vl_vec, vector_dim );
          for ( int i = 0; i < vector_dim; i++ ) {
            if ( cabs(vl_vec2[i]) > 1.e-10 ) {
              iRvl = i;
              zRvl = vl_vec2[i] * ( ( irotl < nrot ) ? 1. : ( projector.parity[0] * projector.parity[1] ) );
              break;
            }
          }
          fprintf ( stdout, "# [test_ref_rotation2] Rl e_%d = ( %16.7e + I %16.7e )   e_%d\n", creal(zRvl), cimag(zRvl), ivl, iRvl );
#endif  /* of if 0 */

        for ( int ivr = 0; ivr < vector_dim; ivr++ ) {

#if 0
          /* find rotated vector index  */
          double _Complex * vr_vec  = init_1level_ztable ( vector_dim );
          double _Complex * vr_vec2 = init_1level_ztable ( vector_dim );

          int iRvr = -1;
          double _Complex zRvr = 0.;
          vr_vec[ivr] = 1.;
          rot_mat_ti_vec ( vr_vec2, Rr, vr_vec, vector_dim );
          for ( int i = 0; i < vector_dim; i++ ) {
            if ( cabs(vr_vec2[i]) > 1.e-10 ) {
              iRvr = i;
              zRvr = vr_vec2[i] * ( ( irotr < nrot ) ? 1. : ( projector.parity[0] * projector.parity[1] ) ); 
              break;
            }
          }
          fprintf ( stdout, "# [test_ref_rotation2] Rr e_%d = ( %16.7e + I %16.7e )   e_%d\n", creal(zRvr), cimag(zRvr), ivr, iRvr );
#endif  /* of if 0 */

          /******************************************************
           * apply spin rotation
           ******************************************************/
#if 0
          for ( int i = 0; i < vector_dim; i++ ) {
          for ( int k = 0; k < vector_dim; k++ ) {

            /* R1 = C x Sr */
            rot_mat_ti_mat ( R1, C[i][k], Sr, spinor_dim );

            /* R2 = Sl^+ x R1 */
            rot_mat_adj_ti_mat ( R2, Sl, R1, spinor_dim );

            /* RCR <- R2 * Rl_i,ivl^* Rr_k,ivr */
            rot_mat_pl_eq_mat_ti_co ( RCR[ivl][ivr], R2, conj(Rl[i][ivl]) * Rr[k][ivr] , spinor_dim );

          }}
#endif  /* of if 0 */

          gamma_matrix_type Sgl, Sgr, gl, gr;
          gamma_matrix_init ( &gl );
          gamma_matrix_init ( &gr );
          gamma_matrix_init ( &Sgl );
          gamma_matrix_init ( &Sgr );

          memcpy ( Sgl.v, Sl[0], 16*sizeof(double _Complex ) );
          memcpy ( Sgr.v, Sr[0], 16*sizeof(double _Complex ) );

          /* Sgl^C C g_ivl Sgl^H */
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gl, &Sgl, 'C', &(gdelta[ivl]), &Sgl, 'H' );
          int igl = ( gl.id == gdelta[0].id ) ? 0 : ( ( gl.id == gdelta[1].id ) ? 1 : ( ( gl.id == gdelta[2].id ) ? 2 : -1 ) );
          if ( igl == -1 ) {
            fprintf ( stderr, "[test_ref_rotation2] Error from gamma index matching igl = %d\n", igl );
            EXIT(12);
          }
          double _Complex zgl = gl.s * Cgamma_basis_matching_coeff[gl.id];

          /* Sgr^C C g_ivr Sgr^H */
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gr, &Sgr, 'C', &(gdelta[ivr]), &Sgr, 'H' );
          int igr = ( gr.id == gdelta[0].id ) ? 0 : ( ( gr.id == gdelta[1].id ) ? 1 : ( ( gr.id == gdelta[2].id ) ? 2 : -1 ) );
          if ( igr == -1 ) {
            fprintf ( stderr, "[test_ref_rotation2] Error from gamma index matching igr = %d\n", igr );
            EXIT(12);
          }
          double _Complex zgr = gr.s * Cgamma_basis_matching_coeff[gr.id];

#if 0
          if ( g_verbose > 2 ) fprintf ( stdout, "vrot r %2d %2d   v %d %d   il %2d %2d   ir %2d %2d   zl %16.6e %16.7e   %16.6e %16.7e   zr %16.6e %16.7e   %16.6e %16.7e\n",
              irotl, irotr, ivl, ivr, iRvl, igl, iRvr, igr, creal(zRvl), cimag(zRvl), creal(zgl), cimag(zgl), creal(zRvr), cimag(zRvr), creal(zgr), cimag(zgr) );
#endif  /* of if 0 */

          if ( g_verbose > 2 ) fprintf ( stdout, "vrot r %2d %2d   v %d %d   il %2d    ir %2d    zl %16.6e %16.7e   zr %16.6e %16.7e\n",
              irotl, irotr, ivl, ivr, igl, igr, creal(zgl), cimag(zgl), creal(zgr), cimag(zgr) );

          /******************************************************
           *
           ******************************************************/
#if 0
          /* R1 = C x Sr */
          rot_mat_ti_mat ( R1, C[iRvl][iRvr], Sr, spinor_dim );

          /* R2 = Sl^+ x R1 */
          rot_mat_adj_ti_mat ( R2, Sl, R1, spinor_dim );

          /* RCR <- R2 * Rl_i,ivl^* Rr_k,ivr */
          rot_mat_pl_eq_mat_ti_co ( RCR[ivl][ivr], R2, conj( zRvl ) * zRvr , spinor_dim );
#endif  /* of if 0 */

          /* R1 = C x Sr */
          rot_mat_ti_mat ( R1, C[igl][igr], Sr, spinor_dim );

          /* R2 = Sl^+ x R1 */
          rot_mat_adj_ti_mat ( R2, Sl, R1, spinor_dim );

          /* RCR <- R2 * Rl_i,ivl^* Rr_k,ivr */
          rot_mat_pl_eq_mat_ti_co ( RCR[ivl][ivr], R2, conj( zgl ) * zgr , spinor_dim );

          /******************************************************
           * projection variants
           ******************************************************/

          /******************************************************
           * irrep matrix for left-applied rotation
           *   = sink side
           ******************************************************/
          double _Complex ** Tirrepl = ( irotl < nrot ) ? projector.rtarget->R[irotl] : projector.rtarget->IR[irotl-nrot];

          /******************************************************
           * irrep matrix for right-applied rotation
           *   = source side
           ******************************************************/
          double _Complex ** Tirrepr = ( irotr < nrot ) ? projector.rtarget->R[irotr] : projector.rtarget->IR[irotr-nrot];

          for ( int ref_snk = 0; ref_snk < irrep_dim; ref_snk++ ) {
          for ( int ref_src = 0; ref_src < irrep_dim; ref_src++ ) {

            for ( int row_snk = 0; row_snk < irrep_dim; row_snk++ ) {
            for ( int row_src = 0; row_src < irrep_dim; row_src++ ) {


              /******************************************************
               * current projection coefficient for chosen irrep rows
               * at source and sink, together with sign factors
               * from rotation+basis projection of gamma matrices
               ******************************************************/
              double _Complex const zcoeff = Tirrepl[row_snk][ref_snk] * conj ( Tirrepr[row_src][ref_src] );

              rot_mat_pl_eq_mat_ti_co ( Cproj[ref_snk][ref_src][row_snk][row_src][ivl][ivr].c[0][0], RCR[ivl][ivr], zcoeff, spinor_dim );

              if ( g_verbose > 2 ) fprintf ( stdout, "# [test_ref_rotation2] rotl %d %2d rotr %d %2d ref %2d %2d row %2d %2d vec %2d %2d z %25.16e %25.16e\n", 
                  irotl/nrot, irotl%nrot, irotr/nrot, irotr%nrot,
                  ref_snk, ref_src, row_snk, row_src, ivl, ivr, dgeps ( creal(zcoeff), 1.e-14 ), dgeps ( cimag(zcoeff), 1.e-14 ) );

          }  // end of loop on row_src
          }  // end of loop on row_snk

        }  // end of loop on ref_src
        }  // end of loop on ref_snk
#if 0
          fini_1level_ztable ( &vr_vec );
          fini_1level_ztable ( &vr_vec2 );
#endif  /* of if 0 */

        }  // end of loop on ivr

#if 0
          fini_1level_ztable ( &vl_vec );
          fini_1level_ztable ( &vl_vec2 );
#endif  /* of if 0 */

        }  // end of loop on ivl

        fini_4level_ztable ( &RCR );
        fini_2level_ztable ( &R2 );
        fini_2level_ztable ( &R1 );

      }  // end of loop on source rotations
    }  // end of loop on sink rotations

    /******************************************************
     * plot the projected data set
     ******************************************************/
    for ( int ref_snk = 0; ref_snk < irrep_dim; ref_snk++ ) {
    for ( int ref_src = 0; ref_src < irrep_dim; ref_src++ ) {

      for ( int row_snk = 0; row_snk < irrep_dim; row_snk++ ) {
      for ( int row_src = 0; row_src < irrep_dim; row_src++ ) {

        for ( int ivl = 0; ivl < vector_dim; ivl++ ) {
        for ( int ivr = 0; ivr < vector_dim; ivr++ ) {

          char name[400];
          sprintf ( name, "Cproj_%d_%d_%d_%d_%d_%d", ref_snk, ref_src, row_snk, row_src, ivl, ivr );
          rot_printf_matrix ( Cproj[ref_snk][ref_src][row_snk][row_src][ivl][ivr].c[0][0], spinor_dim, name, stdout );

        }}
      }}
    }}


    /******************************************************
     * check the rotation property
     ******************************************************/
    twopoint_function_check_reference_rotation_vector_spinor ( Cproj[0][0][0][0][0], &projector, 5.e-12 );
    exit(1);
#if 0
#endif  /* of if 0 */

    /******************************************************
     * deallocate
     ******************************************************/
    little_group_fini ( &little_group );

    fini_little_group_projector ( &projector );

    fini_4level_ztable ( &C );

    for ( int i = 0; i < irrep_dim * irrep_dim * irrep_dim * irrep_dim * vector_dim * vector_dim ; i++ ) {
      twopoint_function_fini ( &( Cproj[0][0][0][0][0][i] ) );
    }
    fini_6level_2pttable ( &Cproj );

  }  // end of loop on 2-point functions

  /******************************************************/
  /******************************************************/

  /******************************************************
   * finalize
   *
   * free the allocated memory, finalize
   ******************************************************/
  free_geometry();

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_ref_rotation2] %s# [test_ref_rotation2] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_ref_rotation2] %s# [test_ref_rotation2] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
