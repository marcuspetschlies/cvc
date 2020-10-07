/****************************************************
 * piN2piN_projection
 * 
 * PURPOSE:
 *   originally copied from piN2piN_correlators.cpp
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
#include <sys/time.h>
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
#include "cvc_timer.h"
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

#define MAX_UDLI_NUM 1000


using namespace cvc;

/***********************************************************
 * main program
 * This code is supposed to be produce table of coefficients
 * to form maximal number of linearly independent interpolating
 * operators for the D-D, N-N, piN-piN case
 ***********************************************************/
int main(int argc, char **argv) {
 
#define _ZCOEFF_EPS 8.e-12

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
  int check_reference_rotation = 0;
  char filename[200];
  double ratime, retime;
  FILE *ofs = NULL;

  int udli_count = 0;
  char udli_list[MAX_UDLI_NUM][500];
  char udli_name[500];
  twopoint_function_type *udli_ptr[MAX_UDLI_NUM];

  struct timeval ta, tb;

  /***********************************************************
   * set Cg basis projection coefficients
   ***********************************************************/
  double const Cgamma_basis_matching_coeff[16] = {
   -1.00,  /*  0 =  Cgy       g0*-i=cgy   */
   -1.00,  /*  1 =  Cgzg5     g1*-i=cgzg5 */
   +1.00,  /*  2 =  Cg0       g2*+i=cg0   */
   +1.00,  /*  3 =  Cgxg5     g3*+i=cgxg5 */
   -1.00,  /*  4 =  Cgyg0     g4*-i=cgyg0 */
   +1.00,  /*  5 =  Cgyg5g0   g5*+i=cgyg5g0 */
   -1.00,  /*  6 =  Cgyg5     g6*-i=cgyg5   */
   -1.00,  /*  7 =  Cgz       g7*-i=cgz */
   -1.00,  /*  8 =  Cg5g0     g8*-i=cg5g0 */
   +1.00,  /*  9 =  Cgx       g9*+i=cgx */
   +1.00,  /* 10 =  Cgzg5g0  g10*+i=cgzg5g0  */
   -1.00,  /* 11 =  C        g11*-i=C  */
   -1.00,  /* 12 =  Cgxg5g0  g12*-i=Cgxg5g0  */
   +1.00,  /* 13 =  Cgxg0    g13*+i=cgxg0  */
   +1.00,  /* 14 =  Cg5      g14*+i=Cg5 */
   -1.00   /* 15 =  Cgzg0    g15*-i=cgzg0*/
  };

  /***********************************************************
   * initialize MPI if used
   ***********************************************************/
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  /***********************************************************
   * evaluate command line arguments
   ***********************************************************/
  while ((c = getopt(argc, argv, "ch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_reference_rotation = 1;
      fprintf ( stdout, "# [piN2piN_projection] check_reference_rotation set to %d\n", check_reference_rotation );
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
  fprintf(stdout, "[piN2piN_projection] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************
   * package-own initialization of MPI parameters
   ***********************************************************/
  mpi_init(argc, argv);

  /***********************************************************
   * report git version
   ***********************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [piN2piN_projection] git version = %s\n", g_gitversion);
  }

  /***********************************************************
   * set geometry
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_projection] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  int const io_proc = get_io_proc ();

  /***********************************************************
   * TEST: report size of twopoint_function_type
   ***********************************************************/
  if ( io_proc == 0 ) {
    fprintf ( stdout, "# [piN2piN_projection] sizeof twopoint_function_type = %lu\n", sizeof ( twopoint_function_type ) );
  }



  /****************************************************
   * set cubic group single/double cover
   * rotation tables
   ****************************************************/
  rot_init_rotation_table();

  /***********************************************************
   * initialize gamma matrix algebra and several
   * gamma basis matrices
   ***********************************************************/
  init_gamma_matrix ();

  /******************************************************
   * set gamma matrices
   *   tmLQCD counting
   ******************************************************/
  gamma_matrix_type gamma[16];
  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set ( &(gamma[i]), i, 1. );
  }


   
  /******************************************************
   * loop on 2-point functions
   ******************************************************/
  for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

    printf("%d %d %d\n", g_twopoint_function_list[i2pt].pf1[0], g_twopoint_function_list[i2pt].pf1[1], g_twopoint_function_list[i2pt].pf1[2]);
    printf("%d %d %d\n", g_twopoint_function_list[i2pt].pf2[0], g_twopoint_function_list[i2pt].pf2[1], g_twopoint_function_list[i2pt].pf2[2]);


    /******************************************************
     * print the 2-point function parameters
     ******************************************************/
    sprintf ( filename, "twopoint_function_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[piN2piN_projection] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    twopoint_function_print ( &(g_twopoint_function_list[i2pt]), "TWPT", ofs );
    fclose ( ofs );

    printf("%d %d %d\n", g_twopoint_function_list[i2pt].pf1[0], g_twopoint_function_list[i2pt].pf1[1], g_twopoint_function_list[i2pt].pf1[2]);
    printf("%d %d %d\n", g_twopoint_function_list[i2pt].pf2[0], g_twopoint_function_list[i2pt].pf2[1], g_twopoint_function_list[i2pt].pf2[2]);



    /****************************************************
     * read little group parameters
     ****************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );
    little_group_type little_group;
    printf("%s\n",g_twopoint_function_list[i2pt].group);
    if ( ( exitstatus = little_group_read ( &little_group, g_twopoint_function_list[i2pt].group, little_group_list_filename ) ) != 0 ) {
      fprintf ( stderr, "[piN2piN_projection] Error from little_group_read, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(2);
    }
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "piN2piN_projection", "little_group_read", g_cart_id == 0 );

    sprintf ( filename, "little_group_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[piN2piN_projection] Error from fopen %s %d\n", __FILE__, __LINE__ );
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
      fprintf ( stderr, "# [piN2piN_projection] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }
    

    /****************************************************
     * parameters for setting the projector
     ****************************************************/
    int ref_row_target          = -1;     // no reference row for target irrep
    int * ref_row_spin          = NULL;   // no reference row for spin matrices
    int refframerot             = -1;     // reference frame rotation
                                          //   added below
    int row_target              = -1;     // no target row
    int cartesian_list[1]       = { 0 };  // not cartesian
    int parity_list[1]          = { 1 };  // intrinsic parity is +1
    const int ** momentum_list  = NULL;   // no momentum list given
    int bispinor_list[1]        = { 1 };  // bispinor yes
    int J2_list[1]              = { 1 };  // spin 1/2
    int Pref[3] = {-1,-1,-1};

    int const Ptot[3] = {
      g_twopoint_function_list[i2pt].pf1[0] + g_twopoint_function_list[i2pt].pf2[0],
      g_twopoint_function_list[i2pt].pf1[1] + g_twopoint_function_list[i2pt].pf2[1],
      g_twopoint_function_list[i2pt].pf1[2] + g_twopoint_function_list[i2pt].pf2[2] };
    printf("%d %d %d\n", g_twopoint_function_list[i2pt].pf1[0], g_twopoint_function_list[i2pt].pf1[1], g_twopoint_function_list[i2pt].pf1[2]);
    printf("%d %d %d\n", g_twopoint_function_list[i2pt].pf2[0], g_twopoint_function_list[i2pt].pf2[1], g_twopoint_function_list[i2pt].pf2[2]);

    /* if ( g_verbose > 1 ) fprintf ( stdout, "# [piN2piN_projection] twopoint_function %3d Ptot = %3d %3d %3d\n", i2pt, 
        Ptot[0], Ptot[1], Ptot[2] ); */

    /****************************************************
     * do we need a reference frame rotation ?
     ****************************************************/
    exitstatus = get_reference_rotation ( Pref, &refframerot, Ptot );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[piN2piN_projection] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(4);
    } else if ( g_verbose > 1 ) {
      fprintf ( stdout, "# [piN2piN_projection] twopoint_function %3d Ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n", i2pt, 
          Ptot[0], Ptot[1], Ptot[2], refframerot, Pref[0], Pref[1], Pref[2]);
    }


    fprintf ( stdout, "# [piN2piN_projection] twopoint_function %3d Ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n", i2pt,
          Ptot[0], Ptot[1], Ptot[2], refframerot, Pref[0], Pref[1], Pref[2]);



    fflush ( stdout );

    /****************************************************
     * set the projector with the info we have
     ****************************************************/
    exitstatus = little_group_projector_set (
        &projector,
        &little_group,
        g_twopoint_function_list[i2pt].irrep ,
        row_target,
        1,
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
      fprintf ( stderr, "[piN2piN_projection] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(3);
    }

    sprintf ( filename, "little_group_projector_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[piN2piN_projection] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    exitstatus = little_group_projector_show ( &projector, ofs, 1 );
    fclose ( ofs );

    

    /****************************************************
     * check, that projector has correct d-vector
     ****************************************************/
    if ( ( projector.P[0] != Ptot[0] ) || ( projector.P[1] != Ptot[1] ) || ( projector.P[2] != Ptot[2] ) ) {
      fprintf ( stderr, "[piN2piN_projection] Error, projector P != Ptot\n" );
      EXIT(12);
    } else {
      if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_projection] projector P == Ptot\n" );
    }


    int const nrot      = projector.rtarget->n;
    int const irrep_dim = projector.rtarget->dim;


    /******************************************************/
    /******************************************************/
    /******************************************************
     * final, projected twopoint function struct for
     * all the reference index choices
     *
     * tp_project = list of projected 2-pt functions
     *   how many ? well,...
     *   n_tp_project =      ref row     row sink    row source
     *
     *   i.e. for each reference row used in the projector
     *   we have nrow x nrow ( source, sink ) operators
     *   
     ******************************************************/
    int const n_tp_project = irrep_dim * irrep_dim * irrep_dim * irrep_dim;

    /******************************************************
     * loop on elements of tp_project
     * - initialize
     * - copy content of current reference element of
     *   g_twopoint_function_list
     ******************************************************/
    printf("n tp project %d irrep_dim %d \n", n_tp_project, irrep_dim);

    const int spin1dimension = g_twopoint_function_list[i2pt].number_of_gammas;
    const int spin1212dimension = g_twopoint_function_list[i2pt].d;
    const int dimension_coeff = g_twopoint_function_list[i2pt].number_of_gammas*g_twopoint_function_list[i2pt].d; 
    double _Complex ** P_matrix = init_2level_ztable (dimension_coeff , dimension_coeff);
    if ( P_matrix == NULL ) {
      fprintf(stderr,"[piN2piN_projection_table] Error in allocating matrix for P\n");
    }

    gamma_matrix_type gl, gr, gf11, gf12, gf2;
    gamma_matrix_init ( &gl );
    gamma_matrix_init ( &gr );


    for ( int ref_snk = 0; ref_snk < irrep_dim; ref_snk++ ) {

      for ( int row_snk = 0; row_snk < irrep_dim; row_snk++ ) {
        /******************************************************
         * curren projection coefficient for chosen irrep rows
         * at source and sink, together with sign factors
         * from rotation+basis projection of gamma matrices
         ******************************************************/

        /******************************************************
         *  loop on all the spin1 degrees of freedom
         * 
         *  for the delta cgx,cgy,cgz
         *  for the nucleon cg5
         *
         *****************************************************/
        for (int spin1degree=0; spin1degree < spin1dimension ; ++ spin1degree ){

          /*********************************************************************  
           *  loop on all the spin1212 degrees of freedom
           *
           *  this loop is on the internal spin indices of the nucleon or delta
           ********************************************************************/
  
          for (int spin1212degree = 0; spin1212degree < spin1212dimension; ++spin1212degree ) {

  
        

            /******************************************************
             * loop on little group elements
             *  - rotations and rotation-reflections
             *  - at sink, left-applied element
             ******************************************************/

            for ( int irotl = 0; irotl < 2*nrot; irotl ++ ) {

              if ( g_verbose > 3 ) fprintf ( stdout, "# [piN2piN_projection] left rotref %2d - %2d\n", irotl / nrot, irotl % nrot );

              /******************************************************
               * set momentum vector ( = spin-1 ) rotation matrix
               *
               *                                          proper rotation          rotation-reflection
               ******************************************************/
              double _Complex ** Rpl = ( irotl < nrot ) ? projector.rp->R[irotl] : projector.rp->IR[irotl-nrot];

              /******************************************************
               * rotate the final momentum vectors pf1, pf2 with Rpl
               * left = sink side
               ******************************************************/
              int pf1[3];
              int pf2[3];
              rot_point ( pf1, g_twopoint_function_list[i2pt].pf1, Rpl );
              rot_point ( pf2, g_twopoint_function_list[i2pt].pf2, Rpl );

              /******************************************************
               * set the spinor ( = spin-1/2 bispinor ) rotation matrix
               *
               *                                          proper rotation               rotation-reflection
               ******************************************************/
              double _Complex ** Rsl = ( irotl < nrot ) ? projector.rspin[0].R[irotl] : projector.rspin[0].IR[irotl-nrot];
          
              memcpy ( gl.v, Rsl[0], 16*sizeof(double _Complex) );

              /* if ( g_verbose > 4 ) gamma_matrix_printf ( &gl, "gl", stdout ); */

              /******************************************************
               * Gamma_{f_1, 1/2} --->
               *
               *   S(R)^* Gamma_{f_1, 1/2} S(R)^H
               *
               *   ****************************
               *   * ANNIHILATION / SINK SIDE *
               *   ****************************
               ******************************************************/
              /* set and rotate gf11 */
              gamma_matrix_set ( &gf11, g_twopoint_function_list[i2pt].list_of_gammas[spin1degree][0], Cgamma_basis_matching_coeff[g_twopoint_function_list[i2pt].list_of_gammas[spin1degree][0]] );
              /* gl^C gf11 gl^H */
              gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf11, &gl, 'C', &gf11, &gl, 'H' );

              /* set and rotate gf12 */
              gamma_matrix_set ( &gf12, g_twopoint_function_list[i2pt].list_of_gammas[spin1degree][1], 1. );
              /* gl^N gf12 gl^H */
              gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf12, &gl, 'N', &gf12, &gl, 'H' );


              /* check for not set or error */
              if ( gf11.id == -1 || gf12.id == -1 ) {
                fprintf ( stderr, "[piN2piN_projection] Error from gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op for gf1 %s %d\n", __FILE__, __LINE__ );
                EXIT(217);
              }

              /******************************************************
               * correct gf11 for basis matching
               ******************************************************/
              gf11.s *= Cgamma_basis_matching_coeff[ gf11.id ];

              /* transcribe gf1 gamma ids to tp.gf1 */
              int gf1[2];
              gf1[0] = gf11.id;
              gf1[1] = gf12.id;

              /******************************************************
               * Gamma_{f_2} ---> S(R) Gamma_{f_2} S(R)^+
               ******************************************************/
              /* set and rotate gf2 */
              gamma_matrix_set ( &gf2, g_twopoint_function_list[i2pt].gf2, 1. );
              /* gl^N gf2 gl^H */
              gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf2, &gl, 'N', &gf2, &gl, 'H' );

              /* transcribe gf2 gamma id to tp.gf2 */
              int gf2_local;
              gf2_local    = gf2.id;


              // contract_diagram_mat_op_ti_zm4x4_field_ti_mat_op ( tp.c[0], Rsl, 'H' , tp.c[0], Rsr, 'N', tp.T );

              /******************************************************
               * projection variants
               ******************************************************/

              /******************************************************
               * irrep matrix for left-applied rotation
               *   = sink side
               ******************************************************/
               double _Complex ** Tirrepl = ( irotl < nrot ) ? projector.rtarget->R[irotl] : projector.rtarget->IR[irotl-nrot];



               double _Complex const zcoeff = 
                            gf11.s                               /* gamma rot sign f_1,1      */
                          * gf12.s                               /* gamma rot sign f_1,2      */
                          * gf2.s                                /* gamma rot sign f_2        */
                          * Tirrepl[row_snk][ref_snk];           /* phase irrep matrix sink   */
               if ( cabs( zcoeff ) < _ZCOEFF_EPS ) {
                /* if ( g_verbose > 4 ) fprintf ( stdout, "# [piN2piN_projection] | zcoeff = %16.7e + I %16.7e | < _ZCOEFF_EPS, continue\n", creal( zcoeff ), cimag( zcoeff ) ); */
                    continue;
               }


            } /* end of loop over rotation elements */

          } /* end of loop over spi1212degree */

        } /* end of loop over spin1degree */

      }  // end of loop on row_snk

    }  // end of loop on ref_sink


    /******************************************************
     * deallocate space inside little_group
     ******************************************************/
    little_group_fini ( &little_group );

    /******************************************************
     * deallocate space inside projector
     ******************************************************/
    fini_little_group_projector ( &projector );

  }  // end of loop on 2-point functions

  /******************************************************/
  /******************************************************/
#if 0
#endif  /* of if 0 */

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
    fprintf(stdout, "# [piN2piN_projection] %s# [piN2piN_projection] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_projection] %s# [piN2piN_projection] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
