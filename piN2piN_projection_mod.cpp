/****************************************************
 * piN2piN_projection_mod
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
#include "cvc_timer.h"
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

#define MAX_UDLI_NUM 10000


using namespace cvc;

/***********************************************************
 * main program
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
  struct timeval start_time, end_time;

  /***********************************************************
   * set Cg basis projection coefficients
   ***********************************************************/
  double const Cgamma_basis_matching_coeff[16] = {
    1.00,  /*  0 =  Cgy        */
   -1.00,  /*  1 =  Cgzg5      */
   -1.00,  /*  2 =  Cg0        */
    1.00,  /*  3 =  Cgxg5      */
    1.00,  /*  4 =  Cgyg0      */
   -1.00,  /*  5 =  Cgyg5g0    */
    1.00,  /*  6 =  Cgyg5      */
   -1.00,  /*  7 =  Cgz        */
    1.00,  /*  8 =  Cg5g0      */
    1.00,  /*  9 =  Cgx        */
    1.00,  /* 10 =  Cgzg5g0    */
    1.00,  /* 11 =  C          */
   -1.00,  /* 12 =  Cgxg5g0    */
   -1.00,  /* 13 =  Cgxg0      */
    1.00,  /* 14 =  Cg5        */
    1.00   /* 15 =  Cgzg0      */
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
   * timer for total time
   ***********************************************************/
  gettimeofday ( &start_time, (struct timezone *)NULL );

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
   * check source coords list
   ******************************************************/
  for ( int i = 0; i < g_source_location_number; i++ ) {
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] +  T_global ) %  T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /******************************************************
   * loop on source locations
   ******************************************************/
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {

    int const t_base = g_source_coords_list[i_src][0];

    /******************************************************
     * loop on coherent source locations
     ******************************************************/
    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int const t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4];
      int const gsx[4] = { t_coherent,
                    ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                    ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                    ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };


      get_point_source_info (gsx, sx, &source_proc_id);

      /******************************************************
       * loop on 2-point functions
       ******************************************************/
      for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

        if ( g_verbose > 4 ) {
          gettimeofday ( &ta, (struct timezone *)NULL );
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

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_projection", "print-twopoint", io_proc == 2 );

        }  /* end of if g_verbose */

        /****************************************************
         * read little group parameters
         ****************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        little_group_type little_group;
        if ( ( exitstatus = little_group_read ( &little_group, g_twopoint_function_list[i2pt].group, little_group_list_filename ) ) != 0 ) {
          fprintf ( stderr, "[piN2piN_projection] Error from little_group_read, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(2);
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_projection", "little_group_read", io_proc == 2 );

        if ( g_verbose > 4 ) {
          sprintf ( filename, "little_group_%d.show", i2pt );
          if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
            fprintf ( stderr, "[piN2piN_projection] Error from fopen %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
          little_group_show ( &little_group, ofs, 1 );
          fclose ( ofs );
        }

        /****************************************************
         * initialize and set projector 
         * for current little group and irrep
         ****************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        little_group_projector_type projector;
        if ( ( exitstatus = init_little_group_projector ( &projector ) ) != 0 ) {
          fprintf ( stderr, "# [piN2piN_projection] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_projection", "init_little_group_projector", io_proc == 2 );

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

        if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_projection] twopoint_function %3d Ptot = %3d %3d %3d\n", i2pt, Ptot[0], Ptot[1], Ptot[2] );

        /****************************************************
         * do we need a reference frame rotation ?
         ****************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        exitstatus = get_reference_rotation ( Pref, &refframerot, Ptot );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[piN2piN_projection] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(4);
        } else if ( g_verbose > 1 ) {
          fprintf ( stdout, "# [piN2piN_projection] twopoint_function %3d Ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n", i2pt, 
          Ptot[0], Ptot[1], Ptot[2], refframerot, Pref[0], Pref[1], Pref[2]);
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_projection", "get_reference_rotation", io_proc == 2 );
        fflush ( stdout );

        /****************************************************
         * set the projector with the info we have
         ****************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

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

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_projection", "little_group_projector_set", io_proc == 2 );

        if ( g_verbose > 4 ) {
          /****************************************************
           * show projector
           ****************************************************/
          gettimeofday ( &tb, (struct timezone *)NULL );

          sprintf ( filename, "little_group_projector_%d.show", i2pt );
          if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
            fprintf ( stderr, "[piN2piN_projection] Error from fopen %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
          exitstatus = little_group_projector_show ( &projector, ofs, 1 );
          fclose ( ofs );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_projection", "show-projector", io_proc == 2 );
        }

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

        /******************************************************
         * set current source coords in 2pt function
         ******************************************************/
        g_twopoint_function_list[i2pt].source_coords[0] = gsx[0];
        g_twopoint_function_list[i2pt].source_coords[1] = gsx[1];
        g_twopoint_function_list[i2pt].source_coords[2] = gsx[2];
        g_twopoint_function_list[i2pt].source_coords[3] = gsx[3];

        /******************************************************
         * this is a temporary twopoint function struct to
         * store one term in the projection sum
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        twopoint_function_type tp;

        twopoint_function_init ( &tp );

        twopoint_function_copy ( &tp, &( g_twopoint_function_list[i2pt] ), 1 );

        if ( twopoint_function_allocate ( &tp ) == NULL ) {
          fprintf ( stderr, "[piN2piN_projection] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
          EXIT(123);
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_projection", "init-copy-allocate-tp", io_proc == 2 );

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
        twopoint_function_type **** tp_project = init_4level_2pttable ( irrep_dim, irrep_dim, irrep_dim, irrep_dim );
        if ( tp_project == NULL ) {
          fprintf ( stderr, "[piN2piN_projection] Error from a init_4level_2pttable %s %d\n", __FILE__, __LINE__ );
          EXIT(124);
        }

        /******************************************************
         * loop on elements of tp_project
         * - initialize
         * - copy content of current reference element of
         *   g_twopoint_function_list
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        for ( int i = 0; i < n_tp_project; i++ ) {
          twopoint_function_type * tp_project_ptr = tp_project[0][0][0];

          twopoint_function_init ( &(tp_project_ptr[i]) );
          twopoint_function_copy ( &(tp_project_ptr[i]), &( g_twopoint_function_list[i2pt]), 1 );

          /* number of data sets in tp_project is always 1
           *   we save the sum of all diagrams in here */
          tp_project_ptr[i].n = 1;
          sprintf ( tp_project_ptr[i].norm, "NA" );
          /* abuse the diagrams name string to label the row-coordinates */
          sprintf ( tp_project_ptr[i].diagrams, "ref_snk%d/ref_src%d/row_snk%d/row_src%d", 
                i                                          / ( irrep_dim * irrep_dim * irrep_dim ), 
              ( i % ( irrep_dim *irrep_dim * irrep_dim ) ) / (             irrep_dim * irrep_dim ),
              ( i % (            irrep_dim * irrep_dim ) ) /                           irrep_dim,
                i %                          irrep_dim   );


          /* allocate memory */
          if ( twopoint_function_allocate ( &(tp_project_ptr[i]) ) == NULL ) {
            fprintf ( stderr, "[piN2piN_projection] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
            EXIT(125);
          }
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_projection", "init-copy-allocate-tp_project", io_proc == 2 );

        gamma_matrix_type gl, gr, gi11, gi12, gi2, gf11, gf12, gf2;
        gamma_matrix_init ( &gl );
        gamma_matrix_init ( &gr );

        /******************************************************
         * loop on little group elements
         *  - rotations and rotation-reflections
         *  - at sink, left-applied element
         ******************************************************/
        for ( int irotl = 0; irotl < 2*nrot; irotl ++ ) {

          /* if ( g_verbose > 4 ) fprintf ( stdout, "# [piN2piN_projection] left rotref %2d - %2d\n", irotl / nrot, irotl % nrot ); */

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
          rot_point ( tp.pf1, g_twopoint_function_list[i2pt].pf1, Rpl );
          rot_point ( tp.pf2, g_twopoint_function_list[i2pt].pf2, Rpl );

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
          gamma_matrix_set ( &gf11, g_twopoint_function_list[i2pt].gf1[0], Cgamma_basis_matching_coeff[ g_twopoint_function_list[i2pt].gf1[0] ] );
          /* gl^C gf11 gl^H */
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf11, &gl, 'C', &gf11, &gl, 'H' );

          /* set and rotate gf12 */
          gamma_matrix_set ( &gf12, g_twopoint_function_list[i2pt].gf1[1], 1. );
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
          tp.gf1[0] = gf11.id;
          tp.gf1[1] = gf12.id;

          /******************************************************
           * Gamma_{f_2} ---> S(R) Gamma_{f_2} S(R)^+
           ******************************************************/
          /* set and rotate gf2 */
          gamma_matrix_set ( &gf2, g_twopoint_function_list[i2pt].gf2, 1. );
          /* gl^N gf2 gl^H */
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf2, &gl, 'N', &gf2, &gl, 'H' );

          /* transcribe gf2 gamma id to tp.gf2 */
          tp.gf2    = gf2.id;

        /******************************************************
         * loop on little group elements
         *  - rotations and rotation-reflections
         *  - at source, right-applied element
         *
         *  --- initial ---
         ******************************************************/
        for ( int irotr = 0; irotr < 2*nrot; irotr ++ ) {

          /* if ( g_verbose > 3 ) fprintf ( stdout, "# [piN2piN_projection] right rotref %2d - %2d\n", irotr / nrot, irotr % nrot ); */

          /*                                          proper rotation          rotation-reflection */
          double _Complex ** Rpr = ( irotr < nrot ) ? projector.rp->R[irotr] : projector.rp->IR[irotr-nrot];

          rot_point ( tp.pi1, g_twopoint_function_list[i2pt].pi1, Rpr );
          rot_point ( tp.pi2, g_twopoint_function_list[i2pt].pi2, Rpr );

          double _Complex ** Rsr = ( irotr < nrot ) ? projector.rspin[0].R[irotr] : projector.rspin[0].IR[irotr-nrot];
          memcpy ( gr.v, Rsr[0], 16*sizeof(double _Complex) );

          /******************************************************
           * Gamma_{i_1, 1/2} --->
           *
           *   S(R) Gamma_{i_1, 1/2} S(R)^t
           *
           *   **************************
           *   * CREATION / SOURCE SIDE *
           *   **************************
           ******************************************************/
          
          /* gi11 <- 2pt gi1[0] */
          gamma_matrix_set ( &gi11, g_twopoint_function_list[i2pt].gi1[0], Cgamma_basis_matching_coeff[ g_twopoint_function_list[i2pt].gi1[0] ] );
#if 0
          /* gr^N gi11 gr^T */
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gi11, &gr, 'N', &gi11, &gr, 'T' );
#endif  /* of if 0 */

          /* gi11 <- gr^C gi11 gr^H */
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gi11, &gr, 'C', &gi11, &gr, 'H' );

          gamma_matrix_set ( &gi12, g_twopoint_function_list[i2pt].gi1[1], 1. );
          /* gr^N gi12 gr^H */
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gi12, &gr, 'N', &gi12, &gr, 'H' );

          if ( gi11.id == -1 || gi12.id == -1 ) {
            fprintf ( stderr, "[piN2piN_projection] Error from gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op for gi1 %s %d\n", __FILE__, __LINE__ );
            EXIT(218);
          }

          /******************************************************
           * correct gi11 sign for basis matching
           ******************************************************/
          gi11.s *= Cgamma_basis_matching_coeff[ gi11.id ];

          /* transcribe */
          tp.gi1[0] = gi11.id;
          tp.gi1[1] = gi12.id;

          /******************************************************
           * Gamma_{i_2} ---> S(R) Gamma_{i_2} S(R)^+
           ******************************************************/
          gamma_matrix_set ( &gi2, g_twopoint_function_list[i2pt].gi2, 1. );
          /* gr^N gi2 gr^H */
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gi2, &gr, 'N', &gi2, &gr, 'H' );

          tp.gi2    = gi2.id;

          /******************************************************
           * TEST
           ******************************************************/
          if ( g_verbose > 2 ) 
            fprintf ( stdout, "# [piN2piN_projection] rot %2d %2d     gf11 %2d %6.2f   gf12 %2d %6.2f   gf2 %2d %6.2f   gi11 %2d %6.2f   gi12 %2d %6.2f   gi2 %2d %6.2f\n", 
              irotl, irotr, gf11.id, gf11.s, gf12.id, gf12.s, gf2.id, gf2.s, gi11.id, gi11.s, gi12.id, gi12.s, gi2.id, gi2.s);

          if ( g_verbose > 4 && io_proc == 2 ) {
            char name[100];
            sprintf (  name, "R%.2d_TWPT_R%.2d", irotl, irotr );
            twopoint_function_print ( &tp, name, stdout );
          }

          /******************************************************
           * fill the diagram with data
           *
           * this takes too long
           ******************************************************/
          /*
          if ( ( exitstatus = twopoint_function_fill_data ( &tp, filename_prefix ) ) != 0 ) {
            fprintf ( stderr, "[piN2piN_projection] Error from twopoint_function_fill_data, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(212);
          }
          */

          /******************************************************
           * fill the diagram with data
           ******************************************************/
          for ( int ids = 0; ids < tp.n; ids++ ) {
            if ( ( exitstatus = twopoint_function_data_location_identifier ( udli_name, &tp, filename_prefix, ids, "#" ) ) != 0 ) {
              fprintf ( stderr, "[piN2piN_projection] Error from twopoint_function_data_location_identifier, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(212);
            }

            /******************************************************
             * check, whether udli_name exists in udli list
             ******************************************************/
            gettimeofday ( &ta, (struct timezone *)NULL );

            int udli_id = -1;
            for ( int i = 0; i < udli_count; i++ ) {
              if ( strcmp ( udli_name, udli_list[i] ) == 0 ) {
                udli_id  = i;
                break;
              }
            }

            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "piN2piN_projection", "check-udli-entry", io_proc == 2 );

            if ( udli_id == -1 ) {
              fprintf ( stdout, "# [piN2piN_projection] could not find udli_name %s in udli_list\n", udli_name );

              gettimeofday ( &ta, (struct timezone *)NULL );

              /******************************************************
               * start new entry in udli list
               ******************************************************/

              /******************************************************
               * check, that number udlis is not exceeded
               ******************************************************/
              if ( udli_count == MAX_UDLI_NUM ) {
                fprintf ( stderr, "[piN2piN_projection] Error, maximal number of udli exceeded\n" );
                EXIT(111);
              } else {
                if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_projection] starting udli entry number %d\n", udli_count );
              }

              udli_ptr[udli_count] = ( twopoint_function_type *)malloc ( sizeof ( twopoint_function_type ) );
              if ( udli_ptr[udli_count] == NULL ) {
                fprintf ( stderr, "[piN2piN_projection] Error from malloc %s %d\n", __FILE__, __LINE__ );
                EXIT(211);
              }

              twopoint_function_init ( udli_ptr[udli_count] );

              twopoint_function_copy ( udli_ptr[udli_count], &tp, 0 );

              udli_ptr[udli_count]->n = 1;

              strcpy ( udli_ptr[udli_count]->name, udli_name );

              twopoint_function_allocate ( udli_ptr[udli_count] );

              /******************************************************
               * fill data from udli
               ******************************************************/
              if ( ( exitstatus = twopoint_function_fill_data_from_udli ( udli_ptr[udli_count] , udli_name , io_proc) ) != 0 ) {
                fprintf ( stderr, "[piN2piN_projection] Error from twopoint_function_fill_data_from_udli, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(212);
              }

              /******************************************************
               * set udli_id on new entry
               ******************************************************/
              udli_id = udli_count;

              /******************************************************
               * set entry in udli_list
               ******************************************************/
              strcpy ( udli_list[udli_id], udli_name );

              /******************************************************
               * count new entry
               ******************************************************/
              udli_count++;

              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "piN2piN_projection", "add-udli-entry", io_proc == 2 );

            } else {
              if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_projection] udli_name %s matches udli_list[%d] %s\n", udli_name, udli_id, udli_list[udli_id] );
            }

            /******************************************************
             * copy data from udli_id entry to current tp
             ******************************************************/
            memcpy ( tp.c[ids][0][0], udli_ptr[udli_id]->c[0][0][0], tp.T * tp.d * tp.d * sizeof(double _Complex ) );

          }  /* end of loop on data sets = diagrams */

          /******************************************************
           * apply diagram norm
           *
           * a little overhead, since this done for each tp,
           * not each udli_ptr only
           ******************************************************/
          gettimeofday ( &ta, (struct timezone *)NULL );
          if ( ( exitstatus = twopoint_function_apply_diagram_norm ( &tp ) ) != 0 ) {
            fprintf ( stderr, "[piN2piN_projection] Error from twopoint_function_apply_diagram_norm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(213);
          }
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_projection", "twopoint_function_apply_diagram_norm", io_proc == 2 );


          /******************************************************
           * sum up data sets in tp
           * - add data sets 1,...,tp.n-1 to data set 0
           ******************************************************/
          gettimeofday ( &ta, (struct timezone *)NULL );
          /*
          for ( int i = 1; i < tp.n; i++ ) {
            contract_diagram_zm4x4_field_pl_eq_zm4x4_field ( tp.c[0], tp.c[i], tp.T );
          }
          */
          if ( ( exitstatus = twopoint_function_accum_diagrams ( tp.c[0], &tp ) ) != 0 ) {
            fprintf ( stderr, "[piN2piN_projection] Error from twopoint_function_accum_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(216);
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_projection", "twopoint_function_accum_diagrams", io_proc == 2 );

          /******************************************************
           * residual rotation of complete 4x4 correlator
           * at sink ( left ) and source ( right )
           *
           * use Rsl, Rsr above
           *
           * tp.c[0] <- Rsl^H x tp.c[0] x Rsr
           ******************************************************/
          gettimeofday ( &ta, (struct timezone *)NULL );

          contract_diagram_mat_op_ti_zm4x4_field_ti_mat_op ( tp.c[0], Rsl, 'H' , tp.c[0], Rsr, 'N', tp.T );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_projection", "contract_diagram_mat_op_ti_zm4x4_field_ti_mat_op", io_proc == 2 );

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

          gettimeofday ( &ta, (struct timezone *)NULL );

          double _Complex ** Tirrepr = ( irotr < nrot ) ? projector.rtarget->R[irotr] : projector.rtarget->IR[irotr-nrot];

          for ( int ref_snk = 0; ref_snk < irrep_dim; ref_snk++ ) {
          for ( int ref_src = 0; ref_src < irrep_dim; ref_src++ ) {

            for ( int row_snk = 0; row_snk < irrep_dim; row_snk++ ) {
            for ( int row_src = 0; row_src < irrep_dim; row_src++ ) {

              /******************************************************
               * curren projection coefficient for chosen irrep rows
               * at source and sink, together with sign factors
               * from rotation+basis projection of gamma matrices
               ******************************************************/
              double _Complex const zcoeff = 
                  gf11.s                               /* gamma rot sign f_1,1      */
                * gf12.s                               /* gamma rot sign f_1,2      */
                * gf2.s                                /* gamma rot sign f_2        */
                * gi11.s                               /* gamma rot sign i_1,1      */
                * gi12.s                               /* gamma rot sign i_1,2      */
                * gi2.s                                /* gamma rot sign i_2        */
                *        Tirrepl[row_snk][ref_snk]     /* phase irrep matrix sink   */
                * conj ( Tirrepr[row_src][ref_src] );  /* phase irrep matrix source */
              
              if ( cabs( zcoeff ) < _ZCOEFF_EPS ) {
                /* if ( g_verbose > 4 ) fprintf ( stdout, "# [piN2piN_projection] | zcoeff = %16.7e + I %16.7e | < _ZCOEFF_EPS, continue\n", creal( zcoeff ), cimag( zcoeff ) ); */
                continue;
              }


              contract_diagram_zm4x4_field_eq_zm4x4_field_pl_zm4x4_field_ti_co (
                  tp_project[ref_snk][ref_src][row_snk][row_src].c[0],
                  tp_project[ref_snk][ref_src][row_snk][row_src].c[0], tp.c[0], zcoeff, tp.T );

              if ( g_verbose > 4 ) fprintf ( stdout, "# [piN2piN_projection] zcoeff = %16.7e   %16.7e\n", creal( zcoeff ), cimag( zcoeff ) );

            }  // end of loop on row_src
            }  // end of loop on row_snk

          }  // end of loop on ref_src
          }  // end of loop on ref_snk

        }  // end of loop on source rotations
        }  // end of loop on sink   rotations

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_projection", "row-col-accum-projection", io_proc == 2 );
#if 0
        /******************************************************
         * check reference index rotations
         ******************************************************/
        if ( check_reference_rotation ) {
          twopoint_function_check_reference_rotation ( tp_project[0][0][0], &projector, 5.e-12 );
        }
#endif  /* of if 0 */

        /******************************************************
         * output of tp_project
         *
         * loop over individiual projection variants
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        for ( int itp = 0; itp < n_tp_project; itp++ ) {

          twopoint_function_type * tp_project_ptr = tp_project[0][0][0];

          /******************************************************
           * multiply group projection normalization
           ******************************************************/
          /* No, not this part, 
           * twopoint_function_get_correlator_phase ( &(tp_project_ptr[itp]) )
           *
           * this has been added in piN2piN_diagrams_complete via
           * via factor in zsign from function contract_diagram_get_correlator_phase */

          /* ztmp given by ( irrep_dim / number of little group members )^2
           *   factor of 4 because ...->n is the number of proper rotations only,
           *   so half the number group elements
           */



          double _Complex const ztmp = (double)( projector.rtarget->dim * projector.rtarget->dim ) / \
                                       ( 4. *    projector.rtarget->n   * projector.rtarget->n   );

          if ( g_verbose > 4 ) fprintf ( stdout, "# [piN2piN_projection] correlator norm = %25.16e %25.16e\n", creal( ztmp ), cimag( ztmp ) );

          exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( tp_project_ptr[itp].c[0], ztmp, tp_project_ptr[itp].T );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[piN2piN_projection] Error from contract_diagram_zm4x4_field_ti_eq_co %s %d\n", __FILE__, __LINE__ );
            EXIT(217)
          }
 
          /******************************************************
           * write to disk
           ******************************************************/
          exitstatus = twopoint_function_write_data ( &( tp_project_ptr[itp] ) );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[piN2piN_projection] Error from twopoint_function_write_data, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(12);
          }

        }  /* end of loop on n_tp_project */

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_projection", "group-projection-norm-output", io_proc == 2 );

        /******************************************************
         * deallocate twopoint_function vars tp and tp_project
         ******************************************************/
        twopoint_function_fini ( &tp );
        for ( int i = 0; i < n_tp_project; i++ ) {
          twopoint_function_fini ( &(tp_project[0][0][0][i]) );
        }
        fini_4level_2pttable ( &tp_project );

        /******************************************************
         * deallocate space inside little_group
         ******************************************************/
        little_group_fini ( &little_group );

        /******************************************************
         * deallocate space inside projector
         ******************************************************/
        fini_little_group_projector ( &projector );

      }  // end of loop on g_twopoint_function_number 2pt functions

      /******************************************************
       * reset udli_count and deallocate udli_ptr
       *
       * NOTE: too much memory ? free udli content
       *       ONLY after loop on 2-point functions
       *
       * new-readin definitely after change of total
       * momentum
       ******************************************************/
      for ( int i = 0; i < udli_count; i++ ) {
        twopoint_function_fini ( udli_ptr[i] );
        free ( udli_ptr[i] );
        udli_ptr[i] = NULL;
      }
      udli_count = 0;

    }  // end of loop on coherent source locations

  }  // end of loop on base source locations

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

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "piN2piN_projection", "total-time", io_proc == 2 );

  
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_projection] %s# [piN2piN_projection] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_projection] %s# [piN2piN_projection] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
