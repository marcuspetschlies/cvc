/****************************************************
 * test_ref_rotation
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
  fprintf(stdout, "[test_ref_rotation] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /***********************************************************
   * report git version
   ***********************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_ref_rotation] git version = %s\n", g_gitversion);
  }

  /***********************************************************
   * set geometry
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0 ) {
    fprintf(stderr, "[test_ref_rotation] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
      fprintf ( stderr, "[test_ref_rotation] Error from fopen %s %d\n", __FILE__, __LINE__ );
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
      fprintf ( stderr, "[test_ref_rotation] Error from little_group_read, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(2);
    }
    
    sprintf ( filename, "little_group_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[test_ref_rotation] Error from fopen %s %d\n", __FILE__, __LINE__ );
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
      fprintf ( stderr, "# [test_ref_rotation] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    /****************************************************
     * dimension parameter governs spin representation
     ****************************************************/
    int const dim = g_twopoint_function_list[i2pt].d;
    if ( g_verbose > 2 ) fprintf ( stdout, "# [test_ref_rotation] data array dimension is %d\n", dim );

    /****************************************************
     * parameters for setting the projector
     ****************************************************/
    int ref_row_target          = -1;     /* no reference row for target irrep */
    int * ref_row_spin          = NULL;   /* no reference row for spin matrices */
    int refframerot             = -1;     /* reference frame rotation added below */
    int row_target              = -1;     /* no target row */
    int cartesian_list[1]       = { 0 };  /* not cartesian */
    int parity_list[1]          = { 1 };  /* intrinsic parity is +1 */
    const int ** momentum_list  = NULL;   /* no momentum list given */
    int bispinor_list[1]        = { ( dim == 4 ) && ( strcmp(g_twopoint_function_list[i2pt].type, "bispinor") == 0 )  };  /* bispinor yes / no */
    int J2_list[1]              = { ( ( dim == 4 ) && ( strcmp(g_twopoint_function_list[i2pt].type, "bispinor") == 0 ) ) ? 1 : dim-1 };  /* 2 x spin + 1 = dim */

    int const Ptot[3] = {
      g_twopoint_function_list[i2pt].pf1[0] + g_twopoint_function_list[i2pt].pf2[0],
      g_twopoint_function_list[i2pt].pf1[1] + g_twopoint_function_list[i2pt].pf2[1],
      g_twopoint_function_list[i2pt].pf1[2] + g_twopoint_function_list[i2pt].pf2[2] };

    /* if ( g_verbose > 1 ) fprintf ( stdout, "# [test_ref_rotation] twopoint_function %3d Ptot = %3d %3d %3d\n", i2pt, 
        Ptot[0], Ptot[1], Ptot[2] ); */

#if 0
    /****************************************************
     * do we need a reference frame rotation ?
     ****************************************************/
    int Pref[3] = {-1,-1,-1};
    exitstatus = get_reference_rotation ( Pref, &refframerot, Ptot );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_ref_rotation] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(4);
    } else if ( g_verbose > 1 ) {
      fprintf ( stdout, "# [test_ref_rotation] twopoint_function %3d Ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n", i2pt, 
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
      fprintf ( stderr, "[test_ref_rotation] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(3);
    }

    sprintf ( filename, "little_group_projector_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[test_ref_rotation] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    exitstatus = little_group_projector_show ( &projector, ofs, 1 );
    fclose ( ofs );

    /****************************************************
     * check, that projector has correct d-vector
     ****************************************************/
    if ( ( projector.P[0] != Ptot[0] ) || ( projector.P[1] != Ptot[1] ) || ( projector.P[2] != Ptot[2] ) ) {
      fprintf ( stderr, "[test_ref_rotation] Error, projector P != Ptot\n" );
      EXIT(12);
    } else {
      if ( g_verbose > 2 ) fprintf ( stdout, "# [test_ref_rotation] projector P == Ptot\n" );
    }

    /****************************************************
     *
     ****************************************************/
    int const irrep_dim = projector.rtarget->dim;
    int const nrot      = projector.rtarget->n;

    /******************************************************
     * example data field
     ******************************************************/
    double _Complex ** C     = init_2level_ztable ( dim, dim );
    /* double _Complex ****** Cproj = init_6level_ztable ( irrep_dim, irrep_dim, irrep_dim, irrep_dim, dim, dim ); */

    twopoint_function_type **** Cproj = init_4level_2pttable ( irrep_dim, irrep_dim, irrep_dim, irrep_dim );

    for ( int i = 0; i < irrep_dim * irrep_dim * irrep_dim * irrep_dim ; i++ ) {
      twopoint_function_init ( &( Cproj[0][0][0][i] ) );
      twopoint_function_copy ( &( Cproj[0][0][0][i] ), &(g_twopoint_function_list[i2pt]) , 0 );
      twopoint_function_allocate ( &( Cproj[0][0][0][i] ) );
    }


    ranlxd ( (double*)(C[0]), 2*dim*dim );

    char Cname[] = "C";
    rot_printf_matrix ( C, dim, Cname, stdout );

    /******************************************************
     * loop on little group elements
     *  - rotations and rotation-reflections
     *  - at sink, left-applied element
     ******************************************************/
    for ( int irotl = 0; irotl < 2*nrot; irotl ++ )
    {

      if ( g_verbose > 3 ) fprintf ( stdout, "# [test_ref_rotation] left rotref %2d - %2d\n", irotl / nrot, irotl % nrot );

      /******************************************************
       * set the spinor ( = spin-1/2 bispinor ) rotation matrix
       *
       *                                          proper rotation               rotation-reflection
       ******************************************************/
      double _Complex ** Rsl = ( irotl < nrot ) ? projector.rspin[0].R[irotl] : projector.rspin[0].IR[irotl-nrot];

      /******************************************************
       * loop on little group elements
       *  - rotations and rotation-reflections
       *  - at source, right-applied element
       *
       *  --- initial ---
       ******************************************************/
      for ( int irotr = 0; irotr < 2*nrot; irotr ++ )
      {

        if ( g_verbose > 3 ) fprintf ( stdout, "# [test_ref_rotation] right rotref %2d - %2d\n", irotr / nrot, irotr % nrot );
        if (irotr > nrot){
         rot_printf_matrix ( Rsl, dim,"test", stdout );
         exit(1);
        }

        double _Complex ** Rsr = ( irotr < nrot ) ? projector.rspin[0].R[irotr] : projector.rspin[0].IR[irotr-nrot];

        /******************************************************
         * residual rotation of complete 4x4 correlator
         * at sink ( left ) and source ( right )
         *
         * use Rsl, Rsr above
         *
         * tp.c[0] <- Rsl^H x tp.c[0] x Rsr
         ******************************************************/
        double _Complex ** R1 = init_2level_ztable ( dim, dim );
        double _Complex ** R2 = init_2level_ztable ( dim, dim );
        double _Complex ** R3 = init_2level_ztable ( dim, dim );

        /* R1 = C x Rsr */
        rot_mat_ti_mat ( R1, C, Rsr, dim );

        /* R2 = Rsl^+ x R1 */
        rot_mat_adj_ti_mat ( R2, Rsl, R1, dim);

        contract_diagram_mat_op_ti_zm4x4_field_ti_mat_op ( &R3, Rsl, 'H' , &C, Rsr, 'N', 1 );

        /* rot_printf_matrix_comp ( R2, R3, 4, "R23_comp", stdout ); */

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
                
            /* contract_diagram_zm4x4_field_eq_zm4x4_field_pl_zm4x4_field_ti_co ( 
                &(Cproj[ref_snk][ref_src][row_snk][row_src]), 
                &(Cproj[ref_snk][ref_src][row_snk][row_src]), &R2, zcoeff, 1 ); */

            rot_mat_pl_eq_mat_ti_co ( Cproj[ref_snk][ref_src][row_snk][row_src].c[0][0], R2, zcoeff, dim );

//            fprintf ( stdout, "# [test_ref_rotation] rotl %d %2d rotr %d %2d ref %2d %2d row %2d %2d z %25.16e %25.16e\n", 
//                irotl/nrot, irotl%nrot, irotr/nrot, irotr%nrot,
//                ref_snk, ref_src, row_snk, row_src, dgeps ( creal(zcoeff), 1.e-14 ), dgeps ( cimag(zcoeff), 1.e-14 ) );

          }  // end of loop on row_src
          }  // end of loop on row_snk

        }  // end of loop on ref_src
        }  // end of loop on ref_snk

        fini_2level_ztable ( &R1 );
        fini_2level_ztable ( &R2 );
        fini_2level_ztable ( &R3 );

      }  // end of loop on source rotations
    }  // end of loop on sink rotations

    /******************************************************
     * plot the projected data set
     ******************************************************/
    for ( int ref_snk = 0; ref_snk < irrep_dim; ref_snk++ ) {
    for ( int ref_src = 0; ref_src < irrep_dim; ref_src++ ) {

      for ( int row_snk = 0; row_snk < irrep_dim; row_snk++ ) {
      for ( int row_src = 0; row_src < irrep_dim; row_src++ ) {
        char name[400];
        sprintf ( name, "Cproj_%d_%d_%d_%d", ref_snk, ref_src, row_snk, row_src );
        rot_printf_matrix ( Cproj[ref_snk][ref_src][row_snk][row_src].c[0][0], dim, name, stdout );
      }}
    }}


    /******************************************************
     * check the rotation property
     ******************************************************/
    twopoint_function_check_reference_rotation ( Cproj[0][0][0], &projector, 1.e-12 );

    /******************************************************
     * deallocate
     ******************************************************/
    little_group_fini ( &little_group );

    fini_little_group_projector ( &projector );

    fini_2level_ztable ( &C );
    /* fini_6level_ztable ( &Cproj ); */

    for ( int i = 0; i < irrep_dim * irrep_dim * irrep_dim * irrep_dim ; i++ ) {
      twopoint_function_fini ( &( Cproj[0][0][0][i] ) );
    }
    fini_4level_2pttable ( &Cproj );
    exit(1);

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
    fprintf(stdout, "# [test_ref_rotation] %s# [test_ref_rotation] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_ref_rotation] %s# [test_ref_rotation] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
