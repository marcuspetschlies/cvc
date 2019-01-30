/****************************************************
 * test_correlator_read
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
 ***********************************************************/
int main(int argc, char **argv) {
 
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
  fprintf(stdout, "[test_correlator_read] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /***********************************************************
   * report git version
   ***********************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_correlator_read] git version = %s\n", g_gitversion);
  }

  /***********************************************************
   * set geometry
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0 ) {
    fprintf(stderr, "[test_correlator_read] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
   * loop on 2-point functions
   ******************************************************/
  for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

    twopoint_function_type * tp = &(g_twopoint_function_list[i2pt]);

    /******************************************************
     * read twopoint function data
     ******************************************************/
    exitstatus = twopoint_function_correlator_from_h5_file ( tp, io_proc );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_correlator_read] Error from twopoint_function_correlator_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(2);
    }

    /******************************************************
     * print the 2-point function parameters
     ******************************************************/
    sprintf ( filename, "twopoint_function_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[test_correlator_read] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    twopoint_function_print ( tp, "TWPT", ofs );

    twopoint_function_show_data ( tp, ofs );

    fclose ( ofs );

  }  // end of loop on reading of 2-point functions

  /****************************************************
   * take all info about group etc. for projector
   * from twopoint function zero
   ****************************************************/
  twopoint_function_type * tp = &(g_twopoint_function_list[0]);

  /****************************************************
   * read little group parameters
   ****************************************************/
  little_group_type little_group;
  if ( ( exitstatus = little_group_read ( &little_group, tp->group, little_group_list_filename ) ) != 0 ) {
    fprintf ( stderr, "[test_correlator_read] Error from little_group_read, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(2);
  }
    
  sprintf ( filename, "little_group_%d.show", 0 );
  if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
    fprintf ( stderr, "[test_correlator_read] Error from fopen %s %d\n", __FILE__, __LINE__ );
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
    fprintf ( stderr, "# [test_correlator_read] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
      tp->pf1[0] + tp->pf2[0],
      tp->pf1[1] + tp->pf2[1],
      tp->pf1[2] + tp->pf2[2] };

  if ( g_verbose > 3 ) fprintf ( stdout, "# [test_correlator_read] twopoint_function Ptot = %3d %3d %3d\n", Ptot[0], Ptot[1], Ptot[2] );

  /****************************************************
   * do we need a reference frame rotation ?
   ****************************************************/
  exitstatus = get_reference_rotation ( Pref, &refframerot, Ptot );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[test_correlator_read] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(4);
  } else if ( g_verbose > 1 ) {
    fprintf ( stdout, "# [test_correlator_read] twopoint_function Ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n", 
        Ptot[0], Ptot[1], Ptot[2], refframerot, Pref[0], Pref[1], Pref[2]);
  }

  fflush ( stdout );

  /****************************************************
   * set the projector with the info we have
   ****************************************************/
  exitstatus = little_group_projector_set (
      &projector,
      &little_group,
      tp->irrep ,
      row_target,
      1,
      J2_list,
      momentum_list,
      bispinor_list,
      parity_list,
      cartesian_list,
      ref_row_target,
      ref_row_spin,
      tp->type,
      refframerot );

  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[test_correlator_read] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(3);
  }

  sprintf ( filename, "little_group_projector_%d.show", 0 );
  if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
    fprintf ( stderr, "[test_correlator_read] Error from fopen %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }
  exitstatus = little_group_projector_show ( &projector, ofs, 1 );
  fclose ( ofs );

  /****************************************************
   * check, that projector has correct d-vector
   ****************************************************/
  if ( ( projector.P[0] != Ptot[0] ) || ( projector.P[1] != Ptot[1] ) || ( projector.P[2] != Ptot[2] ) ) {
    fprintf ( stderr, "[test_correlator_read] Error, projector P != Ptot\n" );
    EXIT(12);
  } else {
    if ( g_verbose > 2 ) fprintf ( stdout, "# [test_correlator_read] projector P == Ptot\n" );
  }

  /****************************************************
   * check the transformation behaviour with respect
   * to reference indices
   *
   * NOTE: Assume, that everything relevant has been read
   ****************************************************/
  /* twopoint_function_check_reference_rotation ( g_twopoint_function_list, &projector, 5.e-12 ); */
  twopoint_function_check_reference_rotation_vector_spinor ( g_twopoint_function_list, &projector, 5.e-12 );

  /******************************************************
   * deallocate space inside little_group
   ******************************************************/
  little_group_fini ( &little_group );

  /******************************************************
   * deallocate space inside projector
   ******************************************************/
  fini_little_group_projector ( &projector );

  /******************************************************/
  /******************************************************/

  /******************************************************
   * finalize
   *
   * free the allocated memory, finalize
   ******************************************************/

  for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {
    twopoint_function_fini ( &(g_twopoint_function_list[i2pt]) );
  }

  free_geometry();

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_correlator_read] %s# [test_correlator_read] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_correlator_read] %s# [test_correlator_read] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
