/****************************************************
 * test_ppp_diags
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
#include "gamma.h"
#include "group_projection.h"
#include "little_group_projector_set.h"

using namespace cvc;

void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}

int main(int argc, char **argv) {

  #if defined CUBIC_GROUP_DOUBLE_COVER
  char const little_group_list_filename[] = "little_groups_2Oh.tab";
  int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_double_cover;
#elif defined CUBIC_GROUP_SINGLE_COVER
  const char little_group_list_filename[] = "little_groups_Oh.tab";
  int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_single_cover;
#endif

  const int P2max = 3;

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;

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
        fprintf(stdout, "# [test_ppp_diags] exit\n");
        exit(1);
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_ppp_diags] Reading input from file %s\n", filename);
  read_input_parser(filename);

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


  int ** p_list = init_2level_itable ( g_sink_momentum_number * g_sink_momentum_number * g_sink_momentum_number, 4 );

  /****************************************************
   * fill list
   * reduce by cut-off on P2
   ****************************************************/
  int count = 0;
  for ( int ip1 = 0; ip1 < g_sink_momentum_number; ip1++ ) {
  for ( int ip2 = 0; ip2 < g_sink_momentum_number; ip2++ ) {
  for ( int ip3 = 0; ip3 < g_sink_momentum_number; ip3++ ) {

    int const P[3] = {
      ( g_sink_momentum_list[ip1][0] + g_sink_momentum_list[ip2][0] + g_sink_momentum_list[ip3][0] ),
      ( g_sink_momentum_list[ip1][1] + g_sink_momentum_list[ip2][1] + g_sink_momentum_list[ip3][1] ),
      ( g_sink_momentum_list[ip1][2] + g_sink_momentum_list[ip2][2] + g_sink_momentum_list[ip3][2] ) };

    int const P2 = P[0] * P[0] + P[1] * P[1] + P[2] * P[2];
    if ( P2 > P2max ) continue;

    p_list[count][0] = ip1;
    p_list[count][1] = ip2;
    p_list[count][2] = ip3;
    p_list[count][3] = 1;
    count++;

  }}}

  /****************************************************
   * remove entries from permutations
   ****************************************************/
  for ( int i = 0; i < count; i++ ) {
    for ( int k = i+1; k < count; k++ ) {
      if ( p_list[k][3] &&  (
          ( ( p_list[i][0] == p_list[k][0] ) && 
            ( p_list[i][1] == p_list[k][1] ) && 
            ( p_list[i][2] == p_list[k][2] )    )  || 
          /* */
          ( ( p_list[i][0] == p_list[k][1] ) && 
            ( p_list[i][1] == p_list[k][2] ) && 
            ( p_list[i][2] == p_list[k][0] )    )  || 
          /* */
          ( ( p_list[i][0] == p_list[k][2] ) && 
            ( p_list[i][1] == p_list[k][0] ) && 
            ( p_list[i][2] == p_list[k][1] )    )  || 
          /* */
          ( ( p_list[i][0] == p_list[k][1] ) && 
            ( p_list[i][1] == p_list[k][0] ) && 
            ( p_list[i][2] == p_list[k][2] )    )  || 
          /* */
          ( ( p_list[i][0] == p_list[k][0] ) && 
            ( p_list[i][1] == p_list[k][2] ) && 
            ( p_list[i][2] == p_list[k][1] )    )  || 
          /* */
          ( ( p_list[i][0] == p_list[k][2] ) && 
            ( p_list[i][1] == p_list[k][1] ) && 
            ( p_list[i][2] == p_list[k][0] )    ) ) )  {

        p_list[k][3] = 0;
      }

    }
  }

  /****************************************************
   * show remaining operators
   ****************************************************/
  int count2 = 0;
  for ( int i = 0; i < count; i++ ) {
    if ( p_list[i][3] ) {
      fprintf ( stdout, " %4d   r = %3d %3d %3d   s = %3d %3d %3d   t = %3d %3d %3d\n", 
          count2,
          g_sink_momentum_list[p_list[i][0]][0],
          g_sink_momentum_list[p_list[i][0]][1],
          g_sink_momentum_list[p_list[i][0]][2],
          g_sink_momentum_list[p_list[i][1]][0],
          g_sink_momentum_list[p_list[i][1]][1],
          g_sink_momentum_list[p_list[i][1]][2],
          g_sink_momentum_list[p_list[i][2]][0],
          g_sink_momentum_list[p_list[i][2]][1],
          g_sink_momentum_list[p_list[i][2]][2] );
      count2++;
    }
  }

  int ** p2_list = init_2level_itable ( count2, 4 );
  int k = 0;
  for ( int i = 0; i < count; i++ ) {
    if ( !p_list[i][3] ) continue;
    p2_list[k][0] = p_list[i][0];
    p2_list[k][1] = p_list[i][1];
    p2_list[k][2] = p_list[i][2];
    p2_list[k][3] = 1;
    k++;
  }

  /****************************************************
   * reduce by lg rotations
   ****************************************************/
  for ( int i = 0; i < count2; i++ ) {
    int const P[3] = {
      ( g_sink_momentum_list[p2_list[i][0]][0] + g_sink_momentum_list[p2_list[i][1]][0] + g_sink_momentum_list[p2_list[i][2]][0] ),
      ( g_sink_momentum_list[p2_list[i][0]][1] + g_sink_momentum_list[p2_list[i][1]][1] + g_sink_momentum_list[p2_list[i][2]][1] ),
      ( g_sink_momentum_list[p2_list[i][0]][2] + g_sink_momentum_list[p2_list[i][1]][2] + g_sink_momentum_list[p2_list[i][2]][2] ) };


    /****************************************************
     * find lg and ref frame rotation
     ****************************************************/
    int refframerot, Pref[3];
    exitstatus = get_reference_rotation ( Pref, &refframerot, P );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_ppp_diags] Error from get_reference_rotation for P = (%3d %3d %3d), status was %d %s %d\n", 
          P[0], P[1], P[2],
          exitstatus, __FILE__, __LINE__);
      EXIT(4);
    } else if ( g_verbose > 1 ) {
      fprintf ( stdout, "# [test_ppp_diags] twopoint_function Ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n",
          P[0], P[1], P[2], refframerot, Pref[0], Pref[1], Pref[2]);
    }

    int ilg = -1;
    for ( int k = 0; k < nlg; k++ ) {
      if ( 
          ( Pref[0] == lg[k].d[0] ) && 
          ( Pref[1] == lg[k].d[1] ) && 
          ( Pref[2] == lg[k].d[2] ) ) {
        ilg = k;
        break;
      }
    }
    if ( ilg == -1 ) {
      fprintf ( stderr, "[test_ppp_diags] Error, could not find little group\n" );
      EXIT(1);
    }
    fprintf ( stdout, "# [test_ppp_diags] P = (%3d %3d %3d) Pref (%3d %3d %3d)  lg %s\n",
        P[0], P[1], P[2],
        Pref[0], Pref[1], Pref[2],
        lg[ilg].name );


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
    int row_target              = -1;     // no target row
    int cartesian_list[1]       = { 1 };  // yes cartesian
    int parity_list[1]          = { 1 };  // intrinsic parity is +1
    const int ** momentum_list  = NULL;   // no momentum list given
    int bispinor_list[1]        = { 0 };  // bispinor no
    int J2_list[1]              = { 2 };  // spin 1/2

    /****************************************************
     * set the projector with the info we have
     ****************************************************/
    exitstatus = little_group_projector_set (
        &projector,
        &(lg[ilg]),
        lg[ilg].lirrep[0],
        row_target,
        1,
        J2_list,
        momentum_list,
        bispinor_list,
        parity_list,
        cartesian_list,
        ref_row_target,
        ref_row_spin,
        "test",
        refframerot );

    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[piN2piN_projection] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(3);
    }



    /****************************************************
     * check lg rotation equivalence
     ****************************************************/
    for ( int k = i+1; k < count2; k++ ) {
      int const Q[3] = {
        ( g_sink_momentum_list[p2_list[k][0]][0] + g_sink_momentum_list[p2_list[k][1]][0] + g_sink_momentum_list[p2_list[k][2]][0] ),
        ( g_sink_momentum_list[p2_list[k][0]][1] + g_sink_momentum_list[p2_list[k][1]][1] + g_sink_momentum_list[p2_list[k][2]][1] ),
        ( g_sink_momentum_list[p2_list[k][0]][2] + g_sink_momentum_list[p2_list[k][1]][2] + g_sink_momentum_list[p2_list[k][2]][2] ) };
      if ( 
          ( P[0] != Q[0] ) ||
          ( P[1] != Q[1] ) ||
          ( P[2] != Q[2] ) ) continue;

    }
  }


  
  /****************************************************
   * finalize
   ****************************************************/

  fini_2level_itable ( &p_list );
  fini_2level_itable ( &p2_list );


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_ppp_diags] %s# [test_ppp_diags] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_ppp_diags] %s# [test_ppp_diags] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
