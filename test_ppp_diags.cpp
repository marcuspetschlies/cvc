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
#include "table_init_c.h"
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

/****************************************************
 * reduce by permutation
 ****************************************************/
inline void reduce_permutation ( int ** p_list, const int num ) {

  /****************************************************
   * remove entries from permutations
   ****************************************************/
  for ( int i = 0; i < num; i++ ) {
    for ( int k = i+1; k < num; k++ ) {
      if ( ( i != k ) && p_list[k][3] &&  (
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
}  /* end of reduce_permutation */

inline int p_eq_q ( int const P[3], int const Q[3] ) {
  return ( ( P[0] == Q[0] ) && ( P[1] == Q[1] ) && ( P[2] == Q[2] ) );
}

inline int p_ne_q ( int const P[3], int const Q[3] ) {
  return ( ( P[0] != Q[0] ) || ( P[1] != Q[1] ) || ( P[2] != Q[2] ) );
}


inline void print_corr_name ( char * corr_name, const int q[3][3], const int b[3][3], const int p[6] ) {

  sprintf ( corr_name, "FX%d_FY%d_FZ%d-IX%d_IY%d_IZ%d-FX%d_FY%d_FZ%d-IX%d_IY%d_IZ%d-FX%d_FY%d_FZ%d-IX%d_IY%d_IZ%d",
             q[p[0]][0],  q[p[0]][1],  q[p[0]][2],
            -b[p[1]][0], -b[p[1]][1], -b[p[1]][2],
             q[p[2]][0],  q[p[2]][1],  q[p[2]][2],
            -b[p[3]][0], -b[p[3]][1], -b[p[3]][2],
             q[p[4]][0],  q[p[4]][1],  q[p[4]][2],
            -b[p[5]][0], -b[p[5]][1], -b[p[5]][2] );
}  /* end of print_corr_name */

/****************************************************
 * 
 ****************************************************/
int main(int argc, char **argv) {

  #if defined CUBIC_GROUP_DOUBLE_COVER
  char const little_group_list_filename[] = "little_groups_2Oh.tab";
  int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_double_cover;
#elif defined CUBIC_GROUP_SINGLE_COVER
  const char little_group_list_filename[] = "little_groups_Oh.tab";
  int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_single_cover;
#endif

  const int P2max = 4;

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
    fprintf(stderr, "[test_ppp_diags] Error from little_group_read_list, status was %d %s %d\n", nlg, __FILE__, __LINE__ );
    EXIT(2);
  }
  fprintf(stdout, "# [test_ppp_diags] number of little groups = %d\n", nlg);

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
  fprintf ( stdout, "# [test_ppp_diags] p level elements %4d\n", count );

  /****************************************************
   * remove entries from permutations
   ****************************************************/
  reduce_permutation ( p_list, count );

  /****************************************************
   * show remaining operators
   ****************************************************/
  int count2 = 0;
  for ( int i = 0; i < count; i++ ) {
    if ( p_list[i][3] ) {
      fprintf ( stdout, "p2 %4d   r = %3d %3d %3d   s = %3d %3d %3d   t = %3d %3d %3d\n", 
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

  fprintf ( stdout, "# [test_ppp_diags] p2 level elements %4d\n", count2 );

  int ** p2_list = init_2level_itable ( count2, 4 );
  count2 = 0;
  for ( int i = 0; i < count; i++ ) {
    if ( !p_list[i][3] ) continue;
    p2_list[count2][0] = p_list[i][0];
    p2_list[count2][1] = p_list[i][1];
    p2_list[count2][2] = p_list[i][2];
    p2_list[count2][3] = 1;
    count2++;
  }


  for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

    int const P[3] = {
        g_total_momentum_list[iptot][0],
        g_total_momentum_list[iptot][1],
        g_total_momentum_list[iptot][2] };

    /****************************************************
     * initialize and set projector
     * for current little group and irrep
     ****************************************************/
    little_group_projector_type projector;
   
    if ( ( exitstatus = init_little_group_projector ( &projector ) ) != 0 ) {
      fprintf ( stderr, "# [test_ppp_diags] Error from init_little_group_projector, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

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
     * parameters for setting the projector
     ****************************************************/
    int ref_row_target          = -1;     // no reference row for target irrep
    int * ref_row_spin          = NULL;   // no reference row for spin matrices
    int row_target              = -1;     // no target row
    int cartesian_list[1]       = { 1 };  // yes cartesian
    int parity_list[1]          = { 1 };  // intrinsic parity is +1
    const int ** momentum_list  = NULL;   // no momentum list given
    int bispinor_list[1]        = { 0 };  // bispinor no
    int J2_list[1]              = { 2 };  // spin 1

    /****************************************************
     * set the projector with the info we have
     ****************************************************/
    exitstatus = little_group_projector_set (
        &(projector),
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
      fprintf ( stderr, "[test_ppp_diags] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(3);
    }

    /****************************************************
     * reduce by lg rotations
     ****************************************************/
    for ( int i = 0; i < count2; i++ ) {
      int const Q[3] = {
        ( g_sink_momentum_list[p2_list[i][0]][0] + g_sink_momentum_list[p2_list[i][1]][0] + g_sink_momentum_list[p2_list[i][2]][0] ),
        ( g_sink_momentum_list[p2_list[i][0]][1] + g_sink_momentum_list[p2_list[i][1]][1] + g_sink_momentum_list[p2_list[i][2]][1] ),
        ( g_sink_momentum_list[p2_list[i][0]][2] + g_sink_momentum_list[p2_list[i][1]][2] + g_sink_momentum_list[p2_list[i][2]][2] ) };

      if ( p_ne_q ( P, Q ) ) continue;

      for ( int irot = 0; irot < projector.rp->n; irot++ ) {
        int q1[3], q2[3], q3[3];
        rot_point ( q1, g_sink_momentum_list[p2_list[i][0]], projector.rp->R[irot] );
        rot_point ( q2, g_sink_momentum_list[p2_list[i][1]], projector.rp->R[irot] );
        rot_point ( q3, g_sink_momentum_list[p2_list[i][2]], projector.rp->R[irot] );

        fprintf ( stdout, " q1 %3d %3d %3d  --->  Rq1 %3d %3d %3d\n", 
            g_sink_momentum_list[p2_list[i][0]][0],
            g_sink_momentum_list[p2_list[i][0]][1],
            g_sink_momentum_list[p2_list[i][0]][2],
            q1[0], q1[1], q1[2] );

        fprintf ( stdout, " q2 %3d %3d %3d  --->  Rq2 %3d %3d %3d\n", 
            g_sink_momentum_list[p2_list[i][1]][0],
            g_sink_momentum_list[p2_list[i][1]][1],
            g_sink_momentum_list[p2_list[i][1]][2],
            q2[0], q2[1], q2[2] );
        fprintf ( stdout, " q3 %3d %3d %3d  --->  Rq3 %3d %3d %3d\n\n\n", 
            g_sink_momentum_list[p2_list[i][2]][0],
            g_sink_momentum_list[p2_list[i][2]][1],
            g_sink_momentum_list[p2_list[i][2]][2],
            q3[0], q3[1], q3[2] );

        /****************************************************
         * check lg rotation equivalence
         ****************************************************/
        for ( int k = i+1; k < count2; k++ ) {
          if ( !p2_list[k][3] ) continue;

          int const b1[3] = {
            g_sink_momentum_list[p2_list[k][0]][0],
            g_sink_momentum_list[p2_list[k][0]][1],
            g_sink_momentum_list[p2_list[k][0]][2] };

          int const b2[3] = {
            g_sink_momentum_list[p2_list[k][1]][0],
            g_sink_momentum_list[p2_list[k][1]][1],
            g_sink_momentum_list[p2_list[k][1]][2] };

          int const b3[3] = {
            g_sink_momentum_list[p2_list[k][2]][0],
            g_sink_momentum_list[p2_list[k][2]][1],
            g_sink_momentum_list[p2_list[k][2]][2] };

          int const B[3] = {
              b1[0] + b2[0] + b3[0],
              b1[1] + b2[1] + b3[1],
              b1[2] + b2[2] + b3[2] };

          if ( p_ne_q ( P, B ) ) continue;

          fprintf ( stdout, "# [test_ppp_diags] comparing q1 %3d %3d %3d q2 %3d %3d %3d q3 %3d %3d %3d      b1 %3d %3d %3d b2 %3d %3d %3d b3 %3d %3d %3d\n",
              q1[0], q1[1], q1[2],
              q2[0], q2[1], q2[2],
              q3[0], q3[1], q3[2],
              b1[0], b1[1], b1[2],
              b2[0], b2[1], b2[2],
              b3[0], b3[1], b3[2] );

          if ( ( p_eq_q ( q1, b1 ) && 
                 p_eq_q ( q2, b2 ) && 
                 p_eq_q ( q3, b3 )    ) ||
               /*  */
               ( p_eq_q ( q1, b2 ) && 
                 p_eq_q ( q2, b3 ) && 
                 p_eq_q ( q3, b1 )    ) ||
               /*  */
               ( p_eq_q ( q1, b3 ) && 
                 p_eq_q ( q2, b1 ) && 
                 p_eq_q ( q3, b2 )    ) ||
               /*  */
               ( p_eq_q ( q1, b2 ) && 
                 p_eq_q ( q2, b1 ) && 
                 p_eq_q ( q3, b3 )    ) ||
               /*  */
               ( p_eq_q ( q1, b1 ) && 
                 p_eq_q ( q2, b3 ) && 
                 p_eq_q ( q3, b2 )    ) ||
               /*  */
               ( p_eq_q ( q1, b3 ) && 
                 p_eq_q ( q2, b2 ) && 
                 p_eq_q ( q3, b1 )    ) ) {

            p2_list[k][3] = 0;
 
            fprintf ( stdout, "# [test_ppp_diags] skipping q1 %3d %3d %3d q2 %3d %3d %3d q3 %3d %3d %3d      b1 %3d %3d %3d b2 %3d %3d %3d b3 %3d %3d %3d\n",
              q1[0], q1[1], q1[2],
              q2[0], q2[1], q2[2],
              q3[0], q3[1], q3[2],
              b1[0], b1[1], b1[2],
              b2[0], b2[1], b2[2],
              b3[0], b3[1], b3[2] );

          }
        }  /* end of loop on k */
      }  /* end of loop on rotations */
    }  /* end of loop on i */

   fini_little_group_projector ( &(projector) );

  }  /* end of loop on total momentum list */

  /****************************************************
   * show remaining operators
   ****************************************************/
  int count3 = 0;
  for ( int i = 0; i < count2; i++ ) {
    if ( p2_list[i][3] ) {
      fprintf ( stdout, "p3 %4d   r = %3d %3d %3d   s = %3d %3d %3d   t = %3d %3d %3d\n", 
          count3,
          g_sink_momentum_list[p2_list[i][0]][0],
          g_sink_momentum_list[p2_list[i][0]][1],
          g_sink_momentum_list[p2_list[i][0]][2],
          g_sink_momentum_list[p2_list[i][1]][0],
          g_sink_momentum_list[p2_list[i][1]][1],
          g_sink_momentum_list[p2_list[i][1]][2],
          g_sink_momentum_list[p2_list[i][2]][0],
          g_sink_momentum_list[p2_list[i][2]][1],
          g_sink_momentum_list[p2_list[i][2]][2] );
      count3++;
    }
  }

  fprintf ( stdout, "# [test_ppp_diags] p3 level elements %4d\n", count3 );

  int ** p3_list = init_2level_itable ( count3, 4 );
  count3 = 0;
  for ( int i = 0; i < count2; i++ ) {
    if ( !p2_list[i][3] ) continue;
    p3_list[count3][0] = p2_list[i][0];
    p3_list[count3][1] = p2_list[i][1];
    p3_list[count3][2] = p2_list[i][2];
    p3_list[count3][3] = 1;
    count3++;
  }

  /****************************************************
   * diagrams
   ****************************************************/


  char ** corr_name = init_2level_ctable ( count2 * count2 * 12, 120 );
  int * corr_name_use = init_1level_itable ( count2 * count2 * 12 );

  int count4=0;
  for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

    int const P[3] = {
        g_total_momentum_list[iptot][0],
        g_total_momentum_list[iptot][1],
        g_total_momentum_list[iptot][2] };

    for ( int i = 0; i < count2; i++ ) {

      int const q[3][3] = { 
        { g_sink_momentum_list[p2_list[i][0]][0],
          g_sink_momentum_list[p2_list[i][0]][1],
          g_sink_momentum_list[p2_list[i][0]][2] },
        { g_sink_momentum_list[p2_list[i][1]][0],
          g_sink_momentum_list[p2_list[i][1]][1],
          g_sink_momentum_list[p2_list[i][1]][2] },
        { g_sink_momentum_list[p2_list[i][2]][0],
          g_sink_momentum_list[p2_list[i][2]][1],
          g_sink_momentum_list[p2_list[i][2]][2] } };

      int const Q[3] = {
           q[0][0] + q[1][0] + q[2][0],
           q[0][1] + q[1][1] + q[2][1],
           q[0][2] + q[1][2] + q[2][2] };

      if ( p_ne_q ( P, Q ) ) continue;

      for ( int k = 0; k < count2; k++ ) {

        int const b[3][3] = {
          { g_sink_momentum_list[p2_list[k][0]][0],
            g_sink_momentum_list[p2_list[k][0]][1],
            g_sink_momentum_list[p2_list[k][0]][2] },
          { g_sink_momentum_list[p2_list[k][1]][0],
            g_sink_momentum_list[p2_list[k][1]][1],
            g_sink_momentum_list[p2_list[k][1]][2] },
          { g_sink_momentum_list[p2_list[k][2]][0],
            g_sink_momentum_list[p2_list[k][2]][1],
            g_sink_momentum_list[p2_list[k][2]][2] } };

        int const B[3] = {
              b[0][0] + b[1][0] + b[2][0],
              b[0][1] + b[1][1] + b[2][1],
              b[0][2] + b[1][2] + b[2][2] };

        if ( p_ne_q ( P, B ) ) continue;

        int const perm0[6]  = { 0,0,1,1,2,2 };
        print_corr_name ( corr_name[12*count4 +  0], q, b, perm0 );

        int const perm1[6]  = { 0,0,1,2,2,1 };
        print_corr_name ( corr_name[12*count4 +  1], q, b, perm1 );

        int const perm2[6]  = { 0,0,2,1,1,2 };
        print_corr_name ( corr_name[12*count4 +  2], q, b, perm2 );

        int const perm3[6]  = { 0,0,2,2,1,1 };
        print_corr_name ( corr_name[12*count4 +  3], q, b, perm3 );

        int const perm4[6]  = { 0,1,2,2,1,0 };
        print_corr_name ( corr_name[12*count4 +  4], q, b, perm4 );

        int const perm5[6]  = { 0,1,1,0,2,2 };
        print_corr_name ( corr_name[12*count4 +  5], q, b, perm5 );

        int const perm6[6]  = { 0,1,1,2,2,0 };
        print_corr_name ( corr_name[12*count4 +  6], q, b, perm6 );

        int const perm7[6]  = { 0,1,0,2,1,2 };
        print_corr_name ( corr_name[12*count4 +  7], q, b, perm7 );

        int const perm8[6]  = { 0,2,2,1,1,0 };
        print_corr_name ( corr_name[12*count4 +  8], q, b, perm8 );

        int const perm9[6]  = { 0,2,1,0,2,1 };
        print_corr_name ( corr_name[12*count4 +  9], q, b, perm9 );

        int const perm10[6] = { 0,2,1,1,2,0 };
        print_corr_name ( corr_name[12*count4 + 10], q, b, perm10 );

        int const perm11[6] = { 0,2,2,0,1,1 };
        print_corr_name ( corr_name[12*count4 + 11], q, b, perm11 );

        corr_name_use[count4] = 1;
        count4++;
      }
    }
  }  /* end of loop on total momenta */
 

  for ( int i =0; i < count4; i++ ) {

    for ( int k = i+1; k < count4; k++ ) {
      if ( corr_name_use[k] &&  strcmp( corr_name[i], corr_name[k] ) == 0 ) {
        corr_name_use[k] = 0;
      }
    }
  }

  int count5 = 0;
  fprintf( stdout, "# p4 level %4d\n", count4 );
  for ( int i =0; i < 12*count4; i++ ) {
    if ( corr_name_use[i] ) {
      fprintf( stdout, "  %6d %s\n", i, corr_name[i] );
      count5++;
    }
  }

  fprintf( stdout, "# p5 level %4d\n", count5 );



  fini_2level_ctable ( &corr_name );
  fini_1level_itable ( &corr_name_use );

  /****************************************************
   * finalize
   ****************************************************/

   for ( int i = 0; i < nlg; i++ ) {
     little_group_fini ( &(lg[i]) );
   }
   free ( lg );


  fini_2level_itable ( &p_list );
  fini_2level_itable ( &p2_list );
  fini_2level_itable ( &p3_list );


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
