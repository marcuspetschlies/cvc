/****************************************************
 * test_lg_momentum_classes.cpp
 *
 * Sat May 13 22:36:53 CEST 2017
 *
 * PURPOSE:
 * DONE:
 * TODO:
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
#include "group_projection.h"
#include "little_group_projector_set.h"

using namespace cvc;

int main(int argc, char **argv) {

#if defined CUBIC_GROUP_DOUBLE_COVER
  char const little_group_list_filename[] = "little_groups_2Oh.tab";
#elif defined CUBIC_GROUP_SINGLE_COVER
  const char little_group_list_filename[] = "little_groups_Oh.tab";
#endif

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
        fprintf(stdout, "# [test_lg_momentum_classes] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_lg_momentum_classes] Reading input from file %s\n", filename);
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
  if(g_cart_id == 0) fprintf(stdout, "# [test_lg_momentum_classes] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_lg_momentum_classes] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_lg_momentum_classes] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /****************************************************/
  /****************************************************/

  /****************************************************
   * initialize and set geometry fields
   ****************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0) {
    fprintf(stderr, "[test_lg_momentum_classes] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
    fprintf(stderr, "[test_lg_momentum_classes] Error from little_group_read_list, status was %d %s %d\n", nlg, __FILE__, __LINE__ );
    EXIT(2);
  }
  fprintf(stdout, "# [test_lg_momentum_classes] number of little groups = %d\n", nlg);

  little_group_show ( lg, stdout, nlg );

  /****************************************************
   * initialize spin-1 rotation table
   * here in spherical basis
   ****************************************************/

  rot_mat_table_type Rspin1;
  init_rot_mat_table ( &Rspin1 );
  exitstatus = set_rot_mat_table_spin ( &Rspin1, 2, 0 );

  /****************************************************
   * loop on chosen total momenta
   ****************************************************/
  for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ )
  {

    int ** momentum_pair_selection = init_2level_itable ( g_sink_momentum_number * g_sink_momentum_number, 5 );
    if ( momentum_pair_selection == NULL ) {
      fprintf(stderr, "[test_lg_momentum_classes] Error from init_2level_itable, status was %d %s %d\n", nlg, __FILE__, __LINE__ );
      EXIT(2);
    }


    /****************************************************
     * set momentum vector
     ****************************************************/
    int Pref[3], refframerot;
    int const P[3] = {  
      g_total_momentum_list[iptot][0],
      g_total_momentum_list[iptot][1],
      g_total_momentum_list[iptot][2] };
      /* lg[ilg].d[0], lg[ilg].d[1], lg[ilg].d[2]  */

    /****************************************************
     * get reference frame rotation
     ****************************************************/
    exitstatus = get_reference_rotation ( Pref, &refframerot, P );
    if ( refframerot == -1 ) refframerot = 0;
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_lg_momentum_classes] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(4);
    } else if ( g_verbose > 1 ) {
      fprintf ( stdout, "# [test_lg_momentum_classes] P = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n",
      P[0], P[1], P[2], refframerot, Pref[0], Pref[1], Pref[2]);
    }

    /****************************************************
     * get little group
     ****************************************************/
    int ilg = -1;
    for ( int i = 0; i < nlg; i++ ) {
      if ( ( Pref[0] == lg[i].d[0] ) && 
           ( Pref[1] == lg[i].d[1] ) && 
           ( Pref[2] == lg[i].d[2] )  ) {
        ilg = i;
      }
    }
    if ( ilg == -1 ) {
      fprintf ( stderr, "[test_lg_momentum_classes] Error, could not find lg %s %d\n", __FILE__, __LINE__);
      EXIT(4);
    } else {
      fprintf ( stdout, "# [test_lg_momentum_classes] P %3d %3d %3d   Pref %3d %3d %3d   lg %s\n",
          P[0], P[1], P[2], Pref[0], Pref[1], Pref[2], lg[ilg].name );
    }

    /****************************************************
     * make cartesian-basis spin-1 reference frame
     * rotation
     ****************************************************/
    double _Complex **refframerot_p = rot_init_rotation_matrix ( 3 );
#if defined CUBIC_GROUP_DOUBLE_COVER
    rot_mat_spin1_cartesian ( refframerot_p, cubic_group_double_cover_rotations[refframerot].n, cubic_group_double_cover_rotations[refframerot].w );
#elif defined CUBIC_GROUP_SINGLE_COVER
    rot_rotation_matrix_spherical_basis_Wigner_D ( refframerot_p, 2, cubic_group_rotations_v2[refframerot].a );
    rot_spherical2cartesian_3x3 ( refframerot_p, refframerot_p );
#endif
    if ( ! ( rot_mat_check_is_real_int ( refframerot_p, 3) ) ) {
      fprintf(stderr, "[test_lg_momentum_classes] Error rot_mat_check_is_real_int refframerot_p %s %d\n", __FILE__, __LINE__);
      EXIT(72);
    }

    /****************************************************
     * list of rotations all rotations, spin 1, spherical
     ****************************************************/
    rot_mat_table_type Rpvec;
    init_rot_mat_table ( &Rpvec );

    exitstatus = alloc_rot_mat_table ( &Rpvec, "SU2", "spin1", 3, lg[ilg].nr );

    for ( int irot = 0; irot < Rpvec.n; irot++ ) {

      /* copy rotation, transform spin-1 to Cartesian basis  */
      rot_spherical2cartesian_3x3 ( Rpvec.R[irot], Rspin1.R[lg[ilg].r[irot]] );
      Rpvec.rid[irot] = lg[ilg].r[irot];

      if ( ! rot_mat_check_is_real_int ( Rpvec.R[irot], 3 ) ) {
        fprintf ( stderr, "[test_lg_momentum_classes] Error, R %d / %d is not real int\n", irot, Rpvec.rid[irot] );
        EXIT(1);
      }


      /* copy reflection-rotation, transform spin-1 to Cartesian basis  */
      rot_spherical2cartesian_3x3 ( Rpvec.IR[irot], Rspin1.IR[lg[ilg].rm[irot]] );
      Rpvec.rmid[irot] = lg[ilg].rm[irot];

      if ( ! rot_mat_check_is_real_int ( Rpvec.IR[irot], 3 ) ) {
        fprintf ( stderr, "[test_lg_momentum_classes] Error, R %d / %d is not real int\n", irot, Rpvec.rmid[irot] );
        EXIT(1);
      }
    }

    for ( int irot = 0; irot < Rpvec.n; irot++ ) {
      // R <- Rref x R
      rot_mat_ti_mat ( Rpvec.R[irot], refframerot_p, Rpvec.R[irot], 3 );
      // R <- R x Rref^+
      rot_mat_ti_mat_adj ( Rpvec.R[irot], Rpvec.R[irot], refframerot_p, 3 );
 
      // IR <- Rref x IR
      rot_mat_ti_mat ( Rpvec.IR[irot], refframerot_p, Rpvec.IR[irot], 3 );
      // IR <- IR x Rref^+
      rot_mat_ti_mat_adj ( Rpvec.IR[irot], Rpvec.IR[irot], refframerot_p, 3 );

    }

    /****************************************************
     * check pairs
     ****************************************************/
    int count_pairs = 0;
    for ( int i = 0; i < g_sink_momentum_number; i++ ) {
      int const p1[3] = {
        g_sink_momentum_list[i][0],
        g_sink_momentum_list[i][1],
        g_sink_momentum_list[i][2] };

    for ( int k = 0; k < g_sink_momentum_number; k++ ) {
      int const p2[3] = {
        g_sink_momentum_list[k][0],
        g_sink_momentum_list[k][1],
        g_sink_momentum_list[k][2] };

      /****************************************************
       * discard momentum pairs,
       * which do not add up to P
       ****************************************************/
      if ( ! (
                 ( ( p1[0] + p2[0] ==  P[0] ) && ( p1[1] + p2[1] ==  P[1] ) && ( p1[2] + p2[2] ==  P[2] ) )
            /*  || ( ( p1[0] + p2[0] == -P[0] ) && ( p1[1] + p2[1] == -P[1] ) && ( p1[2] + p2[2] == -P[2] ) ) */
            ) ) continue;


      /****************************************************
       * initialize momentum_pair_selection entry
       ****************************************************/
      momentum_pair_selection[count_pairs][0] = i;
      momentum_pair_selection[count_pairs][1] = k;
      momentum_pair_selection[count_pairs][2] = -1;
      momentum_pair_selection[count_pairs][3] = -1;
      momentum_pair_selection[count_pairs][4] = -1;
      count_pairs++;

      if ( g_verbose > 0 ) fprintf ( stdout, "# [test_lg_momentum_classes] P %3d %3d %3d i %3d k %3d pair %4d   p1 %3d %3d %3d p2 %3d %3d %3d\n", 
          P[0], P[1], P[2], i, k, count_pairs-1, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2] );
    }}


    for ( int i = 0; i < count_pairs; i++ ) {

      /****************************************************
       * loop on proper rotations
       ****************************************************/
      for ( int irot = 0; irot < Rpvec.n; irot++ ) {

        int p1rot[3] = {0,0,0}, p2rot[3] = {0,0,0};
        /* if ( g_verbose > 0 ) fprintf ( stdout, "# [test_lg_momentum_classes] momentum_pair_selection %3d  i %3d  k %3d  flag %2d\n",
            i, momentum_pair_selection[i][0], momentum_pair_selection[i][1], momentum_pair_selection[i][2] );

        fflush ( stdout );
        fflush ( stderr );
        */


        rot_point ( p1rot, g_sink_momentum_list[momentum_pair_selection[i][0]], Rpvec.R[irot] );
        rot_point ( p2rot, g_sink_momentum_list[momentum_pair_selection[i][1]], Rpvec.R[irot] );

        for ( int k = 0; k < count_pairs; k++ ) {

          if ( momentum_pair_selection[k][2] != -1 ) continue;

          if ( ( p1rot[0] == g_sink_momentum_list[momentum_pair_selection[k][0]][0] &&
                 p1rot[1] == g_sink_momentum_list[momentum_pair_selection[k][0]][1] &&
                 p1rot[2] == g_sink_momentum_list[momentum_pair_selection[k][0]][2] )
            && ( p2rot[0] == g_sink_momentum_list[momentum_pair_selection[k][1]][0] &&
                 p2rot[1] == g_sink_momentum_list[momentum_pair_selection[k][1]][1] &&
                 p2rot[2] == g_sink_momentum_list[momentum_pair_selection[k][1]][2] ) ) {
            momentum_pair_selection[k][2] = i;
            momentum_pair_selection[k][3] = lg[ilg].r[irot]+1;
          }
        }

      }  /* end of loop on proper rotations */

      /****************************************************
       * loop on rotation-reflections
       ****************************************************/
      for ( int irot = 0; irot < Rpvec.n ; irot++ ) {

        int p1rot[3] = {0,0,0}, p2rot[3] = {0,0,0};
        /* if ( g_verbose > 0 ) fprintf ( stdout, "# [test_lg_momentum_classes] momentum_pair_selection %3d  i %3d  k %3d  flag %2d\n",
            i, momentum_pair_selection[i][0], momentum_pair_selection[i][1], momentum_pair_selection[i][2] );
        */

        rot_point ( p1rot, g_sink_momentum_list[momentum_pair_selection[i][0]], Rpvec.IR[irot] );
        rot_point ( p2rot, g_sink_momentum_list[momentum_pair_selection[i][1]], Rpvec.IR[irot] );

        for ( int k = 0; k < count_pairs; k++ ) {

          if ( momentum_pair_selection[k][2] != -1 ) continue;

          if ( ( p1rot[0] == g_sink_momentum_list[momentum_pair_selection[k][0]][0] &&
                 p1rot[1] == g_sink_momentum_list[momentum_pair_selection[k][0]][1] &&
                 p1rot[2] == g_sink_momentum_list[momentum_pair_selection[k][0]][2] )
            && ( p2rot[0] == g_sink_momentum_list[momentum_pair_selection[k][1]][0] &&
                 p2rot[1] == g_sink_momentum_list[momentum_pair_selection[k][1]][1] &&
                 p2rot[2] == g_sink_momentum_list[momentum_pair_selection[k][1]][2] ) ) {
            momentum_pair_selection[k][2] = i;
            momentum_pair_selection[k][4] = lg[ilg].rm[irot]+1;
          }
        }

      }  /* end of loop on proper rotations */

    }  /* end of loop on pairs */

    fprintf ( stdout, "%10s  %3s    %3s %3s  (%3s %3s %3s) (%3s %3s %3s)      %3s  %3s %3s  (%3s %3s %3s) (%3s %3s %3s)\n",
        "# lg", "#pp", "#p1", "#p2", 
        "p1x", "p1y", "p1z",
        "p2x", "p2y", "p2z",
        "#qq", "#R", "#IR",
        "q1x", "q1y", "q1z",
        "q2x", "q2y", "q2z" );

    for ( int i = 0; i < count_pairs; i++ ) {
      fprintf ( stdout, "%10s  %3d    %3d %3d  (%3d %3d %3d) (%3d %3d %3d)      %3d  %3d %3d  (%3d %3d %3d) (%3d %3d %3d)\n",
          lg[ilg].name, i, momentum_pair_selection[i][0], momentum_pair_selection[i][1],
          g_sink_momentum_list[momentum_pair_selection[i][0]][0],
          g_sink_momentum_list[momentum_pair_selection[i][0]][1],
          g_sink_momentum_list[momentum_pair_selection[i][0]][2],
          g_sink_momentum_list[momentum_pair_selection[i][1]][0],
          g_sink_momentum_list[momentum_pair_selection[i][1]][1],
          g_sink_momentum_list[momentum_pair_selection[i][1]][2],
          momentum_pair_selection[i][2],
          momentum_pair_selection[i][3],
          momentum_pair_selection[i][4],
          g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][0] ][0],
          g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][0] ][1],
          g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][0] ][2],
          g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][1] ][0],
          g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][1] ][1],
          g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][1] ][2] );
    }

    sprintf ( filename, "momentum_pairs-PX%d_PY%d_PZ%d.lst" , P[0], P[1], P[2] );
    FILE * ofs = fopen ( filename, "w");
    fprintf( ofs, "p1_list=( " );
    for ( int i = 0, icomp=-1; i < count_pairs; i++ ) {
      if ( momentum_pair_selection[i][2] > icomp ) { 
        fprintf ( ofs, "\"%d,%d,%d\" ", 
            g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][0] ][0],
            g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][0] ][1],
            g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][0] ][2] );
        icomp = momentum_pair_selection[i][2];
      }
    }
    fprintf( ofs, " )\n" );

    fprintf( ofs, "\np2_list=( " );
    for ( int i = 0, icomp=-1; i < count_pairs; i++ ) {
      if ( momentum_pair_selection[i][2] > icomp ) { 
        fprintf ( ofs, "\"%d,%d,%d\" ", 
            g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][1] ][0],
            g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][1] ][1],
            g_sink_momentum_list[ momentum_pair_selection[momentum_pair_selection[i][2]][1] ][2] );
        icomp = momentum_pair_selection[i][2];
      }
    }
    fprintf( ofs, " )\n" );

    fclose ( ofs );

    fini_2level_itable ( &momentum_pair_selection );
    fini_rot_mat_table ( &Rpvec );
    rot_fini_rotation_matrix ( &refframerot_p );

  }  /* end of loop on total momenta */

  fini_rot_mat_table ( &Rspin1 );

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
  free_geometry();

#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_lg_momentum_classes] %s# [test_lg_momentum_classes] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_lg_momentum_classes] %s# [test_lg_momentum_classes] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif
  return(0);
}
