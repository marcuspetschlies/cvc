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

  rot_mat_table_type Rpvec;
  init_rot_mat_table ( &Rpvec );
  exitstatus = set_rot_mat_table_spin ( &Rpvec, 2, 0 );

  /****************************************************
   * loop on little groups
   ****************************************************/
  for ( int ilg = 0; ilg < nlg; ilg++ )
  {

    int ** momentum_pair_selection = init_2level_itable ( g_sink_momentum_number * g_sink_momentum_number, 5 );
    if ( momentum_pair_selection == NULL ) {
      fprintf(stderr, "[test_lg_momentum_classes] Error from init_2level_itable, status was %d %s %d\n", nlg, __FILE__, __LINE__ );
      EXIT(2);
    }

    int const P[3] = {  
      lg[ilg].d[0],
      lg[ilg].d[1],
      lg[ilg].d[2] };

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

      if ( ! (
                 ( ( p1[0] + p2[0] ==  P[0] ) && ( p1[1] + p2[1] ==  P[1] ) && ( p1[2] + p2[2] ==  P[2] ) )
              || ( ( p1[0] + p2[0] == -P[0] ) && ( p1[1] + p2[1] == -P[1] ) && ( p1[2] + p2[2] == -P[2] ) )
            ) ) continue;


      momentum_pair_selection[count_pairs][0] = i;
      momentum_pair_selection[count_pairs][1] = k;
      momentum_pair_selection[count_pairs][2] = -1;
      momentum_pair_selection[count_pairs][3] = -1;
      momentum_pair_selection[count_pairs][4] = -1;
      count_pairs++;

      if ( g_verbose > 0 ) fprintf ( stdout, "# [test_lg_momentum_classes] P %3d %3d %3d  pair %4d   p1 %3d %3d %3d p2 %3d %3d %3d\n", 
          P[0], P[1], P[2], count_pairs, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2] );
    }}


    for ( int i = 0; i < count_pairs; i++ ) {

      /****************************************************
       * loop on proper rotations
       ****************************************************/
      for ( int irot = 0; irot < lg[ilg].nr ; irot++ ) {

        double _Complex **C = rot_init_rotation_matrix ( 3  );

        rot_spherical2cartesian_3x3 ( C, Rpvec.R[lg[ilg].r[irot]] );

        if ( ! rot_mat_check_is_real_int ( C, 3 ) ) {
          fprintf ( stderr, "[test_lg_momentum_classes] Error, R %d / %d is not real int\n", irot, lg[ilg].r[irot] );
          EXIT(1);
        /* } else {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [test_lg_momentum_classes] rot %d / %d is okay\n", irot, lg[ilg].r[irot] );
          */
        }

        int p1rot[3] = {0,0,0}, p2rot[3] = {0,0,0};
        /* if ( g_verbose > 0 ) fprintf ( stdout, "# [test_lg_momentum_classes] momentum_pair_selection %3d  i %3d  k %3d  flag %2d\n",
            i, momentum_pair_selection[i][0], momentum_pair_selection[i][1], momentum_pair_selection[i][2] );

        fflush ( stdout );
        fflush ( stderr );
        */


        rot_point ( p1rot, g_sink_momentum_list[momentum_pair_selection[i][0]], C );
        rot_point ( p2rot, g_sink_momentum_list[momentum_pair_selection[i][1]], C );

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
#if 0
#endif  /* of if 0  */

        rot_fini_rotation_matrix ( &C );

      }  /* end of loop on proper rotations */

      /****************************************************
       * loop on rotation-reflections
       ****************************************************/
      for ( int irot = 0; irot < lg[ilg].nrm ; irot++ ) {

        double _Complex **C = rot_init_rotation_matrix ( 3  );

        rot_spherical2cartesian_3x3 ( C, Rpvec.IR[lg[ilg].rm[irot]] );

        if ( ! rot_mat_check_is_real_int ( C, 3 ) ) {
          fprintf ( stderr, "[test_lg_momentum_classes] Error, IR %d / %d is not real int\n", irot, lg[ilg].rm[irot] );
          EXIT(1);
        /* } else {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [test_lg_momentum_classes] rot %d / %d is okay\n", irot, lg[ilg].r[irot] );
          */
        }

        int p1rot[3] = {0,0,0}, p2rot[3] = {0,0,0};
        /* if ( g_verbose > 0 ) fprintf ( stdout, "# [test_lg_momentum_classes] momentum_pair_selection %3d  i %3d  k %3d  flag %2d\n",
            i, momentum_pair_selection[i][0], momentum_pair_selection[i][1], momentum_pair_selection[i][2] );

        fflush ( stdout );
        fflush ( stderr );
        */


        rot_point ( p1rot, g_sink_momentum_list[momentum_pair_selection[i][0]], C );
        rot_point ( p2rot, g_sink_momentum_list[momentum_pair_selection[i][1]], C );

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
#if 0
#endif  /* of if 0  */

        rot_fini_rotation_matrix ( &C );

      }  /* end of loop on proper rotations */

    }  /* end of loop on pairs */

    for ( int i = 0; i < count_pairs; i++ ) {
      fprintf ( stdout, "%10s     %3d %3d  (%3d %3d %3d) (%3d %3d %3d)      %3d  %3d %3d  (%3d %3d %3d) (%3d %3d %3d)\n",
          lg[ilg].name, momentum_pair_selection[i][0], momentum_pair_selection[i][1],
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
#if 0
#endif  /* of if 0  */

    fini_2level_itable ( &momentum_pair_selection );
  }  /* end of loop on little groups */

  fini_rot_mat_table ( &Rpvec );

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
