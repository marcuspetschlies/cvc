/****************************************************
 * test_get_lg.cpp
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

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  int pvec[3];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:p:")) != -1) {
    switch (c) {
      case 'f':
        strcpy(filename, optarg);
        filename_set=1;
        break;
      case 'p':
        sscanf( optarg, "%d,%d,%d", pvec, pvec+1, pvec+2 );
        fprintf ( stdout, "# [test_get_lg] pvec set to %3d %3d %3d\n", 
            pvec[0], pvec[1], pvec[2] ); 
        break;
      case 'h':
      case '?':
      default:
        fprintf(stdout, "# [test_get_lg] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if ( filename_set == 1 ) {
    fprintf ( stdout, "# [test_get_lg] Reading input from file %s\n", filename );
    read_input_parser ( filename );
  }

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
  if(g_cart_id == 0) fprintf(stdout, "# [test_get_lg] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_get_lg] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_get_lg] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /****************************************************/
  /****************************************************/

  /****************************************************
   * initialize and set geometry fields
   ****************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0) {
    fprintf(stderr, "[test_get_lg] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  geometry();

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
   * initialize spin-1 rotation table
   * here in spherical basis
   ****************************************************/

  rot_mat_table_type Rpvec;
  init_rot_mat_table ( &Rpvec );
  exitstatus = set_rot_mat_table_spin ( &Rpvec, 2, 0 );

  rot_mat_table_printf ( &Rpvec, "Rpvec", stdout);


  exitstatus = get_little_group_elements ( pvec, &Rpvec );

  fini_rot_mat_table ( &Rpvec );

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
    fprintf(stdout, "# [test_get_lg] %s# [test_get_lg] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_get_lg] %s# [test_get_lg] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif
  return(0);
}
