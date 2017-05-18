/****************************************************
 * test_rotations.cpp
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
#include "matrix_init.h"
#include "rotations.h"
#include "ranlxd.h"

using namespace cvc;

void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}

int main(int argc, char **argv) {

  const double ONE_OVER_SQRT2 = 1. / sqrt(2.);
  const int Ndim = 3;

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  // FILE *ofs = NULL;

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
        fprintf(stdout, "# [test_rotations] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_rotations] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    EXIT(1);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(2);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
    if(exitstatus != 0) {
    EXIT(3);
  }
#endif

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
   mpi_init(argc, argv);

 /*********************************
  * set number of openmp threads
  *********************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [p2gg] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_rotations] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

/*
  int n[3] = {-1,1,1};
  double w = 4*M_PI/3.;
*/
/*
  int n[3] = {-1,1,0};
  double w = M_PI;
*/
/*
  int n[3] = {1,1,1};
  double w = 4*M_PI/3.;
*/

  double _Complex **R = NULL;
  double _Complex **A = NULL;
  double _Complex **B = NULL;

  init_rotation_table();

#if 0
  // rlxd_init(2, g_seed);
#endif 

  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [p2gg] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg] initializing unit matrices\n");
    for( unsigned int ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#ifdef HAVE_MPI
  xchange_gauge_field( g_gauge_field );
#endif

  fflush(stdout);
  fflush(stderr);
#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif

  if (g_cart_id == 0 ) {
    fprintf(stdout, "# [test_rotations] plaquette reference value\n");
    fflush(stdout);
  }
  plaquetteria  ( g_gauge_field );

  double *gauge_field_rot = NULL;
  alloc_gauge_field(&gauge_field_rot, VOLUMEPLUSRAND);

  init_2level_buffer ( (double***)(&R), Ndim, 2*Ndim );
  init_2level_buffer ( (double***)(&A), Ndim, 2*Ndim );


  // for(int irot=0; irot < 48; irot++ )
  for(int irot = 46; irot < 47; irot++ )
  {

    if (g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rotations] rotation no. %2d n = (%2d, %2d, %2d) w = %16.7e pi\n", irot,
          cubic_group_double_cover_rotations[irot].n[0], cubic_group_double_cover_rotations[irot].n[1], cubic_group_double_cover_rotations[irot].n[2],
          cubic_group_double_cover_rotations[irot].w);
    }

    rotation_matrix_spherical_basis ( R, Ndim-1, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );
    spherical2cartesian_3x3 (A, R);
    if ( rot_mat_check_is_real_int ( A, Ndim ) ) {
      if (g_cart_id == 0 )
        fprintf(stdout, "# [test_rotations] rot_mat_check_is_real_int matrix A rot %2d ok\n", irot);
    } else {
      EXIT(6);
    }

    char name[6];
    sprintf(name, "A[%.2d]", irot);
    rot_printf_rint_matrix (A, Ndim, name, stdout );


    exitstatus = rot_gauge_field ( gauge_field_rot, g_gauge_field, A);
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_rotations] Error from rot_gauge_field, status was %d\n", exitstatus);
      EXIT(23);
    }
#if 0
#ifdef HAVE_MPI
    xchange_gauge_field ( gauge_field_rot );
#endif
    if (g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rotations] plaquette value rotation %2d\n", irot);
      fflush(stdout);
    }
    plaquetteria  ( gauge_field_rot );


#endif  /* of if 0 */
  }  /* end of loop on rotations */

  fini_2level_buffer ( (double***)(&A) );
  fini_2level_buffer ( (double***)(&R) );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_rotations] %s# [test_rotations] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_rotations] %s# [test_rotations] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
