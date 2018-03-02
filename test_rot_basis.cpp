/****************************************************
 * test_rot_basis.cpp
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
#include "group_projection.h"
#include "ranlxd.h"
#include "scalar_products.h"

#define _SQR(_a) ((_a)*(_a))

using namespace cvc;

void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}

void print_sf (double*sf, char*name) {

  FILE *ofs = fopen(name, "w");
  for( int x0 = 0; x0 < T; x0++ ) {
  for( int x1 = 0; x1 < LX; x1++ ) {
  for( int x2 = 0; x2 < LY; x2++ ) {
  for( int x3 = 0; x3 < LZ; x3++ ) {
    unsigned int ix = g_ipt[x0][x1][x2][x3];
    fprintf(ofs, "# x %3d%3d%3d%3d\n", x0, x1, x2, x3);
    for( int mu=0; mu<12; mu++ ) {
      fprintf(ofs, "%3d %3d %25.16e %25.16e\n", mu/3, mu%3, sf[_GSI(ix)+2*mu], sf[_GSI(ix)+2*mu+1]);
    }
  }}}}
  fclose(ofs);
}  /* end of print_sf */

int main(int argc, char **argv) {

  const double ONE_OVER_SQRT2 = 1. / sqrt(2.);
  const int Ndim = 3;

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  double norm, norm2;
  FILE *ofs = NULL;
  double _Complex **R = NULL;
  double _Complex **A = NULL;
  double _Complex **B = NULL;


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
        fprintf(stdout, "# [test_rot_basis] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_rot_basis] Reading input from file %s\n", filename);
  read_input_parser(filename);

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
    fprintf(stderr, "[test_rot_basis] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  rot_init_rotation_table();

  rlxd_init(2, g_seed);

  rot_mat_table_type spin_1;

  init_rot_mat_table ( &spin_1 );

  exitstatus = set_rot_mat_table_spin ( &spin_1, 2, 0 );


  /***********************************************************
   * loop on rotations
   ***********************************************************/

  for(int irot=0; irot < 24; irot++ )
  {
    char name[20];

    int irot1 = cubic_group_double_cover_identification_table[irot][0];
    int irot2 = cubic_group_double_cover_identification_table[irot][1];

    for ( int ib = 0; ib < 3; ib++ ) {
      int d[3] = {0,0,0}, d2[3], d3[3];
      d[ib] = 1;

      double _Complex **R = rot_init_rotation_matrix ( 3 );

      rot_spherical2cartesian_3x3 ( R, spin_1.R[irot1] );
      if ( !rot_mat_check_is_real_int ( R, 3 ) ) { 
        fprintf(stderr, "[test_rot_basis] rot %2d is not real int\n", irot1 );
        EXIT(1);
      }
      rot_point ( d2, d, R);

      rot_spherical2cartesian_3x3 ( R, spin_1.R[irot2] );
      if ( !rot_mat_check_is_real_int ( R, 3 ) ) { 
        fprintf(stderr, "[test_rot_basis] rot %2d is not real int\n", irot2 );
        EXIT(1);
      }
      rot_point ( d3, d, R);

      fprintf (stdout, "# [test_rot_basis] change of basis coordinates rot %2d  %2d %2d %2d  %16.10f pi and rot %2d  %2d %2d %2d  %16.10f pi\n", irot1+1, 
           cubic_group_double_cover_rotations[irot1].n[0],
           cubic_group_double_cover_rotations[irot1].n[1],
           cubic_group_double_cover_rotations[irot1].n[2],
           cubic_group_double_cover_rotations[irot1].w / M_PI, irot2+1,
           cubic_group_double_cover_rotations[irot2].n[0],
           cubic_group_double_cover_rotations[irot2].n[1],
           cubic_group_double_cover_rotations[irot2].n[2],
           cubic_group_double_cover_rotations[irot2].w / M_PI );

      fprintf(stdout, "  %2d    %2d    %2d\n", d[0], d2[0], d3[0] );
      fprintf(stdout, "  %2d    %2d    %2d\n", d[1], d2[1], d3[1] );
      fprintf(stdout, "  %2d    %2d    %2d\n", d[2], d2[2], d3[2] );

      rot_fini_rotation_matrix ( &R );

    }

    fprintf (stdout, "# [test_rot_basis] =================================\n");

  }  /* end of loop on rotations */

  fini_rot_mat_table ( &spin_1 );


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_rot_basis] %s# [test_rot_basis] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_rot_basis] %s# [test_rot_basis] end of run\n", ctime(&g_the_time));
  }

  free_geometry ();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
