/****************************************************
 * test_zm4x4.cpp
 *
 * So 29. Jul 21:30:12 CEST 2018
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
#include <ctype.h>
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
#include "zm4x4.h"
#include "contract_diagrams.h"
#include "table_init_z.h"

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
        fprintf(stdout, "# [test_zm4x4] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_zm4x4] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
   mpi_init(argc, argv);

 /*********************************
  * set number of openmp threads
  *********************************/
  fprintf(stdout, "# [test_zm4x4] test :  g_num_threads = %d\n", g_num_threads);
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_zm4x4] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_zm4x4] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_zm4x4] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_zm4x4] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  rlxd_init( 2, g_seed);

  unsigned int const N = 179;

  double _Complex *** z1 = init_3level_ztable ( N, 4, 4 );
  double _Complex *** z2 = init_3level_ztable ( N, 4, 4 );
  double _Complex *** z3 = init_3level_ztable ( N, 4, 4 );

  double _Complex const zcoeff = M_PI + I / M_PI;


  ranlxd ( (double*)(z1[0][0]), 32*N );
  ranlxd ( (double*)(z2[0][0]), 32*N );

  memcpy ( z3[0][0], z1[0][0], 16*N*sizeof(double _Complex ) );
  contract_diagram_zm4x4_field_pl_eq_zm4x4_field ( z1, z2, N );

  // contract_diagram_zm4x4_field_eq_zm4x4_field_pl_zm4x4_field_ti_co ( z3, z1, z2, zcoeff, N );

  for ( int i = 0; i < N; i++ ) {
    for ( int k = 0; k < 4; k++ ) {
    for ( int l = 0; l < 4; l++ ) {
      fprintf ( stdout, " %3d %3d %3d  %16.7e\n", i, k, l, cabs( z1[i][k][l] - z3[i][k][l] - z2[i][k][l] ) );
      // fprintf ( stdout, " %3d %3d %3d  %16.7e\n", i, k, l, cabs( z3[i][k][l] - ( z1[i][k][l] + z2[i][k][l] * zcoeff ) ) );
    }}
  }

  fini_3level_ztable ( &z1 );
  fini_3level_ztable ( &z2 );
  fini_3level_ztable ( &z3 );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_zm4x4] %s# [test_zm4x4] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_zm4x4] %s# [test_zm4x4] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
