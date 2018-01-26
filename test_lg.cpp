/****************************************************
 * test_lg.cpp
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
#include "group_projection.h"


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


void print_fp_point_from_sf (double**sf, unsigned int ix, char*name, FILE*ofs) {
  fprintf(ofs, "%s <- array(dim=c(%d, %d))\n", name,12,12);
  for( int mu=0; mu<12; mu++ ) {
  for( int nu=0; nu<12; nu++ ) {
    fprintf(ofs, "%s[%2d,%2d] <- %25.16e +  %25.16e*1.i;\n", name, mu+1, nu+1,
        sf[nu][_GSI(ix)+2*mu], sf[nu][_GSI(ix)+2*mu+1]);
  }}
}  /* end of print_fp_point_from_sf */


int main(int argc, char **argv) {

  const double ONE_OVER_SQRT2 = 1. / sqrt(2.);
  const int Ndim = 3;

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  double norm;
  FILE *ofs = NULL;
  double _Complex **R = NULL;
  double _Complex **A = NULL;
  double _Complex **B = NULL;
  double _Complex **ASpin = NULL;
  char name[12];
  fermion_propagator_type fp1, fp2;


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
        fprintf(stdout, "# [test_lg] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_lg] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
   mpi_init(argc, argv);

 /*********************************
  * set number of openmp threads
  *********************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_lg] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_lg] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_lg] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_lg] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /* rot_init_rotation_table(); */
  little_group_type *lg = NULL;
  int nlg = 0;

  if ( ( nlg = little_group_read_list ( &lg, "little_groups_2Oh.tab") ) <= 0 )
  /* if ( ( nlg = little_group_read_list ( &lg, "little_groups_Oh.tab") ) <= 0 ) */
  {
    fprintf(stderr, "[test_lg] Error from little_group_read_list, status was %d\n", nlg);
    EXIT(2);
  }
  fprintf(stdout, "# [test_lg] number of little groups = %d\n", nlg);

  little_group_show ( lg, stdout, nlg );


#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_lg] %s# [test_lg] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_lg] %s# [test_lg] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
