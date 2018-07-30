/****************************************************
 * test_gamma.cpp
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
#include "gamma.h"

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
        fprintf(stdout, "# [test_gamma] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_gamma] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
   mpi_init(argc, argv);

 /*********************************
  * set number of openmp threads
  *********************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_gamma] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_gamma] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_gamma] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_gamma] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  init_gamma_matrix ();

#if 0
  for ( int i = 0; i < 16; i++ ) {
    int  id;
    double s;
    gamma_matrix_type g1;
    gamma_matrix_set ( &g1, i, 1. );
    // gamma_matrix_adjoint ( &g1, &g1 );
    gamma_matrix_get_id_sign ( &id, &s, &g1 );
    fprintf( stdout, "# [test_gamma] gamma %2d ---> %2d %f\n", i, id, s );
  }
#endif

  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_type gi;
    gamma_matrix_set ( &gi, i, 1. );

  for ( int k = 0; k < 16; k++ ) {
    gamma_matrix_type gk;
    gamma_matrix_set ( &gk, k, 1. );
  
  for ( int l = 0; l < 16; l++ ) {
    gamma_matrix_type gl;
    gamma_matrix_set ( &gl, l, 1. );

    gamma_matrix_type g;


#if 0
    char op1 = 'N';
    char op2 = 'N';
    int id   = g_gamma_mult_table[i][g_gamma_mult_table[l][k]];
    double s = g_gamma_mult_sign[l][k] * g_gamma_mult_sign[i][g_gamma_mult_table[l][k]];

    char op1 = 'N';
    char op2 = 'H';
    int id   = g_gamma_mult_table[i][g_gamma_mult_table[l][k]];
    double s = g_gamma_mult_sign[l][k] * g_gamma_mult_sign[i][g_gamma_mult_table[l][k]] * g_gamma_adjoint_sign[k];

    char op1 = 'H';
    char op2 = 'N';
    int id   = g_gamma_mult_table[i][g_gamma_mult_table[l][k]];
    double s = g_gamma_mult_sign[l][k] * g_gamma_mult_sign[i][g_gamma_mult_table[l][k]] * g_gamma_adjoint_sign[i];

    char op1 = 'T';
    char op2 = 'N';
    int id   = g_gamma_mult_table[i][g_gamma_mult_table[l][k]];
    double s = g_gamma_mult_sign[l][k] * g_gamma_mult_sign[i][g_gamma_mult_table[l][k]] * g_gamma_transposed_sign[i];

    char op1 = 'N';
    char op2 = 'T';
    int id   = g_gamma_mult_table[i][g_gamma_mult_table[l][k]];
    double s = g_gamma_mult_sign[l][k] * g_gamma_mult_sign[i][g_gamma_mult_table[l][k]] * g_gamma_transposed_sign[k];

    char op1 = 'C';
    char op2 = 'N';
    int id   = g_gamma_mult_table[i][g_gamma_mult_table[l][k]];
    double s = g_gamma_mult_sign[l][k] * g_gamma_mult_sign[i][g_gamma_mult_table[l][k]] * g_gamma_transposed_sign[i] * g_gamma_adjoint_sign[i];

    char op1 = 'N';
    char op2 = 'C';
    int id   = g_gamma_mult_table[i][g_gamma_mult_table[l][k]];
    double s = g_gamma_mult_sign[l][k] * g_gamma_mult_sign[i][g_gamma_mult_table[l][k]] * g_gamma_transposed_sign[k] * g_gamma_adjoint_sign[k];
#endif
    char op1 = 'T';
    char op2 = 'C';
    int id   = g_gamma_mult_table[i][g_gamma_mult_table[l][k]];
    double s = g_gamma_mult_sign[l][k] * g_gamma_mult_sign[i][g_gamma_mult_table[l][k]] * g_gamma_transposed_sign[k] * g_gamma_adjoint_sign[k] * g_gamma_transposed_sign[i];
    gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &g, &gi, op1, &gl, &gk, op2 );

    fprintf( stdout, "%2d %c   %2d   %2d %c    %6.3f %2d    %6.3f %2d\n", i, op1, l, k, op2, g.s , g.id, s, id);
  }}}



  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_gamma] %s# [test_gamma] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_gamma] %s# [test_gamma] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
