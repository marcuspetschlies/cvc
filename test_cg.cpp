/****************************************************
 * test_cg.cpp
 *
 * Mo 5. Feb 07:58:16 CET 2018
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
#include "prepare_source.h"
#include "Q_phi.h"
#include "scalar_products.h"
#include "group_projection.h"
#include "clebsch_gordan.h"

using namespace cvc;

void usage() {
  EXIT(0);
}

int main(int argc, char **argv) {

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
        fprintf(stdout, "# [test_cg] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_cg] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_cg] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_cg] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_cg] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /*********************************
   * set up geometry fields
   *********************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_cg] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /******************************************************************/
  /******************************************************************/

#if 0
  /******************************************************************
   * TEST
   *   delta_symbol
   ******************************************************************/
  for ( int a2 = 0; a2 <= 4; a2++ ) {
  for ( int b2 = 0; b2 <= 4; b2++ ) {
  for ( int c2 = 0; c2 <= 4; c2++ ) {

    double d = delta_symbol ( a2, b2, c2 );
    fprintf ( stdout, "# [test_cg] delta_symbol %3.1f   %3.1f   %3.1f  %16.9e\n", a2/2., b2/2., c2/2., d );
  }}}
#endif

  /******************************************************************
   * TEST
   *   delta_symbol
   ******************************************************************/
  int a2 = 3;
  int b2 = 3;
  int c2 = 4;

  fprintf( stdout, "# [test_cg] coupling %3.1f + %3.1f -> %3.1f\n", a2/2., b2/2., c2/2. );
  for ( int gamma2 = -c2; gamma2 <= c2; gamma2+=2 ) {
  for ( int alpha2 = -a2; alpha2 <= a2; alpha2+=2 ) {
  for ( int beta2  = -b2; beta2  <= b2; beta2+=2 ) {
    double d =  clebsch_gordan_coeff ( c2, gamma2, a2, alpha2, b2, beta2 );
    if ( fabs(d) < 5.e-14 ) continue;
    fprintf ( stdout, "  %4.1f     %4.1f %4.1f   %16.9f\n", gamma2/2., alpha2/2., beta2/2., d );
  }}}


  /******************************************************************/
  /******************************************************************/


  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
