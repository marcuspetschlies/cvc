/****************************************************
 * make_random_config.c
 * 
 * Tue May 30 10:40:59 CEST 2017
 *
 * PURPOSE:
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include <getopt.h>

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"

using namespace cvc;

/***********************************************************
 * usage function
 ***********************************************************/
void usage() {
  fprintf(stdout, "Code to generate and write to file a random gauge configuration\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -h? this help\n");
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}
  
  
/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  int exitstatus;

  char filename[200];
  double ratime, retime;
  double heat = 1.;
  double plaq;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:r:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'r':
      heat = atof( optarg );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[make_random_config] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [make_random_config] git version = %s\n", g_gitversion);
  }

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[make_random_config] Error from init_geometry\n");
    EXIT(1);
  }
  geometry();

  exitstatus = alloc_gauge_field( &g_gauge_field, VOLUME );

  random_gauge_field ( g_gauge_field, heat );

  exitstatus = plaquetteria ( g_gauge_field );

  plaquette2 ( &plaq, g_gauge_field );

  sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
  exitstatus = write_lime_gauge_field ( filename, plaq, Nconf, 64 );

  free_geometry();

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [make_random_config] %s# [make_random_config] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [make_random_config] %s# [make_random_config] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
