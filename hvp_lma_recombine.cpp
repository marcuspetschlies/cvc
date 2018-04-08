/****************************************************
 * hvp_lma_recombine.c
 *
 * Do 29. Mär 16:01:03 CEST 2018
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "table_init_d.h"
#include "table_init_z.h"
#include "clover.h"
#include "gsp.h"


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -w                  : check position space WI [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "hvp_lma_recombine";

  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[100];
  // double ratime, retime;
  unsigned int evecs_block_length = 0;
  unsigned int evecs_num = 0;


#ifdef HAVE_LHPC_AFF
  char aff_tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sh?f:b:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'b':
      evecs_block_length = atoi ( optarg );
      break;
    case 'n':
      evecs_num = atoi( optarg );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "p2gg.input");
  // fprintf(stdout, "# [hvp_lma_recombine] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
  mpi_init(argc, argv);

  /***********************************************************
   * set number of openmp threads
   ***********************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hvp_lma_recombine] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hvp_lma_recombine] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hvp_lma_recombine] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * initialize geometry fields
   ***********************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[hvp_lma_recombine] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_contraction(2);
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * read eigenvalues
   ***********************************************************/
  double * evecs_eval = NULL;
  exitstatus = gsp_read_eval( &evecs_eval, evecs_num, filename,  aff_tag);
  if( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma_recombine] Error from gsp_read_eval, status was %d %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(39);
  }

  /***********************************************************
   * set auxilliary eigenvalue fields
   ***********************************************************/
  double * const evecs_lambdainv           = (double*)malloc(evecs_num*sizeof(double));
  double * const evecs_4kappasqr_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if( evecs_lambdainv == NULL || evecs_4kappasqr_lambdainv == NULL ) {
    fprintf(stderr, "[hvp_lma_recombine] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for( unsigned int i = 0; i < evecs_num; i++) {
    evecs_lambdainv[i]           = 2.* g_kappa / evecs_eval[i];
    evecs_4kappasqr_lambdainv[i] = 4.* g_kappa * g_kappa / evecs_eval[i];
  }

  /***********************************************************
   * check evecs_block_length
   ***********************************************************/
  if ( evecs_block_length == 0 ) {
    evecs_block_length = evecs_num;
    if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma_recombine] WARNING, reset evecs_block_length to %u\n", evecs_num );
  }

  /***********************************************************
   * set io process
   ***********************************************************/
  int const io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hvp_lma_recombine] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [hvp_lma_recombine] proc%.4d has io proc id %d\n", g_cart_id, io_proc );



  /***********************************************************/
  /***********************************************************/
 


  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/

  if ( evecs_eval                != NULL ) free ( evecs_eval );
  if ( evecs_lambdainv           != NULL ) free ( evecs_lambdainv );
  if ( evecs_4kappasqr_lambdainv != NULL ) free ( evecs_4kappasqr_lambdainv ); 

  /***********************************************************
   * free clover matrix terms
   ***********************************************************/
  fini_clover ();

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_xchange_eo_propagator();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [hvp_lma_recombine] %s# [hvp_lma_recombine] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [hvp_lma_recombine] %s# [hvp_lma_recombine] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
