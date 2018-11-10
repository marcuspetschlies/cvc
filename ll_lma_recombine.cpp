/****************************************************
 * ll_lma_recombine.c
 *
 * Do 8. Feb 11:51:36 CET 2018
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
#include "set_default.h"
#include "io.h"
#include "read_input_parser.h"
#include "gsp_utils.h"
#include "gsp.h"
#include "matrix_init.h"
#include "rotations.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1


using namespace cvc;

void usage() {
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -p                  : fix eigenvector phase [default no ]\n");
  EXIT(0);
}


int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "ll_lma";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int evecs_num = 0;
  int do_fix_eigenvector_phase = 0;
  unsigned int Vhalf;
  double *evecs_eval = NULL, *evecs_lambdainv=NULL, *evecs_4kappasqr_lambdainv = NULL;
  char filename[100];
  /* double ratime, retime; */
  double **mzz[2], **mzzinv[2];
  double *gauge_field_with_phase = NULL;

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
# else
  void *affw = NULL;
#endif
  char aff_tag[400];

  // in order to be able to initialise QMP if QPhiX is used, we need
  // to allow tmLQCD to intialise MPI via QMP
  // the input file has to be read 
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_init_parallel_and_read_input(argc, argv, 1, "invert.input");
#else
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
#endif

  while ((c = getopt(argc, argv, "h?f:p:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'p':
      do_fix_eigenvector_phase = 1;
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

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [ll_lma_recombine] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /***********************************************************/
  /***********************************************************/

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [ll_lma_recombine] %s# [ll_lma_recombine] start of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [ll_lma_recombine] %s# [ll_lma_recombine] start of run\n", ctime(&g_the_time));
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set number of openmp threads
   ***********************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [ll_lma_recombine] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [ll_lma_recombine] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[ll_lma_recombine] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * intialize geometry
   ***********************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[ll_lma_recombine] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * read eigenvectors and calculate inverses etc.
   ***********************************************/

  exitstatus = 
    init_1level_buffer  ( &evecs_eval, evecs_num ) ||
    init_1level_buffer  ( &evecs_eval_lambdainv, evecs_num ) ||
    init_1level_buffer  ( &evecs_eval_4kappasqr_lambdainv, evecs_num );

  if ( exitstatus != 0 ) {
    fprintf(stderr, "[ll_lma_recombine] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }


  sprintf ( aff_tag, "ll_lma.%.4d", Nconf );
  exitstatus = gsp_read_eval ( &evecs_eval, evecs_num, aff_tag);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[ll_lma_recombine] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * inverse eigenvalues, normalize
   ***********************************************************/
  for( int i = 0; i < evecs_num; i++) {
    evecs_eval_lambdainv[i]           = 2.* g_kappa           / evecs_eval[i];
    evecs_eval_4kappasqr_lambdainv[i] = 4.* g_kappa * g_kappa / evecs_eval[i];
    if( g_cart_id == 0 ) fprintf(stdout, "# [ll_lma_recombine] eval %4d %16.7e\n", i, evecs_eval[i] );
  }

STOPPED HERE
  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * allocate memory for the eigenvector fields
   ***********************************************************/
  gsp_field = (double**)calloc(evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block;
  for( int i = 1; i < evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + _GSI(Vhalf);

  /***********************************************************/
  /***********************************************************/

#ifdef HAVE_MPI
  /***********************************************************
   * set io process
   ***********************************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [ll_lma_recombine] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [ll_lma_recombine] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  if(io_proc == 2) {
    if(g_tr_id != 0) {
      fprintf(stderr, "[ll_lma_recombine] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * fix eigenvector phase
   ***********************************************************/
  if ( do_fix_eigenvector_phase == 1 ) {
    if ( io_proc == 2 ) fprintf(stdout, "# [ll_lma_recombine] fixing eigenvector phase\n");
    exitstatus = fix_eigenvector_phase ( eo_evecs_field, evecs_num );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[ll_lma_recombine] Error from fix_eigenvector_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
  }

  /***********************************************************/
  /***********************************************************/

  sprintf( aff_tag, "/ll/lma/N%d", evecs_num );
  sprintf( filename_prefix, "%s.%.4d.N%d", outfile_prefix, Nconf, evecs_num );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/

  fini_1level_buffer  ( &evecs_eval )
  fini_1level_buffer  ( &evecs_eval_lambdainv );
  fini_1level_buffer  ( &evecs_eval_4kappasqr_lambdainv );

  /* free clover matrix terms */
  fini_clover ();

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [ll_lma_recombine] %s# [ll_lma_recombine] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [ll_lma_recombine] %s# [ll_lma_recombine] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
