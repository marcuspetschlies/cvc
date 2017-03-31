/****************************************************
 * test_openmp.cpp
 *
 * Tue Jan 17 09:09:05 CET 2017
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
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "matrix_init.h"
#include "contract_cvc_tensor.h"
#include "scalar_products.h"

using namespace cvc;

void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {

  const int NREP = 50; 

  int c, i, k, no_fields=0;
  int filename_set = 0;
  int exitstatus;
  unsigned int ix;
  char filename[100];
  double ratime, retime;
  double norm, norm2;
  complex w;
  FILE *ofs;

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
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    exit(557);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    exit(558);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    exit(559);
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
  omp_set_num_threads(g_num_threads);
#else
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_openmp] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(560);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(561);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[test_openmp] Error, &g_gauge_field is NULL\n");
    EXIT(563);
  }
#else
  EXIT(44);
#endif

  const unsigned int Vhalf = VOLUME / 2;
  const size_t sizeof_eo_spinor_field = _GSI(Vhalf)  * sizeof(double);
  const size_t sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);
  const size_t sizeof_halo_spinor_field    = _GSI( (VOLUME+RAND) ) * sizeof(double);

  no_fields = 4;
  g_spinor_field = (double**)malloc(no_fields * sizeof( double* ) );
  g_spinor_field[0] = (double*)malloc( no_fields * sizeof_spinor_field );
  if( g_spinor_field[0] == NULL ) {
    fprintf(stderr, "[test_openmp] Error from malloc\n");
    EXIT(1);
  }
  for( i=1; i<no_fields; i++ ) {
    g_spinor_field[i] = g_spinor_field[i-1] + _GSI( VOLUME);
  }

  /* init rng file */
  init_rng_stat_file (g_seed, NULL);
   
  ranlxd(g_spinor_field[0], 2*_GSI(VOLUME));

  /* real-valued scalar product */
  ratime = _GET_TIME;
  for(i=0; i<NREP; i++) {
    spinor_scalar_product_re(&norm, g_spinor_field[0], g_spinor_field[1], VOLUME);
  }
  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [test_openmp] time for spinor_scalar_product_re = %e seconds\n", (retime-ratime)/NREP );

  /* complex-valued scalar product */
  ratime = _GET_TIME;
  for(i=0; i<NREP; i++) {
    spinor_scalar_product_co(&w, g_spinor_field[0], g_spinor_field[1], VOLUME);
  }
  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [test_openmp] time for spinor_scalar_product_co = %e seconds\n", (retime-ratime)/NREP );

  /* complex-valued scalar product */
  ratime = _GET_TIME;
  for(i=0; i<NREP; i++) {
    spinor_scalar_product_co(&w, g_spinor_field[0], g_spinor_field[1], VOLUME);
    spinor_field_eq_spinor_field_pl_spinor_field_ti_re(g_spinor_field[2], g_spinor_field[0], g_spinor_field[1], M_PI, VOLUME);
  }
  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [test_openmp] time for spinor_field_eq_spinor_field_pl_spinor_field_ti_re = %e seconds\n", (retime-ratime)/NREP );


  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  if(g_spinor_field != NULL) { 
    if( g_spinor_field[0] != NULL ) {
      free(g_spinor_field[0]);
    }
    free(g_spinor_field);
  }

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test] %s# [test] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test] %s# [test] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
