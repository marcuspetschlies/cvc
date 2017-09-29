/****************************************************
 * loops_em.cpp
 *
 * Thu Sep 28 15:27:11 CEST 2017
 *
 * - originally copied from loops_caa_lma.cpp
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
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "matrix_init.h"
#include "clover.h"
#include "scalar_products.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#define _SQR(_a) ((_a)*(_a))

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform loop contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc    [default cvc.input]\n");
  fprintf(stdout, "          -w                  : check position space WI   [default false]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

/******************************************************/
/******************************************************/

int dummy_eo_solver (double * const propagator, double * const source, const int op_id) {
  memcpy(propagator, source, _GSI(VOLUME)/2*sizeof(double) );
  return(0);
}

/******************************************************/
/******************************************************/

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "loops";

  int c;
  int iflavor;
  int filename_set = 0;
  int isource_location;
  unsigned int ix;
  int gsx[4], sx[4];
  int check_position_space_WI=0;
  int exitstatus;
  int source_proc_id = 0;
  int no_eo_fields = 0;
  int io_proc = -1;
  int evecs_num = 0;
  int check_propagator_residual = 0;
  unsigned int Vhalf;
  size_t sizeof_eo_spinor_field;
  double **eo_spinor_field=NULL, **eo_spinor_work=NULL, *eo_evecs_block=NULL;
  double **eo_evecs_field=NULL;
  double ***cvc_loop_eo = NULL;
  // double ***cvc_tp = NULL;
  double *evecs_eval = NULL, *evecs_lambdainv=NULL, *evecs_4kappasqr_lambdainv = NULL;
  char filename[100];
  double ratime, retime;
  double **mzz[2], **mzzinv[2];
  double *gauge_field_with_phase = NULL;
  double ****cvc_loop_lma_x = NULL, ****cvc_loop_lma_p = NULL, *cvc_loop_lexic = NULL;
  double ***cvc_loop_lma_p = NULL, ***cvc_loop_tp = NULL;
  double *jj_tensor_trace = NULL;
  double _Complex ztmp[4];

#ifdef HAVE_MPI
  MPI_Status mstatus;
#endif


#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
  char aff_tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "cwh?f:n:s:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_position_space_WI = 1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'n':
      nev_step_size = atoi (optarg);
      break;
    case 's':
      sample_step_size = atoi (optarg);
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
  /* fprintf(stdout, "# [loops_em] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [loops_em] calling tmLQCD wrapper init functions\n");

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
  mpi_init_xchange_contraction(2);


  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [loops_em] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [loops_em] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [loops_em] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[loops_em] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[loops_em] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  Vhalf = VOLUME / 2;
  sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [loops_em] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [loops_em] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[loops_em] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[loops_em] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[loops_em] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[loops_em] Error from tmLQCD_get_deflator_params, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(9);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [loops_em] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [loops_em] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [loops_em] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [loops_em] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block == NULL) {
    fprintf(stderr, "[loops_em] Error, eo_evecs_block is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(10);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[loops_em] Error, dimension of eigenspace is zero %s %d\n", __FILE__, __LINE__);
    EXIT(11);
  }

  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[loops_em] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  evecs_eval                = (double*)malloc(evecs_num*sizeof(double));
  evecs_lambdainv           = (double*)malloc(evecs_num*sizeof(double));
  evecs_4kappasqr_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if(    evecs_eval                == NULL 
      || evecs_lambdainv           == NULL 
      || evecs_4kappasqr_lambdainv == NULL 
    ) {
    fprintf(stderr, "[loops_em] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for( int i = 0; i < evecs_num; i++) {
    evecs_eval[i]                = ((double*)(g_tmLQCD_defl.evals))[2*i];
    evecs_lambdainv[i]           = 2.* g_kappa / evecs_eval[i];
    evecs_4kappasqr_lambdainv[i] = 4.* g_kappa * g_kappa / evecs_eval[i];
    if( g_cart_id == 0 ) fprintf(stdout, "# [loops_em] eval %4d %16.7e\n", i, evecs_eval[i] );
  }

#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */

  /*************************************************
   * allocate memory for the eigenvector fields
   *************************************************/
  eo_evecs_field = (double**)calloc(evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block;
  for( int i = 1; i < evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + _GSI(Vhalf);

  /*************************************************
   * allocate memory for eo spinor fields 
   * WITH HALO
   *************************************************/
  no_eo_fields = 6;
  exitstatus = init_2level_buffer ( &eo_spinor_work, 6, _GSI((VOLUME+RAND)/2) );
  if ( exitstatus != 0) {
    fprintf(stderr, "[loops_em] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(1);
  }

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[loops_em] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_clover, status was %d\n");
    EXIT(1);
  }

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [loops_em] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [loops_em] proc%.4d is send process\n", g_cart_id);
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
      fprintf(stderr, "[loops_em] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************
   * initialize random number generator
   ***********************************************/
  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[loops_em] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * open AFF file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    sprintf(filename, "%s.%.4d.aff", "loops_em", Nconf );
    fprintf(stdout, "# [loops_em] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[loops_em] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }  /* end of if io_proc == 2 */
#endif

  /***********************************************
   * local lma loops
   ***********************************************/
  double ***local_loop_x = NULL, ***local_loop_p = NULL;

  if( ( exitstatus = init_3level_buffer ( &local_loop_x, 2, 16, 2*Vhalf ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }
  /* position space */
  contract_local_loop_eo_lma ( local_loop_x, eo_evecs_field, evecs_4kappasqr_lambdainv, evecs_num, gauge_field_with_phase, mzz, mzzinv );

  /* momentum space */
  if ( ( exitstatus = cvc_loop_eo_momentum_projection ( &local_loop_p, local_loop_x, 16, g_sink_momentum_list, g_sink_momentum_number) ) != 0 ) {
    fprintf ( stderr, "[loop_em] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /* write to file */
  sprintf(aff_tag, "/loop/local/nev%.4d", evecs_num );
  if( ( exitstatus = cvc_loop_tp_write_to_aff_file ( local_loop_p, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  fini_3level_buffer ( &local_loop_x );
  fini_3level_buffer ( &local_loop_p );

  /***********************************************/
  /***********************************************/

  /***********************************************
   * cvc lma loops
   ***********************************************/

  if( ( exitstatus = init_4level_buffer ( &cvc_loop_lma_x, 2, 4, 2, 2*Vhalf ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_4level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if( ( exitstatus = init_3level_buffer ( &cvc_loop_lma_p, 2, 4, 2*VOLUME ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if( ( exitstatus = init_1level_buffer ( &cvc_loop_lexic, 2*VOLUME ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if( ( exitstatus = init_1level_buffer ( &jj_tensor_trace, 2*VOLUME ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }


  /***********************************************
   * cvc fwd and bwd contraction in position space
   ***********************************************/
  contract_cvc_loop_eo_lma ( cvc_loop_lma_x, eo_evecs_field, evecs_4kappasqr_lambdainv, evecs_num, gauge_field_with_phase, mzz, mzzinv );

  /***********************************************
   * cvc momentum projection fwd
   ***********************************************/
  if( ( exitstatus = cvc_loop_eo_momentum_projection ( &cvc_loop_tp, 4, cvc_loop_lma_x[0], g_sink_momentum_list, g_sink_momentum_number) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  sprintf(aff_tag, "/loop/cvc/fwd/nev%.4d", evecs_num );
  if( ( exitstatus = cvc_loop_tp_write_to_aff_file ( cvc_loop_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  for ( int mu = 0; mu < 4; mu++ ) {
    complex_field_eo2lexic ( cvc_loop_lexic, cvc_loop_lma_x[0][mu][0], cvc_loop_lma_x[0][mu][1] );

    if ( ( exitstatus = ft_4dim ( cvc_loop_lma_p[0][mu], cvc_loop_lexic, 1, (int)( mu==0 )  ) ) != 0 ) {
      fprintf(stderr, "[loops_em] Error from ft_4dim, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }
  }

  /***********************************************
   ***********************************************
   **                                           **
   ** QUESTION: half-link Fourier phase shift?  **
   **                                           **
   ***********************************************
   ***********************************************/

  /***********************************************
   * cvc momentum projection bwd
   ***********************************************/
  if( ( exitstatus = cvc_loop_eo_momentum_projection ( &cvc_loop_tp, 4, cvc_loop_lma_x[1], g_sink_momentum_list, g_sink_momentum_number) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  sprintf(aff_tag, "/loop/cvc/bwd/nev%.4d", evecs_num );
  if( ( exitstatus = cvc_loop_tp_write_to_aff_file ( cvc_loop_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  for ( int mu = 0; mu < 4; mu++ ) {
    complex_field_eo2lexic ( cvc_loop_lexic, cvc_loop_lma_x[1][mu][0], cvc_loop_lma_x[1][mu][1] );

    if ( ( exitstatus = ft_4dim ( cvc_loop_lma_p[1][mu], cvc_loop_lexic, 1, 0 ) ) != 0 ) {
      fprintf(stderr, "[loops_em] Error from ft_4dim, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }
  }

  /***********************************************
   ***********************************************
   **                                           **
   ** QUESTION: half-link Fourier phase shift?  **
   **                                           **
   ***********************************************
   ***********************************************/

  /***********************************************
   * convolution
   ***********************************************/

  /* fwd - fwd - lma - lma */
  co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[0], cvc_loop_lma_p[0], 0, VOLUME );
  exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[0]), jj_tensor_trace, 0 );

  /* fwd - bwd - lma - lma */
  co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[0], cvc_loop_lma_p[1], 0, VOLUME );
  exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[1]), jj_tensor_trace, 0 );

  /* bwd - fwd - lma - lma */
  co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[1], cvc_loop_lma_p[0], 0, VOLUME );
  exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[2]), jj_tensor_trace, 0 );

  /* bwd - bwd - lma - lma */
  co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[1], cvc_loop_lma_p[1], 0, VOLUME );
  exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[3]), jj_tensor_trace, 0 );

  sprintf(aff_tag, "/loop/cvc-conv-cvc/lma-lma/nev%.4d", evecs_num);
  affdir = aff_writer_mkpath(affw, affn, aff_tag);
  if ( ( exitstatus = aff_node_put_complex (affw, affdir, ztmp, (uint32_t)4 ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(5);
  }

  fini_4level_buffer ( &cvc_loop_tp );
  fini_3level_buffer ( &cvc_loop_lma_x );
  fini_1level_buffer ( &cvc_loop_lexic );
  fini_1level_buffer ( &jj_tensor_trace );

  /***********************************************/
  /***********************************************/

  double **eo_stochastic_source = NULL, **eo_stochastic_propagator = NULL;
  double ****cvc_loop_stoch_x = NULL;
  double ***cvc_loop_stoch_p = NULL, cvc_loop_stoch_p_accum = NULL;

  if( ( exitstatus = init_3level_buffer ( &local_loop_x, 2, 16, 2*Vhalf ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( ( exitstatus = init_2level_buffer ( &eo_stochastic_source,     2, _GSI( Vhalf ) ) ) !=  0 ) {
    fprintf(stderr, "[loops_em] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( ( exitstatus = init_2level_buffer ( &eo_stochastic_propagator, 2, _GSI( Vhalf ) ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( ( exitstatus = init_4level_buffer ( &cvc_loop_stoch_x, 2, 4, 2, 2*Vhalf ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_4level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if( ( exitstatus = init_1level_buffer ( &cvc_loop_lexic, 2*VOLUME ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if( ( exitstatus = init_3level_buffer ( &cvc_loop_stoch_p, 2, 4, 2*VOLUME ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if( ( exitstatus = init_3level_buffer ( &cvc_loop_stoch_p_accum, 2, 4, 2*VOLUME ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if( ( exitstatus = init_1level_buffer ( &jj_tensor_trace, 2*VOLUME ) ) != 0 ) {
    fprintf(stderr, "[loops_em] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }


  /***********************************************
   * loop on stochastic samples
   ***********************************************/
  for ( int isample = 0; isample < g_nsample; isample++ ) {

    /***********************************************
     * volume source
     ***********************************************/
    exitstatus = prepare_volume_source ( eo_stochastic_source[0], Vhalf);
    if(exitstatus != 0) {
      fprintf(stderr, "[loops_em] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(33);
    }

    /***********************************************
     * orthogonal projection
     ***********************************************/
    exitstatus = project_propagator_field( eo_stochastic_source[0], eo_stochastic_source[0], 0, eo_evecs_field[0], sample_step_size, evecs_num, Vhalf);
    if (exitstatus != 0) {
      fprintf(stderr, "[loops_em] Error from project_propagator_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(35);
    }

    /***********************************************
     * invert
     ***********************************************/
    memset( eo_spinor_work[0], 0, sizeof_eo_spinor_field );
    memcpy( eo_spinor_work[1], eo_stochastic_source[0], sizeof_eo_spinor_field );
    exitstatus = tmLQCD_invert_eo ( eo_spinor_work[0], eo_spinor_work[1], _OP_ID_UP );
    if(exitstatus != 0) {
      fprintf(stderr, "[loops_em] Error from tmLQCD_invert_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }
    memcpy( eo_stochastic_propagator[0], eo_spinor_work[0], sizeof_eo_spinor_field );

    if( check_propagator_residual ) {
      exitstatus = check_oo_propagator_clover_eo( eo_stochastic_propagator, eo_stochastic_source, &(eo_spinor_work[0]), 
          gauge_field_with_phase, g_mzz_up, g_mzzinv_up, 1 );
      if(exitstatus != 0) {
        fprintf(stderr, "[loops_em] Error from check_oo_propagator_clover_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(19);
      }
    }

    /* multiply with 2 kappa */
    spinor_field_ti_eq_re ( eo_stochastic_propagator[0], 2.*g_kappa, Vhalf );

    /***********************************************
     * local loops
     ***********************************************/
    contract_local_loop_eo_stoch ( local_loop_x, eo_stochastic_propagator, eo_stochastic_source, 1, gauge_field_with_phase, mzz, mzzinv );

    /* momentum space */
    if ( ( exitstatus = cvc_loop_eo_momentum_projection ( &local_loop_p, local_loop_x, 16, g_sink_momentum_list, g_sink_momentum_number) ) != 0 ) {
      fprintf ( stderr, "[loop_em] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }

    /* write to file */
    sprintf(aff_tag, "/loop/local/sample%.4d", isample );
    if( ( exitstatus = cvc_loop_tp_write_to_aff_file ( local_loop_p, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc ) ) != 0 ) {
      fprintf(stderr, "[loops_em] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    /***********************************************
     * cvc fwd and bwd loop
     ***********************************************/
    contract_cvc_loop_eo_stoch ( cvc_loop_stoch_x, eo_stochastic_propagator, eo_stochastic_source, 1, gauge_field_with_phase, mzz, mzzinv );

    /* momentum projection, fwd */
    exitstatus = cvc_loop_eo_momentum_projection ( &cvc_loop_tp, cvc_loop_stoch[0], g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0 ) {
      fprintf(stderr, "[loops_em] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    sprintf(aff_tag, "/loop/cvc/fwd/sample%.4d", isample);
    exitstatus = cvc_loop_tp_write_to_aff_file ( cvc_loop_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[loops_em] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    for ( int mu = 0; mu < 4; mu++ ) {
      complex_field_eo2lexic ( cvc_loop_lexic, cvc_loop_stoch_x[0][mu][0], cvc_loop_stoch_x[0][mu][1] );

      if ( ( exitstatus = ft_4dim ( cvc_loop_stoch_p[0][mu], cvc_loop_lexic, 1, 0 ) ) != 0 ) {
        fprintf(stderr, "[loops_em] Error from ft_4dim, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }
    }

    /* momentum projection, bwd */
    exitstatus = cvc_loop_eo_momentum_projection ( &cvc_loop_tp, cvc_loop_stoch[1], g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0 ) {
      fprintf(stderr, "[loops_em] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    sprintf(aff_tag, "/loop/cvc/bwd/sample%.4d", isample);
    exitstatus = cvc_loop_tp_write_to_aff_file ( cvc_loop_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[loops_em] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    for ( int mu = 0; mu < 4; mu++ ) {
      complex_field_eo2lexic ( cvc_loop_lexic, cvc_loop_stoch_x[1][mu][0], cvc_loop_stoch_x[1][mu][1] );

      if ( ( exitstatus = ft_4dim ( cvc_loop_stoch_p[1][mu], cvc_loop_lexic, 1, 0 ) ) != 0 ) {
        fprintf(stderr, "[loops_em] Error from ft_4dim, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }
    }

    /***********************************************
     * accumulate sum of loops in momentum space
     ***********************************************/
    for ( unsigned int ix = 0; ix < 16 * VOLUME; ix++ ) {
      cvc_loop_stoch_p_accum[0][0][ix] += cvc_loop_stoch_p[0][0][ix];
    }


    /***********************************************/
    /***********************************************/

    /***********************************************
     * cvc loop convolutions for current sample
     ***********************************************/

    /* fwd - fwd - stoch - stoch */
    co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_stoch_p[0], cvc_loop_stoch_p[0], 0, VOLUME );
    exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[0]), jj_tensor_trace, 0 );

    /* fwd - bwd - stoch - stoch */
    co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_stoch_p[0], cvc_loop_stoch_p[1], 0, VOLUME );
    exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[1]), jj_tensor_trace, 0 );

    /* bwd - fwd - stoch - stoch */
    co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_stoch_p[1], cvc_loop_stoch_p[0], 0, VOLUME );
    exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[2]), jj_tensor_trace, 0 );

    /* bwd - bwd - stoch - stoch */
    co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_stoch_p[1], cvc_loop_stoch_p[1], 0, VOLUME );
    exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[3]), jj_tensor_trace, 0 );

    sprintf(aff_tag, "/loop/cvc-conv-cvc/stoch-stoch/sample%.4d", isample);
    affdir = aff_writer_mkpath(affw, affn, aff_tag);
    if ( ( exitstatus = aff_node_put_complex (affw, affdir, ztmp, (uint32_t)4 ) ) != 0 ) {
      fprintf(stderr, "[loops_em] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIEXITT(5);
    }

    /***********************************************/
    /***********************************************/

    /* fwd - fwd - lma - stoch */
    co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[0], cvc_loop_stoch_p[0], 0, VOLUME );
    exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[0]), jj_tensor_trace, 0 );

    /* fwd - fwd - lma - stoch */
    co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[0], cvc_loop_stoch_p[1], 0, VOLUME );
    exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[1]), jj_tensor_trace, 0 );

    /* fwd - fwd - lma - stoch */
    co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[1], cvc_loop_stoch_p[0], 0, VOLUME );
    exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[2]), jj_tensor_trace, 0 );

    /* fwd - fwd - lma - stoch */
    co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[1], cvc_loop_stoch_p[1], 0, VOLUME );
    exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[3]), jj_tensor_trace, 0 );

    sprintf(aff_tag, "/loop/cvc-conv-cvc/lma-stoch/sample%.4d", isample);
    affdir = aff_writer_mkpath(affw, affn, aff_tag);
    if ( ( exitstatus = aff_node_put_complex (affw, affdir, ztmp, (uint32_t)4 ) ) != 0 ) {
      fprintf(stderr, "[loops_em] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIEXITT(5);
    }


    /***********************************************/
    /***********************************************/

    /***********************************************
     * cvc loop convolutions for accumulated loops
     ***********************************************/

    if ( isample % Nsave == 0 ) {
      /* fwd - fwd - stoch - stoch */
      co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_stoch_p[0], cvc_loop_stoch_p[0], 0, VOLUME );
      exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[0]), jj_tensor_trace, 0 );

      /* fwd - bwd - stoch - stoch */
      co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_stoch_p[0], cvc_loop_stoch_p[1], 0, VOLUME );
      exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[1]), jj_tensor_trace, 0 );

      /* bwd - fwd - stoch - stoch */
      co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_stoch_p[1], cvc_loop_stoch_p[0], 0, VOLUME );
      exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[2]), jj_tensor_trace, 0 );

      /* bwd - bwd - stoch - stoch */
      co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_stoch_p[1], cvc_loop_stoch_p[1], 0, VOLUME );
      exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[3]), jj_tensor_trace, 0 );

      /* fwd - fwd - lma - stoch */
      co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[0], cvc_loop_stoch_p[0], 0, VOLUME );
      exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[4]), jj_tensor_trace, 0 );

      /* fwd - fwd - lma - stoch */
      co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[0], cvc_loop_stoch_p[1], 0, VOLUME );
      exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[5]), jj_tensor_trace, 0 );

      /* fwd - fwd - lma - stoch */
      co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[1], cvc_loop_stoch_p[0], 0, VOLUME );
      exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[6]), jj_tensor_trace, 0 );

      /* fwd - fwd - lma - stoch */
      co_field_eq_jj_disc_tensor_trace ( jj_tensor_trace, cvc_loop_lma_p[1], cvc_loop_stoch_p[1], 0, VOLUME );
      exitstatus = co_eq_complex_field_convolute_photon_scalar ( (double*)&(ztmp[7]), jj_tensor_trace, 0 );

      sprintf(aff_tag, "/loop/cvc-conv-cvc/Nsave%.4d/block%.4d", Nsave, isample/Nsave );
      affdir = aff_writer_mkpath(affw, affn, aff_tag);
      if ( ( exitstatus = aff_node_put_complex (affw, affdir, ztmp, (uint32_t)8 ) ) != 0 ) {
        fprintf(stderr, "[loops_em] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIEXITT(5);
      }

    }  /* end of if isample mod Nsave == 0 */

  }  /* end of loop on stochastic samples */

  /***********************************************/
  /***********************************************/

  fini_3level_buffer ( &local_loop_x );
  fini_3level_buffer ( &local_loop_p );
  fini_4level_buffer ( &cvc_loop_stoch_x );
  fini_3level_buffer ( &cvc_loop_tp );

  fini_3level_buffer ( &cvc_loop_stoch_p );
  fini_3level_buffer ( &cvc_loop_stoch_p_accum );
  fini_3level_buffer ( &cvc_loop_lma_p );

  fini_2level_buffer ( &eo_stochastic_source );
  fini_2level_buffer ( &eo_stochastic_propagator );

  fini_1level_buffer ( cvc_loop_lexic );
  fini_1level_buffer ( jj_tensor_trace );

  /***********************************************/
  /***********************************************/

#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[loops_em] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  /***********************************************/
  /***********************************************/

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  fini_2level_buffer ( &eo_spinor_field );
  fini_2level_buffer ( &eo_spinor_work );

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(eo_evecs_block);
#else
  exitstatus = tmLQCD_fini_deflator(_OP_ID_UP);
#endif
  free(eo_evecs_field);

  free ( evecs_eval );
  free ( evecs_lambdainv );
  free ( evecs_4kappasqr_lambdainv );

  /* free clover matrix terms */
  fini_clover ();

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [loops_em] %s# [loops_em] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [loops_em] %s# [loops_em] end of run\n", ctime(&g_the_time));
  }

  return(0);

}