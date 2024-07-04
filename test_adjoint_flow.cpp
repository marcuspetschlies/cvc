/***************************************************************************
 *
 * test_adjoint_flow
 *
 ***************************************************************************/

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

#ifdef _GFLOW_QUDA
#warning "including quda header file quda.h directly "
#include "quda.h"
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
#include "cvc_timer.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "gauge_io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "gamma.h"
#include "clover.h"
#include "gradient_flow.h"
#include "contractions_io.h"
#include "scalar_products.h"
#include "gauge_quda.h"
#include "fermion_quda.h"

using namespace cvc;

/***************************************************************************
 * helper message
 ***************************************************************************/
void usage() {
  fprintf(stdout, "Code for testing gauge field up and download\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  fprintf(stdout, "          -h                  : this message\n");
  EXIT(0);
}

/***************************************************************************
 *
 * MAIN PROGRAM
 *
 ***************************************************************************/
int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "test_adjoint_flow";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[400];
  struct timeval ta, tb, start_time, end_time;
  int gf_niter = 20;
  int gf_ns = 2;
  double gf_dt = 0.01;
  int gf_nb;


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

  gettimeofday ( &start_time, (struct timezone *)NULL );


  /***************************************************************************
   * read input and set the default values
   ***************************************************************************/
  if(filename_set==0) strcpy(filename, "twopt.input");
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_adjoint_flow] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize tmLQCD solvers
   ***************************************************************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
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

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /***************************************************************************
   * report git version
   * make sure the version running here has been commited before program call
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_adjoint_flow] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_adjoint_flow] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_adjoint_flow] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_adjoint_flow] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[test_adjoint_flow] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  
  /***************************************************************************
   * initialize lattice geometry
   *
   * allocate and fill geometry arrays
   ***************************************************************************/
  geometry();


  /***************************************************************************
   * set up some mpi exchangers for
   * (1) even-odd decomposed spinor field
   * (2) even-odd decomposed propagator field
   ***************************************************************************/
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***************************************************************************
   * set up the gauge field
   *
   *   either read it from file or get it from tmLQCD interface
   *
   *   lime format is used
   ***************************************************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [test_adjoint_flow] reading gauge field from file %s\n", filename);

    exitstatus = read_lime_gauge_field_doubleprec(filename);

  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_adjoint_flow] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_adjoint_flow] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[test_adjoint_flow] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  double *gauge_field_with_phase = NULL;
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  /* exitstatus = gauge_field_eq_gauge_field_ti_bcfactor ( &gauge_field_with_phase, g_gauge_field, -1. ); */
  if(exitstatus != 0) {
    fprintf(stderr, "[test_adjoint_flow] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize the clover term, 
   * lmzz and lmzzinv
   *
   *   mzz = space-time diagonal part of the Dirac matrix
   *   l   = light quark mass
   ***************************************************************************/
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  exitstatus = init_clover ( &g_clover, &lmzz, &lmzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_adjoint_flow] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[test_adjoint_flow] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [test_adjoint_flow] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;

  size_t const sizeof_spinor_field = _GSI ( VOLUME ) * sizeof ( double ) ;
  size_t const sizeof_gauge_field  = 72 * VOLUME * sizeof ( double );

#ifdef _GFLOW_QUDA
#if 0
  /***************************************************************************
   * test solve, just to have original gauge field up on device
   ***************************************************************************/
  double ** spinor_work = init_2level_dtable ( 2, _GSI( VOLUME + RAND ) );
  memset(spinor_work[1], 0, sizeof_spinor_field);
  memset(spinor_work[0], 0, sizeof_spinor_field);
  if ( g_cart_id == 0 ) spinor_work[0][0] = 1.;
  exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], 0 );
#  if ( defined GPU_DIRECT_SOLVER )
  if(exitstatus < 0)
#  else
  if(exitstatus != 0)
#  endif
  {
    fprintf(stderr, "[test_adjoint_flow] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(12);
  }
  check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, lmzz[0], lmzzinv[0], 1 );

  // sprintf ( filename, "propagator.0" );
  // write_propagator ( spinor_work[1], filename, 0, 64 );

  fini_2level_dtable ( &spinor_work );

  _loadGaugeQuda( NO_COMPRESSION );
#endif  // of if 0

  double ** h_gauge = init_2level_dtable ( 4, 18*VOLUME );
  if ( h_gauge == NULL )
  {
    fprintf(stderr, "[test_adjoint_flow] Error from init_2level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }

  /***************************************************************************
   * gauge_param initialization
   ***************************************************************************/
  QudaGaugeParam gauge_param;

  init_gauge_param ( &gauge_param );
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;
  gauge_param.type = QUDA_WILSON_LINKS;

  /* reshape gauge field */
  gauge_field_cvc_to_qdp ( h_gauge, gauge_field_with_phase );
  
  /* upload to device */
  loadGaugeQuda ( (void *)h_gauge, &gauge_param );

#endif

#ifdef _GFLOW_QUDA
  double * gauge_field_smeared = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
  if ( gauge_field_smeared == NULL )
  {
    fprintf(stderr, "[test_adjoint_flow] Error from init_1level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }
#endif

  /***************************************************************************
   * initialize random spinor field
   ***************************************************************************/
  double ** spinor_field = init_2level_dtable ( 4, _GSI( VOLUME ) );
  if ( spinor_field == NULL )
  {
    fprintf ( stderr, "[test_adjoint_flow] Error from init_2level_dtable    %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

  /***************************************************************************
   * up- and down-load gauge field
   ***************************************************************************/

#ifdef _GFLOW_QUDA
  /***************************************************************************
   * upload gauge field
   ***************************************************************************/
  // memcpy ( gauge_field_smeared, gauge_field_with_phase, sizeof_gauge_field );
  // unit_gauge_field ( gauge_field_smeared, VOLUME );
  random_gauge_field ( gauge_field_smeared, 1. );

  double pl = 0.;
  plaquette2 ( &pl,  gauge_field_smeared );
  if ( g_cart_id == 0 )
  {
	  fprintf(stdout, "# [test_adjoint_flow] pl = %25.16e    %s %d\n", pl, __FILE__, __LINE__ );
  } 
  
  /* reshape gauge field */
  gauge_field_cvc_to_qdp ( h_gauge, gauge_field_smeared );

  gauge_param.location = QUDA_CPU_FIELD_LOCATION;
  gauge_param.type = QUDA_FLOWED_LINKS;
  
  /* upload to device */
  loadGaugeQuda ( (void *)h_gauge, &gauge_param );


#if 0
  /* to really check, set both gauge fields to zero */
  memset ( h_gauge[0], 0, 4 * 18 * VOLUME * sizeof ( double ) );
  
  memset ( gauge_field_smeared, 0, sizeof_gauge_field );
  
  /* download from device */
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;
  
  saveGaugeQuda ( (void*)h_gauge, &gauge_param );
  
  gauge_field_qdp_to_cvc ( gauge_field_smeared, h_gauge );
 

  double norm_diff = 0.;
  for ( unsigned int ix = 0; ix < 72*VOLUME; ix++ )
  {
    double d = gauge_field_smeared[ix] - gauge_field_with_phase[ix];
    norm_diff += d*d;
  }
  double ndtmp = norm_diff;
  MPI_Allreduce ( &ndtmp , &norm_diff, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid );

  if ( io_proc == 2 )
  {
	  fprintf(stdout, "# [test_adjoint_flow] gauge field norm_diff %25.16e\n", norm_diff );
  }
#endif  // of if 0
    
  //sprintf ( filename, "gauge_field.1" );
  //exitstatus = write_lime_contraction( gauge_field_smeared, filename,  64, 36, "gauge", Nconf, 0 );
  //if ( exitstatus != 0 )
  //{
  //  fprintf ( stderr, "[test_adjoint_flow] Error from write_lime_contraction, status %d   %s %d\n", exitstatus, __FILE__, __LINE__ );
  //  EXIT(12);
  //}
#if 0
#endif  // of if 0

#endif   /* of #ifdef _GFLOW_QUDA  */


  prepare_volume_source ( spinor_field[0], VOLUME );

  memcpy ( spinor_field[1], spinor_field[0], sizeof_spinor_field ); 

    /***************************************************************************
     * flow the gauge field and the spinor field on the gpu
     ***************************************************************************/

#ifdef _GFLOW_QUDA
 
    // _performWuppertalnStep ( spinor_field[0], spinor_field[0], 1, 1. );

    QudaGaugeSmearParam smear_param;
    smear_param.n_steps       = gf_niter;
    smear_param.epsilon       = gf_dt;
    smear_param.meas_interval = 1;
    smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;

    gf_nb = (int) ceil ( pow ( (double)gf_niter , 1./ ( (double)gf_ns + 1. ) ) );
    if ( io_proc == 2 ) fprintf (stdout, "# [test_adjoint_flow] gf_nb = %d   %s %d\n", gf_nb, __FILE__, __LINE__ );

    _performGFlowAdjoint ( spinor_field[0], spinor_field[0], &smear_param, gf_niter, gf_nb, gf_ns );

    //saveGaugeQuda ( (void*)h_gauge, &gauge_param );
    //
    //gauge_field_qdp_to_cvc ( gauge_field_smeared, h_gauge );

     _performGFlowAdjoint ( NULL, NULL, &smear_param, gf_niter, gf_nb, -1 );

#if 0
    sprintf ( filename, "gauge_field.gpuflow" );
    exitstatus = write_lime_contraction( gauge_field_smeared, filename,  64, 36, "gauge", Nconf, 0 );
    if ( exitstatus != 0 )
    {
      fprintf ( stderr, "[test_adjoint_flow] Error from write_lime_contraction, status %d   %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }
#endif  /* of if 0 */

#endif  /* of if _GFLOW_QUDA */


    /***************************************************************************
     * now flow the gauge field and the spinor field the same way on the cpu
     ***************************************************************************/
#ifdef _GFLOW_CVC

    double ** gauge_field_smeared_cpu = init_2level_dtable ( gf_ns+1, 72 * VOLUMEPLUSRAND );
    if ( gauge_field_smeared_cpu == NULL )
    {
      fprintf(stderr, "[test_adjoint_flow] Error from init_1level_dtable   %s %d\n", __FILE__, __LINE__);
      EXIT(12);
    }

    // memcpy ( gauge_field_smeared_cpu[gf_ns], gauge_field_with_phase, sizeof_gauge_field );
    memcpy ( gauge_field_smeared_cpu[gf_ns], gauge_field_smeared, sizeof_gauge_field );

    flow_adjoint_gauge_spinor_field ( gauge_field_smeared_cpu, spinor_field[1], gf_dt, gf_niter, gf_nb, gf_ns );

    //for ( int i = 0; i <= gf_ns; i++ )
    //{
    //  double plaq = 0.;
    //  plaquette2 ( &plaq, gauge_field_smeared_cpu[i] );
    //  if ( io_proc == 2 ) fprintf (stdout, "# [test_adjoint_flow] plaq gauge_field_smeared_cpu %d %25.16e    %s %d\n", i, plaq, __FILE__, __LINE__ );
    //}

    flow_adjoint_step_gauge_spinor_field ( NULL, NULL, 0, 0., 0, 0 );

#endif  /* of if _GFLOW_CVC */

#if defined _GFLOW_CVC && defined _GFLOW_QUDA
    double normg = 0., normc = 0.;
    spinor_scalar_product_re ( &normg, spinor_field[0], spinor_field[0], VOLUME ) ;
    spinor_scalar_product_re ( &normc, spinor_field[1], spinor_field[1], VOLUME ) ;

    spinor_field_pl_eq_spinor_field_ti_re ( spinor_field[1], spinor_field[0] , -1., VOLUME );
    double norm_diff = 0;
    spinor_scalar_product_re ( &norm_diff, spinor_field[1], spinor_field[1], VOLUME ) ;
    if ( io_proc == 2 )
    {
	    fprintf (stdout, "# [] norm_diff %25.16e  normc %25.16e normg %25.16e    %s %d\n", norm_diff, normc, normg, __FILE__, __LINE__);
    }


#endif

#ifdef _GFLOW_CVC
  fini_2level_dtable ( &gauge_field_smeared_cpu );
#endif
  fini_2level_dtable ( &spinor_field );

#ifdef _GFLOW_QUDA
  fini_2level_dtable ( &h_gauge );
  fini_1level_dtable ( &gauge_field_smeared );
#endif

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  if ( g_gauge_field != NULL ) free(g_gauge_field);
#endif
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( &lmzz, &lmzzinv );

  /* free lattice geometry arrays */
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

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "test_adjoint_flow", "runtime", g_cart_id == 0 );

  return(0);

}
