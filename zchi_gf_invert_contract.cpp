/***************************************************************************
 *
 * zchi_gf_invert_contract
 *
 ***************************************************************************/

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

// #include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "cvc_timer.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "contractions_io.h"
#include "gamma.h"
#include "scalar_products.h"

#include "clover.h"

#include "gradient_flow.h"
#ifdef _GFLOW_QUDA
#include "quda.h"
#include "gauge_quda.h"
#endif

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#ifndef _USE_TIME_DILUTION
#define _USE_TIME_DILUTION 1
#endif

#if _USE_TIME_DILUTION
#warning "[zchi_gf_invert_contract] building WITH time dilution"
#else
#warning "[zchi_gf_invert_contract] building WITHOUT time dilution"
#endif

#define MAX_NUM_GF_NSTEP 100

using namespace cvc;

/***************************************************************************
 * helper message
 ***************************************************************************/
void usage() {
  fprintf(stdout, "Code for mixing probe correlators with mixing op at source\n");
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
  
  const char outfile_prefix[] = "kinop_gf";

  const char flavor_tag[4] = { 'u', 'd', 's', 'c' };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  struct timeval ta, tb, start_time, end_time;


  int read_stochastic_source  = 0;
  int write_stochastic_source = 0;
  int gf_nstep = 0;
  int gf_niter_list[MAX_NUM_GF_NSTEP];
  double gf_dt_list[MAX_NUM_GF_NSTEP];

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sSch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 's':
      read_stochastic_source = 1;
      break;
    case 'S':
      write_stochastic_source = 1;
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

  fprintf(stdout, "# [zchi_gf_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [zchi_gf_invert_contract] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [zchi_gf_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [zchi_gf_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[zchi_gf_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[zchi_gf_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [zchi_gf_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [zchi_gf_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[zchi_gf_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[zchi_gf_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[zchi_gf_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize the clover term, 
   * lmzz and lmzzinv
   *
   *   mzz = space-time diagonal part of the Dirac matrix
   *   l   = light quark mass
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &lmzz, &lmzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[zchi_gf_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[zchi_gf_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [zchi_gf_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );
  size_t const sizeof_gauge_field = 72 * VOLUME  * sizeof( double );

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[zchi_gf_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  double ** spinor_work = init_2level_dtable ( 3, _GSI( VOLUME+RAND ) );
  if ( spinor_work == NULL ) {
    fprintf ( stderr, "[zchi_gf_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

#ifdef _GFLOW_QUDA
    /***************************************************************************
     * dummy solve, just to have original gauge field up on device,
     * for subsequent GF application
     ***************************************************************************/
  memset(spinor_work[1], 0, sizeof_spinor_field);
  memset(spinor_work[0], 0, sizeof_spinor_field);
  if ( g_cart_id == 0 ) spinor_work[0][0] = 1.;
  exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], _OP_ID_UP);
#  if ( defined GPU_DIRECT_SOLVER )
  if(exitstatus < 0)
#  else
  if(exitstatus != 0)
#  endif
  {
    fprintf(stderr, "[zchi_gf_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(12);
  }
#endif  /* of if _SMEAR_QUDA */

  /***************************************************************************
   * reshape gauge field
   ***************************************************************************/
#ifdef _GFLOW_QUDA
  double ** h_gauge = init_2level_dtable ( 4, 18*VOLUME );
  if ( h_gauge == NULL )
  {
    fprintf(stderr, "[zchi_gf_invert_contract] Error from init_2level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }
  gauge_field_cvc_to_qdp ( h_gauge, gauge_field_with_phase );


  /***************************************************************************
   * gauge_param initialization
   ***************************************************************************/
  QudaGaugeParam gauge_param;
  init_gauge_param ( &gauge_param );

#elif defined _GFLOW_CVC
  double * gauge_field_gf = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
  if ( gauge_field_gf == NULL )
  {
    fprintf(stderr, "[zchi_gf_invert_contract] Error from init_1level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }
#endif
  double * gauge_field_aux = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
  if ( gauge_field_aux == NULL )
  {
    fprintf(stderr, "[zchi_gf_invert_contract] Error from init_1level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }

  /***************************************************************************
   * gradient flow parameters
   ***************************************************************************/
  gf_nstep = 21;
  gf_niter_list[0] = 0;
  gf_dt_list[0] = 0.01;
  for( int i = 1; i < gf_nstep; i++ )
  {
    gf_niter_list[i] = 1;
    gf_dt_list[i] = 0.01;
  }

  /***************************************************************************
   * loop on samples
   * invert and contract loops
   ***************************************************************************/
  for ( int isample = 0; isample < g_nsample; isample++ ) 
  {

    prepare_volume_source ( spinor_work[0], VOLUME );

    /***************************************************************************
     * write loop field to lime file
     ***************************************************************************/
    if ( write_stochastic_source ) 
    {
      sprintf( filename, "source.c%d.N%d.lime", Nconf, isample );

      exitstatus = write_propagator( spinor_work[0], filename, 0, 64 );
      if ( exitstatus != 0  ) {
        fprintf ( stderr, "[zchi_gf_invert_contract] Error write_lemon_spinor, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }
    }  /* end of if write_stochastic_source */

    /***************************************************************************
     * gradient flow in stochastic source and propagator
     ***************************************************************************/
#ifdef _GFLOW_QUDA
    /* reset: upload original gauge field to device */
    loadGaugeQuda ( (void *)h_gauge, &gauge_param );
#elif defined _GFLOW_CVC
    memcpy ( gauge_field_gf, gauge_field_with_phase, sizeof_gauge_field );
#endif

    /* tm-rotate stochastic propagator at source */
    if( g_fermion_type == _TM_FERMION ) 
    {
      int const tm_rotation_sign = ( ( g_mu > 0 ) ? 1 : -1 ) * ( 1 - 2 * (_OP_ID_UP ) ) ;
      if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [zchi_gf_invert_contract] tm_rotation_sign = %d   %s %d\n", tm_rotation_sign, __FILE__, __LINE__ );
      spinor_field_tm_rotation(spinor_work[2], spinor_work[0], tm_rotation_sign, g_fermion_type, VOLUME);
    }

    /* call to (external/dummy) inverter / solver */
    exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[2], _OP_ID_UP );
#  if ( defined GPU_DIRECT_SOLVER )
    if(exitstatus < 0)
#  else
    if(exitstatus != 0)
#  endif
    {
      fprintf(stderr, "[zchi_gf_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(12);
    }

    if ( check_propagator_residual ) {
      check_residual_clover ( &(spinor_work[1]), &(spinor_work[2]), gauge_field_with_phase, lmzz[_OP_ID_UP], lmzzinv[_OP_ID_UP], 1 );
    }

    /* tm-rotate stochastic propagator at sink, in-place */
    if( g_fermion_type == _TM_FERMION ) 
    {
      int const tm_rotation_sign = ( ( g_mu > 0 ) ? 1 : -1 ) * ( 1 - 2 * (_OP_ID_UP ) ) ;
      if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [zchi_gf_invert_contract] tm_rotation_sign = %d   %s %d\n", tm_rotation_sign, __FILE__, __LINE__ );
      spinor_field_tm_rotation(spinor_work[1], spinor_work[1], tm_rotation_sign, g_fermion_type, VOLUME);
    }

    /***************************************************************************
     * (re-)set gauge field to flowtime zero
     ***************************************************************************/
#ifdef _GFLOW_QUDA
    gauge_field_cvc_to_qdp ( h_gauge, gauge_field_with_phase );

    /* reset: upload original gauge field to device */
    loadGaugeQuda ( (void *)h_gauge, &gauge_param );
#elif defined _GFLOW_CVC
    memcpy ( gauge_field_gf, gauge_field_with_phase, sizeof_gauge_field );
#endif

    /* cumulative flow time */
    double gf_tau = 0;

    /***************************************************************************
     * loop on GF steps
     ***************************************************************************/
    for ( int igf = 0; igf < gf_nstep; igf++ )
    {
      int const gf_niter   = gf_niter_list[igf];
      double const gf_dt   = gf_dt_list[igf];
      double const gf_dtau = gf_niter * gf_dt;
      gf_tau += gf_dtau;
 
      if ( gf_dtau > 0. )
      {
        if ( g_cart_id == 0 ) fprintf(stdout, "# [zchi_gf_invert_contract] GF for dtau = %e\n", gf_dtau );
        gettimeofday ( &ta, (struct timezone *)NULL );
#ifdef _GFLOW_QUDA
        QudaGaugeSmearParam smear_param;
        smear_param.n_steps       = gf_niter;
        smear_param.epsilon       = gf_dt;
        smear_param.meas_interval = 1;
        smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;

        /***************************************************************************
	 * flow the stochastic source
	 *
	 * do not update the gauge field
         ***************************************************************************/

        _performGFlownStep ( spinor_work[0], spinor_work[0], &smear_param, 0 );

#elif defined _GFLOW_CVC
        flow_fwd_gauge_spinor_field ( gauge_field_gf, spinor_work[0], gf_niter, gf_dt, 1, 1, 0 );
#endif    

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "zchi_gf_invert_contract", "forward gradient flow", g_cart_id == 0 );

        /***************************************************************************
         * flow the stochastic propagator
         *
         * update the gauge field
         ***************************************************************************/

        gettimeofday ( &ta, (struct timezone *)NULL );
        /* update resident gaugeFlowed */
#ifdef _GFLOW_QUDA
        _performGFlownStep ( spinor_work[1], spinor_work[1], &smear_param, 1 );

        /* download flowed gauge field */
        saveGaugeQuda ( h_gauge, &gauge_param );
        gauge_field_qdp_to_cvc ( gauge_field_aux, h_gauge );

#elif defined _GFLOW_CVC
        flow_fwd_gauge_spinor_field ( gauge_field_gf, spinor_work[1], gf_niter, gf_dt, 1, 1, 1 );
        memcpy ( gauge_field_aux, gauge_field_gf, sizeof_gauge_field );
#endif
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "zchi_gf_invert_contract", "forward gradient flow", g_cart_id == 0 );

      } else {

        if ( g_cart_id == 0 ) fprintf(stdout, "# [zchi_gf_invert_contract] no GF\n" );
#ifdef _GFLOW_CVC
        memcpy ( gauge_field_aux, gauge_field_gf, sizeof_gauge_field );
#endif

      }  /* end of if do any flow */

#ifdef HAVE_MPI
      xchange_gauge_field( gauge_field_aux );
#endif
      plaquetteria  ( gauge_field_aux );

      /***************************************************************************
       * kinetic operator
       ***************************************************************************/

      double ** Dspinor_field = init_2level_dtable ( 2, _GSI( VOLUME ) );
      if ( Dspinor_field == NULL )
      {
        fprintf ( stderr, "[zchi_gf_invert_contract] Error from init_level_table    %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }

      complex w_total = { 0., 0. };


      for ( int mu = 0; mu < 4; mu++ )
      {

        spinor_field_eq_cov_displ_spinor_field ( Dspinor_field[0], spinor_work[1], mu, 0, gauge_field_aux );

        spinor_field_eq_cov_displ_spinor_field ( Dspinor_field[1], spinor_work[1], mu, 1, gauge_field_aux );

        spinor_field_eq_spinor_field_mi_spinor_field ( Dspinor_field[0], Dspinor_field[0],  Dspinor_field[1], VOLUME );

        spinor_field_eq_gamma_ti_spinor_field ( Dspinor_field[1], mu, Dspinor_field[0], VOLUME );

        complex w = { 0., 0. };

        spinor_scalar_product_co ( &w, spinor_work[0], Dspinor_field[1], VOLUME );

        /***************************************************************************
         * normalize to 1/2 x [ fwd deriv + bwd deriv ] / ( T x L^3 )
         ***************************************************************************/
        w.re *= 0.5 / ( (double)VOLUME * g_nproc );
        w.im *= 0.5 / ( (double)VOLUME * g_nproc );

        w_total.re += w.re;
        w_total.im += w.im;

        if ( io_proc == 2 )
        {
          int const ncdim = 1;
          int const cdim[1] = {2};
          char tag[100];
          sprintf(filename, "%s.c%d.h5", outfile_prefix, Nconf );
          sprintf ( tag, "/s%d/%s/tau%6.4f/mu%d", isample, "up", gf_tau, mu );
          
          write_h5_contraction ( &w, NULL, filename, tag, "double", ncdim, cdim );
        }

      }  /* end of loop on directions mu */

      if ( io_proc == 2 )
      {
        int const ncdim = 1;
        int const cdim[1] = {2};
        char tag[100];
        sprintf(filename, "%s.c%d.h5", outfile_prefix, Nconf );
        sprintf ( tag, "/s%d/%s/tau%6.4f/mu%d", isample, "up", gf_tau, 4 );
          
        write_h5_contraction ( &w_total, NULL, filename, tag, "double", ncdim, cdim );
      }


      fini_2level_dtable ( &Dspinor_field );

    }  /* end of loop on GF steps  */

#if defined _GFLOW_CVC
    flow_fwd_gauge_spinor_field ( NULL, NULL, 0, 0, 0, 0, 0 );
#endif

  }  /* end of loop on samples */

  fini_2level_dtable ( &spinor_work );

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif

#ifdef _GFLOW_QUDA
  fini_2level_dtable ( &h_gauge );
#elif defined _GFLOW_CVC
  fini_1level_dtable ( &gauge_field_gf );
#endif
  fini_1level_dtable ( &gauge_field_aux );
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
  show_time ( &start_time, &end_time, "zchi_gf_invert_contract", "runtime", g_cart_id == 0 );

  return(0);

}
