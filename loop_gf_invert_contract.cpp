/***************************************************************************
 *
 * loop_gf_invert_contract
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
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "contractions_io.h"
// #include "contract_factorized.h"
// #include "contract_diagrams.h"
#include "gamma.h"

#include "clover.h"

#include "pm.h"
#include "gradient_flow.h"
#include "gauge_quda.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#ifndef _USE_TIME_DILUTION
#define _USE_TIME_DILUTION 1
#endif

#if _USE_TIME_DILUTION
#warning "[loop_gf_invert_contract] building WITH time dilution"
#else
#warning "[loop_gf_invert_contract] building WITHOUT time dilution"
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
  
  const char outfile_prefix[] = "loop_gf";

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


  int read_scalar_field  = 0;
  int write_scalar_field = 0;
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
      read_scalar_field = 1;
      break;
    case 'S':
      write_scalar_field = 1;
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

  fprintf(stdout, "# [loop_gf_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [loop_gf_invert_contract] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [loop_gf_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [loop_gf_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[loop_gf_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[loop_gf_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [loop_gf_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [loop_gf_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[loop_gf_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[loop_gf_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[loop_gf_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    fprintf(stderr, "[loop_gf_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[loop_gf_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [loop_gf_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * set the gamma matrices
   ***************************************************************************/

  gamma_matrix_type gamma_v[4];
  gamma_matrix_set( &( gamma_v[0]), 0, 1. );
  gamma_matrix_set( &( gamma_v[1]), 1, 1. );
  gamma_matrix_set( &( gamma_v[2]), 2, 1. );
  gamma_matrix_set( &( gamma_v[3]), 3, 1. );

  gamma_matrix_type gamma_s[1], gamma_p[1] ;
  gamma_matrix_set( &( gamma_s[0]), 4, 1. );
  gamma_matrix_set( &( gamma_p[0]), 5, 1. );

  gamma_matrix_type gamma_a[4];
  gamma_matrix_set( &( gamma_a[0]), 6, 1. );
  gamma_matrix_set( &( gamma_a[1]), 7, 1. );
  gamma_matrix_set( &( gamma_a[2]), 8, 1. );
  gamma_matrix_set( &( gamma_a[3]), 9, 1. );

  gamma_matrix_type sigma_munu[6];
  gamma_matrix_set( &( sigma_munu[0]), 10, 1. );
  gamma_matrix_set( &( sigma_munu[1]), 11, 1. );
  gamma_matrix_set( &( sigma_munu[2]), 12, 1. );
  gamma_matrix_set( &( sigma_munu[3]), 13, 1. );
  gamma_matrix_set( &( sigma_munu[4]), 14, 1. );
  gamma_matrix_set( &( sigma_munu[5]), 15, 1. );


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
    fprintf(stderr, "[loop_gf_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  double ** spinor_work = init_2level_dtable ( 3, _GSI( VOLUME+RAND ) );
  if ( spinor_work == NULL ) {
    fprintf ( stderr, "[loop_gf_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
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
    fprintf(stderr, "[loop_gf_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(12);
  }
#endif  /* of if _SMEAR_QUDA */


  /***************************************************************************
   ***************************************************************************
   **
   ** Part I
   **
   ** prepare stochastic sources for W-type sequential sources and propagators
   **
   ***************************************************************************
   ***************************************************************************/

  double ** scalar_field = NULL;

  if ( g_nsample > 0 ) {
    scalar_field = init_2level_dtable ( g_nsample, 2*VOLUME );
    if( scalar_field == NULL ) {
      fprintf(stderr, "[loop_gf_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(132);
    }
  }

  if ( ! read_scalar_field  && ( g_nsample > 0 ) ) {

    /***************************************************************************
     * draw a stochastic binary source (real, +/1 one per site )
     ***************************************************************************/
    ranbinary ( scalar_field[0], 2 * g_nsample * VOLUME );

    /***************************************************************************
     * write loop field to lime file
     ***************************************************************************/
    if ( write_scalar_field ) {
      
      char field_type[2000];

      sprintf( field_type, "<source_type>%d</source_type><noise_type>binary real</noise_type><coherent_sources>%d</coherent_sources>", g_source_type , g_coherent_source_number );

      for ( int i = 0; i < g_nsample; i++ ) {
        sprintf( filename, "scalar_field.c%d.N%d.lime", Nconf, i );

        exitstatus = write_lime_contraction( scalar_field[i], filename, 64, 1, field_type, Nconf, 0 );
        if ( exitstatus != 0  ) {
          fprintf ( stderr, "[loop_gf_invert_contract] Error write_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(12);
        }
      }
    }  /* end of if write_scalar_field */

  } else {
      
    for ( int i = 0; i < g_nsample; i++ ) {
      sprintf( filename, "scalar_field.c%d.N%d.lime", Nconf, i );

      exitstatus = read_lime_contraction ( scalar_field[i], filename, 1, 0 );
      if ( exitstatus != 0  ) {
        fprintf ( stderr, "[loop_gf_invert_contract] Error read_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }
    }
  }  /* end of if read scalar field */


  /***************************************************************************
   * reshape gauge field
   ***************************************************************************/
#ifdef _GFLOW_QUDA
  double ** h_gauge = init_2level_dtable ( 4, 18*VOLUME );
  if ( h_gauge == NULL )
  {
    fprintf(stderr, "[loop_gf_invert_contract] Error from init_2level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }
  gauge_field_cvc_to_qdp ( h_gauge, gauge_field_with_phase );


  /***************************************************************************
   * Begin of gauge_param initialization
   ***************************************************************************/
  QudaGaugeParam gauge_param;

  gauge_param.struct_size = sizeof ( QudaGaugeParam );
 
  /* gauge_param.location = QUDA_CUDA_FIELD_LOCATION; */
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  gauge_param.X[0] = LX;
  gauge_param.X[1] = LY;
  gauge_param.X[2] = LZ;
  gauge_param.X[3] = T;

  gauge_param.anisotropy    = 1.0;
  gauge_param.tadpole_coeff = 0.0;
  gauge_param.scale         = 0.0;

  gauge_param.type = QUDA_FLOWED_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;  /* expect *gauge[mu], even-odd, spacetime, row-column color */

  gauge_param.t_boundary = QUDA_PERIODIC_T; 

  gauge_param.cpu_prec = QUDA_DOUBLE_PRECISION;

  gauge_param.cuda_prec   = QUDA_DOUBLE_PRECISION;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

  gauge_param.cuda_prec_sloppy   = QUDA_DOUBLE_PRECISION;
  gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;

  gauge_param.cuda_prec_refinement_sloppy   = QUDA_DOUBLE_PRECISION;
  gauge_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;

  gauge_param.cuda_prec_precondition   = QUDA_DOUBLE_PRECISION;
  gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;

  gauge_param.cuda_prec_eigensolver   = QUDA_DOUBLE_PRECISION;
  gauge_param.reconstruct_eigensolver = QUDA_RECONSTRUCT_NO;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = get_gauge_padding ( gauge_param.X );

  gauge_param.site_ga_pad = 0;

  gauge_param.staple_pad   = 0;
  gauge_param.llfat_ga_pad = 0;
  gauge_param.mom_ga_pad   = 0;
  
  gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
  
  gauge_param.staggered_phase_applied = QUDA_STAGGERED_PHASE_NO;

  gauge_param.i_mu = 0.;

  gauge_param.overlap = 0;

  gauge_param.overwrite_mom = false;

  gauge_param.use_resident_gauge  = false;
  gauge_param.use_resident_mom    = false;
  gauge_param.make_resident_gauge = false;
  gauge_param.make_resident_mom   = false;
  gauge_param.return_result_gauge = false;
  gauge_param.return_result_mom   = false;

  gauge_param.gauge_offset = 0;
  gauge_param.mom_offset   = 0;

  /***************************************************************************
   * End of gauge_param initialization
   ***************************************************************************/

  /* TEST PLAQUETTE */
  double * gauge_field_aux = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
  if ( gauge_field_aux == NULL )
  {
    fprintf(stderr, "[loop_gf_invert_contract] Error from init_1level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }

#elif defined _GFLOW_CVC
  double * gauge_field_gf = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
  if ( gauge_field_gf == NULL )
  {
    fprintf(stderr, "[loop_gf_invert_contract] Error from init_1level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }
#endif

  gf_nstep = 2;
  gf_niter_list[0] = 0;
  gf_niter_list[1] = 3;
//  gf_niter_list[2] = 3;
  gf_dt_list[0] = 0.01;
  gf_dt_list[1] = 0.01;
//  gf_dt_list[2] = 0.01;


  /***************************************************************************
   * loop on samples
   * invert and contract loops
   ***************************************************************************/
  for ( int isample = 0; isample < g_nsample; isample++ ) 
  {


    /***************************************************************************
     * gradient flow in stochastic source and propagator
     ***************************************************************************/
#ifdef _GFLOW_QUDA
    /* reset: upload original gauge field to device */
    loadGaugeQuda ( (void *)h_gauge, &gauge_param );
#elif defined _GFLOW_CVC
    memcpy ( gauge_field_gf, gauge_field_with_phase, sizeof_gauge_field );
#endif

    double _Complex **** loop = NULL;

    loop = init_4level_ztable ( gf_nstep, VOLUME, 12, 12 );
    if ( loop  == NULL ) {
      fprintf ( stderr, "[loop_gf_invert_contract] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }


    /***************************************************************************
     * loop on spin and color index 
     *
     * we use spin-color dilution here:
     *   for each inversion we select only a single spin-color component 
     *   from stochastic_source
     ***************************************************************************/
#if _USE_TIME_DILUTION
    for ( int timeslice = 0; timeslice < T_global; timeslice ++ )
    {
#endif

      for ( int ispin = 0; ispin < 4; ispin++ ) 
      {

        for ( int icol = 0; icol < 3; icol++ ) 
        {

          int const isc = 3 * ispin + icol;

          memset ( spinor_work[0], 0, sizeof_spinor_field );
          memset ( spinor_work[1], 0, sizeof_spinor_field );
          memset ( spinor_work[2], 0, sizeof_spinor_field );
 
#if _USE_TIME_DILUTION
          if ( timeslice / T == g_proc_coords[0] ) {
            if ( g_verbose > 2 ) fprintf( stdout, "# [loop_gf_invert_contract] proc %d has global timeslice %d %s %d\n",
                g_cart_id, timeslice, __FILE__, __LINE__ );
            
            size_t const loffset = ( timeslice % T ) * VOL3;
            size_t const offset  = _GSI( loffset );
          
#pragma omp parallel for
            for ( unsigned int ix = 0; ix < VOL3; ix++  ) {
              size_t const iy = offset + _GSI(ix) + 2 * isc;  /* offset for site ix and spin-color isc */
              size_t const iz = 2 * ( loffset + ix );
              spinor_work[0][ iy     ] = scalar_field[isample][ iz     ];
              spinor_work[0][ iy + 1 ] = scalar_field[isample][ iz + 1 ];
            }
          }

#else  /* of if _USE_TIME_DILUTION */

#pragma omp parallel for
          for ( unsigned int ix = 0; ix < VOLUME; ix++  ) {
            size_t const iy = _GSI(ix) + 2 * isc;  /* offset for site ix and spin-color isc */
            spinor_work[0][ iy    ] = scalar_field[isample][ 2 * ix     ];
            spinor_work[0][ iy + 1] = scalar_field[isample][ 2 * ix + 1 ];
          }
#endif

          /* tm-rotate stochastic propagator at source, in-place */
          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation(spinor_work[2], spinor_work[0], 1, g_fermion_type, VOLUME);
          }

          /* call to (external/dummy) inverter / solver */
          exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[2], _OP_ID_UP );
#  if ( defined GPU_DIRECT_SOLVER )
          if(exitstatus < 0)
#  else
          if(exitstatus != 0)
#  endif
          {
            fprintf(stderr, "[loop_gf_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(12);
          }

          if ( check_propagator_residual ) {
            check_residual_clover ( &(spinor_work[1]), &(spinor_work[2]), gauge_field_with_phase, lmzz[_OP_ID_UP], lmzzinv[_OP_ID_UP], 1 );
          }

          /* tm-rotate stochastic propagator at sink */
          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation(spinor_work[1], spinor_work[1], 1, g_fermion_type, VOLUME);
          }

     	  /***************************************************************************
          '* (re-)set gauge field to flowtime zero
     	   ***************************************************************************/
#ifdef _GFLOW_QUDA
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
              if ( g_cart_id == 0 ) fprintf(stdout, "# [loop_gf_invert_contract] GF for dtau = %e\n", gf_dtau );
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

    	      /* TEST PLAQUETTE */
              saveGaugeQuda ( h_gauge, &gauge_param );
  	      gauge_field_qdp_to_cvc ( gauge_field_aux, h_gauge );
#ifdef HAVE_MPI
              xchange_gauge_field( gauge_field_aux );
#endif
	      plaquetteria  ( gauge_field_aux );
	      /* END TEST PLAQUETTE */

#elif defined _GFLOW_CVC
              flow_fwd_gauge_spinor_field ( gauge_field_gf, spinor_work[0], gf_niter, gf_dt, 1, 1, 0 );
#endif    

              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "loop_gf_invert_contract", "forward gradient flow", g_cart_id == 0 );

              /***************************************************************************
	       * flow the stochastic propagators
	       *
	       * update the gauge field
               ***************************************************************************/

              gettimeofday ( &ta, (struct timezone *)NULL );
              /* update resident gaugeFlowed */
#ifdef _GFLOW_QUDA
              _performGFlownStep ( spinor_work[1], spinor_work[1], &smear_param, 1 );

              /* TEST PLAQUETTE */
              saveGaugeQuda ( h_gauge, &gauge_param );
              gauge_field_qdp_to_cvc ( gauge_field_aux, h_gauge );
#ifdef HAVE_MPI
              xchange_gauge_field( gauge_field_aux );
#endif
              plaquetteria  ( gauge_field_aux );
              /* END TEST PLAQUETTE */

#elif defined _GFLOW_CVC
              flow_fwd_gauge_spinor_field ( gauge_field_gf, spinor_work[1], gf_niter, gf_dt, 1, 1, 1 );
#endif
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "loop_gf_invert_contract", "forward gradient flow", g_cart_id == 0 );

            } else {

              if ( g_cart_id == 0 ) fprintf(stdout, "# [loop_gf_invert_contract] no GF\n" );

	    }  /* end of if do any flow */

            /***************************************************************************
             * fill in loop matrix element ksc = (kspin, kcol ), isc = (ispin, icol )
             * as
             * loop[ksc][isc] = prop[ksc] * source[isc]^+
             *
             * with GF: everybody does the the following
             ***************************************************************************/

            gettimeofday ( &ta, (struct timezone *)NULL );

#pragma omp parallel for
            for ( size_t ix = 0; ix < VOLUME;   ix++ )
            {

              size_t const iy = _GSI(ix);

              for ( int ksc = 0; ksc < 12; ksc++ ) 
              {
                /* 
                 * times prop vector element
                 */
                double _Complex const a = spinor_work[1][ iy + 2 * ksc ] + I * spinor_work[1][ iy + 2 * ksc + 1 ];

                for ( int lsc = 0; lsc < 12; lsc++ ) 
                {
                 /* 
                  * complex conjugate of source vector element 
                  */
                  double _Complex const b = spinor_work[0][ iy + 2 * lsc ] - I * spinor_work[0][ iy + 2 * lsc + 1];

                  loop[igf][ix][ksc][lsc] += a * b;
                  }
                }
              }  /* end of loop on volume */

             gettimeofday ( &tb, (struct timezone *)NULL );
             show_time ( &ta, &tb, "loop_gf_invert_contract", "loop-matrix-accumulate", g_cart_id == 0 );

          }  /* end of loop on GF steps  */

#if defined _GFLOW_CVC
          flow_fwd_gauge_spinor_field ( NULL, NULL, 0, 0, 0, 0, 0 );
#endif


        }  /* end of loop on color dilution component */

      }  /* end of loop on spin dilution component */

#if _USE_TIME_DILUTION
    }  /* end of loop on timeslices */
#endif

    double gf_tau = 0;

    /* loop on GF steps */
    for ( int igf = 0; igf < gf_nstep; igf++ )
    {
      int const gf_niter   = gf_niter_list[igf];
      double const gf_dt   = gf_dt_list[igf];
      double const gf_dtau = gf_niter * gf_dt;
      gf_tau += gf_dtau;

      /***************************************************************************
       * write loop field to lime file
       ***************************************************************************/
      sprintf( filename, "loop.up.c%d.N%d.tau%6.4f.lime", Nconf, isample, gf_tau );
      char loop_type[2000];

      sprintf( loop_type, "<source_type>%d</source_type><noise_type>%d</noise_type><dilution_type>spin-color</dilution_type>", g_source_type, g_noise_type );

      exitstatus = write_lime_contraction( (double*)(loop[igf][0][0]), filename, 64, 144, loop_type, Nconf, 0);
      if ( exitstatus != 0  ) {
        fprintf ( stderr, "[loop_gf_invert_contract] Error write_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }

    }  /* end of loop on GF steps  */

    fini_4level_ztable ( &loop );

  }  /* end of loop on samples */

  fini_2level_dtable ( &spinor_work );

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/
  if ( scalar_field != NULL ) fini_2level_dtable ( &scalar_field );

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif

#ifdef _GFLOW_QUDA
  fini_2level_dtable ( &h_gauge );

  /* TEST PLAQUETTE */
  fini_1level_dtable ( &gauge_field_aux );

#elif defined _GFLOW_CVC
  fini_1level_dtable ( &gauge_field_gf );
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
  show_time ( &start_time, &end_time, "loop_gf_invert_contract", "runtime", g_cart_id == 0 );

  return(0);

}
