/***************************************************************************
 *
 * test_gauge_load
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
#include "smearing_techniques.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "gamma.h"
#include "clover.h"
#include "gradient_flow.h"
#include "gluon_operators.h"
#include "scalar_products.h"

using namespace cvc;


int get_gauge_padding ( int x[4] )
{
  int pad = 0;
#ifdef HAVE_MPI
  int volume = x[0] * x[1] * x[2] * x[3];
  int face_size[4];
  for ( int dir=0; dir<4; ++dir )
  {
    face_size[dir] = ( volume / x[dir] ) / 2;
  }
  pad = face_size[0];
  for ( int dir = 1; dir < 4; dir++ )
    if ( face_size[dir] > pad ) pad = face_size[dir];
#endif
  return ( pad );
}  /* end of get_gauge_padding */


/***************************************************************************
 *
 ***************************************************************************/
void gauge_field_cvc_to_qdp ( double ** g_qdp, double * g_cvc )
{

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  size_t const bytes = 18 * sizeof(double);
  
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( int x0=0; x0<T; x0++ )
  {
    for( int x1=0; x1<LX; x1++ )
    {
      for( int x2=0; x2<LY; x2++ )
      {
        for( int x3=0; x3<LZ; x3++ ) 
        {
          /* index running t,z,y,x */
          unsigned int const j = x1 + LX * ( x2 + LY * ( x3 + LZ * x0 ) );
          /* index running t, x, y, z */
          unsigned int const k = x3 + LZ * ( x2 + LY * ( x1 + LX * x0 ) );

          int b = (x0+x1+x2+x3) & 1;
          int qidx = 18 * ( b * VOLUME / 2 + j / 2 );

          memcpy( &(g_qdp[0][qidx]), &(g_cvc[_GGI(k,1)]), bytes );
          memcpy( &(g_qdp[1][qidx]), &(g_cvc[_GGI(k,2)]), bytes );
          memcpy( &(g_qdp[2][qidx]), &(g_cvc[_GGI(k,3)]), bytes );
          memcpy( &(g_qdp[3][qidx]), &(g_cvc[_GGI(k,0)]), bytes );

        }
      }
    }
  }
#ifdef HAVE_OPENMP
} /* end of parallel region */
#endif
  return;
}  /* end of gauge_field_cvc_to_qdp */

/***************************************************************************
 *
 ***************************************************************************/
void gauge_field_qdp_to_cvc ( double * g_cvc, double ** g_qdp )
{

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  size_t const bytes = 18 * sizeof(double);
  
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( int x0=0; x0<T; x0++ )
  {
    for( int x1=0; x1<LX; x1++ )
    {
      for( int x2=0; x2<LY; x2++ )
      {
        for( int x3=0; x3<LZ; x3++ ) 
        {
          unsigned int const j = x1 + LX * ( x2 + LY * ( x3 + LZ * x0 ) );
          unsigned int const k = x3 + LZ * ( x2 + LY * ( x1 + LX * x0 ) );

          int b = (x0+x1+x2+x3) & 1;
          int qidx = 18 * ( b * VOLUME / 2 + j / 2 );

          memcpy( &(g_cvc[_GGI(k,1)]), &(g_qdp[0][qidx]), bytes );
          memcpy( &(g_cvc[_GGI(k,2)]), &(g_qdp[1][qidx]), bytes );
          memcpy( &(g_cvc[_GGI(k,3)]), &(g_qdp[2][qidx]), bytes );
          memcpy( &(g_cvc[_GGI(k,0)]), &(g_qdp[3][qidx]), bytes );
        }
      }
    }
  }
#ifdef HAVE_OPENMP
} /* end of parallel region */
#endif
  return;
}  /* end of gauge_field_cvc_to_qdp */

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
  
  const char outfile_prefix[] = "test_gauge_load";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[400];
  struct timeval ta, tb, start_time, end_time;
  int check_propagator_residual = 0;
  int gf_niter = 1;
  double gf_dt = 0.01;


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
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

  fprintf(stdout, "# [test_gauge_load] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [test_gauge_load] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_gauge_load] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_gauge_load] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_gauge_load] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[test_gauge_load] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [test_gauge_load] reading gauge field from file %s\n", filename);

    exitstatus = read_lime_gauge_field_doubleprec(filename);

  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_gauge_load] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_gauge_load] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[test_gauge_load] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[test_gauge_load] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    fprintf(stderr, "[test_gauge_load] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[test_gauge_load] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [test_gauge_load] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;

  size_t const sizeof_spinor_field = _GSI ( VOLUME ) * sizeof ( double ) ;
  size_t const sizeof_gauge_field  = 72 * VOLUME * sizeof ( double );

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
    fprintf(stderr, "[test_gauge_load] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(12);
  }
  check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, lmzz[0], lmzzinv[0], 1 );

  // sprintf ( filename, "propagator.0" );
  // write_propagator ( spinor_work[1], filename, 0, 64 );

  fini_2level_dtable ( &spinor_work );
#if 0
#endif

  /***************************************************************************
   * up- and down-load gauge field
   ***************************************************************************/

  double * gauge_field_smeared = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
  if ( gauge_field_smeared == NULL )
  {
    fprintf(stderr, "[test_gauge_load] Error from init_1level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }

  double * gauge_field_smeared_cpu = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
  if ( gauge_field_smeared_cpu == NULL )
  {
    fprintf(stderr, "[test_gauge_load] Error from init_1level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }

  double ** h_gauge = init_2level_dtable ( 4, 18*VOLUME );
  if ( h_gauge == NULL )
  {
    fprintf(stderr, "[test_gauge_load] Error from init_2level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }

  double ** spinor_field = init_2level_dtable ( 4, _GSI( VOLUME ) );
  if ( spinor_field == NULL )
  {
    fprintf ( stderr, "[test_gauge_load] Error from init_2level_dtable    %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }


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


  /***************************************************************************
   ***************************************************************************/
#if 0
  /***************************************************************************
   * Begin of inv_param initialization
   ***************************************************************************/

    QudaInvertParam inv_param;
  /* typedef struct QudaInvertParam_s
   */ 
    inv_param.struct_size = sizeof ( QudaInvertParam );

    inv_param.input_location  = QUDA_CPU_FIELD_LOCATION;
    inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

    inv_param.dslash_type = QUDA_WILSON_DSLASH;
    inv_param.inv_type  = QUDA_CG_INVERTER;

    inv_param.mass = 0.; 
    inv_param.kappa = 0.;

    inv_param.m5 = 0.;
    inv_param.Ls = 0;

    /* inv_param.b_5[QUDA_MAX_DWF_LS]; */
    /* inv_param.c_5[QUDA_MAX_DWF_LS]; */

    inv_param.eofa_shift = 0.;
    inv_param.eofa_pm = 0.;
    inv_param.mq1 = 0.;
    inv_param.mq2 = 0.;
    inv_param.mq3 = 0.;

    inv_param.mu = 0.;
    inv_param.epsilon = 0.;

    inv_param.twist_flavor = QUDA_TWIST_NO;

    inv_param.laplace3D = -1; /**< omit this direction from laplace operator: x,y,z,t -> 0,1,2,3 (-1 is full 4D) */

    inv_param.tol = 0.;
    inv_param.tol_restart = 0.;
    inv_param.tol_hq = 0.;

    inv_param.compute_true_res = 0;
    inv_param.true_res = 0.;
    inv_param.true_res_hq = 0.;
    inv_param.maxiter = 0;
    inv_param.reliable_delta = 0.;
    inv_param.reliable_delta_refinement = 0.;
    inv_param.use_alternative_reliable = 0;
    inv_param.use_sloppy_partial_accumulator = 0;

    inv_param.solution_accumulator_pipeline = 0;

    inv_param.max_res_increase = 0;

    inv_param.max_res_increase_total = 0;

    inv_param.max_hq_res_increase = 0;

    inv_param.max_hq_res_restart_total = 0;

    inv_param.heavy_quark_check = 0;

    inv_param.pipeline = 0;

    inv_param.num_offset = 0;

    inv_param.num_src = 0;

    inv_param.num_src_per_sub_partition = 0;

    /* inv_param.split_grid[QUDA_MAX_DIM]; */

    inv_param.overlap = 0;

    /* inv_param.offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param.tol_offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param.tol_hq_offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param.true_res_offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param.iter_res_offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param.true_res_hq_offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param.residue[QUDA_MAX_MULTI_SHIFT]; */

    inv_param.compute_action = 0;

    /* inv_param.action[2] = {0., 0.}; */

    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.solve_type = QUDA_DIRECT_SOLVE;
    inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
    inv_param.dagger = QUDA_DAG_NO;
    inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;
    inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

    inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;

    inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec_refinement_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec_precondition = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;

    inv_param.dirac_order = QUDA_DIRAC_ORDER;

    inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    inv_param.clover_location = QUDA_CPU_FIELD_LOCATION;
    inv_param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param.clover_cuda_prec = QUDA_DOUBLE_PRECISION;
    inv_param.clover_cuda_prec_sloppy = QUDA_DOUBLE_PRECISION; 
    inv_param.clover_cuda_prec_refinement_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param.clover_cuda_prec_precondition = QUDA_DOUBLE_PRECISION;
    inv_param.clover_cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;

    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.use_init_guess = QUDA_USE_INIT_GUESS_NO; 

    inv_param.clover_csw =  0.;
    inv_param.clover_coeff = 0.;
    inv_param.clover_rho = 0.;

    inv_param.compute_clover_trlog = 0;
    /* inv_param.trlogA[2] = {0.,0.}; */

    inv_param.compute_clover = 0;
    inv_param.compute_clover_inverse = 0;
    inv_param.return_clover = 0;
    inv_param.return_clover_inverse = 0;

    inv_param.verbosity = QUDA_SILENT;  

    /* inv_param.sp_pad = 0; */
    inv_param.cl_pad = 0;

    inv_param.iter = 0; 
    inv_param.gflops = 0.;
    inv_param.secs = 0.;

    inv_param.tune = QUDA_TUNE_YES;

    inv_param.Nsteps = 0;

    inv_param.gcrNkrylov = 0;

    inv_param.inv_type_precondition = QUDA_CG_INVERTER;

    inv_param.preconditioner = NULL;

    inv_param.deflation_op = NULL;

    inv_param.eig_param = NULL;

    inv_param.deflate = QUDA_BOOLEAN_FALSE;

    inv_param.dslash_type_precondition = QUDA_WILSON_DSLASH;

    inv_param.verbosity_precondition = QUDA_SILENT;

    inv_param.tol_precondition = 0.;

    inv_param.maxiter_precondition = 0;

    inv_param.omega = 0.;

    /* inv_param.ca_basis = QUDA_CHEBYSHEV_BASIS; */

    inv_param.ca_lambda_min = 0.;

    inv_param.ca_lambda_max = 0.;

    inv_param.precondition_cycle = 0;

    /* inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ; */

    inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;

    inv_param.cuda_prec_ritz = QUDA_DOUBLE_PRECISION;
    
    inv_param.n_ev = 0;

    inv_param.max_search_dim = 0;

    inv_param.rhs_idx = 0;

    inv_param.deflation_grid = 0;

    inv_param.eigenval_tol = 0.;

    inv_param.eigcg_max_restarts = 0;

    inv_param.max_restart_num =0;

    inv_param.inc_tol = 0.;

    inv_param.make_resident_solution = false;

    inv_param.use_resident_solution = false;

    /* inv_param.chrono_make_resident; */

    /* inv_param.chrono_replace_last; */

    /* inv_param.chrono_use_resident; */

    /* inv_param.chrono_max_dim; */
    /* inv_param.chrono_index; */

    /* inv_param.chrono_precision;  */

    /* inv_param.extlib_type; */

    /* inv_param.native_blas_lapack = */

  /***************************************************************************
   * End of inv_param initialization
   ***************************************************************************/
#endif

#if 0
  sprintf ( filename, "gauge_field.0" );
  exitstatus = write_lime_contraction( gauge_field_with_phase, filename,  64, 36, "gauge", Nconf, 0 );
  if ( exitstatus != 0 )
  {
    fprintf ( stderr, "[test_gauge_load] Error from write_lime_contraction, status %d   %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(12);
  }
#endif
  /* for ( gf_dt = 0.1; gf_dt >= 0.00001; gf_dt /= 10. )
  { */

#ifdef _GFLOW_QUDA
    /***************************************************************************
     * upload gauge field
     ***************************************************************************/
  
    memcpy ( gauge_field_smeared, gauge_field_with_phase, sizeof_gauge_field );
    // unit_gauge_field ( gauge_field_smeared, VOLUME );

  
    /* reshape gauge field */
    gauge_field_cvc_to_qdp ( h_gauge, gauge_field_smeared );
  
  
    /* upload to device */
    loadGaugeQuda ( (void *)h_gauge, &gauge_param );
 
#endif   /* of #ifdef _GFLOW_QUDA  */

#if 0
    /* to really check, set both gauge fields to zero */
    memset ( h_gauge[0], 0, 4 * 18 * VOLUME * sizeof ( double ) );
  
    memset ( gauge_field_smeared, 0, sizeof_gauge_field );
  
    /* download from device */
    gauge_param.location = QUDA_CPU_FIELD_LOCATION;
  
    saveGaugeQuda ( (void*)h_gauge, &gauge_param );
  
    gauge_field_qdp_to_cvc ( gauge_field_smeared, h_gauge );
  
    
    sprintf ( filename, "gauge_field.1" );
    exitstatus = write_lime_contraction( gauge_field_smeared, filename,  64, 36, "gauge", Nconf, 0 );
    if ( exitstatus != 0 )
    {
      fprintf ( stderr, "[test_gauge_load] Error from write_lime_contraction, status %d   %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }
#endif

    prepare_volume_source ( spinor_field[0], VOLUME );
#if 0
#pragma omp parallel for
    for ( unsigned int ix = 0; ix < _GSI( VOLUME ); ix++ )
    {
      spinor_field[0][ix] = fabs ( spinor_field[0][ix] );
    }
#endif

#if 0
    int const gsx[4] = {
            g_source_coords_list[0][0],
            g_source_coords_list[0][1],
            g_source_coords_list[0][2],
            g_source_coords_list[0][3] };
    int sx[4], source_proc_id = -1;

    get_point_source_info ( gsx, sx, &source_proc_id );

    memset ( spinor_field[0], 0, sizeof_spinor_field );
    /* if ( g_cart_id == 0 ) spinor_field[0][_GSI(g_ipt[0][5][6][0])+12] = 1.; */
    /* if ( g_cart_id == 3 ) spinor_field[0][_GSI(g_ipt[T-1][5][6][LZ-1])+12] = 1.; */

    if ( g_cart_id == source_proc_id )
    {
      spinor_field[0][_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]])+12] = 1.;
    }
#endif
    /***************************************************************************
     * flow the gauge field and the spinor field on the gpu
     ***************************************************************************/
    
    memcpy ( spinor_field[1], spinor_field[0], sizeof_spinor_field ); 
    /* _performWuppertalnStep ( spinor_field[1], spinor_field[1], 1, 1. ); */

#ifdef _GFLOW_QUDA
 
    QudaGaugeSmearParam smear_param;
    smear_param.n_steps       = gf_niter;
    smear_param.epsilon       = gf_dt;
    smear_param.meas_interval = 1;
    smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;

    _performGFlownStep ( spinor_field[1], spinor_field[1], &smear_param, 1 );

    saveGaugeQuda ( (void*)h_gauge, &gauge_param );

    gauge_field_qdp_to_cvc ( gauge_field_smeared, h_gauge );

#if 0
    sprintf ( filename, "gauge_field.gpuflow" );
    exitstatus = write_lime_contraction( gauge_field_smeared, filename,  64, 36, "gauge", Nconf, 0 );
    if ( exitstatus != 0 )
    {
      fprintf ( stderr, "[test_gauge_load] Error from write_lime_contraction, status %d   %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }
#endif  /* of if 0 */

#endif  /* of if _GFLOW_QUDA */


    /***************************************************************************
     * now flow the gauge field and the spinor field the same way on the cpu
     ***************************************************************************/
#ifdef _GFLOW_CVC
    memcpy ( gauge_field_smeared_cpu, gauge_field_with_phase, sizeof_gauge_field );
    // unit_gauge_field ( gauge_field_smeared_cpu, VOLUME );

    memcpy ( spinor_field[2], spinor_field[0], sizeof_spinor_field );
 
    flow_fwd_gauge_spinor_field ( gauge_field_smeared_cpu, spinor_field[2], gf_niter, gf_dt, 1, 1, 1 );
 
#if 0 
    sprintf ( filename, "gauge_field.cpuflow" );
    exitstatus = write_lime_contraction( gauge_field_smeared_cpu, filename,  64, 36, "gauge", Nconf, 0 );
    if ( exitstatus != 0 )
    {
      fprintf ( stderr, "[test_gauge_load] Error from write_lime_contraction, status %d   %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }
#endif  /* if 0 */

#endif  /* of if _GFLOW_CVC */

    /***************************************************************************
     * || U_gpuflow -  U_cpuflow ||
     ***************************************************************************/

    double normdiff = 0.;
  
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    double nd = 0.;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for ( unsigned int ix = 0; ix < 72*VOLUME; ix++ )
    {
      double const dtmp = gauge_field_smeared_cpu[ix] - gauge_field_smeared[ix];
      nd += dtmp * dtmp;
    }
#ifdef HAVE_OPENMP
#pragma omp critical
{
#endif
  normdiff += nd;
#ifdef HAVE_OPENMP
}
#endif
}
  
#ifdef HAVE_MPI
    double pnormdiff = normdiff;
    if ( MPI_Allreduce ( &pnormdiff, &normdiff, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid ) != MPI_SUCCESS )
    {
      fprintf(stderr, "[test_gauge_load] Error from MPI_Allreduce   %s %d\n", __FILE__, __LINE__);
      EXIT(12);
    }
#endif
 
    normdiff = sqrt ( normdiff );
  
    /***************************************************************************
     * || f_gpuflow -  f_cpuflow ||
     ***************************************************************************/

    spinor_field_eq_spinor_field_mi_spinor_field ( spinor_field[3], spinor_field[1], spinor_field[2], VOLUME );

    double sfdiff = 0.;
    spinor_scalar_product_re ( &sfdiff, spinor_field[3], spinor_field[3], VOLUME );
    sfdiff = sqrt ( sfdiff );
    
    double sfnorm = 0.;
    spinor_scalar_product_re ( &sfnorm, spinor_field[2], spinor_field[2], VOLUME );
    sfnorm = sqrt ( sfnorm );
    
    double sfnorm2 = 0.;
    spinor_scalar_product_re ( &sfnorm2, spinor_field[1], spinor_field[1], VOLUME );
    sfnorm2 = sqrt ( sfnorm2 );

    if ( io_proc == 2 ) fprintf (stdout, "# [test_gauge_field] diff %2d %16.7e   gauge  %16.7e   spinor %16.7e   %16.7e   %16.7e   %s %d\n", 
        gf_niter, gf_dt, normdiff, sfdiff, sfnorm, sfnorm2, __FILE__, __LINE__ );

#if 0
    /* show original and flowed gauge fields in plain text for comparison */
    sprintf ( filename, "gauge-comp.proc%d", g_cart_id );
    FILE * gfs = fopen ( filename, "w" );
    for ( unsigned int ix = 0; ix < VOLUME; ix++ )
    {
      for ( int k = 0; k < 36; k++ )
      {
        fprintf ( gfs, "%25.16e %25.16e     %25.16e %25.16e    %25.16e %25.16e\n",
            gauge_field_with_phase[_GGI(ix,k/9)+2*(k%9)],
            gauge_field_with_phase[_GGI(ix,k/9)+2*(k%9)+1],
            gauge_field_smeared[_GGI(ix,k/9)+2*(k%9)],
            gauge_field_smeared[_GGI(ix,k/9)+2*(k%9)+1],
            gauge_field_smeared_cpu[_GGI(ix,k/9)+2*(k%9)],
            gauge_field_smeared_cpu[_GGI(ix,k/9)+2*(k%9)+1] );
      } 
    }
    fclose ( gfs );
#endif
     
    /* show original and flowed spinor fields in plain text for comparison */
    sprintf ( filename, "spinor.proc%d", g_cart_id );
    FILE * gfs = fopen ( filename, "w" );
    for ( unsigned int ix = 0; ix < VOLUME; ix++ )
    {
      fprintf ( gfs, "# %3d %3d %3d %3d\n",
          g_lexic2coords[ix][0]+g_proc_coords[0]*T,
          g_lexic2coords[ix][1]+g_proc_coords[1]*LX,
          g_lexic2coords[ix][2]+g_proc_coords[2]*LY,
          g_lexic2coords[ix][3]+g_proc_coords[3]*LZ );
      for ( int k = 0; k < 12; k++ )
      {
        fprintf ( gfs, "%2d %2d %25.16e %25.16e     %25.16e %25.16e    %25.16e %25.16e\n",
            k/3, k%3,
            spinor_field[0][_GSI(ix)+2*k], spinor_field[0][_GSI(ix)+2*k+1],
            spinor_field[1][_GSI(ix)+2*k], spinor_field[1][_GSI(ix)+2*k+1],
            spinor_field[2][_GSI(ix)+2*k], spinor_field[2][_GSI(ix)+2*k+1] );
      }
    }
    fclose ( gfs );

  /* } */  /* end of loop on gf_dt */

  fini_2level_dtable ( &h_gauge );
  fini_1level_dtable ( &gauge_field_smeared );
  fini_1level_dtable ( &gauge_field_smeared_cpu );
  fini_2level_dtable ( &spinor_field );


#if 0
  /***************************************************************************
   * another test solve, just to have original gauge field up on device
   ***************************************************************************/
  spinor_work = init_2level_dtable ( 2, _GSI( VOLUME + RAND ) );
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
    fprintf(stderr, "[test_gauge_load] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(12);
  }
  check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, lmzz[0], lmzzinv[0], 1 );

  sprintf ( filename, "propagator.1" );
  write_propagator ( spinor_work[1], filename, 0, 64 );

  fini_2level_dtable ( &spinor_work );
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
  show_time ( &start_time, &end_time, "test_gauge_load", "runtime", g_cart_id == 0 );

  return(0);

}
