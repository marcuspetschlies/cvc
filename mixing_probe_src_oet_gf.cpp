/***************************************************************************
 *
 * mixing_probe_src_oet_gf
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
#include "ranlxd.h"
#include "clover.h"
#include "gradient_flow.h"
#include "contract_cvc_tensor.h"

#include "pm.h"
#include "gauge_quda.h"

#define MAX_NUM_GF_NSTEP 100

using namespace cvc;

/***************************************************************************
 * B diagram gamma L gamma
 ***************************************************************************/
inline void  b_glg ( double _Complex *** const lout, double _Complex *** const lin, gamma_matrix_type * g )
{
  memset ( lout[0][0], 0, 144 * VOLUME * sizeof(double _Complex) );
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int ix = 0; ix < VOLUME; ix++ )
  {
    double _Complex ** const _lin  = lin[ix];
    double _Complex ** const _lout = lout[ix];

    for ( int ia = 0; ia < 4; ia++ )
    {
      for ( int ir = 0; ir < 3; ir++ )
      {
        for ( int ib = 0; ib < 4; ib++ )
        {
          for ( int is = 0; is < 3; is++ )
          {
            double _Complex w = 0.;

            for ( int ic = 0; ic < 4; ic++ )
            {
              for ( int id = 0; id < 4; id++ )
              {

                for( int mu = 0; mu < 4; mu++ )
                {
                  w += g[mu].m[ia][ic] * _lin[3*ic+ir][3*id+is] * g[mu].m[id][ib];
                }
              }
            }

            _lout[3*ia+ir][3*ib+is] += w;
          }
        }
      }
    }
  }
  return;
}  /* b_glg */

/***************************************************************************
 * D diagram gamma L gamma
 ***************************************************************************/
inline void  d_glg ( double _Complex *** const lout, double _Complex *** const lin, gamma_matrix_type * g )
{
  memset ( lout[0][0], 0, 144 * VOLUME * sizeof(double _Complex) );
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int ix = 0; ix < VOLUME; ix++ )
  {
    double _Complex ** const _lin  = lin[ix];
    double _Complex ** const _lout = lout[ix];

    for ( int ia = 0; ia < 4; ia++ )
    {
      for ( int ir = 0; ir < 3; ir++ )
      {
        for ( int ib = 0; ib < 4; ib++ )
        {
          for ( int is = 0; is < 3; is++ )
          {
            double _Complex w = 0.;

            for ( int ic = 0; ic < 4; ic++ )
            {
              for ( int id = 0; id < 4; id++ )
              {
                for( int mu = 0; mu < 4; mu++ )
                {
                  w += g[mu].m[ia][ib] * _lin[3*ic+ir][3*id*is] * g[mu].m[id][ic];
                }
              }
            }

            _lout[ia][ib] += w;

          }
        }
      }
    }
  }
  return;
}  /* end of d_glg */


/***********************************************************
 * qb loop x prop
 ***********************************************************/
inline void qb_glg_prop ( double ** const pout, double ** const pin, double _Complex *** lin , int const n )
{
  for ( int isc = 0; isc < n; isc++ )
  {
#pragma omp parallel for
    for ( unsigned int ix = 0; ix < VOLUME; ix++ )
    {
      double _Complex ** const _lin = lin[ix];
            
      double * const _pin  = pin[isc] + _GSI(ix);
      double * const _pout = pout[isc] + _GSI(ix);

      for ( int ia = 0; ia < 12; ia++ )
      {
        double _Complex w = 0.;

        for ( int ic = 0; ic < 12; ic++ )
        {
          w += _lin[ia][ic] * (  _pin[2*ic+0] + _pin[2*ic+1] * I );
        }

        _pout[2*ia  ] = creal ( w );
        _pout[2*ia+1] = cimag ( w );
      }
    }
  }

  return;

}  /* end of b_glg_prop */

/***********************************************************
 * cc loop x prop
 ***********************************************************/
inline void cc_glg_prop ( double ** const pout, double ** const pin, double _Complex *** lin , int const n )
{
  for ( int isc = 0; isc < n ; isc++ )
  {
#pragma omp parallel for
    for ( unsigned int ix = 0; ix < VOLUME; ix++ )
    {
      double _Complex ** const _lin = lin[ix];

      double * const _pin  = pin[isc] + _GSI(ix);
      double * const _pout = pout[isc] + _GSI(ix);

      for ( int ia = 0; ia < 4; ia++ )
      {
        for ( int ir = 0; ir < 3; ir++ )
        {
          double _Complex w = 0.;

          for ( int ib = 0; ib < 4; ib++ )
          {
            for ( int is = 0; is < 3; is++ )
            {
              w += _lin[3*ia+is][3*ib+is] * (  _pin[2*(3*ib+ir)] + _pin[2*(3*ib+ir)+1] * I );
            }
          }

          _pout[2*(3*ia+ir)  ] = creal ( w );
          _pout[2*(3*ia+ir)+1] = cimag ( w );
        }
      }
    }
  }

  return;
}  /* end of cc_glg_prop */


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
  
  const char outfile_prefix[] = "mx_prb_src_oet";

  const char flavor_tag[4] = { 'u', 'd', 's', 'c' };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400], output_filename[400];
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  struct timeval ta, tb, start_time, end_time;
  int spin_dilution = 4;
  int color_dilution = 1;

  char gamma_tag[2][3] = { "vv", "aa" };

  int gf_nstep = 0;
  int gf_niter_list[MAX_NUM_GF_NSTEP];
  double gf_dt_list[MAX_NUM_GF_NSTEP];


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "srch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'r':
      check_propagator_residual = 1;
      break;
    case 's':
      spin_dilution = atoi ( optarg );
      break;
    case 'c':
      color_dilution = atoi ( optarg );
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

  fprintf(stdout, "# [mixing_probe_src_oet_gf] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [mixing_probe_src_oet_gf] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [mixing_probe_src_oet_gf] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [mixing_probe_src_oet_gf] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[mixing_probe_src_oet_gf] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[mixing_probe_src_oet_gf] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [mixing_probe_src_oet_gf] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [mixing_probe_src_oet_gf] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[mixing_probe_src_oet_gf] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[mixing_probe_src_oet_gf] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[mixing_probe_src_oet_gf] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    fprintf(stderr, "[mixing_probe_src_oet_gf] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[mixing_probe_src_oet_gf] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [mixing_probe_src_oet_gf] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

#if _TEST
  FILE *ofs = fopen ( "mx_prb_src.test", "w" );
#endif

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  int * rng_state = NULL;
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[mixing_probe_src_oet_gf] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   * set the gamma matrices
   ***************************************************************************/

  gamma_matrix_type gamma_list[2][4];

  gamma_matrix_set( &( gamma_list[0][0]), 0, 1. );
  gamma_matrix_set( &( gamma_list[0][1]), 1, 1. );
  gamma_matrix_set( &( gamma_list[0][2]), 2, 1. );
  gamma_matrix_set( &( gamma_list[0][3]), 3, 1. );

  gamma_matrix_set( &( gamma_list[1][0]), 6, 1. );
  gamma_matrix_set( &( gamma_list[1][1]), 7, 1. );
  gamma_matrix_set( &( gamma_list[1][2]), 8, 1. );
  gamma_matrix_set( &( gamma_list[1][3]), 9, 1. );

  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  /***************************************************************************
   * Part Ia
   ***************************************************************************/
  double ** spinor_work = init_2level_dtable ( 3, _GSI( VOLUME+RAND ) );
  if ( spinor_work == NULL ) {
    fprintf ( stderr, "[mixing_probe_src_oet_gf] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

#ifdef _GFLOW_QUDA
    /***************************************************************************
     * dummy solve, just to have original gaugePrecise field up on device,
     ***************************************************************************/
  memset(spinor_work[1], 0, sizeof_spinor_field);
  memset(spinor_work[0], 0, sizeof_spinor_field);
  if ( g_cart_id == 0 ) spinor_work[0][0] = 1.;
  exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], 0);
#  if ( defined GPU_DIRECT_SOLVER )
  if(exitstatus < 0)
#  else
  if(exitstatus != 0)
#  endif
  {
    fprintf(stderr, "[mixing_probe_src_oet_gf] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(12);
  }
#endif  /* of if _GFLOW_QUDA */

  /***************************************************************************
   *
   * 
   *
   ***************************************************************************/

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

#elif defined _GFLOW_CVC
  double * gauge_field_gf = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
  if ( gauge_field_gf == NULL )
  {
    fprintf(stderr, "[loop_gf_invert_contract] Error from init_1level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }
#endif

  /***************************************************************************
   * TEST choice of gf iterations and discretization
   *
   ***************************************************************************/
  gf_nstep = 3;
  gf_niter_list[0] = 0;
  gf_niter_list[1] = 3;
  gf_niter_list[2] = 3;
  gf_dt_list[0] = 0.01;
  gf_dt_list[1] = 0.01;
  gf_dt_list[2] = 0.01;



  /***************************************************************************
   * loop on samples
   *
   ***************************************************************************/
  for ( int isample = g_sourceid; isample <= g_sourceid2; isample += g_sourceid_step )
  {

    /***************************************************************************
     * random source timeslice
     ***************************************************************************/
    double dts;
    ranlxd ( &dts , 1 );
    int gts = (int)(dts * T_global);

#ifdef HAVE_MPI
    if (  MPI_Bcast( &gts, 1, MPI_INT, 0, g_cart_grid ) != MPI_SUCCESS ) {
      fprintf ( stderr, "[mixing_probe_src_oet_gf] Error from MPI_Bcast %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
#endif

    /***************************************************************************
     * local source timeslice and source process ids
     ***************************************************************************/

    int source_timeslice = -1;
    int source_proc_id   = -1;

    exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[mixing_probe_src_oet_gf] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     * one data file per source timeslice
     ***************************************************************************/
    if(io_proc == 2) 
    {
#ifdef HAVE_HDF5
    /***************************************************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     ***************************************************************************/
      sprintf ( output_filename, "%s.%d.t%d.s%d.h5", outfile_prefix, Nconf, gts, isample );
      fprintf(stdout, "# [mixing_probe_src_oet_gf] writing data to file %s\n", output_filename);
#else
      fprintf(stderr, "[mixing_probe_src_oet_gf] Error, no outupt variant selected %s %d\n",  __FILE__, __LINE__);
      EXIT(15);
#endif
    }  /* end of if io_proc == 2 */

    int const spin_color_dilution = spin_dilution * color_dilution;

    /* up and down quark propagators */
    double *** propagator = init_3level_dtable ( 2, spin_color_dilution, _GSI( VOLUME ) );
    if( propagator == NULL ) {
      fprintf(stderr, "[mixing_probe_src_oet_gf] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /* flowed propagators
     */
    double *** propagator_gf = init_3level_dtable ( 2, spin_color_dilution, _GSI( VOLUME ) );
    if ( propagator_gf == NULL ) {
      fprintf(stderr, "[mixing_probe_src_oet_gf] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    double *** b_propagator_gf = init_3level_dtable ( 2, spin_color_dilution, _GSI( VOLUME ) );
    if ( b_propagator_gf == NULL ) {
      fprintf(stderr, "[mixing_probe_src_oet_gf] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /* stochastic timeslice source */
    double ** stochastic_source = init_2level_dtable ( spin_color_dilution, _GSI( VOLUME ) );
    if( stochastic_source == NULL ) {
      fprintf(stderr, "[mixing_probe_src_oet_gf] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     *
     ***************************************************************************/
    if( (exitstatus = init_timeslice_source_oet ( stochastic_source, gts, NULL, spin_dilution, color_dilution, 1 ) ) != 0 ) 
    {
      fprintf(stderr, "[mixing_probe_src_oet_gf] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(64);
    }

    for ( int iflavor = 0; iflavor < 2; iflavor++ ) 
    {

      for( int i = 0; i < spin_color_dilution; i++) 
      {

        gettimeofday ( &ta, (struct timezone *)NULL );

        /***********************************************************
         * flavor-type stochastic timeslice-to-all propagator
         *
         ***********************************************************/
    
        memcpy ( spinor_work[0], stochastic_source[i], sizeof_spinor_field );

        memset ( spinor_work[1], 0, sizeof_spinor_field );

        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
        if(exitstatus < 0) {
          fprintf(stderr, "[mixing_probe_src_oet_gf] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, lmzz[iflavor], lmzzinv[iflavor], 1 );
        }

         memcpy( propagator[iflavor][i], spinor_work[1], sizeof_spinor_field );

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "mixing_probe_src_oet_gf", "forward-light-invert-check", g_cart_id == 0 );
      }

    }  /* end of loop on flavor */

    /***************************************************************************
     ***************************************************************************
     **
     ** contractions
     **
     ***************************************************************************
     ***************************************************************************/

    /***************************************************************************
     *
     ***************************************************************************/
    double * contr = init_1level_dtable ( 2 * T );
    if ( contr == NULL ) {
      fprintf(stderr, "[mixing_probe_src_oet_gf] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(3);
    }

    double _Complex *** loop = init_3level_ztable ( VOLUME, 12, 12 );
    if ( loop  == NULL ) {
      fprintf ( stderr, "[mixing_probe_src_oet_gf] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
        
    double _Complex *** gloopg = init_3level_ztable ( VOLUME, 12, 12 );
    if ( gloopg  == NULL ) {
      fprintf ( stderr, "[mixing_probe_src_oet_gf] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }

    double gf_tau = 0.;
 
    /***********************************************************
     * initially copy all propagator fields
     ***********************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ )
    {
      for ( int isc = 0; isc < spin_color_dilution; isc++ )
      {
        memcpy ( propagator_gf[iflavor][isc], propagator[iflavor][isc], sizeof_spinor_field );
      }
    }

    /***********************************************************
     * loop on gradient flow steps
     ***********************************************************/
    for ( int igf = 0; igf < gf_nstep; igf++ ) 
    {
      int const gf_niter = gf_niter_list[igf];
      double const gf_dt = gf_dt_list[igf];
      double const gf_dtau = gf_niter * gf_dt;
      gf_tau += gf_dtau;

      sprintf ( filename, "loop.up.c%d.N%d.tau%6.4f.lime", Nconf, g_nsample, gf_tau );
        
      exitstatus = read_lime_contraction ( (double*)(loop[0][0]), filename, 144, 0 );
      if ( exitstatus != 0  ) {
        fprintf ( stderr, "[loop_gf_invert_contract] Error read_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }

      int pf[3] = {0,0,0};
      char tag[100];
  
      /***********************************************************
       * Gradient flow parameters
       ***********************************************************/
#ifdef _GFLOW_QUDA
      QudaGaugeSmearParam smear_param;
      smear_param.n_steps       = gf_niter;
      smear_param.epsilon       = gf_dt;
      smear_param.meas_interval = 1;
      smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;
#endif

      for ( int iflavor = 0; iflavor < 2; iflavor++ )
      {
        for ( int isc = 0; isc < spin_color_dilution; isc++ )
        {
          int const update_gauge = ( iflavor == 1 ) && (isc == spin_color_dilution - 1 );
#ifdef _GFLOW_QUDA
          _performGFlownStep ( propagator_gf[iflavor][isc], propagator_gf[iflavor][isc], &smear_param, update_gauge );
#elif defined _GFLOW_CVC
          flow_fwd_gauge_spinor_field ( gauge_field_gf, propagator_gf[iflavor][isc], gf_niter, gf_dt, 1, 1, update_gauge );
#endif
        }
      }

      for ( int iflavor = 0; iflavor < 1; iflavor++ )
      {

        /***********************************************************/
        /***********************************************************/

        for ( int igamma = 0; igamma < 2; igamma++ )
        {

          /***********************************************************
           ***********************************************************
           **
           ** B diagrams: build gamma_mu L gamma_mu
           **
           ***********************************************************
           ***********************************************************/
          b_glg ( gloopg, loop, gamma_list[igamma] );
  
          /***********************************************************
           * qb loop x prop
           ***********************************************************/
          qb_glg_prop ( b_propagator_gf[iflavor], propagator_gf[iflavor], gloopg, spin_color_dilution );
  
          /***********************************************************
           * contract with g5 at source
           ***********************************************************/
          sprintf ( tag, "/qb/f%d-%s-f%d-5/b", iflavor, gamma_tag[igamma], iflavor );
          contract_twopoint ( contr, 4, 5, propagator_gf[1-iflavor], b_propagator_gf[iflavor], spin_dilution, color_dilution );
  
          exitstatus = contract_write_to_h5_file ( &contr, output_filename, tag, &pf, 1, io_proc );
          if(exitstatus != 0) {
            fprintf(stderr, "[mixing_probe_src_oet_gf] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }
  
          /***********************************************************
           * contract with id at source
           ***********************************************************/
          sprintf ( tag, "/qb/f%d-%s-f%d-4/b", iflavor, gamma_tag[igamma], iflavor );
          contract_twopoint ( contr, 4, 4, propagator_gf[1-iflavor], b_propagator_gf[iflavor], spin_dilution, color_dilution );
  
          exitstatus = contract_write_to_h5_file ( &contr, output_filename, tag, &pf, 1, io_proc );
          if(exitstatus != 0) {
            fprintf(stderr, "[mixing_probe_src_oet_gf] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }
  
          /***********************************************************
           * cc B diagram loop x prop
           ***********************************************************/
          cc_glg_prop ( b_propagator_gf[iflavor], propagator_gf[iflavor], gloopg , spin_color_dilution );
  
          /***********************************************************
           * contract with g5 at source
           ***********************************************************/
          sprintf ( tag, "/cc/f%d-%s-f%d-5/b", iflavor, gamma_tag[igamma], iflavor );
          contract_twopoint ( contr, 4, 5, propagator_gf[1-iflavor], b_propagator_gf[iflavor], spin_dilution, color_dilution );
  
          exitstatus = contract_write_to_h5_file ( &contr, output_filename, tag, &pf, 1, io_proc );
          if(exitstatus != 0) {
            fprintf(stderr, "[mixing_probe_src_oet_gf] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }
  
          /***********************************************************
           * contract with id at source
           ***********************************************************/
          sprintf ( tag, "/cc/f%d-%s-f%d-4/b", iflavor, gamma_tag[igamma], iflavor );
          contract_twopoint ( contr, 4, 4, propagator_gf[1-iflavor], b_propagator_gf[iflavor], spin_dilution, color_dilution );
  
          exitstatus = contract_write_to_h5_file ( &contr, output_filename, tag, &pf, 1, io_proc );
          if(exitstatus != 0) {
            fprintf(stderr, "[mixing_probe_src_oet_gf] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }
  
          /***********************************************************/
          /***********************************************************/
  
          /***********************************************************
           ***********************************************************
           **
           ** D diagrams: build tr_spin [ gamma_mu L ] gamma_mu
           **
           ***********************************************************
           ***********************************************************/
  
          d_glg ( gloopg, loop, gamma_list[igamma] );
  
          /***********************************************************
           * qb D diagram loop x prop 
           ***********************************************************/
          qb_glg_prop ( b_propagator_gf[iflavor], propagator_gf[iflavor], gloopg, spin_color_dilution );
  
          /***********************************************************
           * contract with g5 at source
           ***********************************************************/
          sprintf ( tag, "/qb/f%d-%s-f%d-5/d", iflavor, gamma_tag[igamma], iflavor );
          contract_twopoint ( contr, 4, 5, propagator_gf[1-iflavor], b_propagator_gf[iflavor], spin_dilution, color_dilution );
  
          exitstatus = contract_write_to_h5_file ( &contr, output_filename, tag, &pf, 1, io_proc );
          if(exitstatus != 0) {
            fprintf(stderr, "[mixing_probe_src_oet_gf] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }
  
          /***********************************************************
           * contract with id at source
           ***********************************************************/
          sprintf ( tag, "/qb/f%d-%s-f%d-4/d", iflavor, gamma_tag[igamma], iflavor );
          contract_twopoint ( contr, 4, 4, propagator_gf[1-iflavor], b_propagator_gf[iflavor], spin_dilution, color_dilution );
  
          exitstatus = contract_write_to_h5_file ( &contr, output_filename, tag, &pf, 1, io_proc );
          if(exitstatus != 0) {
            fprintf(stderr, "[mixing_probe_src_oet_gf] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }
  
          /***********************************************************
           * cc D diagram loop x prop
           ***********************************************************/
          cc_glg_prop ( b_propagator_gf[iflavor], propagator_gf[iflavor], gloopg, spin_color_dilution );
  
          /***********************************************************
           * contract with g5 at source
           ***********************************************************/
          sprintf ( tag, "/cc/f%d-%s-f%d-5/d", iflavor, gamma_tag[igamma], iflavor );
          contract_twopoint ( contr, 4, 5, propagator_gf[1-iflavor], b_propagator_gf[iflavor], spin_dilution, color_dilution );
  
          exitstatus = contract_write_to_h5_file ( &contr, output_filename, tag, &pf, 1, io_proc );
          if(exitstatus != 0) {
            fprintf(stderr, "[mixing_probe_src_oet_gf] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }
        
          /***********************************************************
           * contract with id at source
           ***********************************************************/
          sprintf ( tag, "/cc/f%d-%s-f%d-4/d", iflavor, gamma_tag[igamma], iflavor );
          contract_twopoint ( contr, 4, 4, propagator_gf[1-iflavor], b_propagator_gf[iflavor], spin_dilution, color_dilution );
  
          exitstatus = contract_write_to_h5_file ( &contr, output_filename, tag, &pf, 1, io_proc );
          if(exitstatus != 0) {
            fprintf(stderr, "[mixing_probe_src_oet_gf] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }
        
        }  /* end of loop on gamma */

      }  /* end of loop on flavor */

    }  /* end of loop on gf steps */

    fini_3level_ztable ( &loop );
    fini_3level_ztable ( &gloopg );
    fini_1level_dtable ( &contr );

    /***************************************************************************
     * free propagator fields
     ***************************************************************************/
    fini_2level_dtable ( &stochastic_source );
    fini_3level_dtable ( &propagator );
    fini_3level_dtable ( &propagator_gf );
    fini_3level_dtable ( &b_propagator_gf );

    exitstatus = init_timeslice_source_oet ( NULL, -1, NULL, 0, 0, -2 );

  }  /* end of loop on samples */

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
  show_time ( &start_time, &end_time, "mixing_probe_src_oet_gf", "runtime", g_cart_id == 0 );

  return(0);

}
