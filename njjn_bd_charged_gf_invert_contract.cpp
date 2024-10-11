/***************************************************************************
 *
 * njjn_bd_charged_gf_invert_contract
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
#include "contract_factorized.h"
#include "contract_diagrams.h"
#include "gamma.h"
#include "clover.h"
#include "fermion_quda.h"
#include "gauge_quda.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#define _PART_PROP   1
#define _PART_TWOP   1  /* N1, N2 */
#define _PART_THREEP 1  /* B/Z and D1c/i sequential diagrams */

#ifndef _TEST_TIMER
#  define _TEST_TIMER
#endif

using namespace cvc;

/* typedef int ( * reduction_operation ) (double**, double*, fermion_propagator_type*, unsigned int ); */

typedef int ( * reduction_operation ) (double**, fermion_propagator_type*, fermion_propagator_type*, fermion_propagator_type*, unsigned int);


/***************************************************************************
 * 
 ***************************************************************************/
static inline int reduce_project_write ( double ** vx, double *** vp, fermion_propagator_type * fa, fermion_propagator_type * fb, fermion_propagator_type * fc, reduction_operation reduce,
    void * writer, char * tag, int (*momentum_list)[3], int momentum_number, int const nd, unsigned int const N, int const io_proc ) {

  int exitstatus;

  /* contraction */
  exitstatus = reduce ( vx, fa, fb, fc, N );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return( 1 );
  }

  /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
  exitstatus = contract_vn_momentum_projection ( vp, vx, nd, momentum_list, momentum_number);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return( 2 );
  }

  if ( g_cart_id == 0 && g_verbose > 2 )
  {
    fprintf (stdout, "# [reduce_project_write] write tag %s    %s %d\n", tag, __FILE__, __LINE__ );
  }

#if defined HAVE_HDF5
  exitstatus = contract_vn_write_h5 ( vp, nd, (char *)writer, tag, momentum_list, momentum_number, io_proc );
#elif defined HAVE_LHPC_AFF
  /* write to AFF file */
  exitstatus = contract_vn_write_aff ( vp, nd, (struct AffWriter_s*)writer, tag, momentum_list, momentum_number, io_proc );
#endif
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_write for tag %s, status was %d %s %d\n", tag, exitstatus, __FILE__, __LINE__ );
    return( 3 );
  }


  return ( 0 );

}  /* end of reduce_project_write */



/***************************************************************************
 * helper message
 ***************************************************************************/
void usage() {
  fprintf(stdout, "Code for FHT-type nucleon-nucleon 2-pt function inversion and contractions\n");
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
  
  const char outfile_prefix[] = "njjn_bd_charged_gf";

  const char flavor_tag[4] = { 'u', 'd', 's', 'c' };

  const int sequential_gamma_sets = 1;
  int const sequential_gamma_num[4] = {4, 4, 1, 1};

  char const sequential_gamma_tag[4][3] = { "vv", "aa", "ss", "pp" };

  char const gamma_id_to_Cg_ascii[16][10] = {
    "Cgy",
    "Cgzg5",
    "Cgt",
    "Cgxg5",
    "Cgygt",
    "Cgyg5gt",
    "Cgyg5",
    "Cgz",
    "Cg5gt",
    "Cgx",
    "Cgzg5gt",
    "C",
    "Cgxg5gt",
    "Cgxgt",
    "Cg5",
    "Cgzgt"
  };


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  struct timeval ta, tb, start_time, end_time;
  int first_solve_dummy = 1;

  /* for gradient flow */
  int gf_niter = 2;
  int gf_ns = 1;
  double gf_dt = 0.01;
  double gf_tau = 0.;
  int gf_nb;

  int const    gamma_f1_number                           = 1;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 };

#if ( defined HAVE_LHPC_AFF ) &&  ! (defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
#endif

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

  fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[njjn_bd_charged_gf_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [njjn_bd_charged_gf_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[njjn_bd_charged_gf_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * set the gamma matrices
   ***************************************************************************/
  gamma_matrix_type sequential_gamma_list[4][4];
  /* vector */
  gamma_matrix_set( &( sequential_gamma_list[0][0] ), 0, 1. );  /*  gamma_0 = gamma_t */
  gamma_matrix_set( &( sequential_gamma_list[0][1] ), 1, 1. );  /*  gamma_1 = gamma_x */
  gamma_matrix_set( &( sequential_gamma_list[0][2] ), 2, 1. );  /*  gamma_2 = gamma_y */
  gamma_matrix_set( &( sequential_gamma_list[0][3] ), 3, 1. );  /*  gamma_3 = gamma_z */
  /* pseudovector */
  gamma_matrix_set( &( sequential_gamma_list[1][0] ), 6, 1. );  /*  gamma_6 = gamma_5 gamma_t */
  gamma_matrix_set( &( sequential_gamma_list[1][1] ), 7, 1. );  /*  gamma_7 = gamma_5 gamma_x */
  gamma_matrix_set( &( sequential_gamma_list[1][2] ), 8, 1. );  /*  gamma_8 = gamma_5 gamma_y */
  gamma_matrix_set( &( sequential_gamma_list[1][3] ), 9, 1. );  /*  gamma_9 = gamma_5 gamma_z */
  /* scalar */
  gamma_matrix_set( &( sequential_gamma_list[2][0] ), 4, 1. );  /*  gamma_4 = id */
  /* pseudoscalar */
  gamma_matrix_set( &( sequential_gamma_list[3][0] ), 5, 1. );  /*  gamma_5 */

  gamma_matrix_type gammafive;
  gamma_matrix_set( &gammafive, 5, 1. );  /*  gamma_5 */


  /***************************************************************************
   * constants
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   * field to read in loops
   ***************************************************************************/
  double _Complex *** loop = NULL;
  loop = init_3level_ztable ( VOLUME, 12, 12 );
  if ( loop  == NULL ) {
    fprintf ( stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

#ifdef _GFLOW_QUDA

  /***************************************************************************
   * gauge_param initialization
   ***************************************************************************/
  double ** h_gauge = init_2level_dtable ( 4, 18*VOLUME );
  if ( h_gauge == NULL )
  {
    fprintf(stderr, "[test_adjoint_flow] Error from init_2level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }

  QudaGaugeParam gauge_param;
  init_gauge_param ( &gauge_param );

  /***************************************************************************
   * prepare upload of gauge field for gradient flow
   ***************************************************************************/
  /* reshape gauge field */
  gauge_field_cvc_to_qdp ( h_gauge, gauge_field_with_phase );

  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  /***************************************************************************
   * either we upload QUDA_WILSON_LINKS explicitly here, or we make
   * a dummy solve to have wrapper upload the original gauge field
   * and have quda instantiate gaugePrecise, which we need in
   * adjoint gradient flow ahead of inversion
   *
   * here we make a dummy solve
   ***************************************************************************/
  if ( first_solve_dummy == 0 )
  {
    // set to Wilson links for this one upload
    gauge_param.type = QUDA_WILSON_LINKS;
    loadGaugeQuda ( (void *)h_gauge, &gauge_param );

  } else if ( first_solve_dummy == 1 )
  {
    double ** spinor_work = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
    if ( spinor_work == NULL ) 
    {
      fprintf ( stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
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
      fprintf(stderr, "[njjn_bd_charged_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(12);
    }
    fini_2level_dtable ( &spinor_work );

  }  // end of if first_solve_dummy

  /***************************************************************************
   * set to flowed links from now on, upload gauge field
   ***************************************************************************/
  gauge_param.type = QUDA_FLOWED_LINKS;
#ifdef _TEST_TIMER
  gettimeofday ( &ta, (struct timezone *)NULL );
#endif
  loadGaugeQuda ( (void *)h_gauge, &gauge_param );
#ifdef _TEST_TIMER
  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "loadGaugeQuda", g_cart_id == 0 );
#endif


  /***************************************************************************
   * set gradient flow parameters
   ***************************************************************************/
  QudaGaugeSmearParam smear_param;
  smear_param.n_steps       = gf_niter;
  smear_param.epsilon       = gf_dt;
  smear_param.meas_interval = 1;
  smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;
  smear_param.restart       = QUDA_BOOLEAN_FALSE;

  gf_tau = gf_niter * gf_dt;

  gf_nb = (int) ceil ( pow ( (double)gf_niter , 1./ ( (double)gf_ns + 1. ) ) );
  if ( io_proc == 2 && g_verbose > 1 )
  {
    fprintf (stdout, "# [njjn_bd_charged_gf_invert_contract] gf_nb = %d   %s %d\n", gf_nb, __FILE__, __LINE__ );
  }

#endif  // of if _GFLOW_QUDA

  /***************************************************************************
   * loop on source locations
   *
   *   each source location is given by 4-coordinates in
   *   global variable
   *   g_source_coords_list[count][0-3] for t,x,y,z
   ***************************************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) 
  {
    /***************************************************************************
     * determine source coordinates,
     * find out, if source_location is in this process
     ***************************************************************************/

    int const gsx[4] = {
        ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global,
        ( g_source_coords_list[isource_location][1] + LX_global ) % LX_global,
        ( g_source_coords_list[isource_location][2] + LY_global ) % LY_global,
        ( g_source_coords_list[isource_location][3] + LZ_global ) % LZ_global };

    int sx[4], source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     * open output file reader
     * we use the AFF format here
     * https://github.com/usqcd-software/aff
     *
     * one data file per source position
     ***************************************************************************/
    void * writer = (void *)NULL;
    if(io_proc == 2) 
    {
#if defined HAVE_HDF5
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.h5", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [njjn_bc_charged_gf_invert_contract] writing data to file %s\n", filename);

      int ncdim = 2;
      int cdim[2] = { g_source_momentum_number, 3 };
      char tag[100];
      sprintf ( tag, "/src_mom" );

      exitstatus = write_h5_contraction ( g_source_momentum_list[0], NULL, filename, tag, "int", ncdim, cdim );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_bc_charged_gf_invert_contract] Error from write_h5_contraction,  status %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(124);
      }

      cdim[0] = g_sink_momentum_number;
      sprintf ( tag, "/snk_mom" );

      exitstatus = write_h5_contraction ( g_sink_momentum_list[0], NULL, filename, tag, "int", ncdim, cdim );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_bc_charged_gf_invert_contract] Error from write_h5_contraction,  status %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(124);
      }

      writer = (void *)filename;

#elif defined HAVE_LHPC_AFF
    /***************************************************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     ***************************************************************************/
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      writer = (void *)affw;
#else
      fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error, no outupt variant selected %s %d\n",  __FILE__, __LINE__);
      EXIT(15);
#endif
    }  /* end of if io_proc == 2 */

    double * point_source_flowed = init_1level_dtable ( _GSI(VOLUME) );
    if ( point_source_flowed == NULL )
    {
      fprintf ( stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_level_table    %s %d\n", __FILE__, __LINE__ );
      EXIT(2);
    }

    /* up and down quark propagator */
    double *** propagator = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
    if( propagator == NULL ) {
      fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }


#if _PART_PROP
    /***************************************************************************
     ***************************************************************************
     **
     ** _PART_PROP
     **
     ** point-to-all propagators with source
     **
     ***************************************************************************
     ***************************************************************************/
        
    /* loop on spin-color components */
    for ( int isc = 0; isc < 12; isc++ )
    {
      memset ( point_source_flowed , 0, sizeof_spinor_field );
      if ( g_cart_id == source_proc_id )
      {
        if ( g_verbose > 2 )
        {
          fprintf (stdout, "# [njjn_bd_charged_gf_invert_contract] proc%.4d sets the source    %s %d\n", g_cart_id, __FILE__, __LINE__ );
        }
        // have process for source position, fill in the source vector
        point_source_flowed[_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]])+2*isc] = 1.;
      }

#ifdef _GFLOW_QUDA

      /***********************************************************
       * apply adjoint flow to the point source
       ***********************************************************/
#if 0
      gettimeofday ( &ta, (struct timezone *)NULL );

      /* upload original gauge field to device */
      loadGaugeQuda ( (void *)h_gauge, &gauge_param );

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "loadGaugeQuda", g_cart_id == 0 );
#endif
      /***********************************************************/
      /***********************************************************/

      smear_param.n_steps       = gf_niter;
      smear_param.epsilon       = gf_dt;
      smear_param.meas_interval = 1;
      smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;
      smear_param.restart       = QUDA_BOOLEAN_TRUE;
#ifdef _TEST_TIMER
      gettimeofday ( &ta, (struct timezone *)NULL );
#endif
      _performGFlowAdjoint ( point_source_flowed, point_source_flowed, &smear_param, gf_niter, gf_nb, gf_ns );
#ifdef _TEST_TIMER
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "_performGFlowAdjoint-restart", g_cart_id == 0 );
#endif
#endif  // of if _GFLOW_QUDA

      /***********************************************************
       * invert on flowed source for flavor up and dn    
       ***********************************************************/
      for ( int iflavor = 0; iflavor < 2; iflavor++ ) 
      {
#ifdef _TEST_TIMER
        gettimeofday ( &ta, (struct timezone *)NULL );
#endif
        /***********************************************************
         * flavor-type point-to-all propagator
         *
         * NEITHER SOURCE NOR SINK SMEARING here, using gradient flow
         *
         * NOTE: quark flavor is controlled by value of iflavor
         ***********************************************************/
        exitstatus = prepare_propagator_from_source ( propagator[iflavor]+isc, &point_source_flowed , 1, iflavor, 
                0, 0, NULL, check_propagator_residual, gauge_field_with_phase, lmzz, NULL );

        if(exitstatus != 0) {
          fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from prepare_propagator_from_source, status %d   %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(12);
        }
#ifdef _TEST_TIMER
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "prepare_propagator_from_source", g_cart_id == 0 );
#endif

        /***********************************************************
         * forward gradient flow at sink
         ***********************************************************/
#ifdef _GFLOW_QUDA

#if 0
        gettimeofday ( &ta, (struct timezone *)NULL );

        loadGaugeQuda ( (void *)h_gauge, &gauge_param );

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "loadGaugeQuda", g_cart_id == 0 );
#endif  // of if 0

        /***********************************************************/
        /***********************************************************/
        smear_param.n_steps       = gf_niter;
        smear_param.epsilon       = gf_dt;
        smear_param.meas_interval = 1;
        smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;
        smear_param.restart       = QUDA_BOOLEAN_TRUE;
#ifdef _TEST_TIMER
        gettimeofday ( &ta, (struct timezone *)NULL );
#endif
        _performGFlowForward ( propagator[iflavor][isc], propagator[iflavor][isc], &smear_param, 0 );
#ifdef _TEST_TIMER
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "_performGFlowForward-restart", g_cart_id == 0 );
#endif

#endif  // of _GFLOW_QUDA
      }  /* end of loop on flavor */

    }  /* end of loop on spin-color component */

#endif  /* of if _PART_PROP */

    /***************************************************************************
     * allocate propagator fields
     *
     * these are ordered as as
     * t,x,y,z,spin,color,spin,color
     * so a 12x12 complex matrix per space-time point
     ***************************************************************************/
    fermion_propagator_type * fp  = create_fp_field ( VOLUME );
    fermion_propagator_type * fp2 = create_fp_field ( VOLUME );
    fermion_propagator_type * fp3 = create_fp_field ( VOLUME );

#if _PART_TWOP
    /***************************************************************************
     ***************************************************************************
     **
     ** Part IIb
     **
     ** point-to-all propagator contractions for baryon 2pts
     **
     ***************************************************************************
     ***************************************************************************/

    /***************************************************************************
     * loop on flavor combinations
     ***************************************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ ) {
#ifdef _TEST_TIMER
      gettimeofday ( &ta, (struct timezone *)NULL );
#endif
      /***************************************************************************
       * vx holds the x-dependent nucleon-nucleon spin propagator,
       * i.e. a 4x4 complex matrix per space time point
       ***************************************************************************/
      double ** vx = init_2level_dtable ( VOLUME, 32 );
      if ( vx == NULL ) {
        fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }

      /***************************************************************************
       * vp holds the nucleon-nucleon spin propagator in momentum space,
       * i.e. the momentum projected vx
       ***************************************************************************/
      double *** vp = init_3level_dtable ( T, g_source_momentum_number, 32 );
      if ( vp == NULL ) {
        fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      /***************************************************************************
       *
       * [ X^T Xb X ] - [ Xb^+ X^* X^+ ]
       *
       ***************************************************************************/

      char  tag_prefix[200], tag[400];
      sprintf ( tag_prefix, "/N-N/%c%c%c/gf%6.4f", flavor_tag[iflavor], flavor_tag[1-iflavor], flavor_tag[iflavor], gf_tau);
       
      /***************************************************************************
       * fill the fermion propagator fp with the 12 spinor fields
       * in propagator of flavor X
       ***************************************************************************/
      assign_fermion_propagator_from_spinor_field ( fp, propagator[iflavor], VOLUME);

      /***************************************************************************
       * fill fp2 with 12 spinor fields from propagator of flavor Xb
       ***************************************************************************/
      assign_fermion_propagator_from_spinor_field ( fp2, propagator[1-iflavor], VOLUME);

      /***************************************************************************
       * contractions for n1, n2
       *
       * if1/2 loop over various Dirac Gamma-structures for
       * baryon interpolators at source and sink
       ***************************************************************************/
      for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
      for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

        /***************************************************************************
         * here we calculate fp3 = Gamma[if2] x propagator[1-iflavor] / fp2 x Gamma[if1]
         ***************************************************************************/
        fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp2, VOLUME );

        fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );

        fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );

        /***************************************************************************
         * diagram n1
         ***************************************************************************/
        sprintf(tag, "%s/Gi_%s/Gf_%s/t1", tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

        exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp, contract_v5, writer, tag, g_source_momentum_list, g_source_momentum_number, 16, VOLUME, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(48);
        }

        /***************************************************************************
         * diagram n2
         ***************************************************************************/
        sprintf(tag, "%s/Gi_%s/Gf_%s/t2", tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

        exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp, contract_v6, writer, tag, g_source_momentum_list, g_source_momentum_number, 16, VOLUME, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(48);
        }

      }}  /* end of loop on Dirac Gamma structures */

      fini_2level_dtable ( &vx );
      fini_3level_dtable ( &vp );

#ifdef _TEST_TIMER
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "n1-n2-reduce-project-write", g_cart_id == 0 );
#endif

    }  /* end of loop on flavor */
  
#endif  /* end of if _PART_TWOP  */

#if _PART_THREEP
    /***************************************************************************
     *
     * sequential inversion and contraction
     *
     ***************************************************************************/

    /***************************************************************************
     * sequential source momenta
     ***************************************************************************/
    int momentum[3] = { 0,0,0};

    /***************************************************************************
     * read the loop for total flowtime gf_tau,
     * average over samples
     ***************************************************************************/

    double _Complex * loop_buffer = init_1level_ztable ( VOLUME * 12 *12 );
    if ( loop_buffer  == NULL ) {
      fprintf ( stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }


    for ( int i = 0; i < g_nsample; i++ )
    {
      char loop_filename[200];
      sprintf ( loop_filename, "loop.up.c%d.N%d.tau%6.4f.lime", Nconf, i, gf_tau );

#ifdef _TEST_TIMER
      gettimeofday ( &tb, (struct timezone *)NULL );
#endif
      exitstatus = read_lime_contraction ( (double*)(loop_buffer), loop_filename, 144, 0 );
      if ( exitstatus != 0  ) {
        fprintf ( stderr, "[njjn_bd_charged_gf_invert_contract] Error read_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
       EXIT(12);
      }
#ifdef _TEST_TIMER
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "read_lime_contraction-loop", g_cart_id == 0 );
#endif

#pragma omp parallel for
      for ( size_t i = 0; i < VOLUME * 12 * 12; i++ )
      {
        loop[0][0][i] += loop_buffer[i];
      }
    }
#pragma omp parallel for
    for ( size_t i = 0; i < VOLUME * 12 * 12; i++ )
    {
      loop[0][0][i] /= (double)g_nsample;
    }

    fini_1level_ztable ( &loop_buffer );

    /***************************************************************************
     * loop on flavor
     ***************************************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ )
    {
      if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] start seq source flavor %3d   %s %d\n", 
          iflavor, __FILE__, __LINE__);

        /***************************************************************************
         * vx holds the x-dependent nucleon-nucleon spin propagator,
         * i.e. a 4x4 complex matrix per space time point
         ***************************************************************************/
        double ** vx = init_2level_dtable ( VOLUME, 32 );
        if ( vx == NULL ) {
          fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
          EXIT(47);
        }

        /***************************************************************************
         * vp holds the nucleon-nucleon spin propagator in momentum space,
         * i.e. the momentum projected vx
         ***************************************************************************/
        double *** vp = init_3level_dtable ( T, g_sink_momentum_number, 32 );
        if ( vp == NULL ) {
          fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(47);
        }

        /***************************************************************************
         ***************************************************************************
         **
         ** Part III
         **
         ** sequential inversion with loop-product sequential sources
         ** and contractions for N - qbar q qbar q - N B,Z,D_1c/i diagrams
         **
         ***************************************************************************
         ***************************************************************************/

        /***************************************************************************
         * loop on 2 types of sequential fht sources
         ***************************************************************************/
        for ( int seq_source_type = 0; seq_source_type <= 1; seq_source_type++ )
        {

          if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] start seq source type %3d   %s %d\n", 
             seq_source_type, __FILE__, __LINE__);

          /***************************************************************************
           * allocate for sequential propagator and source
           ***************************************************************************/
          double ** sequential_propagator = init_2level_dtable ( 12, _GSI( VOLUME ) );
          if( sequential_propagator == NULL ) {
            fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(123);
          }

          double ** sequential_source = init_2level_dtable ( 12,  _GSI(VOLUME) );
          if( sequential_source == NULL ) {
            fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(132);
          }

          char const sequential_propagator_name = ( seq_source_type == 0 ) ? 'd' : 'b';


          /***************************************************************************
           * loop on sequential source gamma matrices
           ***************************************************************************/
          for ( int igamma = 0; igamma < sequential_gamma_sets; igamma++ ) {

            if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] start seq source gamma set %s   %s %d\n", 
               sequential_gamma_tag[igamma], __FILE__, __LINE__);

            /***************************************************************************
             * loop on loop flavor, but not for B diagram, only for D1c/i diagram
             ***************************************************************************/
            for ( int iloop_flavor = 0; iloop_flavor < 2; iloop_flavor++ ) 
            {

              if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_bd_charged_gf_invert_contract] using flavor %d / loop_flavor %d for seq source type %d  %s %d\n", 
                  iflavor, iloop_flavor, seq_source_type, __FILE__, __LINE__ );

              /***************************************************************************
               * add sequential fht vertex
               ***************************************************************************/
              gettimeofday ( &ta, (struct timezone *)NULL );

              /***************************************************************************
               * single sequential source from the propagator
               * 
               * CHECK 1-iflavor flavor-id of propagator
               ***************************************************************************/
              exitstatus = prepare_sequential_fht_loop_source ( 
                    sequential_source, 
                    loop, 
                    propagator[1-iflavor], 
                    sequential_gamma_list[igamma], 
                    sequential_gamma_num[igamma], 
                    NULL, 
                    seq_source_type, ( iloop_flavor == 0 ? NULL : &gammafive ) );

              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[njjn_bd_charged_gf_invert_contract] Error from prepare_sequential_fht_loop_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(123);
              }

              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "prepare-sequential-fht-source", g_cart_id == 0 );

              /***************************************************************************
               * adjoint flow to the sequential source
               *
               * SWITCH OFF UPDATING the flowed gauge field ?
               ***************************************************************************/

              for ( int isc = 0; isc < 12; isc++ )
              {
                smear_param.n_steps       = gf_niter;
                smear_param.epsilon       = gf_dt;
                smear_param.meas_interval = 1;
                smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;
                smear_param.restart       = QUDA_BOOLEAN_TRUE;
#ifdef _TEST_TIMER
                gettimeofday ( &ta, (struct timezone *)NULL );
#endif
                _performGFlowAdjoint ( sequential_source[isc], sequential_source[isc], &smear_param, gf_niter, gf_nb, gf_ns );
#ifdef _TEST_TIMER
                gettimeofday ( &tb, (struct timezone *)NULL );
                show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "_performGFlowAdjoint-restart", g_cart_id == 0 );
#endif
              }

              /***************************************************************************
               * invert the Dirac operator on the sequential source
               ***************************************************************************/
#ifdef _TEST_TIMER
	      gettimeofday ( &ta, (struct timezone *)NULL );
#endif
              exitstatus = prepare_propagator_from_source ( sequential_propagator, sequential_source, 12, iflavor, 0, 0, NULL,
                  check_propagator_residual, gauge_field_with_phase, lmzz, NULL );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[njjn_bd_charged_gf_invert_contract] Error from prepare_propagator_from_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(123);
              }
#ifdef _TEST_TIMER
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "prepare_propagator_from_source", g_cart_id == 0 );
#endif
              /***************************************************************************
               * apply forward flow to sequential_propagator
               ***************************************************************************/
              for ( int isc = 0; isc < 12; isc++ )
              {
                smear_param.n_steps       = gf_niter;
                smear_param.epsilon       = gf_dt;
                smear_param.meas_interval = 1;
                smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;
                smear_param.restart       = QUDA_BOOLEAN_TRUE;
#ifdef _TEST_TIMER
                gettimeofday ( &ta, (struct timezone *)NULL );
#endif
                _performGFlowForward ( sequential_propagator[isc], sequential_propagator[isc], &smear_param, 0 );
#ifdef _TEST_TIMER
                gettimeofday ( &tb, (struct timezone *)NULL );
                show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "_performGFlowForward-restart", g_cart_id == 0 );
#endif
              }

              /***************************************************************************
               *
               * contractions
               *
               ***************************************************************************/
              char correlator_tag[20] = "N-qbGqqbGq-N";
          
              char tag_prefix[200], tag_prefix2[200], tag_prefix3[200], tag[400];

              sprintf ( tag_prefix, "/%s/%c%c%c%c-f%c-f%c/gf%6.4f/nsample%d/Gc_%s",
                      correlator_tag, 
                      sequential_propagator_name, flavor_tag[iflavor], flavor_tag[iloop_flavor], flavor_tag[1-iflavor],
                      flavor_tag[iflavor],
                      flavor_tag[1-iflavor],
                      gf_tau, g_nsample, sequential_gamma_tag[igamma] );

              sprintf ( tag_prefix2, "/%s/f%c-%c%c%c%c-f%c/gf%6.4f/nsample%d/Gc_%s",
                      correlator_tag,
                      flavor_tag[iflavor],
                      sequential_propagator_name, flavor_tag[iflavor], flavor_tag[iloop_flavor], flavor_tag[1-iflavor],
                      flavor_tag[1-iflavor],
                      gf_tau, g_nsample, sequential_gamma_tag[igamma] );

              sprintf ( tag_prefix3, "/%s/f%c-f%c-%c%c%c%c/gf%6.4f/nsample%d/Gc_%s",
                      correlator_tag,
                      flavor_tag[iflavor],
                      flavor_tag[1-iflavor],
                      sequential_propagator_name, flavor_tag[iflavor], flavor_tag[iloop_flavor], flavor_tag[1-iflavor],
                      gf_tau, g_nsample, sequential_gamma_tag[igamma] );

              /***************************************************************************
               * B/D1c/i for uu uu insertion
               ***************************************************************************/
#ifdef _TEST_TIMER
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif
              /***************************************************************************
               * fp2 = b up-after-up-after-up 
               ***************************************************************************/
              assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator, VOLUME);
  
              /***************************************************************************
               * contractions as for t1,...,t4 of N-N type diagrams
               ***************************************************************************/
              for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) 
              {
                /***************************************************************************
                 * fp = fwd up
                 ***************************************************************************/
                assign_fermion_propagator_from_spinor_field ( fp, propagator[iflavor], VOLUME);

                /***************************************************************************
                 * fp <- fp x Gamma_i1 = fwd up x Gamma_i1
                 * in-place
                 ***************************************************************************/
                fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp, gamma_f1_list[if1], fp, VOLUME );

              for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) 
              {
                /***************************************************************************
                 * fp3 = fwd dn
                 ***************************************************************************/
                assign_fermion_propagator_from_spinor_field ( fp3, propagator[1-iflavor], VOLUME);

                /***************************************************************************
                 * fp3 <- Gamma_i2 x fp3 = Gamma[if2] x fwd dn
                 * in-place
                 ***************************************************************************/
                fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp3, VOLUME );
       
                /***************************************************************************
                 * fp3 <- (-1) x sign Gamma_i1 x sigm Gamma_i2 x fp3
                 * in-place
                 ***************************************************************************/
                fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );
      
                /***************************************************************************
                 * diagram t1
                 ***************************************************************************/
                sprintf(tag, "%s/Gf_%s/Gi_%s/t1", tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
 
                /*                                          seq  u   d   */
                exitstatus = reduce_project_write ( vx, vp, fp2, fp, fp3, contract_v5, writer, tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(48);
                }
     
                /***************************************************************************
                 * diagram t2
                 ***************************************************************************/
                sprintf(tag, "%s/Gf_%s/Gi_%s/t2", tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
      
                exitstatus = reduce_project_write ( vx, vp, fp2, fp, fp3, contract_v6, writer, tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(48);
                }

                /***************************************************************************
                 * diagram t1
                 ***************************************************************************/
                sprintf(tag, "%s/Gf_%s/Gi_%s/t1", tag_prefix2, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
     
                /*                                          u   seq  d  */
                exitstatus = reduce_project_write ( vx, vp, fp, fp2, fp3, contract_v5, writer, tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(48);
                }

                /***************************************************************************
                 * diagram t1
                 ***************************************************************************/
                sprintf(tag, "%s/Gf_%s/Gi_%s/t1", tag_prefix3, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
   
                /*                                          u   d    seq */
                exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp2, contract_v5, writer, tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(48);
                }

              }} // end of loop on Dirac gamma structures
#ifdef _TEST_TIMER
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "b-d-reduce-project-write", g_cart_id == 0 );
#endif
              /***************************************************************************/
              /***************************************************************************/

            }  /* end of loop on loop flavor */
              
            /***************************************************************************/
            /***************************************************************************/

          } // end of loop on sequential source gamma matrices
      
          fini_2level_dtable ( &sequential_source );
          fini_2level_dtable ( &sequential_propagator );

        }  /* end of loop on seq. source type */

	fini_2level_dtable ( &vx );
        fini_3level_dtable ( &vp );

      }  /* loop on flavor type */

#endif  /* of if _PART_THREEP */


    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * clean up
     ***************************************************************************/
    free_fp_field ( &fp  );
    free_fp_field ( &fp2 );
    free_fp_field ( &fp3 );

#if defined HAVE_LHPC_AFF && not defined HAVE_HDF5
    /***************************************************************************
     * I/O process id 2 closes its AFF writer
     ***************************************************************************/
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_bd_charged_gf_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    /***************************************************************************
     * free propagator fields
     ***************************************************************************/
    fini_3level_dtable ( &propagator );

  }  /* end of loop on source locations */

  _performGFlowAdjoint ( NULL, NULL, NULL, 0, 0, -1 );

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/
  if ( loop != NULL ) fini_3level_ztable ( &loop );

#if _GFLOW_QUDA
  fini_2level_dtable ( &h_gauge );
#endif

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
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
  show_time ( &start_time, &end_time, "njjn_bd_charged_gf_invert_contract", "runtime", g_cart_id == 0 );

  return(0);

}
