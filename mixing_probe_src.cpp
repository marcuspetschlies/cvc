/***************************************************************************
 *
 * mixing_probe_src
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
#include "pm.h"
#include "gradient_flow.h"

#ifndef _OP_ID_UP
#define _OP_ID_UP 0
#endif
#ifndef _OP_ID_DN
#define _OP_ID_DN 1
#endif

#ifndef _USE_TIME_DILUTION
#define _USE_TIME_DILUTION 1
#endif

#ifndef _GFLOW_STEPS_MAX
#define _GFLOW_STEPS_MAX 20
#endif

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
  
  const char outfile_prefix[] = "mx_prb_src";

  const char flavor_tag[4] = { 'u', 'd', 's', 'c' };

  const int sequential_gamma_sets = 4;
  int const sequential_gamma_num[4] = {4, 4, 1, 1};
  int const sequential_gamma_id[4][4] = {
    { 0,  1,  2,  3 },
    { 6,  7,  8,  9 },
    { 4, -1, -1, -1 },
    { 5, -1, -1, -1 } };

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


  char const gamma_id_to_ascii[16][10] = {
    "gt",
    "gx",
    "gy",
    "gz",
    "id",
    "g5",
    "gtg5",
    "gxg5",
    "gyg5",
    "gzg5",
    "gtgx",
    "gtgy",
    "gtgz",
    "gxgy",
    "gxgz",
    "gygz" 
  };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  double *gauge_field_smeared    = NULL;
  double *gflow_gauge_field      = NULL;
  struct timeval ta, tb, start_time, end_time;

  /*
  int const    gamma_f1_number                           = 4;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 , 11,  8,  2 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 , +1, -1, -1 };
  */

  int const    gamma_f1_number                           = 1;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 };

  int read_loop_field    = 0;
  int write_loop_field   = 0;
  int read_scalar_field  = 0;
  int write_scalar_field = 0;

  int gflow_nstep = 0;
  int gflow_steps[_GFLOW_STEPS_MAX];
  double gflow_dt = 0.;

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char aff_tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sSrwch?f:n:t:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'r':
      read_loop_field = 1;
      break;
    case 'w':
      write_loop_field = 1;
      break;
    case 's':
      read_scalar_field = 1;
      break;
    case 'S':
      write_scalar_field = 1;
      break;
    case 'n':
      gflow_steps[gflow_nstep] = atoi ( optarg );
      gflow_nstep++;
      break;
    case 't':
      gflow_dt = atof ( optarg );
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

  fprintf(stdout, "# [mixing_probe_src] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [mixing_probe_src] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [mixing_probe_src] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [mixing_probe_src] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[mixing_probe_src] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[mixing_probe_src] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [mixing_probe_src] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [mixing_probe_src] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[mixing_probe_src] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[mixing_probe_src] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[mixing_probe_src] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize the clover term, 
   * lmzz and lmzzinv
   *
   *   mzz = space-time diagonal part of the Dirac matrix
   *   l   = light quark mass
   ***************************************************************************/
  exitstatus = init_clover ( &lmzz, &lmzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[mixing_probe_src] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[mixing_probe_src] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [mixing_probe_src] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

#if _TEST
  FILE *ofs = fopen ( "mx_prb_src.test", "w" );
  if( ofs == NULL ) {
    fprintf(stderr, "[mixing_probe_src] Error from fopen %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
#endif


  /***************************************************************************
   * set the gamma matrices
   ***************************************************************************/
#if 0
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
#endif

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

#if _TEST
  /* TEST */
  gamma_print ( &(gamma_s[0]), "g4", ofs );
  gamma_print ( &(gamma_p[0]), "g5", ofs );
  gamma_print ( &(gamma_v[0]), "g0", ofs );
  gamma_print ( &(gamma_v[1]), "g1", ofs );
  gamma_print ( &(gamma_v[2]), "g2", ofs );
  gamma_print ( &(gamma_v[3]), "g3", ofs );
  gamma_print ( &(gamma_a[0]), "g6", ofs );
  gamma_print ( &(gamma_a[1]), "g7", ofs );
  gamma_print ( &(gamma_a[2]), "g8", ofs );
  gamma_print ( &(gamma_a[3]), "g9", ofs );
  /* END */
#endif


  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );
  size_t const sizeof_gauge_field  = 72 * VOLUME * sizeof( double );

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[mixing_probe_src] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

#if ( defined _SMEAR_QUDA ) || ( defined _GFLOW_QUDA )
  {
    /***************************************************************************
     * dummy solve, just to have original gauge field up on device,
     * for subsequent APE smearing
     ***************************************************************************/

    double ** spinor_work = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
    if ( spinor_work == NULL ) {
      fprintf ( stderr, "[mixing_probe_src] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    memset(spinor_work[1], 0, sizeof_spinor_field);
    memset(spinor_work[0], 0, sizeof_spinor_field);
    if ( g_cart_id == 0 ) spinor_work[0][0] = 1.;
    exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], _OP_ID_UP);
#if ( defined GPU_DIRECT_SOLVER )
    if(exitstatus < 0)
#else
    if(exitstatus != 0)
#endif
    {
      fprintf(stderr, "[mixing_probe_src] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(12);
    }

    finit_2level_dtable ( &spinor_work );

  }
#endif  /* of if _SMEAR_QUDA or _GFLOW_QUDA */

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

  if ( g_nsample_oet > 0 ) {
    scalar_field = init_2level_dtable ( g_nsample_oet, 2*VOLUME );
    if( scalar_field == NULL ) {
      fprintf(stderr, "[mixing_probe_src] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(132);
    }
  }

  if ( ! read_scalar_field  && ( g_nsample_oet > 0 ) ) {

    /***************************************************************************
     * draw a stochastic binary source (real, +/1 one per site )
     ***************************************************************************/
    ranbinary ( scalar_field[0], 2 * g_nsample_oet * VOLUME );

    /***************************************************************************
     * write loop field to lime file
     ***************************************************************************/
    if ( write_scalar_field ) {
      sprintf( filename, "scalar_field.c%d.N%d.lime", Nconf, g_nsample_oet );
      
      char field_type[2000];

      sprintf( field_type, "<source_type>%d</source_type><noise_type>binary real</noise_type><coherent_sources>%d</coherent_sources>", g_source_type , g_coherent_source_number );

      for ( int i = 0; i < g_nsample_oet; i++ ) {
        exitstatus = write_lime_contraction( scalar_field[i], filename, 64, 1, field_type, Nconf, ( i > 0 ) );
        if ( exitstatus != 0  ) {
          fprintf ( stderr, "[mixing_probe_src] Error write_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(12);
        }
      }
    }  /* end of if write_loop_field */

  } else {
    sprintf( filename, "scalar_field.c%d.N%d.lime", Nconf, g_nsample_oet );
      
    for ( int i = 0; i < g_nsample_oet; i++ ) {
      exitstatus = read_lime_contraction ( scalar_field[i], filename, 1, i );
      if ( exitstatus != 0  ) {
        fprintf ( stderr, "[mixing_probe_src] Error read_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }
    }
  }  /* end of if read scalar field */

  /***************************************************************************
   * init gflow gauge field
   *
   * DO WE NEED PLUSRAND HERE ?
   ***************************************************************************/
  alloc_gauge_field( &gflow_gauge_field, VOLUME );
  if( gflow_gauge_field == NULL )  {
    fprintf(stderr, "[mixing_probe_src] Error from alloc_gauge_field %s %d\n", __FILE__, __LINE__);
    EXIT(1);
  }

  memcpy ( gflow_gauge_field, gauge_field_with_phase , sizeof_gauge_field );


  /***************************************************************************
   ***************************************************************************
   **
   ** Part Ia
   **
   ** prepare stochastic sources and propagators
   ** to contract the loop for insertion as part of
   ** sequential source
   **
   ***************************************************************************
   ***************************************************************************/

  double _Complex **** loop = NULL;

  loop = init_4level_ztable ( gflow_nstep,  VOLUME, 12, 12 );
  if ( loop  == NULL ) {
    fprintf ( stderr, "[mixing_probe_src] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

  if ( ! read_loop_field ) {

    if ( g_cart_id == 0 ) {
      fprintf ( stdout, "# [mixing_probe_src] produce loop field %s %d\n",  __FILE__, __LINE__ );
    }

    /***************************************************************************
     * loop on samples
     * invert and contract loops
     ***************************************************************************/
    for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

      gettimeofday ( &ta, (struct timezone *)NULL );

      double ** spinor_work = init_2level_dtable ( 3, _GSI( VOLUME+RAND ) );
      if ( spinor_work == NULL ) {
        fprintf ( stderr, "[mixing_probe_src] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
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
      // for ( int timeslice = 0; timeslice < T_global; timeslice ++ )
      for ( int timeslice = 1; timeslice <= 1; timeslice ++ )
      {
#endif

        for ( int ispin = 0; ispin < 4; ispin++ ) 
        {

          for ( int icol = 0; icol < 3; icol++ ) 
          {

            int const isc = 3 * ispin + icol;

            memset ( spinor_work[0], 0, sizeof_spinor_field );
            memset ( spinor_work[1], 0, sizeof_spinor_field );
 
#if _USE_TIME_DILUTION
            if ( timeslice / T == g_proc_coords[0] ) {
              if ( g_verbose > 2 ) fprintf( stdout, "# [mixing_probe_src] proc %d has global timeslice %d %s %d\n",
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

            if ( g_write_source ) {

              sprintf( filename, "stochastic_source.c%d.n%d.t%d.s%d.c%d", Nconf, isample, timeslice, ispin, icol );
              if ( ( exitstatus = write_propagator ( spinor_work[0], filename, 0, g_propagator_precision) ) != 0 ) {
                fprintf(stderr, "[mixing_probe_src] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(2);
              }
            }  /* end of if write source */


#else

#pragma omp parallel for
            for ( unsigned int ix = 0; ix < VOLUME; ix++  ) {
              size_t const iy = _GSI(ix) + 2 * isc;  /* offset for site ix and spin-color isc */
              spinor_work[0][ iy    ] = scalar_field[isample][ 2 * ix     ];
              spinor_work[0][ iy + 1] = scalar_field[isample][ 2 * ix + 1 ];
            }
#endif
            /* keep a copy of the sources field */
            memcpy ( spinor_work[2], spinor_work[0], sizeof_spinor_field );

            /* tm-rotate stochastic propagator at source, in-place */
            if( g_fermion_type == _TM_FERMION ) {
              spinor_field_tm_rotation(spinor_work[2], spinor_work[2], 1, g_fermion_type, VOLUME);
            }

            /* call to (external/dummy) inverter / solver
             *
             * ASSUMING HERE that solver leaves source field unchanged
             *
             * */
            exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[2], _OP_ID_UP );
#if ( defined GPU_DIRECT_SOLVER )
            if(exitstatus < 0)
#else
            if(exitstatus != 0)
#endif
            {
              fprintf(stderr, "[mixing_probe_src] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(12);
            }

            if ( check_propagator_residual ) {
              check_residual_clover ( &(spinor_work[1]), &(spinor_work[2]), gauge_field_with_phase, lmzz[_OP_ID_UP], 1 );
            }

            /* tm-rotate stochastic propagator at sink */
            if( g_fermion_type == _TM_FERMION ) {
              spinor_field_tm_rotation(spinor_work[1], spinor_work[1], 1, g_fermion_type, VOLUME);
            }

            /***************************************************************************/
            /***************************************************************************/

            /***************************************************************************
             * propagator and source are available
             *
             * now we loop over the number of gflow steps
             ***************************************************************************/
            for ( int istep = 0; istep < gflow_nstep; istep++ ) {

              int steps = istep == 0 ? gflow_steps[0] : gflow_steps[istep] - gflow_steps[istep-1];

              /***************************************************************************
               * flow the source and solution
               ***************************************************************************/
              gettimeofday ( &ta, (struct timezone *)NULL );

              flow_fwd_gauge_spinor_field ( gflow_gauge_field, spinor_work, 2, steps, gflow_dt, 1, 1 );
 
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "mixing_probe_src", "flow_fwd_gauge_spinor_field", g_cart_id == 0 );

              /***************************************************************************/
              /***************************************************************************/

              /***************************************************************************
               * fill in loop matrix element ksc = (kspin, kcol ), isc = (ispin, icol )
               * as
               * loop[ksc][isc] = prop[ksc] * source[isc]^+
               ***************************************************************************/
#pragma omp parallel for
                for ( size_t ix = 0; ix < VOLUME;   ix++ )
                {
 
                  size_t const iy = _GSI(ix);

                  for ( int kspin = 0; kspin < 4; kspin++ ) {
                    for ( int kcol = 0; kcol < 3; kcol++ ) {
                      int const ksc = 3 * kspin + kcol;
 
                      for ( int jspin = 0; jspin < 4; jspin++ ) {
                        for ( int jcol = 0; jcol < 3; jcol++ ) {
                          int const jsc = 3 * jspin + jcol;

                          loop[istep][ix][ksc][jsc] +=
                            /* 
                             * complex conjugate of source vector element 
                             */
                            /* ( scalar_field[isample][ 2 * ( ix + loffset ) ] - I * scalar_field[isample][ 2 * ( ix + loffset ) + 1] ) 
                             */
                            ( spinor_work[1][ iy + 2 * jsc  ] - I * spinor_work[1][ iy + 2 * jsc + 1 ] )
                            /* 
                             * times prop vector element
                             */
                          * ( spinor_work[1][ iy + 2 * ksc  ] + I * spinor_work[1][ iy + 2 * ksc + 1 ] );
                        }
                      }
                    }
                  }
                }  /* end of loop on volume */

              /***************************************************************************/
              /***************************************************************************/

            }  /* end of loop on gflow steps */

          }  /* end of loop on color dilution component */

        }  /* end of loop on spin dilution component */

#if _USE_TIME_DILUTION
      }  /* end of loop on timeslices */
#endif
      /* free fields */
      fini_2level_dtable ( &spinor_work );

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "mixing_probe_src", "loop-invert-contract-sample", g_cart_id == 0 );

    }  /* end of loop on samples */

    /***************************************************************************
     * normalize
     ***************************************************************************/
    if ( g_nsample_oet > 1 ) {
      double const norm = 1. / (double)g_nsample_oet;
#pragma omp parallel for
      for ( unsigned long int ix = 0; ix < 144 * VOLUME *gflow_nstep; ix++  ) {
        /* loop[0][0][ix] /= (double)g_nsample; */
        loop[0][0][0][ix] *= norm;
      }
    }

    /***************************************************************************
     * write loop field to lime file
     ***************************************************************************/
    if ( write_loop_field ) {
      char loop_type[2000];
      sprintf( loop_type, "<source_type>%d</source_type><noise_type>%d</noise_type><dilution_type>spin-color</dilution_type>", g_source_type, g_noise_type );
      for ( int i = 0; i < gflow_nstep; i++ ) {
        sprintf( filename, "loop.up.c%d.s%d.dt%6.4f.n%d.lime", Nconf, g_nsample, gflow_dt, gflow_steps[i] );
  
        exitstatus = write_lime_contraction( (double*)(loop[i][0][0]), filename, 64, 144, loop_type, Nconf, 0);
        if ( exitstatus != 0  ) {
          fprintf ( stderr, "[mixing_probe_src] Error write_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(12);
        }
      }

    }  /* end of if write_loop_field */

  } else {

    /***************************************************************************
     * read loop field from lime file
     ***************************************************************************/
    for ( int i = 0; i < gflow_nstep; i++ ) {
      sprintf( filename, "loop.up.c%d.s%d.dt%6.4f.n%d.lime", Nconf, g_nsample, gflow_dt, gflow_steps[i] );

      if ( io_proc == 2 && g_verbose > 0 ) {
        fprintf ( stdout, "# [mixing_probe_src] reading loop field from file %s %s %d\n", filename,  __FILE__, __LINE__ );
      }

      exitstatus = read_lime_contraction ( (double*)(loop[i][0][0]), filename, 144, 0 );
      if ( exitstatus != 0  ) {
        fprintf ( stderr, "[mixing_probe_src] Error read_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }
    }

  }  /* end of if on read stoch. source  */

  /***********************************************
   * if we want to use Jacobi smearing, we need 
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {

#ifndef _SMEAR_QUDA 

    alloc_gauge_field ( &gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy ( gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
#endif
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[mixing_probe_src] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
#ifndef _SMEAR_QUDA 
    }  /* end of if N_ape > 0 */
#endif
  }  /* end of if N_Jacobi > 0 */


  /***************************************************************************
   *
   * point-to-all version
   *
   ***************************************************************************/

  /***************************************************************************
   * loop on source locations
   *
   *   each source location is given by 4-coordinates in
   *   global variable
   *   g_source_coords_list[count][0-3] for t,x,y,z
   ***************************************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

    /***************************************************************************
     * allocate point-to-all propagators,
     * spin-color dilution (i.e. 12 fields per flavor of size 24xVOLUME real )
     ***************************************************************************/

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
      fprintf(stderr, "[mixing_probe_src] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     * open output file reader
     * we use the AFF format here
     * https://github.com/usqcd-software/aff
     *
     * one data file per source position
     ***************************************************************************/
    if(io_proc == 2) {
#if defined HAVE_LHPC_AFF
    /***************************************************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     ***************************************************************************/
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [mixing_probe_src] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[mixing_probe_src] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
#else
      fprintf(stderr, "[mixing_probe_src] Error, no outupt variant selected %s %d\n",  __FILE__, __LINE__);
      EXIT(15);
#endif
    }  /* end of if io_proc == 2 */

    /* up and down quark propagator with source smearing */
    double *** propagator = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
    if( propagator == NULL ) {
      fprintf(stderr, "[mixing_probe_src] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /* up and down quark propagator with source and sink smearing,
     * to use for baryon 2-pt function 
     */
    double *** propagator_snk_smeared = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
    if ( propagator_snk_smeared == NULL ) {
      fprintf(stderr, "[mixing_probe_src] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************/
    /***************************************************************************/

    for ( int iflavor = 0; iflavor < 2; iflavor++ ) 
    {

      gettimeofday ( &ta, (struct timezone *)NULL );

      /***********************************************************
       * flavor-type point-to-all propagator
       *
       * ONLY SOURCE smearing here
       *
       * NOTE: quark flavor is controlled by value of iflavor
       ***********************************************************/
      /*                                     output field         src coords flavor type  src smear  sink smear gauge field for smearing,  for residual check ...                                   */
      exitstatus = point_source_propagator ( propagator[iflavor], gsx,       iflavor,     1,         0,         gauge_field_smeared,       check_propagator_residual, gauge_field_with_phase, lmzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[mixing_probe_src] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(12);
      }
    
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "mixing_probe_src", "forward-light-smear-invert-check", g_cart_id == 0 );

      /***********************************************************
       * sink-smear the flavor-type point-to-all propagator
       * store extra
       ***********************************************************/

      gettimeofday ( &ta, (struct timezone *)NULL );

      for ( int i = 0; i < 12; i++ ) {
        /* copy propagator */
        memcpy ( propagator_snk_smeared[iflavor][i], propagator[iflavor][i], sizeof_spinor_field );

        /* sink-smear propagator */
        exitstatus = Jacobi_Smearing ( gauge_field_smeared, propagator_snk_smeared[iflavor][i], N_Jacobi, kappa_Jacobi);
        if(exitstatus != 0) {
          fprintf(stderr, "[mixing_probe_src] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          return(11);
        }
      }
    
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "mixing_probe_src", "forward-light-sink-smear", g_cart_id == 0 );

      /***********************************************************
       * optionally write the propagator to disc
       *
       * we use the standard lime format here
       * https://github.com/usqcd-software/c-lime
       ***********************************************************/
      if ( g_write_propagator ) {
        /* each spin-color component into a separate file */
        for ( int i = 0; i < 12; i++ ) {
          sprintf ( filename, "propagator_%c.%.4d.t%dx%dy%dz%d.%d.inverted", flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] , i );

          if ( ( exitstatus = write_propagator( propagator[iflavor][i], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[mixing_probe_src] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
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
     * reset gauge field
     ***************************************************************************/
    memcpy ( gflow_gauge_field, gauge_field_with_phase, sizeof_gauge_field );

    /* loop on gflow steps */
    for ( int istep = 0; istep < gflow_nstep; istep++ ) {

      int steps = istep == 0 ? gflow_steps[0] : gflow_steps[istep] - gflow_steps[istep-1];

      /***************************************************************************
       * flow the source and solution
       ***************************************************************************/
      gettimeofday ( &ta, (struct timezone *)NULL );

      flow_fwd_gauge_spinor_field ( gflow_gauge_field, propagator[0], 24, steps, gflow_dt, 1, 1 );

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "mixing_probe_src", "flow_fwd_gauge_spinor_field", g_cart_id == 0 );


      /***************************************************************************
       * loop on flavor combinations
       ***************************************************************************/
      /* for ( int iflavor = 0; iflavor < 2; iflavor++ ) */
      for ( int iflavor = 0; iflavor < 1; iflavor++ )
      {
  
        gettimeofday ( &ta, (struct timezone *)NULL );
  
        /* total number of correlators is 16 + 5
         *
         * 2 [b,d] x 4 [vasp] x 2 [g5,m] = 0,...,15
         * 16 = g5-g5 iso-singlet
         * 17 = 1-1 iso-triplet 3
         * 18 = g5 - g5 charged
         */
        int const ncorr = 2 * 4 * 2 + 3;
  
        double _Complex ** corr_accum = init_2level_ztable ( T, ncorr );
        if ( corr_accum == NULL ) {
          fprintf ( stderr, "[mixing_probe_src] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
          EXIT(2);
        }
  
#ifdef HAVE_OPENMP
        omp_lock_t writelock;
  
        omp_init_lock(&writelock);
  
#pragma omp parallel
{
#endif
  
        double _Complex ** corr = init_2level_ztable ( T, ncorr );
        if ( corr == NULL ) {
          fprintf ( stderr, "[mixing_probe_src] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
          EXIT(2);
        }
  
        double _Complex  ** pm       = init_2level_ztable ( 12, 12 );
        double _Complex  ** pm2      = init_2level_ztable ( 12, 12 );
        double _Complex  **  up      = init_2level_ztable ( 12, 12 );
        double _Complex  **  dn      = init_2level_ztable ( 12, 12 );
        double _Complex  ** gdg      = init_2level_ztable ( 12, 12 );
        double _Complex *** aux      = init_3level_ztable ( 4, 12, 12 );
        double _Complex *** aux2     = init_3level_ztable ( 4, 12, 12 );
  
        double _Complex *** gug      = init_3level_ztable ( 4, 12, 12 );
        double _Complex   * gu       = init_1level_ztable ( 10 );
  
        if (
               pm   == NULL ||
               pm2  == NULL ||
               up   == NULL ||
               dn   == NULL ||
               gdg  == NULL ||
               aux  == NULL ||
               aux2 == NULL ||
               gug  == NULL || 
               gu   == NULL ) {
            fprintf ( stderr, "[mixing_probe_src] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
  
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        /* for ( unsigned int ix = 0; ix < VOLUME; ix++ ) */
        for ( unsigned int ix = 2; ix < 3; ix++ )
        {
          
  
          int const it = ix / VOL3;
  
          unsigned int const iix = _GSI( ix );
  
          /***************************************************************************
           * make the point-wise up and down propagator matrix
           ***************************************************************************/
          /* fill up from 12 spinor fields */
          pm_set_from_sf_point ( up, propagator[iflavor], ix );
  
          /* fill dn from 12 spinor fields */
          pm_set_from_sf_point ( dn, propagator[1-iflavor], ix );
  
#if _TEST
          /* TEST */
          pm_print ( up, "up", ofs );
          pm_print ( dn, "dn", ofs );
          /* END */
#endif
  
          /***************************************************************************
           * auxilliary field
           * gdg <- g5 dn g5
           ***************************************************************************/
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gdg, &(gamma_p[0]), dn, 0. );
  
#if _TEST
          /* TEST */
          pm_print ( gdg, "gdg", ofs );
          /* END */
#endif
  
          /***************************************************************************
           * auxilliary field
           * these store 
           * gug <. sum_c Gamma_c Lu Gamma_c 
           ***************************************************************************/
  
          /***************************************************************************
           * ugu; first time init to zero, then add up
           ***************************************************************************/
          /* scalar */
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[0], &(gamma_s[0]), loop[istep][ix], 0. );
  
          /* pseudoscalar */
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[1], &(gamma_p[0]), loop[istep][ix], 0. );
  
          /* vector */
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], &(gamma_v[0]), loop[istep][ix], 0. );
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], &(gamma_v[1]), loop[istep][ix], 1. );
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], &(gamma_v[2]), loop[istep][ix], 1. );
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], &(gamma_v[3]), loop[istep][ix], 1. );
  
          /* axial-vector */
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], &(gamma_a[0]), loop[istep][ix], 0. );
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], &(gamma_a[1]), loop[istep][ix], 1. );
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], &(gamma_a[2]), loop[istep][ix], 1. );
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], &(gamma_a[3]), loop[istep][ix], 1. );
  
          /***************************************************************************
           * auxilliary field
           *
           * gu  <- Tr[ Gamma_c Lu ] for each c
           * single number per entry
           *
           * NOTE:sequence of gamma_i ( = Gamma_c ) must be the same everywhere
           * id g5 v a
           ***************************************************************************/
          /* scalar */
          pm_eq_gamma_ti_pm ( pm , &(gamma_s[0]), loop[istep][ix] );
          gu[0] = co_eq_tr_pm ( pm );
  
          /* pseudoscalar */
          pm_eq_gamma_ti_pm ( pm , &(gamma_p[0]), loop[istep][ix] );
          gu[1] = co_eq_tr_pm ( pm );
  
          /* vector */
          for ( int i = 0; i < 4; i++ ) {
            pm_eq_gamma_ti_pm ( pm , gamma_v+i, loop[istep][ix] );
            gu[2+i] = co_eq_tr_pm ( pm );
          }
  
          /* pseudovector */
          for ( int i = 0; i < 4; i++ ) {
            pm_eq_gamma_ti_pm ( pm , gamma_a+i, loop[istep][ix] );
            gu[6+i] = co_eq_tr_pm ( pm );
          }
  
#if _TEST
          /* TEST */
          pm_print ( loop[0][ix], "loop", ofs );
          pm_print ( gug[0], "sus", ofs );
          pm_print ( gug[1], "pup", ofs );
          pm_print ( gug[2], "vuv", ofs );
          pm_print ( gug[3], "aua", ofs );
  
          fprintf ( ofs, "gu <- numeric()\n" );
          for ( int i = 0; i < 10; i++ ) {
            fprintf ( ofs, " gu[%d] <- %25.16e + %25.16e*1.i\n", i+1, creal(gu[i]), cimag ( gu[i] ) );
          }
          /* END */
#endif
  
          /***************************************************************************
           * auxilliary field
           * aux, which contains
           *
           * aux = ( g5 D^+ g5 )_fi * ( gf up_ff gf ) * up_fi 
           ***************************************************************************/
          memset ( aux[0][0], 0, 576 * sizeof( double _Complex ) );
  
          for ( int i = 0; i < 4; i++ ) {
            /* pm <- ( gf up gf )  up */
            pm_eq_pm_ti_pm ( pm, gug[i], up );
         
            /* aux <- gdg^+ pm = [ g5 dn^+ g5 ] * ( gc up gc ) * up */
            pm_eq_pm_dag_ti_pm ( aux[i], gdg, pm );
          }
  
          /***************************************************************************
           * auxilliary field
           * aux2, which contains
           *
           * aux2 = ( g5 D^+ g5 )_fi gf  up_fi * Tr[ gf up_ff ]
           ***************************************************************************/
          memset ( aux2[0][0], 0, 576 * sizeof( double _Complex ) );
  
          /* scalar */
          pm_eq_gamma_ti_pm ( pm , &(gamma_s[0]), up );
          pm_eq_pm_dag_ti_pm ( aux2[0], gdg, pm );
          pm_eq_pm_ti_co  ( aux2[0] , aux2[0], gu[0] );
  
          /* pseudoscalar */
          pm_eq_gamma_ti_pm ( pm , &(gamma_p[0]), up );
          pm_eq_pm_dag_ti_pm ( aux2[1], gdg, pm );
          pm_eq_pm_ti_co  ( aux2[1] , aux2[1], gu[1] );
  
          /* vector */
          for ( int i = 0; i < 4; i++ ) {
            /* pm <- gc up */
            pm_eq_gamma_ti_pm ( pm , &(gamma_v[i]), up );
            /* pm2 <- (g5 dn g5)^+ pm */
            pm_eq_pm_dag_ti_pm ( pm2, gdg, pm );
            /* aux2 <- (i>0) aux2 + gu * pm2 */
            pm_eq_pm_pl_pm ( aux2[2], aux2[2], pm2, (double _Complex)(i>0), gu[2+i] );
          }
  
          /* axial vector */
          for ( int i = 0; i < 4; i++ ) {
            /* pm <- gc up */
            pm_eq_gamma_ti_pm ( pm , &(gamma_a[i]), up );
            /* pm2 <- (g5 dn g5)^+ pm */
            pm_eq_pm_dag_ti_pm ( pm2, gdg, pm );
            /* aux2 <- (i>0) aux2 + gu * pm2 */
            pm_eq_pm_pl_pm ( aux2[3], aux2[3], pm2, (double _Complex)(i>0), gu[6+i] );
          }
  
#if _TEST
          /* TEST */
          pm_print ( aux[0], "pDp_sus_u", ofs );
          pm_print ( aux[1], "pDp_pup_u", ofs );
          pm_print ( aux[2], "pDp_vuv_u", ofs );
          pm_print ( aux[3], "pDp_aua_u", ofs );
  
  
          pm_print ( aux2[0], "pDp_s_u_tsu", ofs );
          pm_print ( aux2[0], "pDp_p_u_tpu", ofs );
          pm_print ( aux2[0], "pDp_v_u_tvu", ofs );
          pm_print ( aux2[0], "pDp_a_u_tau", ofs );
  
          /* END OF TEST */
#endif
  
          /***************************************************************************
           * probing op at source, 4q at sink
           * Tr { Gamma_i aux } = Tr { Gamma_i [ g5 dn^+ g5 ]_fi * ( gf up gf ) * up }
           ***************************************************************************/
  
          /***************************************************************************
           * dim-3 operator psibar g5 psi
           ***************************************************************************/
  
          /* pseudoscalar , B */
          for ( int i = 0; i < 4; i++ ) {
            pm_eq_gamma_ti_pm ( pm, &(gamma_p[0]), aux[i] );
            corr[it][i] += co_eq_tr_pm ( pm );
          }
  
          /* pseudoscalar , D */
          for ( int i = 0; i < 4; i++ ) {
            pm_eq_gamma_ti_pm ( pm, &(gamma_p[0]), aux2[i] );
            corr[it][4+i] += co_eq_tr_pm ( pm );
          }
  
          /***************************************************************************
           * dim-4 operator psibar id psi, to be multiplied with m_q
           ***************************************************************************/
  
          /* scalar, B */
          for ( int i = 0; i < 4; i++ ) {
            corr[it][8+i] += co_eq_tr_pm ( aux[i] );
          }
          /* scalar, D */
          for ( int i = 0; i < 4; i++ ) {
            corr[it][12+i] += co_eq_tr_pm ( aux2[i] );
          }
  
          /***************************************************************************/
          /***************************************************************************/
  
          /***************************************************************************
           * pobing operator correlation functions at source and sink
           ***************************************************************************/
  
          /* make the point-wise up and down propagator matrix
           * with sink-smeared propagators
           */
  
          /* fill up from 12 spinor fields */
          pm_set_from_sf_point ( up, propagator_snk_smeared[iflavor], ix );
  
          /* fill dn from 12 spinor fields */
          pm_set_from_sf_point ( dn, propagator_snk_smeared[1-iflavor], ix );
  
          /***************************************************************************
           * auxilliary field
           * gdg <- g5 dn g5
           ***************************************************************************/
          pm_pl_eq_gamma_ti_pm_ti_gamma ( gdg, &(gamma_p[0]), dn, 0. );
  
          /* Tr [ g5 U g5 g5 D^+ g5 ] */
          pm_eq_gamma_ti_pm_dag ( pm, &(gamma_p[0]), gdg );
          pm_eq_pm_ti_pm ( pm2, up, pm );
          pm_eq_gamma_ti_pm ( pm, &(gamma_p[0]), pm2 );
          corr[it][16] += co_eq_tr_pm ( pm );
  
          /* Tr [ Id U Id g5 D^+ g5 ] */
          pm_eq_pm_ti_pm_dag ( pm, up, gdg );
          corr[it][17] += co_eq_tr_pm ( pm );
  
          /* Tr [ g5 U g5 g5 U^+ g5 ] */
          pm_eq_pm_ti_pm_dag ( pm, up, up );
          corr[it][18] += co_eq_tr_pm ( pm );
  
        }  /* end of loop on x */
  
        /***************************************************************************/
        /***************************************************************************/
  
        fini_2level_ztable ( &pm       );
        fini_2level_ztable ( &pm2      );
        fini_2level_ztable ( &up       );
        fini_2level_ztable ( &dn       );
        fini_2level_ztable ( &gdg      );
        fini_3level_ztable ( &aux      );
        fini_3level_ztable ( &aux2     );
        fini_3level_ztable ( &gug      );
        fini_1level_ztable ( &gu       );
  
  
#ifdef HAVE_OPENMP
        omp_set_lock(&writelock);
#endif
        for ( int i = 0; i < T * ncorr; i++ ) corr_accum[0][i] += corr[0][i];
  
#ifdef HAVE_OPENMP
        omp_unset_lock(&writelock);
#endif
  
        fini_2level_ztable ( &corr );
  
#ifdef HAVE_OPENMP
}  /* end of parallel region */
  
        omp_destroy_lock(&writelock);
#endif
  
        double _Complex ** corr_out = init_2level_ztable ( T_global, ncorr );
        if ( corr_out == NULL ) {
          fprintf ( stderr, "[mixing_probe_src] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
          EXIT(2);
        }
  
#ifdef HAVE_MPI
        /***************************************************************************
         * reduce within timeslice
         ***************************************************************************/
        int nitem = 2 * T * ncorr;
        double * buffer = init_1level_dtable ( nitem );
  
        exitstatus = MPI_Allreduce( (double*)(corr_accum[0]), buffer, nitem, MPI_DOUBLE, MPI_SUM, g_ts_comm );
  
        if(exitstatus != MPI_SUCCESS) {
          fprintf(stderr, "[mixing_probe_src] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(4);
        }
  
        /***************************************************************************
         * gather within time ray
         ***************************************************************************/
        exitstatus = MPI_Gather ( buffer, nitem, MPI_DOUBLE, (double*)(corr_out[0]), nitem, MPI_DOUBLE, 0, g_tr_comm);
  
        if(exitstatus != MPI_SUCCESS) {
          fprintf(stderr, "[mixing_probe_src] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(4);
        }
  
        fini_1level_dtable ( &buffer );
#else
        memcpy ( corr_out[0], corr_accum[0], T * ncorr * sizeof ( double _Complex ) );
#endif
  
        fini_2level_ztable ( &corr_accum );
  
  
        /***************************************************************************
         * write to file
         ***************************************************************************/
        if ( io_proc == 2 ) {
          char const diag_list[4][4] = { "b", "d", "m1", "m2" };
          char const op_c_list[4][2] = { "s", "p", "v", "a" };
          char const op_f_list[4][12] = { "g5", "m", "g5", "id" };

          char gflow_tag[100];
          sprintf ( gflow_tag, "gf%6.4f_%d", gflow_dt, gflow_steps[istep] );
  
          double _Complex * buffer = init_1level_ztable ( T_global );
          if ( buffer == NULL ) {
            fprintf ( stderr, "[mixing_probe_src] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
  
          for ( int iop = 0; iop < 2; iop++ ) {
            for ( int idiag = 0; idiag < 2; idiag++ ) {
              for ( int ic = 0; ic < 4; ic++ ) {
                int const corr_id =  4 * ( 2 * iop + idiag ) + ic;
                /* copy data */
                for ( int it = 0; it < T_global; it++ ) {
                  buffer[it] = corr_out[it][corr_id];
                }
                char tag[400];
                sprintf ( tag, "/%s/fl_%c/op_%s/d_%s/c_%s", gflow_tag, flavor_tag[iflavor], op_f_list[iop], diag_list[idiag], op_c_list[ic] );
                exitstatus = write_aff_contraction ( buffer, affw, NULL, tag, T_global, "complex" );
  
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[mixing_probe_src] Error from write_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(4);
                }
  
              }
            }
          }
  
          {
            for ( int i = 0; i < 2; i++ ) {
  
              int const corr_id = 16 + i;
              for ( int it = 0; it < T_global; it++ ) {
                buffer[it] = corr_out[it][corr_id];
              }
              char tag[400];
              sprintf ( tag, "/%s/fl_%c/op_%s/d_%s/c_%s", gflow_tag, flavor_tag[iflavor], op_f_list[2+i], diag_list[2], op_f_list[2+i] );
              exitstatus = write_aff_contraction ( buffer, affw, NULL, tag, T_global, "complex" );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[mixing_probe_src] Error from write_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(4);
              }
            }
          }
  
          {
            int const corr_id = 18;
            for ( int it = 0; it < T_global; it++ ) {
              buffer[it] = corr_out[it][corr_id];
            }
            char tag[400];
            sprintf ( tag, "/%s/fl_%c/op_%s/d_%s/c_%s", gflow_tag, flavor_tag[iflavor], op_f_list[2], diag_list[3], op_f_list[2] );
            exitstatus = write_aff_contraction ( buffer, affw, NULL, tag, T_global, "complex" );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[mixing_probe_src] Error from write_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(4);
            }
          }
  
          fini_1level_ztable ( &buffer );
  
        }  /* end of if io_proc == 2 */
  
        fini_2level_ztable ( &corr_out );
  
      }  /* end of loop on flavor */

    }  /* end of loop on gflow steps */

#ifdef HAVE_LHPC_AFF
    /***************************************************************************
     * I/O process id 2 closes its AFF writer
     ***************************************************************************/
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[mixing_probe_src] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    /***************************************************************************
     * free propagator fields
     ***************************************************************************/
    fini_3level_dtable ( &propagator );
    fini_3level_dtable ( &propagator_snk_smeared );

  }  /* end of loop on source locations */

#if _TEST
  fclose ( ofs );
#endif

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  flow_fwd_gauge_spinor_field ( NULL, NULL, 0, 0, 0., 0, 0 );


  fini_4level_ztable ( &loop );
  fini_2level_dtable ( &scalar_field );

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );
  if ( gauge_field_smeared    != NULL ) free ( gauge_field_smeared );
  if ( gflow_gauge_field      != NULL ) free ( gflow_gauge_field );

  /* free clover matrix terms */
  fini_clover ( );

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
  show_time ( &start_time, &end_time, "mixing_probe_src", "runtime", g_cart_id == 0 );

  return(0);

}
