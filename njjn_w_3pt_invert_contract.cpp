/***************************************************************************
 *
 * njjn_w_3pt_invert_contract
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

#define _OP_ID_UP 0
#define _OP_ID_DN 1

using namespace cvc;

/* typedef int ( * reduction_operation ) (double**, double*, fermion_propagator_type*, unsigned int ); */

typedef int ( * reduction_operation ) (double**, fermion_propagator_type*, fermion_propagator_type*, fermion_propagator_type*, unsigned int);


/***************************************************************************
 * 
 ***************************************************************************/
static inline int reduce_project_write ( double ** vx, double *** vp, fermion_propagator_type * fa, fermion_propagator_type * fb, fermion_propagator_type * fc, reduction_operation reduce,
    struct AffWriter_s *affw, char * tag, int (*momentum_list)[3], int momentum_number, int const nd, unsigned int const N, int const io_proc ) {

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

#if defined HAVE_LHPC_AFF
  /* write to AFF file */
  exitstatus = contract_vn_write_aff ( vp, nd, (struct AffWriter_s *)affw, tag, momentum_list, momentum_number, io_proc );
#endif
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_write for tag %s, status was %d %s %d\n", tag, exitstatus, __FILE__, __LINE__ );
    return( 3 );
  }

  return ( 0 );

}  /* end of reduce_project_write */


/***************************************************************************
 * 
 ***************************************************************************/
static inline int project_write ( double ** vx, double *** vp, struct AffWriter_s *affw, char * tag, int (*momentum_list)[3], int momentum_number, int const nd, int const io_proc ) {

  int exitstatus;

  /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
  exitstatus = contract_vn_momentum_projection ( vp, vx, nd, momentum_list, momentum_number);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return( 2 );
  }

#if defined HAVE_LHPC_AFF
  /* write to AFF file */
  exitstatus = contract_vn_write_aff ( vp, nd, (struct AffWriter_s *)affw, tag, momentum_list, momentum_number, io_proc );
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
  
  const char outfile_prefix[] = "njjn_w";

  const char flavor_tag[4] = { 'u', 'd', 's', 'c' };

  const int sequential_gamma_sets = 2;
  int const sequential_gamma_num[2] = {4, 4 };
  int const sequential_gamma_id[2][4] = {
    { 0,  1,  2,  3 },
    { 6,  7,  8,  9 } };

  char const sequential_gamma_tag[4][3] = { "vv", "aa" };

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
  double *gauge_field_smeared = NULL;
  struct timeval ta, tb, start_time, end_time;

  int const    gamma_f1_number                           = 1;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 };

  int sink_location_number = 0;

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char aff_tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 's':
      sink_location_number = atoi ( optarg );
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

  fprintf(stdout, "# [njjn_w_3pt_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [njjn_w_3pt_invert_contract] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [njjn_w_3pt_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [njjn_w_3pt_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[njjn_w_3pt_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [njjn_w_3pt_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [njjn_w_3pt_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[njjn_w_3pt_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[njjn_w_3pt_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[njjn_w_3pt_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [njjn_w_3pt_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

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
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   * init srand
   ***************************************************************************/
  srand( g_seed );

  /***************************************************************************
   ***************************************************************************
   **
   ** Part I
   **
   ** prepare stochastic sources for W-type sequential sources and propagators
   **
   ***************************************************************************
   ***************************************************************************/

#ifdef _SMEAR_QUDA
  /***************************************************************************
   * dummy solve, just to have original gauge field up on device,
   * for subsequent APE smearing
   ***************************************************************************/

  double ** spinor_work = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
  if ( spinor_work == NULL ) {
    fprintf ( stderr, "[njjn_w_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
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
    fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(12);
  }
#endif  /* of if _SMEAR_QUDA */

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
        fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
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
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) 
  {

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
      fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
      fprintf(stdout, "# [njjn_w_3pt_invert_contract] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
#else
      fprintf(stderr, "[njjn_w_3pt_invert_contract] Error, no outupt variant selected %s %d\n",  __FILE__, __LINE__);
      EXIT(15);
#endif
    }  /* end of if io_proc == 2 */

    /* up and down quark propagator with source smearing */
    double *** propagator = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
    if( propagator == NULL ) {
      fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     ***************************************************************************
     **
     ** Part IIa
     **
     ** point-to-all propagators with source at coherent source
     **
     ***************************************************************************
     ***************************************************************************/
        
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
      /*                                     output field         src coords flavor type  src smear  sink smear gauge field for smearing,  for residual check ...
       */
      exitstatus = point_source_propagator ( propagator[iflavor], gsx,       iflavor,     1,         0,         gauge_field_smeared,       check_propagator_residual, gauge_field_with_phase, lmzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(12);
      }
      
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "njjn_w_3pt_invert_contract", "forward-light-smear-invert-check", g_cart_id == 0 );

    }  /* end of loop on flavor */

    /***************************************************************************
     ***************************************************************************
     **
     ** Part IIb
     **
     ***************************************************************************
     ***************************************************************************/
    for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ )
    {

      for( int isink_location = 0; isink_location < sink_location_number; isink_location++ )
      {

        int gxsink[4] = {
          ( gsx[0] + g_sequential_source_timeslice_list[idt] + T_global ) % T_global,
          rand() % LX_global,
          rand() % LY_global,
          rand() % LZ_global
        };

        int xsink[4], sink_proc_id = -1;
        exitstatus = get_point_source_info (gxsink, xsink, &sink_proc_id);
        if( exitstatus != 0 ) {
          fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(123);
        }

        /* up and down quark propagator with source smearing */
        double *** sink_propagator = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
        if( sink_propagator == NULL ) {
          fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(123);
        }

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
          /*                                     output field         src coords flavor type  src smear  sink smear gauge field for smearing,  for residual check ...
           */
          exitstatus = point_source_propagator ( sink_propagator[iflavor], gsx,       iflavor,     1,         0,         gauge_field_smeared,       check_propagator_residual, gauge_field_with_phase, lmzz );
          if(exitstatus != 0) {
            fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(12);
          }
        
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "njjn_w_3pt_invert_contract", "forward-light-invert-check", g_cart_id == 0 );

        }  /* end of loop on flavor */

        double *** v = init_3level_dtable ( 2, 12, _GSI(VOLUME ) );

        for ( int igamma = 0; igamma < sequential_gamma_sets ; igamma++ )
        {

          /***************************************************************************
           * vx holds the x-dependent nucleon-nucleon spin propagator,
           * i.e. a 4x4 complex matrix per space time point
           ***************************************************************************/
          double ** vx_aux = init_2level_dtable ( VOLUME, 32 );
          if ( vx_aux == NULL ) {
            fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
            EXIT(47);
          }
          double ** vx = init_2level_dtable ( VOLUME, 32 );
          if ( vx == NULL ) {
            fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
            EXIT(47);
          }

          /***************************************************************************
           * vp holds the nucleon-nucleon spin propagator in momentum space,
           * i.e. the momentum projected vx
           ***************************************************************************/
          double *** vp = init_3level_dtable ( T, g_sink_momentum_number, 32 );
          if ( vp == NULL ) {
            fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(47);
          }

          memset ( vx[0], 0, 32*VOLUME *sizeof(double) );

          for ( int igs = 0; igs < sequential_gamma_num[igamma]; igs++ )
          {
            int const gamma_id = sequential_gamma_id[igamma][igs];

            for ( int iflavor = 0; iflavor < 2; iflavor++ )
            {

              for ( int isrc = 0; isrc < 12; isrc++ )
              {
                for ( int isnk = 0; isnk < 12; isnk++ )
                  {
#pragma omp parallel
{
                  double sp[24];
                  complex w;
#pragma omp for
                  for ( unsigned int ix = 0; ix < VOLUME; ix++ ) 
                  {
                    double * const _pi = propagator[iflavor][isrc]        + _GSI(ix);
                    double * const _pf = sink_propagator[1-iflavor][isnk] + _GSI(ix);
                    double * const _v  = v[iflavor][isrc] + _GSI(ix);

                    _fv_eq_gamma_ti_fv ( sp, gamma_id, _pi );

                    _fv_ti_eq_g5 ( sp );

                    _co_eq_fv_dag_ti_fv ( &w, _pf, sp );

                    _v[ 2 * isnk + 0] = w.re;
                    _v[ 2 * isnk + 1] = w.im;

                  }  /* end of loop on volume */

}  /* end of parallel region */

                }  /* end of loop on sink spin-color */

                /* multiply by g5 from right;
                 * from using conjugate sink propagator
                 */

                for ( int isnk = 0; isnk < 12; isnk++ )
                {
                  g5_phi ( v[iflavor][isrc], VOLUME );
                }

              }  /* end of loop on source spin-color */
              
            }  /* end of loop on iflavor */

            fermion_propagator_type * omega  = create_fp_field ( VOLUME );

            for ( int iflavor = 0; iflavor < 2; iflavor++ )
            {

              assign_fermion_propagator_from_spinor_field ( omega, v[iflavor], VOLUME);

              memset ( vx_aux[0], 0, 32*VOLUME *sizeof(double) );

              /* contraction */
              exitstatus = contract_v5 ( vx_aux, omega, delta, omega, VOLUME );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT( 1 );
              }
#pragma omp parallel for
              for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
              {
                vx[0][ix] += vx_aux[0][ix];
              }

              memset ( vx_aux[0], 0, 32*VOLUME *sizeof(double) );

              /* contraction */
              exitstatus = contract_v6 ( vx_aux, omega, delta, omega, VOLUME );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT( 1 );
              }

#pragma omp parallel for
             for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
               {
                vx[0][ix] += vx_aux[0][ix];
              }

            free_fp_field ( &omega  );

          }  /* end of loop on sequential_gamma_num */

          char aff_tag[200];
          sprintf ( aff_tag_prefix, "/%s/w%c%c-f%c-w%c%c/dt%d/Gc_%s/sample%d/qb",
              correlator_tag,
              flavor_tag[iflavor], flavor_tag[iflavor], flavor_tag[1-iflavor],
              flavor_tag[iflavor], flavor_tag[iflavor],
              g_sequential_source_timeslice[idt],
              sequential_gamma_tag[igamma],
              isink_location);

          int momentum[3] = {0,0,0};

          exitstatus = project_write ( vx, vp, affw, aff_tag, &momentum, 1, 16, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_w_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(48);
          }

          fini_2level_dtable ( &vx_aux );
          fini_2level_dtable ( &vx );
          fini_3level_dtable ( &vp );

        }  /* end of loop on sequential_gamma_sets */

        fini_3level_dtable ( &sink_propagator );

      }  /* end of loop on sink locations */

    }  /* end of loop on source - sink -timeseparations */

    free_fp_field ( delta[0] );
    free_fp_field ( delta[1] );


#ifdef HAVE_LHPC_AFF
    /***************************************************************************
     * I/O process id 2 closes its AFF writer
     ***************************************************************************/
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_w_3pt_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  }  /* end of loop on source locations */

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );
  if ( gauge_field_smeared    != NULL ) free ( gauge_field_smeared );

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
  show_time ( &start_time, &end_time, "njjn_w_3pt_invert_contract", "runtime", g_cart_id == 0 );

  return(0);

}
