/***************************************************************************
 *
 * njjn_3pt_invert_contract
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
#include "ilinalg.h"
#include "icontract.h"
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

#ifndef _USE_LOOP
#define _USE_LOOP 1
#endif

#ifndef _NJJN_TEST
#define _NJJN_TEST 0
#endif

#define _P_UUUU_P  1
#define _P_DDDD_P  0

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
  
  const char outfile_prefix[] = "njjn_3pt";

  const char flavor_tag[4] = { 'u', 'd', 's', 'c' };

  const int sequential_gamma_sets = 4;
  int const sequential_gamma_num[4] = {4, 4, 1, 1};
  /* int const sequential_gamma_id[4][4] = {
    { 0,  1,  2,  3 },
    { 6,  7,  8,  9 },
    { 4, -1, -1, -1 },
    { 5, -1, -1, -1 } }; */

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


  /* char const gamma_id_to_ascii[16][10] = {
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
  }; */

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

  /* int read_loop_field    = 1; */
  
#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char aff_tag[400];
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

  fprintf(stdout, "# [njjn_3pt_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [njjn_3pt_invert_contract] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [njjn_3pt_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [njjn_3pt_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[njjn_3pt_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [njjn_3pt_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [njjn_3pt_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[njjn_3pt_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[njjn_3pt_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[njjn_3pt_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize the clover term, 
   * lmzz and lmzzinv
   *
   *   mzz = space-time diagonal part of the Dirac matrix
   *   l   = light quark mass
   *
   *   needed for residuum check
   ***************************************************************************/
  exitstatus = init_clover ( &lmzz, &lmzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[njjn_3pt_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [njjn_3pt_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );
  size_t const sizeof_spinor_field_timeslice = _GSI( VOL3 ) * sizeof( double );

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   ***************************************************************************
   **
   ** read loop field from lime file
   **
   ***************************************************************************
   ***************************************************************************/

  double _Complex *** loop = NULL;
  loop = init_3level_ztable ( VOLUME, 12, 12 );
  if ( loop  == NULL ) {
    fprintf ( stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

#if _USE_LOOP
  sprintf( filename, "loop.up.c%d.N%d.lime", Nconf, g_nsample );

  if ( io_proc == 2 && g_verbose > 0 ) {
    fprintf ( stdout, "# [njjn_3pt_invert_contract] reading loop field from file %s %s %d\n", filename,  __FILE__, __LINE__ );
  }

  exitstatus = read_lime_contraction ( (double*)(loop[0][0]), filename, 144, 0 );
  if ( exitstatus != 0  ) {
    fprintf ( stderr, "[njjn_3pt_invert_contract] Error read_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(12);
  }
#endif

#ifdef _SMEAR_QUDA
  /***************************************************************************
   ***************************************************************************
   **
   ** dummy solve
   ** just to have original gauge field up on device,
   ** for subsequent APE smearing
   **
   ***************************************************************************
   ***************************************************************************/
  {
    double ** spinor_work = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
    if ( spinor_work == NULL ) {
      fprintf ( stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
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
      fprintf(stderr, "[njjn_3pt_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(12);
    }
    fini_2level_dtable ( &spinor_work );
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
        fprintf(stderr, "[njjn_3pt_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
#ifndef _SMEAR_QUDA 
    }  /* end of if N_ape > 0 */
#endif
  }  /* end of if N_Jacobi > 0 */

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


  /***************************************************************************
   ***************************************************************************
   **
   ** loop on source locations
   **
   **   each source location is given by 4-coordinates in
   **   global variable
   **   g_source_coords_list[count][0-3] for t,x,y,z
   **
   ***************************************************************************
   ***************************************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

    /***************************************************************************
     * determine source coordinates,
     * find out, if source_location is in this process
     *
     * gsx = global source coordinates
     *
     * sx  = MPI-local source coordinates (valid only for MPI
     *       process, which has the source point in its sub-lattice)
     *
     * source_proc_id = MPI process id, which has the source point
     ***************************************************************************/

    int const gsx[4] = {
        ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global,
        ( g_source_coords_list[isource_location][1] + LX_global ) % LX_global,
        ( g_source_coords_list[isource_location][2] + LY_global ) % LY_global,
        ( g_source_coords_list[isource_location][3] + LZ_global ) % LZ_global };

    int sx[4], source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[njjn_3pt_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     * open output file reader
     * we use the AFF format here
     * https://github.com/usqcd-software/aff
     *
     * one data file per source position
     ***************************************************************************/
#if defined HAVE_LHPC_AFF
    /***************************************************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     *
     * outfile_prefix is set above
     ***************************************************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [njjn_3pt_invert_contract] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_3pt_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
#else
      fprintf(stderr, "[njjn_3pt_invert_contract] Error, no outupt variant selected %s %d\n",  __FILE__, __LINE__);
      EXIT(15);
#endif
    }  /* end of if io_proc == 2 */

    /***************************************************************************
     * up and down quark propagator
     *
     * allocate memory for the fields
     *
     * the layout is:
     *
     * propagator[i][k][xx]
     * i = 0, 1 = flavor index; we need up and down
     * k = 0, ..., 11 = spin-color index on SOURCE side; color runs faster
     * xx = 0,..., 24 * VOLUME
     *  site-spin-color-real/imag index on sink side
     *  indices run - slowest to fastest - site, spin,color,re/im
     *
     *  for point-to-all up propagator U we thus have
     *  Re ( U(x,y)^{alpha,beta}_{a,b} ) = propatator[0][3*beta+b][ 2 * ( 12 * x + 3 * alpha + a ) + 0 ]
     *  Im ( U(x,y)^{alpha,beta}_{a,b} ) = propatator[0][3*beta+b][ 2 * ( 12 * x + 3 * alpha + a ) + 1 ]
     *
     *  alpha, beta = spin index
     *  a, b        = color index
     *  x = lattice site
     *
     *  ------------------------------------------------------
     *  In genral init_Xlevel_dtable ( n1,...,nX )
     *  genrates an X-dim array of size n1 x n2 x ... nX
     *  cf. e.g. table_init_d.h
     *
     ***************************************************************************/
    double *** propagator = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
    if( propagator == NULL ) {
      fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     *
     * produce point-to-all propagators for both quark flavors
     *
     ***************************************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ ) 
    {

      gettimeofday ( &ta, (struct timezone *)NULL );

      /***********************************************************
       * flavor-type point-to-all propagator
       *
       * WITH SOURCE smearing
       * 
       * WITHOUT SINK smearing
       *
       * NOTE: quark flavor is controlled by value of iflavor
       ***********************************************************/
      /*                                     output field         src coords flavor type  src smear  sink smear gauge field for smearing,  for residual check ...  */
      exitstatus = point_source_propagator ( propagator[iflavor], gsx,       iflavor,     1,         0,         gauge_field_smeared,       check_propagator_residual, gauge_field_with_phase, lmzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[njjn_3pt_invert_contract] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(12);
      }
      
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "njjn_3pt_invert_contract", "forward-light-smear-invert-check", g_cart_id == 0 );

      /***********************************************************
       * optionally write the propagator to disc
       *
       * we use the standard lime format here
       * https://github.com/usqcd-software/c-lime
       ***********************************************************/
      if ( g_write_propagator ) {
        /* each spin-color component into a separate file */
        for ( int i = 0; i < 12; i++ ) {
          sprintf ( filename, "propagator.%c.c%d.t%dx%dy%dz%d.sc%d.inverted", flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] , i );

          if ( ( exitstatus = write_propagator( propagator[iflavor][i], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[njjn_3pt_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
      }

    }  /* end of loop on flavor */

    /***************************************************************************
     * propagator is SOURCE smeared
     *
     * here we need the SOURCE & SINK smeared propagator
     ***************************************************************************/
    double *** propagator_snk_smeared = init_3level_dtable ( 2, 12, _GSI(VOLUME) );
    if ( propagator_snk_smeared == NULL ) {
      fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable, %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    for ( int iflavor = 0; iflavor < 2; iflavor++ ) {
      for ( int i =0; i < 12; i++ ) {
        memcpy ( propagator_snk_smeared[iflavor][i], propagator[iflavor][i], sizeof_spinor_field );
        exitstatus = Jacobi_Smearing ( gauge_field_smeared, propagator_snk_smeared[iflavor][i], N_Jacobi, kappa_Jacobi );
   
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[njjn_3pt_invert_contract] Error from Jacobi_Smearing status %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(47);
        }
      }
    }

    /***************************************************************************
     ***************************************************************************
     **
     ** contractions for baryon 2pts
     **
     ** with point-to-all propagator 
     **
     ** stand-alone part, for checks, we do not need these
     ** results later on
     **
     ***************************************************************************
     ***************************************************************************/

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

    /***************************************************************************
     * loop on flavor combinations
     ***************************************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ ) {

      gettimeofday ( &ta, (struct timezone *)NULL );

      /***************************************************************************
       * vx holds the x-dependent nucleon-nucleon spin propagator,
       * i.e. a 4x4 complex matrix per space time point
       ***************************************************************************/
      double ** vx = init_2level_dtable ( VOLUME, 32 );
      if ( vx == NULL ) {
        fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }

      /***************************************************************************
       * vp holds the nucleon-nucleon spin propagator in momentum space,
       * i.e. the momentum projected vx
       ***************************************************************************/
      double *** vp = init_3level_dtable ( T, g_source_momentum_number, 32 );
      if ( vp == NULL ) {
        fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      /***************************************************************************
       *
       * [ X^T Xb X ] - [ Xb^+ X^* X^+ ]
       *
       * so either 
       *
       * [u^T G_f d ] u - [ dbar G_i ubar^T ] ubar
       *
       * or
       *
       * [d^T G_f u ] d - [ ubar G_i dbar^T ] dbar
       *
       ***************************************************************************/

      char  aff_tag_prefix[200];
      sprintf ( aff_tag_prefix, "/N-N/%c%c%c/T%d_X%d_Y%d_Z%d", flavor_tag[iflavor], flavor_tag[1-iflavor], flavor_tag[iflavor], gsx[0], gsx[1], gsx[2], gsx[3] );
       
      /***************************************************************************
       * fill the fermion propagator fp with the 12 spinor fields
       * in propagator of flavor X
       ***************************************************************************/
      assign_fermion_propagator_from_spinor_field ( fp, propagator_snk_smeared[iflavor], VOLUME);

      /***************************************************************************
       * fill fp2 with 12 spinor fields from propagator of flavor Xb
       ***************************************************************************/
      assign_fermion_propagator_from_spinor_field ( fp2, propagator_snk_smeared[1-iflavor], VOLUME);

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
        sprintf(aff_tag, "%s/Gi_%s/Gf_%s/t1", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

        exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp, contract_v5, affw, aff_tag, g_source_momentum_list, g_source_momentum_number, 16, VOLUME, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[njjn_3pt_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(48);
        }

        /***************************************************************************
         * diagram n2
         ***************************************************************************/
        sprintf(aff_tag, "%s/Gi_%s/Gf_%s/t2", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

        exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp, contract_v6, affw, aff_tag, g_source_momentum_list, g_source_momentum_number, 16, VOLUME, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[njjn_3pt_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(48);
        }

      }}  /* end of loop on Dirac Gamma structures */

      fini_2level_dtable ( &vx );
      fini_3level_dtable ( &vp );

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "njjn_3pt_invert_contract", "n1-n2-reduce-project-write", g_cart_id == 0 );

    }  /* end of loop on flavor */
    
    free_fp_field ( &fp  );
    free_fp_field ( &fp2 );
    free_fp_field ( &fp3 );

#if _P_UUUU_P 
    /***************************************************************************
     ***************************************************************************
     **
     ** construction of sequential source and sequential propagator
     **
     ** reduce to 3-point function
     **
     ** B and D type diagram for 
     **     p-uuuu-p (iflavor = 0 / up ) and
     **     n-dddd-n (iflavor = 1 / dn )
     **
     ***************************************************************************
     ***************************************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ )
    {

      fermion_propagator_type * fup  = create_fp_field ( VOLUME );
      fermion_propagator_type * fdn  = create_fp_field ( VOLUME );

      /* takes the 12 spin-color components of propagator[iflavor]
       * and makes a propagator matrix per lattice site
       */
      assign_fermion_propagator_from_spinor_field ( fup, propagator_snk_smeared[  iflavor], VOLUME);
      assign_fermion_propagator_from_spinor_field ( fdn, propagator_snk_smeared[1-iflavor], VOLUME);

      /***************************************************************************
       * build dn <- Gamma_f dn Gamma_i
       * Gamma_f/i have to be fixed, only one choice
       * Gamma_f/i = Cg5 = 14, sign +1 
       *
       * all IN-PLACE
       ***************************************************************************/

#if _NJJN_TEST
      /* BEGIN OF POINT CHECK */
      FILE * ofs = fopen( "test", "w" );
      unsigned int const test_site = 0;

      printf_fp ( fup[test_site], "up", ofs );
      printf_fp ( fdn[test_site], "dn", ofs );
#endif

      /* fdn <- Gamma_f fdn */
      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fdn, gamma_f1_list[0], fdn, VOLUME );

#if _NJJN_TEST
      printf_fp ( fdn[test_site], "Gdn", ofs);
#endif

      /* fdn <- fdn Gamma_i */
      fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fdn, gamma_f1_list[0], fdn, VOLUME );

#if _NJJN_TEST
      printf_fp ( fdn[test_site], "GdnG", ofs );
#endif

      /* fdn <- fdn sign_f sign_i */
      /* fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fdn, fdn, -gamma_f1_sign[0]*gamma_f1_sign[0], VOLUME ); */

      /***************************************************************************
       * build 
       *
       *   epsilon epsilon D^T U and store in fseq
       *
       * HERE THE NUCLEON INTERPOLATOR MATRICES MUST BE ADDED
       * D -> Gamma D Gamma,
       * cf. N-N 2pt above
       *
       * using _fp... functions from 
       * linalg/fp_linalg_inline.h
       ***************************************************************************/
      fermion_propagator_type * fseq  = create_fp_field ( VOLUME );

#pragma omp parallel
{
      /* auxilliary propagator matrix */
      fermion_propagator_type faux, faux2;
      create_fp ( &faux );
      create_fp ( &faux2 );

#pragma omp for
#if _NJJN_TEST
      for ( unsigned int ix = test_site; ix <=  test_site; ix ++ ) 
#else
      for ( unsigned int ix = 0; ix < VOLUME; ix ++ )
#endif
      {

        fermion_propagator_type _fseq = fseq[ix];

        /***************************************************************************
         * Case 1
         *
         * _fseq = ( 1 + g0) / 2 )  [ epsilon epsilon D^T U  ]^{T_s,*} 
         *
         * USING THAT P = ( 1 + g0 ) / 2 is real and symmetric
         ***************************************************************************/

        /* faux2 = epsilon epsilon fdn^T fup  */
        _fp_eq_fp_eps_contract13_fp( faux2, fdn[ix], fup[ix] );

        // printf_fp ( faux2, "d1u3", ofs );

        /* faux = faux2^+ */
        /* _fp_eq_fp_adjoint ( faux, faux2 ); */

        /* faux = ( faux2^T_s )^* spin-transposed and conjugate */
        _fp_eq_fp_spin_transposed ( faux, faux2 );
        _fp_eq_fp_conj ( faux, faux );

        // printf_fp ( faux, "d1u3tsc", ofs );

        /* faux2 = g0 faux = g0 _fseq^+ */
        _fp_eq_gamma_ti_fp ( faux2, 0, faux );

        // printf_fp ( faux2, "g0_d1u3tsc", ofs );

        /* faux2 <- faux2 + faux = ( 1 + g0 ) faux2^Ts* */
        _fp_pl_eq_fp ( faux2, faux );

        /* faux2 <- faux2 * 0.5 */
        _fp_ti_eq_re ( faux2, 0.5 );

#if _NJJN_TEST
        if ( test_site == ix ) printf_fp ( faux2, "C1", ofs );
#endif

        /* _fseq <- _fseq + faux2 */
        _fp_pl_eq_fp ( _fseq, faux2 );

        /***************************************************************************
         * Case 2
         *
         * _fseq = ( 1 + g0) / 2 )  [ epsilon epsilon D^T U  ]^{T_s,*} 
         *
         * USING THAT P = ( 1 + g0 ) / 2 is real and symmetric
         ***************************************************************************/

        /* faux2 = epsilon epsilon fup fdn^T  */
        _fp_eq_fp_eps_contract24_fp( faux2, fup[ix], fdn[ix] );

        /* faux2 = g0 faux */
        _fp_eq_gamma_ti_fp ( faux, 0, faux2 );

        /* faux <- faux + faux2 */
        _fp_pl_eq_fp ( faux, faux2 );

        /* faux <- faux * 0.5 */
        _fp_ti_eq_re ( faux, 0.5 );

        /* faux = ( faux2^T_s )^* spin-transposed and conjugate */
        _fp_eq_fp_spin_transposed ( faux2, faux );
        _fp_eq_fp_conj ( faux2, faux2 );

#if _NJJN_TEST
        if ( test_site == ix ) printf_fp ( faux2, "C2", ofs );
#endif

        /* _fseq <- _fseq + faux2 */
        _fp_pl_eq_fp ( _fseq, faux2 );


        /***************************************************************************
         * Case 3
         *
         * _fseq = [ Tr_s[ epsilon epsilon U D^T ] ( 1 + g0) / 2 ) ]^{T_s,*}
         *
         * USING THAT P = ( 1 + g0 ) / 2 is real and symmetric
         ***************************************************************************/

        /* faux2 = epsilon epsilon fdn^T fup */
        _fp_eq_fp_eps_contract13_fp( faux,  fdn[ix], fup[ix] );

        /* faux2 = delta_s,s' Tr_spin( faux ) */
        _fp_eq_spintrace_fp ( faux2, faux );

        /* faux = ( faux2^T_s )^* spin-transposed and conjugate */
        _fp_eq_fp_spin_transposed ( faux, faux2 );
        _fp_eq_fp_conj ( faux, faux );

        /* faux2 = g0 faux */
        _fp_eq_gamma_ti_fp ( faux2, 0, faux );

        /* faux2 <- faux2 + faux */
        _fp_pl_eq_fp ( faux2, faux );

        /* faux <- faux * 0.5 */
        _fp_ti_eq_re ( faux2, 0.5 );

#if _NJJN_TEST
        if ( test_site == ix ) printf_fp ( faux2, "C3", ofs );
#endif

        /* _fseq <- _fseq + faux2 */
        _fp_pl_eq_fp ( _fseq, faux2 );

        /***************************************************************************
         * Case 4
         *
         * _fseq = [ Tr_s[ epsilon epsilon ( 1 + g0 ) / 2 U ] D^T ]^{T_s,*}
         *
         * USING THAT P = ( 1 + g0 ) / 2 is real and symmetric
         ***************************************************************************/

        /* faux2 = g0 fup */
        _fp_eq_gamma_ti_fp ( faux2, 0, fup[ix] );

        /* faux <- faux + fup */
        _fp_pl_eq_fp ( faux2, fup[ix] );

        /* faux <- faux * 0.5 */
        _fp_ti_eq_re ( faux2, 0.5 );

        /* faux = delta_s,s' Tr_spin( fup ) */
        _fp_eq_spintrace_fp ( faux, faux2 );

        /* faux = epsilon epsilon faux fdn^T */
        _fp_eq_fp_eps_contract24_fp( faux2,  faux, fdn[ix] );

        /* faux = ( faux2^T_s )^* spin-transposed and conjugate */
        _fp_eq_fp_spin_transposed ( faux, faux2 );
        _fp_eq_fp_conj ( faux, faux );

#if _NJJN_TEST
        if ( test_site == ix ) printf_fp ( faux, "C4", ofs );
#endif

        /* _fseq <- _fseq + faux */
        _fp_pl_eq_fp ( _fseq, faux );
      
#if _NJJN_TEST
        if ( test_site == ix ) printf_fp ( _fseq, "seq", ofs );
#endif

      }  /* end of loop on ix */

      free_fp ( &faux  );
      free_fp ( &faux2 );


}  /* end of parallel region */

#if _NJJN_TEST
      fclose ( ofs );
      /* END OF POINT CHECK */
#endif

      free_fp_field ( &fup  );
      free_fp_field ( &fdn  );


      /***************************************************************************
       ***************************************************************************
       **
       ** invert on this source for a specific sequential source timeslice
       **
       ***************************************************************************
       ***************************************************************************/

      double ** sequential_source = init_2level_dtable ( 12, _GSI( VOLUME ) );
      if( sequential_source == NULL ) {
        fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(123);
      }

      /* get back 12 spinor fields form the fermion propagator matrix field */
      exitstatus = assign_spinor_field_from_fermion_propagator ( sequential_source, fseq, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[njjn_3pt_invert_contract] Error from assign_spinor_field_from_fermion_propagaptor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }

      /* free memory of fseq */
      free_fp_field ( &fseq );

      if ( g_write_sequential_source ) {
        for ( int i = 0; i < 12; i++ ) {
          sprintf ( filename, "sequential_source.%c.c%d.t%dx%dy%dz%d.sc%d",
              flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3], i );
          if ( ( exitstatus = write_propagator( sequential_source[i], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[njjn_3pt_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
      }

      /***************************************************************************
       * multiply sequential source with g5
       ***************************************************************************/
      g5_phi( sequential_source[0], 12*VOLUME );

      /* allocate sequential propagator */
      double ** sequential_propagator = init_2level_dtable ( 12, _GSI( VOLUME ) );
      if( sequential_propagator == NULL ) {
        fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(123);
      }

      for ( int itseq = 0; itseq < g_sequential_source_timeslice_number; itseq++ ) 
      {

        int const gtseq =  ( gsx[0] + g_sequential_source_timeslice_list[itseq] + T_global ) % T_global;
        int const have_tseq = ( gtseq / T == g_proc_coords[0] );
        int const tseq = have_tseq ? gtseq % T : -1;
        size_t const offset = _GSI( tseq * VOL3 );

        /***************************************************************************
         *  preform inversion; 
         *
         *  with 1-iflavor, since we have used g5-hermiticity
         *
         *  with wrapper function prepare_propagator_from_source from
         *  prepare_propagator.cpp
         ***************************************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        /***************************************************************************
         * invert
         *
         * use SOURCE smearing, but NO SINK smearing
         *
         * sink-end is at operator insertion
         ***************************************************************************/
        double * sequential_timeslice_source = init_1level_dtable ( _GSI( VOLUME ) );
        if ( sequential_timeslice_source == NULL ) {
          fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(123);
        }

        for ( int i = 0; i < 12; i++ ) {

          memset ( sequential_timeslice_source, 0, sizeof_spinor_field );

          if ( have_tseq ) {
            if ( g_verbose > 1 ) fprintf ( stdout, "# [] proc %4d has g/tseq %3d / %3d   %s %d\n", g_cart_id, gtseq, tseq, __FILE__, __LINE__ );
            memcpy ( sequential_timeslice_source + offset , sequential_source[i] + offset, sizeof_spinor_field_timeslice );
          }

          exitstatus = prepare_propagator_from_source ( &(sequential_propagator[i]), &sequential_timeslice_source, 1, 1-iflavor, 1, 0, gauge_field_smeared,
              check_propagator_residual, gauge_field_with_phase, lmzz, NULL );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[njjn_3pt_invert_contract] Error from prepare_propagator_from_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(123);
          }
        }  /* end of loop on spin color */

        fini_1level_dtable ( &sequential_timeslice_source );

        /***************************************************************************
         * multiply propagator with g5
         ***************************************************************************/
        g5_phi( sequential_propagator[0], 12*VOLUME );

        if ( g_write_sequential_propagator ) {
          for ( int i = 0; i < 12; i++ ) {
            sprintf ( filename, "sequential_source.%c.c%d.t%dx%dy%dz%d.sc%d.inverted",
                flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3], i );

            if ( ( exitstatus = write_propagator( sequential_propagator[i], filename, 0, g_propagator_precision) ) != 0 ) {
              fprintf(stderr, "[njjn_3pt_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }
          }
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "njjn_3pt_invert_contract", "sequential-source-invert-check-smear", g_cart_id == 0 );

        /***************************************************************************
         ***************************************************************************
         **
         ** finalize contraction and output
         **
         ***************************************************************************
         ***************************************************************************/
        for ( int igamma = 0; igamma < sequential_gamma_sets; igamma++ ) {

          gamma_matrix_type gammafive;
          gamma_matrix_set( &gammafive, 5, 1. );  /*  gamma_5 */

          double ** glg_propagator = init_2level_dtable ( 12, _GSI(VOLUME) );
          if ( glg_propagator == NULL ) { 
            fprintf ( stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(123);
          }

          /***************************************************************************
           * need to set the quark flavor of the loop, which here is always iflavor
           * loop_flavor = iflavor
           ***************************************************************************/
          int const loop_flavor = iflavor;

          for ( int seq_source_type = 0; seq_source_type <= 1; seq_source_type++ )
          {

            /***************************************************************************
             * type of seq. prop. used as tag for diagram type
             ***************************************************************************/
            char const sequential_propagator_name = ( seq_source_type == 0 ) ? 'd' : 'b';

            if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_3pt_invert_contract] start seq source type %3d %c    %s %d\n",
                seq_source_type, sequential_propagator_name, __FILE__, __LINE__);

            memset ( glg_propagator[0], 0, 12 * _GSI(VOLUME) * sizeof ( double )  );

            /***************************************************************************
             * prepare propagator product
             *     G_c Loop   G_c propagator or
             *   [ G_c Loop ] G_c propagator
             *
             * here we use propagator, because we need SOURCE smearing,
             * but NO SINK smearing
             ***************************************************************************/
            exitstatus = prepare_sequential_fht_loop_source (
                glg_propagator, loop, propagator[  iflavor],
                sequential_gamma_list[igamma], sequential_gamma_num[igamma],
                NULL, seq_source_type, ( loop_flavor == 0 ? NULL : &gammafive ) ); 
 
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[njjn_3pt_invert_contract] Error from prepare_sequential_fht_loop_source, status was %d %s %d\n",
                  exitstatus, __FILE__, __LINE__ );
              EXIT(123);
            }

            double * contr_p = init_1level_dtable ( 2 * T );
            if ( contr_p == NULL ) {
              fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
              EXIT(3);
            }

            int sink_momentum[3] = { 0, 0, 0 };

            /***************************************************************************
             * using Gamma_i = unit matrix = Gamma_f;
             *
             * seq^+ glg
             ***************************************************************************/
            contract_twopoint_snk_momentum ( contr_p, 4, 4, sequential_propagator, glg_propagator, 4, 3, sink_momentum, 1 ); 
 

            /***************************************************************************
             * write to file
             *
             * seq_source_type  0 = D-type diagram
             * seq_source_type  1 = B-type diagram
             ***************************************************************************/

            if( io_proc > 0 ) {

              double * contr_p_global = init_1level_dtable ( 2 * T_global );
              if ( contr_p_global == NULL ) {
                fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
                EXIT(3);
              }
#ifdef HAVE_MPI
              exitstatus = MPI_Gather( contr_p , 2*T, MPI_DOUBLE, contr_p_global, 2*T, MPI_DOUBLE, 0, g_tr_comm);
              if(exitstatus != MPI_SUCCESS) {
                fprintf(stderr, "[njjn_3pt_invert_contract] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                return(3);
              }

#else
              memcpy ( contr_p_global, contr_p, 2 * T_global * sizeof ( double ) );
#endif

              if ( io_proc == 2 ) {
         
                char correlator_tag[20] = "NA";
                if ( iflavor == 0 ) {
                  strcpy ( correlator_tag, "p-ubGuubGu-p" );
                } else if ( iflavor == 1 ) {
                  strcpy ( correlator_tag,  "n-dbGddbGd-n" );
                }

                char aff_tag_prefix[200], aff_tag[400];

                sprintf ( aff_tag_prefix, "/%s/nsample%d/Gc_%s/tseq%d", correlator_tag, g_nsample,
                    sequential_gamma_tag[igamma], g_sequential_source_timeslice_list[itseq] );

                if ( g_verbose > 2 ) fprintf ( stdout, "# [njjn_3pt_invert_contract] aff_tag_prefix = %s %s %d\n", aff_tag_prefix, __FILE__, __LINE__ );

                /***************************************************************************
                 * we call this by diagram type = loop seq prop. type
                 ***************************************************************************/
                sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/%c", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[0] ], gamma_id_to_Cg_ascii[ gamma_f1_list[0] ],
                    sequential_propagator_name );

                if ( g_verbose > 2 ) fprintf ( stdout, "# [njjn_3pt_invert_contract] aff_tag = %s %s %d\n", aff_tag, __FILE__, __LINE__ );

                exitstatus = write_aff_contraction ( contr_p_global, affw, NULL, aff_tag, T_global, "complex" );

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[njjn_3pt_invert_contract] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(3);
                }
 
              }  /* end of if io_proc == 2 */
        
              fini_1level_dtable ( &contr_p_global );

            }  /* end of if io_proc > 0 */

            fini_1level_dtable ( &contr_p );

          }  /* end of loop on seq source type */
          
          fini_2level_dtable ( &glg_propagator );

        }  /* end of loop on gamma sets */

      }  /* end of loop on source-sink time separations */

      fini_2level_dtable ( &sequential_source );
      fini_2level_dtable ( &sequential_propagator );

    }  /* end of loop on flavor */
#endif   /* of if _P_UUUU_P  */

#if _P_DDDD_P 
    /***************************************************************************
     ***************************************************************************
     **
     ** construction of sequential source and sequential propagator
     **
     ** reduce to 3-point function
     **
     ** B and D type diagram for 
     **     p-dddd-p (iflavor = 0 / up ) and
     **     n-uuuu-n (iflavor = 1 / dn )
     **
     ***************************************************************************
     ***************************************************************************/
    /* for ( int iflavor = 0; iflavor < 2; iflavor++ )  */
    for ( int iflavor = 0; iflavor < 1; iflavor++ ) 
    {

      fermion_propagator_type * fup  = create_fp_field ( VOLUME );
      fermion_propagator_type * fdn  = create_fp_field ( VOLUME );

      /* takes the 12 spin-color components of propagator[iflavor]
       * and makes a propagator matrix per lattice site
       */
      assign_fermion_propagator_from_spinor_field ( fup, propagator_snk_smeared[  iflavor], VOLUME);
      assign_fermion_propagator_from_spinor_field ( fdn, propagator_snk_smeared[1-iflavor], VOLUME);

      /***************************************************************************
       * build dn <- Gamma_f dn Gamma_i
       * Gamma_f/i have to be fixed, only one choice
       * Gamma_f/i = Cg5 = 14, sign +1 
       *
       * all IN-PLACE
       ***************************************************************************/

#if _NJJN_TEST
      /* BEGIN OF POINT CHECK */
      FILE * ofs = fopen( "test2", "w" );
      unsigned int const test_site = 0;

      printf_fp ( fup[test_site], "up", ofs );
      printf_fp ( fdn[test_site], "dn", ofs );
#endif

      /* fdn <- Gamma_f fdn */
      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fdn, gamma_f1_list[0], fdn, VOLUME );

#if _NJJN_TEST
      printf_fp ( fdn[test_site], "Gdn", ofs);
#endif

      /* fdn <- fdn Gamma_i */
      fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fdn, gamma_f1_list[0], fdn, VOLUME );

#if _NJJN_TEST
      printf_fp ( fdn[test_site], "GdnG", ofs );
#endif

      /* fdn <- fdn sign_f sign_i */
      /* fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fdn, fdn, -gamma_f1_sign[0]*gamma_f1_sign[0], VOLUME ); */

      /***************************************************************************
       * build 
       *
       *   epsilon epsilon D^T U and store in fseq
       *
       * HERE THE NUCLEON INTERPOLATOR MATRICES MUST BE ADDED
       * D -> Gamma D Gamma,
       * cf. N-N 2pt above
       *
       * using _fp... functions from 
       * linalg/fp_linalg_inline.h
       ***************************************************************************/
      fermion_propagator_type * fseq  = create_fp_field ( VOLUME );

#pragma omp parallel
{
      /* auxilliary propagator matrix */
      fermion_propagator_type faux, faux2;
      create_fp ( &faux );
      create_fp ( &faux2 );

#pragma omp for
#if _NJJN_TEST
      for ( unsigned int ix = test_site; ix <=  test_site; ix ++ ) 
#else
      for ( unsigned int ix = 0; ix < VOLUME; ix ++ )
#endif
      {

        fermion_propagator_type _fseq = fseq[ix];

        /***************************************************************************
         * Case 1
         *
         * _fseq = ( 1 + g0) / 2 )  [ epsilon epsilon D^T U  ]^{T_s,*} 
         *
         * USING THAT P = ( 1 + g0 ) / 2 is real and symmetric
         ***************************************************************************/

        /* faux2 = epsilon epsilon fdn^T fup  */
        _fp_eq_fp_eps_contract13_fp( faux2, fdn[ix], fup[ix] );

        // printf_fp ( faux2, "d1u3", ofs );

        /* faux = faux2^+ */
        /* _fp_eq_fp_adjoint ( faux, faux2 ); */

        /* faux = ( faux2^T_s )^* spin-transposed and conjugate */
        _fp_eq_fp_spin_transposed ( faux, faux2 );
        _fp_eq_fp_conj ( faux, faux );

        // printf_fp ( faux, "d1u3tsc", ofs );

        /* faux2 = g0 faux = g0 _fseq^+ */
        _fp_eq_gamma_ti_fp ( faux2, 0, faux );

        // printf_fp ( faux2, "g0_d1u3tsc", ofs );

        /* faux2 <- faux2 + faux = ( 1 + g0 ) faux2^Ts* */
        _fp_pl_eq_fp ( faux2, faux );

        /* faux2 <- faux2 * 0.5 */
        _fp_ti_eq_re ( faux2, 0.5 );

#if _NJJN_TEST
        if ( test_site == ix ) printf_fp ( faux2, "C1", ofs );
#endif

        /* _fseq <- _fseq + faux2 */
        _fp_pl_eq_fp ( _fseq, faux2 );

        /***************************************************************************
         * Case 2
         *
         * _fseq = ( 1 + g0) / 2 )  [ epsilon epsilon D^T U  ]^{T_s,*} 
         *
         * USING THAT P = ( 1 + g0 ) / 2 is real and symmetric
         ***************************************************************************/

        /* faux2 = epsilon epsilon fup fdn^T  */
        _fp_eq_fp_eps_contract24_fp( faux2, fup[ix], fdn[ix] );

        /* faux2 = g0 faux */
        _fp_eq_gamma_ti_fp ( faux, 0, faux2 );

        /* faux <- faux + faux2 */
        _fp_pl_eq_fp ( faux, faux2 );

        /* faux <- faux * 0.5 */
        _fp_ti_eq_re ( faux, 0.5 );

        /* faux = ( faux2^T_s )^* spin-transposed and conjugate */
        _fp_eq_fp_spin_transposed ( faux2, faux );
        _fp_eq_fp_conj ( faux2, faux2 );

#if _NJJN_TEST
        if ( test_site == ix ) printf_fp ( faux2, "C2", ofs );
#endif

        /* _fseq <- _fseq + faux2 */
        _fp_pl_eq_fp ( _fseq, faux2 );


        /***************************************************************************
         * Case 3
         *
         * _fseq = [ Tr_s[ epsilon epsilon U D^T ] ( 1 + g0) / 2 ) ]^{T_s,*}
         *
         * USING THAT P = ( 1 + g0 ) / 2 is real and symmetric
         ***************************************************************************/

        /* faux2 = epsilon epsilon fdn^T fup */
        _fp_eq_fp_eps_contract13_fp( faux,  fdn[ix], fup[ix] );

        /* faux2 = delta_s,s' Tr_spin( faux ) */
        _fp_eq_spintrace_fp ( faux2, faux );

        /* faux = ( faux2^T_s )^* spin-transposed and conjugate */
        _fp_eq_fp_spin_transposed ( faux, faux2 );
        _fp_eq_fp_conj ( faux, faux );

        /* faux2 = g0 faux */
        _fp_eq_gamma_ti_fp ( faux2, 0, faux );

        /* faux2 <- faux2 + faux */
        _fp_pl_eq_fp ( faux2, faux );

        /* faux <- faux * 0.5 */
        _fp_ti_eq_re ( faux2, 0.5 );

#if _NJJN_TEST
        if ( test_site == ix ) printf_fp ( faux2, "C3", ofs );
#endif

        /* _fseq <- _fseq + faux2 */
        _fp_pl_eq_fp ( _fseq, faux2 );

        /***************************************************************************
         * Case 4
         *
         * _fseq = [ Tr_s[ epsilon epsilon ( 1 + g0 ) / 2 U ] D^T ]^{T_s,*}
         *
         * USING THAT P = ( 1 + g0 ) / 2 is real and symmetric
         ***************************************************************************/

        /* faux2 = g0 fup */
        _fp_eq_gamma_ti_fp ( faux2, 0, fup[ix] );

        /* faux <- faux + fup */
        _fp_pl_eq_fp ( faux2, fup[ix] );

        /* faux <- faux * 0.5 */
        _fp_ti_eq_re ( faux2, 0.5 );

        /* faux = delta_s,s' Tr_spin( fup ) */
        _fp_eq_spintrace_fp ( faux, faux2 );

        /* faux = epsilon epsilon faux fdn^T */
        _fp_eq_fp_eps_contract24_fp( faux2,  faux, fdn[ix] );

        /* faux = ( faux2^T_s )^* spin-transposed and conjugate */
        _fp_eq_fp_spin_transposed ( faux, faux2 );
        _fp_eq_fp_conj ( faux, faux );

#if _NJJN_TEST
        if ( test_site == ix ) printf_fp ( faux, "C4", ofs );
#endif

        /* _fseq <- _fseq + faux */
        _fp_pl_eq_fp ( _fseq, faux );
      
#if _NJJN_TEST
        if ( test_site == ix ) printf_fp ( _fseq, "seq", ofs );
#endif

      }  /* end of loop on ix */

      free_fp ( &faux  );
      free_fp ( &faux2 );


}  /* end of parallel region */

#if _NJJN_TEST
      fclose ( ofs );
      /* END OF POINT CHECK */
#endif

      free_fp_field ( &fup  );
      free_fp_field ( &fdn  );


      /***************************************************************************
       * invert on this source for a specific sequential source timeslice
       ***************************************************************************/

      double ** sequential_source = init_2level_dtable ( 12, _GSI( VOLUME ) );
      if( sequential_source == NULL ) {
        fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(123);
      }

      /* get back 12 spinor fields form the fermion propagator matrix field */
      exitstatus = assign_spinor_field_from_fermion_propagator ( sequential_source, fseq, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[njjn_3pt_invert_contract] Error from assign_spinor_field_from_fermion_propagaptor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }

      /* free memory of fseq */
      free_fp_field ( &fseq );

      if ( g_write_sequential_source ) {
        for ( int i = 0; i < 12; i++ ) {
          sprintf ( filename, "sequential_source.%c.c%d.t%dx%dy%dz%d.sc%d",
              flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3], i );
          if ( ( exitstatus = write_propagator( sequential_source[i], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[njjn_3pt_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
      }

      /***************************************************************************
       * multiply sequential source with g5
       ***************************************************************************/
      g5_phi( sequential_source[0], 12*VOLUME );

      /* allocate sequential propagator */
      double ** sequential_propagator = init_2level_dtable ( 12, _GSI( VOLUME ) );
      if( sequential_propagator == NULL ) {
        fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(123);
      }

      for ( int itseq = 0; itseq < g_sequential_source_timeslice_number; itseq++ ) 
      {

        int const gtseq =  ( gsx[0] + g_sequential_source_timeslice_list[itseq] + T_global ) % T_global;
        int const have_tseq = ( gtseq / T == g_proc_coords[0] );
        int const tseq = have_tseq ? gtseq % T : -1;
        size_t const offset = _GSI( tseq * VOL3 );

        /***************************************************************************
         *  preform inversion; 
         *
         *  with 1-iflavor, since we have used g5-hermiticity
         *
         *  with wrapper function prepare_propagator_from_source from
         *  prepare_propagator.cpp
         ***************************************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        /***************************************************************************
         * invert
         *
         * use SOURCE smearing, but NO SINK smearing
         *
         * sink-end is at operator insertion
         ***************************************************************************/
        double * sequential_timeslice_source = init_1level_dtable ( _GSI( VOLUME ) );
        if ( sequential_timeslice_source == NULL ) {
          fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(123);
        }

        for ( int i = 0; i < 12; i++ ) {

          memset ( sequential_timeslice_source, 0, sizeof_spinor_field );

          if ( have_tseq ) {
            if ( g_verbose > 2 ) fprintf ( stdout, "# [] proc %4d has g/tseq %3d / %3d   %s %d\n", g_cart_id, gtseq, tseq, __FILE__, __LINE__ );
            memcpy ( sequential_timeslice_source + offset , sequential_source[i] + offset, sizeof_spinor_field_timeslice );
          }

          exitstatus = prepare_propagator_from_source ( &(sequential_propagator[i]), &sequential_timeslice_source, 1, 1-iflavor, 1, 0, gauge_field_smeared,
              check_propagator_residual, gauge_field_with_phase, lmzz, NULL );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[njjn_3pt_invert_contract] Error from prepare_propagator_from_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(123);
          }
        }  /* end of loop on spin color */

        fini_1level_dtable ( &sequential_timeslice_source );

        /***************************************************************************
         * multiply propagator with g5
         ***************************************************************************/
        g5_phi( sequential_propagator[0], 12*VOLUME );

        if ( g_write_sequential_propagator ) {
          for ( int i = 0; i < 12; i++ ) {
            sprintf ( filename, "sequential_source.%c.c%d.t%dx%dy%dz%d.sc%d.inverted",
                flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3], i );

            if ( ( exitstatus = write_propagator( sequential_propagator[i], filename, 0, g_propagator_precision) ) != 0 ) {
              fprintf(stderr, "[njjn_3pt_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }
          }
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "njjn_3pt_invert_contract", "sequential-source-invert-check-smear", g_cart_id == 0 );

        /***************************************************************************
         ***************************************************************************
         **
         ** finalize contraction and output
         **
         ***************************************************************************
         ***************************************************************************/
        for ( int igamma = 0; igamma < sequential_gamma_sets; igamma++ ) {

          gamma_matrix_type gammafive;
          gamma_matrix_set( &gammafive, 5, 1. );  /*  gamma_5 */

          double ** glg_propagator = init_2level_dtable ( 12, _GSI(VOLUME) );
          if ( glg_propagator == NULL ) { 
            fprintf ( stderr, "[njjn_3pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(123);
          }

          /***************************************************************************
           * need to set the quark flavor of the loop, which here is always iflavor
           * loop_flavor = iflavor
           ***************************************************************************/
          int const loop_flavor = iflavor;

          for ( int seq_source_type = 0; seq_source_type <= 1; seq_source_type++ )
          {

            /***************************************************************************
             * type of seq. prop. used as tag for diagram type
             ***************************************************************************/
            char const sequential_propagator_name = ( seq_source_type == 0 ) ? 'd' : 'b';

            if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_3pt_invert_contract] start seq source type %3d %c    %s %d\n",
                seq_source_type, sequential_propagator_name, __FILE__, __LINE__);

            memset ( glg_propagator[0], 0, 12 * _GSI(VOLUME) * sizeof ( double )  );

            /***************************************************************************
             * prepare propagator product
             *     G_c Loop   G_c propagator or
             *   [ G_c Loop ] G_c propagator
             *
             * here we use propagator, because we need SOURCE smearing,
             * but NO SINK smearing
             ***************************************************************************/
            exitstatus = prepare_sequential_fht_loop_source (
                glg_propagator, loop, propagator[  iflavor],
                sequential_gamma_list[igamma], sequential_gamma_num[igamma],
                NULL, seq_source_type, ( loop_flavor == 0 ? NULL : &gammafive ) ); 
 
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[njjn_3pt_invert_contract] Error from prepare_sequential_fht_loop_source, status was %d %s %d\n",
                  exitstatus, __FILE__, __LINE__ );
              EXIT(123);
            }

            double * contr_p = init_1level_dtable ( 2 * T );
            if ( contr_p == NULL ) {
              fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
              EXIT(3);
            }

            int sink_momentum[3] = { 0, 0, 0 };

            /***************************************************************************
             * using Gamma_i = unit matrix = Gamma_f;
             *
             * seq^+ glg
             ***************************************************************************/
            contract_twopoint_snk_momentum ( contr_p, 4, 4, sequential_propagator, glg_propagator, 4, 3, sink_momentum, 1 ); 
 

            /***************************************************************************
             * write to file
             *
             * seq_source_type  0 = D-type diagram
             * seq_source_type  1 = B-type diagram
             ***************************************************************************/

            if( io_proc > 0 ) {

              double * contr_p_global = init_1level_dtable ( 2 * T_global );
              if ( contr_p_global == NULL ) {
                fprintf(stderr, "[njjn_3pt_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
                EXIT(3);
              }
#ifdef HAVE_MPI
              exitstatus = MPI_Gather( contr_p , 2*T, MPI_DOUBLE, contr_p_global, 2*T, MPI_DOUBLE, 0, g_tr_comm);
              if(exitstatus != MPI_SUCCESS) {
                fprintf(stderr, "[njjn_3pt_invert_contract] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                return(3);
              }

#else
              memcpy ( contr_p_global, contr_p, 2 * T_global * sizeof ( double ) );
#endif

              if ( io_proc == 2 ) {
         
                char correlator_tag[20] = ( iflavor == 0 ) ? "p-dbGddbGd-p" : "n-ubGuubGu-n";

                char aff_tag_prefix[200], aff_tag[400];

                sprintf ( aff_tag_prefix, "/%s/nsample%d/Gc_%s/tseq%d", correlator_tag, g_nsample,
                    sequential_gamma_tag[igamma], g_sequential_source_timeslice_list[itseq] );

                if ( g_verbose > 2 ) fprintf ( stdout, "# [njjn_3pt_invert_contract] aff_tag_prefix = %s %s %d\n", aff_tag_prefix, __FILE__, __LINE__ );

                /***************************************************************************
                 * we call this by diagram type = loop seq prop. type
                 ***************************************************************************/
                sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/%c", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[0] ], gamma_id_to_Cg_ascii[ gamma_f1_list[0] ],
                    sequential_propagator_name );

                if ( g_verbose > 2 ) fprintf ( stdout, "# [njjn_3pt_invert_contract] aff_tag = %s %s %d\n", aff_tag, __FILE__, __LINE__ );

                exitstatus = write_aff_contraction ( contr_p_global, affw, NULL, aff_tag, T_global, "complex" );

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[njjn_3pt_invert_contract] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(3);
                }
 
              }  /* end of if io_proc == 2 */
        
              fini_1level_dtable ( &contr_p_global );

            }  /* end of if io_proc > 0 */

            fini_2level_dtable ( &glg_propagator );
            fini_1level_dtable ( &contr_p );

          }  /* end of loop on seq source type */

        }  /* end of loop on gamma sets */

      }  /* end of loop on source-sink time separations */

      fini_2level_dtable ( &sequential_source );
      fini_2level_dtable ( &sequential_propagator );

    }  /* end of loop on flavor */
#endif   /* of if _P_DDDD_P  */

    /***************************************************************************/
    /***************************************************************************/

#ifdef HAVE_LHPC_AFF
    /***************************************************************************
     * I/O process id 2 closes its AFF writer
     ***************************************************************************/
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_3pt_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
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

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/

  fini_3level_ztable ( &loop );

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
  show_time ( &start_time, &end_time, "njjn_3pt_invert_contract", "runtime", g_cart_id == 0 );

  return(0);

}
