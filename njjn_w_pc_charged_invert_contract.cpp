/***************************************************************************
 *
 * njjn_w_pc_charged_invert_contract
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

#define _PART_IIb 1  /* N1, N2 */
#define _PART_IV  1  /* W type sequential diagrams */

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
 * 
 ***************************************************************************/
static inline int project_write ( double ** vx, double *** vp, void * writer, char * tag, int (*momentum_list)[3], int momentum_number, int const nd, int const io_proc ) {

  int exitstatus;

  /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
  exitstatus = contract_vn_momentum_projection ( vp, vx, nd, momentum_list, momentum_number);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return( 2 );
  }

#if defined HAVE_HDF5
  exitstatus = contract_vn_write_h5 ( vp, nd, (char *)writer, tag, momentum_list, momentum_number, io_proc );
#elif defined HAVE_LHPC_AFF
  /* write to AFF file */
  exitstatus = contract_vn_write_aff ( vp, nd, (struct AffWriter_s *)writer, tag, momentum_list, momentum_number, io_proc );
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
  
  const char outfile_prefix[] = "njjn_w_pc_fch";

  const char flavor_tag[4] = { 'u', 'd', 's', 'c' };

  const int sequential_gamma_sets = 2;
  int const sequential_gamma_num[2] = {4, 4};
  int const sequential_gamma_id[2][4] = {
    { 0,  1,  2,  3 },
    { 6,  7,  8,  9 } };

  char const sequential_gamma_tag[2][3] = { "vv", "aa" };

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

  /*
  int const    gamma_f1_number                           = 4;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 , 11,  8,  2 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 , +1, -1, -1 };
  */

  int const    gamma_f1_number                           = 1;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 };

  int read_scalar_field  = 0;
  int write_scalar_field = 0;

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char aff_tag[400];
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

  fprintf(stdout, "# [njjn_w_pc_charged_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [njjn_w_pc_charged_invert_contract] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [njjn_w_pc_charged_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [njjn_w_pc_charged_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[njjn_w_pc_charged_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [njjn_w_pc_charged_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [njjn_w_pc_charged_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[njjn_w_pc_charged_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [njjn_w_pc_charged_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

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
    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   ***************************************************************************
   **
   ** Part I
   **
   ** prepare stochastic sources for W-type sequential sources and propagators
   **
   ***************************************************************************
   ***************************************************************************/

  double *** scalar_field = NULL;

  if ( g_coherent_source_number * g_nsample_oet > 0 ) {
    scalar_field = init_3level_dtable ( g_coherent_source_number, g_nsample_oet, 2*VOLUME );
    if( scalar_field == NULL ) {
      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(132);
    }
  }

  if ( ! read_scalar_field  && ( g_coherent_source_number * g_nsample_oet > 0 ) ) {

    /***************************************************************************
     * draw a stochastic binary source (real, +/1 one per site )
     ***************************************************************************/
    ranbinary ( scalar_field[0][0], 2 * g_coherent_source_number * g_nsample_oet * VOLUME );

    /***************************************************************************
     * write loop field to lime file
     ***************************************************************************/
    if ( write_scalar_field ) {
      sprintf( filename, "scalar_field.c%d.N%d.lime", Nconf, g_nsample_oet );
      
      char field_type[2000];

      sprintf( field_type, "<source_type>%d</source_type><noise_type>binary real</noise_type><coherent_sources>%d</coherent_sources>", g_source_type , g_coherent_source_number );

      for ( int i = 0; i < g_coherent_source_number * g_nsample_oet; i++ ) {
        exitstatus = write_lime_contraction( scalar_field[0][i], filename, 64, 1, field_type, Nconf, ( i > 0 ) );
        if ( exitstatus != 0  ) {
          fprintf ( stderr, "[njjn_w_pc_charged_invert_contract] Error write_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(12);
        }
      }
    }  /* end of if write_scalar_field */

  } else {
    sprintf( filename, "scalar_field.c%d.N%d.lime", Nconf, g_nsample_oet );
      
    for ( int i = 0; i < g_coherent_source_number * g_nsample_oet; i++ ) {
      exitstatus = read_lime_contraction ( scalar_field[0][i], filename, 1, i );
      if ( exitstatus != 0  ) {
        fprintf ( stderr, "[njjn_w_pc_charged_invert_contract] Error read_lime_contraction, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }
    }
  }  /* end of if read scalar field */

#ifdef _SMEAR_QUDA
    /***************************************************************************
     * dummy solve, just to have original gauge field up on device,
     * for subsequent APE smearing
     ***************************************************************************/

  double ** spinor_work = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
  if ( spinor_work == NULL ) {
    fprintf ( stderr, "[njjn_w_pc_charged_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
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
    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
        fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
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
      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.h5", g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [njjn_w_pc_charged_invert_contract] writing data to file %s\n", filename);

      int ncdim = 2;
      int cdim[2] = { g_source_momentum_number, 3 };
      char tag[100];
      sprintf ( tag, "/src_mom" );

      exitstatus = write_h5_contraction ( g_source_momentum_list[0], NULL, filename, tag, "int", ncdim, cdim );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from write_h5_contraction,  status %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(124);
      }

      cdim[0] = g_sink_momentum_number;
      sprintf ( tag, "/snk_mom" );

      exitstatus = write_h5_contraction ( g_sink_momentum_list[0], NULL, filename, tag, "int", ncdim, cdim );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from write_h5_contraction,  status %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(124);
      }

      writer = (void *)filename;

#elif defined HAVE_LHPC_AFF
    /***************************************************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     ***************************************************************************/
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [njjn_w_pc_charged_invert_contract] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      writer = (void *)affw;
#else
      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error, no outupt variant selected %s %d\n",  __FILE__, __LINE__);
      EXIT(15);
#endif
    }  /* end of if io_proc == 2 */

    /* up and down quark propagator with source smearing */
    double **** propagator = init_4level_dtable ( g_coherent_source_number, 2, 12, _GSI( VOLUME ) );
    if( propagator == NULL ) {
      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /* up and down quark propagator with source and sink smearing,
     * to use for baryon 2-pt function 
     */
    double **** propagator_snk_smeared = init_4level_dtable ( g_coherent_source_number, 2, 12, _GSI( VOLUME ) );
    if ( propagator_snk_smeared == NULL ) {
      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

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
     ***************************************************************************
     **
     ** Part IIa
     **
     ** point-to-all propagators with source at coherent source
     **
     ***************************************************************************
     ***************************************************************************/
    for ( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {
        
      /***************************************************************************
       * coherent source
       ***************************************************************************/
      int const csx[4] = {
        ( gsx[0] + icoh * ( T_global / g_coherent_source_number ) ) % T_global,
        ( gsx[1] + icoh * LX_global / 2 ) % LX_global,
        ( gsx[2] + icoh * LY_global / 2 ) % LY_global,
        ( gsx[3] + icoh * LZ_global / 2 ) % LZ_global };

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
        exitstatus = point_source_propagator ( propagator[icoh][iflavor], csx,       iflavor,     1,         0,         gauge_field_smeared,       check_propagator_residual, gauge_field_with_phase, lmzz );
        if(exitstatus != 0) {
          fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(12);
        }
      
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "njjn_w_pc_charged_invert_contract", "forward-light-smear-invert-check", g_cart_id == 0 );

        /***********************************************************
         * sink-smear the flavor-type point-to-all propagator
         * store extra
         ***********************************************************/

        gettimeofday ( &ta, (struct timezone *)NULL );

        for ( int i = 0; i < 12; i++ ) {
          /* copy propagator */
          memcpy ( propagator_snk_smeared[icoh][iflavor][i], propagator[icoh][iflavor][i], sizeof_spinor_field );

          /* sink-smear propagator */
          exitstatus = Jacobi_Smearing ( gauge_field_smeared, propagator_snk_smeared[icoh][iflavor][i], N_Jacobi, kappa_Jacobi);
          if(exitstatus != 0) {
            fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            return(11);
          }
        }
      
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "njjn_w_pc_charged_invert_contract", "forward-light-sink-smear", g_cart_id == 0 );

        /***********************************************************
         * optionally write the propagator to disc
         *
         * we use the standard lime format here
         * https://github.com/usqcd-software/c-lime
         ***********************************************************/
        if ( g_write_propagator ) {
          /* each spin-color component into a separate file */
          for ( int i = 0; i < 12; i++ ) {
            sprintf ( filename, "propagator_%c.%.4d.t%dx%dy%dz%d.%d.inverted", flavor_tag[iflavor], Nconf, csx[0], csx[1], csx[2], csx[3] , i );

            if ( ( exitstatus = write_propagator( propagator[icoh][iflavor][i], filename, 0, g_propagator_precision) ) != 0 ) {
              fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }
          }
        }

      }  /* end of loop on flavor */

      /***************************************************************************
       ***************************************************************************
       **
       ** Part IIb
       **
       ** point-to-all propagator contractions for baryon 2pts
       **
       ***************************************************************************
       ***************************************************************************/
#if _PART_IIb

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
          fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
          EXIT(47);
        }

        /***************************************************************************
         * vp holds the nucleon-nucleon spin propagator in momentum space,
         * i.e. the momentum projected vx
         ***************************************************************************/
        double *** vp = init_3level_dtable ( T, g_source_momentum_number, 32 );
        if ( vp == NULL ) {
          fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(47);
        }

        /***************************************************************************
         *
         * [ X^T Xb X ] - [ Xb^+ X^* X^+ ]
         *
         ***************************************************************************/

        char  tag[300], tag_prefix[200];
        sprintf ( tag_prefix, "/N-N/%c%c%c/T%d_X%d_Y%d_Z%d", flavor_tag[iflavor], flavor_tag[1-iflavor], flavor_tag[iflavor], csx[0], csx[1], csx[2], csx[3] );
         
        /***************************************************************************
         * fill the fermion propagator fp with the 12 spinor fields
         * in propagator of flavor X
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp, propagator_snk_smeared[icoh][iflavor], VOLUME);

        /***************************************************************************
         * fill fp2 with 12 spinor fields from propagator of flavor Xb
         ***************************************************************************/
        assign_fermion_propagator_from_spinor_field ( fp2, propagator_snk_smeared[icoh][1-iflavor], VOLUME);

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
            fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(48);
          }

          /***************************************************************************
           * diagram n2
           ***************************************************************************/
          sprintf(tag, "%s/Gi_%s/Gf_%s/t2", tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

          exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp, contract_v6, writer, tag, g_source_momentum_list, g_source_momentum_number, 16, VOLUME, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(48);
          }

        }}  /* end of loop on Dirac Gamma structures */

        fini_2level_dtable ( &vx );
        fini_3level_dtable ( &vp );

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "njjn_w_pc_charged_invert_contract", "n1-n2-reduce-project-write", g_cart_id == 0 );

      }  /* end of loop on flavor */
    
#endif  /* end of if _PART_IIb  */

    }  /* end of loop on coherent sources */

    /***************************************************************************
     *
     * sequential inversion and contraction
     *
     ***************************************************************************/

    /***************************************************************************
     * loop on sequential source momenta
     ***************************************************************************/

    int momentum[3] = { 0, 0, 0 };

    if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_w_pc_charged_invert_contract] start seq source mom %3d %3d %3d  %s %d\n", 
          momentum[0], momentum[1], momentum[2], __FILE__, __LINE__);

    /***************************************************************************/
    /***************************************************************************/

#if _PART_IV
    /***************************************************************************
     ***************************************************************************
     **
     ** Part IV
     **
     ** sequential inversion with binary noise insertion for twin-peak diagram
     **
     ***************************************************************************
     ***************************************************************************/

    for ( int isample = g_sample_id_list[isource_location][0]; isample <= g_sample_id_list[isource_location][1]; isample++ ) 
    {

      /***************************************************************************
       * loop on sequential source gamma matrix sets
       ***************************************************************************/
      for ( int igs = 0; igs < 2; igs++ )
      {
                
        /***************************************************************************
         * loop on sequential source gamma matrices inside sets
         ***************************************************************************/
        for ( int ig = 0; ig < 4; ig++ )
        {
                
          /***************************************************************************
           * allocate for sequential propagator and source
           ***************************************************************************/
          double ****** sequential_propagator = init_6level_dtable ( 2, 2, 3, 3,  12, _GSI( VOLUME ) );
          if( sequential_propagator == NULL ) {
            fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(123);
          }

          double ** sequential_source = init_2level_dtable ( 12,  _GSI(VOLUME) );
          if( sequential_source == NULL ) {
            fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(132);
          }


          for ( int iflavor = 0; iflavor < 2; iflavor++ )
          {
          for ( int iflavor2 = 0; iflavor2 < 2; iflavor2++ )
          {

            /***************************************************************************
             * sequential source for "twin-peak" diagram, 
             * needed as both up-after-up and down-after-down type sequential propagator
             ***************************************************************************/

            gettimeofday ( &ta, (struct timezone *)NULL );

            /***************************************************************************
             * auxilliary field sequential_source_accum to accumulate the seq source;
             * abuse memory of sequential_propagator[iflavor] which has not yet been
             * filled, but will be written after seq source construction
             ***************************************************************************/

            for ( int icol1 = 0; icol1 < 3; icol1++ ) 
            {
            for ( int icol2 = 0; icol2 < 3; icol2++ ) 
            {

                double ** sequential_source_accum = init_2level_dtable ( 12, _GSI(VOLUME) );
                if ( sequential_source_accum == NULL ) {
                  fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
                  EXIT(12);
                }

                memset ( sequential_source[0], 0, 12 * sizeof_spinor_field );

                for ( int icoh = 0; icoh < g_coherent_source_number; icoh++)
                {
                  
                    memset ( sequential_source_accum[0], 0, 12 * sizeof_spinor_field );

                    /***************************************************************************
                     * sequential_source_accum = propagator iflavor2
                     ***************************************************************************/
                    exitstatus = prepare_sequential_fht_twinpeak_source_crossed ( sequential_source_accum, propagator[icoh][iflavor2], scalar_field[icoh][isample],
                        sequential_gamma_id[igs][ig], icol1, icol2 );
                    if ( exitstatus != 0 ) {
                      fprintf ( stderr, "[njjn_w_pc_charged_invert_contract] Error from prepare_sequential_fht_twinpeak_source_crossed, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT(123);
                    }

                    spinor_field_pl_eq_spinor_field ( sequential_source[0], sequential_source_accum[0], 12*VOLUME );
                  }

                  fini_2level_dtable ( &sequential_source_accum );

                  gettimeofday ( &tb, (struct timezone *)NULL );
                  show_time ( &ta, &tb, "njjn_w_pc_charged_invert_contract", "prepare-sequential-fht-w-source", g_cart_id == 0 );

                  /***************************************************************************
                   * seq. prop. from seq. source
                   *
                   * INCLUDING SINK-SMEARING
                   *
                   * USING flavor iflavor
                   ***************************************************************************/

                  gettimeofday ( &ta, (struct timezone *)NULL );
                  exitstatus = prepare_propagator_from_source ( sequential_propagator[iflavor][iflavor2][icol1][icol2], sequential_source, 12, iflavor, 0, 1, gauge_field_smeared,
                      check_propagator_residual, gauge_field_with_phase, lmzz, NULL );
                  if ( exitstatus != 0 ) {
                    fprintf ( stderr, "[njjn_w_pc_charged_invert_contract] Error from prepare_propagator_from_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                    EXIT(123);
                  }

                  gettimeofday ( &tb, (struct timezone *)NULL );
                  show_time ( &ta, &tb, "njjn_w_pc_charged_invert_contract", "sequential-source-w-invert-check-smear", g_cart_id == 0 );

                
              }}  /* end of loop on color components */

            }}  /* end of loop on quark propagator flavor */

            /***************************************************************************
             *
             * contractions for W-type diagrams
             *
             ***************************************************************************/
            char correlator_tag[20] = "N-qbGqqbGq-N";

            for ( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) 
            {
 
              /***************************************************************************
               * coherent source
               ***************************************************************************/
              int const csx[4] = {
                ( gsx[0] + icoh * ( T_global / g_coherent_source_number ) ) % T_global,
                ( gsx[1] + icoh * LX_global / 2 ) % LX_global,
                ( gsx[2] + icoh * LY_global / 2 ) % LY_global,
                ( gsx[3] + icoh * LZ_global / 2 ) % LZ_global };

              /***************************************************************************
               * we need for W
               *   p u1u u1d nbar type insertion into proton
               *   p d1d u1d nbar type insertion into proton
               *   --------------------------------------
               *   n d1d d1u pbar type insertion into neutron 
               *   n u1u d1u pbar type insertion into neutron
               *
               *   the neutron correlators follow from the proton case by flavor change
               *
               ***************************************************************************/
              for ( int iflavor = 0; iflavor < 2; iflavor++ ) 
              {
 
                gettimeofday ( &ta, (struct timezone *)NULL );

                /***************************************************************************
                 * vx holds the x-dependent nucleon-nucleon spin propagator,
                 * i.e. a 4x4 complex matrix per space time point
                 ***************************************************************************/
                double ** vx_aux = init_2level_dtable ( VOLUME, 32 );
                if ( vx_aux == NULL ) {
                  fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
                  EXIT(47);
                }
                double ** vx = init_2level_dtable ( VOLUME, 32 );
                if ( vx == NULL ) {
                  fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
                  EXIT(47);
                }

                /***************************************************************************
                 * vp holds the nucleon-nucleon spin propagator in momentum space,
                 * i.e. the momentum projected vx
                 ***************************************************************************/
                double *** vp = init_3level_dtable ( T, g_sink_momentum_number, 32 );
                if ( vp == NULL ) {
                  fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
                  EXIT(47);
                }

                for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) 
                {
                for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) 
                {
                
                  char tag[300], tag_prefix[200];

                  /***************************************************************************
                   ***************************************************************************
                   **
                   ** COLOR-CROSSED CONTRACTIONS
                   **
                   ***************************************************************************
                   ***************************************************************************/

                  /***************************************************************************
                   * BEGIN OF p u1u u1d n   <---> n d1d d1u p (by flavor change)
                   ***************************************************************************/
                  sprintf ( tag_prefix, "/%s/w%c%c-w%c%c-f%c/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/sample%d/Gc_%s",
                      correlator_tag, 
                      flavor_tag[iflavor], flavor_tag[iflavor], 
                      flavor_tag[iflavor], flavor_tag[1-iflavor],
                      flavor_tag[1-iflavor],
                      csx[0], csx[1], csx[2], csx[3], 
                      momentum[0], momentum[1], momentum[2],
                      isample, 
                      gamma_id_to_ascii[ sequential_gamma_id[igs][ig] ] );


                  /***************************************************************************
                   * fp3 = fwd dn
                   ***************************************************************************/
                  assign_fermion_propagator_from_spinor_field ( fp3, propagator_snk_smeared[icoh][1-iflavor], VOLUME);
        
                  /***************************************************************************
                   * calculate fp3 <- Gamma[if2] x fwd dn
                   ***************************************************************************/
                  fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp3, VOLUME );
    
                  fermion_propagator_field_eq_fermion_propagator_field_ti_re  ( fp3, fp3, gamma_f1_sign[if2], VOLUME );
                    
                  /***************************************************************************
                   * color-crossed, diagram t1
                   ***************************************************************************/
                  memset ( vx[0], 0, 32*VOLUME *sizeof(double) );

                  for ( int icol1 = 0; icol1 < 3; icol1++ ) 
                  {
                  for ( int icol2 = 0; icol2 < 3; icol2++ ) 
                  {
 
                    /***************************************************************************
                     * fp2 = seq up-after-up icol1, icol2
                     ***************************************************************************/
                    assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator[iflavor][iflavor][icol1][icol2], VOLUME);
 
                    /***************************************************************************
                     * calculate fp2 <- seq up-after-up  Gamma[if1]
                     ***************************************************************************/
                    fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp2, gamma_f1_list[if1], fp2, VOLUME );

                    fermion_propagator_field_eq_fermion_propagator_field_ti_re ( fp2, fp2, gamma_f1_sign[if1], VOLUME );

                    /***************************************************************************
                     * fp = seq up-after-dn icol2, icol1
                     ***************************************************************************/
                    assign_fermion_propagator_from_spinor_field ( fp,  sequential_propagator[iflavor][1-iflavor][icol2][icol1], VOLUME);

                    /***************************************************************************/

                    memset ( vx_aux[0] , 0, 32 * VOLUME * sizeof ( double ) );

                    /* contraction 
                     *                                 Wuu  Wud D
                     * */
                    exitstatus = contract_v5 ( vx_aux, fp2, fp, fp3, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0] , 0, 32 * VOLUME * sizeof ( double ) );

                    /* contraction 
                     *                                 Wuu  D    Wud
                     */
                    exitstatus = contract_v5 ( vx_aux, fp2, fp3, fp, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0] , 0, 32 * VOLUME * sizeof ( double ) );

                    /* contraction 
                     *                                 Wud Wuu  D
                     */
                    exitstatus = contract_v5 ( vx_aux, fp, fp2, fp3, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0] , 0, 32 * VOLUME * sizeof ( double ) );

                    /* contraction
                     *                                 Wud Wuu  D
                     */
                    exitstatus = contract_v6 ( vx_aux, fp, fp2, fp3, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                  }}  /* end of icol1, icol2 */

                  sprintf(tag, "%s/Gf_%s/Gi_%s/t1", tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

                  exitstatus = project_write ( vx, vp, writer, tag, g_sink_momentum_list, g_sink_momentum_number, 16, io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(48);
                  }
 
                  /***************************************************************************
                   * END OF p u1u u1d n
                   ***************************************************************************/

                  /***************************************************************************/
                  /***************************************************************************/

                  /***************************************************************************
                   * BEGIN OF p d1d u1d n   <---> n u1u d1u p (by flavor change)
                   ***************************************************************************/
                  sprintf ( tag_prefix, "/%s/w%c%c-f%c-w%c%c/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/sample%d/Gc_%s",
                      correlator_tag, 
                      flavor_tag[iflavor], flavor_tag[1-iflavor], 
                      flavor_tag[iflavor],
                      flavor_tag[1-iflavor], flavor_tag[1-iflavor],
                      csx[0], csx[1], csx[2], csx[3], 
                      momentum[0], momentum[1], momentum[2],
                      isample,
                      gamma_id_to_ascii[ sequential_gamma_id[igs][ig] ] );

                  /***************************************************************************
                   * fp3 = fwd up
                   ***************************************************************************/
                  assign_fermion_propagator_from_spinor_field ( fp3, propagator_snk_smeared[icoh][iflavor], VOLUME);

                  /***************************************************************************
                   * calculate fp3 <- fp3 x Gamma[if1]
                   ***************************************************************************/

                  fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );

                  fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, gamma_f1_sign[if1], VOLUME );
        
                  /***************************************************************************
                   * color-crossed, diagram t2
                   ***************************************************************************/
                  memset ( vx[0], 0, 32*VOLUME *sizeof(double) );

                  for ( int icol1 = 0; icol1 < 3; icol1++ ) 
                  {
                  for ( int icol2 = 0; icol2 < 3; icol2++ ) 
                  {
 
                    /***************************************************************************
                     * fp2 = seq dn-after-dn icol1, icol2
                     ***************************************************************************/
                    assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator[1-iflavor][1-iflavor][icol1][icol2], VOLUME);

                    /***************************************************************************
                     * calculate fp2 <- Gamma[if2] x fp2
                     ***************************************************************************/
                    fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp2, gamma_f1_list[if2], fp2, VOLUME );
    
                    fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp2, fp2, gamma_f1_sign[if2], VOLUME );

                    /***************************************************************************
                     * fp = seq up-after-dn icol2, icol1
                     ***************************************************************************/
                    assign_fermion_propagator_from_spinor_field ( fp,  sequential_propagator[iflavor][1-iflavor][icol2][icol1], VOLUME);

                    /***************************************************************************/

                    memset ( vx_aux[0], 0, 32 * VOLUME * sizeof (double ) );

                    /* contraction 
                     *                                 Wud U    Wdd
                     */
                    exitstatus = contract_v5 ( vx_aux, fp, fp3, fp2, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0], 0, 32 * VOLUME * sizeof (double ) );

                    /* contraction
                     *                                 Wud U    Wdd
                     */
                    exitstatus = contract_v6 ( vx_aux, fp, fp3, fp2, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0], 0, 32 * VOLUME * sizeof (double ) );

                    /* contraction 
                     *                                 U    Wdd  Wud
                     */
                    exitstatus = contract_v5 ( vx_aux, fp3, fp2, fp, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0], 0, 32 * VOLUME * sizeof (double ) );

                    /* contraction
                     *                                 U    Wud Wdd
                     */
                    exitstatus = contract_v5 ( vx_aux, fp3, fp, fp2, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                  }}  /* end of icol1, icol2 */

                  sprintf(tag, "%s/Gf_%s/Gi_%s/t2", tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

                  exitstatus = project_write ( vx, vp, writer, tag, g_sink_momentum_list, g_sink_momentum_number, 16, io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(48);
                  }
 
                  /***************************************************************************
                   * END OF p d1d u1d n
                   ***************************************************************************/

                  /***************************************************************************/
                  /***************************************************************************/

                  /***************************************************************************
                   ***************************************************************************
                   **
                   ** NON-COLOR-CROSSED CONTRACTIONS
                   **
                   ***************************************************************************
                   ***************************************************************************/

                  /***************************************************************************
                   * BEGIN OF p u1u u1d n   <---> n d1d d1u p (by flavor change)
                   ***************************************************************************/
                  sprintf ( tag_prefix, "/%s/w%c%c-w%c%c-f%c/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/sample%d/Gc_%s",
                      correlator_tag, 
                      flavor_tag[iflavor], flavor_tag[iflavor], 
                      flavor_tag[iflavor], flavor_tag[1-iflavor],
                      flavor_tag[1-iflavor],
                      csx[0], csx[1], csx[2], csx[3], 
                      momentum[0], momentum[1], momentum[2],
                      isample, 
                      gamma_id_to_ascii[ sequential_gamma_id[igs][ig] ] );


                  /***************************************************************************
                   * fp3 = fwd dn
                   ***************************************************************************/
                  assign_fermion_propagator_from_spinor_field ( fp3, propagator_snk_smeared[icoh][1-iflavor], VOLUME);
        
                  /***************************************************************************
                   * calculate fp3 <- Gamma[if2] x fwd dn
                   ***************************************************************************/
                  fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp3, VOLUME );
    
                  fermion_propagator_field_eq_fermion_propagator_field_ti_re  ( fp3, fp3, gamma_f1_sign[if2], VOLUME );
                    
                  /***************************************************************************
                   * color-crossed, diagram t1
                   ***************************************************************************/
                  memset ( vx[0], 0, 32*VOLUME *sizeof(double) );

                  for ( int icol1 = 0; icol1 < 3; icol1++ ) 
                  {
                  for ( int icol2 = 0; icol2 < 3; icol2++ ) 
                  {
 
                    /***************************************************************************
                     * fp2 = seq up-after-up icol1, icol1
                     ***************************************************************************/
                    assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator[iflavor][iflavor][icol1][icol1], VOLUME);
 
                    /***************************************************************************
                     * calculate fp2 <- seq up-after-up  Gamma[if1]
                     ***************************************************************************/
                    fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp2, gamma_f1_list[if1], fp2, VOLUME );

                    fermion_propagator_field_eq_fermion_propagator_field_ti_re ( fp2, fp2, gamma_f1_sign[if1], VOLUME );

                    /***************************************************************************
                     * fp = seq up-after-dn icol2, icol2
                     ***************************************************************************/
                    assign_fermion_propagator_from_spinor_field ( fp,  sequential_propagator[iflavor][1-iflavor][icol2][icol2], VOLUME);

                    /***************************************************************************/

                    memset ( vx_aux[0] , 0, 32 * VOLUME * sizeof ( double ) );

                    /* contraction 
                     *                                 Wuu  Wud D
                     */
                    exitstatus = contract_v5 ( vx_aux, fp2, fp, fp3, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0] , 0, 32 * VOLUME * sizeof ( double ) );

                    /* contraction 
                     *                                 Wuu  D    Wud
                     */
                    exitstatus = contract_v5 ( vx_aux, fp2, fp3, fp, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0] , 0, 32 * VOLUME * sizeof ( double ) );

                    /* contraction
                     *                                 Wud Wuu  D
                     */
                    exitstatus = contract_v5 ( vx_aux, fp, fp2, fp3, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0] , 0, 32 * VOLUME * sizeof ( double ) );

                    /* contraction 
                     *                                 Wud Wuu  D
                     */
                    exitstatus = contract_v6 ( vx_aux, fp, fp2, fp3, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                  }}  /* end of icol1, icol2 */

                  sprintf(tag, "%s/Gf_%s/Gi_%s/t3", tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

                  exitstatus = project_write ( vx, vp, writer, tag, g_sink_momentum_list, g_sink_momentum_number, 16, io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(48);
                  }
 
                  /***************************************************************************
                   * END OF p u1u u1d n
                   ***************************************************************************/

                  /***************************************************************************/
                  /***************************************************************************/

                  /***************************************************************************
                   * BEGIN OF p d1d u1d n   <---> n u1u d1u p (by flavor change)
                   ***************************************************************************/
                  sprintf ( tag_prefix, "/%s/w%c%c-f%c-w%c%c/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/sample%d/Gc_%s",
                      correlator_tag, 
                      flavor_tag[iflavor], flavor_tag[1-iflavor], 
                      flavor_tag[iflavor],
                      flavor_tag[1-iflavor], flavor_tag[1-iflavor],
                      csx[0], csx[1], csx[2], csx[3], 
                      momentum[0], momentum[1], momentum[2],
                      isample,
                      gamma_id_to_ascii[ sequential_gamma_id[igs][ig] ] );

                  /***************************************************************************
                   * fp3 = fwd up
                   ***************************************************************************/
                  assign_fermion_propagator_from_spinor_field ( fp3, propagator_snk_smeared[icoh][iflavor], VOLUME);

                  /***************************************************************************
                   * calculate fp3 <- fp3 x Gamma[if1]
                   ***************************************************************************/

                  fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );

                  fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, gamma_f1_sign[if1], VOLUME );
        
                  /***************************************************************************
                   * color-crossed, diagram t2
                   ***************************************************************************/
                  memset ( vx[0], 0, 32*VOLUME *sizeof(double) );

                  for ( int icol1 = 0; icol1 < 3; icol1++ ) 
                  {
                  for ( int icol2 = 0; icol2 < 3; icol2++ ) 
                  {
 
                    /***************************************************************************
                     * fp2 = seq dn-after-dn icol1, icol1
                     ***************************************************************************/
                    assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator[1-iflavor][1-iflavor][icol1][icol1], VOLUME);

                    /***************************************************************************
                     * calculate fp2 <- Gamma[if2] x fp2
                     ***************************************************************************/
                    fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp2, gamma_f1_list[if2], fp2, VOLUME );
    
                    fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp2, fp2, gamma_f1_sign[if2], VOLUME );

                    /***************************************************************************
                     * fp = seq up-after-dn icol2, icol2
                     ***************************************************************************/
                    assign_fermion_propagator_from_spinor_field ( fp,  sequential_propagator[iflavor][1-iflavor][icol2][icol2], VOLUME);

                    /***************************************************************************/

                    memset ( vx_aux[0], 0, 32 * VOLUME * sizeof (double ) );

                    /* contraction */
                    exitstatus = contract_v5 ( vx_aux, fp, fp3, fp2, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0], 0, 32 * VOLUME * sizeof (double ) );

                    /* contraction */
                    exitstatus = contract_v6 ( vx_aux, fp, fp3, fp2, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0], 0, 32 * VOLUME * sizeof (double ) );

                    /* contraction */
                    exitstatus = contract_v5 ( vx_aux, fp3, fp2, fp, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                    /***************************************************************************/

                    memset ( vx_aux[0], 0, 32 * VOLUME * sizeof (double ) );

                    /* contraction */
                    exitstatus = contract_v5 ( vx_aux, fp3, fp, fp2, VOLUME );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                      EXIT( 1 );
                    }

#pragma omp parallel for
                    for ( size_t ix = 0; ix < 32*VOLUME; ix++ )
                    {
                      vx[0][ix] -= vx_aux[0][ix];
                    }

                  }}  /* end of icol1, icol2 */

                  sprintf(tag, "%s/Gf_%s/Gi_%s/t4", tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

                  exitstatus = project_write ( vx, vp, writer, tag, g_sink_momentum_list, g_sink_momentum_number, 16, io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(48);
                  }
 
                  /***************************************************************************
                   * END OF p d1d u1d n
                   ***************************************************************************/

                  /***************************************************************************/
                  /***************************************************************************/

                }} // end of loop on Dirac gamma structures

                fini_2level_dtable ( &vx_aux );
                fini_2level_dtable ( &vx );
                fini_3level_dtable ( &vp );
	
                gettimeofday ( &tb, (struct timezone *)NULL );
                show_time ( &ta, &tb, "njjn_w_pc_charged_invert_contract", "w-uuuu-diagram-reduce-project-write", g_cart_id == 0 );

              }  /* end of loop on flavor */

            }  /* end of loop on coherent sources */
          
            fini_2level_dtable ( &sequential_source );
            fini_6level_dtable ( &sequential_propagator );

        }  /* end of loop on gamma id inside gamma set */

      }  /* end of loop on gamma sets */

    }  /* end of loop oet samples  */

#endif  /* of _PART_IV  */

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
        fprintf(stderr, "[njjn_w_pc_charged_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    /***************************************************************************
     * free propagator fields
     ***************************************************************************/
    fini_4level_dtable ( &propagator );
    fini_4level_dtable ( &propagator_snk_smeared );

  }  /* end of loop on source locations */

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/

  if ( scalar_field != NULL ) fini_3level_dtable ( &scalar_field );

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
  show_time ( &start_time, &end_time, "njjn_w_pc_charged_invert_contract", "runtime", g_cart_id == 0 );

  return(0);

}
