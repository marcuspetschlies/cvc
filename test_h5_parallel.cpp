/****************************************************
 * test_h5_parallel
 *
 ****************************************************/

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

#ifdef HAVE_HDF5
#include "hdf5.h"
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
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "read_input_parser.h"
#include "contractions_io.h"

#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "clover.h"
#include "ranlxd.h"
#include "prepare_source.h"
#include "propagator_io.h"


#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_ST 2

using namespace cvc;

void usage() {
  fprintf(stdout, "test h5 parallel writing\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[300];

  struct timeval ta, tb, start_time, end_time;

  double **mzz[2]    = { NULL, NULL }, **mzzinv[2]    = { NULL, NULL };
  double **DW_mzz[2] = { NULL, NULL }, **DW_mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  char output_filename[400];
  int * rng_state = NULL;

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

  gettimeofday ( &start_time, (struct timezone *) NULL );


  /* set the default values */
  if(filename_set==0) sprintf ( filename, "loop.input" );
  /* fprintf(stdout, "# [test_h5_parallel] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_h5_parallel] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
  /* exitstatus = tmLQCD_invert_init(argc, argv, 1); */
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

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_h5_parallel] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[test_h5_parallel] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***************************************************************************
   * some additional xchange operations
   ***************************************************************************/
  mpi_init_xchange_contraction(2);
  mpi_init_xchange_eo_spinor ();
  
  /***************************************************************************
   * initialize own gauge field or get from tmLQCD wrapper
   ***************************************************************************/
  #ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [test_h5_parallel] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_h5_parallel] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[test_h5_parallel] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
  #else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_h5_parallel] Nconf = %d\n", Nconf);
  
  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }
  
  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[test_h5_parallel] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
  #endif
  
  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[test_h5_parallel] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }
  
  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[test_h5_parallel] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }
  
  /***************************************************************************
   * initialize clover, mzz and mzz_inv
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_h5_parallel] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
  
  /***************************************************************************
   * initialize clover, mzz and mzz_inv for Wilson Dirac-operator
   * i.e. twisted mass = 0
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &DW_mzz, &DW_mzzinv, gauge_field_with_phase, 0., g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_h5_parallel] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
  
  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[test_h5_parallel] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [test_h5_parallel] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
  
  
  /***************************************************************************
   * half-VOLUME spinor fields
   * WITH HALO
   * no additional memory, just split up the spinor_work fields
   ***************************************************************************/
  
  
  
  /***************************************************************************
   * allocate memory for contractions
   ***************************************************************************/

  int const ldim[3] = {5, 7, 11 };
  int const lvol = ldim[0] * ldim[1] * ldim[2];

  double *** loop = init_3level_dtable ( ldim[0], ldim[1], 2 * ldim[2] );
  if ( loop == NULL ) {
    fprintf(stderr, "[test_h5_parallel] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }
  
  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_h5_parallel] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }
  
  sprintf ( output_filename, "%s.%.4d.h5", g_outfile_prefix, Nconf );
  if(io_proc == 2 && g_verbose > 1 ) { 
    fprintf(stdout, "# [test_h5_parallel] writing data to file %s\n", output_filename);
  }
  
  
  /*****************************************************************/
  /*****************************************************************/
 
  ranlxd ( loop[0][0], 2*lvol );

  /*for ( int it = 0; it < T; it++ ) 
  {
    for ( int ix = 0; ix < 32*g_sink_momentum_number; ix++ ) {
      loop[it][0][ix] = (double)g_cart_id;
    }
  }*/


#if 0
  /*****************************************************************
   * use gather operation
   *****************************************************************/
  double * gloop = ( io_proc == 2 ) ? init_1level_dtable ( 2 * g_nproc_t * lvol ) : NULL;


  if ( io_proc > 0 )  {
    if ( MPI_Gather ( loop[0][0], 2*lvol, MPI_DOUBLE, gloop, 2*lvol, MPI_DOUBLE, 0, g_tr_comm ) != MPI_SUCCESS ) {
      fprintf(stderr, "[test_h5_parallel] Error from MPI_Gather %s %d\n", __FILE__, __LINE__ );;
      EXIT( 50 );
    }

    if ( io_proc == 2 ) {
  
      double * buffer = init_1level_dtable ( 2 * g_nproc_t * lvol );
      for( int i1 = 0; i1 < ldim[0]; i1++ ) {
      for( int i2 = 0; i2 < ldim[1]; i2++ ) {
        for ( int it = 0; it < g_nproc_t * ldim[2]; it++ ) 
        {
          int const i3 = it % ldim[2];
          int const ip = it / ldim[2];
          int const ix = ip * lvol + i1 * ldim[1]*ldim[2] + i2 * ldim[2] + i3;

          int const iy = i1 * ldim[1] * g_nproc_t * ldim[2] + i2 * g_nproc_t * ldim[2] + it;
          buffer[2*iy  ] = gloop[2*ix  ];
          buffer[2*iy+1] = gloop[2*ix+1];
        }
      }}

      char h5_filename[100] = "test_h5.h5";
      char h5_datatag[100] = "/data-tag";

      int const cdim[3] = { ldim[0], ldim[1], 2 * ldim[2] * g_nproc_t };

      exitstatus = write_h5_contraction ( buffer, NULL, h5_filename, h5_datatag, "double", 3, cdim );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[test_h5_parallel] Error from write_h5_contraction, status %d    %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }

      fini_1level_dtable ( &buffer);
    }
  }

  fini_1level_dtable ( &gloop );
#endif
  
#if 0
  /***************************************************************************
   * write to stdout, pre process
   ***************************************************************************/

  for ( int i0 = 0; i0 < ldim[0]; i0++ )
  {
    for ( int i1 = 0; i1 < ldim[1]; i1++ )
    {
      for ( int i2 = 0; i2 < ldim[2]; i2++ ) 
      {
        fprintf ( stdout, "proc%.4d   %3d %3d %3d  %25.16e %25.16e\n", g_cart_id, i0, i1, i2 + g_proc_coords[0]*ldim[2], loop[i0][i1][2*i2], loop[i0][i1][2*i2+1] );
      }
    }
  }
#endif


  /***************************************************************************
   * write to h5, in parallel
   ***************************************************************************/
  if ( io_proc > 0 ) 
  {
    int const cdim[3] = { ldim[0], ldim[1], 2 * ldim[2] * g_nproc_t };
    char h5_filename[100] = "test_h5_parallel.h5";
    char h5_datatag[100] = "/data-tag";

    exitstatus = write_h5_contraction_parallel ( loop[0][0], h5_filename, h5_datatag, "double", 3, cdim, g_tr_comm );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_h5_parallel] Error from write_h5_contraction_parallel, status %d    %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }
  }

  /***************************************************************************
   * decallocate fields
   ***************************************************************************/
  fini_3level_dtable ( &loop );


  /***************************************************************************
   * fini rng state
   ***************************************************************************/
  fini_rng_state ( &rng_state);

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( &mzz, &mzzinv );
  fini_clover ( &DW_mzz, &DW_mzzinv );

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor ();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *) NULL );

  show_time ( &start_time, &end_time, "test_h5_parallel", "runtime", g_cart_id == 0 );

  return(0);

}
