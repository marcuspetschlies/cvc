/****************************************************
 * twop_invert_contract_stochastic
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
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "Q_phi.h"
#include "clover.h"
#include "ranlxd.h"
#include "smearing_techniques.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_ST 2

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to calculate 2-pt functions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "twop";

  const char fbwd_str[2][4] =  { "fwd", "bwd" };

  int const current_momentum_sqr_max = 3;

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[100];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL, *gauge_field_smeared = NULL;
  char output_filename[400];
  int * rng_state = NULL;

  char const flavor_tag[2] = { 'u', 'd' };

  char data_tag[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "rh?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'r':
      check_propagator_residual = 1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) sprintf ( filename, "%s.input", outfile_prefix );
  /* fprintf(stdout, "# [twop_invert_contract_stochastic] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [twop_invert_contract_stochastic] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize MPI parameters for cvc
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

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [twop_invert_contract_stochastic] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  unsigned int VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field           = _GSI(VOLUME) * sizeof(double);
  size_t const sizeof_spinor_field_timeslice = _GSI(VOLUME) * sizeof(double);

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
    if(g_cart_id==0) fprintf(stdout, "# [twop_invert_contract_stochastic] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [twop_invert_contract_stochastic] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[twop_invert_contract_stochastic] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[twop_invert_contract_stochastic] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [twop_invert_contract_stochastic] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  size_t nelem = _GSI( VOLUME+RAND );
  double ** spinor_work  = init_2level_dtable ( 2, nelem );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * allocate memory for spinor fields
   * WITHOUT halo
   ***************************************************************************/
  nelem = _GSI( VOLUME );
  double ** stochastic_propagator_list = init_2level_dtable ( g_nsample, nelem );
  if ( stochastic_propagator_list == NULL ) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double ** stochastic_source_list = init_2level_dtable ( g_nsample, nelem );
  if ( stochastic_source_list == NULL ) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }
  /* for ( int i = 0; i < rlxd_size(); i++ ) {
    fprintf ( stdout, "rng %2d %10d\n", g_cart_id, rng_state[i] );
  } */

#if ( defined HAVE_HDF5 )
    sprintf ( output_filename, "%s.%.4d.noise%d.h5", outfile_prefix, Nconf, g_noise_type );
#endif
    if(io_proc == 2 && g_verbose > 1 ) { 
      fprintf(stdout, "# [twop_invert_contract_stochastic] writing data to file %s\n", output_filename);
    }

  /***************************************************************************
   * generate all stochastic sources
   ***************************************************************************/
  for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

    exitstatus = prepare_volume_source ( stochastic_source_list[isample], VOLUME );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[twop_invert_contract_stochastic] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(64);
    }
  }

  double *** xid_xi = init_3level_dtable ( ( g_nsample * ( g_nsample - 1 ) ) / 2, g_source_momentum_number, 2 * T );
  if ( xid_xi == NULL ) {
    fprintf(stderr, "[twop_invert_contract_stochastic] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(64);
  }

  int r12 = 0;
  for ( int r1 = 0; r1 < g_nsample; r1++ ) {
    for ( int r2 = r1+1; r2 < g_nsample; r2++ ) {

      double ** xx = init_2level_dtable ( T, 2*VOL3 );
      if ( xx == NULL ) {
        fprintf(stderr, "[twop_invert_contract_stochastic] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(64);
      }

#pragma omp parallel for
      for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {

        unsigned int const iix = _GSI ( ix );

        _d2_pl_eq_fv_dag_ti_fv ( xx[0] + 2*ix, stochastic_source_list[r2] + iix, stochastic_source_list[r1] + iix );
      }

      exitstatus = momentum_projection ( xx[0], xid_xi[r12][0], T, g_source_momentum_number, g_source_momentum_list );
      if(exitstatus != 0) {
        fprintf(stderr, "[twop_invert_contract_stochastic] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

      fini_2level_dtable ( &xx );

      r12++;
    }
  }

  if(io_proc>0) {

    size_t items = ( g_nsample * ( g_nsample - 1 ) ) / 2 * g_source_momentum_number * 2 * T_global;

#ifdef HAVE_MPI
    double * buffer = ( io_proc == 2 ) ?  init_1level_dtable ( items ) : NULL;

    int mitems = ( g_nsample * ( g_nsample - 1 ) ) / 2 * g_source_momentum_number * 2 * T;

    exitstatus = MPI_Gather ( xid_xi[0][0], mitems, MPI_DOUBLE, buffer, mitems, MPI_DOUBLE, 0, g_tr_comm);
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[twop_invert_contract_stochastic] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(3);
    }
#else
    double * buffer = xid_xi[0][0];
#endif

    char tag[400];
    sprintf ( tag, "/xi+-xi" );

    int const ncdim = 3;
    int const cdim[3] = { ( g_nsample * ( g_nsample - 1 ) ) / 2, g_source_momentum_number , 2 * T_global };

    if ( io_proc == 2 ) {
  
      exitstatus = write_h5_contraction ( buffer, NULL, output_filename, tag, "double", ncdim, cdim );
      if(exitstatus != 0 ) {
        fprintf(stderr, "[twop_invert_contract_stochastic] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

    }

#ifdef HAVE_MPI
    if ( io_proc == 2 ) fini_1level_dtable ( &buffer );
#endif

  }

  fini_3level_dtable ( &xid_xi );


  /***************************************************************************
   * stochastic propagators
   ***************************************************************************/

  for ( int gts = 0; gts < T_global; gts++ ) {

    int lts = -1, proc_id = -1;
    
    exitstatus = get_timeslice_source_info (gts, &lts, &proc_id );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[twop_invert_contract_stochastic] Error from get_timeslice_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(3);
    }

   int r12 = 0;
  
   for ( int r1 = 0; r1 < g_nsample; r1++ ) {

      memset ( spinor_work[1], 0, sizeof_spinor_field );

      if ( proc_id == g_cart_id ) {

        size_t const offset = _GSI( lts * VOL3 );
      
        memcpy ( spinor_work[0] + offset, stochastic_source_list[r1] + offset , sizeof_spinor_field_timeslice );

      }

      exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], _OP_ID_UP );
      if(exitstatus < 0) {
        fprintf(stderr, "[twop_invert_contract_stochastic] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(44);
      }

      if ( check_propagator_residual ) {
        check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[_OP_ID_UP], mzzinv[_OP_ID_UP], 1 );
      }

      memcpy( stochastic_propagator_list[r1], spinor_work[1], sizeof_spinor_field );
   }
       

   /***************************************************************************
    *
    ***************************************************************************/
 
    double *** psid_psi = init_3level_dtable ( ( g_nsample * ( g_nsample - 1 ) ) / 2, g_sink_momentum_number, 2 * T );
    if ( psid_psi == NULL ) {
      fprintf(stderr, "[twop_invert_contract_stochastic] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(64);
    }

    int r12 = 0;
    for ( int r1 = 0; r1 < g_nsample; r1++ ) {
      for ( int r2 = r1+1; r2 < g_nsample; r2++ ) {

        double ** xx = init_2level_dtable ( T, 2*VOL3 );
        if ( xx == NULL ) {
          fprintf(stderr, "[twop_invert_contract_stochastic] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(64);
        }

#pragma omp parallel for
        for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {

          unsigned int const iix = _GSI ( ix );

          _d2_pl_eq_fv_dag_ti_fv ( xx[0] + 2*ix, stochastic_source_list[r2] + iix, stochastic_source_list[r1] + iix );
        }

        exitstatus = momentum_projection ( xx[0], psid_psi[r12][0], T, g_sink_momentum_number, g_sink_momentum_list );
        if(exitstatus != 0) {
          fprintf(stderr, "[twop_invert_contract_stochastic] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(3);
        }

        fini_2level_dtable ( &xx );

        r12++;
      }
    }

    if(io_proc>0) {

      size_t items = ( g_nsample * ( g_nsample - 1 ) ) / 2 * g_sink_momentum_number * 2 * T_global;

#ifdef HAVE_MPI
     double * buffer = ( io_proc == 2 ) ?  init_1level_dtable ( items ) : NULL;
 
      int mitems = ( g_nsample * ( g_nsample - 1 ) ) / 2 * g_sink_momentum_number * 2 * T;

      exitstatus = MPI_Gather ( xid_xi[0][0], mitems, MPI_DOUBLE, buffer, mitems, MPI_DOUBLE, 0, g_tr_comm);
      if(exitstatus != MPI_SUCCESS) {
        fprintf(stderr, "[twop_invert_contract_stochastic] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }
#else
      double * buffer = xid_xi[0][0];
#endif

      char tag[400];
      sprintf ( tag, "/psi+-psi/%c-%c/t%d", flavor_tag[_OP_ID_UP], flavor_tag[_OP_ID_UP], gts );

      int const ncdim = 3;
      int const cdim[3] = { ( g_nsample * ( g_nsample - 1 ) ) / 2, g_sink_momentum_number , 2 * T_global };

      if ( io_proc == 2 ) {
  
        exitstatus = write_h5_contraction ( buffer, NULL, output_filename, tag, "double", ncdim, cdim );
        if(exitstatus != 0 ) {
          fprintf(stderr, "[twop_invert_contract_stochastic] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(3);
        }

      }

#ifdef HAVE_MPI
      if ( io_proc == 2 ) fini_1level_dtable ( &buffer );
#endif

    }

    fini_3level_dtable ( &psid_psi );

  }  /* end of loop on global timeslices */

  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_2level_dtable ( &stochastic_propagator_list );
  fini_2level_dtable ( &stochastic_source_list );
  fini_2level_dtable ( &spinor_work );

  /***************************************************************************
   * fini rng state
   ***************************************************************************/
  fini_rng_state ( &rng_state);

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  free( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( &mzz, &mzzinv );


#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif

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

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [twop_invert_contract_stochastic] %s# [twop_invert_contract_stochastic] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [twop_invert_contract_stochastic] %s# [twop_invert_contract_stochastic] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
