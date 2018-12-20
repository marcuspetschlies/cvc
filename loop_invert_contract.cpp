/****************************************************
 * loop_invert_contract
 *
 * PURPOSE:
 * DONE:
 * TODO:
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
#include "contract_loop.h"
#include "ranlxd.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_ST 2

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to calculate loop inversions + contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "loop";

  const char fbwd_str[2][4] =  { "fwd", "bwd" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  size_t sizeof_spinor_field;
  char filename[100];

  struct timeval ta, tb;

  double **mzz[2]    = { NULL, NULL }, **mzzinv[2]    = { NULL, NULL };
  double **DW_mzz[2] = { NULL, NULL }, **DW_mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  int op_id_up = -1;
  /* int op_id_dn = -1; */
  char output_filename[400];
  int * rng_state = NULL;

  char data_tag[400];

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

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) sprintf ( filename, "%s.input", outfile_prefix );
  /* fprintf(stdout, "# [loop_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [loop_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [loop_invert_contract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);

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
    if(g_cart_id==0) fprintf(stdout, "# [loop_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [loop_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[loop_invert_contract] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[loop_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[loop_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[loop_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[loop_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv for Wilson Dirac-operator
   * i.e. twisted mass = 0
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &DW_mzz, &DW_mzzinv, gauge_field_with_phase, 0., g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[loop_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [loop_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************
   * set operator ids depending on fermion type
   ***********************************************************/
  if ( g_fermion_type == _TM_FERMION ) {
    op_id_up = 0;
    /* op_id_dn = 1; */
  } else if ( g_fermion_type == _WILSON_FERMION ) {
    op_id_up = 0;
    /* op_id_dn = 0; */
  }

  /***************************************************************************
   * allocate memory for full-VOLUME spinor fields 
   * WITH HALO
   ***************************************************************************/
  size_t nelem = _GSI( VOLUME+RAND );
  double ** spinor_work  = init_2level_dtable ( 3, nelem );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * half-VOLUME spinor fields
   * WITH HALO
   * no additional memory, just split up the spinor_work fields
   ***************************************************************************/
  double * eo_spinor_work[6] = {
    spinor_work[0], 
    spinor_work[0] + nelem / 2, 
    spinor_work[1], 
    spinor_work[1] + nelem / 2, 
    spinor_work[2], 
    spinor_work[2] + nelem / 2 };

  /***************************************************************************
   * allocate memory for spinor fields
   * WITHOUT halo
   ***************************************************************************/
  nelem = _GSI( VOLUME );
  double * stochastic_propagator = init_1level_dtable ( nelem );
  if ( stochastic_propagator == NULL ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double * DW_stochastic_propagator = init_1level_dtable ( nelem );
  if ( stochastic_propagator == NULL ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double * stochastic_propagator_deriv = init_1level_dtable ( nelem );
  if ( stochastic_propagator == NULL ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double * stochastic_propagator_dderiv = init_1level_dtable ( nelem );
  if ( stochastic_propagator == NULL ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double * stochastic_source = init_1level_dtable ( nelem );
  if ( stochastic_source == NULL ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  /***************************************************************************
   * allocate memory for contractions
   ***************************************************************************/
  double *** loop = init_3level_dtable ( T, g_sink_momentum_number, 32 );
  if ( loop == NULL ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }


  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

#if ( defined HAVE_HDF5 )
  sprintf ( output_filename, "%s.%.4d.h5", outfile_prefix, Nconf );
#endif
  if(io_proc == 2 && g_verbose > 1 ) { 
    fprintf(stdout, "# [loop_invert_contract] writing data to file %s\n", output_filename);
  }

  /***************************************************************************
   * loop on stochastic oet samples
   ***************************************************************************/
  for ( int isample = 0; isample < g_nsample; isample++ ) {

    /***************************************************************************
     * synchronize rng states to state at zero
     *
     * This should not be neccessary!
     ***************************************************************************/
    /*
    exitstatus = sync_rng_state ( rng_state, 0, 0 );
    if(exitstatus != 0) {
      fprintf(stderr, "[loop_invert_contract] Error from sync_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    } */

    /***************************************************************************
     * read stochastic oet source from file
     ***************************************************************************/
    if ( g_read_source ) {
      sprintf(filename, "%s.%.4d.%.5d", filename_prefix, Nconf, isample);
      if ( ( exitstatus = read_lime_spinor( stochastic_source, filename, 0) ) != 0 ) {
        fprintf(stderr, "[loop_invert_contract] Error from read_lime_spinor, status was %d\n", exitstatus);
        EXIT(2);
      }

    /***************************************************************************
     * generate stochastic volume source
     ***************************************************************************/
    } else {

      if( ( exitstatus = prepare_volume_source ( stochastic_source, VOLUME ) ) != 0 ) {
        fprintf(stderr, "[loop_invert_contract] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(64);
      }
      if ( g_write_source ) {
        sprintf(filename, "%s.%.4d.%.5d", filename_prefix, Nconf, isample);
        if ( ( exitstatus = write_propagator( stochastic_source, filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[loop_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }
    }  /* end of if read stochastic source - else */

    /***************************************************************************
     * retrieve current rng state and 0 writes his state
     ***************************************************************************/
    exitstatus = get_rng_state ( rng_state );
    if(exitstatus != 0) {
      fprintf(stderr, "[loop_invert_contract] Error from get_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    exitstatus = save_rng_state ( 0, NULL );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[loop_invert_contract] Error from save_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
      EXIT(38);
    }

    /***************************************************************************
     * invert for stochastic propagator
     *   up flavor
     ***************************************************************************/
    memcpy ( spinor_work[0], stochastic_source, sizeof_spinor_field );

    memset ( spinor_work[1], 0, sizeof_spinor_field );

    exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], op_id_up );
    if(exitstatus < 0) {
      fprintf(stderr, "[loop_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(44);
    }

    if ( check_propagator_residual ) {
      memcpy ( spinor_work[0], stochastic_source, sizeof_spinor_field );
      check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_up], mzzinv[op_id_up], 1 );
    }

    memcpy( stochastic_propagator, spinor_work[1], sizeof_spinor_field);

    if ( g_write_propagator ) {
      sprintf(filename, "%s.%.4d.%.5d.inverted", filename_prefix, Nconf, isample);
      if ( ( exitstatus = write_propagator( stochastic_propagator, filename, 0, g_propagator_precision) ) != 0 ) {
        fprintf(stderr, "[loop_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }
    }

    gettimeofday ( &ta, (struct timezone *) NULL );

    /***************************************************************************
     * apply Wilson Dirac operator
     ***************************************************************************/

    /* decompose lexic stochastic_propagator into even/odd eo_spinor_work */
    spinor_field_lexic2eo ( stochastic_propagator, eo_spinor_work[0], eo_spinor_work[1] );
    /* apply D_W */
    Q_clover_phi_matrix_eo ( eo_spinor_work[2],  eo_spinor_work[3],  eo_spinor_work[0],  eo_spinor_work[1], gauge_field_with_phase,  eo_spinor_work[4], DW_mzz[op_id_up]);
    /* compose full spinor field */
    spinor_field_eo2lexic ( DW_stochastic_propagator, eo_spinor_work[2], eo_spinor_work[3] );

    gettimeofday ( &tb, (struct timezone *)NULL );

    show_time ( &ta, &tb, "loop_invert_contract", "DW", io_proc == 2 );
 
    /***************************************************************************
     *
     * contraction for local loops using std one-end-trick
     *
     ***************************************************************************/

    /***************************************************************************
     * group name for contraction
     ***************************************************************************/
    sprintf ( data_tag, "/conf_%.4d/nstoch_%.4d/%s", Nconf, isample, "Scalar" );

    /***************************************************************************
     * loop contractions
     ***************************************************************************/
    exitstatus = contract_local_loop_stochastic ( loop, stochastic_propagator, stochastic_propagator, g_sink_momentum_number, g_sink_momentum_list );
    if(exitstatus != 0) {
      fprintf(stderr, "[loop_invert_contract] Error from contract_local_loop_stochastic, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(44);
    }

    /***************************************************************************
     * write contraction to file
     ***************************************************************************/
#ifdef HAVE_HDF5
    exitstatus = contract_loop_write_to_h5_file ( loop, output_filename, data_tag, g_sink_momentum_number, 16, io_proc );
#else
    exitstatus = 1;
#endif
    if(exitstatus != 0) {
      fprintf(stderr, "[loop_invert_contract] Error from contract_loop_write_to_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(44);
    }

    /***************************************************************************
     *
     * contraction for local loops using gen one-end-trick
     *
     ***************************************************************************/

    /***************************************************************************
     * group name for contraction
     ***************************************************************************/
    sprintf ( data_tag, "/conf_%.4d/nstoch_%.4d/%s", Nconf, isample, "dOp" );

    /***************************************************************************
     * loop contractions
     ***************************************************************************/
    exitstatus = contract_local_loop_stochastic ( loop, DW_stochastic_propagator, stochastic_propagator, g_sink_momentum_number, g_sink_momentum_list );
    if(exitstatus != 0) {
      fprintf(stderr, "[loop_invert_contract] Error from contract_local_loop_stochastic, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(44);
    }

    /***************************************************************************
     * write contraction to file
     ***************************************************************************/
#ifdef HAVE_HDF5
    exitstatus = contract_loop_write_to_h5_file ( loop, output_filename, data_tag, g_sink_momentum_number, 16, io_proc );
#else
    exitstatus = 1;
#endif
    if(exitstatus != 0) {
      fprintf(stderr, "[loop_invert_contract] Error from contract_loop_write_to_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(44);
    }

    /*****************************************************************
     *
     * contractions for single and double cov displ
     * insertion with std and gen one-end-trick
     *
     *****************************************************************/

    /*****************************************************************
     * loop on fbwd for cov deriv
     *****************************************************************/
    for ( int ifbwd = 0; ifbwd <= 1; ifbwd++ ) {

      /*****************************************************************
       * loop on directions for cov deriv
       *****************************************************************/
      for ( int mu = 0; mu < 4; mu++ ) {

        gettimeofday ( &ta, (struct timezone *)NULL );
        
        /*****************************************************************
         * apply cov deriv direction = mu and fbwd = ifbwd
         *****************************************************************/

        spinor_field_eq_cov_deriv_spinor_field ( stochastic_propagator_deriv, stochastic_propagator, mu, ifbwd, gauge_field_with_phase );

        gettimeofday ( &tb, (struct timezone *)NULL );

        show_time ( &ta, &tb, "loop_invert_contract", "spinor_field_eq_cov_deriv_spinor_field", io_proc == 2 );

        /***************************************************************************
         * group name for contraction
         * std one-end-trick, single deriv
         ***************************************************************************/
        sprintf ( data_tag, "/conf_%.4d/nstoch_%.4d/%s/fbwd%d/dir%d", Nconf, isample, "LoopsD", ifbwd, mu );

        /***************************************************************************
         * loop contractions
         * std one-end-trick
         ***************************************************************************/
        exitstatus = contract_local_loop_stochastic ( loop, stochastic_propagator, stochastic_propagator_deriv, g_sink_momentum_number, g_sink_momentum_list );
        if( exitstatus != 0) {
          fprintf(stderr, "[loop_invert_contract] Error from contract_local_loop_stochastic, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        /***************************************************************************
         * write contraction to file
         ***************************************************************************/
#ifdef HAVE_HDF5
        exitstatus = contract_loop_write_to_h5_file ( loop, output_filename, data_tag, g_sink_momentum_number, 16, io_proc );
#else
        exitstatus = 1;
#endif
        if(exitstatus != 0) {
          fprintf(stderr, "[loop_invert_contract] Error from contract_loop_write_to_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        /***************************************************************************
         * group name for contraction
         * gen one-end-trick, single displ
         ***************************************************************************/
        sprintf ( data_tag, "/conf_%.4d/nstoch_%.4d/%s/%s/dir%d", Nconf, isample, "DW_LoopsD", fbwd_str[ifbwd], mu );

        /***************************************************************************
         * loop contractions
         * std one-end-trick
         ***************************************************************************/
        exitstatus = contract_local_loop_stochastic ( loop, DW_stochastic_propagator, stochastic_propagator_deriv, g_sink_momentum_number, g_sink_momentum_list );
        if( exitstatus != 0) {
          fprintf(stderr, "[loop_invert_contract] Error from contract_local_loop_stochastic, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        /***************************************************************************
         * write contraction to file
         ***************************************************************************/
#ifdef HAVE_HDF5
        exitstatus = contract_loop_write_to_h5_file ( loop, output_filename, data_tag, g_sink_momentum_number, 16, io_proc );
#else
        exitstatus = 1;
#endif
        if(exitstatus != 0) {
          fprintf(stderr, "[loop_invert_contract] Error from contract_loop_write_to_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        /***************************************************************************
         * 2nd loop on fwd / bwd
         ***************************************************************************/
        for ( int kfbwd = 0; kfbwd <= 1; kfbwd++ ) {

          spinor_field_eq_cov_deriv_spinor_field ( stochastic_propagator_dderiv, stochastic_propagator_deriv, mu, kfbwd, gauge_field_with_phase );

          /***************************************************************************
           * group name for contraction
           * std one-end-trick, double displ
           ***************************************************************************/
          sprintf ( data_tag, "/conf_%.4d/nstoch_%.4d/%s/%s/%s/dir%d", Nconf, isample, "LoopsDD", fbwd_str[kfbwd], fbwd_str[ifbwd], mu );

          /***************************************************************************
           * loop contractions
           * std one-end-trick
           ***************************************************************************/
          exitstatus = contract_local_loop_stochastic ( loop, stochastic_propagator, stochastic_propagator_dderiv, g_sink_momentum_number, g_sink_momentum_list );
          if( exitstatus != 0) {
            fprintf(stderr, "[loop_invert_contract] Error from contract_local_loop_stochastic, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(44);
          }

          /***************************************************************************
           * write contraction to file
           ***************************************************************************/
#ifdef HAVE_HDF5
          exitstatus = contract_loop_write_to_h5_file ( loop, output_filename, data_tag, g_sink_momentum_number, 16, io_proc );
#else
          exitstatus = 1;
#endif
          if(exitstatus != 0) {
            fprintf(stderr, "[loop_invert_contract] Error from contract_loop_write_to_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(44);
          }

          /***************************************************************************
           * group name for contraction
           * gen one-end-trick, double displ
           ***************************************************************************/
          sprintf ( data_tag, "/conf_%.4d/nstoch_%.4d/%s/fbwd%d/fbwd%d/dir%d", Nconf, isample, "DW_LoopsDD", kfbwd, ifbwd, mu );

        /***************************************************************************
         * loop contractions
         * std one-end-trick
         ***************************************************************************/
        exitstatus = contract_local_loop_stochastic ( loop, DW_stochastic_propagator, stochastic_propagator_dderiv, g_sink_momentum_number, g_sink_momentum_list );
        if( exitstatus != 0) {
          fprintf(stderr, "[loop_invert_contract] Error from contract_local_loop_stochastic, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        /***************************************************************************
         * write contraction to file
         ***************************************************************************/
#ifdef HAVE_HDF5
        exitstatus = contract_loop_write_to_h5_file ( loop, output_filename, data_tag, g_sink_momentum_number, 16, io_proc );
#else
        exitstatus = 1;
#endif
        if(exitstatus != 0) {
          fprintf(stderr, "[loop_invert_contract] Error from contract_loop_write_to_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        }  /* end of loop on fbwd */

      }  /* end of loop on directions for mu for displ */

    }  /* end of loop on fbwd */

    /*****************************************************************/
    /*****************************************************************/

  }  /* end of loop on oet samples */

  /***************************************************************************
   * decallocate fields
   ***************************************************************************/
  fini_1level_dtable ( &stochastic_propagator        );
  fini_1level_dtable ( &DW_stochastic_propagator     );
  fini_1level_dtable ( &stochastic_propagator_deriv  );
  fini_1level_dtable ( &stochastic_propagator_dderiv );
  fini_1level_dtable ( &stochastic_source            );
  fini_2level_dtable ( &spinor_work                  );
  fini_3level_dtable ( &loop                         );


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

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [loop_invert_contract] %s# [loop_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [loop_invert_contract] %s# [loop_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
