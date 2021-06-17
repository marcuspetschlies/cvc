/****************************************************
 * npv_fht_invert_contract
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
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "contract_factorized.h"

#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to 2-pt function inversion and contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "npv_fht";

  int c;
  int filename_set = 0;
  int gsx[4], sx[4];
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[100];
  // double ratime, retime;
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;

  const int gamma_f1_nucleon_number                                = 1;
  int gamma_f1_nucleon_list[gamma_f1_nucleon_number]               = { 14 }; /*, 11,  8,  2 }; */
  double gamma_f1_nucleon_sign[gamma_f1_nucleon_number]            = { +1 }; /*, +1, -1, -1 }; */
  /* double gamma_f1_nucleon_transposed_sign[gamma_f1_nucleon_number] = { -1, -1, +1, -1 }; */

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char aff_tag[500];
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

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) strcpy(filename, "twopt.input");
  /* fprintf(stdout, "# [npv_fht_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [npv_fht_invert_contract] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  /* exitstatus = tmLQCD_invert_init(argc, argv, 1, 0); */
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
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

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [npv_fht_invert_contract] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [npv_fht_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [npv_fht_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[npv_fht_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[npv_fht_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  size_t sizeof_spinor_field = _GSI( VOLUME ) * sizeof ( double );

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [npv_fht_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [npv_fht_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[npv_fht_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[npv_fht_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[npv_fht_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************************
   * initialize clover, lmzz and lmzzinv
   ***********************************************************/
  exitstatus = init_clover ( &lmzz, &lmzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[npv_fht_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[npv_fht_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [npv_fht_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   ***************************************************************************
   **
   ** packed loops
   **
   ***************************************************************************
   ***************************************************************************/

  /***********************************************************
   * allocate fields
   ***********************************************************/
  double *** loop_field = init_3level_dtable ( VOLUME, 12, 24 );
  if ( loop_field == NULL ) {
    fprintf(stderr, "[npv_fht_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(122);
  }

  double *** loop_field_sum = init_3level_dtable ( VOLUME, 12, 24 );
  if ( loop_field_sum == NULL ) {
    fprintf(stderr, "[npv_fht_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(122);
  }

  double ** spinor_work = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
  if( spinor_work == NULL ) {
    fprintf(stderr, "[npv_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(121);
  }
  
  double * stochastic_source = init_1level_dtable ( _GSI( VOLUME ) );
  if( stochastic_source == NULL ) {
    fprintf(stderr, "[npv_fht_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(120);
  }
  
  double ** stochastic_propagator = init_2level_dtable ( 12, _GSI( VOLUME ) );
  if( stochastic_propagator == NULL ) {
    fprintf(stderr, "[npv_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(119);
  }

  double **** loop_ti_loop_sum = init_4level_dtable ( 12, 12, 12, 24 );
  if( loop_ti_loop_sum == NULL ) {
    fprintf(stderr, "[npv_fht_invert_contract] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(118);
  }

  double **** loop_ti_loop_diag = init_4level_dtable ( 12, 12, 12, 24 );
  if( loop_ti_loop_diag == NULL ) {
    fprintf(stderr, "[npv_fht_invert_contract] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(118);
  }


  memset ( loop_field[0][0], 288 * VOLUME * sizeof(double) );
  /***********************************************************
   * fill the loop field
   ***********************************************************/
  for ( int isample = 0; isample < g_nsample; isample++ ) {

    exitstatus = prepare_volume_source ( stochastic_source, VOLUME );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(118);
    }

    for ( int gts = 0; gts < T_global; gts++ ) {

      int const have_source = ( gts / T == g_proc_coords[0] );
      int const lts = have_source ? gts % T : -1;

      memset ( spinor_work[1], 0, sizeof_spinor_field );

      for ( int isc = 0; isc < 12; isc++ ) {

        if ( have_source ) {
          size_t const offset = lts * _GSI( VOL3 );
          size_t const bytes  = _GSI( VOL3 ) * sizeof(double);

          /* set spin-color source */
#pragma omp parallel for
          for( size_t ix = 0; ix < VOL3; ix++ ) {
            size_t const iix = offset + _GSI(ix) + 2 * isc;
            spinor_work[0][iix  ] = stochastic_source[iix  ];
            spinor_work[0][iix+1] = stochastic_source[iix+1];
          }

        }  /* end of if have source */

        if( g_fermion_type == _TM_FERMION ) spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);

        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], _OP_ID_UP);

        if(exitstatus < 0) {
          fprintf(stderr, "[npv_fht_invert_contract] Error from invert, status was %d\n", exitstatus);
          EXIT(12);
        }

        if ( check_propagator_residual ) {
          memcpy ( spinor_work[0], stochastic_source, sizeof_spinor_field );
          exitstatus = check_residual_clover ( &(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, lmzz[_OP_ID_UP], 1  );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[npv_fht_invert_contract] Error from check_residual_clover, status was %d %s  %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(3);
          }
        }

        if( g_fermion_type == _TM_FERMION ) spinor_field_tm_rotation ( spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);

        /* source process copy timeslice of solution vector */
        if ( have_source ) {
          size_t const offset = lts * _GSI( VOL3 );
          memcpy ( stochastic_propagator[isc] + offset, spinor_work[1] + offset, sizeof_spinor_field_timeslice );
        }  /* end of if have source */

      }  /* end of loop on spin-color */

    }  /* end of loop on timeslices */


    /***********************************************************
     * contract packed vectors to loop
     *
     * note: no reduction in spin and color
     *
     ***********************************************************/
    exitstatus = contract_loop_spin_color_open ( loop_field, stochastic_source, stochastic_propagator );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[npv_fht_invert_contract] Error from contract_loop_spin_color_open, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(117);
    }

    /***********************************************************
     * cumulative sum of space-time dependent loop fields
     ***********************************************************/
#pragma omp parallel for
    for ( size_t ix = 0; ix < _GSI(VOLUME); ix++ ) {
      loop_field_sum[0][0][ix] += loop_field[0][0][ix];
    }
 
    /***********************************************************
     * loop ti loop
     *
     ***********************************************************/
    exitstatus = loop_ti_loop_reduce_spin_color_open_accum ( loop_ti_loop_diag[0][0][0], loop_field );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[npv_fht_invert_contract] Error from loop_ti_loop_reduce_spin_color_open_accum, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(116);
    }

    /***********************************************************/
    /***********************************************************/

  }  /* end of loop on samples */

  /***********************************************************
   * sum of loops times sum of loops
   ***********************************************************/
  exitstatus = loop_ti_loop_reduce_spin_color_open_accum ( loop_ti_loop_sum[0][0][0], loop_field_sum );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[npv_fht_invert_contract] Error from loop_ti_loop_reduce_spin_color_open_accum, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(115);
  }

  /***********************************************************
   * write to AFF file
   ***********************************************************/
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    /* open AFF writer */
    sprintf(filename, "%s.%.4d.loop.aff", outfile_prefix, Nconf );
    if ( g_verbose > 0 ) fprintf(stdout, "# [npv_fht_invert_contract] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    char * aff_status_str = (char*)aff_writer_errstr ( affw );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }

    struct AffNode_s * affn = aff_writer_root(affw);
    if( affn == NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      EXIT(2);
    }

    /* write loop field sum */
    sprintf ( aff_tag, "/loop/sum/S%d", g_nsample );
    struct AffNode_s *affdir = aff_writer_mkpath (affw, affn, aff_tag );
    if ( affdir == NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from aff_writer_mkpath for tag %s %s %d\n", aff_tag, __FILE__, __LINE__);
      EXIT(2);
    }

    if ( ( exitstatus = aff_node_put_complex ( affw, affdir, loop_field_sum[0][0], (uint32_t)(12*12*VOLUME) ) ) != 0 ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from aff_node_put_complex for tag %s, status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    /* write loop_ti_loop_diag */
    sprintf ( aff_tag, "/loop_ti_loop/diag/S%d", g_nsample );
    affdir = aff_writer_mkpath (affw, affn, aff_tag );
    if ( affdir == NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from aff_writer_mkpath for tag %s %s %d\n", aff_tag, __FILE__, __LINE__);
      EXIT(2);
    }

    if ( ( exitstatus = aff_node_put_complex ( affw, affdir, loop_ti_loop_diag[0][0][0], (uint32_t)(12*12*12*12) ) ) != 0 ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from aff_node_put_complex for tag %s, status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    /* write loop_ti_loop_sum */
    sprintf ( aff_tag, "/loop_ti_loop/sum/S%d", g_nsample );
    affdir = aff_writer_mkpath (affw, affn, aff_tag );
    if ( affdir == NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from aff_writer_mkpath for tag %s %s %d\n", aff_tag, __FILE__, __LINE__);
      EXIT(2);
    }

    if ( ( exitstatus = aff_node_put_complex ( affw, affdir, loop_ti_loop_sum[0][0][0], (uint32_t)(12*12*12*12) ) ) != 0 ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from aff_node_put_complex for tag %s, status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    /* close AFF writer */
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(2);
    }

  }  /* end of if io_proc == 2 */
#endif

  /***********************************************************
   * deallocate
   *
   * loop_field_sum will be needed below
   * to construct sequential sources
   ***********************************************************/
  fini_1level_dtable ( &stochastic_source );
  fini_3level_dtable ( &stochastic_propagator );
  
  fini_4level_dtable ( &loop_ti_loop_sum );
  fini_4level_dtable ( &loop_ti_loop_diag );
  fini_3level_dtable ( &loop_field );

  STOPPED HERE

  /***************************************************************************
   ***************************************************************************
   **
   ** point-to-all propagators
   **
   ***************************************************************************
   ***************************************************************************/

  /***********************************************************
   * loop on source locations
   ***********************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

    double ** propagator_up = init_2level_dtable ( 12, _GSI(VOLUME) );
    if ( propagator_up == NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    double ** propagator_dn = init_2level_dtable ( 12, _GSI(VOLUME) );
    if ( propagator_dn == NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    double ** sequential_propagator = init_2level_dtable ( 12, _GSI(VOLUME) );
    if ( sequential_propagator == NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /***********************************************************
     * determine source coordinates, find out,
     * if source_location is in this process
     ***********************************************************/
    gsx[0] = ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global;
    gsx[1] = ( g_source_coords_list[isource_location][1] + LX_global ) % LX_global;
    gsx[2] = ( g_source_coords_list[isource_location][2] + LY_global ) % LY_global;
    gsx[3] = ( g_source_coords_list[isource_location][3] + LZ_global ) % LZ_global;

    int source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

#ifdef HAVE_LHPC_AFF
    /***********************************************
     * writer for aff output file
     ***********************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [npv_fht_invert_contract] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[npv_fht_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#endif

    /**********************************************************
     * propagators with source at gsx
     **********************************************************/

    /***********************************************************
     * up-type point-to-all propagator
     *
     * source smearing yes
     * sink   smeaming no
     ***********************************************************/
    exitstatus = point_source_propagator ( &(spinor_field[0]), gsx, _OP_ID_UP, 1, 0, NULL, check_propagator_residual, gauge_field_with_phase, lmzz );
    if(exitstatus != 0) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from point_source_propagator, status was %d\n", exitstatus);
      EXIT(12);
    }

    /***********************************************************
     * dn-type point-to-all propagator
     *
     * source smearing yes
     * sink   smearing no
     ***********************************************************/
    exitstatus = point_source_propagator ( &(spinor_field[12]), gsx, _OP_ID_DN, 1, 0, NULL, check_propagator_residual, gauge_field_with_phase, lmzz );
    if(exitstatus != 0) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from point_source_propagator, status was %d\n", exitstatus);
      EXIT(12);
    }

    /***************************************************************************
     * Nucleon - Nucleon correlation function
     ***************************************************************************/

    /* allocate propagator fields */
    fermion_propagator_type * fp  = create_fp_field ( VOLUME );
    fermion_propagator_type * fp2 = create_fp_field ( VOLUME );
    fermion_propagator_type * fp3 = create_fp_field ( VOLUME );

    /* up propagator as propagator type field */
    assign_fermion_propagator_from_spinor_field ( fp,  &(spinor_field[ 0]), VOLUME);

    /* dn propagator as propagator type field */
    assign_fermion_propagator_from_spinor_field ( fp2, &(spinor_field[12]), VOLUME);

    double ** v2 = init_2level_dtable ( VOLUME, 32 );
    if ( v2 == NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    double *** vp = init_3level_dtable ( T, g_sink_momentum_number, 32 );
    if ( vp == NULL ) {
      fprintf(stderr, "[npv_fht_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(47);
    }

    /***********************************************************
     * contractions for N - N with up and dn propagagor
     ***********************************************************/
    for ( int if1 = 0; if1 < gamma_f1_nucleon_number; if1++ ) {
    for ( int if2 = 0; if2 < gamma_f1_nucleon_number; if2++ ) {

      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_nucleon_list[if2], fp2, VOLUME );

      fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_nucleon_list[if1], fp3, VOLUME );

      fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_nucleon_sign[if1]*gamma_f1_nucleon_sign[if2], VOLUME );

      sprintf(aff_tag, "/N-N/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/n1",
          gsx[0], gsx[1], gsx[2], gsx[3],
          gamma_f1_nucleon_list[if1], gamma_f1_nucleon_list[if2]);

      exitstatus = contract_v5 ( v2, fp, fp3, fp, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[npv_fht_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[npv_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[npv_fht_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
        EXIT(49);
      }

      /***********************************************************/
      /***********************************************************/

      sprintf(aff_tag, "/N-N/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/n2",
          gsx[0], gsx[1], gsx[2], gsx[3],
          gamma_f1_nucleon_list[if1], gamma_f1_nucleon_list[if2]);

      exitstatus = contract_v6 ( v2, fp, fp3, fp, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[npv_fht_invert_contract] Error from contract_v6, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(48);
      }

      exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[npv_fht_invert_contract] Error from contract_vn_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(48);
      }

      exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[npv_fht_invert_contract] Error from contract_vn_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(49);
      }

    }}  /* end of loop on gamma_f1_nucleon at source and sink */

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * sequential inversions with loop field sum
     ***********************************************************/

    double *** loop_field_sum_gamma_conjugate = init_3level_dtable ( VOLUME, 12, 24 );
    if ( loop_field_sum_gamma_conjugate == NULL ) {
      fprintf ( stderr, "[npv_fht_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(49);
    }
    memset ( loop_field_sum_gamma_conjugate, 0, 12*12*VOLUME*2*sizeof(double) );

    /* gamma_mu ... gamma_mu */
    fermion_propagator_field_pl_eq_gamma_ti_fermion_propagator_field_ti_gamma ( loop_field_sum_gamma_conjugate, 0, loop_field_sum,                 0, VOLUME );
    fermion_propagator_field_pl_eq_gamma_ti_fermion_propagator_field_ti_gamma ( loop_field_sum_gamma_conjugate, 1, loop_field_sum_gamma_conjugate, 1, VOLUME );
    fermion_propagator_field_pl_eq_gamma_ti_fermion_propagator_field_ti_gamma ( loop_field_sum_gamma_conjugate, 2, loop_field_sum_gamma_conjugate, 2, VOLUME );
    fermion_propagator_field_pl_eq_gamma_ti_fermion_propagator_field_ti_gamma ( loop_field_sum_gamma_conjugate, 3, loop_field_sum_gamma_conjugate, 3, VOLUME );

STOPPED HERE


    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * clean up
     ***********************************************************/
    free_fp_field ( &fp  );
    free_fp_field ( &fp2 );
    free_fp_field ( &fp3 );
    fini_2level_dtable ( &v2 );
    fini_3level_dtable ( &vp );
 
    fini_2level_dtable ( &propagator_up );
    fini_2level_dtable ( &propagator_dn );
    fini_2level_dtable ( &sequential_propagator );

    /***************************************************************************/
    /***************************************************************************/

#ifdef HAVE_LHPC_AFF
    /***************************************************************************
     * close AFF writer
     ***************************************************************************/
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[npv_fht_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
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
  free( gauge_field_with_phase );

  fini_2level_dtable ( &spinor_work );

  /* free clover matrix terms */
  fini_clover ( );

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

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [npv_fht_invert_contract] %s# [npv_fht_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [npv_fht_invert_contract] %s# [npv_fht_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
