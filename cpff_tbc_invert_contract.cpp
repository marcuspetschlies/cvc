/****************************************************
 * cpff_tbc_invert_contract.c
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
#include "ranlxd.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_ST 2

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to calculate charged pion FF inversions + contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "cpff_tbc";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  size_t sizeof_spinor_field;
  char filename[100];
  // double ratime, retime;
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  int op_id_up = -1, op_id_dn = -1;
  char output_filename[400];
  int * rng_state = NULL;


  /* int const gamma_current_number = 10;
  int gamma_current_list[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; */
  int const gamma_current_number = 2;
  int gamma_current_list[10] = {0, 1 };

  char data_tag[400];
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
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

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) sprintf ( filename, "%s.input", outfile_prefix );
  /* fprintf(stdout, "# [cpff_tbc_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [cpff_tbc_invert_contract] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
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

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [cpff_tbc_invert_contract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [cpff_tbc_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [cpff_tbc_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[cpff_tbc_invert_contract] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[cpff_tbc_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [cpff_tbc_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************
   * set operator ids depending on fermion type
   ***********************************************************/
  if ( g_fermion_type == _TM_FERMION ) {
    op_id_up = 0;
    op_id_dn = 1;
  } else if ( g_fermion_type == _WILSON_FERMION ) {
    op_id_up = 0;
    op_id_dn = 0;
  }

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  size_t nelem = _GSI( VOLUME+RAND );
  double ** spinor_work  = init_2level_dtable ( 2, nelem );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * allocate memory for spinor fields
   * WITHOUT halo
   ***************************************************************************/
  nelem = _GSI( VOLUME );
  double **** stochastic_propagator_mom_list = init_4level_dtable ( g_source_location_number, g_nsample, g_seq_source_momentum_number, nelem );
  if ( stochastic_propagator_mom_list == NULL ) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double *** stochastic_propagator_zero_list = init_3level_dtable ( g_source_location_number, g_nsample, nelem );
  if ( stochastic_propagator_zero_list == NULL ) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double ** stochastic_source_list = init_2level_dtable ( g_nsample, nelem );
  if ( stochastic_source_list == NULL ) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  double ** sequential_propagator_list = init_2level_dtable ( 1, nelem );
  if ( sequential_propagator_list == NULL ) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   *
   * (I) prepare and keep the stochastic volume sources
   *
   ***************************************************************************/

  /***************************************************************************
   * loop on stochastic samples
   ***************************************************************************/
  for ( int isample = 0; isample < g_nsample; isample++ ) {

    if ( g_read_source ) {
      /***************************************************************************
       * read stochastic source from file
       ***************************************************************************/
      sprintf(filename, "%s.%.4d.%.5d", filename_prefix, Nconf, isample);
      if ( ( exitstatus = read_lime_spinor( stochastic_source_list[isample], filename, 0) ) != 0 ) {
        fprintf(stderr, "[cpff_tbc_invert_contract] Error from read_lime_spinor, status was %d\n", exitstatus);
        EXIT(2);
      }
    } else {
      /***************************************************************************
       * generate stochastic volume source
       ***************************************************************************/
    
      exitstatus = prepare_volume_source( stochastic_source_list[isample], VOLUME );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[cpff_tbc_invert_contract] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(64);
      }
      if ( g_write_source ) {
        sprintf(filename, "%s.%.4d.%.5d", filename_prefix, Nconf, isample);
        if ( ( exitstatus = write_propagator( stochastic_source_list[isample], filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[cpff_tbc_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }

      /***************************************************************************
       * retrieve current rng state and 0 writes his state
       ***************************************************************************/
      exitstatus = get_rng_state ( rng_state );
      if(exitstatus != 0) {
        fprintf(stderr, "[cpff_tbc_invert_contract] Error from get_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }

      exitstatus = save_rng_state ( 0, NULL );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[cpff_tbc_invert_contract] Error from save_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
        EXIT(38);
      }

    }  /* end of if read stochastic source - else */

  }  /* end of loop on samples */


  /***************************************************************************
   *
   * (II) inversions for momentum source, theta = 0
   *
   ***************************************************************************/

  /***************************************************************************
   * multiply the phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * loop on source timeslices
   ***************************************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

    /***************************************************************************
     * local source timeslice and source process ids
     ***************************************************************************/

    int source_timeslice = -1;
    int source_proc_id   = -1;
    int gts              = ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global;

    exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[cpff_tbc_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     * loop on stochastic oet samples
     ***************************************************************************/
    for ( int isample = 0; isample < g_nsample; isample++ ) {

      /***************************************************************************
       * loop on sequential = sink = -source momenta
       ***************************************************************************/
      for ( int imom = 0; imom < g_seq_source_momentum_number; imom++ ) {

        int const source_momentum[3] = {
          g_seq_source_momentum_list[imom][0],
          g_seq_source_momentum_list[imom][1],
          g_seq_source_momentum_list[imom][2] };


        /***************************************************************************
         * extract timeslice from source, add source momentum
         *
         * we abuse the init_sequential_source here, since it is the same
         * operation
         ***************************************************************************/
        memset ( spinor_work[0], 0, sizeof_spinor_field );
        exitstatus = init_sequential_source ( spinor_work[0], stochastic_source_list[isample], gts, source_momentum, 4 );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(64);
        }

        if ( g_write_source ) {
          sprintf(filename, "%s.%.4d.t%d.px%dpy%dpz%d.%.5d", filename_prefix, Nconf, gts,
              source_momentum[0], source_momentum[1], source_momentum[2], isample);
          if ( ( exitstatus = write_propagator( spinor_work[0], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[cpff_tbc_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(2);
          }
        }  /* end of if g_write_sequential_source */

        /***************************************************************************
         * invert
         ***************************************************************************/
        memset ( spinor_work[1], 0, sizeof_spinor_field );

        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], op_id_dn );
        if(exitstatus != 0) {
          fprintf(stderr, "[cpff_tbc_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_dn], 1 );
        }

        memcpy( stochastic_propagator_mom_list[isource_location][isample][imom], spinor_work[1], sizeof_spinor_field);

        if ( g_write_propagator ) {
          for ( int ispin = 0; ispin < 4; ispin++ ) {
            sprintf(filename, "%s.%.4d.t%d.px%dpy%dpz%d.%.5d.inverted", filename_prefix, Nconf, gts, 
                source_momentum[0], source_momentum[1], source_momentum[2], isample);
            if ( ( exitstatus = write_propagator( stochastic_propagator_mom_list[isource_location][isample][imom], filename, 0, g_propagator_precision) ) != 0 ) {
              fprintf(stderr, "[cpff_tbc_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }
          }
        }

      }  /* end of loop on momenta */

    }  /* end of loop on stochastic samples */

  }  /* end of loop on source timeslices */

  /***************************************************************************
   * free clover term matrices
   ***************************************************************************/
  fini_clover ( &mzz, &mzzinv );

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   *
   * (III) loop on tbc phases
   *
   ***************************************************************************/
  for ( int itheta = 0; itheta < g_tbc_phase_number; itheta++ ) {

    /***************************************************************************
     * (III.1) inversions for theta = - tbc
     *
     * NOTE: we keep tbc phase in time direction, no change of sign there
     ***************************************************************************/

    double theta_t =  g_tbc_phase_list[itheta][0];
    double theta_x = -g_tbc_phase_list[itheta][1];
    double theta_y = -g_tbc_phase_list[itheta][2];
    double theta_z = -g_tbc_phase_list[itheta][3];

    complex co_tbc_phase[4] = {
      { cos ( theta_t * M_PI / (double)T_global ),
        sin ( theta_t * M_PI / (double)T_global ) },
      { cos ( theta_x * M_PI / (double)LX_global ),
        sin ( theta_x * M_PI / (double)LX_global ) },
      { cos ( theta_y * M_PI / (double)LY_global ),
        sin ( theta_y * M_PI / (double)LY_global ) },
      { cos ( theta_z * M_PI / (double)LZ_global ),
        sin ( theta_z * M_PI / (double)LZ_global ) } };

    /***************************************************************************
     * multiply the phase to the gauge field
     ***************************************************************************/
    exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_tbc_phase );
    if(exitstatus != 0) {
      fprintf(stderr, "[cpff_tbc_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    /***************************************************************************
     * check plaquettes
     ***************************************************************************/
    exitstatus = plaquetteria ( gauge_field_with_phase );
    if(exitstatus != 0) {
      fprintf(stderr, "[cpff_tbc_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    /***************************************************************************
     * initialize clover, mzz and mzz_inv
     ***************************************************************************/
    exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }


    /***************************************************************************
     * loop on source timeslices
     ***************************************************************************/
    for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

      /***************************************************************************
       * local source timeslice and source process ids
       ***************************************************************************/

      int source_timeslice = -1;
      int source_proc_id   = -1;
      int gts              = ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global;

      exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[cpff_tbc_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(123);
      }

      /***************************************************************************
       * loop on stochastic oet samples
       ***************************************************************************/
      for ( int isample = 0; isample < g_nsample; isample++ ) {

        /***************************************************************************
         * invert
         ***************************************************************************/
        size_t const offset = source_timeslice * _GSI(LX*LY*LZ);
        size_t const sizeof_spinor_field_timeslice = _GSI(LX*LY*LZ) * sizeof(double);

        memset ( spinor_work[0], 0, sizeof_spinor_field );
        if ( source_proc_id == g_cart_id ) {
          /* if process has the sources timeslice, then copy the local timeslice */
          memcpy ( spinor_work[0] + offset, stochastic_source_list[isample]+offset, sizeof_spinor_field_timeslice );
        }

        memset ( spinor_work[1], 0, sizeof_spinor_field );

        exitstatus = _TMLQCD_INVERT_TBC ( spinor_work[1], spinor_work[0], op_id_dn );
        if(exitstatus != 0) {
          fprintf(stderr, "[cpff_tbc_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_dn], 1 );
        }

        memcpy( stochastic_propagator_zero_list[isource_location][isample], spinor_work[1], sizeof_spinor_field);

        if ( g_write_propagator ) {
          sprintf(filename, "%s.%.4d.t%d.px%dpy%dpz%d.%.5d.inverted", filename_prefix, Nconf, gts, 0,0,0, isample);
          if ( ( exitstatus = write_propagator( stochastic_propagator_zero_list[isource_location][isample], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[cpff_tbc_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }

      }  /* end of loop on stochastic samples */

      /***************************************************************************/
      /***************************************************************************/

      /***************************************************************************
       *
       * (III.2) contractions for 2-point functions
       *
       ***************************************************************************/

      /***************************************************************************
       * output filename
       ***************************************************************************/
#if ( defined HAVE_HDF5 )
      sprintf ( output_filename, "%s.%.4d.t%d.h5", outfile_prefix, Nconf, gts );
#endif
      if(io_proc == 2 && g_verbose > 1 ) { 
        fprintf(stdout, "# [cpff_tbc_invert_contract] writing data to file %s\n", output_filename);
      }

      /***************************************************************************
       * loop on stochastic samples
       ***************************************************************************/
      for ( int isample = 0; isample < g_nsample; isample++ ) {

        /***************************************************************************
         * loop on gamma matrix
         *
         * Note: source_gamma = -1 expected by contract_twopoint_snk_momentum
         *       if we do not have spin-diluted propagators
         ***************************************************************************/
        // for ( int isrc_gamma = 0; isrc_gamma < g_source_gamma_id_number; isrc_gamma++ ) {
          int const source_gamma = 5;

        for ( int isnk_gamma = 0; isnk_gamma < g_source_gamma_id_number; isnk_gamma++ ) {
          int const sink_gamma = g_source_gamma_id_list[isnk_gamma];

          double ** contr_p = init_2level_dtable ( g_seq_source_momentum_number, 2*T );
          if ( contr_p == NULL ) {
            fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(47);
          }

          /***************************************************************************
           * loop on momenta
           ***************************************************************************/
          for ( int imom = 0; imom < g_seq_source_momentum_number; imom++ ) {

            int const source_momentum[3] = {
              g_seq_source_momentum_list[imom][0],
              g_seq_source_momentum_list[imom][1],
              g_seq_source_momentum_list[imom][2] };

            int const sink_momentum[3] = { -source_momentum[0], -source_momentum[1], -source_momentum[2] };

            /***************************************************************************
             * contract
             ***************************************************************************/

            contract_twopoint_snk_momentum ( contr_p[imom], -1,  sink_gamma,
                &(stochastic_propagator_zero_list[isource_location][isample]), &(stochastic_propagator_mom_list[isource_location][isample][imom]), 1, 1, sink_momentum, 1);

          }  /* end of loop on momenta */

          /***************************************************************************
           * data key
           ***************************************************************************/
          sprintf ( data_tag, "/d+-g-d-g/theta%d/t%d/s%d/gf%d/gi%d", itheta, gts, isample, sink_gamma, source_gamma );


#if ( defined HAVE_HDF5 )          
          exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_seq_source_momentum_list, g_seq_source_momentum_number, io_proc );
#endif
          if(exitstatus != 0) {
            fprintf(stderr, "[cpff_tbc_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }
        
          fini_2level_dtable ( &contr_p );

        }  /* end of loop on gamma at sink */
        // }  /* end of loop on gammas at source */


      }  /* end of loop on samples */

    }  /* end of loop on source timeslices */

    /***************************************************************************
     * free clover term matrices
     ***************************************************************************/
    fini_clover ( &mzz, &mzzinv );

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     *
     * (III.4) inversions for theta = tbc phase
     *
     ***************************************************************************/
    theta_x = g_tbc_phase_list[itheta][1];
    theta_y = g_tbc_phase_list[itheta][2];
    theta_z = g_tbc_phase_list[itheta][3];
    theta_t = g_tbc_phase_list[itheta][0];

    co_tbc_phase[0].re = cos ( theta_t * M_PI / (double)T_global );
    co_tbc_phase[0].im = sin ( theta_t * M_PI / (double)T_global );

    co_tbc_phase[1].re = cos ( theta_x * M_PI / (double)LX_global );
    co_tbc_phase[1].im = sin ( theta_x * M_PI / (double)LX_global );

    co_tbc_phase[2].re = cos ( theta_y * M_PI / (double)LY_global );
    co_tbc_phase[2].im = sin ( theta_y * M_PI / (double)LY_global ),

    co_tbc_phase[3].re = cos ( theta_z * M_PI / (double)LZ_global );
    co_tbc_phase[3].im = sin ( theta_z * M_PI / (double)LZ_global );

    /***************************************************************************
     * multiply the phase to the gauge field
     ***************************************************************************/
    exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_tbc_phase );
    if(exitstatus != 0) {
      fprintf(stderr, "[cpff_tbc_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    /***************************************************************************
     * check plaquettes
     ***************************************************************************/
    exitstatus = plaquetteria ( gauge_field_with_phase );
    if(exitstatus != 0) {
      fprintf(stderr, "[cpff_tbc_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    /***************************************************************************
     * initialize clover, mzz and mzz_inv
     ***************************************************************************/
    exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }

    /***************************************************************************
     * loop on source timeslices
     ***************************************************************************/
    for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

      /***************************************************************************
       * local source timeslice and source process ids
       ***************************************************************************/

      int source_timeslice = -1;
      int source_proc_id   = -1;
      int gts              = ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global;

      exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[cpff_tbc_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(123);
      }


      /***************************************************************************
       * output filename
       ***************************************************************************/
#if ( defined HAVE_HDF5 )
      sprintf ( output_filename, "%s.%.4d.t%d.h5", outfile_prefix, Nconf, gts );
#endif
      if(io_proc == 2 && g_verbose > 1 ) { 
        fprintf(stdout, "# [cpff_tbc_invert_contract] writing data to file %s\n", output_filename);
      }

      /***************************************************************************
       * loop on stochastic samples
       ***************************************************************************/
      for ( int isample = 0; isample < g_nsample; isample++ ) {

        /*****************************************************************
         * loop on sequential source momenta p_f
         *****************************************************************/
        for( int iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {

          int seq_source_momentum[3] = { g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2] };

          /*****************************************************************
           * loop on sequential source gamma ids
           *****************************************************************/
          for ( int iseq_gamma = 0; iseq_gamma < g_sequential_source_gamma_id_number; iseq_gamma++ ) {

            int const seq_source_gamma = g_sequential_source_gamma_id_list[iseq_gamma];

            /*****************************************************************
             * loop on sequential source timeslices
             *****************************************************************/
            for ( int iseq_timeslice = 0; iseq_timeslice < g_sequential_source_timeslice_number; iseq_timeslice++ ) {

              /*****************************************************************
               * global sequential source timeslice
               * NOTE: counted from current source timeslice
               *****************************************************************/
              int const gtseq = ( gts + g_sequential_source_timeslice_list[iseq_timeslice] + T_global ) % T_global;

              /*****************************************************************
               *
               * (III.5) invert for sequential timeslice propagator
               *
               *****************************************************************/

              /*****************************************************************
               * prepare sequential timeslice source 
               *****************************************************************/
              exitstatus = init_sequential_source ( spinor_work[0], stochastic_propagator_mom_list[isource_location][isample][iseq_mom], gtseq, seq_source_momentum, seq_source_gamma );
              if( exitstatus != 0 ) {
                fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(64);
              }

              if ( g_write_sequential_source ) {
                sprintf(filename, "%s.%.4d.t%d.qx%dqy%dqz%d.g%d.dt%d.%.5d", filename_prefix, Nconf, gts,
                    seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], seq_source_gamma,
                    g_sequential_source_timeslice_list[iseq_timeslice], isample);
                if ( ( exitstatus = write_propagator( spinor_work[0], filename, 0, g_propagator_precision) ) != 0 ) {
                  fprintf(stderr, "[cpff_tbc_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(2);
                }
              }  /* end of if g_write_sequential_source */

              memset ( spinor_work[1], 0, sizeof_spinor_field );

              exitstatus = _TMLQCD_INVERT_TBC ( spinor_work[1], spinor_work[0], op_id_up );
              if(exitstatus != 0) {
                fprintf(stderr, "[cpff_tbc_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(44);
              }

              if ( check_propagator_residual ) {
                check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_up], 1 );
              }

              memcpy( sequential_propagator_list[0], spinor_work[1], sizeof_spinor_field );

              if ( g_write_sequential_propagator ) {
                sprintf ( filename, "%s.%.4d.t%d.qx%dqy%dqz%d.g%d.dt%d.%.5d.inverted", filename_prefix, Nconf, gts,
                    seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], seq_source_gamma, 
                    g_sequential_source_timeslice_list[iseq_timeslice], isample);
                if ( ( exitstatus = write_propagator( sequential_propagator_list[0], filename, 0, g_propagator_precision) ) != 0 ) {
                  fprintf(stderr, "[cpff_tbc_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(2);
                }
              }  /* end of if g_write_sequential_propagator */

              /*****************************************************************/
              /*****************************************************************/

              /*****************************************************************
               *
               * (III..6) contractions for local current insertion
               *
               *****************************************************************/

              /*****************************************************************
               * loop on local gamma matrices
               *****************************************************************/
              for ( int icur_gamma = 0; icur_gamma < gamma_current_number; icur_gamma++ ) {

                int const gamma_current = gamma_current_list[icur_gamma];

                int const gamma_source = 5;

                double ** contr_p = init_2level_dtable ( 1, 2*T );
                if ( contr_p == NULL ) {
                  fprintf(stderr, "[cpff_tbc_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
                  EXIT(47);
                }

                int const current_momentum[3] = {
                  -2 * seq_source_momentum[0],
                  -2 * seq_source_momentum[1],
                  -2 * seq_source_momentum[2] };

                  contract_twopoint_snk_momentum ( contr_p[0], -1,  gamma_current, 
                      &(stochastic_propagator_zero_list[isource_location][isample]), 
                      &(sequential_propagator_list[0]), 1, 1, current_momentum, 1);

                sprintf ( data_tag, "/d+-g-sud/theta%d/t%d/s%d/dt%d/gf%d/gc%d/gi%d/pfx%dpfy%dpfz%d/", 
                    itheta, gts, isample, g_sequential_source_timeslice_list[iseq_timeslice],
                    seq_source_gamma, gamma_current, gamma_source,
                    seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2] );

#if ( defined HAVE_HDF5 )
                exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, &current_momentum, 1, io_proc );
#endif
                if(exitstatus != 0) {
                  fprintf(stderr, "[cpff_tbc_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(3);
                }
              
                fini_2level_dtable ( &contr_p );

              }  /* end of loop on current gamma id */

            }  /* loop on sequential source timeslices */

          }  /* end of loop on sequential source gamma ids */

        }  /* end of loop on sequential source momenta */

      }  /* end of loop on stochastic samples */

    }  /* end of loop on source timeslices */

#if 0
#endif  /* of if 0 */

    /***************************************************************************
     * free clover term matrices
     ***************************************************************************/
    fini_clover ( &mzz, &mzzinv );

  }  /* end of loop on tbc phases */

    
  /***************************************************************************
   *
   * (IV) free the allocated memory, finalize
   *
   ***************************************************************************/

  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_4level_dtable ( &stochastic_propagator_mom_list );
  fini_3level_dtable ( &stochastic_propagator_zero_list );
  fini_2level_dtable ( &stochastic_source_list );
  fini_2level_dtable ( &sequential_propagator_list );

  fini_2level_dtable ( &spinor_work );

  /***************************************************************************
   * fini rng state
   ***************************************************************************/
  fini_rng_state ( &rng_state);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  /***************************************************************************
   * clover matrix terms have been deallocated above
   ***************************************************************************/

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
    fprintf(stdout, "# [cpff_tbc_invert_contract] %s# [cpff_tbc_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [cpff_tbc_invert_contract] %s# [cpff_tbc_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
