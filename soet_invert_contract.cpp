/****************************************************
 * soet_invert_contract
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
  fprintf(stdout, "Code to calculate stochastic oet inversions + contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "soet";

  const char fbwd_str[2][4] =  { "fwd", "bwd" };

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
  int spin_dilution = 1;
  int color_dilution = 1;

  char data_tag[400];
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
#endif

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
    /* case 's':
      spin_dilution = atoi ( optarg );
      break;
    case 'c':
      color_dilution = atoi ( optarg );
      break;
      */
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
  /* fprintf(stdout, "# [soet_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [soet_invert_contract] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
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

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [soet_invert_contract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[soet_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [soet_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [soet_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[soet_invert_contract] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[soet_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[soet_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[soet_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[soet_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[soet_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[soet_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [soet_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

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
  double ** spinor_work  = init_2level_dtable ( 4, nelem );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[soet_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * allocate memory for spinor fields
   * WITHOUT halo
   ***************************************************************************/
  int const spin_color_dilution = spin_dilution * color_dilution;
  nelem = _GSI( VOLUME );
  double ** stochastic_propagator_mom_list = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_propagator_mom_list == NULL ) {
    fprintf(stderr, "[soet_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double ** stochastic_propagator_zero_list = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_propagator_zero_list == NULL ) {
    fprintf(stderr, "[soet_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double ** stochastic_propagator_zero_list2 = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_propagator_zero_list == NULL ) {
    fprintf(stderr, "[soet_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double ** stochastic_source_list = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_source_list == NULL ) {
    fprintf(stderr, "[soet_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[soet_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }
  /* for ( int i = 0; i < rlxd_size(); i++ ) {
    fprintf ( stdout, "rng %2d %10d\n", g_cart_id, rng_state[i] );
  } */


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
      fprintf(stderr, "[soet_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

#if ( defined HAVE_LHPC_AFF ) && !(defined HAVE_HDF5 )
    /***************************************************************************
     * output filename
     ***************************************************************************/
    sprintf ( output_filename, "%s.%.4d.t%d.aff", outfile_prefix, Nconf, gts );
    /***************************************************************************
     * writer for aff output file
     ***************************************************************************/
    if(io_proc == 2) {
      affw = aff_writer ( output_filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[soet_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#elif ( defined HAVE_HDF5 )
    sprintf ( output_filename, "%s.%.4d.t%d.h5", outfile_prefix, Nconf, gts );
#endif
    if(io_proc == 2 && g_verbose > 1 ) { 
      fprintf(stdout, "# [soet_invert_contract] writing data to file %s\n", output_filename);
    }

    /***************************************************************************
     * loop on stochastic oet samples
     ***************************************************************************/
    for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

      /***************************************************************************
       * synchronize rng states to state at zero
       ***************************************************************************/
      exitstatus = sync_rng_state ( rng_state, 0, 0 );
      if(exitstatus != 0) {
        fprintf(stderr, "[soet_invert_contract] Error from sync_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }

      /***************************************************************************
       * read stochastic oet source from file
       ***************************************************************************/
      if ( g_read_source ) {
        for ( int i = 0; i < spin_color_dilution; i++ ) {
          sprintf(filename, "%s.%.4d.t%d.%d.%.5d", filename_prefix, Nconf, gts, i, isample);
          if ( ( exitstatus = read_lime_spinor( stochastic_source_list[i], filename, 0) ) != 0 ) {
            fprintf(stderr, "[soet_invert_contract] Error from read_lime_spinor, status was %d\n", exitstatus);
            EXIT(2);
          }
        }
        /* recover the ran field */
        exitstatus = init_timeslice_source_oet(stochastic_source_list, gts, NULL, spin_dilution, color_dilution,  -1 );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[soet_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(64);
        }

      /***************************************************************************
       * generate stochastic oet source
       ***************************************************************************/
      } else {
        /* call to initialize the ran field 
         *   penultimate argument is momentum vector for the source, NULL here
         *   final argument in arg list is 1
         */
        if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gts, NULL, spin_dilution, color_dilution, 1 ) ) != 0 ) {
          fprintf(stderr, "[soet_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(64);
        }
        if ( g_write_source ) {
          for ( int i = 0; i < spin_color_dilution; i++ ) {
            sprintf(filename, "%s.%.4d.t%d.%d.%.5d", filename_prefix, Nconf, gts, i, isample);
            if ( ( exitstatus = write_propagator( stochastic_source_list[i], filename, 0, g_propagator_precision) ) != 0 ) {
              fprintf(stderr, "[soet_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(2);
            }
          }
        }
      }  /* end of if read stochastic source - else */

      /***************************************************************************
       * retrieve current rng state and 0 writes his state
       ***************************************************************************/
      exitstatus = get_rng_state ( rng_state );
      if(exitstatus != 0) {
        fprintf(stderr, "[soet_invert_contract] Error from get_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }

      exitstatus = save_rng_state ( 0, NULL );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[soet_invert_contract] Error from save_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
        EXIT(38);
      }


      /***************************************************************************
       * invert for stochastic timeslice propagator at zero momentum dn flavor
       ***************************************************************************/
      for( int i = 0; i < spin_color_dilution; i++) {
        memcpy ( spinor_work[0], stochastic_source_list[i], sizeof_spinor_field );
        memcpy ( spinor_work[2], spinor_work[0], sizeof_spinor_field );

        /***************************************************************************
         * dn type
         ***************************************************************************/
        memset ( spinor_work[1], 0, sizeof_spinor_field );
        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], op_id_dn );
        if(exitstatus < 0) {
          fprintf(stderr, "[soet_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[2]), gauge_field_with_phase, mzz[op_id_dn], mzzinv[op_id_dn], 1 );
        }

        memcpy( stochastic_propagator_zero_list[i], spinor_work[1], sizeof_spinor_field);

        /***************************************************************************
         * up type
         ***************************************************************************/
        memset ( spinor_work[1], 0, sizeof_spinor_field );
        memcpy ( spinor_work[0], spinor_work[2], sizeof_spinor_field );
        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], op_id_up );
        if(exitstatus < 0) {
          fprintf(stderr, "[soet_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[2]), gauge_field_with_phase, mzz[op_id_up], mzzinv[op_id_up], 1 );
        }

        memcpy( stochastic_propagator_zero_list2[i], spinor_work[1], sizeof_spinor_field);

      }
      if ( g_write_propagator ) {
        for ( int i = 0; i < spin_color_dilution; i++ ) {
          sprintf(filename, "%s.%.4d.t%d.%d.%.5d.inverted", filename_prefix, Nconf, gts, i, isample);
          if ( ( exitstatus = write_propagator( stochastic_propagator_zero_list[i], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[soet_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
      }

      /***************************************************************************
       * loop on gamma matrices at source
       ***************************************************************************/

      for ( int isrc_gamma = 0; isrc_gamma < g_source_gamma_id_number; isrc_gamma++ ) {

        /***************************************************************************
         * invert for stochastic timeslice propagator at source momenta
         ***************************************************************************/
          for ( int isrc_mom = 0; isrc_mom < g_source_momentum_number; isrc_mom++ ) {

          /***************************************************************************
           * NOTE: we take the negative of the momentum in the list
           * since we use it in the daggered timeslice propagator
           ***************************************************************************/
          int source_momentum[3] = {
            g_source_momentum_list[isrc_mom][0],
            g_source_momentum_list[isrc_mom][1],
            g_source_momentum_list[isrc_mom][2] };

          /***************************************************************************
           * prepare stochastic timeslice source at source momentum
           ***************************************************************************/
          exitstatus = init_timeslice_source_oet ( stochastic_source_list, gts, source_momentum, spin_dilution, color_dilution, 0 );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[soet_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(64);
          } 

          /***************************************************************************
           * invert
           ***************************************************************************/
          for( int i = 0; i < spin_color_dilution; i++) {
 
            memcpy ( spinor_work[0], stochastic_source_list[i], sizeof_spinor_field );
            g5_phi ( spinor_work[0], VOLUME );
            spinor_field_eq_gamma_ti_spinor_field ( spinor_work[0], g_source_gamma_id_list[isrc_gamma], spinor_work[0], VOLUME );
            memcpy ( spinor_work[2], spinor_work[0], sizeof_spinor_field );
            
            memset ( spinor_work[1], 0, sizeof_spinor_field );

            if ( g_write_source ) {
              sprintf(filename, "%s.%.4d.t%d.px%dpy%dpz%d.%d.%.5d", filename_prefix, Nconf, gts, 
              source_momentum[0], source_momentum[1], source_momentum[2], i, isample);
              if ( ( exitstatus = write_propagator( spinor_work[0], filename, 0, g_propagator_precision) ) != 0 ) {
                fprintf(stderr, "[soet_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(2);
              }
            }

            exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], op_id_dn );
            if(exitstatus < 0) {
              fprintf(stderr, "[soet_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(44);
            }

            if ( check_propagator_residual ) {
              check_residual_clover ( &(spinor_work[1]), &(spinor_work[2]), gauge_field_with_phase, mzz[op_id_dn], mzzinv[op_id_dn], 1 );
            }

            memcpy( stochastic_propagator_mom_list[i], spinor_work[1], sizeof_spinor_field);

          }  /* end of loop on spinor components */

          if ( g_write_propagator ) {
            for ( int i = 0; i < spin_color_dilution; i++ ) {
              sprintf(filename, "%s.%.4d.t%d.px%dpy%dpz%d.g%d.%d.%.5d.inverted", filename_prefix, Nconf, gts,
                  source_momentum[0], source_momentum[1], source_momentum[2], g_source_gamma_id_list[isrc_gamma], i, isample);
              if ( ( exitstatus = write_propagator( stochastic_propagator_mom_list[i], filename, 0, g_propagator_precision) ) != 0 ) {
                fprintf(stderr, "[soet_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(2);
              }
            }
          }

          /*****************************************************************
           * contractions
           *****************************************************************/
          for ( int isnk_gamma = 0; isnk_gamma < g_source_gamma_id_number; isnk_gamma++ ) {

            /*****************************************************************
             * charged
             *****************************************************************/
        
            /* allocate contraction fields in position and momentum space */
            double * contr_x = init_1level_dtable ( 2 * VOLUME );
            if ( contr_x == NULL ) {
              fprintf(stderr, "[soet_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
              EXIT(3);
            }

            double ** contr_p = init_2level_dtable ( g_sink_momentum_number , 2 * T );
            if ( contr_p == NULL ) {
              fprintf(stderr, "[soet_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
              EXIT(3);
            }

            /* contractions in x-space */
            contract_twopoint_xdep ( contr_x, g_source_gamma_id_list[isrc_gamma], g_source_gamma_id_list[isnk_gamma], 
                stochastic_propagator_mom_list,
                stochastic_propagator_zero_list, spin_dilution, color_dilution, 1, 1., 64 );

            /* momentum projection at sink */
            exitstatus = momentum_projection ( contr_x, contr_p[0], T, g_sink_momentum_number, g_sink_momentum_list );
            if(exitstatus != 0) {
              fprintf(stderr, "[soet_invert_contract] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(3);
            }

            sprintf ( data_tag, "/d+-g-d-g/t%d/s%d/gf%d/gi%d/pix%dpiy%dpiz%d", gts, isample,
                g_source_gamma_id_list[isnk_gamma], g_source_gamma_id_list[isrc_gamma],
                g_source_momentum_list[isrc_mom][0], g_source_momentum_list[isrc_mom][1], g_source_momentum_list[isrc_mom][2] );

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
            exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
#elif ( defined HAVE_HDF5 )          
            exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
#endif
            if(exitstatus != 0) {
              fprintf(stderr, "[soet_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(3);
            }
 
            /* deallocate the contraction fields */       
            fini_1level_dtable ( &contr_x );
            fini_2level_dtable ( &contr_p );

            /*****************************************************************
             * neutral
             *****************************************************************/

            /* allocate contraction fields in position and momentum space */
            contr_x = init_1level_dtable ( 2 * VOLUME );
            if ( contr_x == NULL ) {
              fprintf(stderr, "[soet_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
              EXIT(3);
            }

            contr_p = init_2level_dtable ( g_sink_momentum_number , 2 * T );
            if ( contr_p == NULL ) {
              fprintf(stderr, "[soet_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
              EXIT(3);
            }

            /* contractions in x-space */
            contract_twopoint_xdep ( contr_x, g_source_gamma_id_list[isrc_gamma], g_source_gamma_id_list[isnk_gamma], 
                stochastic_propagator_mom_list,
                stochastic_propagator_zero_list2, spin_dilution, color_dilution, 1, 1., 64 );

            /* momentum projection at sink */
            exitstatus = momentum_projection ( contr_x, contr_p[0], T, g_sink_momentum_number, g_sink_momentum_list );
            if(exitstatus != 0) {
              fprintf(stderr, "[soet_invert_contract] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(3);
            }

            sprintf ( data_tag, "/u+-g-d-g/t%d/s%d/gf%d/gi%d/pix%dpiy%dpiz%d", gts, isample,
                g_source_gamma_id_list[isnk_gamma], g_source_gamma_id_list[isrc_gamma],
                g_source_momentum_list[isrc_mom][0], g_source_momentum_list[isrc_mom][1], g_source_momentum_list[isrc_mom][2] );

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
            exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
#elif ( defined HAVE_HDF5 )          
            exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
#endif
            if(exitstatus != 0) {
              fprintf(stderr, "[soet_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(3);
            }
 
            /* deallocate the contraction fields */       
            fini_1level_dtable ( &contr_x );
            fini_2level_dtable ( &contr_p );



          }  /* end of loop on gamma at sink */

        }  /* end of loop on source momenta */
      }  /* end of loop on source gamma */

      /*****************************************************************/
      /*****************************************************************/

      exitstatus = init_timeslice_source_oet ( NULL, -1, NULL, 0, 0, -2 );

    }  /* end of loop on oet samples */

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[soet_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  }  /* end of loop on source timeslices */

  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_2level_dtable ( &stochastic_propagator_mom_list );
  fini_2level_dtable ( &stochastic_propagator_zero_list );
  fini_2level_dtable ( &stochastic_propagator_zero_list2 );
  fini_2level_dtable ( &stochastic_source_list );
  fini_2level_dtable ( &spinor_work );

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
    fprintf(stdout, "# [soet_invert_contract] %s# [soet_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [soet_invert_contract] %s# [soet_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
