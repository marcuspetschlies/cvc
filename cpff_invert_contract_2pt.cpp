/****************************************************
 * cpff_invert_contract_2pt
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

#define _DERIV  1
#define _DDERIV 0

#define FLAVOR_MAX_NUMBER 12

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to calculate charged pion FF inversions + contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "cpff";

  const char fbwd_str[2][4] =  { "fwd", "bwd" };
  
  const char flavor_tag[2] =  { 'u', 'd' };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  size_t sizeof_spinor_field;
  char filename[400];
  double **mzz[FLAVOR_MAX_NUMBER], **mzzinv[FLAVOR_MAX_NUMBER];
  double *gauge_field_with_phase = NULL;
  char output_filename[400];
  int * rng_state = NULL;
  int spin_dilution = 4;
  int color_dilution = 1;
  int flavor_number = 0;
  double flavor_mu[FLAVOR_MAX_NUMBER];


  char data_tag[400];
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "rh?f:s:c:q:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'r':
      check_propagator_residual = 1;
      break;
    case 's':
      spin_dilution = atoi ( optarg );
      break;
    case 'c':
      color_dilution = atoi ( optarg );
      break;
    case 'q':
      flavor_mu[flavor_number] = atof ( optarg );
      flavor_number++;
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
  /* fprintf(stdout, "# [cpff_invert_contract_2pt] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [cpff_invert_contract_2pt] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [cpff_invert_contract_2pt] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [cpff_invert_contract_2pt] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [cpff_invert_contract_2pt] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[cpff_invert_contract_2pt] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[cpff_invert_contract_2pt] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[cpff_invert_contract_2pt] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[cpff_invert_contract_2pt] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[cpff_invert_contract_2pt] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[cpff_invert_contract_2pt] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [cpff_invert_contract_2pt] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************
   * set operator ids depending on fermion type
   ***********************************************************/
#if 0
  if ( g_fermion_type == _TM_FERMION ) {
    op_id_up = 0;
    op_id_dn = 1;
  } else if ( g_fermion_type == _WILSON_FERMION ) {
    op_id_up = 0;
    op_id_dn = 0;
  }
#endif

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  size_t nelem = _GSI( VOLUME+RAND );
  double ** spinor_work  = init_2level_dtable ( 2, nelem );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * allocate memory for spinor fields
   * WITHOUT halo
   ***************************************************************************/
  int const spin_color_dilution = spin_dilution * color_dilution;
  nelem = _GSI( VOLUME );

  double *** stochastic_propagator_mom_list = init_3level_dtable ( 2, spin_color_dilution, nelem );
  if ( stochastic_propagator_mom_list == NULL ) {
    fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double *** stochastic_propagator_zero_list = init_3level_dtable ( flavor_number, spin_color_dilution, nelem );
  if ( stochastic_propagator_zero_list == NULL ) {
    fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double ** stochastic_source_list = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_source_list == NULL ) {
    fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }
 
  if ( g_verbose > 4 ) {
    for ( int i = 0; i < rlxd_size(); i++ ) {
      fprintf ( stdout, "rng %2d %10d\n", g_cart_id, rng_state[i] );
    }
  }

#if ( defined HAVE_LHPC_AFF ) && !(defined HAVE_HDF5 )
  /***************************************************************************
   * output filename
   ***************************************************************************/
  sprintf ( output_filename, "%s.%.4d.aff", outfile_prefix, Nconf );
  /***************************************************************************
   * writer for aff output file
   ***************************************************************************/
  if(io_proc == 2) {
    affw = aff_writer ( output_filename);
    const char * aff_status_str = aff_writer_errstr ( affw );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[cpff_invert_contract_2pt] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }  /* end of if io_proc == 2 */
#elif ( defined HAVE_HDF5 )
  sprintf ( output_filename, "%s.%.4d.h5", outfile_prefix, Nconf );
#endif
  if(io_proc == 2 && g_verbose > 1 ) { 
    fprintf(stdout, "# [cpff_invert_contract_2pt] writing data to file %s\n", output_filename);
  }

  /***************************************************************************
   * loop on source timeslices
   ***************************************************************************/
  for( int isource_location = 0; isource_location < T_global; isource_location++ )
  {

    /***************************************************************************
     * local source timeslice and source process ids
     ***************************************************************************/

    int source_timeslice = -1;
    int source_proc_id   = -1;
    int gts              = isource_location;

    exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[cpff_invert_contract_2pt] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
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
        fprintf(stderr, "[cpff_invert_contract_2pt] Error from sync_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }

      /***************************************************************************
       * read stochastic oet source from file
       ***************************************************************************/
      if ( g_read_source ) {
        for ( int i = 0; i < spin_color_dilution; i++ ) {
          sprintf(filename, "%s.%.4d.t%d.%d.%.5d", filename_prefix, Nconf, gts, i, isample);
          if ( ( exitstatus = read_lime_spinor( stochastic_source_list[i], filename, 0) ) != 0 ) {
            fprintf(stderr, "[cpff_invert_contract_2pt] Error from read_lime_spinor, status was %d\n", exitstatus);
            EXIT(2);
          }
        }
        /* recover the ran field */
        exitstatus = init_timeslice_source_oet(stochastic_source_list, gts, NULL, spin_dilution, color_dilution,  -1 );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
          fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(64);
        }
        if ( g_write_source ) {
          for ( int i = 0; i < spin_color_dilution; i++ ) {
            sprintf(filename, "%s.%.4d.t%d.%d.%.5d", filename_prefix, Nconf, gts, i, isample);
            if ( ( exitstatus = write_propagator( stochastic_source_list[i], filename, 0, g_propagator_precision) ) != 0 ) {
              fprintf(stderr, "[cpff_invert_contract_2pt] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
        fprintf(stderr, "[cpff_invert_contract_2pt] Error from get_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }

      exitstatus = save_rng_state ( 0, NULL );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[cpff_invert_contract_2pt] Error from save_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
        EXIT(38);
      }

      /***************************************************************************
       * invert for stochastic timeslice propagator at zero momentum, all flavors
       ***************************************************************************/
      for ( int iflavor = 0; iflavor < flavor_number; iflavor++ ) {

        if ( iflavor % 2 == 0 ) {

          /***************************************************************************
           * initialize clover, mzz and mzz_inv
           ***************************************************************************/
          exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, flavor_mu[iflavor], g_csw );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
        }

        for ( int i = 0; i < spin_color_dilution; i++ ) {

          memcpy ( spinor_work[0], stochastic_source_list[i], sizeof_spinor_field );

          memset ( spinor_work[1], 0, sizeof_spinor_field );

          exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
          if(exitstatus < 0) {
            fprintf(stderr, "[cpff_invert_contract_2pt] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(44);
          }

          if ( check_propagator_residual ) {
            check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor%2], mzzinv[iflavor%2], 1 );
          }

          memcpy( stochastic_propagator_zero_list[iflavor][i], spinor_work[1], sizeof_spinor_field);
        }  /* end of spin-color component */

        if ( iflavor % 2 == 1 ) {
          /* free clover matrix terms */
          fini_clover ( &mzz, &mzzinv );
        }
      }

      /***************************************************************************
       * loop on all flavor type
       ***************************************************************************/
      for ( int iflavor1 = 0; iflavor1 < 2; iflavor1++ ) 
      {
        char flavor_str1[20];
        if ( iflavor1 < 2 ) {
          sprintf ( flavor_str1, "%c", flavor_tag[iflavor1] );
        } else {
          if ( flavor_mu[iflavor1] < 0 ) {
            sprintf( flavor_str1, "sm%6.4f", fabs( flavor_mu[iflavor1] ) );
          } else {
            sprintf( flavor_str1, "sp%6.4f", flavor_mu[iflavor1] );
          }
        }

        for ( int iflavor2 = 0; iflavor2 < 2; iflavor2++ )
        {
          char flavor_str2[20];
          if ( iflavor2 < 2 ) {
            sprintf ( flavor_str2, "%c", flavor_tag[iflavor2] );
          } else {
            if ( flavor_mu[iflavor2] < 0 ) {
              sprintf( flavor_str2, "sm%6.4f", fabs( flavor_mu[iflavor2] ) );
           } else {
              sprintf( flavor_str2, "sp%6.4f", flavor_mu[iflavor2] );
            }
          }
          
          for ( int isrc_gamma = 0; isrc_gamma < g_source_gamma_id_number; isrc_gamma++ ) {
          for ( int isnk_gamma = 0; isnk_gamma < g_sink_gamma_id_number; isnk_gamma++ ) {
          
            /* allocate contraction fields in position and momentum space */
            double * contr_x = init_1level_dtable ( 2 * VOLUME );
            if ( contr_x == NULL ) {
              fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
              EXIT(3);
            }
  
            double ** contr_p = init_2level_dtable ( g_sink_momentum_number , 2 * T );
            if ( contr_p == NULL ) {
              fprintf(stderr, "[cpff_invert_contract_2pt] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
              EXIT(3);
            }
  
            /* contractions in x-space */
            contract_twopoint_xdep ( contr_x, g_source_gamma_id_list[isrc_gamma], g_sink_gamma_id_list[isnk_gamma], 
                stochastic_propagator_zero_list[iflavor1], 
                stochastic_propagator_mom_list[iflavor2],
                spin_dilution, color_dilution, 1, 1., 64 );
  
            /* momentum projection at sink */
            int sink_momentum[3] = {0,0,0};
            int source_momentum[3] = {0,0,0};
            exitstatus = momentum_projection ( contr_x, contr_p[0], T, 1, &sink_momentum );
            if(exitstatus != 0) {
              fprintf(stderr, "[cpff_invert_contract_2pt] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(3);
            }
  
            sprintf ( data_tag, "/%s+-g-%s-g/t%d/s%d/gf%d/pfx%dpfy%dpfz%d/gi%d/pix%dpiy%dpiz%d", flavor_str1, flavor_str2,
                gts, isample,
                g_sink_gamma_id_list[isnk_gamma], 
                sink_momentum[0], sink_momentum[1], sink_momentum[2],
                g_source_gamma_id_list[isrc_gamma],
                source_momentum[0], source_momentum[1], source_momentum[2] );
  
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
            exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, &sink_momentum, 1, io_proc );
#elif ( defined HAVE_HDF5 )          
            exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, &sink_momentum, 1, io_proc );
#endif
            if(exitstatus != 0) {
              fprintf(stderr, "[cpff_invert_contract_2pt] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(3);
            }
   
            /* deallocate the contraction fields */       
            fini_1level_dtable ( &contr_x );
            fini_2level_dtable ( &contr_p );
  
          }  /* end of loop on gamma at sink */
          }  /* end of loop on gammas at source */
   
        }  /* end of loop on flavor2 */
  
      }  /* end of loop on flavor1 */

      exitstatus = init_timeslice_source_oet ( NULL, -1, NULL, 0, 0, -2 );

    }  /* end of loop on oet samples */

  }  /* end of loop on source timeslices */

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  if(io_proc == 2) {
    const char * aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[cpff_invert_contract_2pt] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */


  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_3level_dtable ( &stochastic_propagator_zero_list );
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
    fprintf(stdout, "# [cpff_invert_contract_2pt] %s# [cpff_invert_contract_2pt] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [cpff_invert_contract_2pt] %s# [cpff_invert_contract_2pt] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
