/****************************************************
 * kaon2pt_invert_contract
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
#include "table_init_i.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "Q_phi.h"
#include "clover.h"
#include "ranlxd.h"
#include "smearing_techniques.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to calculate charged pion FF inversions + contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "kaon";

  const char flavor_tag[4][2] =  { "l", "l", "s", "s" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double **smzz[2] = { NULL, NULL }, **smzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL, *gauge_field_smeared = NULL;
  char output_filename[400];
  int * rng_state = NULL;
  int spin_dilution = 4;
  int color_dilution = 1;
  double g_mus =0.;
  int rng_status_read = 0;

  char data_tag[400];
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "prh?f:s:c:m:")) != -1) {
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
    case 'm':
      g_mus = atof ( optarg );
      break; 
    case 'p':
      rng_status_read = 1;
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
  /* fprintf(stdout, "# [kaon2pt_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [kaon2pt_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [kaon2pt_invert_contract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  size_t const sizeof_spinor_field = _GSI(VOLUME) * sizeof(double);

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
    if(g_cart_id==0) fprintf(stdout, "# [kaon2pt_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [kaon2pt_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[kaon2pt_invert_contract] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[kaon2pt_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv for light quark
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv for strange quark
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &smzz, &smzzinv, gauge_field_with_phase, g_mus, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [kaon2pt_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

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

  double const muval[4] =  { g_mu, -g_mu, g_mus, -g_mus };

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  size_t nelem = _GSI( VOLUME+RAND );
  double ** spinor_work  = init_2level_dtable ( 2, nelem );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * allocate memory for spinor fields
   * WITHOUT halo
   ***************************************************************************/
  int const spin_color_dilution = spin_dilution * color_dilution;
  nelem = _GSI( VOLUME );

  double *** stochastic_propagator_mom_smeared_list = init_3level_dtable ( g_source_momentum_number, spin_color_dilution, nelem );
  if ( stochastic_propagator_mom_smeared_list == NULL ) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  /***************************************************************************/

  double *** stochastic_propagator_zero_smeared_list = init_3level_dtable ( 1, spin_color_dilution, nelem );
  if ( stochastic_propagator_zero_smeared_list == NULL ) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  /***************************************************************************/

  double ** stochastic_source_list = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_source_list == NULL ) {
    fprintf(stderr, "[kaon2pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

#ifdef _SMEAR_QUDA
  /***************************************************************************
   * dummy solve, just to have original gauge field up on device,
   * for subsequent APE smearing
   ***************************************************************************/
  memset(spinor_work[1], 0, sizeof_spinor_field);
  memset(spinor_work[0], 0, sizeof_spinor_field);
  if ( g_cart_id == 0 ) spinor_work[0][0] = 1.;
  exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], 0 );
#  if ( defined GPU_DIRECT_SOLVER )
  if(exitstatus < 0)
#  else
  if(exitstatus != 0)
#  endif
  {
    fprintf(stderr, "[njjn_fht_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(12);
  }
#endif  /* of if _SMEAR_QUDA */

  /***********************************************
   * if we want to use Jacobi smearing, we need
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {

    alloc_gauge_field ( &gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy ( gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
    }  /* end of if N_aoe > 0 */
  }  /* end of if N_Jacobi > 0 */

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/

  if ( rng_status_read ) {

    exitstatus = read_rng_state ( &rng_state, 0, NULL );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[kaon2pt_invert_contract] Error from read_rng_state %s %d\n", __FILE__, __LINE__ );;
      EXIT( 50 );
    }

  } else {
    exitstatus = init_rng_state ( g_seed, &rng_state);
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[kaon2pt_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
      EXIT( 50 );
    }
  }
 
  if ( g_verbose > 0 ) {
    for ( int i = 0; i < rlxd_size(); i++ ) {
      fprintf ( stdout, "rng proc %2d start %10d\n", g_cart_id, rng_state[i] );
    }
  }
  
  /***************************************************************************
   * loop on source timeslices
   ***************************************************************************/
  for ( int isample = g_sourceid; isample <= g_sourceid2; isample += g_sourceid_step )
  {

    /***************************************************************************
     * random source timeslice
     ***************************************************************************/
    double dts;
    ranlxd ( &dts , 1 );
    int gts = (int)(dts * T_global);

    if ( g_cart_id == 0 ) fprintf ( stdout, "# [] gts = %d %s %d\n", gts, __FILE__, __LINE__ );

#ifdef HAVE_MPI
    if (  MPI_Bcast( &gts, 1, MPI_INT, 0, g_cart_grid ) != MPI_SUCCESS ) {
      fprintf ( stderr, "[kaon2pt_invert_contract] Error from MPI_Bcast %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
#endif

    /***************************************************************************
     * local source timeslice and source process ids
     ***************************************************************************/

    int source_timeslice = -1;
    int source_proc_id   = -1;

    exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[kaon2pt_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

#if ( defined HAVE_LHPC_AFF ) && !(defined HAVE_HDF5 )
    /***************************************************************************
     * output filename
     ***************************************************************************/
    sprintf ( output_filename, "%s.%.4d.t%d.s%d.aff", g_outfile_prefix, Nconf, gts, isample );
    /***************************************************************************
     * writer for aff output file
     ***************************************************************************/
    if(io_proc == 2) {
      affw = aff_writer ( output_filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#elif ( defined HAVE_HDF5 )
    sprintf ( output_filename, "%s.%.4d.t%d.s%d.h5", g_outfile_prefix, Nconf, gts, isample );
#endif
    if(io_proc == 2 && g_verbose > 1 ) { 
      fprintf(stdout, "# [kaon2pt_invert_contract] writing data to file %s\n", output_filename);
    }

    /***************************************************************************
     * re-initialize random number generator
     ***************************************************************************/
    /*
    if ( ! g_read_source ) {
      sprintf(filename, "rng_stat.%.4d.tsrc%.3d.stochastic-oet.out", Nconf, gts );
      exitstatus = init_rng_stat_file ( ( ( gts + 1 ) * 10000 + g_seed ), filename );
      if(exitstatus != 0) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }
    }
    */

    /***************************************************************************
     * synchronize rng states to state at zero
     ***************************************************************************/
    exitstatus = sync_rng_state ( rng_state, 0, 0 );
    if(exitstatus != 0) {
      fprintf(stderr, "[kaon2pt_invert_contract] Error from sync_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    /***************************************************************************
     * read stochastic oet source from file
     ***************************************************************************/
    if ( g_read_source ) {
      /* for ( int i = 0; i < spin_color_dilution; i++ ) { */
        sprintf ( filename, "%s.c%d.t%d.s%d", filename_prefix, Nconf, gts, isample);
        if ( ( exitstatus = read_lime_spinor( stochastic_source_list[0], filename, 0) ) != 0 ) {
          fprintf(stderr, "[kaon2pt_invert_contract] Error from read_lime_spinor, status was %d\n", exitstatus);
          EXIT(2);
        }
      /* } */
      /* recover the ran field */
      exitstatus = init_timeslice_source_oet(stochastic_source_list, gts, NULL, spin_dilution, color_dilution,  -1 );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
        fprintf(stderr, "[kaon2pt_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(64);
      }
      if ( g_write_source ) {
        for ( int i = 0; i < spin_color_dilution; i++ ) {
          sprintf(filename, "%s.c%d.t%d.i%d.s%d", filename_prefix, Nconf, gts, i, isample);
          if ( ( exitstatus = write_propagator( stochastic_source_list[i], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[kaon2pt_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
      fprintf(stderr, "[kaon2pt_invert_contract] Error from get_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    exitstatus = save_rng_state ( 0, NULL );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[kaon2pt_invert_contract] Error from save_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
      EXIT(38);
    }

          
    /***************************************************************************
     * prepare stochastic timeslice source at source momentum
     *
     * FOR FLAVOR UP
     ***************************************************************************/

    for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

      int source_momentum[3] = { 
        g_source_momentum_list[ipi][0],
        g_source_momentum_list[ipi][1],
        g_source_momentum_list[ipi][2] };

      exitstatus = init_timeslice_source_oet ( stochastic_source_list, gts, source_momentum, spin_dilution, color_dilution, 0 );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(64);
      }

      if ( g_write_source ) {
        for ( int i = 0; i < spin_color_dilution; i++ ) {
          sprintf(filename, "%s.c%d.t%d.i%d.s%d.px%dpy%dpz%d", filename_prefix, Nconf, gts, i, isample,
              source_momentum[0], source_momentum[1], source_momentum[2] );
          if ( ( exitstatus = write_propagator( stochastic_source_list[i], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[kaon2pt_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
      }

      if ( N_Jacobi > 0 ) {
        /***************************************************************************
         * SOURCE SMEARING
         ***************************************************************************/
        for( int i = 0; i < spin_color_dilution; i++) {
          exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_source_list[i], N_Jacobi, kappa_Jacobi);
          if(exitstatus != 0) {
            fprintf(stderr, "[kaon2pt_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            return(11);
          }
        }
      }
    
      /***************************************************************************
       * invert for stochastic timeslice propagator at zero momentum
       *   dn flavor
       *   this one will run from source to sink as part of the sequential
       *   propagator
       ***************************************************************************/
      for( int i = 0; i < spin_color_dilution; i++) {
   
        memcpy ( spinor_work[0], stochastic_source_list[i], sizeof_spinor_field );
  
        memset ( spinor_work[1], 0, sizeof_spinor_field );
  
        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], 0 );
        if(exitstatus < 0) {
          fprintf(stderr, "[kaon2pt_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[0], mzzinv[0], 1 );
        }
   
        memcpy( stochastic_propagator_mom_smeared_list[ipi][i], spinor_work[1], sizeof_spinor_field);

        if ( N_Jacobi > 0 ) {
          /***************************************************************************
           * SINK SMEARING
           ***************************************************************************/
          exitstatus = Jacobi_Smearing ( gauge_field_smeared,  stochastic_propagator_mom_smeared_list[ipi][i], N_Jacobi, kappa_Jacobi);
          if(exitstatus != 0) {
            fprintf(stderr, "[kaon2pt_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            return(11);
          }
        }
  
      }  /* end of loop on spin color dilution indices */

    }  /* end of loop on source momenta */
    

    /***************************************************************************
     * invert for stochastic timeslice propagator at zero momentum
     *
     * ONLY FOR FLAVOR STRANGE, UP-TYPE
     ***************************************************************************/
  
    /***************************************************************************
    * 3-momentum vector at source set to zero
     ***************************************************************************/
    int source_momentum[3] = { 0, 0, 0 };

    /***************************************************************************
     * prepare stochastic timeslice source at source momentum
     ***************************************************************************/
    exitstatus = init_timeslice_source_oet ( stochastic_source_list, gts, source_momentum, spin_dilution, color_dilution, 0 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[kaon2pt_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(64);
    }
      
    if ( g_write_source ) {
      for ( int i = 0; i < spin_color_dilution; i++ ) {
        sprintf(filename, "%s.c%d.t%d.i%d.s%d.zeromom", filename_prefix, Nconf, gts, i, isample );
        if ( ( exitstatus = write_propagator( stochastic_source_list[i], filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[kaon2pt_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }
    }

    /***************************************************************************
     * SOURCE SMEARING
     ***************************************************************************/
    if ( N_Jacobi > 0 ) {
      for( int i = 0; i < spin_color_dilution; i++) {
        exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_source_list[i], N_Jacobi, kappa_Jacobi);
        if(exitstatus != 0) {
          fprintf(stderr, "[kaon2pt_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          return(11);
        }
      }
    }

    /***************************************************************************
     * invert for flavor id 2 (up-type strange)
     ***************************************************************************/
    for( int i = 0; i < spin_color_dilution; i++) {

      memcpy ( spinor_work[0], stochastic_source_list[i], sizeof_spinor_field );

      memset ( spinor_work[1], 0, sizeof_spinor_field );
  
      exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], 2 );
      if(exitstatus < 0) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(44);
      }

      if ( check_propagator_residual ) {
        check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, smzz[0], smzzinv[0], 0 );
      }
 
      memcpy( stochastic_propagator_zero_smeared_list[0][i], spinor_work[1], sizeof_spinor_field);
  
      /***************************************************************************
       * SINK SMEARING
       ***************************************************************************/
      if ( N_Jacobi > 0 ) {
        exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_propagator_zero_smeared_list[0][i], N_Jacobi, kappa_Jacobi);
        if(exitstatus != 0) {
          fprintf(stderr, "[kaon2pt_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(11);
        }
      }
    }  /* end of loop on spinor components */

    /***************************************************************************
     * contractions for 2-point functions
     *
     * loop on Gamma strcutures
     ***************************************************************************/
    int source_gamma = 5;
    int sink_gamma   = source_gamma;

    for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {
      int source_momentum[3] = {
        g_source_momentum_list[ipi][0],
        g_source_momentum_list[ipi][1],
        g_source_momentum_list[ipi][2]
      };

      int sink_momentum[3] = {
        -source_momentum[0],
        -source_momentum[1],
        -source_momentum[2]
      };
          

      /* allocate contraction fields in position and momentum space */
      double * contr_x = init_1level_dtable ( 2 * VOLUME );
      if ( contr_x == NULL ) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(3);
      }
    
      double * contr_p = init_1level_dtable ( 2 * T );
      if ( contr_p == NULL ) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(3);
      }
    
      /***************************************************************************
       *
       ***************************************************************************/
      contract_twopoint_xdep ( contr_x, source_gamma, sink_gamma, 
          stochastic_propagator_zero_smeared_list[0], 
          stochastic_propagator_mom_smeared_list[ipi],
          spin_dilution, color_dilution, 1, 1., 64 );
    
      /* momentum projection at sink */
      exitstatus = momentum_projection ( contr_x, contr_p, T, 1, &sink_momentum );
      if(exitstatus != 0) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }
    
     sprintf ( data_tag, "/%s-gf-%s-gi/mu%6.4f/mu%6.4f/t%d/s%d/gf%d/gi%d/pix%dpiy%dpiz%d", flavor_tag[3], flavor_tag[0],
         muval[3], muval[0],
         gts, isample,
         sink_gamma, source_gamma,
         source_momentum[0], source_momentum[1], source_momentum[2] );
    
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
      exitstatus = contract_write_to_aff_file ( &contr_p, affw, data_tag, &sink_momentum, 1, io_proc );
#elif ( defined HAVE_HDF5 )          
      exitstatus = contract_write_to_h5_file ( &contr_p, output_filename, data_tag, &sink_momentum, 1, io_proc );
#endif
      if(exitstatus != 0) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }
     
      /***************************************************************************/
      /***************************************************************************/

      memset ( contr_x , 0, 2 * VOLUME * sizeof(double) );
      memset ( contr_p , 0, 2 * T * sizeof (double ) );

      contract_twopoint_xdep ( contr_x, source_gamma, sink_gamma,
        stochastic_propagator_mom_smeared_list[0],
        stochastic_propagator_mom_smeared_list[ipi],
        spin_dilution, color_dilution, 1, 1., 64 );

      /* momentum projection at sink */
      exitstatus = momentum_projection ( contr_x, contr_p, T, 1, &sink_momentum );
      if(exitstatus != 0) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

     sprintf ( data_tag, "/%s-gf-%s-gi/mu%6.4f/mu%6.4f/t%d/s%d/gf%d/gi%d/pix%dpiy%dpiz%d", flavor_tag[1], flavor_tag[0],
         muval[1], muval[0],
         gts, isample,
         sink_gamma, source_gamma,
         source_momentum[0], source_momentum[1], source_momentum[2] );

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
      exitstatus = contract_write_to_aff_file ( &contr_p, affw, data_tag, &sink_momentum, 1, io_proc );
#elif ( defined HAVE_HDF5 )
      exitstatus = contract_write_to_h5_file ( &contr_p, output_filename, data_tag, &sink_momentum, 1, io_proc );
#endif
      if(exitstatus != 0) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

      /***************************************************************************/
      /***************************************************************************/

      /* deallocate the contraction fields */       
      fini_1level_dtable ( &contr_x );
      fini_1level_dtable ( &contr_p );
  
    }  /* end of loop on momenta at source */
  
    /*****************************************************************/
    /*****************************************************************/

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[kaon2pt_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */

#endif  /* of ifdef HAVE_LHPC_AFF */

    exitstatus = init_timeslice_source_oet ( NULL, -1, NULL, 0, 0, -2 );


    /***************************************************************************
     * sync + retrieve + save rng state
     ***************************************************************************/

    if ( g_verbose > 0 ) {
      for ( int i = 0; i < rlxd_size(); i++ ) {
        fprintf ( stdout, "rng proc %2d sample %d  %10d\n", g_cart_id, isample, rng_state[i] );
      }
    }

    exitstatus = sync_rng_state ( rng_state, 0, 0 );
    if(exitstatus != 0) {
      fprintf(stderr, "[kaon2pt_invert_contract] Error from sync_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    exitstatus = get_rng_state ( rng_state );
    if(exitstatus != 0) {
      fprintf(stderr, "[kaon2pt_invert_contract] Error from get_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    exitstatus = save_rng_state ( 0, NULL );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[kaon2pt_invert_contract] Error from save_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
      EXIT(38);
    }


  }  /* end of loop on samples */

  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_3level_dtable ( &stochastic_propagator_mom_smeared_list );
  fini_3level_dtable ( &stochastic_propagator_zero_smeared_list );
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
  free( gauge_field_smeared );

  /* free clover matrix terms */
  fini_clover ( &mzz, &mzzinv );
  fini_clover ( &smzz, &smzzinv );


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
    fprintf(stdout, "# [kaon2pt_invert_contract] %s# [kaon2pt_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [kaon2pt_invert_contract] %s# [kaon2pt_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
