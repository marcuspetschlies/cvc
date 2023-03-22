/****************************************************
 * avgx23_invert_contract
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

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

#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_ST 2

#define _DERIV  1
#define _DDERIV 0

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to calculate charged pion FF inversions + contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "avgx";

  const char fbwd_str[2][4] =  { "fwd", "bwd" };
  
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

  char data_tag[400];
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
#endif

  int const idx_map_dd[12][2] = {
    {0,1},
    {0,2},
    {0,3},
    {1,0},
    {1,2},
    {1,3},
    {2,0},
    {2,1},
    {2,3},
    {3,0},
    {3,1},
    {3,2} 
  };

  int const idx_map_ddd[24][3] = {
    {0,1,2},
    {0,1,3},
    {0,2,1},
    {0,2,3},
    {0,3,1},
    {0,3,2},
    {1,0,2},
    {1,0,3},
    {1,2,0},
    {1,2,3},
    {1,3,0},
    {1,3,2},
    {2,0,1},
    {2,0,3},
    {2,1,0},
    {2,1,3},
    {2,3,0},
    {2,3,1},
    {3,0,1},
    {3,0,2},
    {3,1,0},
    {3,1,2},
    {3,2,0},
    {3,2,1} 
  };

  struct timeval ta, tb, start_time, end_time;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "rh?f:s:c:m:")) != -1) {
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
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  gettimeofday ( &start_time, (struct timezone *)NULL );

  /* set the default values */
  if(filename_set==0) sprintf ( filename, "%s.input", outfile_prefix );
  /* fprintf(stdout, "# [avgx23_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [avgx23_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [avgx23_invert_contract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[avgx23_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [avgx23_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [avgx23_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[avgx23_invert_contract] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[avgx23_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[avgx23_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[avgx23_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[avgx23_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv for light quark
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[avgx23_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv for strange quark
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &smzz, &smzzinv, gauge_field_with_phase, g_mus, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[avgx23_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[avgx23_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [avgx23_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************
   * set valence twisted mass values
   ***********************************************************/
  double const muval[4] =  { g_mu, -g_mu, g_mus, -g_mus };

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  size_t nelem = _GSI( VOLUME+RAND );
  double ** spinor_work  = init_2level_dtable ( 2, nelem );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[avgx23_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
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
    fprintf(stderr, "[avgx23_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  /***************************************************************************/

  double ** stochastic_propagator_zero_list = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_propagator_zero_list == NULL ) {
    fprintf(stderr, "[avgx23_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double ** stochastic_propagator_zero_smeared_list = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_propagator_zero_smeared_list == NULL ) {
    fprintf(stderr, "[avgx23_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  /***************************************************************************/

  double ** stochastic_source_list = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_source_list == NULL ) {
    fprintf(stderr, "[avgx23_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
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

#ifndef _SMEAR_QUDA 

    alloc_gauge_field ( &gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy ( gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
#endif
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[avgx23_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
#ifndef _SMEAR_QUDA
    }  /* end of if N_aoe > 0 */

    exitstatus = plaquetteria( gauge_field_smeared );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

#endif

  }  /* end of if N_Jacobi > 0 */

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[avgx23_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }
 
  if ( g_verbose > 4 ) {
    for ( int i = 0; i < rlxd_size(); i++ ) {
      fprintf ( stdout, "rng %2d %10d\n", g_cart_id, rng_state[i] );
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

#ifdef HAVE_MPI
    if (  MPI_Bcast( &gts, 1, MPI_INT, 0, g_cart_grid ) != MPI_SUCCESS ) {
      fprintf ( stderr, "[avgx23_invert_contract] Error from MPI_Bcast %s %d\n", __FILE__, __LINE__ );
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
      fprintf(stderr, "[avgx23_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
        fprintf(stderr, "[avgx23_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#elif ( defined HAVE_HDF5 )
    sprintf ( output_filename, "%s.%.4d.t%d.s%d.h5", g_outfile_prefix, Nconf, gts, isample );
#endif
    if(io_proc == 2 && g_verbose > 1 ) { 
      fprintf(stdout, "# [avgx23_invert_contract] writing data to file %s\n", output_filename);
    }

    /***************************************************************************
     * synchronize rng states to state at zero
     ***************************************************************************/
    exitstatus = sync_rng_state ( rng_state, 0, 0 );
    if(exitstatus != 0) {
      fprintf(stderr, "[avgx23_invert_contract] Error from sync_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    /***************************************************************************
     * read stochastic oet source from file
     ***************************************************************************/
    if ( g_read_source ) {
      for ( int i = 0; i < spin_color_dilution; i++ ) {
        sprintf(filename, "%s.%.4d.t%d.%d.%.5d", filename_prefix, Nconf, gts, i, isample);
        if ( ( exitstatus = read_lime_spinor( stochastic_source_list[i], filename, 0) ) != 0 ) {
          fprintf(stderr, "[avgx23_invert_contract] Error from read_lime_spinor, status was %d\n", exitstatus);
          EXIT(2);
        }
      }
      /* recover the ran field */
      exitstatus = init_timeslice_source_oet(stochastic_source_list, gts, NULL, spin_dilution, color_dilution,  -1 );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[avgx23_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
        fprintf(stderr, "[avgx23_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(64);
      }
      if ( g_write_source ) {
        for ( int i = 0; i < spin_color_dilution; i++ ) {
          sprintf(filename, "%s.%.4d.t%d.%d.%.5d", filename_prefix, Nconf, gts, i, isample);
          if ( ( exitstatus = write_propagator( stochastic_source_list[i], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[avgx23_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
      fprintf(stderr, "[avgx23_invert_contract] Error from get_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    exitstatus = save_rng_state ( 0, NULL );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[avgx23_invert_contract] Error from save_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
      EXIT(38);
    }

    /***************************************************************************
     * prepare stochastic timeslice source at source momentum
     *
     * FOR ALL FLAVORS, LIGHT = UP, DN and STRANGE = SP, SM
     ***************************************************************************/

    int source_momentum[3] = { 0, 0, 0 };
    exitstatus = init_timeslice_source_oet ( stochastic_source_list, gts, source_momentum, spin_dilution, color_dilution, 0 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[avgx23_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(64);
    }

    if ( N_Jacobi > 0 ) {
      gettimeofday ( &ta, (struct timezone *)NULL );
      /***************************************************************************
       * SOURCE SMEARING
       ***************************************************************************/

      for( int i = 0; i < spin_color_dilution; i++) {
        exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_source_list[i], N_Jacobi, kappa_Jacobi);
        if(exitstatus != 0) {
          fprintf(stderr, "[avgx23_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          return(11);
        }
      }
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "avgx23_invert_contract", "Jacobi_Smearing-diluted-stochastic-source", g_cart_id == 0 );
    }

    /***************************************************************************
     * loop on quark flavor
     *
     * this flavor is used to apply dd and ddd to
     * runs over all flavors u, d, s+, s-
     ***************************************************************************/
    for ( int iflavor = 0; iflavor < 4; iflavor++ )
    {

      /***************************************************************************
       * invert for stochastic timeslice propagator at zero momentum
       *   dn flavor
       *   this one will run from source to sink as part of the sequential
       *   propagator
       ***************************************************************************/
      for( int i = 0; i < spin_color_dilution; i++) {

        memcpy ( spinor_work[0], stochastic_source_list[i], sizeof_spinor_field );

  
        memset ( spinor_work[1], 0, sizeof_spinor_field );
  
        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
        if(exitstatus < 0) {
          fprintf(stderr, "[avgx23_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        if ( check_propagator_residual ) {
          if ( iflavor < 2 ) {
            check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
          } else {
            check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, smzz[iflavor%2], smzzinv[iflavor%2], 1 );
          }
        }
 
        memcpy( stochastic_propagator_zero_list[i],         spinor_work[1], sizeof_spinor_field);

        memcpy( stochastic_propagator_zero_smeared_list[i], spinor_work[1], sizeof_spinor_field);
 
        if ( N_Jacobi > 0 ) {
          gettimeofday ( &ta, (struct timezone *)NULL );

          /***************************************************************************
           * SOURCE SMEARING
           ***************************************************************************/
          exitstatus = Jacobi_Smearing ( gauge_field_smeared,  stochastic_propagator_zero_smeared_list[i], N_Jacobi, kappa_Jacobi);
          if(exitstatus != 0) {
            fprintf(stderr, "[avgx23_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            return(11);
          }
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "avgx23_invert_contract", "Jacobi_Smearing-sequential-propagator", g_cart_id == 0 );
        }

      }  /* end of loop on spin color dilution indices */

      /*****************************************************************/
      /*****************************************************************/

      /*****************************************************************
       * displacement fwd, bwd of stoch prop zero, no smearing
       * at sink
       *****************************************************************/
      gettimeofday ( &ta, (struct timezone *)NULL );

      double ***** stochastic_propagator_zero_ddispl_list = init_5level_dtable (12, 2, 2, spin_color_dilution, _GSI ( VOLUME ) );
      if ( stochastic_propagator_zero_ddispl_list == NULL )
      {
        fprintf( stderr, "[avgx23_invert_contract] Error from init_5level_dtable  %s %d\n", __FILE__, __LINE__ );
        EXIT(12);
      }
      double ****** stochastic_propagator_zero_dddispl_list = init_6level_dtable (24, 2, 2, 2, spin_color_dilution, _GSI ( VOLUME ) );
      if ( stochastic_propagator_zero_dddispl_list == NULL )
      {
        fprintf( stderr, "[avgx23_invert_contract] Error from init_6level_dtable  %s %d\n", __FILE__, __LINE__ );
        EXIT(12);
      }

      for ( int isc = 0; isc < spin_color_dilution; isc++ )
      {
        for ( int k = 0; k < 4; k++ ) 
        {
          for ( int ifbwd = 0; ifbwd < 2; ifbwd++ ) 
          {
            exitstatus = spinor_field_eq_cov_displ_spinor_field ( spinor_work[0], stochastic_propagator_zero_list[isc],
                    idx_map_dd[3*k][0], ifbwd, gauge_field_with_phase );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[avgx23_invert_contract] Error from spinor_field_eq_cov_displ_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(33);
            }

            for ( int l = 0; l < 3; l++ )
            {
              for ( int ifbwd2 = 0; ifbwd2 < 2; ifbwd2++ ) 
              {
                exitstatus = spinor_field_eq_cov_displ_spinor_field ( stochastic_propagator_zero_ddispl_list[3*k+l][ifbwd2][ifbwd][isc], spinor_work[0],
                    idx_map_dd[3*k+l][1], ifbwd2, gauge_field_with_phase );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[avgx23_invert_contract] Error from spinor_field_eq_cov_displ_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                   EXIT(33);
                }
            
                for ( int m = 0; m < 2; m++ )
                {
                  for ( int ifbwd3 = 0; ifbwd3 < 2; ifbwd3++ ) 
                  {
                    exitstatus = spinor_field_eq_cov_displ_spinor_field ( stochastic_propagator_zero_dddispl_list[2*(3*k+l)+m][ifbwd3][ifbwd2][ifbwd][isc],
                        stochastic_propagator_zero_ddispl_list[3*k+l][ifbwd2][ifbwd][isc],
                        idx_map_ddd[2*(3*k+l)+m][2], ifbwd3, gauge_field_with_phase );
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[avgx23_invert_contract] Error from spinor_field_eq_cov_displ_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(33);
                    }


                  }  /* end of loop on fbwd3 */
                }  /* end of 3rd direction m */
                  
              }  /* end of loop on fbwd2 */
            }  /* end of 3rd direction l */

          }  /* end of loop on fbwd */
        }  /* end of 3rd direction k */

      }  /* end of loop on isc */
          
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "avgx23_invert_contract", "apply-dd-and-ddd-to-zero-mom-prop", g_cart_id == 0 );

      /***************************************************************************/
      /***************************************************************************/
  
      /***************************************************************************
       * invert for stochastic timeslice propagator at source momenta
       *
       * ONLY FOR FLAVORS iflavor2, see below
       ***************************************************************************/
      for ( int isrc_mom = 0; isrc_mom < g_source_momentum_number; isrc_mom++ ) 
      {

        /***************************************************************************
         * 3-momentum vector at source
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
          fprintf(stderr, "[avgx23_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(64);
        }

        /***************************************************************************
         * source-smearing
         ***************************************************************************/
        if ( N_Jacobi > 0 ) {
          gettimeofday ( &ta, (struct timezone *)NULL );
 
          for( int i = 0; i < spin_color_dilution; i++) {
            exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_source_list[i], N_Jacobi, kappa_Jacobi);
            if(exitstatus != 0) {
              fprintf(stderr, "[avgx23_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              return(11);
            }
          }
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "avgx23_invert_contract", "Jacobi_Smearing-diluted-stochastic-source", g_cart_id == 0 );
        }

        /***************************************************************************
         * invert for flavors iflavor2, which takes values
         *   iflavor and 2 * ( 1 - ( iflavor / 2) ) + ( iflavor % 2 )
         *
         * general layout of 3-point function is as follows:
         *
         *
         *           /\                      /\
         *          /  \                    /  \
         *         /    \_               _ /    \_
         *  X    |/_    |\  X   => Xbar   /|    |\  X
         *       /        \              /        \
         *      /____\ ____\            /____\_____\
         *           /                       /
         *      
         *           Y                       Y
         *
         *  X and Y must have OPPOSITE SIGN of twisted quark mass
         *  or
         *  Xbar and Y must have SAME SIGN of twisted quark mass, so
         *
         *        iflavor  = Xbar =   u      ,   d      ,   s+,    ,   s-
         *        iflavor2 = Y    = { u, s+ }, { d, s- }, { s+, u }, { s-, d }
         *
         ***************************************************************************/
        int const flavor_seqsrc_list[2] = { iflavor, 2 * ( 1 - ( iflavor / 2) ) + ( iflavor % 2 ) };

        for ( int ifl2 = 0; ifl2 < 2; ifl2++ ) 
        {

          int const iflavor2 = flavor_seqsrc_list[ifl2];

          if ( ( iflavor / 2 == 1 ) && ( iflavor2 / 2 == 1 ) ) continue; /* skip quark flow with only strange flavor propagators */
      
          for( int i = 0; i < spin_color_dilution; i++) 
          {

            memcpy ( spinor_work[0], stochastic_source_list[i], sizeof_spinor_field );
 
            memset ( spinor_work[1], 0, sizeof_spinor_field );

            exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor2 );
            if(exitstatus < 0) {
              fprintf(stderr, "[avgx23_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(44);
            }

            if ( check_propagator_residual ) 
            {
              double ** const _mzz    = ( iflavor2/2 == 0 ) ? mzz[iflavor2]    : smzz[iflavor2-2];
              double ** const _mzzinv = ( iflavor2/2 == 0 ) ? mzzinv[iflavor2] : smzzinv[iflavor2-2];

              check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, _mzz, _mzzinv, 1 );
            }

            memcpy( stochastic_propagator_mom_list[i], spinor_work[1], sizeof_spinor_field);

          }  /* end of loop on spinor components */

          /***************************************************************************
           * SINK SMEARING of the momentum propagator
           *
           * need both smeared and unsmeared later on, so keep in separate list
           ***************************************************************************/
          double ** stochastic_propagator_mom_smeared_list = init_2level_dtable ( spin_color_dilution, _GSI(VOLUME) );
          if ( stochastic_propagator_mom_smeared_list == NULL ) {
            fprintf(stderr, "[avgx23_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(48);
          }

          memcpy ( stochastic_propagator_mom_smeared_list[0], stochastic_propagator_mom_list[0], spin_color_dilution * sizeof_spinor_field );

          if ( N_Jacobi > 0 ) 
          {
            gettimeofday ( &ta, (struct timezone *)NULL );
  
            for( int i = 0; i < spin_color_dilution; i++) {
 
              exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_propagator_mom_smeared_list[i], N_Jacobi, kappa_Jacobi);
              if(exitstatus != 0) {
                fprintf(stderr, "[avgx23_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                return(11);
              }
            }
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "avgx23_invert_contract", "Jacobi_Smearing-diluted-stochastic-propagator", g_cart_id == 0 );
          }
 
          /***************************************************************************/
          /***************************************************************************/

          /***************************************************************************
           ***************************************************************************
           **
           ** contractions for 2-point functions
           **
           ** no need for a separate loop on sink gamma, same as at source
           **
           ** for Tr [ X(0) Gf Y(p) Gi ] = Xbar(0)^+ g5 Gf Y(p) Gi g5
           ** iflavor:  X = u,d,s+,s-, at zero momentum
           ** iflavor2: Y = according to formula
           **
           ***************************************************************************
           ***************************************************************************/
            
          gettimeofday ( &ta, (struct timezone *)NULL );

          int source_gamma = 5;

          int sink_gamma   = source_gamma;
        
          /* allocate contraction fields in position and momentum space */
          double * contr_x = init_1level_dtable ( 2 * VOLUME );
          if ( contr_x == NULL ) {
            fprintf(stderr, "[avgx23_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(3);
          }
  
          double * contr_p = init_1level_dtable ( 2 * T );
          if ( contr_p == NULL ) {
            fprintf(stderr, "[avgx23_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(3);
          }
  
          /***************************************************************************
           * Xbar(0)^+ g5 Gf Y(p) Gi g5
           *
           ***************************************************************************/
          contract_twopoint_xdep ( contr_x, source_gamma, sink_gamma, 
              stochastic_propagator_zero_smeared_list, 
              stochastic_propagator_mom_smeared_list,
              spin_dilution, color_dilution, 1, 1., 64 );
  
          /***************************************************************************
           * p_sink = -p_source, only one vector
           ***************************************************************************/
          int sink_momentum[3] = {
            -source_momentum[0],
            -source_momentum[1],
            -source_momentum[2] };

          /* momentum projection at sink */
          exitstatus = momentum_projection ( contr_x, contr_p, T, 1, &sink_momentum );
          if(exitstatus != 0) {
            fprintf(stderr, "[avgx23_invert_contract] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(3);
          }
  
          sprintf ( data_tag, "/%s-gf-%s-gi/mu%6.4f/mu%6.4f", 
              flavor_tag[2*(iflavor/2)+1-(iflavor%2)], flavor_tag[iflavor2],
              muval[2*(iflavor/2) + 1-(iflavor%2)], muval[iflavor2] );
  
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
          exitstatus = contract_write_to_aff_file ( &contr_p, affw, data_tag, &sink_momentum, 1, io_proc );
#elif ( defined HAVE_HDF5 )          
          exitstatus = contract_write_to_h5_file ( &contr_p, output_filename, data_tag, &sink_momentum, 1, io_proc );
#endif
          if(exitstatus != 0) {
            fprintf(stderr, "[avgx23_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }
   
          /* deallocate the contraction fields */       
          fini_1level_dtable ( &contr_x );
          fini_1level_dtable ( &contr_p );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "avgx23_invert_contract", "contract-io-twop", g_cart_id == 0 );

          /*****************************************************************/
          /*****************************************************************/

          /*****************************************************************
           *****************************************************************
           **
           ** sequential propagator part, 3-point function
           **
           *****************************************************************
           *****************************************************************/

          /*****************************************************************
           * loop on sequential source timeslices
           *****************************************************************/
          for ( int iseq_timeslice = 0; iseq_timeslice < g_sequential_source_timeslice_number; iseq_timeslice++ )
          {

            /*****************************************************************
             * global sequential source timeslice
             * NOTE: counted from current source timeslice
             *****************************************************************/
            int gtseq = ( gts + g_sequential_source_timeslice_list[iseq_timeslice] + T_global ) % T_global;

            double ** sequential_propagator_list = init_2level_dtable ( spin_color_dilution, nelem );
            if ( sequential_propagator_list == NULL ) {
              fprintf(stderr, "[avgx23_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
              EXIT(48);
            }

            /*****************************************************************
             * seq source mom is zero always
             *****************************************************************/
            int current_momentum[3] = { 0, 0, 0 };

            int sink_momentum[3] = {
                -( current_momentum[0] + source_momentum[0] ),
                -( current_momentum[1] + source_momentum[1] ),
                -( current_momentum[2] + source_momentum[2] ) };

            /*****************************************************************
             *****************************************************************
             **
             ** iflavor3 - after - iflavor2
             **
             ** iflavor3 is opposite tm sign compared to iflavor
             **
             ** SEQ SOURCE FROM LIGHT PROPAGATOR INCLUDING MOMENTUM PHASE AT SOURCE
             **
             ** the seq source momentum is the one at sink
             **
             *****************************************************************
             *****************************************************************/
            int iflavor3 = 2 * ( iflavor / 2 ) + 1 - ( iflavor % 2 );

            /*****************************************************************
             * invert for sequential timeslice propagator
             *****************************************************************/
            for ( int i = 0; i < spin_color_dilution; i++ ) 
            {

              if ( g_cart_id == 0 && g_verbose > 2 ) {
                fprintf ( stdout, "# [avgx23_invert_contract] start LIGHT-AFTER-LIGHT with tseq = %d, pf = %3d %3d %3d, pi = %3d %3d %3d, sc = %d  %s %d\n",
                    gtseq,
                    sink_momentum[0], sink_momentum[1], sink_momentum[2],
                    source_momentum[0], source_momentum[1], source_momentum[2],
                    i,
                    __FILE__, __LINE__ );
              }
                
              /*****************************************************************
               * prepare sequential timeslice source 
               *
               * THROUGH THE SINK, so use the SINK SMEARED stochastic zero momentum propagator
               *****************************************************************/
     
              /*****************************************************************
               * sequential source
               *****************************************************************/
              exitstatus = init_sequential_source ( spinor_work[0], stochastic_propagator_mom_smeared_list[i], gtseq, sink_momentum, sink_gamma );
              if( exitstatus != 0 ) {
                fprintf(stderr, "[avgx23_invert_contract] Error from init_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(64);
              }

              if ( N_Jacobi > 0 ) {
                gettimeofday ( &ta, (struct timezone *)NULL );

                /***************************************************************************
                 * SINK SMEARING THE SEQUENTIAL SOURCE
                 ***************************************************************************/
                exitstatus = Jacobi_Smearing ( gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);
                if(exitstatus != 0) {
                  fprintf(stderr, "[avgx23_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  return(11);
                } 
                gettimeofday ( &tb, (struct timezone *)NULL );
                show_time ( &ta, &tb, "avgx23_invert_contract", "Jacobi_Smearing-sequential-source", g_cart_id == 0 );
              }

              memset ( spinor_work[1], 0, sizeof_spinor_field );

              /***************************************************************************
               * invert on the sequential source
               ***************************************************************************/
              exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor3 );
              if(exitstatus < 0) {
                fprintf(stderr, "[avgx23_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(44);
              }

              if ( check_propagator_residual ) 
              {
                double ** const _mzz    = ( iflavor3 / 2 == 0 ) ? mzz[iflavor3]    : smzz[iflavor3-2];
                double ** const _mzzinv = ( iflavor3 / 2 == 0 ) ? mzzinv[iflavor3] : smzzinv[iflavor3-2];

                check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, _mzz, _mzzinv, 1 );
              }

              /***************************************************************************
               * NO SMEARING AT THIS END OF THE PRPOAGATOR
               *
               * this end runs to the insertion
               ***************************************************************************/

              memcpy( sequential_propagator_list[i], spinor_work[1], sizeof_spinor_field );
            }  /* end of loop on oet spin components */

            /*****************************************************************/
            /*****************************************************************/

            gettimeofday ( &ta, (struct timezone *)NULL );

            /*****************************************************************
             * contractions for covariant displacement insertion
             *****************************************************************/
                    
            double * contr_p = init_1level_dtable (  2*T );
            if ( contr_p == NULL ) {
              fprintf(stderr, "[avgx23_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
              EXIT(47);
            }
                

            /*****************************************************************
             * loop on directions for 2 covariant displacements
             *****************************************************************/
            for ( int k = 0; k < 12; k++ ) 
            {
              int const mu = idx_map_dd[k][0];
              int const nu = idx_map_dd[k][1];
                    
              /*****************************************************************
               * loop on fbwd for covariant displacements
               *****************************************************************/
              for ( int ifbwd = 0; ifbwd <= 1; ifbwd++ )
              {
                for ( int ifbwd2 = 0; ifbwd2 <= 1; ifbwd2++ )
                {

                  for ( int kappa = 0; kappa < 4; kappa++)
                  {
                    if ( mu == kappa || nu == kappa ) continue;

                    memset ( contr_p, 0, 2 * T * sizeof ( double ) );

                    /*****************************************************************
                     * DD contraction
                     * [ DD fwd(0) ] ^+ g5 Gc seq Gi g5
                     * Gc = current_gamma
                     * Gi = source_gamma 
                     *
                     * seq was produced for flavor iflavor3 - after - iflavor2
                     *
                     *****************************************************************/
                    contract_twopoint_snk_momentum ( contr_p, source_gamma, kappa,
                          stochastic_propagator_zero_ddispl_list[k][ifbwd2][ifbwd], 
                          sequential_propagator_list, spin_dilution, color_dilution, current_momentum, 1);

                    sprintf ( data_tag, "/DD%s-gc-%s%s-gi/mu%6.4f/mu%6.4f/mu%6.4f/dt%d/gc%d_d%d_%s_d%d_%s/", 
                        flavor_tag[2*(iflavor/2) + 1-(iflavor%2)],
                        flavor_tag[iflavor3], flavor_tag[iflavor2],
                        muval[2*(iflavor/2)+1-(iflavor%2)], muval[iflavor3], muval[iflavor2],
                        g_sequential_source_timeslice_list[iseq_timeslice],
                        kappa, 
                        nu, fbwd_str[ifbwd2], 
                        mu, fbwd_str[ifbwd] );

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
                    exitstatus = contract_write_to_aff_file ( &contr_p, affw, data_tag, &sink_momentum, 1, io_proc );
#elif ( defined HAVE_HDF5 )
                    exitstatus = contract_write_to_h5_file ( &contr_p, output_filename, data_tag, &sink_momentum, 1, io_proc );
#endif
                    if(exitstatus != 0) {
                      fprintf(stderr, "[avgx23_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(3);
                    }
              
                  }  /* end of loop on kappa => current gamma */

                  /*****************************************************************
                   * DDD contraction
                   * [ DDD fwd(0) ] ^+ g5 Gc seq Gi g5
                   * Gc = current_gamma
                   * Gi = source_gamma 
                   *
                   * seq was produced for flavor iflavor3 - after - iflavor2
                   *
                   *****************************************************************/
                  for ( int ilda = 0; ilda < 2; ilda++ )
                  {

                    /* skip any momentum vector, which has at least one component equal to zero */
                    if ( ( source_momentum[0] == 0 ) || ( source_momentum[1] == 0 ) || ( source_momentum[2] == 0 ) ) continue;

                    int const lambda = idx_map_ddd[2*k+ilda][2];

                    for ( int ifbwd3 = 0; ifbwd3 < 2; ifbwd3++ )
                    {

                      for ( int kappa = 0; kappa < 4; kappa++)
                      {
                        if ( mu == kappa || nu == kappa || lambda == kappa ) continue;

                        memset ( contr_p, 0, 2 * T * sizeof ( double ) );

                        contract_twopoint_snk_momentum ( contr_p, source_gamma, kappa, 
                           stochastic_propagator_zero_dddispl_list[2*k+ilda][ifbwd3][ifbwd2][ifbwd], 
                           sequential_propagator_list, spin_dilution, color_dilution, current_momentum, 1);

                        sprintf ( data_tag, "/DDD%s-gc-%s%s-gi/mu%6.4f/mu%6.4f/mu%6.4f/dt%d/gc%d_d%d_%s_d%d_%s/d%d_%s", 
                            flavor_tag[2*(iflavor/2) + 1-(iflavor%2)],
                            flavor_tag[iflavor3], flavor_tag[iflavor2],
                            muval[2*(iflavor/2)+1-(iflavor%2)], muval[iflavor3], muval[iflavor2],
                            g_sequential_source_timeslice_list[iseq_timeslice],
                            kappa, 
                            lambda, fbwd_str[ifbwd3], 
                            nu,     fbwd_str[ifbwd2], 
                            mu,     fbwd_str[ifbwd] );

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
                        exitstatus = contract_write_to_aff_file ( &contr_p, affw, data_tag, &sink_momentum, 1, io_proc );
#elif ( defined HAVE_HDF5 )
                        exitstatus = contract_write_to_h5_file ( &contr_p, output_filename, data_tag, &sink_momentum, 1, io_proc );
#endif
                        if(exitstatus != 0) {
                          fprintf(stderr, "[avgx23_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                          EXIT(3);
                        }
              
                      }  /* end of loop on kappa => current gamma */

                    }  /* end of loop on fbwd3 */
                  }  /* end of loop on ilda => lambda */

                }  /* end of loop on fbwd2 */
              }  /* end of loop on fbwd */
                  
            }  /* end of loop on k => nu, mu */
              
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "avgx23_invert_contract", "contract-io-gdd-and-gddd-threep", g_cart_id == 0 );

            /*****************************************************************/
            /*****************************************************************/

            fini_1level_dtable ( &contr_p );

            fini_2level_dtable ( &sequential_propagator_list );

          }  /* end of loop on seq source timeslice */

          fini_2level_dtable ( &stochastic_propagator_mom_smeared_list );

        }  /* end of loop on ifl2 => flavor2 */

      }  /* end of loop on source momentum */
              
      /*****************************************************************/
      /*****************************************************************/

      fini_5level_dtable ( &stochastic_propagator_zero_ddispl_list );

      fini_6level_dtable ( &stochastic_propagator_zero_dddispl_list );

    }  /* loop on flavor */


#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[avgx23_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */

#endif  /* of ifdef HAVE_LHPC_AFF */

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * free and clean up
     ***************************************************************************/
    exitstatus = init_timeslice_source_oet ( NULL, -1, NULL, 0, 0, -2 );
    
  }  /* end of loop on samples => source timeslices */

  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_2level_dtable ( &stochastic_propagator_mom_list );
  fini_2level_dtable ( &stochastic_propagator_zero_list );
  fini_2level_dtable ( &stochastic_propagator_zero_smeared_list );
  fini_2level_dtable ( &stochastic_source_list );
  fini_2level_dtable ( &spinor_work );

  /***************************************************************************
   * fini rng state
   ***************************************************************************/
  fini_rng_state ( &rng_state);

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  if ( gauge_field_with_phase != NULL ) free( gauge_field_with_phase );
  if ( gauge_field_smeared    != NULL ) free( gauge_field_smeared );

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


  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "avgx23_invert_contract", "runtime", g_cart_id == 0 );

  return(0);

}
