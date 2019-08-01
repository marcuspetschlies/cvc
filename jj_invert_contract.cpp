/****************************************************
 * jj_invert_contract
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
#include "smearing_techniques.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_ST 2

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to calculate JJ inversions + contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}


/***************************************************************************
 * extract oet spin-color component from volume source
 ***************************************************************************/
inline void get_spin_color_oet_comp ( double * const r_out, double * const r_in, int const is, int const ic, int const sd, int const cd , int const ts_loc ) {
 
  int const spin_dim  = ( sd == 0 ) ? 0 : 4 / sd;
  int const color_dim = ( cd == 0 ) ? 0 : 3 / cd;
  int const isc = cd * is + ic;
  unsigned int const VOL3 = LX * LY * LZ;

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int ix = 0; ix < VOL3; ix++ ) {
    unsigned int const iix = _GSI( ts_loc * VOL3 + ix );

    /* set ith spin-component in ith spinor field */

    for ( int j = 0; j < spin_dim;  j++ ) {
    for ( int k = 0; k < color_dim; k++ ) {
      r_out[ iix + 2 * ( 3 * ( is * spin_dim + j ) + ( ic * color_dim + k ) )     ] = r_in[ iix + 2 * ( color_dim * j + k )     ];
      r_out[ iix + 2 * ( 3 * ( is * spin_dim + j ) + ( ic * color_dim + k ) ) + 1 ] = r_in[ iix + 2 * ( color_dim * j + k ) + 1 ];

    }}  /* end of loop on non-diluted spin-color indices */
  }  /* of loop on 3-volume */
  return;
}  /* end of get_spin_color_oet_comp */


/***************************************************************************
 *
 * MAIN
 *
 ***************************************************************************/
int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "cpff";

  const char fbwd_str[2][4] =  { "fwd", "bwd" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[100];
  // double ratime, retime;
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double * gauge_field_smeared = NULL;
  double * gauge_field_with_phase = NULL;
  int op_id_up = -1, op_id_dn = -1;
  char output_filename[400];
  int * rng_state = NULL;
  int spin_dilution = 4;
  int color_dilution = 1;
  int nflavor = 2;


  /* vertices */
  int const gamma_vertex_number    = 8;
  int const gamma_vertex_id[2][8]  = { 
                                       { 6,  7,  8,  9,  4, 10, 11, 12 },
                                       { 0,  1,  2,  3,  5, 15, 14, 13 } };

  char const gamma_vertex_name[8][8] = { "gt", "gx", "gy",  "gz" , "1", "gxgt" , "gygt", "gzgt" };
 
  char const flavor_combination[2][10] = { "dn-g-up-g", "up-g-up-g" };

  char data_tag[500];
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "rh?f:s:c:")) != -1) {
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
  /* fprintf(stdout, "# [jj_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

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
    fprintf(stdout, "# [jj_invert_contract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[jj_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field           = _GSI(VOLUME) * sizeof(double);
  size_t const sizeof_spinor_field_timeslice = _GSI(VOL3)   * sizeof(double);

  /***************************************************************************
   *
   * check spin - color  dilution scheme
   * FOR NOW ONLY ACCEPT STD OET 4,1
   *
   ***************************************************************************/
  if ( spin_dilution != 4 || color_dilution != 1 )  {
    fprintf(stderr, "[jj_invert_contract] Error, (spin,color)-dilution must be (4,1) %s %d\n", __FILE__, __LINE__);
    EXIT(171);
  }

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
    if(g_cart_id==0) fprintf(stdout, "# [jj_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [jj_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[jj_invert_contract] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[jj_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[jj_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***************************************************************************
   * APE-smearing of gauge field
   ***************************************************************************/
  if( g_gaussian_smearing_level_number > 0 ) {

    /***************************************************************************
     * NOTE: gauge_field_smeared, needs boundary
     ***************************************************************************/
    alloc_gauge_field ( &gauge_field_smeared, VOLUMEPLUSRAND );

    memcpy ( gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double) );

    if ( N_ape > 0 ) {
      exitstatus = APE_Smearing ( gauge_field_smeared, alpha_ape, N_ape );
      if ( exitstatus !=  0 ) {
        fprintf(stderr, "[jj_invert_contract] Error from APE_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(47);
      }

    }  /* end of if N_ape > 0 */

    /* check the plaquettes for the smeared gauge field */
    exitstatus = plaquetteria ( gauge_field_smeared );
    if(exitstatus != 0) {
      fprintf(stderr, "[jj_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

  }  /* end of if g_gaussian_smearing_level_number > 0 */

  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[jj_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes for the original gauge field with boundary phase
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[jj_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[jj_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[jj_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [jj_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

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
   * transcribe the smearing level list
   ***************************************************************************/
  int smearing_level_number = g_gaussian_smearing_level_number + 1;
  gaussian_smearing_level_type * smearing_level_list = (gaussian_smearing_level_type*) malloc ( smearing_level_number * sizeof ( gaussian_smearing_level_type ) );
  for ( int i = 0; i < smearing_level_number; i++ ) {
    smearing_level_list[i].n     = ( i == 0 ) ? 0  : g_gaussian_smearing_level[i-1].n;
    smearing_level_list[i].alpha = ( i == 0 ) ? 0. : g_gaussian_smearing_level[i-1].alpha;
  }

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  double ** spinor_work  = init_2level_dtable ( 3, _GSI( VOLUME+RAND ) );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[jj_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * allocate memory for spinor fields
   * WITHOUT halo
   ***************************************************************************/
  int const spin_color_dilution = spin_dilution * color_dilution;

  double ** stochastic_propagator_mom = init_2level_dtable ( spin_color_dilution, _GSI( VOLUME ) );
  if ( stochastic_propagator_mom == NULL ) {
    fprintf(stderr, "[jj_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  /* this is a big field */
  double ***** stochastic_propagator_zero = init_5level_dtable ( smearing_level_number, g_source_location_number, nflavor, spin_color_dilution, _GSI ( VOLUME ) );
  if ( stochastic_propagator_zero == NULL ) {
    fprintf(stderr, "[jj_invert_contract] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  double * stochastic_source = init_1level_dtable ( _GSI ( VOLUME ) );
  if ( stochastic_source == NULL ) {
    fprintf(stderr, "[jj_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  double * stochastic_source_smeared = init_1level_dtable ( _GSI ( VOLUME ) );
  if ( stochastic_source_smeared == NULL ) {
    fprintf(stderr, "[jj_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

#if 0
  double * stochastic_source_zero_smeared = init_2level_dtable ( spin_color_dilution, _GSI ( VOLUME ) );
  if ( stochastic_source_zero_smeared == NULL ) {
    fprintf(stderr, "[jj_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }
#endif

  double ** stochastic_source_mom_smeared = init_2level_dtable ( spin_color_dilution, _GSI ( VOLUME ) );
  if ( stochastic_source_mom_smeared == NULL ) {
    fprintf(stderr, "[jj_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[jj_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   * make the source momentum phase field
   *
   *   source_momentum = - sink_momentum ( given )
   ***************************************************************************/
  g_source_momentum_number = g_sink_momentum_number;

  for ( int i = 0; i < g_sink_momentum_number; i++ ) {
    g_source_momentum_list[i][0] = -g_sink_momentum_list[i][0];
    g_source_momentum_list[i][1] = -g_sink_momentum_list[i][1];
    g_source_momentum_list[i][2] = -g_sink_momentum_list[i][2];
  }

  double _Complex ** ephase = init_2level_ztable ( g_source_momentum_number, VOL3 );
  if ( ephase == NULL ) {
    fprintf ( stderr, "[jj_invert_contract] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

  make_phase_field_timeslice ( ephase, g_source_momentum_number, g_source_momentum_list );

  /***************************************************************************
   *
   * loop on stochastic oet samples
   *
   ***************************************************************************/
  for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

    /***************************************************************************
     *
     * output file
     *
     ***************************************************************************/
#if ( defined HAVE_LHPC_AFF ) && !(defined HAVE_HDF5 )
    /***************************************************************************
     * AFF format
     ***************************************************************************/
    sprintf ( output_filename, "%s.%.4d.aff", outfile_prefix, Nconf );
  
    if(io_proc == 2) {
      affw = aff_writer ( output_filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[jj_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#elif ( defined HAVE_HDF5 )
    /***************************************************************************
     * HDF5 format
     ***************************************************************************/
    sprintf ( output_filename, "%s.%.4d.h5", outfile_prefix, Nconf );
#endif
    if(io_proc == 2 && g_verbose > 1 ) { 
      fprintf(stdout, "# [jj_invert_contract] writing data to file %s\n", output_filename);
    }

    /***************************************************************************
     * synchronize rng states to state at zero
     ***************************************************************************/
    exitstatus = sync_rng_state ( rng_state, 0, 0 );
    if(exitstatus != 0) {
      fprintf(stderr, "[jj_invert_contract] Error from sync_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    if ( g_read_source ) {
      /***************************************************************************
       * read stochastic volume source from file
       ***************************************************************************/
      for ( int i = 0; i < spin_color_dilution; i++ ) {
        sprintf ( filename, "%s.%.4d.%.5d", filename_prefix, Nconf, isample);
        if ( ( exitstatus = read_lime_spinor( stochastic_source, filename, 0) ) != 0 ) {
          fprintf(stderr, "[jj_invert_contract] Error from read_lime_spinor, status was %d\n", exitstatus);
          EXIT(2);
        }
      }
    } else {
      /***************************************************************************
       * generate stochastic volume source
       ***************************************************************************/
      exitstatus = prepare_volume_source ( stochastic_source, VOLUME );

      if( exitstatus != 0 ) {
        fprintf(stderr, "[jj_invert_contract] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(64);
      }
      if ( g_write_source ) {
        sprintf(filename, "%s.%.4d.%.5d", filename_prefix, Nconf, isample);
        if ( ( exitstatus = write_propagator( stochastic_source, filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[jj_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }
    }  /* end of if read stochastic source - else */

    /***************************************************************************
     * retrieve current rng state and 0 writes his state
     ***************************************************************************/
    exitstatus = get_rng_state ( rng_state );
    if(exitstatus != 0) {
      fprintf(stderr, "[jj_invert_contract] Error from get_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }

    exitstatus = save_rng_state ( 0, NULL );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[jj_invert_contract] Error from save_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
      EXIT(38);
    }

    /***************************************************************************
     *
     * loop on source smearing levels
     *
     *   applied to stochastic source at zero momentum only
     *
     ***************************************************************************/
    for ( int ismear = 0; ismear < smearing_level_number; ismear++ ) {

      /***************************************************************************
       * before first smearing, copy the original stochastic oet source set
       ***************************************************************************/
      if ( ismear == 0 ) {
        memcpy ( stochastic_source_smeared, stochastic_source, sizeof_spinor_field );
      }

      /***************************************************************************
       * source smearing parameters
       ***************************************************************************/
      int const    nstep_src = ( ismear == 0 ) ? smearing_level_list[ismear].n : smearing_level_list[ismear].n - smearing_level_list[ismear-1].n;
      double const alpha_src = smearing_level_list[ismear].alpha;

      if ( g_verbose > 2 && g_cart_id == 0 ) fprintf ( stdout, "# [jj_invert_contract] source smearing level %2d parameters N %3d A %6.4f\n", ismear, nstep_src, alpha_src );

      exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_source_smeared, nstep_src, alpha_src );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[jj_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(72);
      }

      if ( g_write_source ) {
        /* write smeared stochastic source at smearing level ismear */
        sprintf(filename, "%s.%.4d.%.5d.Nsrc%d_Asrc%6.4f", filename_prefix, Nconf, isample,
            smearing_level_list[ismear].n, smearing_level_list[ismear].alpha );

        exitstatus = write_propagator( stochastic_source_smeared, filename, 0, g_propagator_precision);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[jj_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }  /* end of if write smeared zero-momentum source */

      /***************************************************************************
       * loop on source timeslices
       ***************************************************************************/
      for( int itsrc = 0; itsrc < g_source_location_number; itsrc++ ) {

        /***************************************************************************
         * local source timeslice and source process ids
         ***************************************************************************/

        int source_timeslice = -1;
        int source_proc_id   = -1;
        int gts              = ( g_source_coords_list[itsrc][0] +  T_global ) %  T_global;

        exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[jj_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(123);
        }

        /***************************************************************************
         * invert for oet stochastic timeslice propagator at zero momentum
         *   for all flavors
         ***************************************************************************/
        for( int is = 0; is < spin_dilution;  is++) {
        for( int ic = 0; ic < color_dilution; ic++) {

          int const isc = is*color_dilution + ic;

          /***************************************************************************
           * init source to zero
           * if have the source timeslice, then extract spin-color component is,ic
           * for local source_timeslice
           ***************************************************************************/
          memset ( spinor_work[2], 0, sizeof_spinor_field );
          if ( source_proc_id == g_cart_id ) {
            get_spin_color_oet_comp ( spinor_work[2], stochastic_source_smeared, is, ic, spin_dilution, color_dilution , source_timeslice );
          }

          /***************************************************************************
           * loop on flavors
           ***************************************************************************/
          for ( int iflavor = 0; iflavor < nflavor; iflavor++ ) {

            /* init solution field to zero */
            memset ( spinor_work[1], 0, sizeof_spinor_field );
            /* copy source to spinor work 0 */
            memcpy ( spinor_work[1], spinor_work[2], sizeof_spinor_field );

            /* call solver via tmLQCD */
            exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
            if(exitstatus < 0) {
              fprintf(stderr, "[jj_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(44);
            }

            /* check propagator residual with original smeared stochastic source */
            if ( check_propagator_residual ) {
              check_residual_clover ( &(spinor_work[1]), &(spinor_work[2]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
            }

            /* copy solution into solution spinor field component */
            memcpy( stochastic_propagator_zero[ismear][itsrc][iflavor][isc], spinor_work[1], sizeof_spinor_field );

            if ( g_write_propagator ) {
              sprintf(filename, "%s.%.4d.fl%d.t%d.d%d.s%.5d.Nsrc%d_Asrc%6.4f.inverted", 
                  filename_prefix, Nconf, iflavor, gts, isc, isample, 
                  smearing_level_list[ismear].n, smearing_level_list[ismear].alpha );
              exitstatus = write_propagator( spinor_work[1], filename, 0, g_propagator_precision);
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[jj_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(2);
              }
            }
          }  /* end of loop on iflavor */
        }  /* end of loop on color dilution */
        }  /* end of loop on spin dilution */
      }  /* end of loop on source timeslices */

    }  /* end of loop on smearing levels */

    /***************************************************************************
     *                                                                         *
     * all done for zero momentum inversions                                   *
     *                                                                         *
     * now we restart with the momentum-source inversions                      *
     *                                                                         *
     ***************************************************************************/

    /***************************************************************************
     *
     * loop on (sink-)momenta of 2-point function
     *
     ***************************************************************************/
    for ( int isnk_mom = 0; isnk_mom < g_sink_momentum_number; isnk_mom++ ) {

      /***************************************************************************
       * the momentum at source
       ***************************************************************************/
      int source_momentum[3] = { g_source_momentum_list[isnk_mom][0], g_source_momentum_list[isnk_mom][1], g_source_momentum_list[isnk_mom][2] };

      /***************************************************************************
       * apply the source momentum phase to all timeslices
       * of the original stochastic source
       ***************************************************************************/
      for ( int it = 0; it < T; it++ ) {
        size_t const offset =  it * _GSI(VOL3);
        spinor_field_eq_spinor_field_ti_complex_field ( stochastic_source_smeared + offset, stochastic_source + offset, (double*)(ephase[isnk_mom]), VOL3 );
      }

      /***************************************************************************
       *
       * loop on source smearing levels
       *
       *   now applied to momentum-source
       *
       ***************************************************************************/
      for ( int ismear = 0; ismear < smearing_level_number; ismear++ ) {

        /***************************************************************************
         * source smearing parameters
         ***************************************************************************/
        int const    nstep_src = ( ismear == 0 ) ? smearing_level_list[ismear].n : smearing_level_list[ismear].n - smearing_level_list[ismear-1].n;
        double const alpha_src = smearing_level_list[ismear].alpha;

        if ( g_verbose > 2 && g_cart_id == 0 ) 
          fprintf ( stdout, "# [jj_invert_contract] momentum-source smearing level %2d parameters N %3d A %6.4f\n", ismear, nstep_src, alpha_src );

        exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_source_smeared, nstep_src, alpha_src );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[jj_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(72);
        }

        if ( g_write_source ) {
          /* write smeared stochastic source at smearing level ismear */
          sprintf(filename, "%s.%.4d.px%dpy%dpz%d.%.5d.Nsrc%d_Asrc%6.4f", filename_prefix, Nconf,
              source_momentum[0], source_momentum[1], source_momentum[2], isample,
              smearing_level_list[ismear].n, smearing_level_list[ismear].alpha );

          exitstatus = write_propagator( stochastic_source_smeared, filename, 0, g_propagator_precision);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[jj_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }  /* end of if write smeared momentum source */

        /***************************************************************************
         * loop on source timeslices
         ***************************************************************************/
        for( int itsrc = 0; itsrc < g_source_location_number; itsrc++ ) {

          /***************************************************************************
           * local source timeslice and source process ids
           ***************************************************************************/

          int source_timeslice = -1;
          int source_proc_id   = -1;
          int gts              = ( g_source_coords_list[itsrc][0] +  T_global ) %  T_global;

          exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[jj_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(123);
          }

          /***************************************************************************
           * invert for oet stochastic timeslice propagator with source momentum
           *   for up-type flavor only
           ***************************************************************************/
          for( int is = 0; is < spin_dilution;  is++) {
          for( int ic = 0; ic < color_dilution; ic++) {

            int const isc = is*color_dilution + ic;

            /***************************************************************************
             * init source to zero
             * if have the source timeslice, then extract spin-color component is,ic
             * for local source_timeslice
             ***************************************************************************/
            memset ( spinor_work[2], 0, sizeof_spinor_field );
            if ( source_proc_id == g_cart_id ) {
              get_spin_color_oet_comp ( spinor_work[2], stochastic_source_smeared, is, ic, spin_dilution, color_dilution , source_timeslice );
            }

            /* init solution field to zero */
            memset ( spinor_work[1], 0, sizeof_spinor_field );

            /* copy source again, to preserve original source */
            memcpy ( spinor_work[0], spinor_work[2], sizeof_spinor_field );

            /* call solver via tmLQCD */
            exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], op_id_up );
            if(exitstatus < 0) {
              fprintf(stderr, "[jj_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(44);
            }

            if ( check_propagator_residual ) {
              check_residual_clover ( &(spinor_work[1]), &(spinor_work[2]), gauge_field_with_phase, mzz[op_id_up], mzzinv[op_id_up], 1 );
            }

            memcpy( stochastic_propagator_mom[isc], spinor_work[1], sizeof_spinor_field);

            if ( g_write_propagator ) {
              sprintf(filename, "%s.%.4d.t%d.px%dpy%dpz%d.d%d.s%.5d.Nsrc%d_Asrc%6.4f.inverted", filename_prefix, Nconf, gts,
                  source_momentum[0], source_momentum[1], source_momentum[2], isc, isample,
                  smearing_level_list[ismear].n, smearing_level_list[ismear].alpha );
              exitstatus = write_propagator( stochastic_propagator_mom[isc], filename, 0, g_propagator_precision);
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[jj_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(2);
              }
            }
          }  /* end of loop on color dilution */
          }  /* end of loop on spin dilution */

          /*****************************************************************
           * loop on levels of sink smearing
           *
           *   applied to both zero-momentum and momentum timeslice
           *   propagator
           *
           *****************************************************************/
          for ( int ksmear = 0; ksmear < smearing_level_number; ksmear++ ) {

            /***************************************************************************
             * sink smearing parameters
             *   nstep_snk is difference current - previous nstep
             *   WE TAKE IT, THAT alpha_snk IS THE SAME FOR ALL SMEARING LEVELS
             *   AND THE SAME AS alpha for source smearing
             *
             *   For ksmear == 1, don't do anything, just use stochastic_propagator_... as is
             ***************************************************************************/
            int const nstep_snk = ( ksmear == 0 ) ? smearing_level_list[ksmear].n : smearing_level_list[ksmear].n - smearing_level_list[ksmear-1].n;
            double const alpha_snk = smearing_level_list[ksmear].alpha;

            if ( g_verbose > 2 && io_proc == 2 ) fprintf ( stdout, "# [jj_invert_contract] sink smearing level %2d parameters N %3d A %6.4f\n", ksmear, nstep_snk, alpha_snk );

            for ( int i = 0; i < spin_color_dilution; i++ ) {
                
              exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_propagator_mom[i], nstep_snk, alpha_snk  );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[jj_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(72);
              }

              for ( int iflavor = 0; iflavor < nflavor; iflavor++ ) {
                 exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_propagator_zero[ismear][itsrc][iflavor][i], nstep_snk, alpha_snk  );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[jj_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(72);
                }
              }
            }  /* end of loop on spin-color dilution components */

            /*****************************************************************
             *
             * contractions for 2-point functons
             *
             *****************************************************************/

            int sink_momentum[3] = {
                g_sink_momentum_list[isnk_mom][0],
                g_sink_momentum_list[isnk_mom][1],
                g_sink_momentum_list[isnk_mom][2] };

            for ( int isrc_gamma = 0; isrc_gamma < gamma_vertex_number; isrc_gamma++ ) {
            for ( int isnk_gamma = 0; isnk_gamma < gamma_vertex_number; isnk_gamma++ ) {
        
              for ( int iflavor = 0; iflavor < nflavor; iflavor++ ) {

                double * contr_p = init_1level_dtable ( 2*T );
                if ( contr_p == NULL ) {
                  fprintf(stderr, "[jj_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
                  EXIT(47);
                }

                contract_twopoint_snk_momentum (
                    contr_p,
                    gamma_vertex_id[iflavor][isrc_gamma],
                    gamma_vertex_id[iflavor][isnk_gamma], 
                    stochastic_propagator_zero[ismear][itsrc][iflavor],
                    stochastic_propagator_mom,
                    spin_dilution, color_dilution, sink_momentum, 1);

                sprintf ( data_tag, "/%s/t%d/s%d/Nsnk%d_Asnk%6.4f/Nsrc%d_Asrc%6.4f/Gf_%s/Gi_%s/PX%d_PY%d_PZ%d", flavor_combination[iflavor], gts, isample,
                    smearing_level_list[ksmear].n, smearing_level_list[ksmear].alpha,
                    smearing_level_list[ismear].n, smearing_level_list[ismear].alpha,
                    gamma_vertex_name[isnk_gamma], gamma_vertex_name[isrc_gamma],
                    sink_momentum[0], sink_momentum[1], sink_momentum[2] );

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
                exitstatus = contract_write_to_aff_file ( &contr_p, affw, data_tag, &sink_momentum, 1, io_proc );
#elif ( defined HAVE_HDF5 )          
                exitstatus = contract_write_to_h5_file ( &contr_p, output_filename, data_tag, &sink_momentum, 1, io_proc );
#endif
                if(exitstatus != 0) {
                  fprintf(stderr, "[jj_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  return(3);
                }
 
                /* deallocate the contraction field */
                fini_1level_dtable ( &contr_p );

              }  /* end of loop on iflavor */

            }  /* end of loop on gamma at sink */
            }  /* end of loop on gammas at source */

          }  /* end of loop on sink smearing levels */

        }  /* end of loop on source timeslices */

      }  /* end of loop on source smearing levels */

    }  /* end of loop on sink momenta */

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[jj_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  }  /* end of loop on oet samples */

  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_2level_dtable ( &stochastic_propagator_mom );
  fini_5level_dtable ( &stochastic_propagator_zero );
  fini_1level_dtable ( &stochastic_source );
  fini_1level_dtable ( &stochastic_source_smeared );
  fini_2level_dtable ( &spinor_work );

  fini_2level_ztable ( &ephase );

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
    fprintf(stdout, "# [jj_invert_contract] %s# [jj_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [jj_invert_contract] %s# [jj_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
