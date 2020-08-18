/****************************************************
 * loop_invert_contract
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


#define _FIRST_DERIV_CONTRACTION
#undef _SECOND_DERIV_CONTRACTION

int main(int argc, char **argv) {
  
  const char fbwd_str[2][4] =  { "fwd", "bwd" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  size_t sizeof_spinor_field;
  char filename[300];

  struct timeval ta, tb, start_time, end_time;

  double **mzz[2]    = { NULL, NULL }, **mzzinv[2]    = { NULL, NULL };
  double **DW_mzz[2] = { NULL, NULL }, **DW_mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  int op_id_up = -1;
  int op_id_dn = -1;
  char output_filename[400];
  int * rng_state = NULL;
  int restart = 0;

  char data_tag[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "rch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'r':
      restart = 1;
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
    op_id_dn = 1;
  } else if ( g_fermion_type == _WILSON_FERMION ) {
    op_id_up = 0;
    op_id_dn = 0;
   } else {
     fprintf(stderr, "[loop_invert_contract] Error, unrecognized fermion type %d %s %d\n", g_fermion_type, __FILE__, __LINE__ );
     EXIT(1);
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
  nelem = 288 * VOLUME;
  double * stochastic_propagator = init_1level_dtable ( nelem );
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
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[loop_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

#if ( defined HAVE_HDF5 )
  sprintf ( output_filename, "%s.%.4d.h5", g_outfile_prefix, Nconf );
#endif
  if(io_proc == 2 && g_verbose > 1 ) { 
    fprintf(stdout, "# [loop_invert_contract] writing data to file %s\n", output_filename);
  }

#if ( defined HAVE_HDF5 )
  /***************************************************************************
   * write momentum configuration
   ***************************************************************************/
  if ( ( io_proc == 2 ) && !restart ) {

    for ( int i = 0; i < g_sink_momentum_number; i++ ) {
      g_sink_momentum_list[i][0] *= -1;
      g_sink_momentum_list[i][1] *= -1;
      g_sink_momentum_list[i][2] *= -1;
    }

    int const dims[2] = { g_sink_momentum_number, 3 };
    exitstatus = write_h5_contraction ( g_sink_momentum_list[0], NULL, output_filename, "/Momenta_list_xyz", 3*g_sink_momentum_number , "int", 2, dims );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[loop_invert_contract] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
      EXIT( 54 );
    }

    for ( int i = 0; i < g_sink_momentum_number; i++ ) {
      g_sink_momentum_list[i][0] *= -1;
      g_sink_momentum_list[i][1] *= -1;
      g_sink_momentum_list[i][2] *= -1;
    }

    char message[1000];
    strcpy( message, "<dirac_gamma_basis>tmlqcd</dirac_gamma_basis>\n<noise_type>Z2xZ2</noise_type>" );
    exitstatus = write_h5_attribute ( output_filename, "Description", message );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[loop_invert_contract] Error from write_h5_attribute, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
      EXIT( 54 );
    }
    sprintf( message, "<kappa>%.8f</kappa>\n<mu>%.8f</mu>\n<Csw>%.8f</Csw>", g_kappa, g_mu, g_csw );
    exitstatus = write_h5_attribute ( output_filename, "Ensemble-info", message );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[loop_invert_contract] Error from write_h5_attribute, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
      EXIT( 54 );
    }
    sprintf( message, "<count>%d</count\n<phase_sign>%s</phase_sign>", g_sink_momentum_number, "-1" );
    exitstatus = write_h5_attribute ( output_filename, "Momenta", message );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[loop_invert_contract] Error from write_h5_attribute, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );;
      EXIT( 54 );
    }

  }
#endif

  /***************************************************************************
   * loop on stochastic oet samples
   ***************************************************************************/
  for ( int isample = 0; isample < g_nsample; isample++ )
  /* for ( int isample = g_sourceid; isample <= g_sourceid2; isample += g_sourceid_step ) */
  {

    /***************************************************************************
     * read stochastic oet source from file
     ***************************************************************************/
    if ( g_read_source ) {
      sprintf(filename, "%s.%.4d.%.5d", filename_prefix, Nconf, isample);
      /* exitstatus = read_lime_spinor( stochastic_source, filename, 0); */
      exitstatus = read_lime_propagator( stochastic_source, filename, 0);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[loop_invert_contract] Error from read_lime, status was %d\n", exitstatus);
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
     * invert for stochastic propagator
     *   up flavor
     ***************************************************************************/
    if ( ! g_read_propagator ) {
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

    } else {
      sprintf(filename, "%s.%.4d.%.5d", filename_prefix2, Nconf, isample);
 
      /* exitstatus = read_lime_spinor( stochastic_source, filename, 0); */
      exitstatus = read_lime_propagator( stochastic_propagator, filename, 0);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[loop_invert_contract] Error from read_lime, status was %d\n", exitstatus);
        EXIT(2);
      }
    }


    for ( int ir = 0; ir < 1; ir++ ) {
    for ( int ia = 0; ia < 1; ia++ ) {
      for ( int it = 0; it < T; it++ ) {
      for ( int iz = 0; iz < LZ; iz++ ) {
      for ( int iy = 0; iy < LY; iy++ ) {
      for ( int ix = 0; ix < LX; ix++ ) {
        for ( int is = 0; is < 4; is++ ) {
        for ( int ib = 0; ib < 3; ib++ ) {
          /* ir,ia; is,ib */
          /* int const isc = 3*(4*(3*ir+ia) + is) + ib; */
          
          /* ir,is; ia,ib */
          // int const isc = 3 * ( 3 * ( 4 * ir + is ) + ia ) + ib;

          /* is,ir; ib,ia */
          int const isc = 3 * ( 3 * ( 4 * is + ir ) + ib ) + ia;

          /* ia,ib; ir,is */
          // int const isc =  4*(4*(3*ia+ib)+ir ) + is;

          /* no */
          /* int const isc =  4*(4*(3*ib+ia)+is ) + ir; */
            3*(4*(3*ir+ia) + is) + ib;
        fprintf ( stdout, " %2d %2d    %3d %3d %3d %3d    %2d %2d    %3d   %25.16e %25.16e\n", 
            ir, ia, 
            ix, iy, iz, it, 
            is, ib, isc,
            stochastic_source[  288 * g_ipt[it][ix][iy][iz] + 2 * isc    ],
            stochastic_source[  288 * g_ipt[it][ix][iy][iz] + 2 * isc +1 ] );
        }}
      }}
    }}}}

#if 0
    /***************************************************************************
     *
     * CONTRACTION FOR LOCAL LOOPS USING STD ONE-END-TRICK
     *
     ***************************************************************************/

    /***************************************************************************
     * group name for contraction
     ***************************************************************************/
    sprintf ( data_tag, "/conf_%.4d/nstoch_%.4d/%s", Nconf, isample+1, "Scalar" );

    /***************************************************************************
     * loop contractions
     ***************************************************************************/
    exitstatus = contract_local_loop_stochastic ( loop, stochastic_propagator_g5, stochastic_propagator, g_sink_momentum_number, g_sink_momentum_list );
    if(exitstatus != 0) {
      fprintf(stderr, "[loop_invert_contract] Error from contract_local_loop_stochastic, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(44);
    }

    /***************************************************************************
     * factor -1 for STD-OET
     ***************************************************************************/
    complex_field_ti_eq_re ( loop[0][0], -1., T * g_sink_momentum_number * 16 );

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

    /*****************************************************************/
    /*****************************************************************/
#endif
  }  /* end of loop on oet samples */

  /***************************************************************************
   * decallocate fields
   ***************************************************************************/
  fini_1level_dtable ( &stochastic_propagator        );
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

  gettimeofday ( &end_time, (struct timezone *) NULL );

  show_time ( &start_time, &end_time, "loop_invert_contract", "runtime", g_cart_id == 0 );

  return(0);

}
