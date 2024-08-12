/****************************************************
 * vv2pt_simple_invert_contract
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

using namespace cvc;

void usage() {
  fprintf(stdout, "Code for high-statistics V-V 2pt\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "kaon";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL, *gauge_field_smeared = NULL;
  char output_filename[400];
  int const spin_dilution  = 1;
  int const color_dilution = 1;

  struct timeval ta, tb;

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
  /* fprintf(stdout, "# [vv2pt_simple_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [vv2pt_simple_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [vv2pt_simple_invert_contract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  size_t const sizeof_spinor_field           = _GSI(VOLUME) * sizeof(double);

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
    if(g_cart_id==0) fprintf(stdout, "# [vv2pt_simple_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [vv2pt_simple_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[vv2pt_simple_invert_contract] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[vv2pt_simple_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv for light quark
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [vv2pt_simple_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  size_t nelem = _GSI( VOLUME+RAND );
  double ** spinor_work  = init_2level_dtable ( 2, nelem );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * allocate memory for spinor fields
   * WITHOUT halo
   ***************************************************************************/
  nelem = _GSI( VOLUME );

  int const spin_color_dilution = spin_dilution * color_dilution;

  /***************************************************************************/

  double *** stochastic_propagator_mom = init_3level_dtable ( 2, spin_color_dilution, nelem );
  if ( stochastic_propagator_mom == NULL ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  /***************************************************************************/

  double *** stochastic_propagator_zero = init_3level_dtable ( 2, spin_color_dilution, nelem );
  if ( stochastic_propagator_zero == NULL ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  /***************************************************************************/

  double ** stochastic_source_list = init_2level_dtable ( spin_color_dilution, nelem );
  if ( stochastic_source_list == NULL ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  int * rng_state = NULL;
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   * output filename
   ***************************************************************************/
  sprintf ( output_filename, "%s.%d.h5", g_outfile_prefix, Nconf );

  if(io_proc == 2 && g_verbose > 1 ) { 
    fprintf(stdout, "# [vv2pt_simple_invert_contract] writing data to file %s\n", output_filename);
  }

  exitstatus = write_h5_contraction ( g_sink_gamma_id_list, NULL, output_filename, "snk_gamma", "int", 1, &g_sink_gamma_id_number );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(123);
  }

  exitstatus = write_h5_contraction ( g_source_gamma_id_list, NULL, output_filename, "src_gamma", "int", 1, &g_source_gamma_id_number );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(123);
  }

  int mom_cdim[2] = {g_source_momentum_number, 3};
  exitstatus = write_h5_contraction ( g_source_momentum_list[0], NULL, output_filename, "src_mom", "int", 2, mom_cdim );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(123);
  }


  /***************************************************************************
   * zero momentum propagators
   ***************************************************************************/
  for ( int isample = 0; isample < T_global; isample++ )
  {

    int const gts = isample;

    int source_timeslice = -1;
    int source_proc_id   = -1;

    exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[vv2pt_simple_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }
      
    for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ )
    {
      int pi[3] = {
        g_source_momentum_list[ipi][0],
        g_source_momentum_list[ipi][1],
        g_source_momentum_list[ipi][2] 
      };
      int pf[3] = { -pi[0], -pi[1], -pi[2] };

      if ( ipi == 0 && ! ( pi[0] == 0 && pi[1] == 0 && pi[2] == 0 ) )
      {
        if ( io_proc == 2 ) fprintf (stderr, "[vv2pt_simple_invert_contract] Error, first source momentum must be zero    %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }

      /***************************************************************************
       * stochastic oet timeslice sources
       ***************************************************************************/
      exitstatus = init_timeslice_source_oet(stochastic_source_list, gts, pi, spin_dilution, color_dilution, ipi == 0 ) ;
      if( exitstatus != 0 ) 
      {
        fprintf(stderr, "[vv2pt_simple_invert_contract] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(64);
      }

      for ( int iflavor = 0; iflavor < 2; iflavor++ )
      {
        for ( int isc = 0; isc < spin_color_dilution; isc++ )
        {
          memset ( spinor_work[1], 0, sizeof_spinor_field );
      
          memcpy ( spinor_work[0], stochastic_source_list[isc], sizeof_spinor_field );

          exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], 0 );
          if(exitstatus < 0) 
          {
            fprintf(stderr, "[vv2pt_simple_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(44);
          }

          if ( check_propagator_residual ) 
          {
            check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[0], mzzinv[0], 1 );
          }
    
          memcpy( stochastic_propagator_mom[iflavor][isc], spinor_work[1], sizeof_spinor_field );
          if ( ipi == 0 )
          {
            /* zero momentum case, copy extra, for each flavor */
            memcpy( stochastic_propagator_zero[iflavor][isc], spinor_work[1], sizeof_spinor_field );
          }

        }  /* end of loop on spin-color components */

      }  /* end of loop on flavors */

      /* allocate contraction fields in position and momentum space */
      double **** contr = init_4level_dtable ( 2, g_source_gamma_id_number, g_sink_gamma_id_number, 2*T );
      if ( contr == NULL ) {
        fprintf(stderr, "[vv2pt_simple_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(3);
      }
    
      /***************************************************************************
       * contractions for 2-point functions
       *
       * loop on Gamma strcutures
       ***************************************************************************/
      for ( int igi = 0; igi < g_source_gamma_id_number; igi++ )
      {
        int source_gamma = g_source_gamma_id_list[igi];

        for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ )
        {
          int sink_gamma = g_sink_gamma_id_list[igf];

          gettimeofday ( &ta, (struct timezone *)NULL );
          /***************************************************************************
           * gig5 D^+ g5gf U = gi U gf U
           ***************************************************************************/
          contract_twopoint_snk_momentum ( contr[0][igi][igf], source_gamma, sink_gamma, stochastic_propagator_zero[1], stochastic_propagator_mom[0],
              spin_dilution, color_dilution, pf, 1 );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "vv2pt_simple_invert_contract", "contract_twopoint_snk_momentum", io_proc==2 );

          gettimeofday ( &ta, (struct timezone *)NULL );
          /***************************************************************************
           * gig5 U^+ g5gf D = gi D g5 D
           ***************************************************************************/
          contract_twopoint_snk_momentum ( contr[1][igi][igf], source_gamma, sink_gamma, stochastic_propagator_zero[0], stochastic_propagator_mom[1],
              spin_dilution, color_dilution, pf, 1 );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "vv2pt_simple_invert_contract", "contract_twopoint_snk_momentum", io_proc==2 );

        }
      }
      
      if ( io_proc > 0 ) 
      {
        double **** buffer = init_4level_dtable ( T, 2, g_source_gamma_id_number, 2*g_sink_gamma_id_number );
        for ( int ifl = 0; ifl < 2; ifl++ )
        {
          for ( int igi = 0; igi < g_source_gamma_id_number; igi++ )
          {
            for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ )
            {
#pragma omp parallel for
              for ( int it = 0; it < T; it++ )
              {
                buffer[it][ifl][igi][2*igf  ] = contr[ifl][igi][igf][2*it  ];
                buffer[it][ifl][igi][2*igf+1] = contr[ifl][igi][igf][2*it+1];
              }
            }
          }
        }
        
        double * gcontr = ( io_proc == 2 ) ? init_1level_dtable ( T_global * 4 * g_source_gamma_id_number * g_sink_gamma_id_number ) :  init_1level_dtable ( 2 );
#ifdef HAVE_MPI
        exitstatus = MPI_Gather ( buffer[0][0][0], T * 4 * g_source_gamma_id_number * g_sink_gamma_id_number, MPI_DOUBLE,
               gcontr, T * 4 * g_source_gamma_id_number * g_sink_gamma_id_number, MPI_DOUBLE, 0,
               g_tr_comm);
        if ( exitstatus != MPI_SUCCESS )
        {
          fprintf (stderr, "[vv2pt_simple_invert_contract] Error from MPI_Gather    %s %d\n", __FILE__, __LINE__);
          EXIT(12);
        }
#else
        memcpy ( gcontr, buffer[0][0][0], T * 4 * g_source_gamma_id_number * g_sink_gamma_id_number * sizeof(double) );
#endif
        if ( io_proc == 2 )
        {
          sprintf ( data_tag, "/t%d/pfx%dpfy%dpfz%d", gts, pf[0], pf[1], pf[2]);

          int const ncdim = 4;
          int const cdim[4] = { T_global, 2, g_source_gamma_id_number, g_sink_gamma_id_number};

          gettimeofday ( &ta, (struct timezone *)NULL );

          exitstatus = write_h5_contraction ( gcontr, NULL, output_filename, data_tag, "double", ncdim, cdim );
          if(exitstatus != 0) {
            fprintf(stderr, "[vv2pt_simple_invert_contract] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(3);
          }
      
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "vv2pt_simple_invert_contract", "write_h5_contraction", io_proc==2 );

        }
      
        fini_4level_dtable ( &buffer );
        fini_1level_dtable ( &gcontr );

      }
     
      fini_4level_dtable ( &contr );


    }  /* end of loop on source momenta */
      
    exitstatus = init_timeslice_source_oet ( NULL, -1, NULL, 0, 0, -2 );

  }  /* end of loop on samples */

  fini_rng_state ( &rng_state );

  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_3level_dtable ( &stochastic_propagator_zero );
  fini_3level_dtable ( &stochastic_propagator_mom );
  fini_2level_dtable ( &stochastic_source_list );
  fini_2level_dtable ( &spinor_work );

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  free( gauge_field_with_phase );
  free( gauge_field_smeared );

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
    fprintf(stdout, "# [vv2pt_simple_invert_contract] %s# [vv2pt_simple_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [vv2pt_simple_invert_contract] %s# [vv2pt_simple_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
