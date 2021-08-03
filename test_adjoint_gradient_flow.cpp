/***************************************************************************
 *
 * test_adjoint_gradient_flow
 *
 ***************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
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
#include "cvc_timer.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "gauge_io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "contractions_io.h"
#include "contract_factorized.h"
#include "contract_diagrams.h"
#include "gamma.h"
#include "clover.h"
#include "gradient_flow.h"
#include "gluon_operators.h"
#include "contract_cvc_tensor.h"
#include "scalar_products.h"

using namespace cvc;

/***************************************************************************
 *
 ***************************************************************************/
static inline void _fv_cvc_eq_convert_fv_szin ( double * const r , double * const s ) {
  double _spinor1[24];
  _fv_eq_gamma_ti_fv ( _spinor1, 2, s );
  _fv_eq_fv ( r, _spinor1 );
}  /* end of _fv_cvc_eq_convert_fv_szin */

/***************************************************************************
 *
 ***************************************************************************/
static inline void spinor_field_cvc_eq_convert_spinor_field_szin (double * const r , double * const s , unsigned int const N ) {
#pragma omp parallel for
  for ( unsigned int i = 0; i < N; i++ ) {
    double * const _r = r + _GSI(i);
    double * const _s = s + _GSI(i);
    _fv_cvc_eq_convert_fv_szin ( _r , _s );
  }
}  /* end of spinor_field_cvc_eq_convert_spinor_field_szin */


/***************************************************************************
 * helper message
 ***************************************************************************/
void usage() {
  fprintf(stdout, "Code for FHT-type nucleon-nucleon 2-pt function inversion and contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  fprintf(stdout, "          -h                  : this message\n");
  EXIT(0);
}

/***************************************************************************
 *
 * MAIN PROGRAM
 *
 ***************************************************************************/
int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "test_gf";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[400];
  struct timeval ta, tb, start_time, end_time;
  int check_propagator_residual = 0;
  unsigned int gf_niter = 10;
  unsigned int gf_ns    = 0;
  unsigned int gf_m     = 0;
  double gf_dt          = 0.01;

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?f:e:n:s:m:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'n':
      gf_niter = atoi ( optarg );
      break;
    case 'e':
      gf_dt = atof ( optarg );
      break;
    case 's':
      gf_ns = atoi ( optarg );
      break;
    case 'm':
      gf_m = atoi ( optarg );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  gettimeofday ( &start_time, (struct timezone *)NULL );


  /***************************************************************************
   * read input and set the default values
   ***************************************************************************/
  if(filename_set==0) strcpy(filename, "twopt.input");
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_adjoint_gradient_flow] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize tmLQCD solvers
   ***************************************************************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
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
  mpi_init_xchange_contraction(2);

  /***************************************************************************
   * report git version
   * make sure the version running here has been commited before program call
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_adjoint_gradient_flow] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_adjoint_gradient_flow] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_adjoint_gradient_flow] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_adjoint_gradient_flow] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[test_adjoint_gradient_flow] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  
  /***************************************************************************
   * initialize lattice geometry
   *
   * allocate and fill geometry arrays
   ***************************************************************************/
  geometry();


  /***************************************************************************
   * set up some mpi exchangers for
   * (1) even-odd decomposed spinor field
   * (2) even-odd decomposed propagator field
   ***************************************************************************/
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***************************************************************************
   * set up the gauge field
   *
   *   either read it from file or get it from tmLQCD interface
   *
   *   lime format is used
   ***************************************************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [test_adjoint_gradient_flow] reading gauge field from file %s\n", filename);

    exitstatus = read_lime_gauge_field_doubleprec(filename);

  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_adjoint_gradient_flow] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_adjoint_gradient_flow] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[test_adjoint_gradient_flow] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  double *gauge_field_with_phase = NULL;
  /* exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up ); */
  exitstatus = gauge_field_eq_gauge_field_ti_bcfactor ( &gauge_field_with_phase, g_gauge_field, -1. );
  if(exitstatus != 0) {
    fprintf(stderr, "[test_adjoint_gradient_flow] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize the clover term, 
   * lmzz and lmzzinv
   *
   *   mzz = space-time diagonal part of the Dirac matrix
   *   l   = light quark mass
   ***************************************************************************/
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  exitstatus = init_clover ( &lmzz, &lmzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_adjoint_gradient_flow] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[test_adjoint_gradient_flow] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [test_adjoint_gradient_flow] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  size_t const sizeof_gauge_field = 72 * ( VOLUME ) * sizeof( double );
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_adjoint_gradient_flow] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

#if defined HAVE_LHPC_AFF
  if ( io_proc == 2 ) {
    /***************************************************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     ***************************************************************************/
    sprintf(filename, "%s.c%d.aff", outfile_prefix, Nconf );
    fprintf(stdout, "# [test_adjoint_gradient_flow] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    const char * aff_status_str = aff_writer_errstr ( affw );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }
#endif

  /***************************************************************************
   *
   * 
   *
   ***************************************************************************/

  double ** spinor_field = init_2level_dtable ( 12, _GSI( VOLUME ) );
  if ( spinor_field == NULL ) {
    fprintf(stderr, "[test_adjoint_gradient_flow] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(44);
  }

  /***************************************************************************
   * prepare an up-type propagator from oet timeslice source
   ***************************************************************************/
  if ( g_read_propagator ) {

    int source_proc_id = -1;
    int sx[4];
    exitstatus = get_point_source_info ( g_source_coords_list[0], sx, &source_proc_id );


    double * buffer = init_1level_dtable ( 288 * VOLUME );
    if ( buffer == NULL ) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(44);
    }

    strcpy ( filename, filename_prefix2 );

    exitstatus = read_lime_propagator ( buffer, filename, g_propagator_position);
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from  read_lime_propagator, status %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(44);
    }

    for ( int isc = 0; isc < 12; isc++ ) {

      for ( unsigned int ix = 0; ix < VOLUME; ix ++ ) {

        for ( int ir = 0; ir < 12; ir++ ) {

          // int const is = 3 * ( 3 * ( 4 * (isc/3) + (ir/3) ) + (isc%3) ) + (ir%3);
          int const is = 3 * ( 3 * ( 4 * (ir/3) + (isc/3) ) + (ir%3) ) + (isc%3);

          unsigned int iy = 288 * ix + 2*is;

          spinor_field[isc][_GSI(ix) + 2 * ir     ] = buffer[iy  ];
          spinor_field[isc][_GSI(ix) + 2 * ir + 1 ] = buffer[iy+1];
        }
      }
    }

    fini_1level_dtable ( &buffer );

    if ( check_propagator_residual ) {
      double ** spinor_work  = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
      if ( spinor_work == NULL ) {
        fprintf(stderr, "[test_adjoint_gradient_flow] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(44);
      }

      for( int i = 0; i < 12; i++ ) {
        memcpy ( spinor_work[1], spinor_field[i], sizeof_spinor_field);
        memset ( spinor_work[0], 0, sizeof_spinor_field );
        if ( g_cart_id == source_proc_id ) {
          spinor_work[0][ _GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]]) + 2*i ] = 1.;
        }

        spinor_field_cvc_eq_convert_spinor_field_szin ( spinor_work[0], spinor_work[0] , VOLUME );
        spinor_field_cvc_eq_convert_spinor_field_szin ( spinor_work[1], spinor_work[1] , VOLUME );


        check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, lmzz[0], 1 );
      }

      fini_2level_dtable ( &spinor_work );

    }

  } else {

#if 0
    int source_timeslice = -1;
    int source_proc_id   = -1;
    int gts              = ( g_source_coords_list[0][0] +  T_global ) %  T_global;

    exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    double ** spinor_work  = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
    if ( spinor_work == NULL ) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(44);
    }

    /* s0 <- timeslice source, no spin or color dilution */
    exitstatus = init_timeslice_source_oet ( &(spinor_work[0]), gts, NULL, 1, 1, 1 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from init_timeslice_source_oet status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /* init s1 */
    memset ( spinor_work[1], 0, sizeof_spinor_field );

    /* s1 <- D_up^-1 s0 */
    exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], 0 );
    if(exitstatus < 0) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(44);
    }

    /* || s0 - D s1 || */
    if ( check_propagator_residual ) {
      check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, lmzz[0], 1 );
    }  

    memcpy ( spinor_field[0], spinor_work[1], sizeof_spinor_field );

    if ( g_write_propagator ) {
      sprintf ( filename, "%s.c%d.t%d", filename_prefix2, Nconf, gts );
      if ( ( exitstatus = write_propagator( spinor_field[0], filename, 0, g_propagator_precision) ) != 0 ) {
        fprintf(stderr, "[test_adjoint_gradient_flow] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }
    }

    fini_2level_dtable ( &spinor_work );
#endif

#if 0
    sprintf ( filename, "%s.t%6.4f.n%u", filename_prefix2 , gf_m * gf_dt, gf_m );
    exitstatus = read_lime_spinor ( spinor_field[0], filename, g_propagator_position);
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    sprintf ( filename, "%s.t%6.4f.n%u", filename_prefix2 , 0 * gf_dt, 0 );
    exitstatus = read_lime_spinor ( spinor_field[1], filename, g_propagator_position);
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }
#endif

    /* eta */
    random_spinor_field ( spinor_field[0], VOLUME );

    random_spinor_field ( spinor_field[1], VOLUME );

    double * gfaux = init_1level_dtable ( 72 * VOLUME );
    memcpy ( gfaux, g_gauge_field, sizeof_gauge_field );
    memcpy ( spinor_field[2], spinor_field[1], sizeof_spinor_field );
    flow_fwd_gauge_spinor_field ( gfaux, spinor_field[2], gf_m, gf_dt, 1, 1 );
    fini_1level_dtable ( &gfaux );

    flow_fwd_gauge_spinor_field ( NULL, NULL, 0, 0., 0, 0 );

  }  /* end of if read_propagator else */

#if 0
  complex csp = {0.,0.};
 
  spinor_scalar_product_co ( &csp, spinor_field[0], spinor_field[2], VOLUME );
  if ( io_proc == 2 ) fprintf ( stdout, "# [test_adjoint_gradient_flow] t %6.4f m %3u csp %25.16e %25.16e %s %d\n",  gf_m * gf_dt, gf_m,
      csp.re, csp.im , __FILE__, __LINE__ );
#endif

  char data_tag[100];


  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * prepare for GF on gauge field
   ***************************************************************************/
  double ** gauge_field_smeared = init_2level_dtable ( gf_ns+1, 72 * VOLUME );

  unsigned int gf_nb = (unsigned int) ceil ( pow ( (double)gf_m , 1./ ( (double)gf_ns + 1. ) ) );

  if ( g_verbose > 2 ) fprintf ( stdout, "# [test_adjoint_gradient_flow] gf_m = %3d gf_ns = %d  gf_nb = %3d %s %d\n", gf_m, gf_ns, gf_nb , __FILE__, __LINE__ );

  /***************************************************************************
   ***************************************************************************
   **
   ** GF^+ application
   **
   ***************************************************************************
   ***************************************************************************/
  for ( int isc = 0; isc < 12; isc++ ) {

    memcpy ( gauge_field_smeared[gf_ns], g_gauge_field, sizeof_gauge_field );

    gettimeofday ( &ta, (struct timezone *)NULL );

    flow_adjoint_gauge_spinor_field ( gauge_field_smeared, spinor_field[isc], gf_dt, gf_m, gf_nb, gf_ns );

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "test_adjoint_gradient_flow", "flow_adjoint_gauge_spinor_field", g_cart_id == 0 );

    sprintf ( filename, "%s.t%6.4f.m%u.nb%u.ns%u.sc%d", filename_prefix3, gf_dt*gf_m, gf_m, gf_nb, gf_ns, isc );
    if ( ( exitstatus = write_propagator( spinor_field[isc], filename, 0, g_propagator_precision) ) != 0 ) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }
  
    flow_adjoint_step_gauge_spinor_field ( NULL, NULL, 0, 0., 0, 0 );
  }  /* end of loop on isc */
#if 0
  spinor_scalar_product_co ( &csp, spinor_field[0], spinor_field[1], VOLUME );
  if ( io_proc == 2 ) fprintf ( stdout, "# [test_adjoint_gradient_flow] t %6.4f m %3u csp %25.16e %25.16e %s %d\n",  0 * gf_dt, 0,
      csp.re, csp.im , __FILE__, __LINE__ );
#endif
  /***************************************************************************/
  /***************************************************************************/

  exitstatus = init_timeslice_source_oet ( NULL, -1, NULL, 0, 0, -2 );

  fini_2level_dtable ( &spinor_field );
  fini_2level_dtable ( &gauge_field_smeared );



#ifdef HAVE_LHPC_AFF
  /***************************************************************************
   * I/O process id 2 closes its AFF writer
   ***************************************************************************/
  if(io_proc == 2) {
    const char * aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_adjoint_gradient_flow] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  if ( g_gauge_field != NULL ) free(g_gauge_field);
#endif
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( );

  /* free lattice geometry arrays */
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

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "test_adjoint_gradient_flow", "runtime", g_cart_id == 0 );

  return(0);

}
