/***************************************************************************
 *
 * zchi
 *
 ***************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
/* #include <complex.h> */
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

#ifdef _GFLOW_QUDA
#warning "including quda header file quda.h directly "
#include "quda.h"
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
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "gamma.h"
#include "clover.h"
#include "gradient_flow.h"
#include "gluon_operators.h"
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
 *
 ***************************************************************************/
int read_propagator ( double ** const spinor_field , char * const filename, int const check_propagator_residual, int const source_proc_id , double * const gf,  double ** const mzz, double ** const mzzinv, int const sx[4] ) {
  
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  int exitstatus;

  double * buffer = init_1level_dtable ( 288 * VOLUME );
  if ( buffer == NULL ) {
    fprintf(stderr, "[read_propagator] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  exitstatus = read_lime_propagator ( buffer, filename, g_propagator_position);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[read_propagator] Error from  read_lime_propagator, status %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(2);
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
      fprintf(stderr, "[read_propagator] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
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


      check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gf, mzz, mzzinv, 1 );

    }

    fini_2level_dtable ( &spinor_work );

  }

}  /* end of read_propagator */

/***************************************************************************/
/***************************************************************************/

/***************************************************************************
 * helper message
 ***************************************************************************/
void usage() {
  fprintf(stdout, "Code for Z_chi inversion and contractions\n");
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
  
  const char outfile_prefix[] = "zchi";

  char flavor_tag[2][4] = { "up", "dn" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[400];
  struct timeval ta, tb, start_time, end_time;
  int check_propagator_residual = 0;
  unsigned int gf_niter = 10;
  double gf_dt = 0.01;


#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?f:e:n:")) != -1) {
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

  fprintf(stdout, "# [zchi] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [zchi] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [zchi] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [zchi] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[zchi] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[zchi] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [zchi] reading gauge field from file %s\n", filename);

    exitstatus = read_lime_gauge_field_doubleprec(filename);

  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [zchi] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[zchi] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[zchi] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  double *gauge_field_with_phase = NULL;
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  /* exitstatus = gauge_field_eq_gauge_field_ti_bcfactor ( &gauge_field_with_phase, g_gauge_field, -1. ); */
  if(exitstatus != 0) {
    fprintf(stderr, "[zchi] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
  exitstatus = init_clover ( &g_clover, &lmzz, &lmzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[zchi] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[zchi] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [zchi] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_gauge_field = 72 * ( VOLUME ) * sizeof( double );
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[zchi] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

#if defined HAVE_HDF5
  if ( io_proc == 2 ) {
    /***************************************************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     ***************************************************************************/
    sprintf(filename, "%s.c%d.h5", outfile_prefix, Nconf );
    fprintf(stdout, "# [zchi] writing data to file %s\n", filename);
  }
#endif

  /***************************************************************************
   * fermion fields
   ***************************************************************************/

  double * stochastic_source = init_1level_dtable ( _GSI( VOLUME ) );

  double ** stochastic_propagator = init_2level_dtable ( 2, _GSI( ( VOLUME ) ) );

  if ( stochastic_source == NULL || stochastic_propagator == NULL )
  {
    fprintf(stderr, "[zchi] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(44);
  }

  /***************************************************************************
   * loop on samples
   ***************************************************************************/
  for ( int isample = 0; isample < g_nsample; isample++ )
  {
    /***************************************************************************
     * prepare a volume source
     ***************************************************************************/
#if 0
    prepare_volume_source ( stochastic_source, VOLUME );

    sprintf ( filename, "stochastic_source.%d", isample );
    if ( ( exitstatus = write_propagator( stochastic_source, filename, 0, g_propagator_precision) ) != 0 ) {
      fprintf(stderr, "[zchi] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }
#endif
    sprintf ( filename, "stochastic_source.%d", isample );
    if ( ( exitstatus = read_lime_spinor( stochastic_source, filename, 0 ) ) != 0 ) {
      fprintf(stderr, "[zchi] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }



    double ** spinor_work  = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
    if ( spinor_work == NULL ) {
      fprintf(stderr, "[zchi] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(44);
    }

    for ( int iflavor = 0; iflavor < 2; iflavor++ )
    {
      memcpy ( spinor_work[0], stochastic_source, sizeof_spinor_field ); 
      memset ( spinor_work[1], 0, sizeof_spinor_field );

      /* s1 <- D_flavor^-1 s0 */
      exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
      if(exitstatus < 0) {
        fprintf(stderr, "[zchi] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(44);
      }

      /* || s0 - D s1 || */
      if ( check_propagator_residual ) {
        check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, lmzz[iflavor], lmzzinv[iflavor], 1 );
      }  

      memcpy ( stochastic_propagator[iflavor], spinor_work[1], sizeof_spinor_field );
    }

    fini_2level_dtable ( &spinor_work );

    /***************************************************************************
     * contractions
     ***************************************************************************/

    for ( int iflavor = 0; iflavor < 2; iflavor++ )
    {
      double ** Dspinor_field = init_2level_dtable ( 2, _GSI( VOLUME ) );

      for ( int mu = 0; mu < 4; mu++ )
      {

        spinor_field_eq_cov_displ_spinor_field ( Dspinor_field[0], stochastic_propagator[iflavor], mu, 0, gauge_field_with_phase );
      
        spinor_field_eq_cov_displ_spinor_field ( Dspinor_field[1], stochastic_propagator[iflavor], mu, 1, gauge_field_with_phase );

        spinor_field_eq_spinor_field_mi_spinor_field ( Dspinor_field[0], Dspinor_field[0],  Dspinor_field[1], VOLUME );

        spinor_field_eq_gamma_ti_spinor_field ( Dspinor_field[1], mu, Dspinor_field[0], VOLUME );

        complex w = { 0., 0. };

        spinor_scalar_product_co ( &w, stochastic_source, Dspinor_field[1], VOLUME );

        if ( io_proc == 2 )
        {
          int const ncdim = 1;
          int const cdim[1] = {2};
          char tag[100];
          sprintf(filename, "%s.c%d.h5", outfile_prefix, Nconf );
          sprintf ( tag, "/s%d/%s/mu%d", isample, flavor_tag[iflavor], mu );

          write_h5_contraction ( &w, NULL, filename, tag, "double", ncdim, cdim );
        }

      }

      fini_2level_dtable ( &Dspinor_field );
  
    }  /* end of loop on flavor */

  }  /* end of loop on samples */


  /***************************************************************************
   * deallocate (static) fields
   ***************************************************************************/

  fini_2level_dtable ( &stochastic_propagator );
  fini_1level_dtable ( &stochastic_source );

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
  fini_clover ( &lmzz, &lmzzinv );

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
  show_time ( &start_time, &end_time, "zchi", "runtime", g_cart_id == 0 );

  return(0);

}
