/****************************************************
 * hlbl_lm_contract
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
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

#  ifdef HAVE_KQED
#    include "KQED.h"
#  endif

#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM

// #include "cvc_complex.h"
#include "iblas.h"
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
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "clover.h"
#include "scalar_products.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#define _WITH_TIMER 1


using namespace cvc;

/***********************************************************
 * KQED kernel function pointer
 ***********************************************************/
/* void QED_kernel_L0( const double xv[4] , const double yv[4] , const struct QED_kernel_temps t , double kerv[6][4][4][4] ) ; */
typedef void (*QED_kernel_LX_ptr)( const double xv[4], const double yv[4], const struct QED_kernel_temps t, double kerv[6][4][4][4] );

/***********************************************************
 * x must be in { 0, ..., L-1 }
 * mapping as in 2006.16224, eq. 8
 ***********************************************************/
inline void site_map (int xv[4], int const x[4] )
{
  xv[0] = ( x[0] >= T_global   / 2 ) ? (x[0] - T_global )  : x[0];
  xv[1] = ( x[1] >= LX_global  / 2 ) ? (x[1] - LX_global)  : x[1];
  xv[2] = ( x[2] >= LY_global  / 2 ) ? (x[2] - LY_global)  : x[2];
  xv[3] = ( x[3] >= LZ_global  / 2 ) ? (x[3] - LZ_global)  : x[3];

  return;
}

/***********************************************************
 * as above, but set L/2 to 0 and -L/2 to 0
 ***********************************************************/
inline void site_map_zerohalf (int xv[4], int const x[4] )
{
  xv[0] = ( x[0] > T_global   / 2 ) ? x[0] - T_global   : (  ( x[0] < T_global   / 2 ) ? x[0] : 0 );
  xv[1] = ( x[1] > LX_global  / 2 ) ? x[1] - LX_global  : (  ( x[1] < LX_global  / 2 ) ? x[1] : 0 );
  xv[2] = ( x[2] > LY_global  / 2 ) ? x[2] - LY_global  : (  ( x[2] < LY_global  / 2 ) ? x[2] : 0 );
  xv[3] = ( x[3] > LZ_global  / 2 ) ? x[3] - LZ_global  : (  ( x[3] < LZ_global  / 2 ) ? x[3] : 0 );

  return;
}

/***********************************************************/
/***********************************************************/

void usage() {
  fprintf(stdout, "Code to perform contractions for hlbl tensor\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {

  double const mmuon = 105.6583745 /* MeV */  / 197.3269804 /* MeV fm */;
  double const alat[2] = { 0.07957, 0.00013 };  /* fm */


  int const ysign_num = 4;
  int const ysign_comb[16][4] = {
    { 1, 1, 1, 1},
    { 1, 1,-1,-1},
    { 1,-1, 1,-1},
    { 1,-1,-1, 1},
    {-1, 1,-1, 1},
    {-1,-1, 1, 1},
    {-1, 1, 1,-1},
    { 1, 1, 1,-1},
    { 1, 1,-1, 1},
    { 1,-1, 1, 1},
    { 1,-1,-1,-1},
    {-1, 1, 1, 1},
    {-1, 1,-1,-1},
    {-1,-1, 1,-1},
    {-1,-1,-1, 1},
    {-1,-1,-1,-1}
  };

  int idx_comb[6][2] = {
        {0,1},
        {0,2},
        {0,3},
        {1,2},
        {1,3},
        {2,3} };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  int first_solve_dummy = 0;
  struct timeval start_time, end_time;
  int ymax = 0;
  int evec_num =  0;
  int evec_test = 0;

  struct timeval ta, te;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "tch?f:y:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'y':
      ymax = atoi ( optarg );
      break;
    case 'n':
      evec_num = atoi ( optarg );
      break;
    case 't':
      evec_test = 1;
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
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [hlbl_lm_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [hlbl_lm_contract] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
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

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [hlbl_lm_contract] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hlbl_lm_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hlbl_lm_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hlbl_lm_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[hlbl_lm_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  size_t sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  if(!(strcmp(gaugefilename_prefix,"identity")==0)) 
  {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [hlbl_lm_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else
  {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [hlbl_lm_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }

#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[hlbl_lm_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[hlbl_lm_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
#if _WITH_TIMER
    gettimeofday ( &ta, (struct timezone *)NULL );
#endif
  /* exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up ); */
  exitstatus = gauge_field_eq_gauge_field_ti_bcfactor ( &gauge_field_with_phase, g_gauge_field, -1. );

  if(exitstatus != 0) {
    fprintf(stderr, "[hlbl_lm_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }
#if _WITH_TIMER
  gettimeofday ( &te, (struct timezone *)NULL );
  show_time ( &ta, &te, "hlbl_lm_contract", "gauge_field_eq_gauge_field_ti_phase", g_cart_id == 0 );
#endif


#if _WITH_TIMER
  gettimeofday ( &ta, (struct timezone *)NULL );
#endif
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[hlbl_lm_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }
#if _WITH_TIMER
  gettimeofday ( &te, (struct timezone *)NULL );
  show_time ( &ta, &te, "hlbl_lm_contract", "plaquetteria", g_cart_id == 0 );
#endif

  /***********************************************
   * initialize clover, mzz and mzz_inv
   *
   * FOR ZERO TWISTED MASS g_mu = 0
   ***********************************************/
#if _WITH_TIMER
  gettimeofday ( &ta, (struct timezone *)NULL );
#endif
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, 0, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hlbl_lm_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
#if _WITH_TIMER
  gettimeofday ( &te, (struct timezone *)NULL );
  show_time ( &ta, &te, "hlbl_lm_contract", "init_clover", g_cart_id == 0 );
#endif


  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hlbl_lm_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [hlbl_lm_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   *
   ***********************************************************/

  double ** spinor_field = init_2level_dtable ( 4, _GSI( (size_t)(VOLUME) ));
  if( spinor_field == NULL ) 
  {
    fprintf(stderr, "[hlbl_lm_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }

  /***********************************************************
   * filename for output data
   ***********************************************************/
  char output_filename[400];
  sprintf ( output_filename, "%s.%d.h5", g_outfile_prefix, Nconf );

  /***********************************************************
   * space for eigenvectors
   ***********************************************************/
  double ** evec_field = init_2level_dtable ( evec_num, _GSI( (size_t)(VOLUME) ));
  if( evec_field == NULL ) 
  {
    fprintf(stderr, "[hlbl_lm_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }

  /***********************************************************
   * read eigenvectors from file
   ***********************************************************/
  for ( int ievec = 0; ievec < evec_num; ievec++ )
  {
    sprintf (filename, "%s/eigVec_eV%d", filename_prefix, ievec);
#if _WITH_TIMER
    gettimeofday ( &ta, (struct timezone *)NULL );
#endif
    exitstatus = read_lime_spinor ( evec_field[ievec], filename, 0);
#if _WITH_TIMER
    gettimeofday ( &te, (struct timezone *)NULL );
    show_time ( &ta, &te, "hlbl_lm_contract", "read_lime_spinor", g_cart_id == 0 );
#endif
    if ( exitstatus != 0 )
    {
      fprintf( stderr, "[hlbl_lm_contract] Error from read_lime_spinor, status %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }
  }

  /***********************************************************
   * test eigenpair properties
   ***********************************************************/
  if ( evec_test )
  {

    double ** vv = init_2level_dtable ( evec_num, 2*evec_num );
    if ( vv == NULL ) 
    {
      fprintf(stderr, "[hlbl_lm_contract] Error from init_2level_ztable  %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }


    /* work fields, even-odd */
    double ** eo_spinor_work = init_2level_dtable (5, _GSI( (VOLUME+RAND)/2 ) );
    if ( eo_spinor_work == NULL )
    {
      fprintf(stderr, "[hlbl_lm_contract] Error from init_2level_buffer  %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    for ( int iv = 0; iv < evec_num; iv++ )
    {
      /* rotate from ukqcd to cvc gamma basis */
      memcpy ( spinor_field[0], evec_field[iv], sizeof_spinor_field );
#if _WITH_TIMER
      gettimeofday ( &ta, (struct timezone *)NULL );
#endif
      rotate_propagator_ETMC_UKQCD ( spinor_field[0], VOLUME );
#if _WITH_TIMER
      gettimeofday ( &te, (struct timezone *)NULL );
      show_time ( &ta, &te, "hlbl_lm_contract", "rotate_propagator_ETMC_UKQCD", g_cart_id == 0 );
#endif

    /* Q_clover_phi_matrix_eo */
      spinor_field_lexic2eo ( spinor_field[0], eo_spinor_work[0], eo_spinor_work[1] );
#if _WITH_TIMER
      gettimeofday ( &ta, (struct timezone *)NULL );
#endif
      Q_clover_phi_matrix_eo ( eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_phase, eo_spinor_work[4], mzz[0] );
#if _WITH_TIMER
      gettimeofday ( &te, (struct timezone *)NULL );
      show_time ( &ta, &te, "hlbl_lm_contract", "Q_clover_phi_matrix_eo", g_cart_id == 0 );
#endif

      spinor_field_eo2lexic ( spinor_field[0], eo_spinor_work[2], eo_spinor_work[3] );

      /* g5 application */
      g5_phi ( spinor_field[0], VOLUME );

      /* rotate from cvc back to ukqcd gamma basis */
#if _WITH_TIMER
      gettimeofday ( &ta, (struct timezone *)NULL );
#endif
      rotate_propagator_ETMC_UKQCD ( spinor_field[0], VOLUME );
#if _WITH_TIMER
      gettimeofday ( &te, (struct timezone *)NULL );
      show_time ( &ta, &te, "hlbl_lm_contract", "rotate_propagator_ETMC_UKQCD", g_cart_id == 0 );
#endif

      /* V x v_iv */
      double _Complex * p = init_1level_ztable ( evec_num );
      if ( p == NULL )
      {
        fprintf(stderr, "[hlbl_lm_contract] Error from init_1level_ztable  %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }

      /* projection on V-basis */
      double _Complex BLAS_ALPHA  = 1.;
      double _Complex BLAS_BETA   = 0.;
      char BLAS_TRANSA = 'C';
      char BLAS_TRANSB = 'N';
      int BLAS_M      = evec_num;
      int BLAS_K      = 12 * VOLUME;
      int BLAS_N      = 1;
      double _Complex * BLAS_A      = (double _Complex*)evec_field[0];
      double _Complex * BLAS_B      = (double _Complex*)(spinor_field[0]);
      double _Complex * BLAS_C      = p;
      int BLAS_LDA    = BLAS_K;
      int BLAS_LDB    = BLAS_K;
      int BLAS_LDC    = BLAS_M;

      _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
      /* allreduce across all processes */
      if( (p_buffer = (double _Complex*)malloc( bytes_p )) == NULL ) {
        fprintf(stderr, "[project_propagator_field] Error from malloc\n");
        return(2);
      }

      memcpy(p_buffer, p, bytes_p);
      status = MPI_Allreduce(p_buffer, p, 2*items_p, MPI_DOUBLE, MPI_SUM, g_cart_grid);
      if (status != MPI_SUCCESS) {
        fprintf(stderr, "[project_propagator_field] Error from MPI_Allreduce, status was %d\n", status);
        return(1);
      }
      free(p_buffer); p_buffer = NULL;
#endif

      for ( int k = 0; k < evec_num; k++ )
      {
        vv[iv][2*k  ] = creal ( p[k] );
        vv[iv][2*k+1] = cimag ( p[k] );
      }

      fini_1level_ztable ( &p );
    }   /* end of loop on evecs */

//     for ( int iv = 0; iv <= 2; iv++ )
//     {
//       for ( int iw = 0; iw <= 2; iw++ )
//       {
//         complex w = {0., 0.};
// 
// #if _WITH_TIMER
//         gettimeofday ( &ta, (struct timezone *)NULL );
// #endif
//         spinor_scalar_product_co ( &w, spinor_field[iw], spinor_field[iv], VOLUME );
// #if _WITH_TIMER
//         gettimeofday ( &te, (struct timezone *)NULL );
//         show_time ( &ta, &te, "hlbl_lm_contract", "spinor_scalar_product_co", g_cart_id == 0 );
// #endif
// 
//         if ( io_proc == 2 ) fprintf (stdout, "# [hlbl_lm_contract] sp %3d %3d  %25.16e %25.16e\n", iw, iv, w.re, w.im);
//       }
//     }

    /* deacllocate */
    fini_2level_dtable ( &eo_spinor_work );


    if ( io_proc == 2 )
    {
      sprintf ( filename, "vv.%d", Nconf );
      FILE * fs = fopen ( filename, "w" );

      for ( int k = 0; k < evec_num; k++ )
      {
        for ( int l = 0; l < evec_num; l++ )
        {
          fprintf( fs, "%4d %4d %25.16e %25.16e\n", k, l, vv[k][2*l], vv[k][2*l+1] );
        }
      }

      fclose ( fs );
    }


    fini_2level_dtable ( &vv );

  }  /* end of if evec_test */

#if 0
#endif  /* if 0 */

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/
          
  fini_2level_dtable ( &evec_field );
  fini_2level_dtable ( &spinor_field );


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
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "hlbl_lm_contract", "runtime", g_cart_id == 0 );

  return(0);
}
