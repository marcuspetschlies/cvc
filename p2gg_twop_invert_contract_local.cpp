/****************************************************
 * p2gg_twop_invert_contract_local
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
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"

#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

/****************************************************
 * defines for 2-pt gamma vertex combinations
 * and charged / neutral
 *
 * ALL IN TwISTED BASIS
 *
 * TWISTED BASIS  --- PHYSICAL BASIS
 * S C                s c
 * P C                p c
 * V C                a c
 * A C                v c
 *
 * S N                p n
 * P N                s n
 * V N                v n
 * A N                a n
 ****************************************************/

/****************************************************
 * NEUTRAL combinations
 ****************************************************/
/* VVN = vvn */
#ifndef _V_V_N 
#  define _V_V_N 1
#endif

/* SVN = pvn */
#ifndef _S_V_N
#  define _S_V_N 0
#endif

/* VSN = vpn */
#ifndef _V_S_N
#  define _V_S_N 0
#endif

/* SSN = ppn */
#ifndef _S_S_N
#  define _S_S_N 1
#endif

/* SAN = pan */
#ifndef _S_A_N 
#  define _S_A_N 0
#endif

/* ASN = apn */
#ifndef _A_S_N 
#  define _A_S_N 0
#endif

/* AAN = aan */
#ifndef _A_A_N
#  define _A_A_N 0
#endif

/****************************************************
 * CHARGED combinations
 ****************************************************/

/* AAC = vvc */
#ifndef _A_A_C
#  define _A_A_C 0
#endif

/* APC = vpc */
#ifndef _A_P_C
#  define _A_P_C 0
#endif

/* PAC = pvc */
#ifndef _P_A_C
#  define _P_A_C 0
#endif

/* PPC = ppc */
#ifndef _P_P_C
#  define _P_P_C 0
#endif

/* VVC = aac  */
#ifndef _V_V_C
#  define _V_V_C 0
#endif

/* VPC = apc */
#ifndef _V_P_C
#  define _V_P_C 0
#endif

/* PVC = PAC */
#ifndef _P_V_C
#  define _P_V_C 0
#endif

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -w                  : check position space WI [default false]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {

  /* char const outfile_prefix[] = "p2gg_local"; */
#if 0
  /*                            gt  gx  gy  gz */
  int const gamma_v_list[4] = {  0,  1,  2,  3 };
  int const gamma_v_num = 4;
#endif
  /*                            gx  gy  gz */
  int const gamma_v_list[3] = { 1,  2,  3 };
  int const gamma_v_num = 3;

#if 0
  /*                            gtg5 gxg5 gyg5 gzg5 */
  int const gamma_a_list[4] = {    6,   7,   8,   9 };
  int const gamma_a_num = 4;
#endif
  /* vector, axial vector */
  /* int const gamma_va_list[8] = { 0,  1,  2,  3,  6,   7,   8,   9 }; */
  /* int const gamma_va_num = 8; */

  /*                             id  g5 */
  /* int const gamma_sp_list[2] = { 4  , 5 }; */
  /* int const gamma_sp_num = 2; */

  int const gamma_s = 4;
  int const gamma_s_list[1] = { 4 };
  int const gamma_s_num = 1;

#if 0
  int const gamma_p = 5;
  int const gamma_p_list[1] = { 5 };
  int const gamma_p_num = 1;
#endif

  int c;
  int filename_set = 0;
  int gsx[4], sx[4];
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  size_t sizeof_spinor_field;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  struct timeval start_time, end_time, ta, tb;
  int spin_dilution = 4;
  int color_dilution = 1;
  char h5_tag[100];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
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
  /* fprintf(stdout, "# [p2gg_twop_invert_contract_local] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_twop_invert_contract_local] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [p2gg_twop_invert_contract_local] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_twop_invert_contract_local] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_twop_invert_contract_local] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_twop_invert_contract_local] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [p2gg_twop_invert_contract_local] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg_twop_invert_contract_local] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg_twop_invert_contract_local] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg_twop_invert_contract_local] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_twop_invert_contract_local] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [p2gg_twop_invert_contract_local] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  int * rng_state = NULL;
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[vv2pt_simple_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***********************************************************
   * allocate eo_spinor_field
   ***********************************************************/
  int spin_color_dilution = spin_dilution * color_dilution;

  double ** source = init_2level_dtable ( spin_color_dilution, _GSI( (size_t)VOLUME));
  if( source == NULL ) {
    fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_level_table %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }
  
  double *** propagator = init_3level_dtable ( 2, spin_color_dilution, _GSI( (size_t)VOLUME));
  if( propagator == NULL ) {
    fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_level_table %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }
  
  double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ) );
  if( propagator == NULL ) {
    fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_level_table %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }
  
  /***********************************************************
   ***********************************************************
   **
   ** loop on source locations
   **
   ***********************************************************
   ***********************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) 
  {
    /***********************************************************
     * determine source coordinates, find out, if source_location is in this process
     ***********************************************************/
    gsx[0] = ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global;
    gsx[1] = ( g_source_coords_list[isource_location][1] + LX_global ) % LX_global;
    gsx[2] = ( g_source_coords_list[isource_location][2] + LY_global ) % LY_global;
    gsx[3] = ( g_source_coords_list[isource_location][3] + LZ_global ) % LZ_global;

    int source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /**********************************************************
     * outputf filename
     **********************************************************/
    if(io_proc == 2) 
    {
      sprintf ( filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.h5", g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [p2gg_twop_invert_contract_local] writing data to file %s\n", filename);
    }

    /**********************************************************
     * write momentum and gamma lists to file
     **********************************************************/

    if ( io_proc == 2 )
    {
      int const mom_cdim[2] = { g_sink_momentum_number, 3 };
      exitstatus = write_h5_contraction ( g_sink_momentum_list[0], NULL, filename, "/mom_snk", "int", 2, mom_cdim );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(123);
      }

      exitstatus = write_h5_contraction ( (void*)gamma_s_list, NULL, filename, "/gamma_s", "int", 1, &gamma_s_num );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(123);
      }

      exitstatus = write_h5_contraction ( (void*)gamma_v_list, NULL, filename, "/gamma_v", "int", 1, &gamma_v_num );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(123);
      }
    }

    /**********************************************************
     **********************************************************
     **
     ** propagators with source at gsx
     **
     **********************************************************
     **********************************************************/

    exitstatus = init_point_source_oet ( source, gsx[0], &(gsx[1]), spin_dilution, color_dilution, 1 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_point_source_oet, status %d    %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(21);
    }

    /**********************************************************
     * propagators
     **********************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ )
    {
      for ( int isc = 0; isc < spin_color_dilution; isc++ )
      {
        memset ( spinor_work[1], 0, sizeof_spinor_field );

        memcpy ( spinor_work[0], source[isc], sizeof_spinor_field );

        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
        if(exitstatus < 0)
        {
          fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        if ( check_propagator_residual )
        {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
        }

        memcpy( propagator[iflavor][isc], spinor_work[1], sizeof_spinor_field );

      }  /* end of loop on spin-color components */

    }  /* end of loop on flavors */

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     *
     * local - local 2-point  u - u neutral
     *
     ***************************************************************************/

    /***************************************************************************
     * contraction vector -vector
     ***************************************************************************/
    double * contr_x = init_1level_dtable ( 2*VOLUME );
    if ( contr_x == NULL ) {
      fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_level_table %s %d\n", __FILE__, __LINE__);
      EXIT(3);
    }
    
    double ** contr_p = NULL;
    double **** contr_accum = NULL;

#if _V_V_N
    gettimeofday ( &ta, (struct timezone *)NULL );

#define _GAMMA_I_NUM  gamma_v_num
#define _GAMMA_I_LIST gamma_v_list
#define _GAMMA_F_NUM  gamma_v_num
#define _GAMMA_F_LIST gamma_v_list

    contr_accum = init_4level_dtable ( T, g_sink_momentum_number, _GAMMA_F_NUM, 2 * _GAMMA_I_NUM );
    if ( contr_accum == NULL ) {
      fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_level_table %s %d\n", __FILE__, __LINE__);
      EXIT(3);
    }

    contr_p = init_2level_dtable ( g_sink_momentum_number, 2*T );
    if ( contr_p == NULL ) {
      fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_level_table %s %d\n", __FILE__, __LINE__);
      EXIT(3);
    }

    sprintf ( h5_tag, "/local-local/u-v-u-v" );

    for ( int igi = 0; igi < _GAMMA_I_NUM; igi++ )
    {
      int idsource = _GAMMA_I_LIST[igi];

      for ( int igf = 0; igf < _GAMMA_F_NUM; igf++ )
      {
        int idsink = _GAMMA_F_LIST[igf];

        /***************************************************************************
         * gig5 D^+ g5gf U = gi U gf U
         ***************************************************************************/
        memset ( contr_x, 0, 2*VOLUME*sizeof(double) );
        contract_twopoint_xdep ( contr_x, idsource, idsink, propagator[1], propagator[0], spin_dilution, color_dilution, 1, 1, 64 );

        memset ( contr_p[0], 0, g_sink_momentum_number * 2*T * sizeof(double) );
        exitstatus = momentum_projection ( contr_x, contr_p[0], T, g_sink_momentum_number, g_sink_momentum_list );
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(3);
        }

#pragma omp parallel for
        for ( int it = 0; it < T; it++ )
        {
          for ( int ip = 0; ip < g_sink_momentum_number; ip++ )
          {
            contr_accum[it][ip][igf][2*igi  ] = contr_p[ip][2*it  ];
            contr_accum[it][ip][igf][2*igi+1] = contr_p[ip][2*it+1];
          }
        }

      }  /* end of loop on idsink */
    }  /* end of loop on idsource */

    if ( io_proc > 0 ) 
    {
      double * write_buffer = NULL;
#ifdef HAVE_MPI
      int mitems = 2 * T * g_sink_momentum_number * _GAMMA_F_NUM * _GAMMA_I_NUM;
 
      if ( io_proc == 2 ) 
      {
        write_buffer = init_1level_dtable ( 2 * T_global * g_sink_momentum_number * _GAMMA_F_NUM * _GAMMA_I_NUM ); 
        if ( write_buffer == NULL ) {
          fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_level_table %s %d\n", __FILE__, __LINE__);
          EXIT(3);
        }
      }

      /***************************************************************************
       * io_proc's 1 and 2 gather the data to g_tr_id = 0 into zbuffer
       ***************************************************************************/
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
      exitstatus = MPI_Gather ( contr_accum[0][0][0], mitems, MPI_DOUBLE, write_buffer, mitems, MPI_DOUBLE, 0, g_tr_comm);
#  else
      exitstatus = MPI_Gather ( contr_accum[0][0][0], mitems, MPI_DOUBLE, write_buffer, mitems, MPI_DOUBLE, 0, g_cart_grid);
#  endif
      if(exitstatus != MPI_SUCCESS) {
        fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(4);
      }
#else
      write_buffer = contr_accum[0][0][0];
#endif
      if ( io_proc == 2 ) 
      {
        int const ncdim = 4;
        int const cdim[4] = { T_global, g_sink_momentum_number, _GAMMA_F_NUM, 2 * _GAMMA_I_NUM };

        exitstatus = write_h5_contraction ( write_buffer, NULL, filename, h5_tag, "double", ncdim, cdim );

        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(3);
        }

#ifdef HAVE_MPI
        fini_1level_dtable ( &write_buffer );
#endif
      }
      fini_4level_dtable ( &contr_accum );

    }  /* end of if io_proc > 0 */
     
    fini_4level_dtable ( &contr_accum );
    fini_2level_dtable ( &contr_p );

#undef _GAMMA_I_NUM
#undef _GAMMA_I_LIST
#undef _GAMMA_F_NUM
#undef _GAMMA_F_LIST

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_twop_invert_contract_local", "v-v-n", io_proc==2 );
#endif  /* of if _V_V_N */

    /***************************************************************************
     * contraction s - s
     ***************************************************************************/
#if _S_S_N
    gettimeofday ( &ta, (struct timezone *)NULL );

#define    _GAMMA_I_NUM  gamma_s_num
#define    _GAMMA_I_LIST gamma_s_list
#define    _GAMMA_F_NUM  gamma_s_num
#define    _GAMMA_F_LIST gamma_s_list

    contr_accum = init_4level_dtable ( T, g_source_momentum_number, _GAMMA_F_NUM, 2 * _GAMMA_I_NUM );
    if ( contr_accum == NULL ) {
      fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_level_table %s %d\n", __FILE__, __LINE__);
      EXIT(3);
    }

    contr_p = init_2level_dtable ( g_source_momentum_number, 2*T );
    if ( contr_p == NULL ) {
      fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_level_table %s %d\n", __FILE__, __LINE__);
      EXIT(3);
    }

    sprintf ( h5_tag, "/local-local/u-s-u-s" );

    for ( int igi = 0; igi < _GAMMA_I_NUM; igi++ )
    {
      int idsource = _GAMMA_I_LIST[igi];

      for ( int igf = 0; igf < _GAMMA_F_NUM; igf++ )
      {
        int idsink = _GAMMA_F_LIST[igf];

        /***************************************************************************
         * gig5 D^+ g5gf U = gi U gf U
         ***************************************************************************/
        memset ( contr_x, 0, 2*VOLUME*sizeof(double) );
        contract_twopoint_xdep ( contr_x, idsource, idsink, propagator[1], propagator[0], spin_dilution, color_dilution, 1, 1, 64 );

        memset ( contr_p[0], 0, g_source_momentum_number * 2*T * sizeof(double) );
        exitstatus = momentum_projection ( contr_x, contr_p[0], T, g_source_momentum_number, g_source_momentum_list );
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(3);
        }

#pragma omp parallel for
        for ( int it = 0; it < T; it++ )
        {
          for ( int ip = 0; ip < g_source_momentum_number; ip++ )
          {
            contr_accum[it][ip][igf][2*igi  ] = contr_p[ip][2*it  ];
            contr_accum[it][ip][igf][2*igi+1] = contr_p[ip][2*it+1];
          }
        }

      }  /* end of loop on idsink */
    }  /* end of loop on idsource */

    if ( io_proc > 0 ) 
    {
      double * write_buffer = NULL;
#ifdef HAVE_MPI
      int mitems = 2 * T * g_source_momentum_number * _GAMMA_F_NUM * _GAMMA_I_NUM;
 
      if ( io_proc == 2 ) 
      {
        write_buffer = init_1level_dtable ( 2 * T_global * g_source_momentum_number * _GAMMA_F_NUM * _GAMMA_I_NUM ); 
        if ( write_buffer == NULL ) {
          fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from init_level_table %s %d\n", __FILE__, __LINE__);
          EXIT(3);
        }
      }

      /***************************************************************************
       * io_proc's 1 and 2 gather the data to g_tr_id = 0 into zbuffer
       ***************************************************************************/
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
      exitstatus = MPI_Gather ( contr_accum[0][0][0], mitems, MPI_DOUBLE, write_buffer, mitems, MPI_DOUBLE, 0, g_tr_comm);
#  else
      exitstatus = MPI_Gather ( contr_accum[0][0][0], mitems, MPI_DOUBLE, write_buffer, mitems, MPI_DOUBLE, 0, g_cart_grid);
#  endif
      if(exitstatus != MPI_SUCCESS) {
        fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(4);
      }
#else
      write_buffer = contr_accum[0][0][0];
#endif
      if ( io_proc == 2 ) 
      {
        int const ncdim = 4;
        int const cdim[4] = { T_global, g_source_momentum_number, _GAMMA_F_NUM, 2 * _GAMMA_I_NUM };

        exitstatus = write_h5_contraction ( write_buffer, NULL, filename, h5_tag, "double", ncdim, cdim );

        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg_twop_invert_contract_local] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(3);
        }

#ifdef HAVE_MPI
        fini_1level_dtable ( &write_buffer );
#endif
      }

      fini_4level_dtable ( &contr_accum );

    }  /* end of if io_proc > 0 */
     
    fini_4level_dtable ( &contr_accum );
    fini_2level_dtable ( &contr_p );

#undef _GAMMA_I_NUM
#undef _GAMMA_I_LIST
#undef _GAMMA_F_NUM
#undef _GAMMA_F_LIST

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_twop_invert_contract_local", "s-s-n", io_proc==2 );
#endif  /* of if _S_S_N */

    fini_1level_dtable ( &contr_x );

  }  /* end of loop on source locations */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

  fini_rng_state ( &rng_state );

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  fini_2level_dtable ( &source );
  fini_3level_dtable ( &propagator );
  fini_2level_dtable ( &spinor_work );

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
  show_time ( &start_time, &end_time, "p2gg_twop_invert_contract_local", "runtime", g_cart_id == 0 );

  return(0);
}
