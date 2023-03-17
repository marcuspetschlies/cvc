/****************************************************
 * p2gg_local
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
#include "contractions_io.h"
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
 * defines for tensors to be contracted
 ****************************************************/
#ifndef _NEUTRAL_LVC_LVC_TENSOR
#  define _NEUTRAL_LVC_LVC_TENSOR 1
#endif

#ifndef _CHARGED_LVC_LVC_TENSOR
#  define _CHARGED_LVC_LVC_TENSOR 1
#endif

#if _NEUTRAL_LVC_LVC_TENSOR
#warning "[p2gg_local] contract neutral lvc - lvc tensor"
#else
#warning "[p2gg_local] DO NOT contract neutral lvc - lvc tensor"
#endif

#if _CHARGED_LVC_LVC_TENSOR
#warning "[p2gg_local]  contract chargd lvc - lvc tensor"
#else
#warning "[p2gg_local] DO NOT contract chargd lvc - lvc tensor"
#endif

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
#  define _S_V_N 1
#endif

/* VSN = vpn */
#ifndef _V_S_N
#  define _V_S_N 1
#endif

/* SSN = ppn */
#ifndef _S_S_N
#  define _S_S_N 1
#endif

#if 0
/* SAN = pan */
#ifndef _S_A_N 
#  define _S_A_N 1
#endif

/* ASN = apn */
#ifndef _A_S_N 
#  define _A_S_N 1 
#endif

/* AAN = aan */
#ifndef _A_A_N
#  define _A_A_N 1
#endif
#endif

/****************************************************
 * CHARGED combinations
 ****************************************************/

/* AAC = vvc */
#ifndef _A_A_C
#  define _A_A_C 1
#endif

/* APC = vpc */
#ifndef _A_P_C
#  define _A_P_C 1
#endif

/* PAC = pvc */
#ifndef _P_A_C
#  define _P_A_C 1
#endif

/* PPC = ppc */
#ifndef _P_P_C
#  define _P_P_C 1
#endif

#if 0
/* VVC = aac  */
#ifndef _V_V_C
#  define _V_V_C 1
#endif

/* VPC = apc */
#ifndef _V_P_C
#  define _V_P_C 1
#endif

/* PVC = PAC */
#ifndef _P_V_C
#  define _P_V_C 1
#endif
#endif

using namespace cvc;

int contract_local_local_2pt ( double ** const chi, double ** const phi, 
    const int * const gamma_snk_list, int const gamma_snk_num, const int * const gamma_src_list, int const gamma_src_num, 
    int (* const momentum_list)[3], int const momentum_num, char * const filename, char * data_tag, int const io_proc )
{
  
  int exitstatus;

  /* allocate contraction fields in position and momentum space */
  double * contr_x = init_1level_dtable ( 2 * VOLUME );
  if ( contr_x == NULL ) {
    fprintf(stderr, "[contract_local_local_2pt] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }
    
  double ** contr_p = init_2level_dtable ( momentum_num, 2 * T );
  if ( contr_p == NULL ) {
    fprintf(stderr, "[contract_local_local_2pt] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
    return(2);
  }
  
  double **** contr_accum = init_4level_dtable ( T, gamma_snk_num, gamma_src_num, 2*momentum_num );
  if ( contr_accum == NULL ) {
    fprintf(stderr, "[contract_local_local_2pt] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
    return(3);
  }

  for ( int igsrc = 0; igsrc < gamma_src_num; igsrc++ ) 
  {
    for ( int igsnk = 0; igsnk < gamma_snk_num; igsnk++ ) 
    {

      memset ( contr_x , 0, 2 * VOLUME * sizeof ( double ) );

      /***************************************************************************
       * Xbar(0)^+ g5 Gf Y(p) Gi g5
       *
       ***************************************************************************/
      contract_twopoint_xdep ( contr_x, gamma_src_list[igsrc], gamma_snk_list[igsnk], chi, phi, 4, 3, 1, 1., 64 );
    
      /***************************************************************************
       * momentum projection at sink
       ***************************************************************************/
      exitstatus = momentum_projection ( contr_x, contr_p[0], T, momentum_num, momentum_list );
      if(exitstatus != 0) {
        fprintf(stderr, "[contract_local_local_2pt] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

#pragma omp parallel for
      for ( int it = 0; it < T; it++ ) 
      {
        for ( int ip = 0; ip < momentum_num; ip++ ) 
        {
          contr_accum[it][igsnk][igsrc][2*ip  ] = contr_p[ip][2*it  ];
          contr_accum[it][igsnk][igsrc][2*ip+1] = contr_p[ip][2*it+1];
        }
      }
    
    }
  }
  
#ifdef HAVE_MPI
  double * write_buffer = NULL;
  if ( io_proc > 0 ) {
    int mitems = 2 * T * momentum_num * gamma_src_num * gamma_snk_num;
 
    if ( io_proc == 2 ) {
      write_buffer = init_1level_dtable ( 2 * T_global * momentum_num * gamma_src_num * gamma_snk_num );
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
      fprintf(stderr, "[contract_local_local_2pt] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(4);
    }
  }
#else
  double * write_buffer = contr_accum[0][0][0];
#endif

  if ( io_proc == 2 ) {

    int const ncdim = 4;
    int const cdim[4] = { T_global, gamma_snk_num, gamma_src_num, 2*momentum_num };

    exitstatus = write_h5_contraction ( write_buffer, NULL, filename, data_tag, "double", ncdim, cdim );

    if(exitstatus != 0) {
      fprintf(stderr, "[contract_local_local_2pt] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(3);
    }

#ifdef HAVE_MPI
    fini_1level_dtable ( &write_buffer );
#endif

  }

  /* deallocate the contraction fields */       
  fini_1level_dtable ( &contr_x );
  fini_2level_dtable ( &contr_p );
  fini_4level_dtable ( &contr_accum );

  return(0);
                  
}

/***************************************************************************/
/***************************************************************************/

void usage() {
  fprintf(stdout, "Code to perform P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -w                  : check position space WI [default false]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

/***************************************************************************/
/***************************************************************************/

int main(int argc, char **argv) {

  /* char const outfile_prefix[] = "p2gg_local"; */

  /*                            gt  gx  gy  gz */
  int const gamma_v_list[4] = {  0,  1,  2,  3 };
  int const gamma_v_num = 4;

  /*                            gtg5 gxg5 gyg5 gzg5 */
  int const gamma_a_list[4] = {    6,   7,   8,   9 };
  int const gamma_a_num = 4;

  int const gamma_s = 4;
  int const gamma_s_list[1] = { 4 };
  int const gamma_s_num = 1;

  int const gamma_p = 5;
  int const gamma_p_list[1] = { 5 };
  int const gamma_p_num = 1;
  

  int c;
  int filename_set = 0;
  int gsx[4], sx[4];
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  int first_solve_dummy = 1;
  struct timeval start_time, end_time;
  char tag[400];

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
  /* fprintf(stdout, "# [p2gg_local] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_local] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [p2gg_local] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_local] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_local] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_local] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_local] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  size_t const sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [p2gg_local] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg_local] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg_local] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg_local] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_local] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_local] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[p2gg_local] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_local] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [p2gg_local] proc%.4d has io proc id %d\n", g_cart_id, io_proc );



  /***********************************************************
   * allocate spinor_field
   ***********************************************************/
  double ** spinor_field = init_2level_dtable ( 36, _GSI( (size_t)VOLUME) );
  if( spinor_field == NULL ) {
    fprintf(stderr, "[p2gg_local] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }

  /***********************************************************
   * allocate spinor_field
   ***********************************************************/
  double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ) );
  if( spinor_work == NULL ) {
    fprintf(stderr, "[p2gg_local] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }
  
  /***********************************************************
   ***********************************************************
   **
   ** dummy inversion for solver tuning
   **
   ** use volume source
   **
   ***********************************************************
   ***********************************************************/

  if ( first_solve_dummy ) {
    /***********************************************************
     * initialize rng state
     ***********************************************************/
    exitstatus = init_rng_stat_file ( g_seed, NULL );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
      EXIT( 50 );
    }

    if( ( exitstatus = prepare_volume_source ( spinor_field[0], VOLUME ) ) != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(64);
    }

    memset ( spinor_field[1], 0, sizeof_spinor_field);

    /* full_spinor_work[1] = D^-1 full_spinor_work[0],
     * flavor id 0 
     */
    exitstatus = _TMLQCD_INVERT ( spinor_field[1], spinor_field[0], 0 );
    if(exitstatus < 0) {
      fprintf(stderr, "[p2gg_local] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }

    /* check residuum */
    exitstatus = check_residual_clover ( &(spinor_field[1]), &(spinor_field[0]), gauge_field_with_phase, mzz[0], mzzinv[0], 1);
    if ( exitstatus != 0 ) 
    {
      fprintf(stderr, "[p2gg_local] Error from check_residual_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }

  }  /* end of first_solve_dummy */


  /***********************************************************
   ***********************************************************
   **
   ** loop on source locations
   **
   ***********************************************************
   ***********************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

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
      fprintf(stderr, "[p2gg_local] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***********************************************
     ***********************************************
     **
     ** output file
     **
     ***********************************************
     ***********************************************/
    if(io_proc == 2) {
      sprintf ( filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.h5", g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [p2gg_local] writing data to file %s\n", filename);
    }  /* end of if io_proc == 2 */

    /**********************************************************
     * write gamma lists to h5 file
     **********************************************************/
    if ( io_proc == 2 ) {

      int const nvdim = 1;
      int const vdim[1] = { 4 };

      exitstatus = write_h5_contraction ( (void*)gamma_v_list, NULL, filename, "/gamma_v", "int", nvdim, vdim );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_local] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

      int const nadim = 1;
      int const adim[1] = { 4 };

      exitstatus = write_h5_contraction ( (void*)gamma_a_list, NULL, filename, "/gamma_a", "int", nadim, adim );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_local] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

      int const npdim = 1;
      int const pdim[1] = { 1 };

      exitstatus = write_h5_contraction ( (void*)gamma_p_list, NULL, filename, "/gamma_p", "int", npdim, pdim );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_local] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

      int const nsdim = 1;
      int const sdim[1] = { 1 };

      exitstatus = write_h5_contraction ( (void*)gamma_s_list, NULL, filename, "/gamma_s", "int", nsdim, sdim );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_local] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

      int const nqdim = 2;
      int qdim[2] = { g_source_momentum_number, 3  };

      exitstatus = write_h5_contraction ( (void*)(g_source_momentum_list[0]), NULL, filename, "/mom_src", "int", nqdim, qdim );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_local] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

      qdim[0] = g_sink_momentum_number;

      exitstatus = write_h5_contraction ( (void*)(g_sink_momentum_list[0]), NULL, filename, "/mom_snk", "int", nqdim, qdim );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_local] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(3);
      }

    }

    /**********************************************************
     **********************************************************
     **
     ** propagators with source at gsx
     ** up and down type
     **
     **********************************************************
     **********************************************************/

    for ( int iflavor = 0; iflavor <=1; iflavor++ ) 
    {
      for( int is = 0; is < 12; is++) {

        memset ( spinor_work[0], 0, sizeof_spinor_field );
        memset ( spinor_work[1], 0, sizeof_spinor_field );

        if(source_proc_id == g_cart_id)  {
          spinor_work[0][_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]])+2*is] = 1.;
        }

        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
        if(exitstatus < 0) {
          fprintf(stderr, "[avgx_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
        }

        memcpy ( spinor_field[iflavor*12+is], spinor_work[1], sizeof_spinor_field );
      }  /* end of loop on spin-color component */
    
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
#if _V_V_N
    sprintf ( tag, "/local-local/u-v-u-v" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[ 0]), 
       gamma_v_list, gamma_v_num, gamma_v_list, gamma_v_num,
       g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_V_N */

    /***************************************************************************
     * contraction scalar - vector
     ***************************************************************************/
#if _S_V_N
    sprintf ( tag, "/local-local/u-s-u-v" );
    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[ 0]),
       gamma_s_list, gamma_s_num, gamma_v_list, gamma_v_num,
       g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _S_V_N */

    /***************************************************************************
     * contraction vector - scalar
     ***************************************************************************/
#if _V_S_N
    sprintf ( tag, "/local-local/u-v-u-s" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[ 0]),
       gamma_v_list, gamma_v_num, gamma_s_list, gamma_s_num,
       g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_S_N */

    /***************************************************************************
     * contraction s - s
     ***************************************************************************/
#if _S_S_N
    sprintf ( tag, "/local-local/u-s-u-s" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[ 0]),
       gamma_s_list, gamma_s_num, gamma_s_list, gamma_s_num,
       g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _S_S_N */

    /***************************************************************************
     * different set of momenta here
     *
     ***************************************************************************/

    /***************************************************************************
     * contraction axial - axial
     ***************************************************************************/
#if _A_A_N
    sprintf ( tag, "/local-local/u-a-u-a" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[ 0]),
       gamma_a_list, gamma_a_num, gamma_a_list, gamma_a_num,
       g_source_momentum_list, g_source_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _A_A_N */

    /***************************************************************************
     * contraction scalar - axial
     ***************************************************************************/
#if _S_A_N
    sprintf ( tag, "/local-local/u-s-u-a" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[ 0]),
       gamma_s_list, gamma_s_num, gamma_a_list, gamma_a_num,
       g_source_momentum_list, g_source_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* _S_A_N */

    /***************************************************************************
     * contraction axial - scalar
     ***************************************************************************/
#if _A_S_N
    sprintf ( tag, "/local-local/u-a-u-s" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[ 0]),
       gamma_a_list, gamma_a_num, gamma_s_list, gamma_s_num,
       g_source_momentum_list, g_source_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif   /* of if _A_S_N */

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     ***************************************************************************
     **
     ** local - local 2-point  d - u charged
     **
     ***************************************************************************
     ***************************************************************************/

    /***************************************************************************
     * contraction axial - axial
     ***************************************************************************/
#if _A_A_C
    sprintf ( tag, "/local-local/d-a-u-a" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[ 0]), &(spinor_field[ 0]),
       gamma_a_list, gamma_a_num, gamma_a_list, gamma_a_num,
       g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _A_A_C */

    /***************************************************************************
     * contraction pseudoscalar - axial
     ***************************************************************************/
#if _P_A_C
    sprintf ( tag, "/local-local/d-p-u-a" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[ 0]), &(spinor_field[ 0]),
       gamma_p_list, gamma_p_num, gamma_a_list, gamma_a_num,
       g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _P_A_C */

    /***************************************************************************
     * contraction axial - pseudoscalar
     ***************************************************************************/
#if _A_P_C
    sprintf ( tag, "/local-local/d-a-u-p" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[ 0]), &(spinor_field[ 0]),
       gamma_a_list, gamma_a_num, gamma_p_list, gamma_p_num,
       g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _A_P_C */

    /***************************************************************************
     * contraction pseudoscalar - pseudoscalar
     ***************************************************************************/
#if _P_P_C
    sprintf ( tag, "/local-local/d-p-u-p" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[ 0]), &(spinor_field[ 0]),
       gamma_p_list, gamma_p_num, gamma_p_list, gamma_p_num,
       g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _P_P_C */

#if 0
    /***************************************************************************
     * different set of momenta here
     ***************************************************************************/

    /***************************************************************************
     * contraction vector - vector
     ***************************************************************************/
#if _V_V_C
    sprintf ( tag, "/local-local/d-v-u-v" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[ 0]), &(spinor_field[ 0]),
       gamma_v_list, gamma_v_num, gamma_v_list, gamma_v_num,
       g_source_momentum_list, g_source_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_V_C */

    /***************************************************************************
     * contraction vector - pseudoscalar
     ***************************************************************************/
#if _V_P_C
    sprintf ( tag, "/local-local/d-v-u-p" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[ 0]), &(spinor_field[ 0]),
       gamma_v_list, gamma_v_num, gamma_p_list, gamma_p_num,
       g_source_momentum_list, g_source_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_P_C */

    /***************************************************************************
     * contraction vector - vector
     ***************************************************************************/
#if _P_V_C
    sprintf ( tag, "/local-local/d-p-u-v" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[ 0]), &(spinor_field[ 0]),
       gamma_p_list, gamma_p_num, gamma_v_list, gamma_v_num,
       g_source_momentum_list, g_source_momentum_number, filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _P_V_C */

#endif

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     ***************************************************************************
     **
     ** local - local 2-point  u - d charged
     **
     ***************************************************************************
     ***************************************************************************/

    /***************************************************************************
     * contraction axial - axial
     ***************************************************************************/
#if _A_A_C
    sprintf ( tag, "/local-local/u-a-d-a" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[12]),
       gamma_a_list, gamma_a_num, gamma_a_list, gamma_a_num,
       g_sink_momentum_list, g_sink_momentum_number,  filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _A_A_C */

    /***************************************************************************
     * contraction pseudoscalar - axial
     ***************************************************************************/
#if _P_A_C
    sprintf ( tag, "/local-local/u-p-d-a" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[12]),
       gamma_p_list, gamma_p_num, gamma_a_list, gamma_a_num,
       g_sink_momentum_list, g_sink_momentum_number,  filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif /* of if _P_A_C */

    /***************************************************************************
     * contraction axial - pseudoscalar
     ***************************************************************************/
#if _A_P_C
    sprintf ( tag, "/local-local/u-a-d-p" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[12]),
       gamma_a_list, gamma_a_num, gamma_p_list, gamma_p_num,
       g_sink_momentum_list, g_sink_momentum_number,  filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _A_P_C  */

    /***************************************************************************
     * contraction pseudoscalar - pseudoscalar
     ***************************************************************************/
#if _P_P_C
    sprintf ( tag, "/local-local/u-p-d-p" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[12]),
       gamma_p_list, gamma_p_num, gamma_p_list, gamma_p_num,
       g_sink_momentum_list, g_sink_momentum_number,  filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _P_P_C */

#if 0
    /***************************************************************************
     * different set of momenta here
     ***************************************************************************/

    /***************************************************************************
     * contraction vector - vector
     ***************************************************************************/
#if _V_V_C
    sprintf ( tag, "/local-local/u-v-d-v" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[12]),
       gamma_v_list, gamma_v_num, gamma_v_list, gamma_v_num,
       g_source_momentum_list, g_source_momentum_number,  filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_V_C */

    /***************************************************************************
     * contraction vector - pseudoscalar
     ***************************************************************************/
#if _V_P_C
    sprintf ( tag, "/local-local/u-v-d-p" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[12]),
       gamma_v_list, gamma_v_num, gamma_p_list, gamma_p_num,
       g_source_momentum_list, g_source_momentum_number,  filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_P_C */

    /***************************************************************************
     * contraction pseudoscalar - vector
     ***************************************************************************/
#if _P_V_C
    sprintf ( tag, "/local-local/u-p-d-v" );

    exitstatus = contract_local_local_2pt (
       &(spinor_field[12]), &(spinor_field[12]),
       gamma_p_list, gamma_p_num, gamma_v_list, gamma_v_num,
       g_source_momentum_list, g_source_momentum_number,  filename, tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _P_V_C */

#endif
    
    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     ***************************************************************************
     **
     ** P -> gamma gamma contractions
     **
     ***************************************************************************
     ***************************************************************************/

    /***************************************************************************
     * loop on quark flavors
     ***************************************************************************/
    for( int iflavor = 1; iflavor >= 0; iflavor-- )
    {

      /***************************************************************************
       * loop on sequential source gamma matrices
       ***************************************************************************/
      for ( int iseq_source_momentum = 0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) {

        g_seq_source_momentum[0] = g_seq_source_momentum_list[iseq_source_momentum][0];
        g_seq_source_momentum[1] = g_seq_source_momentum_list[iseq_source_momentum][1];
        g_seq_source_momentum[2] = g_seq_source_momentum_list[iseq_source_momentum][2];

        if( g_verbose > 2 && g_cart_id == 0) fprintf(stdout, "# [p2gg_local] using sequential source momentum no. %2d = (%d, %d, %d)\n", iseq_source_momentum,
            g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);

          /***************************************************************************
           * loop on sequential source time slices
           ***************************************************************************/
          for ( int isequential_source_timeslice = 0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++) {

            g_sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];
            /* shift sequential source timeslice by source timeslice gsx[0] */
            int const g_shifted_sequential_source_timeslice = ( gsx[0] + g_sequential_source_timeslice + T_global ) % T_global;

            if( g_verbose > 2 && g_cart_id == 0) 
              fprintf(stdout, "# [p2gg_local] using sequential source timeslice %d / %d\n", g_sequential_source_timeslice, g_shifted_sequential_source_timeslice);

            /***************************************************************************
             * flavor-dependent sequential source momentum
             ***************************************************************************/
            int const seq_source_momentum[3] = { (1 - 2*iflavor) * g_seq_source_momentum[0],
                                                 (1 - 2*iflavor) * g_seq_source_momentum[1],
                                                 (1 - 2*iflavor) * g_seq_source_momentum[2] };

            if( g_verbose > 2 && g_cart_id == 0)
              fprintf(stdout, "# [p2gg_local] using flavor-dependent sequential source momentum (%d, %d, %d)\n",
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2]);

            /***************************************************************************
             ***************************************************************************
             **
             ** invert and contract for flavor X - after - X
             **
             ***************************************************************************
             ***************************************************************************/

            int sequential_source_gamma_id = -1;

#if _NEUTRAL_LVC_LVC_TENSOR

            /***************************************************************************
             * set sequential source gamma id
             ***************************************************************************/
            sequential_source_gamma_id = gamma_s;

            if( g_verbose > 2 && g_cart_id == 0) fprintf(stdout, "# [p2gg_local] using sequential source gamma id = %d\n", sequential_source_gamma_id);

            /***************************************************************************
             * prepare sequential source and sequential propagator
             ***************************************************************************/
            for( int is = 0; is < 12; is++ ) 
            {
              int const spinor_field_id     = iflavor * 12 + is;
              int const seq_spinor_field_id = 24 + is;

              memset ( spinor_work[0], 0, sizeof_spinor_field );
              memset ( spinor_work[1], 0, sizeof_spinor_field );

              exitstatus = init_sequential_source ( spinor_work[0], spinor_field[ spinor_field_id ],
                  g_shifted_sequential_source_timeslice, seq_source_momentum, sequential_source_gamma_id );

              if(exitstatus != 0) {
                fprintf(stderr, "[p2gg_local] Error from iinit_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(25);
              }

              /* full_spinor_work[1] = D^-1 full_spinor_work[0] */
              exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
              if(exitstatus < 0) {
                fprintf(stderr, "[p2gg_local] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(19);
              }

              /* check residuum */  
              if ( check_propagator_residual ) {
                check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
              }

              /* copy solution into place */
              memcpy ( spinor_field [ seq_spinor_field_id ], spinor_work[1], sizeof_spinor_field );

            }  /* end of loop on spin-color */

            /***************************************************************************
             * contraction for P - local - local tensor
             ***************************************************************************/
            /* flavor-dependent tag */
            sprintf ( tag, "/p-lvc-lvc/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d",
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                sequential_source_gamma_id, g_sequential_source_timeslice, iflavor );

            /***************************************************************************
             * contract
             *
             * Tr[ G_v ( X G_seq X ) G_v g5 Xbar^+ g5 ] = Tr[ G_v ( X G_seq X ) G_v X]
             ***************************************************************************/
            exitstatus = contract_local_local_2pt (
                &(spinor_field[ ( 1 - iflavor ) * 12]),
                &(spinor_field[24]),
                gamma_v_list, 4, gamma_v_list, 4,
                g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

            if( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(1);
            }

#endif  /* of if _NEUTRAL_LVC_LVC_TENSOR */

            /***************************************************************************/
            /***************************************************************************/

            /***************************************************************************
             ***************************************************************************
             **
             ** invert and contract for flavor Xbar - after - X
             **
             ** the upper is the old version; we want to try the following:
             **
             ** invert and contract for flavor X - after - Xbar
             **
             ** to keep the same flavor in the solver
             ***************************************************************************
             ***************************************************************************/
#if _CHARGED_LVC_LVC_TENSOR
            /***************************************************************************
             * set sequential source gamma id
             ***************************************************************************/
            sequential_source_gamma_id = gamma_p;
            if( g_verbose > 2 && g_cart_id == 0) fprintf(stdout, "# [p2gg_local] using sequential source gamma id = %d\n", sequential_source_gamma_id);

            for( int is = 0; is < 12; is++ ) 
            {
              int const spinor_field_id     = ( 1 - iflavor ) * 12 + is;
              
              int const seq_spinor_field_id = 24 + is;

              memset ( spinor_work[0], 0, sizeof_spinor_field);
              memset ( spinor_work[1], 0, sizeof_spinor_field);

               exitstatus = init_sequential_source ( spinor_work[0], spinor_field[ spinor_field_id ],
                  g_shifted_sequential_source_timeslice, seq_source_momentum, sequential_source_gamma_id );

              if(exitstatus != 0) {
                fprintf(stderr, "[p2gg_local] Error from init_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(25);
              }

              /* full_spinor_work[1] = D^-1 full_spinor_work[0] */
              exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
              if(exitstatus < 0) {
                fprintf(stderr, "[p2gg_local] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(19);
              }

              /* check residuum */  
              if ( check_propagator_residual ) {
                check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
              }

              /* copy solution into place */
              memcpy ( spinor_field [ seq_spinor_field_id ], spinor_work[1], sizeof_spinor_field );

            }  /* end of loop on spin-color */

            /***************************************************************************/
            /***************************************************************************/

            /***************************************************************************
             * contraction for P - local - local tensor
             ***************************************************************************/
            /* flavor-dependent AFF tag */
            sprintf ( tag, "/p-lvc-lvc/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d-%d_%d",
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                sequential_source_gamma_id, g_sequential_source_timeslice, iflavor, 1-iflavor, 1-iflavor );

            /***************************************************************************
             * contract
             *
             * Tr[ G_a ( Xbar G_seq X ) G_v g5 Xbar^+ g5 ] = Tr[ X G_a Xbar G_seq X G_v ]
             ***************************************************************************/
            exitstatus = contract_local_local_2pt (
                &(spinor_field[ iflavor * 12]),
                &(spinor_field[24]),
                gamma_a_list, 4, gamma_v_list, 4,
                g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

            if( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(1);
            }

            /* flavor-dependent AFF tag */
            sprintf ( tag, "/p-lvc-lvc/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d-%d_%d",
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                sequential_source_gamma_id, g_sequential_source_timeslice, iflavor, 1-iflavor, iflavor );

            /***************************************************************************
             * contract 
             * Tr[ G_v ( Xbar G_seq X ) G_a g5 X^+ g5 ] = Tr[ G_v Xbar G_seq X G_a Xbar ]
             ***************************************************************************/
            exitstatus = contract_local_local_2pt (
                &(spinor_field[ ( 1 - iflavor ) * 12]),
                &(spinor_field[24]),
                gamma_v_list, 4, gamma_a_list, 4,
                g_sink_momentum_list, g_sink_momentum_number, filename, tag, io_proc );

            if( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_local] Error from contract_local_local_2pt, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(1);
            }
#endif  /* of if _CHARGED_LVC_LVC_TENSOR */

          }  /* end of loop on sequential source timeslices */

      }  /* end of loop on sequential source momentum */

    }  /* end of loop on flavor */

  }  /* end of loop on source locations */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  fini_2level_dtable ( &spinor_field );
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
  show_time ( &start_time, &end_time, "p2gg_local", "runtime", g_cart_id == 0 );

  return(0);
}
