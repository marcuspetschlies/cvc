/****************************************************
 * p2gg_invert_contract_local
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
 * defines for tensors to be contracted
 ****************************************************/
#ifndef _NEUTRAL_CVC_LVC_TENSOR
#  define _NEUTRAL_CVC_LVC_TENSOR 1
#endif

#ifndef _NEUTRAL_LVC_LVC_TENSOR
#  define _NEUTRAL_LVC_LVC_TENSOR 1
#endif

#ifndef _CHARGED_LVC_LVC_TENSOR
#  define _CHARGED_LVC_LVC_TENSOR 1
#endif

#if _NEUTRAL_LVC_LVC_TENSOR
#warning "[p2gg_invert_contract_local] contract neutral lvc - lvc tensor"
#else
#warning "[p2gg_invert_contract_local] DO NOT contract neutral lvc - lvc tensor"
#endif

#if _NEUTRAL_CVC_LVC_TENSOR
#warning "[p2gg_invert_contract_local] contract neutral cvc - lvc tensor"
#else
#warning "[p2gg_invert_contract_local] DO NOT contract neutral cvc - lvc tensor"
#endif

#if _CHARGED_LVC_LVC_TENSOR
#warning "[p2gg_invert_contract_local]  contract chargd lvc - lvc tensor"
#else
#warning "[p2gg_invert_contract_local] DO NOT contract chargd lvc - lvc tensor"
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

  /*                            gt  gx  gy  gz */
  int const gamma_v_list[4] = {  0,  1,  2,  3 };
  int const gamma_v_num = 4;

  /*                            gtg5 gxg5 gyg5 gzg5 */
  int const gamma_a_list[4] = {    6,   7,   8,   9 };
  int const gamma_a_num = 4;

  /* vector, axial vector */
  /* int const gamma_va_list[8] = { 0,  1,  2,  3,  6,   7,   8,   9 }; */
  /* int const gamma_va_num = 8; */

  /*                             id  g5 */
  /* int const gamma_sp_list[2] = { 4  , 5 }; */
  /* int const gamma_sp_num = 2; */

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
  unsigned int Vhalf;
  size_t sizeof_eo_spinor_field;
  size_t sizeof_spinor_field;
  double **eo_spinor_field=NULL, **eo_spinor_work=NULL;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  int check_position_space_WI = 0;
  int first_solve_dummy = 1;
  struct timeval start_time, end_time;


#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char aff_tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "wch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'w':
      check_position_space_WI = 1;
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
  /* fprintf(stdout, "# [p2gg_invert_contract_local] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_invert_contract_local] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [p2gg_invert_contract_local] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_invert_contract_local] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_invert_contract_local] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_invert_contract_local] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_invert_contract_local] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  Vhalf                  = VOLUME / 2;
  sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);
  sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [p2gg_invert_contract_local] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg_invert_contract_local] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg_invert_contract_local] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg_invert_contract_local] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /*************************************************
   * allocate memory for eo spinor fields 
   * WITH HALO
   *************************************************/
  int const no_eo_fields = 6;
  eo_spinor_work  = init_2level_dtable ( (size_t)no_eo_fields, _GSI( (size_t)(VOLUME+RAND)/2) );
  if ( eo_spinor_work == NULL ) {
    fprintf(stderr, "[p2gg_invert_contract_local] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_invert_contract_local] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_invert_contract_local] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[p2gg_invert_contract_local] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_invert_contract_local] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [p2gg_invert_contract_local] proc%.4d has io proc id %d\n", g_cart_id, io_proc );



  /***********************************************************
   * allocate eo_spinor_field
   ***********************************************************/
  eo_spinor_field = init_2level_dtable ( 72, _GSI( (size_t)Vhalf));
  if( eo_spinor_field == NULL ) {
    fprintf(stderr, "[p2gg_invert_contract_local] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
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
      fprintf(stderr, "[p2gg_invert_contract_local] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
      EXIT( 50 );
    }

    double * full_spinor_work[2]  = { eo_spinor_work[0],  eo_spinor_work[2] };
    double * full_spinor_field[3] = { eo_spinor_field[0], eo_spinor_field[2], eo_spinor_field[4] };

    if( ( exitstatus = prepare_volume_source ( full_spinor_field[0], VOLUME ) ) != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(64);
    }

    spinor_field_lexic2eo ( full_spinor_field[0], eo_spinor_field[2], eo_spinor_field[3] );

    memset ( full_spinor_work[1], 0, sizeof_spinor_field);
    memcpy ( full_spinor_work[0], full_spinor_field[0], sizeof_spinor_field);

    /* full_spinor_work[1] = D^-1 full_spinor_work[0],
     * flavor id 0 
     */
    exitstatus = _TMLQCD_INVERT ( full_spinor_work[1], full_spinor_work[0], 0 );
    if(exitstatus < 0) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }

    /* full -> eo-precon
     * full_spinor_work[0] = eo_spinor_work[0,1] <- full_spinor_work[1]
     */
    spinor_field_lexic2eo ( full_spinor_work[1], eo_spinor_work[0], eo_spinor_work[1] );

    /* check residuum */
    exitstatus = check_residuum_eo (
        &( eo_spinor_field[2]), &(eo_spinor_field[3]),
        &( eo_spinor_work[0] ), &( eo_spinor_work[1]),
        gauge_field_with_phase, mzz[0], mzzinv[0], 1 );

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
      fprintf(stderr, "[p2gg_invert_contract_local] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***********************************************************
     * init Usource and source_proc_id
     *
     * NOTE: here it must be either 
     *         g_gauge_field with argument phase == co_phase_up
     *       or
     *         gauge_field_with_phase with argument phase == NULL 
     ***********************************************************/
    init_contract_cvc_tensor_usource( gauge_field_with_phase, gsx, NULL);

#ifdef HAVE_LHPC_AFF
    /***********************************************
     ***********************************************
     **
     ** writer for aff output file
     **
     ***********************************************
     ***********************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [p2gg_invert_contract_local] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_invert_contract_local] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#endif

    /**********************************************************
     **********************************************************
     **
     ** propagators with source at gsx
     **
     **********************************************************
     **********************************************************/

    /**********************************************************
     * up-type propagators
     **********************************************************/
    exitstatus = point_to_all_fermion_propagator_clover_full2eo ( &(eo_spinor_field[0]), &(eo_spinor_field[12]), _OP_ID_UP,
        gsx, gauge_field_with_phase, mzz[0], mzzinv[0], check_propagator_residual );

    if ( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_mixed] Error from point_to_all_fermion_propagator_clover_full2eo status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(21);
    }

    /**********************************************************
     * dn-type propagators
     **********************************************************/
    exitstatus = point_to_all_fermion_propagator_clover_full2eo ( &(eo_spinor_field[24]), &(eo_spinor_field[36]), _OP_ID_DN,
        gsx, gauge_field_with_phase, mzz[1], mzzinv[1], check_propagator_residual );

    if ( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_mixed] Error from point_to_all_fermion_propagator_clover_full2eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(21);
    }

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     *
     * local - local 2-point  u - u neutral
     *
     ***************************************************************************/
    /* AFF tag */
    sprintf(aff_tag, "/local-local/u-gf-u-gi/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );

    /***************************************************************************
     * contraction vector -vector
     ***************************************************************************/
#if _V_V_N
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_v_list, gamma_v_num, gamma_v_list, gamma_v_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_V_N */

    /***************************************************************************
     * contraction scalar - vector
     ***************************************************************************/
#if _S_V_N
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_s_list, gamma_s_num, gamma_v_list, gamma_v_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _S_V_N */

    /***************************************************************************
     * contraction vector - scalar
     ***************************************************************************/
#if _V_S_N
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_v_list, gamma_v_num, gamma_s_list, gamma_s_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_S_N */

    /***************************************************************************
     * contraction s - s
     ***************************************************************************/
#if _S_S_N
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_s_list, gamma_s_num, gamma_s_list, gamma_s_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_a_list, gamma_a_num, gamma_a_list, gamma_a_num,
       g_source_momentum_list, g_source_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _A_A_N */

    /***************************************************************************
     * contraction scalar - axial
     ***************************************************************************/
#if _S_A_N
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_s_list, gamma_s_num, gamma_a_list, gamma_a_num,
       g_source_momentum_list, g_source_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* _S_A_N */

    /***************************************************************************
     * contraction axial - scalar
     ***************************************************************************/
#if _A_S_N
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_a_list, gamma_a_num, gamma_s_list, gamma_s_num,
       g_source_momentum_list, g_source_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    /* AFF tag */
    sprintf(aff_tag, "/local-local/d-gf-u-gi/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );

    /***************************************************************************
     * contraction axial - axial
     ***************************************************************************/
#if _A_A_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_a_list, gamma_a_num, gamma_a_list, gamma_a_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _A_A_C */

    /***************************************************************************
     * contraction pseudoscalar - axial
     ***************************************************************************/
#if _P_A_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_p_list, gamma_p_num, gamma_a_list, gamma_a_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _P_A_C */

    /***************************************************************************
     * contraction axial - pseudoscalar
     ***************************************************************************/
#if _A_P_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_a_list, gamma_a_num, gamma_p_list, gamma_p_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _A_P_C */

    /***************************************************************************
     * contraction pseudoscalar - pseudoscalar
     ***************************************************************************/
#if _P_P_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_p_list, gamma_p_num, gamma_p_list, gamma_p_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _P_P_C */

    /***************************************************************************
     * different set of momenta here
     ***************************************************************************/

    /***************************************************************************
     * contraction vector - vector
     ***************************************************************************/
#if _V_V_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_v_list, gamma_v_num, gamma_v_list, gamma_v_num,
       g_source_momentum_list, g_source_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_V_C */

    /***************************************************************************
     * contraction vector - pseudoscalar
     ***************************************************************************/
#if _V_P_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_v_list, gamma_v_num, gamma_p_list, gamma_p_num,
       g_source_momentum_list, g_source_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_P_C */

    /***************************************************************************
     * contraction vector - vector
     ***************************************************************************/
#if _P_V_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       gamma_p_list, gamma_p_num, gamma_v_list, gamma_v_num,
       g_source_momentum_list, g_source_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _P_V_C */

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     ***************************************************************************
     **
     ** local - local 2-point  u - d charged
     **
     ***************************************************************************
     ***************************************************************************/
    /* AFF tag */
    sprintf(aff_tag, "/local-local/u-gf-d-gi/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );

    /***************************************************************************
     * contraction axial - axial
     ***************************************************************************/
#if _A_A_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       gamma_a_list, gamma_a_num, gamma_a_list, gamma_a_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _A_A_C */

    /***************************************************************************
     * contraction pseudoscalar - axial
     ***************************************************************************/
#if _P_A_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       gamma_p_list, gamma_p_num, gamma_a_list, gamma_a_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif /* of if _P_A_C */

    /***************************************************************************
     * contraction axial - pseudoscalar
     ***************************************************************************/
#if _A_P_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       gamma_a_list, gamma_a_num, gamma_p_list, gamma_p_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _A_P_C  */

    /***************************************************************************
     * contraction pseudoscalar - pseudoscalar
     ***************************************************************************/
#if _P_P_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       gamma_p_list, gamma_p_num, gamma_p_list, gamma_p_num,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _P_P_C */

    /***************************************************************************
     * different set of momenta here
     ***************************************************************************/

    /***************************************************************************
     * contraction vector - vector
     ***************************************************************************/
#if _V_V_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       gamma_v_list, gamma_v_num, gamma_v_list, gamma_v_num,
       g_source_momentum_list, g_source_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_V_C */

    /***************************************************************************
     * contraction vector - pseudoscalar
     ***************************************************************************/
#if _V_P_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       gamma_v_list, gamma_v_num, gamma_p_list, gamma_p_num,
       g_source_momentum_list, g_source_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _V_P_C */

    /***************************************************************************
     * contraction pseudoscalar - vector
     ***************************************************************************/
#if _P_V_C
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       gamma_p_list, gamma_p_num, gamma_v_list, gamma_v_num,
       g_source_momentum_list, g_source_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
#endif  /* of if _P_V_C */

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * mixed hvp tensor
     *
     * sink   operator --- conserved vector current
     * source operator --- local vector current
     ***************************************************************************/
#if _NEUTRAL_CVC_LVC_TENSOR
    double ** cl_tensor_eo = init_2level_dtable ( 2, 32 * (size_t)Vhalf );
    if( cl_tensor_eo == NULL ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(24);
    }

    contract_cvc_local_tensor_eo ( cl_tensor_eo[0], cl_tensor_eo[1],
        &(eo_spinor_field[24]), &(eo_spinor_field[36]), &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
        gauge_field_with_phase );

    double *** cvc_tp = init_3level_dtable ( g_sink_momentum_number, 16, 2*T);
    if ( cvc_tp == NULL ) {
      fprintf ( stderr, "[p2gg_invert_contract_local] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }

    exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cl_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(26);
    }
    /* write results to file */
    sprintf(aff_tag, "/hvp/u-cvc-u-lvc/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = cvc_tensor_tp_write_to_aff_file ( cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }
    fini_3level_dtable ( &cvc_tp );

    /* check position space WI */
    if(check_position_space_WI) {
      if( g_cart_id == 0 && g_verbose > 0 ) fprintf ( stdout, "# [p2gg_invert_contract_local] check position space WI for cvc-lcv tensor %s %d\n", __FILE__, __LINE__ );
      exitstatus = cvc_tensor_eo_check_wi_position_space ( cl_tensor_eo );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_invert_contract_local] Error from cvc_tensor_eo_check_wi_position_space for mixed, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }
    }

    fini_2level_dtable ( &cl_tensor_eo );

#endif  /* of if _NEUTRAL_CVC_LVC_TENSOR */

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
    /* for( int iflavor = 0; iflavor <= 1; iflavor++ ) */
    for( int iflavor = 1; iflavor >= 0; iflavor-- )
    {

      /***************************************************************************
       * loop on sequential source gamma matrices
       ***************************************************************************/
      for ( int iseq_source_momentum = 0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) {

        g_seq_source_momentum[0] = g_seq_source_momentum_list[iseq_source_momentum][0];
        g_seq_source_momentum[1] = g_seq_source_momentum_list[iseq_source_momentum][1];
        g_seq_source_momentum[2] = g_seq_source_momentum_list[iseq_source_momentum][2];

        if( g_verbose > 2 && g_cart_id == 0) fprintf(stdout, "# [p2gg_invert_contract_local] using sequential source momentum no. %2d = (%d, %d, %d)\n", iseq_source_momentum,
            g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);

        /***************************************************************************
         * loop on sequential source gamma matrices
         ***************************************************************************/
/*
        for( int isequential_source_gamma_id = 0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++) {

          int const sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];
          if( g_verbose > 2 && g_cart_id == 0) fprintf(stdout, "# [p2gg_invert_contract_local] using sequential source gamma id no. %2d = %d\n",
              isequential_source_gamma_id, sequential_source_gamma_id);
*/
          /***************************************************************************
           * loop on sequential source time slices
           ***************************************************************************/
          for ( int isequential_source_timeslice = 0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++) {

            g_sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];
            /* shift sequential source timeslice by source timeslice gsx[0] */
            int const g_shifted_sequential_source_timeslice = ( gsx[0] + g_sequential_source_timeslice + T_global ) % T_global;

            if( g_verbose > 2 && g_cart_id == 0) 
              fprintf(stdout, "# [p2gg_invert_contract_local] using sequential source timeslice %d / %d\n", g_sequential_source_timeslice, g_shifted_sequential_source_timeslice);

            /***************************************************************************
             * flavor-dependent sequential source momentum
             ***************************************************************************/
            int const seq_source_momentum[3] = { (1 - 2*iflavor) * g_seq_source_momentum[0],
                                                 (1 - 2*iflavor) * g_seq_source_momentum[1],
                                                 (1 - 2*iflavor) * g_seq_source_momentum[2] };

            if( g_verbose > 2 && g_cart_id == 0)
              fprintf(stdout, "# [p2gg_invert_contract_local] using flavor-dependent sequential source momentum (%d, %d, %d)\n",
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2]);

            /***************************************************************************
             ***************************************************************************
             **
             ** invert and contract for flavor X - after - X
             **
             ***************************************************************************
             ***************************************************************************/

            int sequential_source_gamma_id = -1;

#if _NEUTRAL_LVC_LVC_TENSOR || _NEUTRAL_CVC_LVC_TENSOR

            /***************************************************************************
             * set sequential source gamma id
             ***************************************************************************/
            sequential_source_gamma_id = gamma_s;

            if( g_verbose > 2 && g_cart_id == 0) fprintf(stdout, "# [p2gg_invert_contract_local] using sequential source gamma id = %d\n", sequential_source_gamma_id);

            /***************************************************************************
             * prepare sequential source and sequential propagator
             ***************************************************************************/
            for( int is = 0; is < 12; is++ ) 
            {
              int eo_spinor_field_id_e     = iflavor * 24 + is;
              int eo_spinor_field_id_o     = eo_spinor_field_id_e + 12;
              int eo_seq_spinor_field_id_e = 48 + is;
              int eo_seq_spinor_field_id_o = eo_seq_spinor_field_id_e + 12;

              exitstatus = init_clover_eo_sequential_source(
                  eo_spinor_field[ eo_seq_spinor_field_id_e ], eo_spinor_field[ eo_seq_spinor_field_id_o ],
                  eo_spinor_field[ eo_spinor_field_id_e     ], eo_spinor_field[ eo_spinor_field_id_o     ] ,
                  g_shifted_sequential_source_timeslice, gauge_field_with_phase, mzzinv[iflavor][0],
                  seq_source_momentum, sequential_source_gamma_id, eo_spinor_work[0]);
              if(exitstatus != 0) {
                fprintf(stderr, "[p2gg_invert_contract_local] Error from init_clover_eo_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(25);
              }

              if ( g_write_sequential_source ) {
                double * spinor_field_write = init_1level_dtable ( _GSI(VOLUME) );
                int const isc = is % 12;
                int const imu = is / 12;
                int shift[4] = {0,0,0,0};
                if ( imu < 4 ) shift[imu]++;

                sprintf ( filename, "source.%.4d.t%dx%dy%dz%d.t%d.g%d.px%dpy%dpz%d.fl%d.%d", Nconf, 
                    (gsx[0]+shift[0])%T_global,
                    (gsx[1]+shift[1])%LX_global,
                    (gsx[2]+shift[2])%LY_global,
                    (gsx[3]+shift[3])%LZ_global,
                    g_shifted_sequential_source_timeslice, sequential_source_gamma_id,
                    seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], iflavor, isc );

                spinor_field_eo2lexic ( spinor_field_write, eo_spinor_field[ eo_seq_spinor_field_id_e ], eo_spinor_field[ eo_seq_spinor_field_id_o ] );

                if ( ( exitstatus = write_propagator( spinor_field_write, filename, 0, g_propagator_precision) ) != 0 ) {
                  fprintf(stderr, "[p2gg_invert_contract_local] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(2);
                }
              }

              double *full_spinor_work[2] = { eo_spinor_work[0], eo_spinor_work[2] };

              memset ( full_spinor_work[1], 0, sizeof_spinor_field);
              /* eo-precon -> full */
              spinor_field_eo2lexic ( full_spinor_work[0], eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_field[eo_seq_spinor_field_id_o] );

              /* full_spinor_work[1] = D^-1 full_spinor_work[0] */
              exitstatus = _TMLQCD_INVERT ( full_spinor_work[1], full_spinor_work[0], iflavor );
              if(exitstatus < 0) {
                fprintf(stderr, "[p2gg_invert_contract_local] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(19);
              }

              /* full -> eo-precon 
               * full_spinor_work[0] = eo_spinor_work[0,1] <- full_spinor_work[1]
               * */
              spinor_field_lexic2eo ( full_spinor_work[1], eo_spinor_work[0], eo_spinor_work[1] );
              
              /* check residuum */  
              if ( check_propagator_residual ) {
                exitstatus = check_residuum_eo ( 
                    &( eo_spinor_field[eo_seq_spinor_field_id_e]), &(eo_spinor_field[eo_seq_spinor_field_id_o]),
                    &( eo_spinor_work[0] ),                        &( eo_spinor_work[1] ),
                    gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
              }
 
              /* copy solution into place */
              memcpy ( eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_work[0], sizeof_eo_spinor_field );
              memcpy ( eo_spinor_field[eo_seq_spinor_field_id_o], eo_spinor_work[1], sizeof_eo_spinor_field );

            }  /* end of loop on spin-color */

#if _NEUTRAL_LVC_LVC_TENSOR
            /***************************************************************************
             * contraction for P - local - local tensor
             ***************************************************************************/
            /* flavor-dependent AFF tag */
            sprintf(aff_tag, "/p-lvc-lvc/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d", gsx[0], gsx[1], gsx[2], gsx[3],
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                sequential_source_gamma_id, g_sequential_source_timeslice, iflavor );

            /***************************************************************************
             * contract
             *
             * Tr[ G_v ( X G_seq X ) G_v g5 Xbar^+ g5 ] = Tr[ G_v ( X G_seq X ) G_v X]
             ***************************************************************************/
            exitstatus = contract_local_local_2pt_eo (
                &(eo_spinor_field[ ( 1 - iflavor ) * 24]), &(eo_spinor_field[ ( 1 - iflavor ) * 24 + 12]),
                &(eo_spinor_field[48]), &(eo_spinor_field[60]),
                gamma_v_list, 4, gamma_v_list, 4,
                g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

            if( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(1);
            }
#endif  /* of if _NEUTRAL_LVC_LVC_TENSOR */

#if _NEUTRAL_CVC_LVC_TENSOR
            /***************************************************************************
             * contract for mixed P - cvc - lvc tensor
             ***************************************************************************/
            /* flavor-dependent aff tag  */
            sprintf(aff_tag, "/p-cvc-lvc/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d",
                gsx[0], gsx[1], gsx[2], gsx[3], 
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                sequential_source_gamma_id, g_sequential_source_timeslice, iflavor );

            double ** cl_tensor_eo = init_2level_dtable ( 2, 32 * (size_t)Vhalf);
            if( cl_tensor_eo == NULL ) {
              fprintf(stderr, "[p2gg_invert_contract_local] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
              EXIT(24);
            }
             /* contraction for P - cvc - lvc tensor */
            contract_cvc_local_tensor_eo ( cl_tensor_eo[0], cl_tensor_eo[1],
                &(eo_spinor_field[ ( 1 - iflavor ) * 24]), &(eo_spinor_field[ ( 1 - iflavor ) * 24 + 12]), &(eo_spinor_field[48]), &(eo_spinor_field[60]),
                gauge_field_with_phase );

               /* momentum projections */
            cvc_tp = init_3level_dtable ( g_sink_momentum_number, 16, 2*T);
            if ( cvc_tp == NULL ) {
              fprintf ( stderr, "[p2gg_invert_contract_local] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
              EXIT(12);
            }

            exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cl_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg_invert_contract_local] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(26);
            }

            /* write results to file */
            exitstatus = cvc_tensor_tp_write_to_aff_file ( cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if(exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_invert_contract_local] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(45);
            }
            fini_3level_dtable ( &cvc_tp );

            /* check position space WI */
            if(check_position_space_WI) {
              if( g_cart_id == 0 && g_verbose > 0 ) fprintf ( stdout, "# [p2gg_invert_contract_local] check position space WI for p-cvc-lvc tensor fl %d %s %d\n",
                 iflavor, __FILE__, __LINE__ );
              exitstatus = cvc_tensor_eo_check_wi_position_space ( cl_tensor_eo );
              if(exitstatus != 0) {
                fprintf(stderr, "[p2gg_invert_contract_local] Error from cvc_tensor_eo_check_wi_position_space for full, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(38);
              }
            }

            fini_2level_dtable ( &cl_tensor_eo );

#endif  /* of if _NEUTRAL_CVC_LVC_TENSOR */

#endif  /* of if _NEUTRAL_LVC_LVC_TENSOR || _NEUTRAL_CVC_LVC_TENSOR */

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
            if( g_verbose > 2 && g_cart_id == 0) fprintf(stdout, "# [p2gg_invert_contract_local] using sequential source gamma id = %d\n", sequential_source_gamma_id);

            for( int is = 0; is < 12; is++ ) 
            {
              int eo_spinor_field_id_e     = ( 1 - iflavor ) * 24 + is;
              int eo_spinor_field_id_o     = eo_spinor_field_id_e + 12;
              int eo_seq_spinor_field_id_e = 48 + is;
              int eo_seq_spinor_field_id_o = eo_seq_spinor_field_id_e + 12;

              exitstatus = init_clover_eo_sequential_source(
                  eo_spinor_field[ eo_seq_spinor_field_id_e ], eo_spinor_field[ eo_seq_spinor_field_id_o ],
                  eo_spinor_field[ eo_spinor_field_id_e     ], eo_spinor_field[ eo_spinor_field_id_o     ] ,
                  g_shifted_sequential_source_timeslice, gauge_field_with_phase, mzzinv[1-iflavor][0],
                  seq_source_momentum, sequential_source_gamma_id, eo_spinor_work[0]);

              if(exitstatus != 0) {
                fprintf(stderr, "[p2gg_invert_contract_local] Error from init_clover_eo_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(25);
              }

              double *full_spinor_work[2] = { eo_spinor_work[0], eo_spinor_work[2] };

              memset ( full_spinor_work[1], 0, sizeof_spinor_field);
              /* eo-precon -> full */
              spinor_field_eo2lexic ( full_spinor_work[0], eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_field[eo_seq_spinor_field_id_o] );

              /* full_spinor_work[1] = D^-1 full_spinor_work[0] */
              exitstatus = _TMLQCD_INVERT ( full_spinor_work[1], full_spinor_work[0], iflavor );
              if(exitstatus < 0) {
                fprintf(stderr, "[p2gg_invert_contract_local] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(19);
              }

              /* full -> eo-precon 
               * full_spinor_work[0] = eo_spinor_work[0,1] <- full_spinor_work[1]
               * */
              spinor_field_lexic2eo ( full_spinor_work[1], eo_spinor_work[0], eo_spinor_work[1] );
              
              /* check residuum */  
              if ( check_propagator_residual ) {
                exitstatus = check_residuum_eo ( 
                    &( eo_spinor_field[eo_seq_spinor_field_id_e]), &(eo_spinor_field[eo_seq_spinor_field_id_o]),
                    &( eo_spinor_work[0] ),                        &( eo_spinor_work[1] ),
                    gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
              }
 
              /* copy solution into place */
              memcpy ( eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_work[0], sizeof_eo_spinor_field );
              memcpy ( eo_spinor_field[eo_seq_spinor_field_id_o], eo_spinor_work[1], sizeof_eo_spinor_field );

            }  /* end of loop on spin-color */

            /***************************************************************************/
            /***************************************************************************/

            /***************************************************************************
             * contraction for P - local - local tensor
             ***************************************************************************/
            /* flavor-dependent AFF tag */
            sprintf(aff_tag, "/p-lvc-lvc/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d-%d_%d", gsx[0], gsx[1], gsx[2], gsx[3],
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                sequential_source_gamma_id, g_sequential_source_timeslice, iflavor, 1-iflavor, 1-iflavor );

            /***************************************************************************
             * contract
             *
             * Tr[ G_a ( Xbar G_seq X ) G_v g5 Xbar^+ g5 ] = Tr[ X G_a Xbar G_seq X G_v ]
             ***************************************************************************/
            exitstatus = contract_local_local_2pt_eo (
                &(eo_spinor_field[ iflavor * 24]), &(eo_spinor_field[ iflavor * 24 + 12]),
                &(eo_spinor_field[48]), &(eo_spinor_field[60]),
                gamma_a_list, 4, gamma_v_list, 4,
                g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

            if( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(1);
            }

            /* flavor-dependent AFF tag */
            sprintf(aff_tag, "/p-lvc-lvc/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d-%d_%d", gsx[0], gsx[1], gsx[2], gsx[3],
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                sequential_source_gamma_id, g_sequential_source_timeslice, iflavor, 1-iflavor, iflavor );

            /***************************************************************************
             * contract 
             * Tr[ G_v ( Xbar G_seq X ) G_a g5 X^+ g5 ] = Tr[ G_v Xbar G_seq X G_a Xbar ]
             ***************************************************************************/
            exitstatus = contract_local_local_2pt_eo (
                &(eo_spinor_field[ ( 1 - iflavor ) * 24]), &(eo_spinor_field[ ( 1 - iflavor ) * 24 + 12]),
                &(eo_spinor_field[48]), &(eo_spinor_field[60]),
                gamma_v_list, 4, gamma_a_list, 4,
                g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

            if( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_invert_contract_local] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(1);
            }
#endif  /* of if _CHARGED_LVC_LVC_TENSOR */

          }  /* end of loop on sequential source timeslices */

/*        } */  /* end of loop on sequential source gamma id */
      }  /* end of loop on sequential source momentum */
    }  /* end of loop on flavor */


    /***************************************************************************/
    /***************************************************************************/

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_invert_contract_local] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  }  /* end of loop on source locations */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  fini_2level_dtable ( &eo_spinor_field );
  fini_2level_dtable ( &eo_spinor_work );

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
  show_time ( &start_time, &end_time, "p2gg_invert_contract_local", "runtime", g_cart_id == 0 );

  return(0);
}
