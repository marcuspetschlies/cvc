/****************************************************
 * twopt_invert_contract.c
 *
 * Wed Jul  5 14:08:25 CEST 2017
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
#include "contract_factorized.h"

#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_SP 2
#define _OP_ID_SM 3


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to 2-pt function inversion and contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "twopt";

  int c;
  int filename_set = 0;
  int gsx[4], sx[4];
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  unsigned int Vhalf;
  size_t sizeof_eo_spinor_field;
  size_t sizeof_spinor_field;
  double **eo_spinor_field=NULL;
  char filename[100];
  // double ratime, retime;
  double **lmzz[2], **lmzzinv[2];
  double **smzz[2], **smzzinv[2];
  double *gauge_field_with_phase = NULL;



#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char aff_tag[400];
#endif

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

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cpff.input");
  /* fprintf(stdout, "# [twopt_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [twopt_invert_contract] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
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

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [twopt_invert_contract] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [twopt_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [twopt_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[twopt_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[twopt_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [twopt_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [twopt_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[twopt_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[twopt_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[twopt_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************************
   * initialize clover, lmzz and lmzzinv
   ***********************************************************/
  exitstatus = init_clover ( &g_clover, &lmzz, &lmzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[twopt_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[twopt_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [twopt_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * allocate eo_spinor_field
   ***********************************************************/
  eo_spinor_field = init_2level_dtable ( 96, _GSI( Vhalf ) );
  if( eo_spinor_field == NULL ) {
    fprintf(stderr, "[twopt_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }
  
  /***********************************************************
   * loop on source locations
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
      fprintf(stderr, "[twopt_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

#ifdef HAVE_LHPC_AFF
    /***********************************************
     ***********************************************
     **
     ** writer for aff output file
     **
     ***********************************************
     ***********************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [twopt_invert_contract] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[twopt_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#endif

    /**********************************************************
     * propagators with source at gsx
     **********************************************************/

    /**********************************************************
     * l+ - type propagators
     **********************************************************/
    exitstatus = point_to_all_fermion_propagator_clover_full2eo ( &(eo_spinor_field[0]), &(eo_spinor_field[12]), _OP_ID_UP,
        gsx, gauge_field_with_phase, lmzz[0], lmzzinv[0], check_propagator_residual );

    if ( exitstatus != 0 ) {
      fprintf(stderr, "[twopt_invert_contract] Error from point_to_all_fermion_propagator_clover_full2eo status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(21);
    }

    /**********************************************************
     * l- - type propagators
     **********************************************************/
    exitstatus = point_to_all_fermion_propagator_clover_full2eo ( &(eo_spinor_field[24]), &(eo_spinor_field[36]), _OP_ID_DN,
        gsx, gauge_field_with_phase, lmzz[1], lmzzinv[1], check_propagator_residual );

    if ( exitstatus != 0 ) {
      fprintf(stderr, "[twopt_invert_contract] Error from point_to_all_fermion_propagator_clover_full2eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(21);
    }

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * local - local 2-point u-u
     ***************************************************************************/
    sprintf(aff_tag, "/twopt/l+l+/t%dx%dy%dz%d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[24]), &(eo_spinor_field[36]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       g_source_gamma_id_list, g_source_gamma_id_number,
       g_source_gamma_id_list, g_source_gamma_id_number,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[twopt_invert_contract] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }

    /***************************************************************************
     * local - local 2-point d-u
     ***************************************************************************/
    sprintf(aff_tag, "/twopt/l-l+/t%dx%dy%dz%d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
       g_source_gamma_id_list, g_source_gamma_id_number,
       g_source_gamma_id_list, g_source_gamma_id_number,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[twopt_invert_contract] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * st-type propagators
     *
     * fix: check_propagator_residual = 0 here; needs 
     *      2nd set of mzz and mzzinv for strange quark mass;
     *      either that or use the actual clover field in 
     *      application of Dirac operator
     ***************************************************************************/

    for ( int imass = 0; imass < g_twisted_masses_number; imass++ ) {

      if ( g_cart_id == 0 && g_verbose > 1 ) fprintf ( stdout, "# [twopt_invert_contract] using mass no. %d = %16.7f\n", imass, g_twisted_masses_list[imass] );

      /***************************************************************************
       * initialize clover, lmzz and lmzzinv
       ***************************************************************************/
      exitstatus = init_clover ( &g_clover, &smzz, &smzzinv, gauge_field_with_phase, g_twisted_masses_list[imass], g_csw );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[twopt_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      /***************************************************************************/
      /***************************************************************************/

      /***************************************************************************
       * s+ - type propagators
      /***************************************************************************/
      exitstatus = point_to_all_fermion_propagator_clover_full2eo ( &(eo_spinor_field[48]), &(eo_spinor_field[60]), _OP_ID_SP,
          gsx, gauge_field_with_phase, smzz[0], smzzinv[0], check_propagator_residual );

      if ( exitstatus != 0 ) {
        fprintf(stderr, "[twopt_invert_contract] Error from point_to_all_fermion_propagator_clover_full2eo status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(21);
      }

      /***************************************************************************
       * s- - type propagators
       ***************************************************************************/
      exitstatus = point_to_all_fermion_propagator_clover_full2eo ( &(eo_spinor_field[72]), &(eo_spinor_field[84]), _OP_ID_SM,
          gsx, gauge_field_with_phase, smzz[1], smzzinv[1], check_propagator_residual );

      if ( exitstatus != 0 ) {
        fprintf(stderr, "[twopt_invert_contract] Error from point_to_all_fermion_propagator_clover_full2eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(21);
      }

      /***************************************************************************/
      /***************************************************************************/

      /***************************************************************************
       * local - local 2-point s+ - u
       ***************************************************************************/
      sprintf(aff_tag, "/twopt/s+u+/t%dx%dy%dz%d/m%6.4f", gsx[0], gsx[1], gsx[2], gsx[3], g_twisted_masses_list[imass] );
      exitstatus = contract_local_local_2pt_eo (
         &(eo_spinor_field[72]), &(eo_spinor_field[84]),
         &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
         g_source_gamma_id_list, g_source_gamma_id_number,
         g_source_gamma_id_list, g_source_gamma_id_number,
         g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

      if( exitstatus != 0 ) {
        fprintf(stderr, "[twopt_invert_contract] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }

      /***************************************************************************
       * local - local 2-point s- - u
       ***************************************************************************/
      sprintf(aff_tag, "/twopt/s-l+/t%dx%dy%dz%d/m%6.4f", gsx[0], gsx[1], gsx[2], gsx[3], g_twisted_masses_list[imass] );
      exitstatus = contract_local_local_2pt_eo (
         &(eo_spinor_field[60]), &(eo_spinor_field[72]),
         &(eo_spinor_field[ 0]), &(eo_spinor_field[12]),
         g_source_gamma_id_list, g_source_gamma_id_number,
         g_source_gamma_id_list, g_source_gamma_id_number,
         g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

      if( exitstatus != 0 ) {
        fprintf(stderr, "[twopt_invert_contract] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }

      /***************************************************************************
       * local - local 2-point s- - s+
       ***************************************************************************/
      sprintf(aff_tag, "/twopt/s-s+/t%dx%dy%dz%d/m%6.4f", gsx[0], gsx[1], gsx[2], gsx[3], g_twisted_masses_list[imass] );
      exitstatus = contract_local_local_2pt_eo (
         &(eo_spinor_field[48]), &(eo_spinor_field[60]),
         &(eo_spinor_field[48]), &(eo_spinor_field[60]),
         g_source_gamma_id_list, g_source_gamma_id_number,
         g_source_gamma_id_list, g_source_gamma_id_number,
         g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

      if( exitstatus != 0 ) {
        fprintf(stderr, "[twopt_invert_contract] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }

      /***************************************************************************
       * local - local 2-point s+ - s+
       ***************************************************************************/
      sprintf(aff_tag, "/twopt/s-s+/t%dx%dy%dz%d/m%6.4f", gsx[0], gsx[1], gsx[2], gsx[3], g_twisted_masses_list[imass] );
      exitstatus = contract_local_local_2pt_eo (
         &(eo_spinor_field[72]), &(eo_spinor_field[84]),
         &(eo_spinor_field[48]), &(eo_spinor_field[60]),
         g_source_gamma_id_list, g_source_gamma_id_number,
         g_source_gamma_id_list, g_source_gamma_id_number,
         g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

      if( exitstatus != 0 ) {
        fprintf(stderr, "[twopt_invert_contract] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }

      /***************************************************************************/
      /***************************************************************************/

      fini_clover ( &smzz, &smzzinv );

      /***************************************************************************/
      /***************************************************************************/

      /***************************************************************************
       * omega baryon
       *
       * use smearing ?
       * can also use up, dn eo propagator space for reordering
       ***************************************************************************/

      double ** spinor_field = init_2level_dtable ( 24, _GSI((size_t)VOLUME) );
      if ( spinor_field == NULL ) {
        fprintf(stderr, "[twopt_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(1);
      }

      /* full volume s+ propagator */
      for ( int i = 0; i < 12; i++ ) {
        spinor_field_lexic2eo ( spinor_field[   i], eo_spinor_field[48+i], eo_spinor_field[60+i] );
      }
 
      /* full volume sm propagator */
      for ( int i = 0; i < 12; i++ ) {
        spinor_field_lexic2eo ( spinor_field[12+i], eo_spinor_field[72+i], eo_spinor_field[84+i] );
      }

      if( g_fermion_type == _TM_FERMION ) {
        spinor_field_tm_rotation ( spinor_field[ 0], spinor_field[ 0], +1, g_fermion_type, 12*VOLUME);

        spinor_field_tm_rotation ( spinor_field[12], spinor_field[12], -1, g_fermion_type, 12*VOLUME);
      }

      /* allocate propagator fields */
      fermion_propagator_type * fp  = create_fp_field ( VOLUME );
      fermion_propagator_type * fp2 = create_fp_field ( VOLUME );
      fermion_propagator_type * fp3 = create_fp_field ( VOLUME );

      /* sp propagator as propagator type field */
      assign_fermion_propagator_from_spinor_field ( fp,  &(spinor_field[ 0]), VOLUME);

      /* sm propagator as propagator type field */
      assign_fermion_propagator_from_spinor_field ( fp2, &(spinor_field[12]), VOLUME);

      double ** v2 = init_2level_dtable ( VOLUME, 32 );
      if ( v2 == NULL ) {
        fprintf(stderr, "[twopt_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }

      double *** vp = init_3level_dtable ( T, g_sink_momentum_number, 32 );
      if ( vp == NULL ) {
        fprintf(stderr, "[twopt_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      /* vertex list */
      int const    gamma_f1_number                    = 3;
      int const    gamma_f1_list[gamma_f1_number]     = { 9,  0,  7 };
      double const gamma_f1_src_sign[gamma_f1_number] = {-1, +1, +1 };
      double const gamma_f1_snk_sign[gamma_f1_number] = {+1, +1, -1 }; 


      /***********************************************************
       * contractions for Omega^- - Omega^-
       *   with s+ ( and s- ) propagator
       ***********************************************************/
      for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {

        /* fp2 <- fp x Gamma_i1 */
        fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp2, gamma_f1_list[if1], fp, VOLUME );
        fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp2, fp2, gamma_f1_src_sign[if1], VOLUME );

      for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

        /* fp3 <- Gamma_f1 x fp */
        fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp, VOLUME );
        fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp3, fp3, gamma_f1_snk_sign[if2], VOLUME );

        /***********************************************************/
        /***********************************************************/
  
        sprintf(aff_tag, "/omega-omega/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/d1",
            gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_list[if1], gamma_f1_list[if2]);

        exitstatus = contract_v5 ( v2, fp2, fp3, fp, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[twopt_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[twopt_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[twopt_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
          EXIT(49);
        }

        /***********************************************************/
        /***********************************************************/

        sprintf(aff_tag, "/omega-omega/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/d6",
            gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_list[if1], gamma_f1_list[if2]);

        exitstatus = contract_v6 ( v2, fp, fp2, fp3, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[twopt_invert_contract] Error from contract_v6, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[twopt_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[twopt_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
          EXIT(49);
        }

      }}  /* end of loop on i1, f1 vertex */

      free_fp_field ( &fp  );
      free_fp_field ( &fp2 );
      free_fp_field ( &fp3 );
      fini_2level_dtable ( &v2 );
      fini_3level_dtable ( &vp );
 
      fini_2level_dtable ( &spinor_field );

    }  /* end of loop on extra masses */

    /***************************************************************************/
    /***************************************************************************/

#ifdef HAVE_LHPC_AFF
    /***************************************************************************
     * close AFF writer
     ***************************************************************************/
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[twopt_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
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

  /* free clover matrix terms */
  fini_clover ( &lmzz, &lmzzinv );

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

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [twopt_invert_contract] %s# [twopt_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [twopt_invert_contract] %s# [twopt_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
