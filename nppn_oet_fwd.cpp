/****************************************************
 * nppn_oet_fwd
 * 
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
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
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "contractions_io.h"
#include "matrix_init.h"
#include "project.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "contract_factorized.h"
#include "clover.h"
#include "dummy_solver.h"
#include "scalar_products.h"

using namespace cvc;


/***********************************************************
 * usage function
 ***********************************************************/
void usage() {
  fprintf(stdout, "Code to perform contractions for nppn 4-pt. function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -h? this help\n");
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}
  
  
/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
  
  const int max_num_diagram = 6;


  int c;
  int filename_set                   = 0;
  int exitstatus;
  int op_id_up= -1, op_id_dn         = -1;
  int source_proc_id                 = 0;
  int read_forward_propagator        = 0;
  int read_sequential_propagator     = 0;
  int check_propagator_residual      = 0;

/***********************************************************/
/***********************************************************/

  char filename[200];
  double ratime, retime;
#ifndef HAVE_TMLQCD_LIBWRAPPER
  double plaq_r = 0.;
#endif
  double *gauge_field_smeared = NULL, *gauge_field_with_phase = NULL;
  double **mzz[2], **mzzinv[2];

/*******************************************************************
 * Gamma components for the piN and Delta:
 *                                                                 */
  /* vertex i2, gamma_5 only */
  const int gamma_i2_number             = 2;
  int gamma_i2_list[gamma_i2_number]    = {  5,  4 };
  double gamma_i2_sign[gamma_i2_number] = { +1, +1 };

  /* vertex f2, gamma_5 and id,  vector indices and pseudo-vector */
  const int gamma_f2_number                        = 2;
  int gamma_f2_list[gamma_f2_number]               = {  5,  4 };
  double gamma_f2_sign[gamma_f2_number]            = { +1, +1 };

  /* vertex f1 for nucleon-type, C g5, C, C g0 g5, C g0 */
  const int gamma_f1_nucleon_number                                = 1;
  int gamma_f1_nucleon_list[gamma_f1_nucleon_number]               = { 14 };
  double gamma_f1_nucleon_sign[gamma_f1_nucleon_number]            = { +1 };

/*
 *******************************************************************/

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
  char aff_tag[200];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "pqch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      fprintf(stdout, "# [nppn_oet_fwd] will check_propagator_residual\n");
      break;
    case 'p':
      read_forward_propagator = 1;
      fprintf(stdout, "# [nppn_oet_fwd] will read forward propagator\n");
      break;
    case 'q':
      read_sequential_propagator = 1;
      fprintf(stdout, "# [nppn_oet_fwd] will read sequential propagator\n");
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# reading input from file %s\n", filename);
  read_input_parser(filename);

  if(g_fermion_type == -1 ) {
    fprintf(stderr, "# [nppn_oet_fwd] fermion_type must be set\n");
    exit(1);
  } else {
    fprintf(stdout, "# [nppn_oet_fwd] using fermion type %d\n", g_fermion_type);
  }

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [nppn_oet_fwd] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    EXIT(14);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(15);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    EXIT(16);
  }
#endif

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[nppn_oet_fwd] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [nppn_oet_fwd] git version = %s\n", g_gitversion);
  }

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_geometry\n");
    EXIT(1);
  }
  geometry();

#ifdef HAVE_MPI
  if ( check_propagator_residual ) {
    mpi_init_xchange_eo_spinor();
  }
#endif

  unsigned int const VOL3 = LX*LY*LZ;
  size_t const sizeof_spinor_field           = _GSI(VOLUME) * sizeof(double);
  size_t const sizeof_spinor_field_timeslice = _GSI(VOL3)   * sizeof(double);


  int const io_proc = get_io_proc ();

  /***********************************************
   * read the gauge field or obtain from tmLQCD
   ***********************************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
#ifndef HAVE_TMLQCD_LIBWRAPPER
  switch(g_gauge_file_format) {
    case 0:
      sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
      exitstatus = read_lime_gauge_field_doubleprec(filename);
      break;
    case 1:
      sprintf(filename, "%s.%.5d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "\n# [nppn_oet_fwd] reading gauge field from file %s\n", filename);
      exitstatus = read_nersc_gauge_field(g_gauge_field, filename, &plaq_r);
      break;
  }
  if(exitstatus != 0) {
    fprintf(stderr, "[nppn_oet_fwd] Error, could not read gauge field\n");
    EXIT(21);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[nppn_oet_fwd] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[nppn_oet_fwd] Error, g_gauge_field is NULL\n");
    EXIT(5);
  }
#endif

#ifdef HAVE_MPI
  xchange_gauge_field ( g_gauge_field );
#endif
  /* measure the plaquette */
  
  if ( ( exitstatus = plaquetteria  ( g_gauge_field ) ) != 0 ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from plaquetteria, status was %d\n", exitstatus);
    EXIT(2);
  }

  /***********************************************
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {

    alloc_gauge_field(&gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy(gauge_field_smeared, tmLQCD_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[nppn_oet_fwd] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
    }  /* end of if N_aoe > 0 */
  }  /* end of if N_Jacobi > 0 */

  if ( check_propagator_residual ) {
    /***********************************************************
     * multiply the phase to the gauge field
     ***********************************************************/
    exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, tmLQCD_gauge_field, co_phase_up );
    if(exitstatus != 0) {
      fprintf(stderr, "[nppn_oet_fwd] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }
  
    /***********************************************
     * initialize clover, mzz and mzz_inv
     ***********************************************/
    exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[nppn_oet_fwd] Error from init_clover, status was %d\n", exitstatus );
      EXIT(1);
    }
  
  }  /* end of if check propagator residual */

  /***********************************************************
   * set operator ids depending on fermion type
   ***********************************************************/
  if(g_fermion_type == _TM_FERMION) {
    op_id_up = 0;
    op_id_dn = 1;
  } else if(g_fermion_type == _WILSON_FERMION) {
    op_id_up = 0;
    op_id_dn = 0;
  }

  /******************************************************
   * check source coords list
   ******************************************************/
  for ( int i = 0; i < g_source_location_number; i++ ) {
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] + T_global ) % T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * stochastic oet 
   * each timeslice, 4 for oet, 2 flavors
   ***********************************************************/
  double *** stochastic_source_oet = init_2level_dtable ( 4, _GSI(VOLUME) );
  if( stochastic_source_oet == NULL ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__ );
    EXIT(44);
  }
  double **** stochastic_propagator_oet = init_4level_dtable ( T_global, 4, 2, _GSI(VOLUME) );
  if( stochastic_propagator_oet == NULL ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__ );
    EXIT(44);
  }

  /*****************************************************************
   * initialize rng state
   *****************************************************************/
  int * rng_state = NULL;
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  if ( g_verbose > 4 ) {
    for ( int i = 0; i < rlxd_size(); i++ ) {
      fprintf ( stdout, "rng %2d %10d\n", g_cart_id, rng_state[i] );
    }
  }

  if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gts, NULL, spin_dilution, color_dilution, 1 ) ) != 0 ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(64);
  }

  /*****************************************************************
   * PART Ia stochastic propagators
   *****************************************************************/

  int source_momentum[3] = { 0, 0, 0 };

  /***************************************************************************
   * loop on global timeslices
   ***************************************************************************/
  for ( int gts = 0; gts < T_global; gts++ ) {

    /***************************************************************************
     * init every time, i.e. final argument is 1
     ***************************************************************************/
    exitstatus = init_timeslice_source_oet ( stochastic_source_oet, gts, source_momentum, 4, 1, 1 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[nppn_oet_fwd] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(64);
    }
  }
  
  /***************************************************************************
   * SOURCE SMEARING
   ***************************************************************************/
  for ( int i = 0; i < 4; i++ ) {

    exitstatus = Jacobi_Smearing ( gauge_field_smeared, stochastic_source_oet[i], N_Jacobi, kappa_Jacobi);
    if(exitstatus != 0) {
      fprintf(stderr, "[nppn_oet_fwd] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(11);
    }

  }

  /***************************************************************************
   * solve DE with smeared oet sources
   * for up-type and dn-type
   ***************************************************************************/
  for ( int iflavor = 0; iflavor <= 1; iflavor ++ ) {
  
    for ( int gts = 0; gts < T_global; gts++ ) {

      double ** spinor_work  = init_2level_dtable ( 3, _GSI( VOLUME+RAND ) );
      if ( spinor_work == NULL ) {
        fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }

      int source_timeslice = -1;
      int source_proc_id   = -1;

      exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[nppn_oet_fwd] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(123);
      }

      for( int i = 0; i < 4; i++) {

        /***************************************************************************
         * source process copy source timeslice to [0]
         ***************************************************************************/
        memset ( spinor_work[0], 0, sizeof_spinor_field );
        if ( source_proc_id == g_cart_id ) memcpy ( spinor_work[0] + source_timeslice * _GSI(VOL3), stochastic_source_oet[i] + source_timeslice * _GSI(VOL3), sizeof_spinor_field_timeslice );

        /***************************************************************************
         * [1] init to zero
         ***************************************************************************/
        memset ( spinor_work[1], 0, sizeof_spinor_field );

        /***************************************************************************
         * tm-rotate to twisted basis
         ***************************************************************************/
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], 1-2*iflavor, g_fermion_type, VOLUME);
        }

        /***************************************************************************
         * copy source for residual check
         ***************************************************************************/
        memcpy ( spinor_work[2] , spinor_work[0], sizeof_spinor_field );

        /***************************************************************************
         * call solver
         * [1] = D_flavor^-1 [0]
         ***************************************************************************/
        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
        if(exitstatus < 0) {
          fprintf(stderr, "[nppn_oet_fwd] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }

        /***************************************************************************
         * check residual with original source
         ***************************************************************************/
        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[2]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
        }

        /***************************************************************************
         * tm-rotate to physical basis
         ***************************************************************************/
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation ( spinor_work[1], spinor_work[1], 1-2*iflavor, g_fermion_type, VOLUME);
        }

        /***************************************************************************
         * SINK SMEARING
         ***************************************************************************/
        exitstatus = Jacobi_Smearing ( gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);
        if(exitstatus != 0) {
          fprintf(stderr, "[nppn_oet_fwd] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(11);
        }

        /***************************************************************************
         * store propagator
         ***************************************************************************/
        memcpy( stochastic_propagator_oet[gts][iflavor][i], spinor_work[1], sizeof_spinor_field );

      }  /* end of loop on spin color dilution indices */

    }  /* end of loop on timeslices */

  }  /* end of loop flavor */

  /*****************************************************************
   * PART Ib up-type, dn-type propagator
   *****************************************************************/

  double *** propagator_list = init_3level_dtable ( 2, 12, _GSI(VOLUME) );
  if( propagator_list == NULL ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__ );
    EXIT(44);
  }

  /***********************************************************/
  /***********************************************************/

  /* loop on source locations */
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {

    int const gsx[4] = {
      g_source_coords_list[i_src][0],
      g_source_coords_list[i_src][1],
      g_source_coords_list[i_src][2],
      g_source_coords_list[i_src][3] };

    if(io_proc == 2) {
      sprintf(filename, "%s.%d.t%dx%dy%dz%d.aff", "nppn", Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [nppn_oet_fwd] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[nppn_oet_fwd] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      }
    }  /* end of if io_proc == 2 */

    int sx[4], source_proc_id;
    get_point_source_info (gsx, sx, &source_proc_id);

    /***********************************************************
     * point-to-all propagator
     *
     * RETURNED FIELD IS IN PHYSICAL BASIS
     ***********************************************************/
    for ( int iflavor = 0; iflavor <= 1; iflavor ++ ) {
      int const op_id = iflavor;

      exitstatus = point_source_propagator ( propagator_list[iflavor], gsx, op_id, 1, 1, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, mzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[nppn_oet_fwd] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(12);
      }
    }  /* end of loop on flavor */

    /*****************************************************************/
    /*****************************************************************/

    /*****************************************************************
     * PART II N-N
     *****************************************************************/
    for ( int iflavor = 0; iflavor <= 1; iflavor ++ ) {

      fermion_propagator_type * fp  = create_fp_field ( VOLUME );
      fermion_propagator_type * fp2 = create_fp_field ( VOLUME );
      fermion_propagator_type * fp3 = create_fp_field ( VOLUME );

      /* up propagator as propagator type field */
      assign_fermion_propagator_from_spinor_field ( fp,  propagator_list[iflavor], VOLUME);

      /* dn propagator as propagator type field */
      assign_fermion_propagator_from_spinor_field ( fp2, propagator_list[1-iflavor], VOLUME);

      double ** v2 = init_2level_dtable ( VOLUME, 32 );
      if ( v2 == NULL ) {
        fprintf(stderr, "[nppn_oet_fwd] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      double ** vp = init_3level_dtable ( T, g_sink_momentum_number, 32 );
      if ( vp == NULL ) {
        fprintf(stderr, "[nppn_oet_fwd] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      /***********************************************************
       * contractions for N - N with up and dn propagagor
       ***********************************************************/
      for ( int if1 = 0; if1 < gamma_f1_nucleon_number; if1++ ) {
      for ( int if2 = 0; if2 < gamma_f1_nucleon_number; if2++ ) {

        fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_nucleon_list[if2], fp2, VOLUME );

        fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_nucleon_list[if1], fp3, VOLUME );

        fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp3, fp3, -gamma_f1_nucleon_sign[if1]*gamma_f1_nucleon_sign[if2], VOLUME );


        sprintf(aff_tag, "/N-N/%c%c%c/t%dx%dy%dz%d/gi%d/gf%d/n1",
            flavor_tag[iflavor], flavor_tag[1-iflavor], flavor_tag[iflavor],
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_nucleon_list[if1], gamma_f1_nucleon_list[if2]);

        exitstatus = contract_v5 ( v2, fp, fp3, fp, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from contract_v5, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(49);
        }

        /***********************************************************/
        /***********************************************************/

        sprintf(aff_tag, "/N-N/%c%c%c/t%dx%dy%dz%d/gi%d/gf%d/n2",
          flavor_tag[iflavor], flavor_tag[1-iflavor], flavor_tag[iflavor],
          gsx[0], gsx[1], gsx[2], gsx[3],
          gamma_f1_nucleon_list[if1], gamma_f1_nucleon_list[if2]);

        exitstatus = contract_v6 ( v2, fp, fp3, fp, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from contract_v6, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(49);
        }

      }}

      free_fp_field ( &fp3 );
      fini_2level_dtable ( &v2 );
      fini_3level_dtable ( &vp );

    }  /* end of loop on flavor */

    /*****************************************************************/
    /*****************************************************************/

    /*****************************************************************
     * PART III M-M
     *****************************************************************/

    for ( int iflavor = 0; iflavor <=1 ; iflavor ++ ) {
    for ( int iflavor2 = 0; iflavor2 <=1 ; iflavor2 ++ ) {

      for ( int igi2 = 0; igi2 < gamma_i2_number; igi2++ ) {
      for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

        double ** v3 = init_2level_dtable ( VOLUME, 2 );
        if ( v3 == NULL ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__ );
          EXIT(47);
        }

        double *** vp = init_3level_dtable ( T, g_sink_momentum_number, 2 );
        if ( vp == NULL ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__ );
          EXIT(47);
        }

        /*****************************************************************
         * contractions for the charged pion - pion correlator
         *****************************************************************/
        sprintf(aff_tag, "/m-m/%c%c/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d", 
            flavor_tag[1-iflavor], flavor_tag[iflavor2],
            gsx[0], gsx[1], gsx[2], gsx[3], gamma_i2_number[igi2], gamma_f2_number[igf2] );

        contract_twopoint_xdep (v3[0], gamma_i2_list[igi2], gamma_f2_list[igf2], propagator_list[iflavor], propagator_list[iflavor2], 3, 1, 1., 64);

        exitstatus = contract_vn_momentum_projection ( vp, v3, 1, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 1, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(49);
        }
 
       fini_2level_dtable ( &v3 ); 
       fini_3level_dtable ( &vp ); 
      }}  /* end of loop on gamma matrices */

    }}  /* end of loop on flavors */

    /*****************************************************************/
    /*****************************************************************/

    /*****************************************************************
     * PART IV P-NN-P
     *****************************************************************/

    /*****************************************************************
     * PART IVa fwd-phi
     *
     * for B, Z
     *****************************************************************/

    for ( int idt = 1; idt <= g_source_operator_separation; idt++ ) {

      int const gts = ( gsx[0] - idt + T_global ) % T_global;

      for ( int iflavor_p = 0; iflavor_p <= 1; iflavor_p ++ ) {
      for ( int iflavor_f = 0; iflavor_f <= 1; iflavor_f ++ ) {



      }}
    }


    /*****************************************************************
     * PART IVb fwd-fwd-phi
     *
     * for B, Z
     *****************************************************************/

    /*****************************************************************
     * PART IVc fwd-phi-phi
     *
     * for W
     *****************************************************************/

    /*****************************************************************
     * PART IVd fwd-phi
     *
     * for W
     *****************************************************************/

    /* loop on sequential source momenta */
    for( int iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {

      /***********************************************************
       * sequential propagator U^{-1} g5 exp(ip) D^{-1}: tfii
       ***********************************************************/
      if(g_cart_id == 0) fprintf(stdout, "# [nppn_oet_fwd] sequential inversion fpr pi2 = (%d, %d, %d)\n", 
          g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);

        double **prop_list = (double**)malloc(g_coherent_source_number * sizeof(double*));
        if(prop_list == NULL) {
          fprintf(stderr, "[nppn_oet_fwd] Error from malloc\n");
          EXIT(43);
        }

        for( int is=0;is<n_s*n_c;is++) {

          /* build sequential source */
          exitstatus = init_coherent_sequential_source( spinor_work[0], prop_list, g_source_coords_list[i_src][0], g_coherent_source_number, g_seq_source_momentum_list[iseq_mom], 5);
          if(exitstatus != 0) {
            fprintf(stderr, "[nppn_oet_fwd] Error from init_coherent_sequential_source, status was %d\n", exitstatus);
            EXIT(14);
          }

          /* source-smear the coherent source */
          exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);

          /* tm-rotate sequential source */
          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation(spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
          }

          memset(spinor_work[1], 0, sizeof_spinor_field);
          /* invert */
          exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], op_id_up);
          if(exitstatus != 0) {
            fprintf(stderr, "[nppn_oet_fwd] Error from tmLQCD_invert, status was %d\n", exitstatus);
            EXIT(12);
          }

          if ( check_propagator_residual ) {
            check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_up], 1 );
          }

          /* tm-rotate at sink */
          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
          }

          /* sink-smear the coherent-source propagator */
          exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);

          memcpy( sequential_propagator_list[is], spinor_work[1], sizeof_spinor_field);

        }  /* end of loop on spin-color component */
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [nppn_oet_fwd] time for seq propagator = %e seconds\n", retime-ratime);

        free(prop_list);
      
      /***********************************************/
      /***********************************************/

      /***********************************************
       * contractions involving sequential propagator
       ***********************************************/

      double **v1 = NULL, **v2 = NULL, **v3 = NULL, ***vp = NULL;
      fermion_propagator_type *fp = NULL;
      fp  = create_fp_field ( VOLUME );

      exitstatus= init_2level_buffer ( &v3, VOLUME, 24 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 24 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[nppn_oet_fwd] Error from init_3level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      /* sequential propagator as propagator type field */
      assign_fermion_propagator_from_spinor_field ( fp, sequential_propagator_list, VOLUME);

      for ( int i = 0; i < gamma_f2_number; i++ ) {

        for ( int i_sample = 0; i_sample < g_nsample; i_sample++ ) {


          /*****************************************************************
           * xi^+ - gf2 - ud
           *****************************************************************/

          /* spinor_work <- Gamma_f2^+ stochastic source */
          spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f2_list[i], stochastic_source_list[i_sample], VOLUME );
          spinor_field_ti_eq_re ( spinor_work[0], gamma_f2_adjoint_sign[i], VOLUME);

          sprintf(aff_tag, "/v3/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-ud/sample%.2d", 
              g_source_coords_list[i_src][0],
              g_source_coords_list[i_src][1],
              g_source_coords_list[i_src][2],
              g_source_coords_list[i_src][3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f2_list[i], i_sample);

          exitstatus = contract_v3  ( v3, spinor_work[0], fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_v3, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#if 0
          /*****************************************************************
           * phi^+ - gf2 - ud
           *****************************************************************/
          /* spinor_work <- Gamma_f2^+ stochastic propagator */
          spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f2_list[i], stochastic_propagator_list[i_sample], VOLUME );
          spinor_field_ti_eq_re ( spinor_work[0], gamma_f2_adjoint_sign[i], VOLUME);

          sprintf(aff_tag, "/v3/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-ud/sample%.2d", 
              g_source_coords_list[i_src][0],
              g_source_coords_list[i_src][1],
              g_source_coords_list[i_src][2],
              g_source_coords_list[i_src][3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f2_list[i], i_sample);

          exitstatus = contract_v3  ( v3, spinor_work[0], fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_v3, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#endif

        }  /* end of loop on samples */
      }  /* end of loop on gf2  */

      fini_2level_buffer ( &v3 );
      fini_3level_buffer ( &vp );
      free_fp_field ( &fp );

      /***********************************************/
      /***********************************************/

      /* loop on coherent source locations */
      for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
        int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

        gsx[0] = t_coherent;
        gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
        gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
        gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

        get_point_source_info (gsx, sx, &source_proc_id);


        /***********************************************/
        /***********************************************/

        double **v1 = NULL, **v2 = NULL, **v3 = NULL, ***vp = NULL;
        fermion_propagator_type *fp = NULL, *fp2 = NULL, *fp3 = NULL, *fp4 = NULL;
        fp  = create_fp_field ( VOLUME );
        fp2 = create_fp_field ( VOLUME );
        fp3 = create_fp_field ( VOLUME );
        fp4 = create_fp_field ( VOLUME );

        /* sequential propagator as propagator type field */
        assign_fermion_propagator_from_spinor_field ( fp, sequential_propagator_list, VOLUME);

        /* up propagator as propagator type field */
        assign_fermion_propagator_from_spinor_field ( fp2,  &(propagator_list_up[i_coherent * n_s*n_c]), VOLUME);

        /***********************************************************/
        /***********************************************************/

        exitstatus= init_2level_buffer ( &v2, VOLUME, 32 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 32 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_3level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        /***********************************************************
         * contractions for pi N - D with up sequential propagagor
         ***********************************************************/
        for ( int if1 = 0; if1 < gamma_f1_nucleon_number; if1++ ) {

          /* fp3 <- fp x Gamma_i1 */
          fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_nucleon_list[if1], fp, VOLUME );
          fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp3, fp3, -gamma_f1_nucleon_sign[if1], VOLUME );

        for ( int if2 = 0; if2 < gamma_f1_delta_number; if2++ ) {

          /* fp4 <- Gamma_f2 x fp2 */
          fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp4, gamma_f1_delta_list[if2], fp2, VOLUME );
          fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp4, fp4, gamma_f1_delta_snk_sign[if2], VOLUME );

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t1",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v5 ( v2, fp3, fp4, fp2, VOLUME );
          if ( exitstatus != 0 ) {
           fprintf(stderr, "[nppn_oet_fwd] Error from contract_v5, status was %d\n", exitstatus);
             EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t2",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v5 ( v2, fp3, fp2, fp4, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t4",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v5 ( v2, fp2, fp3, fp4, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t6",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v6 ( v2, fp2, fp4, fp3, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_v6, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***********************************************************/
          /***********************************************************/

          fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp4, gamma_f1_delta_list[if2], fp3, VOLUME );
          fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp4, fp4, gamma_f1_delta_snk_sign[if2], VOLUME );

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t3",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v5 ( v2, fp2, fp4, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t5",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v6 ( v2, fp2, fp2, fp4, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_v6, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

        }}  /* end of gf1_delta, gf1_nucleon */

        fini_2level_buffer ( &v2 );
        fini_3level_buffer ( &vp );
        free_fp_field ( &fp4 );

        /*****************************************************************/
        /*****************************************************************/

        /*****************************************************************
         * contraction for pi x pi - rho 2-point function
         *****************************************************************/

        exitstatus= init_2level_buffer ( &v3, VOLUME, 2 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 2 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_3level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        /*****************************************************************/
        /*****************************************************************/

        /*****************************************************************
         * contractions for the pi^+ pi^- rho^0 correlator
         *****************************************************************/
        for ( int igf = 0; igf < gamma_rho_number; igf++ ) {

          sprintf(aff_tag, "/mxm-m/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              5,  gamma_rho_list[igf] );

          contract_twopoint_xdep (v3[0], 5, gamma_rho_list[igf], sequential_propagator_list, &(propagator_list_dn[i_coherent * n_s*n_c]), n_c, 1, 1., 64);

          exitstatus = contract_vn_momentum_projection ( vp, v3, 1, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 1, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
        }  /* end of loop on gamma rho at sink */

        fini_2level_buffer ( &v3 );
        fini_3level_buffer ( &vp );

        /*****************************************************************/
        /*****************************************************************/

        exitstatus= init_2level_buffer ( &v1, VOLUME, 72 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_2level_buffer ( &v2, VOLUME, 384 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 384 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_3level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        assign_fermion_propagator_from_spinor_field ( fp,  sequential_propagator_list, VOLUME);
        assign_fermion_propagator_from_spinor_field ( fp2,  &(propagator_list_up[i_coherent * n_s*n_c]), VOLUME);
        assign_fermion_propagator_from_spinor_field ( fp3,  &(propagator_list_dn[i_coherent * n_s*n_c]), VOLUME);

        for ( int i = 0; i < gamma_f1_nucleon_number; i++ ) {
          for ( int i_sample = 0; i_sample < g_nsample; i_sample++ ) {
  
#if 0
            /* multiply with Dirac structure at vertex f1 */
            spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f1_nucleon_list[i], stochastic_source_list[i_sample], VOLUME );
            spinor_field_ti_eq_re ( spinor_work[0], gamma_f1_nucleon_sign[i], VOLUME);


            /*****************************************************************
             * xi - gf1 - ud 
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (1) xi - gf1 - ud - u
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-ud-u/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3], 
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

            /*****************************************************************
             * (2) xi - gf1 - ud - d
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-ud-d/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp3, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

            /*****************************************************************/
            /*****************************************************************/

            /*****************************************************************
             * xi - gf1 - u
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp2, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (3) xi - gf1 - u - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-u-ud/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

            /*****************************************************************/
            /*****************************************************************/
  
            /*****************************************************************
             * xi - gf1 - d
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp3, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (4) xi - gf1 - d - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-d-ud/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
#endif  /* of if 0 */

            /*****************************************************************/
            /*****************************************************************/

            /* multiply with Dirac structure at vertex f1 */
            spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f1_nucleon_list[i], stochastic_propagator_list[i_sample], VOLUME );
            spinor_field_ti_eq_re ( spinor_work[0], gamma_f1_nucleon_sign[i], VOLUME);
  
            /*****************************************************************
             * phi - gf1 - ud
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (1) phi - gf1 - ud - u
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-ud-u/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

#if 0
            /*****************************************************************
             * (2 )phi - gf1 - ud - d
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-ud-d/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp3, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
  
#endif  /* end of if 0  */

            /*****************************************************************/
            /*****************************************************************/
  
            /*****************************************************************
             * phi - gf1 - u
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp2, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (3) phi - gf1 - u - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-u-ud/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
  
            /*****************************************************************/
            /*****************************************************************/
  
#if 0
            /*****************************************************************
             * phi - gf1 - d
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp3, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (4) phi - gf1 - d - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-d-ud/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
#endif  /* end of if 0  */

          }  /* end of loop on samples */
        }  /* end of loop on gf1  */

        fini_2level_buffer ( &v1 );
        fini_2level_buffer ( &v2 );
        fini_3level_buffer ( &vp );


        free_fp_field ( &fp  );
        free_fp_field ( &fp2 );
        free_fp_field ( &fp3 );

      }  /* end of loop on coherent source timeslices */

    }  /* end of loop on sequential momentum list */

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[nppn_oet_fwd] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(111);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */
  }  /* end of loop on base source locations */

  fini_2level_buffer ( &sequential_propagator_list );
  fini_2level_buffer ( &propagator_list_up );
  fini_2level_buffer ( &propagator_list_dn );

  /***********************************************************/
  /***********************************************************/


  /***********************************************************/
  /***********************************************************/

  fini_2level_buffer ( &stochastic_propagator_list );
  fini_2level_buffer ( &stochastic_source_list );

  /***********************************************************/
  /***********************************************************/

  if ( do_contraction_oet ) {

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   ***********************************************
   **
   ** stochastic contractions using the 
   **   one-end-trick
   **
   ***********************************************
   ***********************************************/
  exitstatus = init_2level_buffer ( &stochastic_propagator_list, 4, _GSI(VOLUME) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &stochastic_propagator_zero_list, 4, _GSI(VOLUME) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &stochastic_source_list, 4, _GSI(VOLUME) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &propagator_list_up, 12, _GSI(VOLUME) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &propagator_list_dn, 12, _GSI(VOLUME) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }


  /* loop on base source locations */
  for( int i_src=0; i_src < g_source_location_number; i_src++) {

    /* base source timeslice */
    int t_base = g_source_coords_list[i_src][0];

#ifdef HAVE_LHPC_AFF
    /***********************************************
     * open aff output file
     ***********************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN_oet", Nconf, t_base );
      fprintf(stdout, "# [nppn_oet_fwd] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[nppn_oet_fwd] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      }
    }  /* end of if io_proc == 2 */
#endif

    /* loop on coherent source locations */
    for(int i_coherent = 0; i_coherent < g_coherent_source_number; i_coherent++) {

      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;
      gsx[0] = t_coherent;
      gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
      gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
      gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

      /***********************************************
       * up-type forward propagator
       ***********************************************/
      if ( read_forward_propagator ) {
        for ( int isc = 0; isc < n_s*n_c; isc++ ) {
          sprintf ( filename, "source.%c.%.4d.t%dx%dy%dz%d.%.2d.inverted", 'u', Nconf, gsx[0], gsx[1], gsx[2], gsx[3], isc );

          exitstatus = read_lime_spinor ( propagator_list_up[isc], filename, 0 );
          if(exitstatus != 0) {
            fprintf(stderr, "[nppn_oet_fwd] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(12);
          }
        }
      } else {
        exitstatus = point_source_propagator ( propagator_list_up, gsx, op_id_up, 1, 1, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, mzz );
        if(exitstatus != 0) {
          fprintf(stderr, "[nppn_oet_fwd] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(12);
        }
      }

      if ( read_forward_propagator ) {
        for ( int isc = 0; isc < n_s*n_c; isc++ ) {
          sprintf ( filename, "source.%c.%.4d.t%dx%dy%dz%d.%.2d.inverted", 'd', Nconf, gsx[0], gsx[1], gsx[2], gsx[3], isc );

          exitstatus = read_lime_spinor ( propagator_list_dn[isc], filename, 0 );
          if(exitstatus != 0) {
            fprintf(stderr, "[nppn_oet_fwd] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(12);
          }
        }
      } else {
        exitstatus = point_source_propagator ( propagator_list_dn, gsx, op_id_dn, 1, 1, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, mzz );
        if(exitstatus != 0) {
          fprintf(stderr, "[nppn_oet_fwd] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(12);
        }
      }

      double **v1 = NULL, **v2 = NULL;
      double **v3 = NULL, ***vp = NULL;
      fermion_propagator_type *fp = NULL, *fp2 = NULL, *fp3 = NULL;
      fp  = create_fp_field ( VOLUME );
      fp2 = create_fp_field ( VOLUME );
      fp3 = create_fp_field ( VOLUME );

      /* fp  <- up propagator as fermion_propagator_type */
      assign_fermion_propagator_from_spinor_field ( fp,  propagator_list_up, VOLUME);
      /* fp2 <- dn propagator as fermion_propagator_type */
      assign_fermion_propagator_from_spinor_field ( fp2, propagator_list_dn, VOLUME);

      /******************************************************
       * re-initialize random number generator
       ******************************************************/
      if ( ! read_stochastic_source_oet ) {
        sprintf(filename, "rng_stat.%.4d.tsrc%.3d.stochastic-oet.out", Nconf, gsx[0]);
        exitstatus = init_rng_stat_file ( ( ( gsx[0] + 1 ) * 10000 + g_seed ), filename );
        if(exitstatus != 0) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(38);
        }
      }

      /* loop on oet samples */
      for( int isample=0; isample < g_nsample_oet; isample++) {

        /*****************************************************************
         * read stochastic oet source from file
         *****************************************************************/
        if ( read_stochastic_source_oet ) {
          for ( int ispin = 0; ispin < 4; ispin++ ) {
            sprintf(filename, "%s-oet.%.4d.t%.2d.%.2d.%.5d", filename_prefix, Nconf, gsx[0], ispin, isample);
            if ( ( exitstatus = read_lime_spinor( stochastic_source_list[ispin], filename, 0) ) != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from read_lime_spinor, status was %d\n", exitstatus);
              EXIT(2);
            }
          }
          /* recover the random field */
          if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gsx[0], NULL, -1 ) ) != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(64);
          }

        /*****************************************************************
         * generate stochastic oet source
         *****************************************************************/
        } else {
          /* dummy call to initialize the ran field, we do not use the resulting stochastic_source_list */
          if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gsx[0], NULL, 1 ) ) != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(64);
          }
          if ( write_stochastic_source_oet ) {
            for ( int ispin = 0; ispin < 4; ispin++ ) {
              sprintf(filename, "%s-oet.%.4d.t%.2d.%.2d.%.5d", filename_prefix, Nconf, gsx[0], ispin, isample);
              if ( ( exitstatus = write_propagator( stochastic_source_list[ispin], filename, 0, 64) ) != 0 ) {
                fprintf(stderr, "[nppn_oet_fwd] Error from write_propagator, status was %d\n", exitstatus);
                EXIT(2);
              }
            }
          }
        }  /* end of if read stochastic source - else */

        /*****************************************************************
         * invert for stochastic timeslice propagator at zero momentum
         *****************************************************************/
        if ( !read_stochastic_propagator_oet ) {
          for( int i = 0; i < 4; i++) {
            memcpy(spinor_work[0], stochastic_source_list[i], sizeof_spinor_field);

            /* source-smearing stochastic momentum source */
            if ( ( exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi) ) != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(5);
            }

            /* tm-rotate stochastic source */
            if( g_fermion_type == _TM_FERMION ) {
              spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
            }

            memset(spinor_work[1], 0, sizeof_spinor_field);

            exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], op_id_up);
            if(exitstatus != 0) {
              fprintf(stderr, "[nppn_oet_fwd] Error from tmLQCD_invert, status was %d\n", exitstatus);
              EXIT(44);
            }

            if ( check_propagator_residual ) {
              check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_up], 1 );
            }

            /* tm-rotate stochastic propagator at sink */
            if( g_fermion_type == _TM_FERMION ) {
              spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
            }

            /* sink smearing stochastic propagator */
            if ( ( exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi) ) != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(5);
            }
 
            memcpy( stochastic_propagator_zero_list[i], spinor_work[1], sizeof_spinor_field);
          }

        } else {
          for( int i = 0; i < 4; i++) {
            sprintf(filename, "%s-oet.%c.%.4d.t%.2d.%.2d.%.5d", filename_prefix2, 'u', Nconf, gsx[0], i, isample);
            if ( g_cart_id == 0 ) fprintf( stdout, "# [nppn_oet_fwd] Reading stochastic propagator oet mom from file %s %s %d\n", filename, __FILE__, __LINE__ );

            exitstatus = read_lime_spinor ( stochastic_propagator_zero_list[i], filename, 0 );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(47);
            }
          }
        }


        /*****************************************************************
         * calculate V3
         *
         * phi^+ g5 Gamma_f2 ( pf2 ) U ( z_1xi )
         * phi^+ g5 Gamma_f2 ( pf2 ) D
         *****************************************************************/

        exitstatus= init_2level_buffer ( &v3, VOLUME, 24 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 24 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[nppn_oet_fwd] Error from init_3level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        for ( int if2 = 0; if2 < gamma_f2_number; if2++ ) {

          for( int ispin = 0; ispin < 4; ispin++ ) {

            /*****************************************************************
             * (1) phi - gf2 - u
             *****************************************************************/

            /* spinor_work <- Gamma_f2^+ x stochastic_propagator, up to sign */
            spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f2_list[if2], stochastic_propagator_zero_list[ispin], VOLUME );
            /* spinor_work <- g5 spinor_work */
            g5_phi( spinor_work[0], VOLUME );
            /* spinor_work <- spinor_work x sign for Gamma_f2^+ */
            spinor_field_ti_eq_re ( spinor_work[0], gamma_f2_g5_adjoint_sign[if2], VOLUME);

            sprintf(aff_tag, "/v3-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-u/sample%.2d/d%d", gsx[0], gsx[1], gsx[2], gsx[3],
                0, 0, 0, gamma_f2_list[if2], isample, ispin);

            exitstatus = contract_v3 ( v3, spinor_work[0], fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v3, status was %d\n", exitstatus);
              EXIT(47);
            }
 
            exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }

            exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

#if 0
            /*****************************************************************/
            /*****************************************************************/

            /*****************************************************************
             * (2) phi - gf2 - d
             *****************************************************************/
            sprintf(aff_tag, "/v3-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-d/sample%.2d/d%d", gsx[0], gsx[1], gsx[2], gsx[3],
                0, 0, 0,
                gamma_f2_list[if2], isample, ispin);

            exitstatus = contract_v3 ( v3, spinor_work[0], fp2, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_v3, status was %d\n", exitstatus);
              EXIT(47);
            }
 
            exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }

            exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
#endif  /* of if 0 */
          }  /* end of loop on spin components ispin */

        }  /* end of loop on gamma_f2_list */

        fini_2level_buffer ( &v3 );
        fini_3level_buffer ( &vp );

        /*****************************************************************/
        /*****************************************************************/

        /*****************************************************************
         * loop on sequential source momenta p_i2
         *****************************************************************/
        for( int iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {

          int seq_source_momentum[3] = { g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2] };

          if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gsx[0], seq_source_momentum, 0 ) ) != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(64);
          }

          /*****************************************************************/
          /*****************************************************************/

          /*****************************************************************
           * invert for stochastic timeslice propagator 
           *   with sequential * momentum p_i2
           *****************************************************************/
          if ( !read_stochastic_propagator_oet ) {
            for( int i = 0; i < 4; i++) {
              memcpy(spinor_work[0], stochastic_source_list[i], sizeof_spinor_field);

              /* source-smearing stochastic momentum source */
              if ( ( exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi) ) != 0 ) {
                fprintf(stderr, "[nppn_oet_fwd] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(12);
              }

              /* tm-rotate stochastic source */
              if( g_fermion_type == _TM_FERMION ) {
                spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
              }

              memset(spinor_work[1], 0, sizeof_spinor_field);

              exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], op_id_up);
              if(exitstatus != 0) {
                fprintf(stderr, "[nppn_oet_fwd] Error from tmLQCD_invert, status was %d\n", exitstatus);
                EXIT(44);
              }

              if ( check_propagator_residual ) {
                check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_up], 1 );
              }

              /* tm-rotate stochastic propagator at sink */
              if( g_fermion_type == _TM_FERMION ) {
                 spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
             }

              /* sink smearing stochastic propagator */
              exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);

              memcpy( stochastic_propagator_list[i], spinor_work[1], sizeof_spinor_field);
            }
          } else {
            for( int i = 0; i < 4; i++) {
              sprintf(filename, "%s-oet.%c.%.4d.t%.2d.px%dpy%dpz%d.%.2d.%.5d", filename_prefix2, 'u', Nconf, gsx[0], 
                  g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                  i, isample );
              if ( g_cart_id == 0 ) fprintf( stdout, "# [nppn_oet_fwd] Reading stochastic propagator oet mom from file %s %s %d\n", filename, __FILE__, __LINE__ );

              exitstatus = read_lime_spinor ( stochastic_propagator_list[i], filename, 0 );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[nppn_oet_fwd] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(47);
              }
            }
          }

          /*****************************************************************/
          /*****************************************************************/

          /*****************************************************************
           * contraction for pion - pion 2-point function
           *****************************************************************/

          exitstatus= init_2level_buffer ( &v3, VOLUME, 2 );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 2 );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from init_3level_buffer, status was %d\n", exitstatus);
            EXIT(47);
          }

          sprintf(aff_tag, "/m-m/t%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/sample%.2d/gi%.2d/gf%.2d", gsx[0],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              isample, 5, 5);

          contract_twopoint_xdep (v3[0], 5, 5, stochastic_propagator_zero_list, stochastic_propagator_list, 1, 1, 1., 64);

          exitstatus = contract_vn_momentum_projection ( vp, v3, 1, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 1, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          fini_2level_buffer ( &v3 );
          fini_3level_buffer ( &vp );


          /*****************************************************************/
          /*****************************************************************/

          /*****************************************************************
           * calculate V2 and V4
           *
           * z_1phi = V4
           * z_3phi = V2
           *****************************************************************/

            exitstatus= init_2level_buffer ( &v1, VOLUME, 72 );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
              EXIT(47);
            }

            exitstatus= init_2level_buffer ( &v2, VOLUME, 384 );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from init_2level_buffer, status was %d\n", exitstatus);
              EXIT(47);
            }

            exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 384 );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[nppn_oet_fwd] Error from init_3level_buffer, status was %d\n", exitstatus);
              EXIT(47);
            }


            for( int if1 = 0; if1 < gamma_f1_nucleon_number; if1++ ) {

              for( int ispin = 0; ispin < 4; ispin++ ) {

                /* spinor_work <- Gamma_f1^t x stochastic propagator */
                spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f1_nucleon_list[if1], stochastic_propagator_list[ispin], VOLUME );
                spinor_field_ti_eq_re ( spinor_work[0], gamma_f1_nucleon_transposed_sign[if1], VOLUME);

                /*****************************************************************/
                /*****************************************************************/

                /*****************************************************************
                 * phi - gf1 - d
                 *****************************************************************/
                exitstatus = contract_v1 ( v1, spinor_work[0], fp2, VOLUME  );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[nppn_oet_fwd] Error from contract_v1, status was %d\n", exitstatus);
                  EXIT(47);
                }
  
                /*****************************************************************
                 * (3) phi - gf1 - d - u
                 *****************************************************************/
                sprintf(aff_tag, "/v2-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-d-u/sample%.2d/d%d", gsx[0], gsx[1], gsx[2], gsx[3],
                    g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                    gamma_f1_nucleon_list[if1], isample, ispin);

                exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[nppn_oet_fwd] Error from contract_v4, status was %d\n", exitstatus);
                  EXIT(47);
                }

                exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
                  EXIT(48);
                }

                exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
                  EXIT(49);
                }

              }  /* end of loop on ispin */

              /*****************************************************************/
              /*****************************************************************/

              /* fp3 <- Gamma_f1 x fp2 = Gamma_i1 x D */
              fermion_propagator_field_eq_gamma_ti_fermion_propagator_field (fp3, gamma_f1_nucleon_list[if1], fp2, VOLUME );
              fermion_propagator_field_eq_fermion_propagator_field_ti_re ( fp3, fp3, gamma_f1_nucleon_sign[if1], VOLUME );


              for( int ispin = 0; ispin < 4; ispin++ ) {
                /*****************************************************************
                 * (4) phi - gf1 - d - u
                 *****************************************************************/
                sprintf(aff_tag, "/v4-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-d-u/sample%.2d/d%d", gsx[0], gsx[1], gsx[2], gsx[3],
                    g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                    gamma_f1_nucleon_list[if1], isample, ispin);

                exitstatus = contract_v4 ( v2, stochastic_propagator_list[ispin], fp3, fp, VOLUME );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[nppn_oet_fwd] Error from contract_v4, status was %d\n", exitstatus);
                  EXIT(47);
                }

                exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
                  EXIT(48);
                }

                exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[nppn_oet_fwd] Error from contract_vn_write_aff, status was %d\n", exitstatus);
                  EXIT(49);
                }

              }  /* end of loop on ispin */
            }  /* end of loop on gamma at f1 */


            fini_2level_buffer ( &v1 );
            fini_2level_buffer ( &v2 );
            fini_3level_buffer ( &vp );

        }  /* end of loop on sequential source momenta pi2 */

      } /* end of loop on oet samples */

      free_fp_field( &fp  );
      free_fp_field( &fp2 );
      free_fp_field( &fp3 );

    }  /* end of loop on coherent sources */

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[nppn_oet_fwd] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(111);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */
  } /* end of loop on base sources */ 


  fini_2level_buffer ( &propagator_list_up );
  fini_2level_buffer ( &propagator_list_dn );

  /***********************************************************/
  /***********************************************************/

  }  /* end of if do_contraction_oet */

  /***********************************************************/
  /***********************************************************/


  /***********************************************
   * free gauge fields and spinor fields
   ***********************************************/

  if(g_gauge_field        != NULL ) free(g_gauge_field);
  if( gauge_field_smeared != NULL ) free(gauge_field_smeared);

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  if ( check_propagator_residual ) {
    fini_clover ();
    if( gauge_field_with_phase != NULL) free( gauge_field_with_phase );
  }

  /***************************************************************************
   * fini rng state
   ***************************************************************************/
  fini_rng_state ( &rng_state);


  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  if ( check_propagator_residual ) mpi_fini_xchange_eo_spinor();
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [nppn_oet_fwd] %s# [nppn_oet_fwd] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [nppn_oet_fwd] %s# [nppn_oet_fwd] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
