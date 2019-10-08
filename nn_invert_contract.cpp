/****************************************************
 * nn_invert_contract
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
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
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

#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

using namespace cvc;

void rotatePropsOF(double *rot_b, int beta, int b, double omega, char flavour, int T, int Lx, int Ly, int Lz, double * sp ) {
 
  double const gamma5[4] = {1., 1., -1., -1.};
  double const flavourFactor = ( flavour == 'u' ) ? 1. : -1.;

  double * s = sp;

  for (int t = 0; t < T; t++) {
    for (int x = 0; x < Lx; x++) {
      for (int y = 0; y < Ly; y++) {
        for (int z = 0; z < Lz; z++) {

          for (int alpha = 0; alpha < 4; alpha++) {

            for (int a = 0; a < 3; a++) {

              double _Complex expo  = cexp ( I * omega * gamma5[alpha] * flavourFactor / 2. );

              double _Complex expo2 = cexp ( I * omega * gamma5[beta]  * flavourFactor / 2. );

              // printf("with alpha = %d and beta = %d:\texp = %.8e + %.8e\t
              // using gamma5 = %.1e\n",alpha, beta, creal(expo), cimag(expo),
              // creal(gamma5[alpha*4+beta])); do rotation
              int p = (((((t * Lx + x) * Ly + y) * Lz + z) * 4 + alpha) * 3 + a) * 2;
             
              double _Complex cnum = s[p] + I * s[p + 1];
              // printf("prop = %.8e + i*%.8e,\texpo = %.8e + i*%.8e\n",
              // prop[p], prop[p + 1], creal(expo), cimag(expo));
              cnum *= expo * expo2;
              rot_b[p    ] = creal(cnum);
              rot_b[p + 1] = cimag(cnum);
            }
          }
        }
      }
    }
  }
}





double _Complex cgamma5[16];

void setGamma() {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      int index = i * 4 + j;
      if (i == j) {
        
        cgamma5[index] = 0;
      } else if (i + j == 3) {
        cgamma5[index] = 0;
        
      } else if ((i == 2 && j == 0) || (j == 3 && i == 1)) {
        cgamma5[index] = 0;
      } else if ((i == 0 && j == 2) || (j == 1 && i == 3)) {
        cgamma5[index] = 0;
      } else if (i == 0 && j == 1) {
        cgamma5[index] = -I;
      } else if (i == 3 && j == 2) {
        cgamma5[index] = I;
      } else if (i == 1 && j == 0) {
        cgamma5[index] = I;
      } else if (i == 2 && j == 3) {
        cgamma5[index] = -I;
      }

      else {
        cgamma5[index] = 0;
      }
    }
  }
}



int epsilon(int _a, int _b, int _c) {
  int a = _a + 1;
  int b = _b + 1;
  int c = _c + 1;
  // totally antisymmetric tensor
  if (a > 3 || a < 1 || b > 3 || b < 1 || c > 3 || c < 1) {
    printf("epsilon is out of bounds!\n");
    exit(1);
  }

  if (a + b + c != 6 || a == b || a == c || b == c) {
    return 0;
  } else if (a == 1) {
    if (b == 2) {
      return 1;
    } else {
      return -1;
    }
  } else if (a == 2) {
    if (b == 1) {
      return -1;
    } else {
      return 1;
    }
  } else {
    if (b == 1) {
      return 1;
    } else {
      return -1;
    }
  }
}



void usage() {
  fprintf(stdout, "Code to 2-pt function inversion and contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "twopt";

  char const gamma_id_to_Cg_ascii[16][10] = {
    "Cgy",
    "Cgzg5",
    "Cgt",
    "Cgxg5",
    "Cgygt",
    "Cgyg5gt",
    "Cgyg5",
    "Cgz",
    "Cg5gt",
    "Cgx",
    "Cgzg5gt",
    "C",
    "Cgxg5gt",
    "Cgxgt",
    "Cg5",
    "Cgzgt"
  };

  int c;
  int filename_set = 0;
  int gsx[4], sx[4];
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  double **spinor_field=NULL;
  char filename[100];
  // double ratime, retime;
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  double *gauge_field_smeared = NULL;

  int const    gamma_f1_number                           = 1;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 }; /*, 11,  8,  2 ; */
  double const gamma_f1_sign[gamma_f1_number]            = { +1 }; /*, +1, -1, -1 ; */
  /* double const gamma_f1_transposed_sign[gamma_f1_number] = { -1, -1, +1, -1 }; */

#if 0
  int const    gamma_f1_number                = 3;
  int const    gamma_f1_list[gamma_f1_number] = { 9,  0,  7 };
  double const gamma_f1_sign[gamma_f1_number] = {-1, +1, +1 };
#endif  /* of if 0 */

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
  if(filename_set==0) strcpy(filename, "twopt.input");
  /* fprintf(stdout, "# [nn_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [nn_invert_contract] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  /* exitstatus = tmLQCD_invert_init(argc, argv, 1, 0); */
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
    fprintf(stdout, "# [nn_invert_contract] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [nn_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [nn_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[nn_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[nn_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

#if defined STOCHASTIC
  size_t sizeof_spinor_field = _GSI( VOLUME ) * sizeof ( double );
#endif

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [nn_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [nn_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[nn_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[nn_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[nn_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {

    alloc_gauge_field ( &gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy ( gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[nn_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
    }  /* end of if N_aoe > 0 */
  }  /* end of if N_Jacobi > 0 */



  /***********************************************************
   * initialize clover, lmzz and lmzzinv
   ***********************************************************/
  exitstatus = init_clover ( &lmzz, &lmzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[nn_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[nn_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [nn_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * allocate spinor_field
   ***********************************************************/
  spinor_field = init_2level_dtable ( 24, _GSI( VOLUME ) );
  if( spinor_field == NULL ) {
    fprintf(stderr, "[nn_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }


  /***************************************************************************
   ***************************************************************************
   **
   ** point-to-all version
   **
   ***************************************************************************
   ***************************************************************************/

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
      fprintf(stderr, "[nn_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

#if 0
    for ( int ia = 0; ia < 12; ia++ ) {

      size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof ( double );

      double ** spinor_work = init_2level_dtable ( 2, _GSI(VOLUME+RAND) );

      /***************************************************************************
       * up-type propagator
       ***************************************************************************/

      spinor_work[0][_GSI( g_ipt[gsx[0]][gsx[1]][gsx[2]][gsx[3]] ) + 2*ia ] = 1.;

      exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], _OP_ID_UP );
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_factorized] Error from tmLQCD_invert, status was %d\n", exitstatus);
        EXIT(12);
      }

      rotatePropsOF ( spinor_work[0], ia/3, ia%3, M_PI/2., 'u', T, LX, LY, LZ, spinor_work[1] );

      if ( g_write_propagator ) {
        sprintf ( filename, "Tpsource.%.4d.t%dx%dy%dz%d.%d.inverted", Nconf, gsx[0], gsx[1], gsx[2], gsx[3] , ia );
        if ( ( exitstatus = write_propagator( spinor_work[0], filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[nn_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }

      sprintf ( filename, "Rpsource.%.4d.t%dx%dy%dz%d.%d.inverted", Nconf, gsx[0], gsx[1], gsx[2], gsx[3] , ia );
      if ( ( exitstatus = read_lime_spinor ( spinor_work[1], filename, 0 ) ) ) {
        fprintf(stderr, "[nn_invert_contract] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      double normu = 0.;

      spinor_field_norm_diff ( &normu, spinor_work[1], spinor_work[0], VOLUME );

      /***************************************************************************
       * dn-type propagator
       ***************************************************************************/
      memset ( spinor_work[0], 0, sizeof_spinor_field );
      memset ( spinor_work[1], 0, sizeof_spinor_field );

      spinor_work[0][_GSI( g_ipt[gsx[0]][gsx[1]][gsx[2]][gsx[3]] ) + 2*ia ] = 1.;

      exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], _OP_ID_DN );
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_factorized] Error from tmLQCD_invert, status was %d\n", exitstatus);
        EXIT(12);
      }

      // rotatePropsOF ( spinor_work[0], ia/3, ia%3, M_PI/2., 'd', T, LX, LY, LZ, spinor_work[1] );
      rotatePropsOF ( spinor_work[0], ia/3, ia%3, M_PI/2., 'd', T, LX, LY, LZ, spinor_work[1] );

      if ( g_write_propagator ) {
        sprintf ( filename, "Tnsource.%.4d.t%dx%dy%dz%d.%d.inverted", Nconf, gsx[0], gsx[1], gsx[2], gsx[3] , ia );
        if ( ( exitstatus = write_propagator( spinor_work[0], filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[nn_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }

      sprintf ( filename, "Rnsource.%.4d.t%dx%dy%dz%d.%d.inverted", Nconf, gsx[0], gsx[1], gsx[2], gsx[3] , ia );
      if ( ( exitstatus = read_lime_spinor ( spinor_work[1], filename, 0 ) ) ) {
        fprintf(stderr, "[nn_invert_contract] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      double normd = 0.;

      spinor_field_norm_diff ( &normd, spinor_work[1], spinor_work[0], VOLUME );


      fprintf ( stdout, "# [nn_invert_contract] norm sc %2d  u %16.7e   d %15.6e\n", ia, normu, normd );

      fini_2level_dtable ( &spinor_work );
    }
#endif  /* of if 0 */

#ifdef HAVE_LHPC_AFF
    /***********************************************
     * writer for aff output file
     ***********************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [nn_invert_contract] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[nn_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#endif

    /**********************************************************
     * propagators with source at gsx
     **********************************************************/


    /***********************************************************
     * up-type point-to-all propagator
     ***********************************************************/
    exitstatus = point_source_propagator ( &(spinor_field[0]), gsx, _OP_ID_UP, 1, 1, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, lmzz );
    if(exitstatus != 0) {
      fprintf(stderr, "[nn_invert_contract] Error from point_source_propagator, status was %d\n", exitstatus);
      EXIT(12);
    }

    if ( g_write_propagator ) {
      for ( int i = 0; i < 12; i++ ) {
        sprintf ( filename, "Rpsource.%.4d.t%dx%dy%dz%d.%d.inverted", Nconf, 
            gsx[0], gsx[1], gsx[2], gsx[3] , i );

        if ( ( exitstatus = write_propagator( spinor_field[i], filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[nn_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }
    }

    /***********************************************************
     * dn-type point-to-all propagator
     ***********************************************************/
    exitstatus = point_source_propagator ( &(spinor_field[12]), gsx, _OP_ID_DN, 1, 1, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, lmzz );
    if(exitstatus != 0) {
      fprintf(stderr, "[nn_invert_contract] Error from point_source_propagator, status was %d\n", exitstatus);
      EXIT(12);
    }

    if ( g_write_propagator ) {
      for ( int i = 0; i < 12; i++ ) {
        sprintf ( filename, "Rnsource.%.4d.t%dx%dy%dz%d.%d.inverted", Nconf, 
            gsx[0], gsx[1], gsx[2], gsx[3] , i );

        if ( ( exitstatus = write_propagator( spinor_field[12+i], filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[nn_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }
    }

#if 0
#endif  /* of if 0 */

    /***************************************************************************
     * allocate propagator fields
     ***************************************************************************/
    fermion_propagator_type * fp  = create_fp_field ( VOLUME );
    fermion_propagator_type * fp2 = create_fp_field ( VOLUME );
    fermion_propagator_type * fp3 = create_fp_field ( VOLUME );

    double ** v2 = init_2level_dtable ( VOLUME, 32 );
    if ( v2 == NULL ) {
      fprintf(stderr, "[nn_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    double *** vp = init_3level_dtable ( T, g_sink_momentum_number, 32 );
    if ( vp == NULL ) {
      fprintf(stderr, "[nn_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(47);
    }

    /***************************************************************************
     * spin 1/2 2-point correlation function
     ***************************************************************************/

    /* up propagator as propagator type field */
    assign_fermion_propagator_from_spinor_field ( fp,  &(spinor_field[ 0]), VOLUME);

    /* dn propagator as propagator type field */
    assign_fermion_propagator_from_spinor_field ( fp2, &(spinor_field[12]), VOLUME);

    /***********************************************************
     * contractions
     ***********************************************************/
    for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
    for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp2, VOLUME );

      fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );

      fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );

      double *** vpsum = init_3level_dtable ( T, g_sink_momentum_number, 32 );
      if ( vpsum == NULL ) {
        fprintf(stderr, "[nn_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      /***********************************************************
       * diagram n1
       ***********************************************************/
      sprintf(aff_tag, "/N-N/t%.2dx%.2dy%.2dz%.2d/gi_%s/gf_%s/n1",
          gsx[0], gsx[1], gsx[2], gsx[3],
          gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

      exitstatus = contract_v5 ( v2, fp, fp3, fp, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[nn_invert_contract] Error from contract_v5, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[nn_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[nn_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
        EXIT(49);
      }

#pragma omp parallel for
      for ( int i = 0; i < g_sink_momentum_number * T * 32; i++ ) {
        vpsum[0][0][i] += vp[0][0][i];
      }

      /***********************************************************
       * diagram n2
       ***********************************************************/

      sprintf(aff_tag, "/N-N/t%.2dx%.2dy%.2dz%.2d/gi_%s/gf_%s/n2",
          gsx[0], gsx[1], gsx[2], gsx[3],
          gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

      exitstatus = contract_v6 ( v2, fp, fp3, fp, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[nn_invert_contract] Error from contract_v6, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[nn_invert_contract] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[nn_invert_contract] Error from contract_vn_write_aff, status was %d\n", exitstatus);
        EXIT(49);
      }

#pragma omp parallel for
      for ( int i = 0; i < g_sink_momentum_number * T * 32; i++ ) {
        vpsum[0][0][i] += vp[0][0][i];
      }
     
      /***********************************************************/
      /***********************************************************/

      for ( int i = 0; i < g_sink_momentum_number; i++ ) {

        double _Complex *** diagram = init_3level_ztable ( T, 4, 4 );
        double _Complex * diagram_tr = init_1level_ztable ( T );

        for ( int it = 0; it < T; it++ ) {
          memcpy ( diagram[it][0], vpsum[it][i], 16 * sizeof(double _Complex ) );
        }


        exitstatus = correlator_add_baryon_boundary_phase ( diagram, gsx[0], 1, T );

        exitstatus = correlator_add_source_phase ( diagram, g_sink_momentum_list[i], &(gsx[1]), T );

        exitstatus = reorder_to_relative_time ( diagram, diagram, gsx[0], 1, T );

        exitstatus = correlator_spin_parity_projection ( diagram, diagram, 1., T);

        exitstatus = contract_diagram_co_eq_tr_zm4x4_field ( diagram_tr, diagram, T );

        sprintf(aff_tag, "/N-N/t%.2dx%.2dy%.2dz%.2d/gi_%s/gf_%s/PX%d_PY%d_PZ%d/n1+n2/P+/tr", gsx[0], gsx[1], gsx[2], gsx[3], 
            gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ],
            g_sink_momentum_list[i][0], g_sink_momentum_list[i][1], g_sink_momentum_list[i][2]);
      
        exitstatus = write_aff_contraction ( diagram_tr, affw, NULL, aff_tag, T );

        fini_3level_ztable ( &diagram );
        fini_1level_ztable ( &diagram_tr );
      }

      fini_3level_dtable ( &vpsum );
   
      /***************************************************************************
       * "by hand" version
       ***************************************************************************/

      unsigned int const VOL3 = LX * LY * LZ;
      gamma_matrix_type Gamma_i1, Gamma_f1;
     
      gamma_matrix_set ( &Gamma_i1, gamma_f1_list[if1], 1. );
      gamma_matrix_set ( &Gamma_f1, gamma_f1_list[if2], 1. );


      gamma_matrix_printf ( &Gamma_i1, "gi1", stdout );
      gamma_matrix_printf ( &Gamma_f1, "gf1", stdout );


      double _Complex **** corr = init_4level_ztable ( 2, T_global, 4, 4 );

      int const epsilon_num = 6;
      int const epsilon_perm[6][3] = { {0,1,2}, {1,2,0}, {2,0,1}, {1,0,2}, {0,2,1}, {2,1,0} };
      int const epsilon_sign[6] = {  1,  1,  1,  -1,  -1,  -1 };

      for ( int igamma  = 0; igamma  < 4; igamma++  ) {
      for ( int ialpha = 0; ialpha < 4; ialpha++ ) {
      for ( int ibeta = 0; ibeta < 4; ibeta++ ) {

        for ( int ip = 0; ip < epsilon_num; ip++ ) {

          int const ia = epsilon_perm[ip][0];
          int const ib = epsilon_perm[ip][1];
          int const ic = epsilon_perm[ip][2];

          /* for ( unsigned int it = 0; it < T; it++ ) */
          for ( unsigned int it = 0; it < 1; it++ )
          {
            /* for ( unsigned int ix = 0; ix < VOL3; ix++ ) */
            for ( unsigned int ix = 0; ix < 1; ix++ )
            {

              unsigned int const iix = it * VOL3 + ix;
              unsigned int const iit = it + g_proc_coords[0] * T;

              for ( int k1=0; k1 < 12; k1++ ) {
              for ( int k2=0; k2 < 12; k2++ ) {
                fprintf( stdout, " %2d %2d %2d %2d   %25.16e %25.16e    %25.16e %25.16e\n", k1/3+1, k1%3+1, k2/3+1, k2%3+1,
                    spinor_field[k2][_GSI(iix)+2*k1], spinor_field[k2][_GSI(iix)+2*k1+1],
                    spinor_field[12+k2][_GSI(iix)+2*k1], spinor_field[12+k2][_GSI(iix)+2*k1+1] );
              }}


            for ( int igammap = 0; igammap < 4; igammap++ ) {
            for ( int ialphap = 0; ialphap < 4; ialphap++ ) {
            for ( int ibetap  = 0; ibetap  < 4; ibetap++ ) {

              for ( int ip2 = 0; ip2 < epsilon_num; ip2++ ) {

                int const iap = epsilon_perm[ip2][0];
                int const ibp = epsilon_perm[ip2][1];
                int const icp = epsilon_perm[ip2][2];

                int const idx_a  = 3 * ialpha  + ia;
                int const idx_ap = 3 * ialphap + iap;

                int const idx_b  = 3 * ibeta  + ib;
                int const idx_bp = 3 * ibetap + ibp;

                int const idx_c  = 3 * igamma  + ic;
                int const idx_cp = 3 * igammap + icp;

                double _Complex const DD   = spinor_field[12+idx_b][_GSI(iix)+2*idx_bp] + I * spinor_field[12+idx_b][_GSI(iix)+2*idx_bp+1];

                double _Complex const UU_1 = spinor_field[idx_a][_GSI(iix)+2*idx_ap] + I * spinor_field[idx_a][_GSI(iix)+2*idx_ap+1];

                double _Complex const UU_2 = spinor_field[idx_c][_GSI(iix)+2*idx_cp] + I * spinor_field[idx_c][_GSI(iix)+2*idx_cp+1];

                corr[1][iit][igammap][igamma] += Gamma_i1.m[ialpha][ibeta] * Gamma_f1.m[ialphap][ibetap] * DD * UU_1 * UU_2 * epsilon_sign[ip] * epsilon_sign[ip2];

                double _Complex const UU_3 = spinor_field[idx_a][_GSI(iix)+2*idx_cp] + I * spinor_field[idx_a][_GSI(iix)+2*idx_cp+1];

                double _Complex const UU_4 = spinor_field[idx_c][_GSI(iix)+2*idx_ap] + I * spinor_field[idx_c][_GSI(iix)+2*idx_ap+1];

                corr[0][iit][igammap][igamma] -= Gamma_i1.m[ialpha][ibeta] * Gamma_f1.m[ialphap][ibetap] * DD * UU_3 * UU_4 * epsilon_sign[ip] * epsilon_sign[ip2];
                   
              }  /* end of loop on sink side color permutations */
            }}}

            }  /* end of loop on ix */
          }  /* end of loop on it */

        }  /* end of loop on source side color permuations */

      }}}


      /***************************************************************************
       * after Timo's version
       ***************************************************************************/

      double _Complex **** corrt = init_4level_ztable ( 2, T_global, 4, 4 );
      setGamma();

      /* for (unsigned int t = 0; t < T; t++) */
      for (unsigned int t = 0; t < 1; t++)
      {

        for (int gamma = 0; gamma < 4; gamma++) {
          for (int gamma_p = 0; gamma_p < 4; gamma_p++) {
    
            double _Complex cn1 = 0., cn2 = 0.;
    
            for (int alpha = 0; alpha < 4; alpha++) {
              for (int a = 0; a < 3; a++) {
    
                // rotatePropsOF(r_ua, alpha, a, omega, 'u', T, Lx, Ly, Lz);
                // rotateProp(s, r_ua, omega, 'u', T, Lx, Ly, Lz);
                double * r_ua = spinor_field[ 3 * alpha + a ];
    
                for (int beta = 0; beta < 4; beta++) {
                  for (int b = 0; b < 3; b++) {
                    // rotatePropsOF(r_db, beta, b, omega, 'd', T, Lx, Ly, Lz);
                    double * r_db = spinor_field[ 12 + 3 * beta + b ];
    
                    for (int c = 0; c < 3; c++) {
                      // rotatePropsOF(r_uc, gamma, c, omega, 'u', T, Lx, Ly, Lz);
                      // rotateProp(s, r_uc, omega, 'u', T, Lx, Ly, Lz);
                      double * r_uc = spinor_field[ 3 * gamma + c ];
    
                      /* for (unsigned int x = 0; x < LX; x++)  */
                      for (unsigned int x = 0; x < 1; x++) 
                      {
                        /* for (unsigned int y = 0; y < LY; y++) */
                        for (unsigned int y = 0; y < 1; y++)
                        {
                          /* for (unsigned int z = 0; z < LZ; z++) */
                          for (unsigned int z = 0; z < 1; z++)
                          {
    
                            for (int alpha_p = 0; alpha_p < 4; alpha_p++) {
                              for (int beta_p = 0; beta_p < 4; beta_p++) {
    
                                for (int a_p = 0; a_p < 3; a_p++) {
                                  for (int b_p = 0; b_p < 3; b_p++) {
                                    for (int c_p = 0; c_p < 3; c_p++) {
    
                                      int temp =
                                          epsilon(a, b, c) * epsilon(a_p, b_p, c_p);
                                      if (temp != 0) {
                                        double _Complex  cnum = cgamma5[alpha_p * 4 + beta_p] * cgamma5[alpha * 4 + beta];

                                        // double _Complex cnum = Gamma_f1.m[0][alpha_p * 4 + beta_p] * Gamma_f1.m[0][alpha * 4 + beta];


                                        // projector(1, gamma, gamma_p);
                                        if (creal(cnum) != 0 || cimag(cnum) != 0) {
                                          int index_bp = (((((t * LX + x) * LY + y) * LZ + z) * 4 + beta_p) * 3 + b_p) * 2;

                                          int index_ap = (((((t * LX + x) * LY + y) * LZ + z) * 4 + alpha_p) * 3 + a_p) * 2;

                                          int index_cp = (((((t * LX + x) * LY + y) * LZ + z) * 4 + gamma_p) * 3 + c_p) * 2;
    
                                          double _Complex cnum_db =
                                                             r_db[index_bp] +
                                                             I * r_db[index_bp + 1],
                                                         cnum_ua =
                                                             r_ua[index_ap] +
                                                             I * r_ua[index_ap + 1],
                                                         cnum_uc =
                                                             r_uc[index_cp] +
                                                             I * r_uc[index_cp + 1];
    
                                          double _Complex intermediate2 =
                                              cnum_db * cnum_ua *
                                              cnum_uc; // first term in bracket
    
                                          intermediate2 *=
                                              cnum * temp; // multiplying with
                                          // Cgamma5 and P+
                                          //
                                          // plus sign here
                                          cn2 += intermediate2;
    
                                          cnum_ua = r_ua[index_cp] +
                                                    I * r_ua[index_cp + 1];
                                          cnum_uc = r_uc[index_ap] +
                                                    I * r_uc[index_ap + 1];
    
                                          double _Complex intermediate =
                                              cnum_db * cnum_ua *
                                              cnum_uc; // second term in bracket
    
                                          intermediate *=
                                              cnum * temp; // multiplying with
                                          // Cgamma5 and P+
                                          //
                                          // minus sign here
                                          cn1 -= intermediate;
    
                                        } else {
                                          /*  */
                                        }
                                      } else {
                                        /*  */
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            corrt[0][t][gamma_p][gamma] = cn1;
            corrt[1][t][gamma_p][gamma] = cn2;

          }  /* gammap */
        }  /* gamma */
      }  /* time */


      for ( unsigned int it = 0; it < T_global; it++ ) {

        for ( int igammap = 0; igammap < 4; igammap++ ) {
        for ( int igamma  = 0; igamma  < 4; igamma++  ) {

          fprintf ( stdout, " %3d  %2d %2d   %25.16e %25.16e    %25.16e %25.16e    %25.16e %25.16e    %25.16e %25.16e\n", it, igammap, igamma,

               creal( corr[0][it][igammap][igamma] ),
               cimag( corr[0][it][igammap][igamma] ),
               creal( corr[1][it][igammap][igamma] ),
               cimag( corr[1][it][igammap][igamma] ),

               creal( corrt[0][it][igammap][igamma] ),
               cimag( corrt[0][it][igammap][igamma] ),
               creal( corrt[1][it][igammap][igamma] ),
               cimag( corrt[1][it][igammap][igamma] ) 

#if 0              
               creal( corr[0][it][igamma][igammap] ),
               cimag( corr[0][it][igamma][igammap] ),
               creal( corr[1][it][igamma][igammap] ),
               cimag( corr[1][it][igamma][igammap] ) 
#endif  /* of if 0 */
               );
        }}
      }

      fini_4level_ztable ( &corr );
      fini_4level_ztable ( &corrt );

    }}
#if 0
#endif  /* of if 0 */

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * clean up
     ***************************************************************************/
    free_fp_field ( &fp  );
    free_fp_field ( &fp2 );
    free_fp_field ( &fp3 );
    fini_2level_dtable ( &v2 );
    fini_3level_dtable ( &vp );
 
    /***************************************************************************/
    /***************************************************************************/

#ifdef HAVE_LHPC_AFF
    /***************************************************************************
     * close AFF writer
     ***************************************************************************/
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[nn_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  }  /* end of loop on source locations */

  fini_2level_dtable ( &spinor_field );


  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  if ( gauge_field_smeared != NULL ) free ( gauge_field_smeared );

  /* free clover matrix terms */
  fini_clover ( );

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
    fprintf(stdout, "# [nn_invert_contract] %s# [nn_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [nn_invert_contract] %s# [nn_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
