/****************************************************
 * njn_invert_contract
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
  int const    gamma_f1_list[gamma_f1_number]            = { 14 }; /*, 11,  8,  2 }; */
  double const gamma_f1_sign[gamma_f1_number]            = { +1 }; /*, +1, -1, -1 }; */
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
  /* fprintf(stdout, "# [njn_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [njn_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [njn_invert_contract] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [njn_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [njn_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[njn_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[njn_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof ( double );
  unsigned int const VOL3 = LX * LY * LZ;

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [njn_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [njn_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[njn_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[njn_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[njn_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

#if 0
  /***********************************************
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {

    alloc_gauge_field ( &gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy ( gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[njn_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
    }  /* end of if N_aoe > 0 */
  }  /* end of if N_Jacobi > 0 */
#endif  /* of if 0 */


  /***********************************************************
   * initialize clover, lmzz and lmzzinv
   ***********************************************************/
  exitstatus = init_clover ( &lmzz, &lmzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njn_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[njn_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [njn_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * allocate spinor_field
   ***********************************************************/
  spinor_field = init_2level_dtable ( 48, _GSI( VOLUME ) );
  if( spinor_field == NULL ) {
    fprintf(stderr, "[njn_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
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
      fprintf(stderr, "[njn_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /**********************************************************
     * propagators with source at gsx
     **********************************************************/

    /***********************************************************
     * up-type point-to-all propagator
     ***********************************************************/

    for ( int ia = 0; ia < 12; ia++ ) {
      double ** spinor_work = init_2level_dtable ( 3, _GSI(VOLUME+RAND) );

      memset ( spinor_work[0], 0, sizeof_spinor_field );
      memset ( spinor_work[1], 0, sizeof_spinor_field );

      unsigned int const ixsrc = g_ipt[gsx[0]][gsx[1]][gsx[2]][gsx[3]];

      spinor_work[0][_GSI(ixsrc)+2*ia] = 1.;

      exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], _OP_ID_UP );
      if(exitstatus != 0) {
        fprintf(stderr, "[njn_invert_contract] Error from tmLQCD_invert, status was %d\n", exitstatus);
        EXIT(12);
      }

      memcpy ( spinor_field[ia], spinor_work[1], sizeof_spinor_field );

      fini_2level_dtable ( &spinor_work );

    }

    /***************************************************************************
     * sequential source
     ***************************************************************************/
    for ( int ia = 0; ia < 12; ia++ ) {

      double ** spinor_work = init_2level_dtable ( 3, _GSI(VOLUME+RAND) );

      int const tseq = ( gsx[0] + g_src_snk_time_separation ) % T_global;

      /***************************************************************************
       * forward
       ***************************************************************************/
      memcpy ( spinor_work[0], spinor_field[ia], sizeof_spinor_field );
      memset ( spinor_work[1], 0, sizeof_spinor_field );

      for ( unsigned int ix = 0; ix < VOL3; ix++ ) {

        unsigned int const iy = ix + VOL3 * tseq;
        unsigned int const iy2 = g_iup[iy][0];
        double sp1[24];

        _fv_eq_cm_ti_fv ( spinor_work[1]+_GSI(iy), gauge_field_with_phase + _GGI(iy,0), spinor_work[0]+_GSI(iy) );
        _fv_eq_gamma_ti_fv ( sp1, 0, spinor_work[1]+_GSI(iy) );
        _fv_mi_eq_fv ( spinor_work[1]+_GSI(iy), sp1 );
        _fv_ti_eq_re ( spinor_work[1]+_GSI(iy), -0.5 );

      }

      exitstatus = _TMLQCD_INVERT ( spinor_work[0], spinor_work[1], _OP_ID_UP );
      if(exitstatus != 0) {
        fprintf(stderr, "[njn_invert_contract] Error from tmLQCD_invert, status was %d\n", exitstatus);
        EXIT(12);
      }

      memcpy ( spinor_field[12+ia], spinor_work[0], sizeof_spinor_field );

      /***************************************************************************
       * backward
       ***************************************************************************/
      memcpy ( spinor_work[0], spinor_field[ia], sizeof_spinor_field );
      memset ( spinor_work[1], 0, sizeof_spinor_field );

      for ( unsigned int ix = 0; ix < VOL3; ix++ ) {

        unsigned int const iy = ix + VOL3 * tseq;
        unsigned int const iy2 = g_idn[iy][0];
        double sp1[24];

        _fv_eq_cm_dag_ti_fv ( spinor_work[1]+_GSI(iy), gauge_field_with_phase + _GGI(iy2,0), spinor_work[0]+_GSI(iy2) );
        _fv_eq_gamma_ti_fv ( sp1, 0, spinor_work[1]+_GSI(iy) );
        _fv_pl_eq_fv ( spinor_work[1]+_GSI(iy), sp1 );
        _fv_ti_eq_re ( spinor_work[1]+_GSI(iy), 0.5 );

      }

      exitstatus = _TMLQCD_INVERT ( spinor_work[0], spinor_work[1], _OP_ID_UP );
      if(exitstatus != 0) {
        fprintf(stderr, "[njn_invert_contract] Error from tmLQCD_invert, status was %d\n", exitstatus);
        EXIT(12);
      }

      memcpy ( spinor_field[24+ia], spinor_work[0], sizeof_spinor_field );


      fini_2level_dtable ( &spinor_work );
    }

    for ( int isc = 0; isc < 12; isc++ ) {

      for ( int idt = 0; idt <= g_src_snk_time_separation; idt++ ) {

        int const it = ( gsx[0] + idt ) % T_global;

        for ( int ix = 0; ix < LX; ix++ ) {
        for ( int iy = 0; iy < LY; iy++ ) {
        for ( int iz = 0; iz < LZ; iz++ ) {

          unsigned int const iix = g_ipt[it][ix][iy][iz];
          unsigned int const iix2 = g_iup[iix][0];

          fprintf ( stdout, "# [njn_invert_contract] sc %2d x %3d %3d %3d %3d\n", isc, it, ix, iy, iz );

          for ( int ia = 0; ia < 12; ia++ ) {
            fprintf ( stdout, "%3d  %25.16e %25.16e %25.16e %25.16e %25.16e %25.16e\n",
                it, ix, iy, iz, ia,
                spinor_field[   isc][_GSI(iix) +2*ia], spinor_field[   isc][_GSI(iix) +2*ia+1],
                spinor_field[12+isc][_GSI(iix) +2*ia], spinor_field[12+isc][_GSI(iix) +2*ia+1],
                spinor_field[24+isc][_GSI(iix2)+2*ia], spinor_field[24+isc][_GSI(iix2)+2*ia+1] );
          }

        }}}
      }
    }

  }  /* end of loop on source locations */

  fini_2level_dtable ( &spinor_field );


  /***************************************************************************/
  /***************************************************************************/

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

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
    fprintf(stdout, "# [njn_invert_contract] %s# [njn_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [njn_invert_contract] %s# [njn_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
