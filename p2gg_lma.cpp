/****************************************************
 * p2gg_lma.c
 *
 * Wed Jul 12 09:27:00 CEST 2017
 *
 * - originally copied from p2gg_caa_lma.cpp
 * - only low-mode contribution to correlators
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
#include "matrix_init.h"
#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -w                  : check position space WI [default false]\n");
  EXIT(0);
}

int dummy_eo_solver (double * const propagator, double * const source, const int op_id) {
  memcpy(propagator, source, _GSI(VOLUME)/2*sizeof(double) );
  return(0);
}

int main(int argc, char **argv) {
  
  /*
   * sign for g5 Gamma^\dagger g5
   *                                                0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   * */
  const int sequential_source_gamma_id_sign[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };

  const char outfile_prefix[] = "p2gg_lma";

  int c;
  int filename_set = 0;
  int isource_location;
  int gsx[4], sx[4];
  int check_position_space_WI=0;
  int exitstatus;
  int source_proc_id = 0;
  int iseq_source_momentum;
  int isequential_source_gamma_id, isequential_source_timeslice;
  int io_proc = -1;
  int evecs_num = 0;
  int check_propagator_residual = 0;
  unsigned int Vhalf;
  size_t sizeof_eo_spinor_field;
  double **eo_spinor_field=NULL, **eo_spinor_work=NULL, *eo_evecs_block=NULL;
  double **eo_evecs_field=NULL;
  double **cvc_tensor_eo = NULL, contact_term[2][8];
  double ***cvc_tp = NULL;
  double *evecs_eval = NULL, *evecs_lambdainv=NULL, *evecs_4kappasqr_lambdainv = NULL;
  char filename[100];
  double ratime, retime;
  double **mzz[2], **mzzinv[2];
  double *gauge_field_with_phase = NULL;
  double ***eo_source_buffer = NULL;



#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
  char aff_tag[400];
#endif

  // in order to be able to initialise QMP if QPhiX is used, we need
  // to allow tmLQCD to intialise MPI via QMP
  // the input file has to be read 
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_init_parallel_and_read_input(argc, argv, 1, "invert.input");
#else
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
#endif

  while ((c = getopt(argc, argv, "cwh?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_position_space_WI = 1;
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
  ratime = _GET_TIME;

  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [p2gg_lma] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_lma] calling tmLQCD wrapper init functions\n");

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

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_lma] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_lma] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_lma] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg_lma] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  Vhalf = VOLUME / 2;
  sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [p2gg_lma] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg_lma] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg_lma] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg_lma] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[p2gg_lma] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_lma] Error from tmLQCD_get_deflator_params, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(9);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [p2gg_lma] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [p2gg_lma] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [p2gg_lma] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [p2gg_lma] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block == NULL) {
    fprintf(stderr, "[p2gg_lma] Error, eo_evecs_block is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(10);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[p2gg_lma] Error, dimension of eigenspace is zero %s %d\n", __FILE__, __LINE__);
    EXIT(11);
  }

  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[p2gg_lma] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  evecs_eval                = (double*)malloc(evecs_num*sizeof(double));
  evecs_lambdainv           = (double*)malloc(evecs_num*sizeof(double));
  evecs_4kappasqr_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if(    evecs_eval                == NULL 
      || evecs_lambdainv           == NULL 
      || evecs_4kappasqr_lambdainv == NULL 
    ) {
    fprintf(stderr, "[p2gg_lma] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for( int i = 0; i < evecs_num; i++) {
    evecs_eval[i]                = ((double*)(g_tmLQCD_defl.evals))[2*i];
    evecs_lambdainv[i]           = 2.* g_kappa / evecs_eval[i];
    evecs_4kappasqr_lambdainv[i] = 4.* g_kappa * g_kappa / evecs_eval[i];
    if( g_cart_id == 0 ) fprintf(stdout, "# [p2gg_lma] eval %4d %16.7e\n", i, evecs_eval[i] );
  }

#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */

  /*************************************************
   * allocate memory for the eigenvector fields
   *************************************************/
  eo_evecs_field = (double**)calloc(evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block;
  for( int i = 1; i < evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + _GSI(Vhalf);

  /*************************************************
   * allocate memory for eo spinor fields 
   * WITH HALO
   *************************************************/
  exitstatus = init_2level_buffer ( &eo_spinor_work, 6, _GSI((VOLUME+RAND)/2) );
  if ( exitstatus != 0) {
    fprintf(stderr, "[p2gg_lma] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(1);
  }

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_lma] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[p2gg_lma] Error from init_clover, status was %d\n", exitstatus);
    EXIT(1);
  }

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [p2gg_lma] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [p2gg_lma] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  if(io_proc == 2) {
    if(g_tr_id != 0) {
      fprintf(stderr, "[p2gg_lma] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************************
   * allocate eo_spinor_field
   ***********************************************************/
  exitstatus = init_2level_buffer ( &eo_spinor_field, 360, _GSI(Vhalf));
  if( exitstatus != 0 ) {
    fprintf(stderr, "[p2gg_lma] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(123);
  }
  
  /***********************************************************
   ***********************************************************
   **
   ** loop on source locations
   **
   ***********************************************************
   ***********************************************************/
  for(isource_location=0; isource_location < g_source_location_number; isource_location++) {

    /***********************************************************
     * determine source coordinates, find out, if source_location is in this process
     ***********************************************************/
    gsx[0] = g_source_coords_list[isource_location][0];
    gsx[1] = g_source_coords_list[isource_location][1];
    gsx[2] = g_source_coords_list[isource_location][2];
    gsx[3] = g_source_coords_list[isource_location][3];

    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_lma] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***********************************************************
     * init Usource and source_proc_id
     *
     * NOTE: here it must be g_gauge_field, 
     *       NOT gauge_field_with_phase,
     *       because the boundary phase is multiplied at
     *       source inside init_contract_cvc_tensor_usource
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
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [p2gg_lma] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_lma] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#endif

    /**********************************************************
     **********************************************************
     **
     ** low-mode propagators with source at gsx and gsx + mu
     **
     **********************************************************
     **********************************************************/

    for( int mu = 0; mu < 5; mu++)
    {  /* loop on shifts in direction mu */
      /**********************************************************
       * shifted source coords and source proc coords
       **********************************************************/
      int g_shifted_source_coords[4], shifted_source_coords[4], shifted_source_proc_id = 0;
      memcpy( g_shifted_source_coords, gsx, 4*sizeof(int));
      switch(mu) {
        case 0: g_shifted_source_coords[0] = ( g_shifted_source_coords[0] + 1 ) % T_global; break;;
        case 1: g_shifted_source_coords[1] = ( g_shifted_source_coords[1] + 1 ) % LX_global; break;;
        case 2: g_shifted_source_coords[2] = ( g_shifted_source_coords[2] + 1 ) % LY_global; break;;
        case 3: g_shifted_source_coords[3] = ( g_shifted_source_coords[3] + 1 ) % LZ_global; break;;
      }

      /* source info for shifted source location */
      if( (exitstatus = get_point_source_info (g_shifted_source_coords, shifted_source_coords, &shifted_source_proc_id) ) != 0 ) {
        fprintf(stderr, "[p2gg_lma] Error from get_point_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(15);
      }

      /**********************************************************
       * A^-1 g5 for up-type propagators from sub-space
       * inversion
       **********************************************************/
      for( int is = 0; is < 12; is++ )
      {

        /* A^-1 g5 source */
        exitstatus = init_clover_eo_spincolor_pointsource_propagator (eo_spinor_field[mu*12+is], eo_spinor_field[mu*12+60+is],
            g_shifted_source_coords, is, gauge_field_with_phase, g_mzzinv_up[0], (int)(shifted_source_proc_id == g_cart_id), eo_spinor_work[0]);
        if(exitstatus != 0 ) {
          fprintf(stderr, "[p2gg_lma] Error from init_eo_spincolor_pointsource_propagator; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(18);
        }
      }  /* end of loop on spin-color */

      /**********************************************************
       * A^-1 g5 for dn-type propagators from sub-space
       * inversion
       **********************************************************/
      for( int is = 0; is < 12; is++ )
      {
        /* A^-1 g5 source */
        exitstatus = init_clover_eo_spincolor_pointsource_propagator (eo_spinor_field[120+mu*12+is], eo_spinor_field[180+mu*12+is],
            g_shifted_source_coords, is, gauge_field_with_phase, g_mzzinv_dn[0], (int)(shifted_source_proc_id == g_cart_id), eo_spinor_work[0]);
        if(exitstatus != 0 ) {
          fprintf(stderr, "[p2gg_lma] Error from init_eo_spincolor_pointsource_propagator; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(18);
        }
      }  /* end of loop on spin-color */


    }  /* end of loop on shift direction mu */

    /**********************************************************/
    /**********************************************************/

    /**********************************************************
     * up-type inversion on deflation subspace
     **********************************************************/
    if ( check_propagator_residual ) {
      exitstatus = init_3level_buffer( &eo_source_buffer, 2, 60, _GSI(Vhalf));
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_lma] Error from init_3level_buffer; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(18);
      }
    }  /* end of if check_propagator_residual */


    if ( check_propagator_residual ) {
      memcpy ( eo_source_buffer[0][0], eo_spinor_field[ 0] , 60 * sizeof_eo_spinor_field );
      memcpy ( eo_source_buffer[1][0], eo_spinor_field[60] , 60 * sizeof_eo_spinor_field );
    }
    exitstatus = Q_clover_eo_invert_subspace ( 
        &(eo_spinor_field[0]),   &(eo_spinor_field[60]),
        &(eo_spinor_field[0]),   &(eo_spinor_field[60]),
        60, eo_evecs_block, evecs_lambdainv, evecs_num, gauge_field_with_phase, mzz,  mzzinv, 0, eo_spinor_work
    );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_lma] Error from Q_clover_eo_invert_subspace, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(28);
    }

    if ( check_propagator_residual ) {
      exitstatus = check_subspace_propagator_clover_eo(
          &(eo_spinor_field[0]), &(eo_spinor_field[60]),
          eo_source_buffer[0], eo_source_buffer[1],
          60, eo_evecs_block, evecs_4kappasqr_lambdainv, evecs_num, gauge_field_with_phase, mzz, mzzinv, 0);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_lma] Error fromeck_subspace_propagator_clover_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(28);
      }
    }  /* end of if check_propagator_residual */


    /**********************************************************
     * dn-type inversion on deflation subspace
     **********************************************************/
    if ( check_propagator_residual ) {
      memcpy ( eo_source_buffer[0][0], eo_spinor_field[120] , 60 * sizeof_eo_spinor_field );
      memcpy ( eo_source_buffer[1][0], eo_spinor_field[180] , 60 * sizeof_eo_spinor_field );
    }
    exitstatus = Q_clover_eo_invert_subspace ( 
        &(eo_spinor_field[120]),   &(eo_spinor_field[180]),
        &(eo_spinor_field[120]),   &(eo_spinor_field[180]),
        60, eo_evecs_block, evecs_lambdainv, evecs_num, gauge_field_with_phase, mzz,  mzzinv, 1, eo_spinor_work
    );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_lma] Error from Q_clover_eo_invert_subspace, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(28);
    }

    if ( check_propagator_residual ) {
      exitstatus = check_subspace_propagator_clover_eo(
          &(eo_spinor_field[120]), &(eo_spinor_field[180]),
          eo_source_buffer[0], eo_source_buffer[1],
          60, eo_evecs_block, evecs_4kappasqr_lambdainv, evecs_num, gauge_field_with_phase, mzz, mzzinv, 1);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_lma] Error fromeck_subspace_propagator_clover_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(28);
      }
    }  /* end of if check_propagator_residual */

    if ( check_propagator_residual ) {
      fini_3level_buffer ( &eo_source_buffer );
    }  /* end of if check_propagator_residual */

    /**********************************************************/
    /**********************************************************/

    /* allocate memory for contractions, initialize */
    exitstatus = init_2level_buffer( &cvc_tensor_eo, 2, 32*Vhalf);
    if( exitstatus != 0) {
      fprintf(stderr, "[p2gg_lma] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(24);
    }

    /***************************************************************************
     * low-mode tensor
     ***************************************************************************/
    memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
    memset(contact_term[0], 0, 8*sizeof(double));
    /* contraction */
    contract_cvc_tensor_eo ( cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[0], &(eo_spinor_field[120]), &(eo_spinor_field[180]),
       &(eo_spinor_field[0]), &(eo_spinor_field[60]), gauge_field_with_phase );

    /* subtract contact term */
    cvc_tensor_eo_subtract_contact_term (cvc_tensor_eo, contact_term[0], gsx, (int)( source_proc_id == g_cart_id ) );

    /* momentum projections */
    exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cvc_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg_lma] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(26);
    }
    /* write results to file */
    sprintf(aff_tag, "/hvp/lm/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = cvc_tensor_tp_write_to_aff_file ( cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_lma] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }
    fini_3level_buffer(&cvc_tp);

    /* check position space WI */
    if(check_position_space_WI) {
      exitstatus = cvc_tensor_eo_check_wi_position_space ( cvc_tensor_eo );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_lma] Error from cvc_tensor_eo_check_wi_position_space for lm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }
    }

    fini_2level_buffer( &cvc_tensor_eo );

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * local - cvc 2-point
     ***************************************************************************/
    sprintf(aff_tag, "/local-cvc/lm/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_local_cvc_2pt_eo (
        &(eo_spinor_field[120]), &(eo_spinor_field[180]),
        &(eo_spinor_field[0]), &(eo_spinor_field[60]),
        g_sequential_source_gamma_id_list, g_sequential_source_gamma_id_number, g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_lma] Error from contract_local_cvc_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * local - local 2-point
     ***************************************************************************/
    sprintf(aff_tag, "/local-local/lm/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[168]), &(eo_spinor_field[228]),
       &(eo_spinor_field[ 48]), &(eo_spinor_field[108]),
       g_sequential_source_gamma_id_list, g_sequential_source_gamma_id_number,
       g_sequential_source_gamma_id_list, g_sequential_source_gamma_id_number,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_lma] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * P -> gamma gamma contractions
     ***************************************************************************/

    /***************************************************************************
     * loop on sequential source gamma matrices
     ***************************************************************************/
    for(iseq_source_momentum=0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) {

      g_seq_source_momentum[0] = g_seq_source_momentum_list[iseq_source_momentum][0];
      g_seq_source_momentum[1] = g_seq_source_momentum_list[iseq_source_momentum][1];
      g_seq_source_momentum[2] = g_seq_source_momentum_list[iseq_source_momentum][2];

      if(g_cart_id == 0) fprintf(stdout, "# [p2gg_lma] using sequential source momentum no. %2d = (%d, %d, %d)\n", iseq_source_momentum,
          g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);

      /***************************************************************************
       * loop on sequential source gamma matrices
       ***************************************************************************/
      for(isequential_source_gamma_id=0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++) {

        int sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];
        if(g_cart_id == 0) fprintf(stdout, "# [p2gg_lma] using sequential source gamma id no. %2d = %d\n", isequential_source_gamma_id, sequential_source_gamma_id);

        /***************************************************************************
         * loop on sequential source time slices
         ***************************************************************************/
        for(isequential_source_timeslice=0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++) {

          g_sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];
          /* shift sequential source timeslice by source timeslice gsx[0] */
          int g_shifted_sequential_source_timeslice = ( gsx[0] + g_sequential_source_timeslice + T_global ) % T_global;

          if(g_cart_id == 0) fprintf(stdout, "# [p2gg_lma] using sequential source timeslice %d / %d\n", g_sequential_source_timeslice, g_shifted_sequential_source_timeslice);

          /* allocate memory for contractions, initialize */
          exitstatus = init_2level_buffer( &cvc_tensor_eo, 2, 32*Vhalf);
          if( exitstatus != 0) {
            fprintf(stderr, "[p2gg_lma] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(24);
          }
          memset(contact_term[0], 0, 8*sizeof(double));
          memset(contact_term[1], 0, 8*sizeof(double));


          for( int iflavor = 1; iflavor >= 0; iflavor-- )
          {

            /* flavor-dependent sequential source momentum */
            int seq_source_momentum[3] = { (1 - 2*iflavor) * g_seq_source_momentum[0],
                                           (1 - 2*iflavor) * g_seq_source_momentum[1],
                                           (1 - 2*iflavor) * g_seq_source_momentum[2] };

            if(g_cart_id == 0) fprintf(stdout, "# [p2gg_lma] using flavor-dependent sequential source momentum (%d, %d, %d)\n",
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2]);


            /***************************************************************************
             * prepare sequential sources
             ***************************************************************************/
            for( int is = 0; is < 60; is++ ) {
              int eo_spinor_field_id_e     = iflavor * 120 + is;
              int eo_spinor_field_id_o     = eo_spinor_field_id_e + 60;
              int eo_seq_spinor_field_id_e = 240 + is;
              int eo_seq_spinor_field_id_o = eo_seq_spinor_field_id_e + 60;

              exitstatus = init_clover_eo_sequential_source(
                  eo_spinor_field[ eo_seq_spinor_field_id_e ], eo_spinor_field[ eo_seq_spinor_field_id_o ],
                  eo_spinor_field[ eo_spinor_field_id_e     ], eo_spinor_field[ eo_spinor_field_id_o     ] ,
                  g_shifted_sequential_source_timeslice, gauge_field_with_phase, mzzinv[iflavor][0],
                  seq_source_momentum, sequential_source_gamma_id, eo_spinor_work[0]);
              if(exitstatus != 0) {
                fprintf(stderr, "[p2gg_lma] Error from init_clover_eo_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(25);
              }

            }  /* end of loop on spin-color and shift direction */

            /**********************************************************
             * sequential inversion on deflation subspace of type iflavor
             **********************************************************/
            if ( check_propagator_residual ) {
              exitstatus = init_3level_buffer( &eo_source_buffer, 2, 60, _GSI(Vhalf));
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[p2gg_lma] Error from init_3level_buffer; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(18);
              }
            }  /* end of if check_propagator_residual */


            if ( check_propagator_residual ) {
              memcpy ( eo_source_buffer[0][0], eo_spinor_field[240] , 60 * sizeof_eo_spinor_field );
              memcpy ( eo_source_buffer[1][0], eo_spinor_field[300] , 60 * sizeof_eo_spinor_field );
            }
            exitstatus = Q_clover_eo_invert_subspace ( 
                &(eo_spinor_field[240]),   &(eo_spinor_field[300]),
                &(eo_spinor_field[240]),   &(eo_spinor_field[300]),
                60, eo_evecs_block, evecs_lambdainv, evecs_num, gauge_field_with_phase, mzz,  mzzinv, iflavor, eo_spinor_work
            );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_lma] Error from Q_clover_eo_invert_subspace, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(28);
            }

            if ( check_propagator_residual ) {
              exitstatus = check_subspace_propagator_clover_eo(
                  &(eo_spinor_field[240]), &(eo_spinor_field[300]),
                  eo_source_buffer[0], eo_source_buffer[1],
                  60, eo_evecs_block, evecs_4kappasqr_lambdainv, evecs_num, gauge_field_with_phase, mzz, mzzinv, iflavor);
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[p2gg_lma] Error fromeck_subspace_propagator_clover_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(28);
              }

              fini_3level_buffer ( &eo_source_buffer );
            }  /* end of if check_propagator_residual */

            /***************************************************************************
             * P - cvc - cvc tensor
             ***************************************************************************/

            if ( iflavor == 1 ) {
              memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
              memset(contact_term[1], 0, 8*sizeof(double));
            } else if ( iflavor == 0 ) {
              /* note: 4 x 4 x 2 x VOLUME/2 COMPLEX elements in cvc_tensor_eo[0/1] */
              complex_field_eq_complex_field_conj_ti_re (cvc_tensor_eo[0], (double)sequential_source_gamma_id_sign[ sequential_source_gamma_id ], 16*VOLUME );
              memset( contact_term[0], 0, 8*sizeof(double) );
            }

            /* flavor-dependent aff tag  */
            sprintf(aff_tag, "/pgg/lm/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d",
                                  gsx[0], gsx[1], gsx[2], gsx[3], 
                                  g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
                                  sequential_source_gamma_id, g_sequential_source_timeslice, iflavor);

            /* contraction */
            contract_cvc_tensor_eo ( 
                cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[0], 
                &(eo_spinor_field[ ( 1 - iflavor ) * 120]), &(eo_spinor_field[ ( 1 - iflavor ) * 120 + 60]),
                &(eo_spinor_field[240]), &(eo_spinor_field[300]),
                gauge_field_with_phase );

            /* write the contact term to file */
            exitstatus = cvc_tensor_eo_write_contact_term_to_aff_file (contact_term[0], affw, aff_tag, io_proc );
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg_lma] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(26);
            }

          }  /* end of loop on flavor */

          /***************************************************************************/
          /***************************************************************************/

          /* subtract contact term */
          cvc_tensor_eo_subtract_contact_term (cvc_tensor_eo, contact_term[0], gsx, (int)( source_proc_id == g_cart_id ) );

          /* momentum projections */
          exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cvc_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg_lma] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(26);
          }
             
          /* flavor-dependent aff tag  */
          sprintf(aff_tag, "/pgg/lm/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d",
                                gsx[0], gsx[1], gsx[2], gsx[3], 
                                g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
                                sequential_source_gamma_id, g_sequential_source_timeslice);

          /* write results to file */
          exitstatus = cvc_tensor_tp_write_to_aff_file ( cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if(exitstatus != 0 ) {
            fprintf(stderr, "[p2gg_lma] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(45);
          }
          fini_3level_buffer(&cvc_tp);

          /* check position space WI */
          if(check_position_space_WI) {
            exitstatus = cvc_tensor_eo_check_wi_position_space ( cvc_tensor_eo );
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg_lma] Error from cvc_tensor_eo_check_wi_position_space for lm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(38);
            }
          }

#if 0
          /* TEST */
          sprintf(filename, "p2gg_x.t%.2dx%.2dy%.2dz%.2d.tseq%.2d.g%.2d.px%.2dpy%.2dpz%.2d.%.4d.ascii", 
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_sequential_source_timeslice, sequential_source_gamma_id,
              g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
              Nconf );
          FILE *ofs = fopen( filename, "w" );
          for ( int x0 = 0; x0 < T; x0++ ) {
          for ( int x1 = 0; x1 < LX; x1++ ) {
          for ( int x2 = 0; x2 < LY; x2++ ) {
          for ( int x3 = 0; x3 < LZ; x3++ ) {
            unsigned int ix = g_ipt[x0][x1][x2][x3];
            unsigned int ixeosub = g_lexic2eosub[ix];
            int ieo = 1 - g_iseven[ix];
            fprintf(ofs, "# t = %2d, x = %2d, y = %2d, z = %2d, ieo = %d\n", x0, x1, x2, x3, ieo);
            for ( int mu = 0; mu < 16; mu++ ) {
              /* fprintf(ofs, "%3d %25.16e %25.16e\n", mu,  cvc_tensor_eo[ieo][ 2 * ( 16*ixeosub + mu ) ], cvc_tensor_eo[ieo][ 2 * ( 16*ixeosub + mu ) + 1 ] ); */
              fprintf(ofs, "%3d %25.16e %25.16e\n", mu,  cvc_tensor_eo[ieo][ _GWI(mu,ixeosub,Vhalf)], cvc_tensor_eo[ieo][ _GWI(mu,ixeosub,Vhalf) + 1 ] );
            }
          }}}}
          fclose(ofs);
#endif  /* of if 0 */

          fini_2level_buffer( &cvc_tensor_eo );

          /***************************************************************************/
          /***************************************************************************/

        }  /* end of loop on sequential source timeslices */
      }  /* end of loop on sequential source gamma id */
    }  /* end of loop on sequential source momentum */


#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_lma] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
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

  fini_2level_buffer ( &eo_spinor_field );
  fini_2level_buffer ( &eo_spinor_work );

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(eo_evecs_block);
#else
  exitstatus = tmLQCD_fini_deflator(_OP_ID_UP);
#endif
  free(eo_evecs_field);

  free ( evecs_eval );
  free ( evecs_lambdainv );

  /* free clover matrix terms */
  fini_clover ();

  free_geometry();

  retime = _GET_TIME;

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
    fprintf(stdout, "# [p2gg_lma] time for p2gg_lma = %e seconds\n", retime - ratime );
    fprintf(stdout, "# [p2gg_lma] %s# [p2gg_lma] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_lma] %s# [p2gg_lma] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
