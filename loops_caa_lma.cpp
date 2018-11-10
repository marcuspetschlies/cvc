/****************************************************
 * loops_caa_lma.cpp
 *
 * Wed Jul 26 15:12:32 CEST 2017
 *
 * - originally copied from p2gg_caa_lma.cpp
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
#include "matrix_init.h"
#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#define _SQR(_a) ((_a)*(_a))

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc loop contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc    [default cvc.input]\n");
  fprintf(stdout, "          -w                  : check position space WI   [default false]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

int dummy_eo_solver (double * const propagator, double * const source, const int op_id) {
  memcpy(propagator, source, _GSI(VOLUME)/2*sizeof(double) );
  return(0);
}


int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "loops";

  int c;
  int iflavor;
  int filename_set = 0;
  int isource_location;
  unsigned int ix;
  int gsx[4], sx[4];
  int check_position_space_WI=0;
  int exitstatus;
  int source_proc_id = 0;
  int no_eo_fields = 0;
  int io_proc = -1;
  int evecs_num = 0;
  int check_propagator_residual = 0;
  unsigned int Vhalf;
  size_t sizeof_eo_spinor_field;
  double **eo_spinor_field=NULL, **eo_spinor_work=NULL, *eo_evecs_block=NULL, *eo_sample_block=NULL, **eo_sample_field = NULL;
  double **eo_stochastic_source = NULL, ***eo_stochastic_propagator = NULL;
  double **eo_evecs_field=NULL;
  double ***cvc_loop_eo = NULL;
  // double ***cvc_tp = NULL;
  double *evecs_eval = NULL, *evecs_lambdainv=NULL, *evecs_4kappasqr_lambdainv = NULL;
  char filename[100];
  double ratime, retime;
  double **mzz[2], **mzzinv[2];
  double *gauge_field_with_phase = NULL;
  double ***eo_source_buffer = NULL;
  double ***cvc_loop_eo_wi = NULL;
  int nev_step_size = 10, sample_step_size = 10;

#ifdef HAVE_MPI
  MPI_Status mstatus;
#endif


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

  while ((c = getopt(argc, argv, "cwh?f:n:s:")) != -1) {
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
    case 'n':
      nev_step_size = atoi (optarg);
      break;
    case 's':
      sample_step_size = atoi (optarg);
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
  if(filename_set==0) strcpy(filename, "cvc.input");
  /* fprintf(stdout, "# [loops_caa_lma] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [loops_caa_lma] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [loops_caa_lma] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [loops_caa_lma] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [loops_caa_lma] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[loops_caa_lma] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[loops_caa_lma] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [loops_caa_lma] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [loops_caa_lma] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[loops_caa_lma] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[loops_caa_lma] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[loops_caa_lma] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[loops_caa_lma] Error from tmLQCD_get_deflator_params, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(9);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [loops_caa_lma] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [loops_caa_lma] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [loops_caa_lma] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [loops_caa_lma] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block == NULL) {
    fprintf(stderr, "[loops_caa_lma] Error, eo_evecs_block is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(10);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[loops_caa_lma] Error, dimension of eigenspace is zero %s %d\n", __FILE__, __LINE__);
    EXIT(11);
  }

  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[loops_caa_lma] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  evecs_eval                = (double*)malloc(evecs_num*sizeof(double));
  evecs_lambdainv           = (double*)malloc(evecs_num*sizeof(double));
  evecs_4kappasqr_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if(    evecs_eval                == NULL 
      || evecs_lambdainv           == NULL 
      || evecs_4kappasqr_lambdainv == NULL 
    ) {
    fprintf(stderr, "[loops_caa_lma] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for( int i = 0; i < evecs_num; i++) {
    evecs_eval[i]                = ((double*)(g_tmLQCD_defl.evals))[2*i];
    evecs_lambdainv[i]           = 2.* g_kappa / evecs_eval[i];
    evecs_4kappasqr_lambdainv[i] = 4.* g_kappa * g_kappa / evecs_eval[i];
    if( g_cart_id == 0 ) fprintf(stdout, "# [loops_caa_lma] eval %4d %16.7e\n", i, evecs_eval[i] );
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
  no_eo_fields = 6;
  exitstatus = init_2level_buffer ( &eo_spinor_work, 6, _GSI((VOLUME+RAND)/2) );
  if ( exitstatus != 0) {
    fprintf(stderr, "[loops_caa_lma] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(1);
  }

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[loops_caa_lma] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[] Error from init_clover, status was %d\n");
    EXIT(1);
  }

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [loops_caa_lma] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [loops_caa_lma] proc%.4d is send process\n", g_cart_id);
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
      fprintf(stderr, "[loops_caa_lma] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************************
   * allocate eo_spinor_field
   ***********************************************************/
  exitstatus = init_2level_buffer ( &eo_spinor_field, 48, _GSI(Vhalf));
  if( exitstatus != 0 ) {
    fprintf(stderr, "[loops_caa_lma] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(123);
  }


#if 0
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
      fprintf(stderr, "[loops_caa_lma] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
      fprintf(stdout, "# [loops_caa_lma] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[loops_caa_lma] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#endif

    if ( g_cart_id == 0 ) {
      exitstatus = init_3level_buffer( &cvc_loop_eo_wi, 2, 4, 2);
      if( exitstatus != 0) {
        fprintf(stderr, "[loops_caa_lma] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(24);
      }
    }

    /**********************************************************
     **********************************************************
     **
     ** propagators with source at gsx and gsx + mu
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
        case 0: g_shifted_source_coords[0] = ( g_shifted_source_coords[0] - 1 + T_global  ) % T_global; break;;
        case 1: g_shifted_source_coords[1] = ( g_shifted_source_coords[1] - 1 + LX_global ) % LX_global; break;;
        case 2: g_shifted_source_coords[2] = ( g_shifted_source_coords[2] - 1 + LY_global ) % LY_global; break;;
        case 3: g_shifted_source_coords[3] = ( g_shifted_source_coords[3] - 1 + LZ_global ) % LZ_global; break;;
      }

      /* source info for shifted source location */
      if( (exitstatus = get_point_source_info (g_shifted_source_coords, shifted_source_coords, &shifted_source_proc_id) ) != 0 ) {
        fprintf(stderr, "[loops_caa_lma] Error from get_point_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(15);
      }

      /**********************************************************
       * up-type propagators
       **********************************************************/
      exitstatus = point_to_all_fermion_propagator_clover_eo ( &(eo_spinor_field[0]), &(eo_spinor_field[12]),  _OP_ID_UP,
          g_shifted_source_coords, gauge_field_with_phase, g_mzz_up, g_mzzinv_up, check_propagator_residual, eo_spinor_work );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[loops_caa_lma] Error from point_to_all_fermion_propagator_clover_eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(21);
      }

      /**********************************************************
       * dn-type propagators
       **********************************************************/
      exitstatus = point_to_all_fermion_propagator_clover_eo ( &(eo_spinor_field[24]), &(eo_spinor_field[36]),  _OP_ID_DN,
          g_shifted_source_coords, gauge_field_with_phase, g_mzz_dn, g_mzzinv_dn, check_propagator_residual, eo_spinor_work );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[loops_caa_lma] Error from point_to_all_fermion_propagator_clover_eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(21);
      }

      /* allocate memory for contractions, initialize */
      exitstatus = init_3level_buffer( &cvc_loop_eo, 2, 4, 2*Vhalf);
      if( exitstatus != 0) {
        fprintf(stderr, "[loops_caa_lma] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(24);
      }

      contract_cvc_loop_eo ( cvc_loop_eo, &(eo_spinor_field[0]), &(eo_spinor_field[12]), &(eo_spinor_field[24]), &(eo_spinor_field[36]), gauge_field_with_phase );

      if ( shifted_source_proc_id == g_cart_id ) {
        fprintf(stdout, "# [loops_caa_lma] x %3d %3d %3d %3d\n", g_shifted_source_coords[0], g_shifted_source_coords[1], g_shifted_source_coords[2], g_shifted_source_coords[3] );
        unsigned int ix = g_ipt[shifted_source_coords[0]][shifted_source_coords[1]][shifted_source_coords[2]][shifted_source_coords[3]];
        unsigned int ixeosub = g_lexic2eosub[ix];
        int ieo = g_iseven[ix] ? 0 : 1;

        for ( int k = 0; k < 4; k++ ) {
          fprintf(stdout, "%3d %25.16e%25.16e\n", k, cvc_loop_eo[ieo][k][2*ixeosub], cvc_loop_eo[ieo][k][2*ixeosub+1]);
        }
      }

      if ( shifted_source_proc_id == 0 ) {
        if ( g_cart_id == 0 ) {
          unsigned int ix = g_ipt[shifted_source_coords[0]][shifted_source_coords[1]][shifted_source_coords[2]][shifted_source_coords[3]];
          unsigned int ixeosub = g_lexic2eosub[ix];
          int ieo = g_iseven[ix] ? 0 : 1;
          if ( mu == 4 ) {
            for ( int k = 0; k < 4; k++ ) {
              cvc_loop_eo_wi[0][k][0] = cvc_loop_eo[ieo][k][2*ixeosub  ];
              cvc_loop_eo_wi[0][k][1] = cvc_loop_eo[ieo][k][2*ixeosub+1];
            }
          } else {
            cvc_loop_eo_wi[1][mu][0] = cvc_loop_eo[ieo][mu][2*ixeosub  ];
            cvc_loop_eo_wi[1][mu][1] = cvc_loop_eo[ieo][mu][2*ixeosub+1];
          }
        }
#ifdef HAVE_MPI
      } else {
        if ( g_cart_id == source_proc_id ) {
          unsigned int ix = g_ipt[shifted_source_coords[0]][shifted_source_coords[1]][shifted_source_coords[2]][shifted_source_coords[3]];
          unsigned int ixeosub = g_lexic2eosub[ix];
          int ieo = g_iseven[ix] ? 0 : 1;
          if ( mu == 4 ) {
            double buffer[8] = {
              cvc_loop_eo[ieo][0][2*ixeosub], cvc_loop_eo[ieo][0][2*ixeosub+1],
              cvc_loop_eo[ieo][1][2*ixeosub], cvc_loop_eo[ieo][1][2*ixeosub+1],
              cvc_loop_eo[ieo][2][2*ixeosub], cvc_loop_eo[ieo][2][2*ixeosub+1],
              cvc_loop_eo[ieo][3][2*ixeosub], cvc_loop_eo[ieo][3][2*ixeosub+1] };
            exitstatus = MPI_Send( buffer, 8, MPI_DOUBLE, 0, 101, g_cart_grid );
            if( exitstatus != MPI_SUCCESS) {
              fprintf(stderr, "[loops_caa_lma] Error from MPI_Send, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(24);
            }
          } else {
            exitstatus = MPI_Send( cvc_loop_eo[ieo][mu]+2*ixeosub, 2, MPI_DOUBLE, 0, 101, g_cart_grid );
            if( exitstatus != MPI_SUCCESS) {
              fprintf(stderr, "[loops_caa_lma] Error from MPI_Send, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(24);
            }
          }
        } else if ( g_cart_id == 0 ) {
          if ( mu == 4 ) {
            exitstatus = MPI_Recv( cvc_loop_eo_wi[0][0], 8, MPI_DOUBLE, shifted_source_proc_id, 101, g_cart_grid, &mstatus );
            if( exitstatus != MPI_SUCCESS) {
              fprintf(stderr, "[loops_caa_lma] Error from MPI_Recv, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(24);
            }
          } else {
            exitstatus = MPI_Recv( cvc_loop_eo_wi[1][mu], 2, MPI_DOUBLE, shifted_source_proc_id, 101, g_cart_grid, &mstatus );
            if( exitstatus != MPI_SUCCESS) {
              fprintf(stderr, "[loops_caa_lma] Error from MPI_Recv, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(24);
            }
          }
        }
#endif
      }


      fini_3level_buffer( &cvc_loop_eo );

    }  /* end of loop on shift direction mu */

    /***************************************************************************/
    /***************************************************************************/


    if ( g_cart_id == 0 ) {
      fprintf(stdout, "# [loops_caa_lma] WI x %3d %3d %3d %3d\n", gsx[0], gsx[1], gsx[2], gsx[3] );
      for ( int k = 0; k < 4; k++ ) {
        fprintf(stdout, "%3d %25.16e%25.16e %25.16e%25.16e\n", k, cvc_loop_eo_wi[0][k][0], cvc_loop_eo_wi[0][k][1], cvc_loop_eo_wi[1][k][0], cvc_loop_eo_wi[1][k][1] );
      }
      double normr = 0., normi = 0.;
      normr = cvc_loop_eo_wi[0][0][0] - cvc_loop_eo_wi[1][0][0]
            + cvc_loop_eo_wi[0][1][0] - cvc_loop_eo_wi[1][1][0]
            + cvc_loop_eo_wi[0][2][0] - cvc_loop_eo_wi[1][2][0]
            + cvc_loop_eo_wi[0][3][0] - cvc_loop_eo_wi[1][3][0];

      normi = cvc_loop_eo_wi[0][0][1] - cvc_loop_eo_wi[1][0][1]
            + cvc_loop_eo_wi[0][1][1] - cvc_loop_eo_wi[1][1][1]
            + cvc_loop_eo_wi[0][2][1] - cvc_loop_eo_wi[1][2][1]
            + cvc_loop_eo_wi[0][3][1] - cvc_loop_eo_wi[1][3][1];
      fprintf(stdout, "# [loops_caa_lma] WI %25.16e %25.16e\n", normr, normi);

      fini_3level_buffer( &cvc_loop_eo_wi );
    }


    /***************************************************************************/
    /***************************************************************************/


#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[loops_caa_lma] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */


  }  /* end of loop on source locations */
#endif  /* of if 0 */

#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    sprintf(filename, "%s.%.4d.aff", "loops_lma", Nconf );
    fprintf(stdout, "# [loops_caa_lma] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[loops_caa_lma] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }  /* end of if io_proc == 2 */
#endif

  double ***cvc_loop_lma = NULL, **cvc_wi = NULL;
  exitstatus = init_3level_buffer ( &cvc_loop_lma, 2, 4, 2*Vhalf );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[loops_caa_lma] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  /***********************************************/
  /***********************************************/

  for ( int nev = nev_step_size; nev <= evecs_num; nev += nev_step_size ) {

    double **eo_evecs_ptr = &(eo_evecs_field[nev-nev_step_size]);
    double *evecs_norm_ptr = &(evecs_4kappasqr_lambdainv[nev-nev_step_size]);

    contract_cvc_loop_eo_lma ( cvc_loop_lma, eo_evecs_ptr, evecs_norm_ptr, nev_step_size, gauge_field_with_phase, mzz, mzzinv );

    if ( check_position_space_WI ) {
      exitstatus = cvc_loop_eo_check_wi_position_space_lma ( &cvc_wi, cvc_loop_lma, eo_evecs_ptr, evecs_norm_ptr, nev_step_size, gauge_field_with_phase, mzz, mzzinv  );
      if(exitstatus != 0 ) {
        fprintf(stderr, "[loops_caa_lma] Error from cvc_loop_eo_check_wi_position_space_lma, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }
    }

    double ***cvc_loop_tp = NULL;

    exitstatus = cvc_loop_eo_momentum_projection ( &cvc_loop_tp, cvc_loop_lma, g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0 ) {
      fprintf(stderr, "[loops_caa_lma] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    sprintf(aff_tag, "/cvc-loop/nev%.4d", nev);
    exitstatus = cvc_loop_tp_write_to_aff_file ( cvc_loop_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[loops_caa_lma] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }
 
    if ( check_position_space_WI ) {
      exitstatus = cvc_loop_eo_check_wi_momentum_space_lma ( cvc_wi, cvc_loop_tp, g_sink_momentum_list, g_sink_momentum_number );
      if(exitstatus != 0 ) {
        fprintf(stderr, "[loops_caa_lma] Error from cvc_loop_eo_check_wi_momentum_space_lma, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }
    }


    fini_3level_buffer ( &cvc_loop_tp );
#if 0
#endif
  }  /* end of loop on nev */ 

  fini_3level_buffer ( &cvc_loop_lma );
  if ( check_position_space_WI ) {
    fini_2level_buffer ( &cvc_wi );
  }

#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[loops_caa_lma] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  /***********************************************/
  /***********************************************/

  /***********************************************
   * stochastic part
   ***********************************************/

#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    sprintf(filename, "%s.%.4d.b%d.aff", "loops_stoch", Nconf, sample_step_size );
    fprintf(stdout, "# [loops_caa_lma] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[loops_caa_lma] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }  /* end of if io_proc == 2 */
#endif

  /***********************************************
   * initialize random number generator
   ***********************************************/
  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_mixed] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }
 
  double ***cvc_loop_stoch = NULL, ***cvc_loop_tp = NULL, *sample_norm_ptr = NULL;
  size_t bytes = g_sink_momentum_number * 4 * 2*T * sizeof(double);

  exitstatus = init_3level_buffer ( &cvc_loop_stoch, 2, 4, 2*Vhalf );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[loops_caa_lma] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  exitstatus = init_3level_buffer( &cvc_loop_tp, g_sink_momentum_number, 4, 2*T);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[loops_caa_lma] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  exitstatus = init_2level_buffer ( &eo_sample_field, sample_step_size, _GSI( Vhalf ) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[loops_caa_lma] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }
  eo_sample_block = eo_sample_field[0];

  if ( (sample_norm_ptr = (double*)malloc ( sample_step_size * sizeof(double) ) ) == NULL ) {
    fprintf(stderr, "[loops_caa_lma] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(1);
  }
  for ( int i = 0; i < sample_step_size; i++ ) { sample_norm_ptr[i] = 1.; }

  /***********************************************
   * loop on stochastic samples
   ***********************************************/
  for( int isample = sample_step_size; isample <= g_nsample; isample += sample_step_size ) {

    /***********************************************
     * make a source
     ***********************************************/
    exitstatus = prepare_volume_source ( eo_sample_field[0], sample_step_size*Vhalf);
    if(exitstatus != 0) {
      fprintf(stderr, "[loops_caa_lma] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(33);
    }

    for ( int i = 0; i < sample_step_size; i++ ) {
      /* work1 <- C sample[i] */
      C_clover_oo ( eo_spinor_work[1], eo_sample_field[i], gauge_field_with_phase, eo_spinor_work[2], g_mzz_up[1], g_mzzinv_up[0] );

      /* weighted parallel projection */
      /* work1 <- V  (2 kappa)^2/Lambda V^+ work1 */
      exitstatus = project_propagator_field_weighted ( eo_spinor_work[1], eo_spinor_work[1], 1, eo_evecs_block, evecs_4kappasqr_lambdainv, 1, evecs_num, Vhalf);
      if (exitstatus != 0) {
        fprintf(stderr, "[loops_caa_lma] Error from project_propagator_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(35);
      }

      /* work0 <- Cbar work1 */
      C_clover_oo ( eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_phase, eo_spinor_work[2], g_mzz_dn[1], g_mzzinv_dn[0] );

      /* work1 <- sample - work0 */
      spinor_field_eq_spinor_field_mi_spinor_field( eo_spinor_work[1], eo_sample_field[i], eo_spinor_work[0], Vhalf);

      /* invert */
      memset(eo_spinor_work[0], 0, sizeof_eo_spinor_field);
      exitstatus = tmLQCD_invert_eo(eo_spinor_work[0], eo_spinor_work[1], _OP_ID_DN);
      if(exitstatus != 0) {
        fprintf(stderr, "[loops_caa_lma] Error from tmLQCD_invert_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(19);
      }
      memcpy( eo_sample_field[i], eo_spinor_work[0], sizeof_eo_spinor_field );

      if( check_propagator_residual ) {
        exitstatus = check_oo_propagator_clover_eo( &(eo_sample_field[i]), &(eo_spinor_work[1]), &(eo_spinor_work[2]), gauge_field_with_phase, g_mzz_dn, g_mzzinv_dn, 1 );
        if(exitstatus != 0) {
          fprintf(stderr, "[loops_caa_lma] Error from check_oo_propagator_clover_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(19);
        }
      }
    }  /* end of loop on samples within block */

    /* contract */
    contract_cvc_loop_eo_lma ( cvc_loop_stoch, eo_sample_field, sample_norm_ptr, sample_step_size, gauge_field_with_phase, mzz, mzzinv );

    /* momentum projection */
    memset ( cvc_loop_tp[0][0], 0, bytes);
    exitstatus = cvc_loop_eo_momentum_projection ( &cvc_loop_tp, cvc_loop_stoch, g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0 ) {
      fprintf(stderr, "[loops_caa_lma] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    /* output */
    sprintf(aff_tag, "/cvc-loop/block%.4d", isample);
    exitstatus = cvc_loop_tp_write_to_aff_file ( cvc_loop_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[loops_caa_lma] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

  }  /* end of loop on stochastic samples */

  fini_3level_buffer ( &cvc_loop_stoch );
  fini_3level_buffer ( &cvc_loop_tp );
  fini_2level_buffer ( &eo_sample_field ); eo_sample_block = NULL;
  free ( sample_norm_ptr );


#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[loops_caa_lma] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */


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
  free ( evecs_4kappasqr_lambdainv );

  /* free clover matrix terms */
  fini_clover ();

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
    fprintf(stdout, "# [loops_caa_lma] %s# [loops_caa_lma] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [loops_caa_lma] %s# [loops_caa_lma] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
