/****************************************************
 * p2gg.c
 *
 * Tue Mar 14 10:22:55 CET 2017
 *
 * - originally copied from p2gg_xspace.c
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
#include "project.h"
#include "matrix_init.h"
#include "dummy_solver.h"

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

  const char outfile_prefix[] = "p2gg";

  int c, i, mu;
  int iflavor;
  int filename_set = 0;
  int isource_location;
  unsigned int ix;
  int gsx[4], sx[4];
  int check_position_space_WI=0;
  int exitstatus;
  int source_proc_id = 0;
  int iseq_source_momentum;
  int isequential_source_gamma_id, isequential_source_timeslice;
  int no_eo_fields = 0;
  int io_proc = -1;
  int evecs_num = 0;
  int check_propagator_residual = 0;
  unsigned int Vhalf;
  size_t sizeof_eo_spinor_field;
  double **eo_spinor_field=NULL, **eo_spinor_work=NULL, *eo_evecs_block=NULL, *eo_sample_block=NULL,   **eo_sample_field=NULL;
  double **eo_evecs_field=NULL;
  double *cvc_tensor_eo[2] = {NULL, NULL}, contact_term[2][8], *cvc_tensor_lexic=NULL;
  double ***cvc_tp = NULL;
  double *evecs_eval = NULL, *evecs_lambdainv=NULL;
  double *uprop_list_e[60], *uprop_list_o[60], *tprop_list_e[60], *tprop_list_o[60], *dprop_list_e[60], *dprop_list_o[60];
  char filename[100];
  double ratime, retime;
  double plaq;
  double **mzz[2], **mzzinv[2];
  double *gauge_field_with_phase = NULL;



#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  /* struct AffNode_s *affn = NULL, *affdir=NULL; */
  char * aff_status_str;
  /* double _Complex *aff_buffer = NULL; */
  /* char aff_buffer_path[200]; */
  char aff_tag[400];
  /*  uint32_t aff_buffer_size; */
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

  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [p2gg] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg] calling tmLQCD wrapper init functions\n");

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
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [p2gg] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(6);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg] Error, &g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[p2gg] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg] Error from tmLQCD_get_deflator_params, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(9);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [p2gg] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [p2gg] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [p2gg] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [p2gg] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block == NULL) {
    fprintf(stderr, "[p2gg] Error, eo_evecs_block is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(10);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[p2gg] Error, dimension of eigenspace is zero %s %d\n", __FILE__, __LINE__);
    EXIT(11);
  }

  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[p2gg] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  evecs_eval = (double*)malloc(evecs_num*sizeof(double));
  evecs_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if(evecs_eval == NULL || evecs_lambdainv == NULL ) {
    fprintf(stderr, "[p2gg] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for(i=0; i<evecs_num; i++) {
    evecs_eval[i] = ((double*)(g_tmLQCD_defl.evals))[2*i];
    evecs_lambdainv[i] = 2.* g_kappa / evecs_eval[i];
    if( g_cart_id == 0 ) fprintf(stdout, "# [p2gg] eval %4d %16.7e\n", i, evecs_eval[i] );
  }

#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */



  /*************************************************
   * allocate memory for the eigenvector fields
   *************************************************/
  eo_evecs_field = (double**)calloc(evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block;
  for(i=1; i<evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + _GSI(Vhalf);

  /*************************************************
   * allocate memory for samples
   *************************************************/
  eo_sample_block = (double*)malloc(2*g_nsample * sizeof_eo_spinor_field);
  if(eo_sample_block == NULL) {
    fprintf(stderr, "[p2gg] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }
  eo_sample_field    = (double**)calloc(2*g_nsample, sizeof(double*));
  eo_sample_field[0] = eo_sample_block;
  for(i=1; i<2*g_nsample; i++) eo_sample_field[i] = eo_sample_field[i-1] + _GSI(Vhalf);

  /*************************************************
   * allocate memory for eo spinor fields 
   * WITHOUT HALO
   *************************************************/
  no_eo_fields = 360;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));

  eo_spinor_field[0] = (double*)malloc( no_eo_fields * sizeof_eo_spinor_field);
  if(eo_spinor_field[0] == NULL) {
    fprintf(stderr, "[p2gg] Error from calloc %s %d\n", __FILE__, __LINE__);
    EXIT(13);
  }
  for(i=1; i<no_eo_fields; i++) eo_spinor_field[i] = eo_spinor_field[i-1] + _GSI(Vhalf);

  /*************************************************
   * allocate memory for eo spinor fields 
   * WITH HALO
   *************************************************/
  no_eo_fields = 6;
  eo_spinor_work = (double**)malloc( no_eo_fields * sizeof(double*) );
  eo_spinor_work[0] = (double*)malloc( no_eo_fields * _GSI( ((VOLUME+RAND)/2) ) * sizeof(double) );
  if(eo_spinor_field[0] == NULL) {
    fprintf(stderr, "[p2gg] Error from calloc %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  for(i=1; i<no_eo_fields; i++) { eo_spinor_work[i] = eo_spinor_work[i-1] + _GSI( ((VOLUME+RAND)/2) ); }


  /*************************************************
   *************************************************
   **
   ** set the up-, dn- and seq-propagator list
   **
   *************************************************
   *************************************************/
  
  for(i=0; i<60; i++) {
    /* up + even */
    uprop_list_e[i] = eo_spinor_field[      i];
    /* up + odd */
    uprop_list_o[i] = eo_spinor_field[ 60 + i];

    /* up + even */
    dprop_list_e[i] = eo_spinor_field[120 + i];
    /* up + odd */
    dprop_list_o[i] = eo_spinor_field[180 + i];

    /* up + even */
    tprop_list_e[i] = eo_spinor_field[240 + i];
    /* up + odd */
    tprop_list_o[i] = eo_spinor_field[300 + i];
  }
  
  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  clover_term_init(&g_clover, 6);
  clover_term_init(&g_mzz_up, 6);
  clover_term_init(&g_mzz_dn, 6);


  ratime = _GET_TIME;
  clover_term_eo (g_clover, gauge_field_with_phase);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_term_eo = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (g_mzz_up, g_clover, g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (g_mzz_dn, g_clover, -g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  clover_term_fini( &g_clover );
  clover_term_init(&g_mzzinv_up, 8);
  clover_term_init(&g_mzzinv_dn, 8);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (g_mzzinv_up, g_mzz_up);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (g_mzzinv_dn, g_mzz_dn);
  retime = _GET_TIME;
  if ( g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  mzz[0] = g_mzz_up;
  mzz[1] = g_mzz_dn;
  mzzinv[0] = g_mzzinv_up;
  mzzinv[1] = g_mzzinv_dn;

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [p2gg] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [p2gg] proc%.4d is send process\n", g_cart_id);
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
      fprintf(stderr, "[p2gg] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************
   ***********************************************
   **
   ** stochastic propagators
   **
   ***********************************************
   ***********************************************/

  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /* make a source */
  exitstatus = prepare_volume_source(eo_sample_block, g_nsample*Vhalf);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(33);
  }

  /* orthogonal projection */
  exitstatus = project_propagator_field ( eo_sample_block, eo_sample_block, 0, eo_evecs_block, g_nsample, evecs_num, Vhalf);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg] Error from project_propagator_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(35);
  }

  for(i = 0; i < g_nsample; i++) {
    /* invert */
    memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
    memcpy(eo_spinor_work[0], eo_sample_field[i], sizeof_eo_spinor_field);
    exitstatus = _TMLQCD_INVERT_EO(eo_spinor_work[1], eo_spinor_work[0], _OP_ID_UP);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg] Error from _TMLQCD_INVERT_EO, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }
    memcpy(eo_sample_field[g_nsample+i], eo_spinor_work[1], sizeof_eo_spinor_field);
  }  /* end of loop on samples */

  if( check_propagator_residual ) {
    exitstatus = check_oo_propagator_clover_eo( &(eo_sample_field[g_nsample]), eo_sample_field, eo_spinor_work, gauge_field_with_phase, g_mzz_up, g_mzzinv_up, g_nsample );
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg] Error from check_oo_propagator_clover_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }
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
      fprintf(stderr, "[p2gg] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    init_contract_cvc_tensor_usource(g_gauge_field, gsx, co_phase_up);

    
    /**********************************************************
     **********************************************************
     **
     ** propagators with source at gsx and gsx + mu
     **
     **********************************************************
     **********************************************************/

    for(mu=0; mu<5; mu++)
    {  /* loop on shifts in direction mu */
      /**********************************************************
       * shifted source coords and source proc coords
       **********************************************************/
      int g_shifted_source_coords[4], shifted_source_coords[4], shifted_source_proc_id=0;
      memcpy( g_shifted_source_coords, gsx, 4*sizeof(int));
      switch(mu) {
        case 0: g_shifted_source_coords[0] = ( g_shifted_source_coords[0] + 1 ) % T_global; break;;
        case 1: g_shifted_source_coords[1] = ( g_shifted_source_coords[1] + 1 ) % LX_global; break;;
        case 2: g_shifted_source_coords[2] = ( g_shifted_source_coords[2] + 1 ) % LY_global; break;;
        case 3: g_shifted_source_coords[3] = ( g_shifted_source_coords[3] + 1 ) % LZ_global; break;;
      }

      /* source info for shifted source location */
      if( (exitstatus = get_point_source_info (g_shifted_source_coords, shifted_source_coords, &shifted_source_proc_id) ) != 0 ) {
        fprintf(stderr, "[p2gg] Error from get_point_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(15);
      }

      /**********************************************************
       * up-type propagators
       **********************************************************/
      for(i=0; i<12; i++) 
      {

        int eo_spinor_field_id_e =      mu * 12 + i;
        int eo_spinor_field_id_o = 60 + mu * 12 + i;

        /* A^-1 g5 source */
        exitstatus = init_clover_eo_spincolor_pointsource_propagator (eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
            g_shifted_source_coords, i, gauge_field_with_phase, g_mzzinv_up[0], (int)(shifted_source_proc_id == g_cart_id), eo_spinor_work[0]);
        if(exitstatus != 0 ) {
          fprintf(stderr, "[p2gg] Error from init_eo_spincolor_pointsource_propagator; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(18);
        }

        /* C^-1 */
        if(g_cart_id == 0) fprintf(stdout, "# [p2gg] calling _TMLQCD_INVERT_EO\n");
        memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
        memcpy(eo_spinor_work[0], eo_spinor_field[eo_spinor_field_id_o], sizeof_eo_spinor_field);
        exitstatus = _TMLQCD_INVERT_EO(eo_spinor_work[1], eo_spinor_work[0], _OP_ID_UP);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg] Error from _TMLQCD_INVERT_EO, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(19);
        }
        memcpy(eo_spinor_field[eo_spinor_field_id_o], eo_spinor_work[1], sizeof_eo_spinor_field);

        /* B^-1 excl. C^-1 */
        exitstatus = fini_clover_eo_propagator (eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o], eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
            gauge_field_with_phase, g_mzzinv_up[0], eo_spinor_work[0]);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg] Error from fini_eo_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(20);
        }

      }  /* end of loop on spin-color */
      if( check_propagator_residual ) {
        check_point_source_propagator_clover_eo( &(eo_spinor_field[12*mu]), &(eo_spinor_field[60+12*mu]), eo_spinor_work, gauge_field_with_phase, g_mzz_up, g_mzzinv_up, g_shifted_source_coords, 12 );
        if(exitstatus != 0 ) {
          fprintf(stderr, "[p2gg] Error from check_point_source_propagator_clover_eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(21);
        }
      }



      /**********************************************************
       * dn-type propagators
       **********************************************************/
      for(i=0; i<12; i++) {
        /* A^-1 g5 source */
        int eo_spinor_field_id_e = 120 + mu * 12 + i;
        int eo_spinor_field_id_o = 180 + mu * 12 + i;

        exitstatus = init_clover_eo_spincolor_pointsource_propagator (eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
            g_shifted_source_coords, i, gauge_field_with_phase, g_mzzinv_dn[0], shifted_source_proc_id==g_cart_id, eo_spinor_work[0]);
        if(exitstatus != 0 ) {
          fprintf(stderr, "[p2gg] Error from init_eo_spincolor_pointsource_propagator; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(21);
        }

        /* C^-1 */
        if(g_cart_id == 0) fprintf(stdout, "# [p2gg] calling _TMLQCD_INVERT_EO\n");
        memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
        memcpy(eo_spinor_work[0], eo_spinor_field[eo_spinor_field_id_o], sizeof_eo_spinor_field);
        exitstatus = _TMLQCD_INVERT_EO(eo_spinor_work[1], eo_spinor_work[0], _OP_ID_DN);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg] Error from _TMLQCD_INVERT_EO, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(22);
        }
        memcpy(eo_spinor_field[eo_spinor_field_id_o], eo_spinor_work[1], sizeof_eo_spinor_field);

        /* B^-1 excl. C^-1 */
        exitstatus = fini_clover_eo_propagator (eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o], eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
            gauge_field_with_phase, g_mzzinv_dn[0], eo_spinor_work[0]);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg] Error from fini_eo_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(23);
        }
      }  /* end of loop on spin-color */
      if ( check_propagator_residual ) {
        check_point_source_propagator_clover_eo( &(eo_spinor_field[120+12*mu]), &(eo_spinor_field[180+12*mu]), eo_spinor_work, gauge_field_with_phase, g_mzz_dn, g_mzzinv_dn, g_shifted_source_coords, 12 );
        if(exitstatus != 0 ) {
          fprintf(stderr, "[p2gg] Error from check_point_source_propagator_clover_eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(21);
        }
      }

      /**********************************************************
       * A^-1 g5 for up-type propagators from sub-space
       * inversion
       **********************************************************/
      for(i=0; i<12; i++)
      {
        int eo_spinor_field_id_e = 240 + mu * 12 + i;
        int eo_spinor_field_id_o = eo_spinor_field_id_e + 60;

        /* A^-1 g5 source */
        exitstatus = init_clover_eo_spincolor_pointsource_propagator (eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
          g_shifted_source_coords, i, gauge_field_with_phase, g_mzzinv_up[0], (int)(shifted_source_proc_id == g_cart_id), eo_spinor_work[0]);
        if(exitstatus != 0 ) {
          fprintf(stderr, "[p2gg] Error from init_eo_spincolor_pointsource_propagator; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(18);
        }
      }  /* end of loop on spin-color */

    }    /* end of loop on shift direction mu */


#ifdef HAVE_LHPC_AFF
    /***********************************************
     ***********************************************
     **
     ** writer for aff output file
     **
     ***********************************************
     ***********************************************/
    if(io_proc == 2) {
      aff_status_str = (char*)aff_version();
      fprintf(stdout, "# [p2gg] using aff version %s\n", aff_status_str);

      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [p2gg] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#endif

    /***************************************************************************
     ***************************************************************************
     **
     ** contractions for the photon hvp tensor
     **
     ***************************************************************************
     ***************************************************************************/

    exitstatus = Q_clover_eo_invert_subspace ( &(eo_spinor_field[240]), &(eo_spinor_field[300]),
                                               &(eo_spinor_field[240]), &(eo_spinor_field[300]), 60, eo_evecs_block, evecs_lambdainv, evecs_num,
                                               gauge_field_with_phase, mzz, mzzinv, 0, eo_spinor_work );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg] Error from Q_clover_eo_invert_subspace, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(28);
    }

    /* allocate memory for contractions, initialize */
    cvc_tensor_eo[0] = (double*)malloc( 32*VOLUME *sizeof(double) );
    if( cvc_tensor_eo[0] == NULL ) {
      fprintf(stderr, "[p2gg] could not allocate memory for eo contraction fields %s %d\n", __FILE__, __LINE__);
      EXIT(24);
    }
    cvc_tensor_eo[1] = cvc_tensor_eo[0] + 32 * Vhalf;

    /***************************************************************************
     * full tensor
     ***************************************************************************/
    memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
    memset(contact_term[0], 0, 8*sizeof(double));
    /* contraction */
    contract_cvc_tensor_eo ( cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[0], dprop_list_e, dprop_list_o, uprop_list_e, uprop_list_o , gauge_field_with_phase );

    /* subtract contact term */
    cvc_tensor_eo_subtract_contact_term (cvc_tensor_eo, contact_term[0], gsx, (int)( source_proc_id == g_cart_id ) );

    /* momentum projections */
    exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cvc_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(26);
    }
    /* write results to file */
    sprintf(aff_tag, "/full/hvp/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = cvc_tensor_tp_write_to_aff_file ( cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }
    fini_3level_buffer(&cvc_tp);

    /* check position space WI */
    if(check_position_space_WI) {
      exitstatus = cvc_tensor_eo_check_wi_position_space ( cvc_tensor_eo );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg] Error from cvc_tensor_eo_check_wi_position_space for full, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }
    }

    /***************************************************************************
     * low-mode tensor
     ***************************************************************************/
    memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
    memset(contact_term[0], 0, 8*sizeof(double));
    /* contraction */
    contract_cvc_tensor_eo ( cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[0], dprop_list_e, dprop_list_o, tprop_list_e, tprop_list_o , gauge_field_with_phase );

    /* subtract contact term */
    cvc_tensor_eo_subtract_contact_term (cvc_tensor_eo, contact_term[0], gsx, (int)( source_proc_id == g_cart_id ) );

    /* momentum projections */
    exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cvc_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(26);
    }
    /* write results to file */
    sprintf(aff_tag, "/lm/hvp/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = cvc_tensor_tp_write_to_aff_file ( cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }
    fini_3level_buffer(&cvc_tp);

    /* check position space WI */
    if(check_position_space_WI) {
      exitstatus = cvc_tensor_eo_check_wi_position_space ( cvc_tensor_eo );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg] Error from cvc_tensor_eo_check_wi_position_space for lm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }
    }

    /***************************************************************************
     * high-mode tensor
     ***************************************************************************/

    for(mu = 0; mu< 5; mu++ ) {

      int g_shifted_source_coords[4], shifted_source_coords[4], shifted_source_proc_id=0;
      memcpy( g_shifted_source_coords, gsx, 4*sizeof(int));
      switch(mu) {
        case 0: g_shifted_source_coords[0] = ( g_shifted_source_coords[0] + 1 ) % T_global; break;;
        case 1: g_shifted_source_coords[1] = ( g_shifted_source_coords[1] + 1 ) % LX_global; break;;
        case 2: g_shifted_source_coords[2] = ( g_shifted_source_coords[2] + 1 ) % LY_global; break;;
        case 3: g_shifted_source_coords[3] = ( g_shifted_source_coords[3] + 1 ) % LZ_global; break;;
      }

      /**********************************************************
       * A^-1 g5 for up-type propagators from sub-space
       * inversion
       **********************************************************/
      for(i=0; i<12; i++)
      {
        /* eo spinor field indices */
        int eo_spinor_field_id_e = 240 + mu * 12 + i;
        int eo_spinor_field_id_o = eo_spinor_field_id_e + 60;
        /* A^-1 g5 source */
        exitstatus = init_clover_eo_spincolor_pointsource_propagator (eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
          g_shifted_source_coords, i, gauge_field_with_phase, g_mzzinv_up[0], (int)(shifted_source_proc_id == g_cart_id), eo_spinor_work[0]);
        if(exitstatus != 0 ) {
          fprintf(stderr, "[p2gg] Error from init_eo_spincolor_pointsource_propagator; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(18);
        }
        /* set even part to zero */
        memset( eo_spinor_field[eo_spinor_field_id_e], 0, sizeof_eo_spinor_field );
      }  /* end of loop on spin-color */
    }    /* end of loop on shift direction mu */

    exitstatus = Q_clover_eo_invert_subspace_stochastic ( &(eo_spinor_field[240]), &(eo_spinor_field[300]),
                                                          &(eo_spinor_field[240]), &(eo_spinor_field[300]), 60,
                                                          eo_sample_field[g_nsample], eo_sample_field[0], g_nsample,
                                                          gauge_field_with_phase, mzz, mzzinv, 0, eo_spinor_work );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg] Error from Q_clover_eo_invert_subspace_stochastic, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(28);
    }

    memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
    memset(contact_term[0], 0, 8*sizeof(double));
    /* contraction */
    contract_cvc_tensor_eo ( cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[0], dprop_list_e, dprop_list_o, tprop_list_e, tprop_list_o , gauge_field_with_phase );

    /* subtract contact term */
    cvc_tensor_eo_subtract_contact_term (cvc_tensor_eo, contact_term[0], gsx, (int)( source_proc_id == g_cart_id ) );

    /* momentum projections */
    exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cvc_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(26);
    }
    /* write results to file */
    sprintf(aff_tag, "/hm/hvp/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = cvc_tensor_tp_write_to_aff_file ( cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }
    fini_3level_buffer(&cvc_tp);

    /* check position space WI */
    if(check_position_space_WI) {
      exitstatus = cvc_tensor_eo_check_wi_position_space ( cvc_tensor_eo );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg] Error from cvc_tensor_eo_check_wi_position_space for lm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }
    }


    /* free tensor array */
    free( cvc_tensor_eo[0] ); cvc_tensor_eo[0] = NULL; cvc_tensor_eo[1] = NULL;

#if 0
#endif  /* of if 0 */

    /***************************************************************************
     ***************************************************************************
     **
     ** contractions for the P -> gg tensor
     **
     ***************************************************************************
     ***************************************************************************/

    /***************************************************************************
     * loop on sequential source gamma matrices
     ***************************************************************************/
    for(iseq_source_momentum=0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) {

      g_seq_source_momentum[0] = g_seq_source_momentum_list[iseq_source_momentum][0];
      g_seq_source_momentum[1] = g_seq_source_momentum_list[iseq_source_momentum][1];
      g_seq_source_momentum[2] = g_seq_source_momentum_list[iseq_source_momentum][2];

      if(g_cart_id == 0) fprintf(stdout, "# [p2gg] using sequential source momentum no. %2d = (%d, %d, %d)\n", iseq_source_momentum,
        g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);

      /***************************************************************************
       * loop on sequential source gamma matrices
       ***************************************************************************/
      for(isequential_source_gamma_id=0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++) {
  
        int sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];
        if(g_cart_id == 0) fprintf(stdout, "# [p2gg] using sequential source gamma id no. %2d = %d\n", isequential_source_gamma_id, sequential_source_gamma_id);

        /***************************************************************************
         * loop on sequential source time slices
         ***************************************************************************/
        for(isequential_source_timeslice=0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++) {

          g_sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];
          /* shift sequential source timeslice by source timeslice gsx[0] */
          int g_shifted_sequential_source_timeslice = ( gsx[0] + g_sequential_source_timeslice + T_global ) % T_global;

          if(g_cart_id == 0) fprintf(stdout, "# [p2gg] using sequential source timeslice %d / %d\n", g_sequential_source_timeslice, g_shifted_sequential_source_timeslice);


          /***************************************************************************
           ***************************************************************************
           **
           **  contractions for P - cvc - cvc 3pt with low-mode sequential propagator 
           **
           ***************************************************************************
           ***************************************************************************/


          /*************************************************
           * allocate memory for the contractions
           *************************************************/
          cvc_tensor_eo[0] = (double*)malloc( 32*VOLUME *sizeof(double) );
          if( cvc_tensor_eo[0] == NULL ) {
            fprintf(stderr, "[p2gg] could not allocate memory for eo contraction fields %s %d\n", __FILE__, __LINE__);
            EXIT(24);
          }
          cvc_tensor_eo[1] = cvc_tensor_eo[0] + 32 * Vhalf;
          memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
          memset(contact_term[0], 0, 8*sizeof(double));
          memset(contact_term[1], 0, 8*sizeof(double));

          /* loop on dn-type and up-type flavor */
          // for(iflavor=1; iflavor>=0; iflavor--) 
          for(iflavor=0; iflavor<1; iflavor++ ) 
          {

            if ( iflavor == 1 ) {
              memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
              memset(contact_term[1], 0, 8*sizeof(double));
            } else if ( iflavor == 0 ) {
              complex_field_eq_complex_field_conj_ti_re (cvc_tensor_eo[0], (double)sequential_source_gamma_id_sign[ sequential_source_gamma_id ], 16*VOLUME );
              memset( contact_term[0], 0, 8*sizeof(double) );
            }

            ratime = _GET_TIME;

             /* flavor-dependent sequential source momentum */
            int seq_source_momentum[3] = { (1 - 2*iflavor) * g_seq_source_momentum[0],
                                           (1 - 2*iflavor) * g_seq_source_momentum[1],
                                           (1 - 2*iflavor) * g_seq_source_momentum[2] };

            if(g_cart_id == 0) fprintf(stdout, "# [p2gg] using flavor-dependent sequential source momentum (%d, %d, %d)\n", 
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2]);

            for(i=0; i<60; i++) {
                int eo_spinor_field_id_e     = iflavor * 120 + i;
                int eo_spinor_field_id_o     = eo_spinor_field_id_e + 60;
                int eo_seq_spinor_field_id_e = 240 + i;
                int eo_seq_spinor_field_id_o = eo_seq_spinor_field_id_e + 60;

                exitstatus = init_clover_eo_sequential_source(
                    eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_field[eo_seq_spinor_field_id_o],
                    eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
                    g_shifted_sequential_source_timeslice, gauge_field_with_phase, mzzinv[iflavor][0], seq_source_momentum, sequential_source_gamma_id, eo_spinor_work[0]);
                if(exitstatus != 0) {
                  fprintf(stderr, "[p2gg] Error from init_clover_eo_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(25);
                }
            }  /* end of loop on spin-color */

            /* invert sequential source on eigenvector subspace */
            exitstatus = Q_clover_eo_invert_subspace ( &(eo_spinor_field[240]), &(eo_spinor_field[300]),
                                                       &(eo_spinor_field[240]), &(eo_spinor_field[300]),
                                                       60, eo_evecs_block, evecs_lambdainv, evecs_num, gauge_field_with_phase, mzz, mzzinv, iflavor, eo_spinor_work);
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg] Error from Q_clover_eo_invert_subspace, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(27);
            }

            /* contraction function */
            if ( iflavor == 1 ) {
              contract_cvc_tensor_eo ( cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[iflavor], uprop_list_e, uprop_list_o, tprop_list_e, tprop_list_o , gauge_field_with_phase );
            } else {
              contract_cvc_tensor_eo ( cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[iflavor], dprop_list_e, dprop_list_o, tprop_list_e, tprop_list_o , gauge_field_with_phase );
            }
          }  /* end of loop on iflavor */

          /* subtract contact term */
          cvc_tensor_eo_subtract_contact_term ( cvc_tensor_eo, contact_term[0], gsx, (int)( source_proc_id == g_cart_id ) );

          /* momentum projections */
          exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cvc_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(28);
          }

          /***************************************************************************
           * write results to file
           ***************************************************************************/
          sprintf(aff_tag, "/lm/PJJ/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d",
                  gsx[0], gsx[1], gsx[2], gsx[3], g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2], sequential_source_gamma_id, g_sequential_source_timeslice);

          exitstatus = cvc_tensor_tp_write_to_aff_file (cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(29);
          }
          fini_3level_buffer(&cvc_tp);
          free( cvc_tensor_lexic ); cvc_tensor_lexic = NULL;
          free( cvc_tensor_eo[0] ); cvc_tensor_eo[0] = NULL; cvc_tensor_eo[1] = NULL;

          /***************************************************************************
           ***************************************************************************
           **
           **  end of low-mode contractions for P - cvc - cvc 3pt
           **
           ***************************************************************************
           ***************************************************************************/


          /***************************************************************************/
          /***************************************************************************/


          /***************************************************************************
           ***************************************************************************
           **
           **  contractions for P - cvc - cvc 3pt with high-mode sequential propagator 
           **
           ***************************************************************************
           ***************************************************************************/

          /*************************************************
           * allocate memory for the contractions
           *************************************************/
          cvc_tensor_eo[0] = (double*)malloc( 32 * VOLUME * sizeof(double) );
          if( cvc_tensor_eo[0] == NULL  ) {
            fprintf(stderr, "[p2gg] could not allocate memory for contr. fields %s %d\n", __FILE__, __LINE__);
            EXIT(24);
          }
          cvc_tensor_eo[1] = cvc_tensor_eo[0] + 32 * Vhalf;

          /* loop on up-type and down-type flavor */
          for( iflavor=1; iflavor>=0; iflavor-- ) {

            if ( iflavor == 1 ) {
              memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
              memset(contact_term[1], 0, 8*sizeof(double));
            } else {
              complex_field_eq_complex_field_conj_ti_re (cvc_tensor_eo[0], (double)sequential_source_gamma_id_sign[ sequential_source_gamma_id ], 16*VOLUME );
              memset(contact_term[0], 0, 8*sizeof(double));
            }
            ratime = _GET_TIME;

             /* flavor-dependent sequential source momentum */
            const int seq_source_momentum[3] = { (1 - 2*iflavor) * g_seq_source_momentum[0],
                                                 (1 - 2*iflavor) * g_seq_source_momentum[1],
                                                 (1 - 2*iflavor) * g_seq_source_momentum[2] };

            if(g_cart_id == 0) fprintf(stdout, "# [p2gg] using flavor-dependent sequential source momentum (%d, %d, %d)\n", 
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2]);

            
            for(i=0; i<60; i++) {
              int eo_spinor_field_id_e     = iflavor * 120 + i;
              int eo_spinor_field_id_o     = eo_spinor_field_id_e + 60;
              int eo_seq_spinor_field_id_e = 240 + i;
              int eo_seq_spinor_field_id_o = eo_seq_spinor_field_id_e + 60;

              exitstatus = init_clover_eo_sequential_source(
                  eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_field[eo_seq_spinor_field_id_o],
                  eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
                  g_shifted_sequential_source_timeslice, gauge_field_with_phase, mzzinv[iflavor][0], g_seq_source_momentum, sequential_source_gamma_id, eo_spinor_work[0]);
              if(exitstatus != 0) {
                fprintf(stderr, "[p2gg] Error from init_clover_eo_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(25);
              }
              /* set even part of sequential source to zero */
              memset( eo_spinor_field[eo_seq_spinor_field_id_e] , 0, sizeof_eo_spinor_field );
            }  /* end of loop on spin-color */


            /* stochastic inversion on odd, orthogonal subspace */
            exitstatus = Q_clover_eo_invert_subspace_stochastic ( &(eo_spinor_field[240]), &(eo_spinor_field[300]),
                                                                  &(eo_spinor_field[240]), &(eo_spinor_field[300]),
                                                                  60, eo_sample_field[g_nsample], eo_sample_field[0], g_nsample,
                                                                  gauge_field_with_phase, mzz, mzzinv, iflavor, eo_spinor_work );
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg] Error from Q_clover_eo_invert_subspace_stochastic, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(41);
            }

            /* contract */
            if( iflavor == 1 ) {
              contract_cvc_tensor_eo(cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[iflavor], uprop_list_e, uprop_list_o, tprop_list_e, tprop_list_o, gauge_field_with_phase);
            } else {
              contract_cvc_tensor_eo(cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[iflavor], dprop_list_e, dprop_list_o, tprop_list_e, tprop_list_o, gauge_field_with_phase);
            }

          }  /* end of loop on iflavor */
  
          /* subtract contact term */
          cvc_tensor_eo_subtract_contact_term ( cvc_tensor_eo, contact_term[0], gsx, (int)( source_proc_id == g_cart_id ) );

          /* momentum projections */
          exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cvc_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(28);
          }

          /***************************************************************************
           * write results to file
           ***************************************************************************/
          sprintf(aff_tag, "/hm/PJJ/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d",
                  gsx[0], gsx[1], gsx[2], gsx[3], g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2], sequential_source_gamma_id, g_sequential_source_timeslice);

          exitstatus = cvc_tensor_tp_write_to_aff_file (cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(29);
          }
          fini_3level_buffer(&cvc_tp);
          free( cvc_tensor_lexic ); cvc_tensor_lexic = NULL;
          free( cvc_tensor_eo[0] ); cvc_tensor_eo[0] = NULL; cvc_tensor_eo[1] = NULL;

          /***************************************************************************
           ***************************************************************************
           **
           ** end of contractions for P - cvc - cvc 3pt
           **   with high-mode sequential propagator 
           **
           ***************************************************************************
           ***************************************************************************/

        }  /* end of loop on sequential source timeslices */

      }  /* end of loop on sequential gamma id */

    }  /* end of loop on sequential source momentum */

#if 0
#endif /* of if 0 */

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
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

  free(eo_spinor_work[0]);
  free(eo_spinor_work);

  free(eo_spinor_field[0]);
  free(eo_spinor_field);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(eo_evecs_block);
#else
  exitstatus = tmLQCD_fini_deflator(_OP_ID_UP);
#endif
  free(eo_evecs_field);

  free(eo_sample_block);
  free(eo_sample_field);

  free ( evecs_eval );
  free ( evecs_lambdainv );


  /* free remaining clover matrix terms */
  clover_term_fini( &g_mzz_up    );
  clover_term_fini( &g_mzz_dn    );
  clover_term_fini( &g_mzzinv_up );
  clover_term_fini( &g_mzzinv_dn );

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
    fprintf(stdout, "# [p2gg] %s# [p2gg] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg] %s# [p2gg] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
