/****************************************************
* p2gg_noexdefl2.c
*
* Fri Dec 30 20:00:50 CET 2016
*
* - originally copied from p2gg.c
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
  int isample;
  unsigned int ix;
  int gsx[4], sx[4];
  int check_position_space_WI=0;
  int exitstatus;
  int source_proc_coords[4], source_proc_id = 0;
  int iseq_source_momentum;
  int isequential_source_gamma_id, isequential_source_timeslice;
  int io_proc = -1;
  unsigned int VOL3;
  size_t sizeof_spinor_field, sizeof_spinor_field_timeslice;
  double **stochastic_propagator_list = NULL, **propagator_list_up=NULL,
         **propagator_list_dn=NULL, **sequential_propagator_list=NULL, **stochastic_source_list=NULL;
  double **spinor_work=NULL;
  double *cvc_tensor[2] = {NULL, NULL}, contact_term[2][8];
  double *fwd_list[5][12], *bwd_list[5][12];
  char filename[100];
  double ratime, retime;
  double plaq;
  /* double **mzz[2], **mzzinv[2]; */
  
  
#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex *aff_buffer = NULL;
  char aff_buffer_path[200];
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
  
  while ((c = getopt(argc, argv, "wh?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
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
  
  g_the_time = time(NULL);
  
  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  fprintf(stdout, "# [p2gg_noexdefl2] Reading input from file %s\n", filename);
  read_input_parser(filename);
  
#ifdef HAVE_TMLQCD_LIBWRAPPER
  
  fprintf(stdout, "# [p2gg_noexdefl2] calling tmLQCD wrapper init functions\n");
  
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
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_noexdefl2] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
  {
  fprintf(stdout, "# [p2gg_noexdefl2] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
  }
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_noexdefl2] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif
  
  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg_noexdefl2] Error from init_geometry\n");
    EXIT(4);
  }
  
  geometry();
  
  VOL3 = LX*LY*LZ;
  sizeof_spinor_field           = _GSI(VOLUME) * sizeof(double);
  sizeof_spinor_field_timeslice = _GSI(VOL3)   * sizeof(double);
  
#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [p2gg_noexdefl2] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg_noexdefl2] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg_noexdefl2] Nconf = %d\n", Nconf);
  
  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }
  
  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(6);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg_noexdefl2] Error, &g_gauge_field is NULL\n");
    EXIT(7);
  }
#endif
  
#ifdef HAVE_MPI
  xchange_gauge();
#endif
  
  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_noexdefl2] measured plaquette value: %25.16e\n", plaq);
  
  /*************************************************
   * allocate memory for samples
   *************************************************/
  no_fields = g_nsample;
  stochastic_source_list = (double**)malloc(no_fields * sizeof(double*) );
  stochastic_source_list[0] = (double*)malloc(no_fields * sizeof_spinor_field );
  if(stochastic_source_list[0] == NULL) {
    fprintf(stderr, "[p2gg_noexdefl2] Error from malloc\n");
    EXIT(12);
  }
  for(i=1; i < no_fields; i++) stochastic_source_list[i] = stochastic_source_list[i-1] + _GSI(VOLUME);
  
  no_fields = g_nsample;
  stochastic_propagator_list = (double**)malloc(no_fields * sizeof(double*) );
  stochastic_propagator_list[0] = (double*)malloc(no_fields * sizeof_spinor_field);
  if(stochastic_propagator_list[0] == NULL) {
    fprintf(stderr, "[p2gg_noexdefl2] Error from malloc\n");
    EXIT(12);
  }
  for(i=1; i<no_fields; i++) stochastic_propagator_list[i] = stochastic_propagator_list[i-1] + _GSI(VOLUME);
  
  /*************************************************
   * allocate memory for spinor fields 
   * WITHOUT HALO
   *************************************************/
  no_fields = 60;
  propagator_list_up = (double**)calloc(no_fields, sizeof(double*));
  propagator_list_up[0] = (double*)malloc( no_fields * sizeof_spinor_field);
  if(propagator_list_up[0] == NULL) {
    fprintf(stderr, "[p2gg_noexdefl2] Error from calloc\n");
    EXIT(13);
  }
  for(i=1; i<no_fields; i++) propagator_list_up[i] = propagator_list_up[i-1] + _GSI(VOLUME);
  
  propagator_list_dn = (double**)calloc(no_fields, sizeof(double*));
  propagator_list_dn[0] = (double*)malloc( no_fields * sizeof_spinor_field);
  if(propagator_list_dn[0] == NULL) {
    fprintf(stderr, "[p2gg_noexdefl2] Error from calloc\n");
    EXIT(13);
  }
  for(i=1; i<no_fields; i++) propagator_list_dn[i] = propagator_list_dn[i-1] + _GSI(VOLUME);
  
  sequential_propagator_list = (double**)calloc(no_fields, sizeof(double*));
  sequential_propagator_list[0] = (double*)malloc( no_fields * sizeof_spinor_field);
  if(sequential_propagator_list[0] == NULL) {
    fprintf(stderr, "[p2gg_noexdefl2] Error from calloc\n");
    EXIT(13);
  }
  for(i=1; i<no_fields; i++) sequential_propagator_list[i] = sequential_propagator_list[i-1] + _GSI(VOLUME);
  
  /*************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   *************************************************/
  no_fields = 2;
  spinor_work = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&spinor_work[i], VOLUME+RAND);
  
  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
#if 0
  clover_term_init(&g_clover, 6);
  
  clover_term_init(&g_mzz_up, 6);
  clover_term_init(&g_mzz_dn, 6);
  clover_term_init(&g_mzzinv_up, 8);
  clover_term_init(&g_mzzinv_dn, 8);
  
  ratime = _GET_TIME;
  clover_term_eo (g_clover, g_gauge_field);
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
  
  ratime = _GET_TIME;
  clover_mzz_inv_matrix (g_mzzinv_up, g_mzz_up);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);
  
  ratime = _GET_TIME;
  clover_mzz_inv_matrix (g_mzzinv_dn, g_mzz_dn);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);
  
  mzz[0] = g_mzz_up;
  mzz[1] = g_mzz_dn;
  mzzinv[0] = g_mzzinv_up;
  mzzinv[1] = g_mzzinv_dn;
#endif

  /***********************************************************
   * init gamma permutations
   ***********************************************************/
  init_contract_cvc_tensor_gperm();
  
#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [p2gg_noexdefl2] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [p2gg_noexdefl2] proc%.4d is send process\n", g_cart_id);
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
      fprintf(stderr, "[p2gg_noexdefl2] Error, io proc must be id 0 in g_tr_comm\n");
      EXIT(14);
    }
  }
#endif
  
  /***********************************************
   ***********************************************
   **
   ** stochastic timeslice propagators for
   **  one-end-trick
   **
   ***********************************************
   ***********************************************/
  
  /***********************************************
   * init rng
   ***********************************************/
  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_noexdefl2] Error from init_rng_stat_file status was %d\n", exitstatus);
    EXIT(38);
  }
  
  /* initialize block of 4 x g_nsample_oet spinor fields to zero */
  memset( stochastic_source_list[0], 0, g_nsample * sizeof_spinor_field );
  memset( stochastic_propagator_list[0], 0, g_nsample * sizeof_spinor_field );
  
  /*******************************************************
   * loop on inversions for individual global timeslices
   *******************************************************/
  exitstatus = prepare_volume_source ( stochastic_source_list[0], g_nsample * VOLUME );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[] Error from init_timeslice_source_oet, status was %d\n", exitstatus);
    return(1);
  }

  for(isample = 0; isample < g_nsample; isample++) {
  
    memcpy( spinor_work[0], stochastic_source_list[isample], sizeof_spinor_field );
  
    memset( spinor_work[1], 0, sizeof_spinor_field);
    /* invert for up-type */
    exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], _OP_ID_UP, 0);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg_noexdefl2] Error from tmLQCD_invert, status was %d\n", exitstatus);
      EXIT(19);
    }
    memcpy(stochastic_propagator_list[isample], spinor_work[1], sizeof_spinor_field);
  
  }  /* end of loop on samples */
  
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
    fprintf(stdout, "# [p2gg_noexdefl2] using aff version %s\n", aff_status_str);
  
    sprintf(filename, "%s.%.4d.aff", outfile_prefix, Nconf);
    fprintf(stdout, "# [p2gg_noexdefl2] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[p2gg_noexdefl2] Error from aff_writer, status was %s\n", aff_status_str);
      EXIT(15);
    }
  
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[p2gg_noexdefl2] Error, aff writer is not initialized\n");
      EXIT(16);
    }
  
  }  /* end of if io_proc == 2 */
#endif
  
  
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
    source_proc_id = 0;
#if HAVE_MPI
    source_proc_coords[0] = gsx[0] / T;
    source_proc_coords[1] = gsx[1] / LX;
    source_proc_coords[2] = gsx[2] / LY;
    source_proc_coords[3] = gsx[3] / LZ;
  
    if(g_cart_id == 0) {
      fprintf(stdout, "# [p2gg_noexdefl2] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [p2gg_noexdefl2] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
    }
  
    MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
    if(source_proc_id == g_cart_id) {
      fprintf(stdout, "# [p2gg_noexdefl2] process %2d has source location\n", source_proc_id);
    }
#endif
    sx[0] = gsx[0] % T;
    sx[1] = gsx[1] % LX;
    sx[2] = gsx[2] % LY;
    sx[3] = gsx[3] % LZ;
  
    /* init Usource and source_proc_id */
    init_contract_cvc_tensor_usource(g_gauge_field, gsx, co_phase_up);
  
    
    /**********************************************************
     **********************************************************
     **
     ** propagators with source at gsx and gsx + mu
     **
     **********************************************************
     **********************************************************/
  
    for(mu=0; mu<5; mu++) {  /* loop on shifts in direction */
      /**********************************************************
       * shifted source coords and source proc coords
       **********************************************************/
      int shifted_source_coords[4], shifted_source_proc_id=0;
      memcpy( shifted_source_coords, gsx, 4*sizeof(int));
      switch(mu) {
        case 0: shifted_source_coords[0] = ( shifted_source_coords[0] + 1 ) % T_global; break;;
        case 1: shifted_source_coords[1] = ( shifted_source_coords[1] + 1 ) % LX_global; break;;
        case 2: shifted_source_coords[2] = ( shifted_source_coords[2] + 1 ) % LY_global; break;;
        case 3: shifted_source_coords[3] = ( shifted_source_coords[3] + 1 ) % LZ_global; break;;
      }
#if HAVE_MPI
      int shifted_source_proc_coords[4];
      shifted_source_proc_coords[0] = shifted_source_coords[0] / T;
      shifted_source_proc_coords[1] = shifted_source_coords[1] / LX;
      shifted_source_proc_coords[2] = shifted_source_coords[2] / LY;
      shifted_source_proc_coords[3] = shifted_source_coords[3] / LZ;
  
      if(g_cart_id == 0) {
        fprintf(stdout, "# [p2gg_noexdefl2] global shifted source coordinates: (%3d,%3d,%3d,%3d)\n",  shifted_source_coords[0], shifted_source_coords[1],
            shifted_source_coords[2], shifted_source_coords[3]);
        fprintf(stdout, "# [p2gg_noexdefl2] shifted source proc coordinates: (%3d,%3d,%3d,%3d)\n",  shifted_source_proc_coords[0], shifted_source_proc_coords[1],
            shifted_source_proc_coords[2], shifted_source_proc_coords[3]);
      }
  
      exitstatus = MPI_Cart_rank(g_cart_grid, shifted_source_proc_coords, &shifted_source_proc_id);
      if(exitstatus != MPI_SUCCESS ) {
        fprintf(stderr, "[p2gg_noexdefl2] Error from MPI_Cart_rank; status was %d\n", exitstatus);
        EXIT(17);
      }
      if(shifted_source_proc_id == g_cart_id) {
        fprintf(stdout, "# [p2gg_noexdefl2] process %2d has shifted source location\n", shifted_source_proc_id);
      }
#endif
  
      /**********************************************************
       * up-type and dn-type propagators
       **********************************************************/
      for(i=0; i<12; i++) {
  
        int spinor_field_id = mu * 12 + i;
  
        memset(spinor_work[1], 0, sizeof_spinor_field);
        memset(spinor_work[0], 0, sizeof_spinor_field);
        /* process, which has source location, set point source */
        if ( source_proc_id == g_cart_id ) {
          spinor_work[0][_GSI( g_ipt[shifted_source_coords[0]%T][shifted_source_coords[1]%LX][shifted_source_coords[2]%LY][shifted_source_coords[3]%LZ] ) + 2*i] = 1.0;
        }
  
        /* invert for up */
        exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], _OP_ID_UP, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg_noexdefl2] Error from tmLQCD_invert, status was %d\n", exitstatus);
          EXIT(19);
        }
        memcpy( propagator_list_up[spinor_field_id], spinor_work[1], sizeof_spinor_field );
  
        memset(spinor_work[1], 0, sizeof_spinor_field);
        /* invert for dn */
        exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], _OP_ID_DN, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg_noexdefl2] Error from tmLQCD_invert, status was %d\n", exitstatus);
          EXIT(19);
        }
        memcpy( propagator_list_dn[spinor_field_id], spinor_work[1], sizeof_spinor_field );
  
  
      }  /* end of loop on spin-color */
  
    }    /* end of loop on shift direction mu */
  
  
    /***************************************************************************
     * loop on sequential source gamma matrices
     ***************************************************************************/
    for(iseq_source_momentum=0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) {
  
      g_seq_source_momentum[0] = g_seq_source_momentum_list[iseq_source_momentum][0];
      g_seq_source_momentum[1] = g_seq_source_momentum_list[iseq_source_momentum][1];
      g_seq_source_momentum[2] = g_seq_source_momentum_list[iseq_source_momentum][2];
  
      if(g_cart_id == 0) fprintf(stdout, "# [p2gg_noexdefl2] using sequential source momentum no. %2d = (%d, %d, %d)\n", iseq_source_momentum,
        g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);
  
      /***************************************************************************
       * loop on sequential source gamma matrices
       ***************************************************************************/
      for(isequential_source_gamma_id=0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++) {
  
        int sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];
        if(g_cart_id == 0) fprintf(stdout, "# [p2gg_noexdefl2] using sequential source gamma id no. %2d = %d\n", isequential_source_gamma_id, sequential_source_gamma_id);
  
        /***************************************************************************
         * loop on sequential source time slices
         ***************************************************************************/
        for(isequential_source_timeslice=0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++) {
  
          g_sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];
          /* shift sequential source timeslice by source timeslice gsx[0] */
          int shifted_sequential_source_timeslice = ( gsx[0] + g_sequential_source_timeslice ) % T_global;
  
          if(g_cart_id == 0) fprintf(stdout, "# [p2gg_noexdefl2] using sequential source timeslice %d\n", g_sequential_source_timeslice);
  
  
          /***************************************************************************
           ***************************************************************************
           **
           **  contractions for P - cvc - cvc 3pt with sequential propagator 
           **
           ***************************************************************************
           ***************************************************************************/
  
          /*************************************************
           * allocate memory for the contractions
           *************************************************/
          unsigned int items = 32 * VOLUME;
          size_t       bytes = items * sizeof(double);
          cvc_tensor[0] = (double*)malloc(bytes);
          cvc_tensor[1] = (double*)malloc(bytes);
          if( cvc_tensor[0] == NULL || cvc_tensor[1] == NULL ) {
            fprintf(stderr, "[p2gg_noexdefl2] could not allocate memory for contr. fields\n");
            EXIT(24);
          }
  
          memset(cvc_tensor[0], 0, bytes );
          memset(cvc_tensor[1], 0, bytes );
          memset(contact_term[0], 0, 8*sizeof(double));
          memset(contact_term[1], 0, 8*sizeof(double));
  
          /* loop on up-type and down-type flavor */
          for(iflavor=0; iflavor<2; iflavor++) {
  
            ratime = _GET_TIME;
  
             /* flavor-dependent sequential source momentum */
            int seq_source_momentum[3] = { (1 - 2*iflavor) * g_seq_source_momentum[0],
                                           (1 - 2*iflavor) * g_seq_source_momentum[1],
                                           (1 - 2*iflavor) * g_seq_source_momentum[2] };
  
            if(g_cart_id == 0) fprintf(stdout, "# [p2gg_noexdefl2] using flavor-dependent sequential source momentum (%d, %d, %d)\n", 
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2]);
 
           if( iflavor == 0 )  {
  
              exitstatus = prepare_seqn_stochastic_vertex_propagator_sliced3d (sequential_propagator_list, stochastic_propagator_list, stochastic_source_list,
                  propagator_list_up, g_nsample, 60, seq_source_momentum, sequential_source_gamma_id );
           } else {
              exitstatus = prepare_seqn_stochastic_vertex_propagator_sliced3d (sequential_propagator_list, stochastic_source_list, stochastic_propagator_list,
                  propagator_list_dn, g_nsample, 60, seq_source_momentum, sequential_source_gamma_id );
           }
           if( exitstatus != 0 ) {
             fprintf(stderr, "[p2gg_noexdefl2] Error from prepare_seqn_stochastic_vertex_propagator_sliced3d, status was %d\n", exitstatus);
             EXIT(43);
           }

            /***************************************************************************
             * set fwd_list_eo and bwd_list_eo
             ***************************************************************************/
            for(mu=0; mu<5; mu++) {
              for(i=0; i<12; i++) {
                /* the sequential propagator */
                fwd_list[mu][i] = sequential_propagator_list[12*mu+i];
  
                /* dn propagator or up propagator for iflavor = 0 / 1 */
                bwd_list[mu][i] = (iflavor==0) ? propagator_list_dn[12*mu+i] : propagator_list_up[12*mu+i];
  
              }
            }
  
            ratime = _GET_TIME;
            contract_cvc_tensor(cvc_tensor[iflavor], contact_term[iflavor], fwd_list, bwd_list, NULL, NULL);
  
            retime = _GET_TIME;
            if(g_cart_id==0) fprintf(stdout, "# [p2gg_noexdefl2] time for cvc_tensor contraction = %e seconds\n", retime-ratime);
  
          }  /* end of loop on iflavor */
  
          /***************************************************************************
           * combine up-type and dn-type part, normalisation of contractions
           ***************************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix)
#endif
          for(ix=0; ix<16*VOLUME; ix++) {
            /* real part */
            cvc_tensor[0][2*ix  ] += sequential_source_gamma_id_sign[ sequential_source_gamma_id ] * cvc_tensor[1][2*ix  ];
            cvc_tensor[0][2*ix  ] *= -0.25;
            /* imaginary part */
            cvc_tensor[0][2*ix+1] -= sequential_source_gamma_id_sign[ sequential_source_gamma_id ] * cvc_tensor[1][2*ix+1];
            cvc_tensor[0][2*ix+1] *= -0.25;
          }
          free(cvc_tensor[1]);
  
          /***************************************************************************
           * subtract contact term
           ***************************************************************************/
          if(source_proc_id == g_cart_id) {
            ix = g_ipt[sx[0]][sx[1]][sx[2]][sx[3]];
            cvc_tensor[0][_GWI( 0,ix, VOLUME)  ] -= contact_term[0][ 0];
            cvc_tensor[0][_GWI( 0,ix, VOLUME)+1] -= contact_term[0][ 1];
            cvc_tensor[0][_GWI( 5,ix, VOLUME)  ] -= contact_term[0][ 2];
            cvc_tensor[0][_GWI( 5,ix, VOLUME)+1] -= contact_term[0][ 3];
            cvc_tensor[0][_GWI(10,ix, VOLUME)  ] -= contact_term[0][ 4];
            cvc_tensor[0][_GWI(10,ix, VOLUME)+1] -= contact_term[0][ 5];
            cvc_tensor[0][_GWI(15,ix, VOLUME)  ] -= contact_term[0][ 6];
            cvc_tensor[0][_GWI(15,ix, VOLUME)+1] -= contact_term[0][ 7];
          }
  
          /***************************************************************************
           * momentum projections
           ***************************************************************************/
  
          double ***cvc_tp = NULL;
          exitstatus = init_3level_buffer(&cvc_tp, g_sink_momentum_number, 16, 2*T);
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg_noexdefl2] Error from init_3level_buffer, status was %d\n", exitstatus);
            EXIT(26);
          }
  
          ratime = _GET_TIME;
  
          exitstatus = momentum_projection (cvc_tensor[0], cvc_tp[0][0], T*16, g_sink_momentum_number, g_sink_momentum_list);
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg_noexdefl2] Error from momentum_projection, status was %d\n", exitstatus);
            EXIT(26);
          }
          retime = _GET_TIME;
          if(g_cart_id==0) fprintf(stdout, "# [p2gg_noexdefl2] time for momentum projection = %e seconds\n", retime-ratime);
  
          /***************************************************************************
           * write results to file
           ***************************************************************************/
          ratime = _GET_TIME;
  
          items = g_sink_momentum_number * 16 * T_global;
          bytes = items * sizeof(double _Complex);
          if(io_proc == 2) {
            aff_buffer = (double _Complex*)malloc(bytes);
            if(aff_buffer == NULL) {
              fprintf(stderr, "[p2gg_noexdefl2] Error from malloc\n");
              EXIT(27);
            }
          }
  
#ifdef HAVE_MPI
          i = g_sink_momentum_number * 32 * T;
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
          if(io_proc>0) {
            exitstatus = MPI_Gather(cvc_tp[0][0], i, MPI_DOUBLE, aff_buffer, i, MPI_DOUBLE, 0, g_tr_comm);
            if(exitstatus != MPI_SUCCESS) {
              fprintf(stderr, "[p2gg_noexdefl2] Error from MPI_Gather, status was %d\n", exitstatus);
              EXIT(28);
            }
          }
#else
          exitstatus = MPI_Gather(cvc_tp[0][0], i, MPI_DOUBLE, aff_buffer, i, MPI_DOUBLE, 0, g_cart_grid);
          if(exitstatus != MPI_SUCCESS) {
            fprintf(stderr, "[p2gg_noexdefl2] Error from MPI_Gather, status was %d\n", exitstatus);
            EXIT(44);
          }
#endif
#else
          memcpy(aff_buffer, cvc_tp[0][0], bytes);
#endif
  
          fini_3level_buffer(&cvc_tp);
  
          if(io_proc == 2) {
            for(i=0; i<g_sink_momentum_number; i++) {
              sprintf(aff_buffer_path, "/lm/PJJ/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/px%.2dpy%.2dpz%.2d", 
                  gsx[0], gsx[1], gsx[2], gsx[3],
                  g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
                  g_sequential_source_gamma_id, g_sequential_source_timeslice, 
                  g_sink_momentum_list[i][0], g_sink_momentum_list[i][1], g_sink_momentum_list[i][2]);
              fprintf(stdout, "# [p2gg_noexdefl2] current aff path = %s\n", aff_buffer_path);
              affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
              exitstatus = aff_node_put_complex (affw, affdir, aff_buffer+16*T_global*i, (uint32_t)T_global*16);
              if(exitstatus != 0) {
                fprintf(stderr, "[p2gg_noexdefl2] Error from aff_node_put_double, status was %d\n", exitstatus);
                EXIT(29);
              }
            }
            free(aff_buffer);
          }  /* if io_proc == 2 */
  
          retime = _GET_TIME;
          if(g_cart_id==0) fprintf(stdout, "# [p2gg_noexdefl2] time for saving momentum space results = %e seconds\n", retime-ratime);
          
          if(check_position_space_WI) {
            exitstatus = check_cvc_wi_position_space (cvc_tensor[0]);
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg_noexdefl2] Error from check_cvc_wi_position_space, status was %d\n", exitstatus);
              EXIT(38);
            }
          }
  
          free(cvc_tensor[0]);
  
          /***************************************************************************
           ***************************************************************************
           **
           **  end of contractions for P - cvc - cvc 3pt
           **
           ***************************************************************************
           ***************************************************************************/
  
        }  /* end of loop on sequential gamma id */
  
      }  /* end of loop on sequential source momentum */
  
    }  /* end of loop on sequential source timeslices */
  
  }  /* end of loop on source locations */

#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[p2gg_noexdefl2] Error from aff_writer_close, status was %s\n", aff_status_str);
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

  free( propagator_list_up[0] );
  free( propagator_list_up );
  free( propagator_list_dn[0] );
  free( propagator_list_dn );

  free( sequential_propagator_list[0] );
  free( sequential_propagator_list );

  free( stochastic_source_list[0] );
  free( stochastic_source_list );
  free( stochastic_propagator_list[0] );
  free( stochastic_propagator_list );

  free( spinor_work[0] );
  free( spinor_work[1] );

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_noexdefl2] %s# [p2gg_noexdefl2] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_noexdefl2] %s# [p2gg_noexdefl2] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
