/****************************************************
 * p2gg_mixed.c
 *
 * Thu Apr 27 12:45:44 CEST 2017
 *
 * - originally copied from p2gg.cpp
 * - calculates mixed products of eigenvector or stochastic
 *   fields with point-to-all propagators
 *
 * - transient version ?
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
#include "scalar_products.h"
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

  const char outfile_prefix[] = "p2gg_mixed";
  const int block_size = 120;

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
  char filename[100];
  double ratime, retime;
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
  /* fprintf(stdout, "# [p2gg_mixed] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_mixed] calling tmLQCD wrapper init functions\n");

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
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_mixed] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_mixed] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_mixed] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg_mixed] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [p2gg_mixed] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg_mixed] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg_mixed] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(6);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg_mixed] Error, &g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[p2gg_mixed] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_mixed] Error from tmLQCD_get_deflator_params, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(9);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [p2gg_mixed] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [p2gg_mixed] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [p2gg_mixed] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [p2gg_mixed] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block == NULL) {
    fprintf(stderr, "[p2gg_mixed] Error, eo_evecs_block is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(10);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[p2gg_mixed] Error, dimension of eigenspace is zero %s %d\n", __FILE__, __LINE__);
    EXIT(11);
  }

  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[p2gg_mixed] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  evecs_eval = (double*)malloc(evecs_num*sizeof(double));
  evecs_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if(evecs_eval == NULL || evecs_lambdainv == NULL ) {
    fprintf(stderr, "[p2gg_mixed] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for(i=0; i<evecs_num; i++) {
    evecs_eval[i] = ((double*)(g_tmLQCD_defl.evals))[2*i];
    evecs_lambdainv[i] = 2.* g_kappa / evecs_eval[i];
    if( g_cart_id == 0 ) fprintf(stdout, "# [p2gg_mixed] eval %4d %16.7e\n", i, evecs_eval[i] );
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
  eo_sample_block = (double*)malloc( g_nsample * sizeof_eo_spinor_field);
  if(eo_sample_block == NULL) {
    fprintf(stderr, "[p2gg_mixed] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }
  eo_sample_field    = (double**)calloc( g_nsample, sizeof(double*));
  eo_sample_field[0] = eo_sample_block;
  for(i=1; i < g_nsample; i++) eo_sample_field[i] = eo_sample_field[i-1] + _GSI(Vhalf);

  /*************************************************
   * allocate memory for eo spinor fields 
   * WITHOUT HALO
   *************************************************/
  no_eo_fields = 120;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));

  eo_spinor_field[0] = (double*)malloc( no_eo_fields * sizeof_eo_spinor_field);
  if(eo_spinor_field[0] == NULL) {
    fprintf(stderr, "[p2gg_mixed] Error from calloc %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[p2gg_mixed] Error from calloc %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  for(i=1; i<no_eo_fields; i++) { eo_spinor_work[i] = eo_spinor_work[i-1] + _GSI( ((VOLUME+RAND)/2) ); }

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_mixed] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg-mixed] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(39);
  }

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [p2gg_mixed] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [p2gg_mixed] proc%.4d is send process\n", g_cart_id);
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
      fprintf(stderr, "[p2gg_mixed] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************
   ***********************************************
   **
   ** stochastic propagators
   **
   ** in-place
   ** for down-type propagator
   **
   ***********************************************
   ***********************************************/

  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_mixed] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }


  /* make a source */
  exitstatus = prepare_volume_source(eo_sample_block, g_nsample*Vhalf);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_mixed] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(33);
  }

  for(i = 0; i < g_nsample; i++) {
    /* orthogonal projection */
#if 0
    double norm;
    spinor_scalar_product_re ( &norm, eo_sample_field[i], eo_sample_field[i], Vhalf );
    if ( g_cart_id == 0 ) { fprintf(stdout, "# [p2gg_mixed] sample norm %25.16e\n", sqrt(norm) ); }
#endif

    /* apply C 
     * work1 <- C sample
     */
    C_clover_oo ( eo_spinor_work[1], eo_sample_field[i], gauge_field_with_phase, eo_spinor_work[2], g_mzz_up[1], g_mzzinv_up[0] );

    /* weighted parallel projection 
     * work1 <- V 2kappa/Lambda V^+ work1
     */
    exitstatus = project_propagator_field_weighted ( eo_spinor_work[1], eo_spinor_work[1], 1, eo_evecs_block, evecs_lambdainv, 1, evecs_num, Vhalf);
    if (exitstatus != 0) {
      fprintf(stderr, "[p2gg_mixed] Error from project_propagator_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(35);
    }

    /* work1 <- work1 * 2 kappa */
    spinor_field_ti_eq_re ( eo_spinor_work[1], 2*g_kappa, Vhalf);
    /* apply Cbar 
     * work0 <- Cbar work1
     */
    C_clover_oo ( eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_phase, eo_spinor_work[2], g_mzz_dn[1], g_mzzinv_dn[0] );

    /* work1 <- sample - work0 */
    spinor_field_eq_spinor_field_mi_spinor_field( eo_spinor_work[1], eo_sample_field[i], eo_spinor_work[0], Vhalf);

#if 0
    /* TEST W W^+ projection */
    spinor_scalar_product_re ( &norm, eo_spinor_work[1], eo_spinor_work[1], Vhalf );
    if ( g_cart_id == 0 ) { fprintf(stdout, "# [p2gg_mixed] proj orth sample norm %25.16e\n", sqrt(norm) ); }
    spinor_scalar_product_re ( &norm, eo_spinor_work[0], eo_spinor_work[0], Vhalf );
    if ( g_cart_id == 0 ) { fprintf(stdout, "# [p2gg_mixed] proj para sample norm %25.16e\n", sqrt(norm) ); }
    norm = sqrt( norm );

    for ( int  ievec=0; ievec<evecs_num; ievec++ ) {
      double norm2;
      complex w, w2;
      C_clover_oo ( eo_spinor_work[2], eo_evecs_field[ievec], gauge_field_with_phase, eo_spinor_work[3], g_mzz_dn[1], g_mzzinv_dn[0] );
      spinor_field_ti_eq_re ( eo_spinor_work[2], 2.*g_kappa/sqrt(evecs_eval[ievec]) , Vhalf);
      spinor_scalar_product_re ( &norm2, eo_spinor_work[2], eo_spinor_work[2], Vhalf );
      spinor_scalar_product_co ( &w, eo_spinor_work[2], eo_spinor_work[1], Vhalf );
      spinor_scalar_product_co ( &w2, eo_spinor_work[2], eo_spinor_work[0], Vhalf );
      if ( g_cart_id == 0 ) {
        fprintf(stdout, " vdag s %4d %25.16e %25.16e %25.16e %25.16e %25.16e\n", ievec, w.re/norm, w.im/norm, w2.re/norm, w2.im/norm , sqrt(norm2) );
      }
    }
#endif  /* of if 0 */

    /* invert */
    memset(eo_spinor_work[0], 0, sizeof_eo_spinor_field);
    exitstatus = _TMLQCD_INVERT_EO(eo_spinor_work[0], eo_spinor_work[1], _OP_ID_DN);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg_mixed] Error from _TMLQCD_INVERT_EO, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }
    memcpy( eo_sample_field[i], eo_spinor_work[0], sizeof_eo_spinor_field );

    if( check_propagator_residual ) {
      exitstatus = check_oo_propagator_clover_eo( &(eo_sample_field[i]), &(eo_spinor_work[1]), &(eo_spinor_work[2]), gauge_field_with_phase, g_mzz_dn, g_mzzinv_dn, 1 );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_mixed] Error from check_oo_propagator_clover_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(19);
      }
    }

  }  /* end of loop on samples */


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
      fprintf(stderr, "[p2gg_mixed] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

#ifdef HAVE_LHPC_AFF
    /***********************************************
     ***********************************************
     **
     ** writer for aff output file
     **
     ** one file per source location
     **
     ***********************************************
     ***********************************************/
    if(io_proc == 2) {
      aff_status_str = (char*)aff_version();
      fprintf(stdout, "# [p2gg_mixed] using aff version %s\n", aff_status_str);

      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [p2gg_mixed] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_mixed] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#endif



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
     ** up-type propagators with source at gsx and gsx + mu
     **
     **********************************************************
     **********************************************************/

    for(mu=0; mu<5; mu++)
    {  /* loop on shifts in direction mu */
      /**********************************************************
       * shifted source coords and source proc coords
       **********************************************************/
      int g_shifted_source_coords[4];
      memcpy( g_shifted_source_coords, gsx, 4*sizeof(int));
      switch(mu) {
        case 0: g_shifted_source_coords[0] = ( g_shifted_source_coords[0] + 1 ) % T_global; break;;
        case 1: g_shifted_source_coords[1] = ( g_shifted_source_coords[1] + 1 ) % LX_global; break;;
        case 2: g_shifted_source_coords[2] = ( g_shifted_source_coords[2] + 1 ) % LY_global; break;;
        case 3: g_shifted_source_coords[3] = ( g_shifted_source_coords[3] + 1 ) % LZ_global; break;;
      }

      /**********************************************************
       * up-type propagators
       **********************************************************/
      exitstatus = point_to_all_fermion_propagator_clover_eo ( &(eo_spinor_field[mu*12]), &(eo_spinor_field[60+12*mu]),  _OP_ID_UP,
        g_shifted_source_coords, gauge_field_with_phase, g_mzz_up, g_mzzinv_up, check_propagator_residual, eo_spinor_work );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_mixed] Error from point_to_all_fermion_propagator_clover_eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(21);
      }

      if ( mu < 4 ) {
        /**********************************************************
         * apply backward cvc vertex
         *   with constant gauge field from source location
         *
         **********************************************************/
        exitstatus = apply_constant_cvc_vertex_at_source ( &(eo_spinor_field[mu*12]), mu, 1, Vhalf );
      }

    }  /* end of loop on shifts */

    /**********************************************************
     **********************************************************
     **
     ** contractions for local vertex
     **
     ** up-type propagator
     **
     **********************************************************
     **********************************************************/
    sprintf( aff_tag, "/lm/up-prop/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_vdag_gloc_spinor_field ( 
        &(eo_spinor_field[0]), &(eo_spinor_field[60]), 60,
        eo_evecs_field, evecs_num, g_seq_source_momentum_number, g_seq_source_momentum_list, g_sequential_source_gamma_id_number, g_sequential_source_gamma_id_list,
        affw, aff_tag, io_proc, gauge_field_with_phase, mzz, mzzinv);

    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_mixed] Error from contract_vdag_gloc_spinor_field; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(22);
    }

    sprintf( aff_tag, "/hm/up-prop/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_vdag_gloc_spinor_field ( 
        &(eo_spinor_field[0]), &(eo_spinor_field[60]), 60,
        eo_sample_field, g_nsample, g_seq_source_momentum_number, g_seq_source_momentum_list, g_sequential_source_gamma_id_number, g_sequential_source_gamma_id_list,
        affw, aff_tag, io_proc, gauge_field_with_phase, mzz, mzzinv);

    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_mixed] Error from contract_vdag_gloc_spinor_field; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(22);
    }

    sprintf( aff_tag, "/lm/up-prop/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_vdag_cvc_spinor_field (
        &(eo_spinor_field[0]), &(eo_spinor_field[60]), 60,
        eo_evecs_field, evecs_num, g_seq_source_momentum_number, g_seq_source_momentum_list,
        affw, aff_tag, io_proc, gauge_field_with_phase, mzz, mzzinv, 60 );

    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_mixed] Error from contract_vdag_cvc_spinor_field; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(22);
    }

    sprintf( aff_tag, "/hm/up-prop/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_vdag_cvc_spinor_field (
        &(eo_spinor_field[0]), &(eo_spinor_field[60]), 60,
        eo_sample_field, g_nsample, g_seq_source_momentum_number, g_seq_source_momentum_list,
        affw, aff_tag, io_proc, gauge_field_with_phase, mzz, mzzinv, 60 );

    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_mixed] Error from contract_vdag_cvc_spinor_field; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(22);
    }


    /**********************************************************
     **********************************************************
     **
     ** dn-type propagators with source at gsx and gsx + mu
     **
     **********************************************************
     **********************************************************/

    for(mu=0; mu<5; mu++)
    {  /* loop on shifts in direction mu */
      /**********************************************************
       * shifted source coords and source proc coords
       **********************************************************/
      int g_shifted_source_coords[4];
      memcpy( g_shifted_source_coords, gsx, 4*sizeof(int));
      switch(mu) {
        case 0: g_shifted_source_coords[0] = ( g_shifted_source_coords[0] + 1 ) % T_global; break;;
        case 1: g_shifted_source_coords[1] = ( g_shifted_source_coords[1] + 1 ) % LX_global; break;;
        case 2: g_shifted_source_coords[2] = ( g_shifted_source_coords[2] + 1 ) % LY_global; break;;
        case 3: g_shifted_source_coords[3] = ( g_shifted_source_coords[3] + 1 ) % LZ_global; break;;
      }

      /**********************************************************
       * dn-type propagators
       **********************************************************/
      exitstatus = point_to_all_fermion_propagator_clover_eo ( &(eo_spinor_field[mu*12]), &(eo_spinor_field[60+12*mu]),  _OP_ID_DN,
        g_shifted_source_coords, gauge_field_with_phase, g_mzz_dn, g_mzzinv_dn, check_propagator_residual, eo_spinor_work );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_mixed] Error from point_to_all_fermion_propagator_clover_eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(21);
      }

      if ( mu < 4 ) {
        /**********************************************************
         * apply backward cvc vertex
         *   with constant gauge field from source location
         *
         **********************************************************/
        exitstatus = apply_constant_cvc_vertex_at_source ( &(eo_spinor_field[mu*12]), mu, 1, Vhalf );
      }

    }  /* end of loop on shifts */

    /**********************************************************
     **********************************************************
     **
     ** contractions for local vertex
     **
     ** dn-type propagator
     **
     **********************************************************
     **********************************************************/
    sprintf( aff_tag, "/lm/dn-prop/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_vdag_gloc_spinor_field ( 
        &(eo_spinor_field[0]), &(eo_spinor_field[60]), 60,
        eo_evecs_field, evecs_num, g_seq_source_momentum_number, g_seq_source_momentum_list, g_sequential_source_gamma_id_number, g_sequential_source_gamma_id_list,
        affw, aff_tag, io_proc, gauge_field_with_phase, mzz, mzzinv);

    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_mixed] Error from contract_vdag_gloc_spinor_field; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(22);
    }

    sprintf( aff_tag, "/hm/dn-prop/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_vdag_gloc_spinor_field ( 
        &(eo_spinor_field[0]), &(eo_spinor_field[60]), 60,
        eo_sample_field, g_nsample, g_seq_source_momentum_number, g_seq_source_momentum_list, g_sequential_source_gamma_id_number, g_sequential_source_gamma_id_list,
        affw, aff_tag, io_proc, gauge_field_with_phase, mzz, mzzinv);

    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_mixed] Error from contract_vdag_gloc_spinor_field; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(22);
    }

    sprintf( aff_tag, "/lm/dn-prop/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_vdag_cvc_spinor_field (
        &(eo_spinor_field[0]), &(eo_spinor_field[60]), 60,
        eo_evecs_field, evecs_num, g_seq_source_momentum_number, g_seq_source_momentum_list,
        affw, aff_tag, io_proc, gauge_field_with_phase, mzz, mzzinv, 60 );

    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_mixed] Error from contract_vdag_cvc_spinor_field; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(22);
    }

    sprintf( aff_tag, "/hm/dn-prop/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_vdag_cvc_spinor_field (
        &(eo_spinor_field[0]), &(eo_spinor_field[60]), 60,
        eo_sample_field, g_nsample, g_seq_source_momentum_number, g_seq_source_momentum_list,
        affw, aff_tag, io_proc, gauge_field_with_phase, mzz, mzzinv, 60 );

    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_mixed] Error from contract_vdag_cvc_spinor_field; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(22);
    }

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_mixed] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
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
  fini_clover();

  free_geometry();

#if 0
#endif  /* of if 0 */


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
    fprintf(stdout, "# [p2gg_mixed] %s# [p2gg_mixed] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_mixed] %s# [p2gg_mixed] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
