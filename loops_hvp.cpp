/****************************************************
 * loops_hvp.cpp
 *
 * Fr 22. Dez 14:21:36 CET 2017
 *
 * - originally copied from loops_em.cpp
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
#include "fft.h"
#include "Q_phi.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#define _SQR(_a) ((_a)*(_a))

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform loop contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc    [default cvc.input]\n");
  fprintf(stdout, "          -w                  : check position space WI   [default false]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

/******************************************************/
/******************************************************/

int dummy_eo_solver (double * const propagator, double * const source, const int op_id) {
  memcpy(propagator, source, _GSI(VOLUME)/2*sizeof(double) );
  return(0);
}

/******************************************************/
/******************************************************/

int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  int check_position_space_WI=0;
  int exitstatus;
  int io_proc = -1;
  int evecs_num = 0;
  int check_propagator_residual = 0;
  int transverse_projection = 0;
  int switch_evecs = 0;
  unsigned int Vhalf, VOL3half;
  size_t sizeof_eo_spinor_field, sizeof_eo_spinor_field_timeslice;
  double **eo_stochastic_source = NULL, **eo_stochastic_propagator = NULL;
  double **eo_spinor_field=NULL, **eo_spinor_work=NULL, *eo_evecs_block=NULL;
  double **eo_evecs_field=NULL;
  double **eo_evecs_field2=NULL;
  double *evecs_eval = NULL, *evecs_lambdainv=NULL, *evecs_4kappasqr_lambdainv = NULL;
  char filename[100], contype[600];
  /* double ratime, retime; */
  double **mzz[2], **mzzinv[2];
  double *gauge_field_with_phase = NULL;
  double ***local_loop_x = NULL, ***local_loop_p = NULL;
  double ****cvc_loop_lma_x = NULL;
  double ***cvc_loop_tp = NULL;
  double **cvc_wi = NULL;

  double _Complex ztmp;

#ifdef HAVE_MPI
  MPI_Status mstatus;
#endif


#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
  char aff_tag[400];
  struct AffNode_s *affn = NULL, *affdir=NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "cwh?f:p:s:")) != -1) {
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
    case 'p':
      transverse_projection = atoi( optarg );
      break;
    case 's':
      switch_evecs = atoi( optarg );
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
  /* fprintf(stdout, "# [loops_hvp] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [loops_hvp] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [loops_hvp] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [loops_hvp] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [loops_hvp] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[loops_hvp] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[loops_hvp] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  Vhalf = VOLUME / 2;
  VOL3half = LX * LY * LZ / 2;
  sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);
  sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [loops_hvp] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [loops_hvp] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[loops_hvp] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[loops_hvp] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /******************************************************/
  /******************************************************/

#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[loops_hvp] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[loops_hvp] Error from tmLQCD_get_deflator_params, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(9);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [loops_hvp] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [loops_hvp] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [loops_hvp] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [loops_hvp] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block == NULL) {
    fprintf(stderr, "[loops_hvp] Error, eo_evecs_block is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(10);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[loops_hvp] Error, dimension of eigenspace is zero %s %d\n", __FILE__, __LINE__);
    EXIT(11);
  }

#if 0
  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[loops_hvp] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }
#endif  /* of if 0 */

  evecs_eval                = (double*)malloc(evecs_num*sizeof(double));
  evecs_lambdainv           = (double*)malloc(evecs_num*sizeof(double));
  evecs_4kappasqr_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if(    evecs_eval                == NULL 
      || evecs_lambdainv           == NULL 
      || evecs_4kappasqr_lambdainv == NULL 
    ) {
    fprintf(stderr, "[loops_hvp] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for( int i = 0; i < evecs_num; i++) {
    evecs_eval[i]                = ((double*)(g_tmLQCD_defl.evals))[2*i];
    evecs_lambdainv[i]           = 2.* g_kappa / evecs_eval[i];
    evecs_4kappasqr_lambdainv[i] = 4.* g_kappa * g_kappa / evecs_eval[i];
    if( g_cart_id == 0 ) fprintf(stdout, "# [loops_hvp] eval %4d %16.7e\n", i, evecs_eval[i] );
  }

#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */

  /******************************************************/
  /******************************************************/

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
    fprintf(stderr, "[loops_hvp] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(1);
  }

  if ( switch_evecs == 1 ) {
    /*************************************************
     * allocate memory for dn-type eo evecs
     * NO HALO
     *************************************************/
    if ( ( exitstatus = init_2level_buffer ( &eo_evecs_field2, evecs_num, _GSI(Vhalf) ) ) != 0 ) {
      fprintf(stderr, "[loops_hvp] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }

  }

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[loops_hvp] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from init_clover, status was %d\n", exitstatus);
    EXIT(1);
  }

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [loops_hvp] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [loops_hvp] proc%.4d is send process\n", g_cart_id);
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
      fprintf(stderr, "[loops_hvp] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * check eigenvectors
   ***********************************************************/
  exitstatus = init_2level_buffer ( &eo_spinor_field, 2, _GSI(Vhalf));
  if( exitstatus != 0 ) {
    fprintf(stderr, "[loops_caa_lma] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(123);
  }

  for( int i = 0; i < evecs_num; i++)
  {
    double norm;
    complex w;

    /*
    w.re =  eo_evecs_field[i][0] / ( eo_evecs_field[i][0] * eo_evecs_field[i][0] + eo_evecs_field[i][1] * eo_evecs_field[i][1]);
    w.im = -eo_evecs_field[i][1] / ( eo_evecs_field[i][0] * eo_evecs_field[i][0] + eo_evecs_field[i][1] * eo_evecs_field[i][1]);
#ifdef HAVE_MPI
    MPI_Bcast(&w, 2, MPI_DOUBLE, 0, g_cart_grid);
#endif
    spinor_field_eq_spinor_field_ti_co ( eo_evecs_field[i], eo_evecs_field[i], w, Vhalf);
    */

    if ( switch_evecs == 0 ) {
      /* apply C[ dn-mu = - ] = Cbar */
      C_clover_oo (eo_spinor_field[0], eo_evecs_field[i],  gauge_field_with_phase, eo_spinor_work[2], g_mzz_dn[1], g_mzzinv_dn[0]);
      /* apply C[ up-mu = + ] = C  */
      C_clover_oo (eo_spinor_field[1], eo_spinor_field[0], gauge_field_with_phase, eo_spinor_work[2], g_mzz_up[1], g_mzzinv_up[0]);

      spinor_scalar_product_re(&norm, eo_evecs_field[i],  eo_evecs_field[i], Vhalf);
      spinor_scalar_product_co(&w,    eo_spinor_field[1], eo_evecs_field[i], Vhalf);

      w.re *= 4.*g_kappa*g_kappa;
      w.im *= 4.*g_kappa*g_kappa;

      if(g_cart_id == 0) {
        fprintf(stdout, "# [loops_hvp] evec %.4d norm = %25.16e w = %25.16e +I %25.16e diff = %25.16e\n", i, norm, w.re, w.im, fabs( w.re-evecs_eval[i]));
      }

    } else if ( switch_evecs == 1 ) {
    /***********************************************************
     * keep Cbar v_i
     ***********************************************************/
      if ( io_proc == 2 && i == 0 ) fprintf( stdout, "# [loops_hvp] switching eigenvector flavor\n");
      /* apply C[ dn-mu = - ] = Cbar */
      C_clover_oo (eo_evecs_field2[i], eo_evecs_field[i],  gauge_field_with_phase, eo_spinor_work[2], g_mzz_up[1], g_mzzinv_up[0]);
      spinor_scalar_product_re ( &norm, eo_evecs_field[i], eo_evecs_field[i], Vhalf);
      spinor_scalar_product_co ( &w,    eo_evecs_field2[i], eo_evecs_field2[i], Vhalf);

      w.re *= 4.*g_kappa*g_kappa;
      w.im *= 4.*g_kappa*g_kappa;

      if(g_cart_id == 0) {
        fprintf(stdout, "# [loops_hvp] evec %.4d norm = %25.16e w = %25.16e +I %25.16e diff = %25.16e\n", i, norm, w.re, w.im, fabs( w.re-evecs_eval[i]));
      }

      spinor_scalar_product_re ( &norm, eo_evecs_field2[i], eo_evecs_field2[i], Vhalf);
      norm = 1. / sqrt( norm );
      spinor_field_ti_eq_re ( eo_evecs_field2[i], norm, Vhalf );

    }  /* end of if swtich evecs equals 1 */

  }  /* end of loops on eigenvectors */

  fini_2level_buffer ( &eo_spinor_field );

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * initialize random number generator
   ***********************************************/
  if ( ( exitstatus = init_rng_stat_file (g_seed, NULL) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * open AFF file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    if ( switch_evecs == 0 ) {
      sprintf(filename, "%s.%.4d.aff", "loops_hvp", Nconf );
    } else if ( switch_evecs == 1 )  {
      sprintf(filename, "%s.%.4d.aff", "loops_hvp_switched", Nconf );
    }
    fprintf(stdout, "# [loops_hvp] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[loops_hvp] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      EXIT(1);
    }
  }  /* end of if io_proc == 2 */
#endif

  /***********************************************************/
  /***********************************************************/

#if 0
  /***********************************************
   * local lma loops
   ***********************************************/

  if( ( exitstatus = init_3level_buffer ( &local_loop_x, 2, 16, 2*Vhalf ) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }
  /* position space */

  if ( switch_evecs == 0 ) {
    contract_local_loop_eo_lma ( local_loop_x, eo_evecs_field, evecs_4kappasqr_lambdainv, evecs_num, gauge_field_with_phase, mzz, mzzinv );
  } else if ( switch_evecs == 1 ) {
    contract_local_loop_eo_lma ( local_loop_x, eo_evecs_field2, evecs_4kappasqr_lambdainv, evecs_num, gauge_field_with_phase, mzz, mzzinv );
  }

  /* momentum space */
  if ( ( exitstatus = cvc_loop_eo_momentum_projection ( &local_loop_p, local_loop_x, 16, g_sink_momentum_list, g_sink_momentum_number) ) != 0 ) {
    fprintf ( stderr, "[loop_em] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /* write to file */
  sprintf(aff_tag, "/loop/local/nev%.4d", evecs_num );
  if( ( exitstatus = cvc_loop_tp_write_to_aff_file ( local_loop_p, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc ) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }


  fini_3level_buffer ( &local_loop_x );
  fini_3level_buffer ( &local_loop_p );
#endif
  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * cvc lma loops
   ***********************************************/
  if( ( exitstatus = init_4level_buffer ( &cvc_loop_lma_x, 2, 4, 2, 2*Vhalf ) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from init_4level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  /***********************************************
   * cvc fwd and bwd contraction in position space
   ***********************************************/
  if ( switch_evecs == 0 ) {
    contract_cvc_loop_eo_lma ( cvc_loop_lma_x, eo_evecs_field, evecs_4kappasqr_lambdainv, evecs_num, gauge_field_with_phase, mzz, mzzinv );
  } else if ( switch_evecs == 1 ) {
    contract_cvc_loop_eo_lma ( cvc_loop_lma_x, eo_evecs_field2, evecs_4kappasqr_lambdainv, evecs_num, gauge_field_with_phase, mzz, mzzinv );
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * add up fwd and bwd
   ***********************************************/

  double ***cvc_loop_lma_aux = NULL;
  if( ( exitstatus = init_3level_buffer ( &cvc_loop_lma_aux, 2, 4, 2*Vhalf ) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  memset ( cvc_loop_lma_aux[0][0], 0, 16*Vhalf*sizeof(double) );
  for ( int mu = 0; mu < 4; mu++ ) {
    complex_field_pl_eq_complex_field ( cvc_loop_lma_aux[0][mu], cvc_loop_lma_x[0][mu][0], Vhalf );
    complex_field_pl_eq_complex_field ( cvc_loop_lma_aux[0][mu], cvc_loop_lma_x[1][mu][0], Vhalf );
    complex_field_pl_eq_complex_field ( cvc_loop_lma_aux[1][mu], cvc_loop_lma_x[0][mu][1], Vhalf );
    complex_field_pl_eq_complex_field ( cvc_loop_lma_aux[1][mu], cvc_loop_lma_x[1][mu][1], Vhalf );
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * cvc momentum projection of
   * combined fwd + bwd
   ***********************************************/
  if( ( exitstatus = cvc_loop_eo_momentum_projection ( &cvc_loop_tp, cvc_loop_lma_aux, 4, g_sink_momentum_list, g_sink_momentum_number) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * add half-link momentum shift
   ***********************************************/
  if ( ( exitstatus = cvc_loop_eo_momentum_shift ( cvc_loop_tp, g_sink_momentum_list, g_sink_momentum_number) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from cvc_loop_eo_momentum_shift, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * write t-pvec-result momentum to AFF file
   ***********************************************/
  sprintf(aff_tag, "/loop/cvc/nev%.4d", evecs_num );
  if( ( exitstatus = cvc_loop_tp_write_to_aff_file ( cvc_loop_tp, 4, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc ) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * check Ward identity in position space
   ***********************************************************/
  if ( check_position_space_WI ) {

    /********************************************
     * check in position space
     ********************************************/
    if ( switch_evecs == 0 ) {
      exitstatus = cvc_loop_eo_check_wi_position_space_lma ( &cvc_wi, cvc_loop_lma_aux, eo_evecs_field, evecs_4kappasqr_lambdainv, evecs_num,
            gauge_field_with_phase, mzz, mzzinv  );
    } else if (switch_evecs == 1 ) {
      exitstatus = cvc_loop_eo_check_wi_position_space_lma ( &cvc_wi, cvc_loop_lma_aux, eo_evecs_field2, evecs_4kappasqr_lambdainv, evecs_num,
            gauge_field_with_phase, mzz, mzzinv  );
    }

    if ( exitstatus != 0 ) {
      fprintf(stderr, "[loops_hvp] Error from cvc_loop_eo_check_wi_position_space_lma, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    /***********************************************************/
    /***********************************************************/

    /********************************************
     * check on momentum space
     ********************************************/
    if ( ( exitstatus = cvc_loop_eo_check_wi_momentum_space_lma ( cvc_wi, cvc_loop_tp, g_sink_momentum_list, g_sink_momentum_number ) ) != 0 ) {
      fprintf(stderr, "[loops_hvp] Error from cvc_loop_eo_check_wi_momentum_space_lma, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

  }  /* end of if check_position_space_WI */

  /***********************************************************/
  /***********************************************************/

  fini_3level_buffer ( &cvc_loop_lma_aux );
  fini_3level_buffer ( &cvc_loop_tp );
  fini_4level_buffer ( &cvc_loop_lma_x );
  if ( check_position_space_WI ) {
    fini_2level_buffer ( &cvc_wi );
  }


  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * stochastic part
   ***********************************************************/

  double ****cvc_loop_stoch_x = NULL, ***cvc_loop_stoch_p = NULL;

  if( ( exitstatus = init_3level_buffer ( &local_loop_x, 2, 16, 2*Vhalf ) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( ( exitstatus = init_2level_buffer ( &eo_stochastic_source,     2, _GSI( Vhalf ) ) ) !=  0 ) {
    fprintf(stderr, "[loops_hvp] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( ( exitstatus = init_2level_buffer ( &eo_stochastic_propagator, 2, _GSI( Vhalf ) ) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( ( exitstatus = init_4level_buffer ( &cvc_loop_stoch_x, 2, 4, 2, 2*Vhalf ) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from init_4level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if( ( exitstatus = init_3level_buffer ( &cvc_loop_stoch_p, 2, 4, 2*VOLUME ) ) != 0 ) {
    fprintf(stderr, "[loops_hvp] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  /***********************************************************/
  /***********************************************************/
 
  /***********************************************
   * loop on stochastic samples
   ***********************************************/
  for ( int isample = 0; isample < g_nsample; isample++ ) {

    /***********************************************
     * volume source
     ***********************************************/
    exitstatus = prepare_volume_source ( eo_stochastic_source[1], Vhalf);
    if(exitstatus != 0) {
      fprintf(stderr, "[loops_hvp] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(33);
    }

    /***********************************************
     * TEST */
    double norm2;
    spinor_scalar_product_re ( &norm2, eo_stochastic_source[1], eo_stochastic_source[1], Vhalf );
    if ( io_proc == 2 ) fprintf(stdout, "# [loops_hvp] %2d norm2 of volume source = %e\n", isample, norm2 );
    /* END OF TEST 
     ***********************************************/

    /***********************************************************/
    /***********************************************************/

    for ( int t_src = 0; t_src < T_global; t_src++ )
    /* for ( int t_src = 0; t_src < 2; t_src++ ) */
    /* for ( int t_src = T_global; t_src <= T_global; t_src++ ) */
    {

      int have_source = 0;

      if ( t_src == T_global ) {
        have_source = 1;
        memcpy ( eo_stochastic_source[0], eo_stochastic_source[1], sizeof_eo_spinor_field );
      } else {
        int have_source = ( g_proc_coords[0] == t_src / T );

        /***********************************************
         * prepare timeslice source
         ***********************************************/
        memset ( eo_stochastic_source[0], 0, sizeof_eo_spinor_field );
        if ( have_source ) {
          fprintf(stdout, "# [loops_hvp] proc %4d = ( %d, %d, %d, %d) has t_src = %3d \n", g_cart_id,
              g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3], t_src);
          unsigned int shift = _GSI( (t_src%T) * VOL3half   );
          memcpy ( eo_stochastic_source[0]+shift, eo_stochastic_source[1]+shift, sizeof_eo_spinor_field_timeslice );
        }
      }  /* end of if t_src == T_global */

      /***********************************************
       * TEST */
      spinor_scalar_product_re ( &norm2, eo_stochastic_source[0], eo_stochastic_source[0], Vhalf );
      if ( io_proc == 2 ) fprintf(stdout, "# [loops_hvp] %2d %3d norm2 of volume source = %e\n", isample, t_src, norm2 );
      /* END OF TEST
       ***********************************************/

      /***********************************************************/
      /***********************************************************/

      /***********************************************
       * orthogonal projection
       ***********************************************/
      if ( switch_evecs == 0 ) {
        exitstatus = project_propagator_field( eo_stochastic_source[0], eo_stochastic_source[0], 0, eo_evecs_field[0], 1, evecs_num, Vhalf);
      } else if ( switch_evecs == 1 ) {
        exitstatus = project_propagator_field( eo_stochastic_source[0], eo_stochastic_source[0], 0, eo_evecs_field2[0], 1, evecs_num, Vhalf);
      }
      if (exitstatus != 0) {
        fprintf(stderr, "[loops_hvp] Error from project_propagator_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(35);
      }

      /***********************************************************/
      /***********************************************************/

      /***********************************************
       * invert
       ***********************************************/
      memset( eo_spinor_work[0], 0, sizeof_eo_spinor_field );
      memcpy( eo_spinor_work[1], eo_stochastic_source[0], sizeof_eo_spinor_field );

      if ( switch_evecs == 0 ) {
        exitstatus = tmLQCD_invert_eo ( eo_spinor_work[0], eo_spinor_work[1], _OP_ID_UP );
      } else if ( switch_evecs == 1 ) {
        exitstatus = tmLQCD_invert_eo ( eo_spinor_work[0], eo_spinor_work[1], _OP_ID_DN );
      }
      if(exitstatus != 0) {
        fprintf(stderr, "[loops_hvp] Error from tmLQCD_invert_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(19);
      }
      memcpy( eo_stochastic_propagator[0], eo_spinor_work[0], sizeof_eo_spinor_field );

      if( check_propagator_residual ) {
        exitstatus = check_oo_propagator_clover_eo( eo_stochastic_propagator, eo_stochastic_source, &(eo_spinor_work[0]), 
            gauge_field_with_phase, g_mzz_up, g_mzzinv_up, 1 );
        if(exitstatus != 0) {
          fprintf(stderr, "[loops_hvp] Error from check_oo_propagator_clover_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(19);
        }
      }

      /***********************************************************/
      /***********************************************************/

      /***********************************************
       * multiply with 2 kappa
       ***********************************************/
      spinor_field_ti_eq_re ( eo_stochastic_propagator[0], 2.*g_kappa, Vhalf );

      /***********************************************************/
      /***********************************************************/

      /***********************************************
       * local loops
       ***********************************************/
      memset ( local_loop_x[0][0], 0, 2*16*2*Vhalf*sizeof(double) );
      contract_local_loop_eo_stoch ( local_loop_x, eo_stochastic_propagator, eo_stochastic_source, 1, gauge_field_with_phase, mzz, mzzinv );

      /***********************************************
       * momentum space
       ***********************************************/
      if ( ( exitstatus = cvc_loop_eo_momentum_projection ( &local_loop_p, local_loop_x, 16, g_sink_momentum_list, g_sink_momentum_number) ) != 0 ) {
        fprintf ( stderr, "[loop_em] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      /***********************************************
       * write to file
       ***********************************************/
      sprintf(aff_tag, "/loop/local/sample%.4d/tsrc%.2d", isample, t_src );
      if( ( exitstatus = cvc_loop_tp_write_to_aff_file ( local_loop_p, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc ) ) != 0 ) {
        fprintf(stderr, "[loops_hvp] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      /***********************************************************/
      /***********************************************************/

      /***********************************************
       * cvc fwd and bwd loop
       ***********************************************/
      memset ( cvc_loop_stoch_x[0][0][0], 0, 2*4*2*2*Vhalf*sizeof(double) );
      contract_cvc_loop_eo_stoch ( cvc_loop_stoch_x, eo_stochastic_propagator, eo_stochastic_source, 1, gauge_field_with_phase, mzz, mzzinv );

      /***********************************************************/
      /***********************************************************/

      /***********************************************
       * add fwd and bwd
       ***********************************************/
      double ***cvc_loop_stoch_aux = NULL;
      if( ( exitstatus = init_3level_buffer ( &cvc_loop_stoch_aux, 2, 4, 2*Vhalf ) ) != 0 ) {
        fprintf(stderr, "[loops_hvp] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }

      memset ( cvc_loop_stoch_aux[0][0], 0, 16*Vhalf*sizeof(double) );
      for ( int mu = 0; mu < 4; mu++ ) {
        complex_field_pl_eq_complex_field ( cvc_loop_stoch_aux[0][mu], cvc_loop_stoch_x[0][mu][0], Vhalf );
        complex_field_pl_eq_complex_field ( cvc_loop_stoch_aux[0][mu], cvc_loop_stoch_x[1][mu][0], Vhalf );
        complex_field_pl_eq_complex_field ( cvc_loop_stoch_aux[1][mu], cvc_loop_stoch_x[0][mu][1], Vhalf );
        complex_field_pl_eq_complex_field ( cvc_loop_stoch_aux[1][mu], cvc_loop_stoch_x[1][mu][1], Vhalf );
      }

      /***********************************************************/
      /***********************************************************/

      /***********************************************
       * momentum projection, fwd
       ***********************************************/
      exitstatus = cvc_loop_eo_momentum_projection ( &cvc_loop_tp, cvc_loop_stoch_aux, 4, g_sink_momentum_list, g_sink_momentum_number);
      if(exitstatus != 0 ) {
        fprintf(stderr, "[loops_hvp] Error from cvc_loop_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      /***********************************************************/
      /***********************************************************/

      /***********************************************************
       * add half-link momentum shift
       ***********************************************************/
      if ( ( exitstatus = cvc_loop_eo_momentum_shift ( cvc_loop_tp, g_sink_momentum_list, g_sink_momentum_number) ) != 0 ) {
        fprintf(stderr, "[loops_hvp] Error from cvc_loop_eo_momentum_shift, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      /***********************************************************/
      /***********************************************************/

      /***********************************************
       * write to AFF file
       ***********************************************/
      sprintf(aff_tag, "/loop/cvc/sample%.4d/tsrc%.2d", isample, t_src);
      exitstatus = cvc_loop_tp_write_to_aff_file ( cvc_loop_tp, 4, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
      if(exitstatus != 0 ) {
        fprintf(stderr, "[loops_hvp] Error from cvc_loop_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }

      /***********************************************************/
      /***********************************************************/

      /***********************************************
       * check Ward identity in position space
       ***********************************************/
      if ( check_position_space_WI ) {

        if ( ( exitstatus = cvc_loop_eo_check_wi_position_space_stoch ( &cvc_wi, cvc_loop_stoch_aux, eo_stochastic_propagator, eo_stochastic_source, 1, gauge_field_with_phase, mzz, mzzinv ) ) != 0 ) {
          fprintf(stderr, "[loops_hvp] Error from cvc_loop_eo_check_wi_position_space_stoch; status was %d\n", exitstatus );
          EXIT(1);
        }

        /***********************************************************/
        /***********************************************************/

        if ( ( exitstatus = cvc_loop_eo_check_wi_momentum_space_lma ( cvc_wi, cvc_loop_tp, g_sink_momentum_list, g_sink_momentum_number ) ) != 0 ) {
          fprintf(stderr, "[loops_hvp] Error from cvc_loop_eo_check_wi_momentum_space_lma, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }

        /***********************************************************/
        /***********************************************************/

        /***********************************************************
         * clear cvc_wi
         ***********************************************************/
        fini_2level_buffer ( &cvc_wi );

      }  /* end of if check_position_space_WI */


      /***********************************************************/
      /***********************************************************/

      fini_3level_buffer ( &cvc_loop_stoch_aux );

    }  /* end of loop on timeslices */

  }  /* end of loop on stochastic samples */

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * free memory
   ***********************************************/
  fini_3level_buffer ( &local_loop_x );
  fini_3level_buffer ( &local_loop_p );
  fini_4level_buffer ( &cvc_loop_stoch_x );
  fini_3level_buffer ( &cvc_loop_tp );

  fini_2level_buffer ( &eo_stochastic_source );
  fini_2level_buffer ( &eo_stochastic_propagator );

  fini_3level_buffer ( &cvc_loop_stoch_p );

#if 0
#endif  /* of if 0 */

  /***********************************************/
  /***********************************************/

  /***********************************************
   * close AFF file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[loops_hvp] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */


  /***********************************************/
  /***********************************************/

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/

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
  fini_2level_buffer ( &eo_evecs_field2 );

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
    fprintf(stdout, "# [loops_hvp] %s# [loops_hvp] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [loops_hvp] %s# [loops_hvp] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
