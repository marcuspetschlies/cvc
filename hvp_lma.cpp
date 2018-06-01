/****************************************************
 * hvp_lma.c
 *
 * Sun Aug 20 14:42:09 CEST 2017
 *
 * - originally copied from hvp_caa_lma.cpp
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
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "matrix_init.h"
#include "table_init_d.h"
#include "table_init_z.h"
#include "clover.h"
#include "scalar_products.h"
#include "contract_cvc_tensor_eo_lm_factors.h"
#include "gsp_utils.h"

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
  
  const char outfile_prefix[] = "hvp_lma";

  int c;
  int filename_set = 0;
  int exitstatus;
  int sort_eigenvalues = 0;
  char filename[100];
  // double ratime, retime;
  double **mzz[2], **mzzinv[2];
  double *gauge_field_with_phase = NULL;
  unsigned int evecs_block_length = 0;
  unsigned int evecs_num = 0;

#ifdef HAVE_LHPC_AFF
  char aff_tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sh?f:b:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'b':
      evecs_block_length = atoi ( optarg );
      break;
    case 's':
      sort_eigenvalues = 1;
      break;
    case 'n':
      evecs_num = atoi( optarg );
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
  /* fprintf(stdout, "# [hvp_lma] Reading input from file %s\n", filename); */
  read_input_parser(filename);


  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hvp_lma] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hvp_lma] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hvp_lma] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[hvp_lma] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_contraction(2);
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  unsigned int const Vhalf = VOLUME / 2;
  unsigned int const VOL3half =  LX * LY * LZ / 2;
  size_t const sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    // read the gauge field
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [hvp_lma] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    // initialize unit matrices
    if(g_cart_id==0) fprintf(stdout, "\n# [hvp_lma] initializing unit matrices\n");
    exitstatus = unit_gauge_field( g_gauge_field, VOLUME );
  }
#ifdef HAVE_MPI
  xchange_gauge_field ( g_gauge_field );
#endif
  plaquetteria ( g_gauge_field );

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[hvp_lma] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************************
   * allocate eigenvectors
   ***********************************************************/
  double * eo_evecs_block = init_1level_dtable ( evecs_num * _GSI(Vhalf) );
  if( eo_evecs_block == NULL ) {
    fprintf(stderr, "[hvp_lma] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(8);
  }

  /***********************************************************
   * read eigenvectors
   ***********************************************************/
  exitstatus = deflator_read_evecs ( eo_evecs_block, evecs_num, "partfile", filename_prefix , 64 );
  if( exitstatus != 0) {
    fprintf(stderr, "[hvp_lma] Error from deflator_read_evecs status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * allocate eigenvalues
   ***********************************************************/
  //double * const evecs_eval = (double*)malloc(evecs_num*sizeof(double));
  //if( evecs_eval == NULL ) {
  //  fprintf(stderr, "[hvp_lma] Error from malloc %s %d\n", __FILE__, __LINE__);
  //  EXIT(39);
  //}

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set eigenvector field
   ***********************************************************/
  double ** eo_evecs_field = (double**) malloc ( evecs_num * sizeof(double*) );
  if ( eo_evecs_field == NULL ) {
    fprintf ( stderr, "[hvp_lma] Error from malloc %s %d\n", __FILE__, __LINE__ );
    EXIT(42);
  }
  eo_evecs_field[0] = eo_evecs_block;
  for( unsigned int i = 1; i < evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + _GSI(Vhalf);

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set eigenvalues
   ***********************************************************/
  double * evecs_eval = NULL;
  exitstatus = check_eigenpairs ( eo_evecs_field, &evecs_eval, evecs_num, gauge_field_with_phase, mzz, mzzinv );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma] Error from check_eigenpairs, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(17);
  }

  /***********************************************************/
  /***********************************************************/

#if 0
  /***********************************************************
   * sort eigenvalues
   ***********************************************************/

  if ( sort_eigenvalues ) {
    unsigned int * const sort_map = sort_by_dvalue_mapping ( evecs_eval, evecs_num );
    if( sort_map == NULL  ) {
      fprintf(stderr, "[hvp_lma] Error from sort_by_dvalue_mapping %s %d\n", __FILE__, __LINE__);
      EXIT(43);
    }

    exitstatus = sort_dfield_by_map ( eo_evecs_block, evecs_num, sort_map, _GSI(Vhalf) );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[hvp_lma] Error from sort_dfield_by_map, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(44);
    }

    exitstatus = sort_dfield_by_map ( evecs_eval, evecs_num, sort_map, 1 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[hvp_lma] Error from sort_dfield_by_map, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }

    //exitstatus = check_eigenpairs ( eo_evecs_field, &evecs_eval, evecs_num, gauge_field_with_phase, mzz, mzzinv );
    //if( exitstatus != 0 ) {
    //  fprintf(stderr, "[hvp_lma] Error from check_eigenpairs, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    //  EXIT(17);
    //}
    free ( sort_map );
  }
#endif  // of if 0

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * reconstruct additional eigenvector
   ***********************************************************/

  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[hvp_lma] Error from init_rng_stat_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(2);
  }

  double ** eo_spinor_work  = init_2level_dtable ( 2, _GSI((VOLUME+RAND)/2) );
  double ** eo_spinor_field = init_2level_dtable ( 2, _GSI(Vhalf) );

  random_spinor_field ( eo_spinor_work[0], Vhalf );

  exitstatus = project_spinor_field( eo_spinor_field[0], eo_spinor_work[0], 0, eo_evecs_field[0], evecs_num, Vhalf );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[hvp_lma] Error from project_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(2);
  }

#if 0
  exitstatus = project_spinor_field( eo_spinor_field[1], eo_spinor_field[0], 1, eo_evecs_field[0], evecs_num, Vhalf );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[hvp_lma] Error from project_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(2);
  }
  double norm = 0.;
  spinor_scalar_product_re( &norm, eo_spinor_field[1], eo_spinor_field[1], Vhalf );
  fprintf ( stdout, "# [hvp_lma] norm after double projection %25.16e\n", sqrt(norm) );
#endif  // of if 0

  double norm = 0.;
  spinor_scalar_product_re( &norm, eo_spinor_field[0], eo_spinor_field[0], Vhalf );
  norm = 1. / sqrt ( norm );
  spinor_field_ti_eq_re ( eo_spinor_field[0], norm, Vhalf );

  // apply C Cbar
  C_clover_oo ( eo_spinor_work[0],  eo_spinor_field[0], gauge_field_with_phase, eo_spinor_work[1], mzz[1][1], mzzinv[1][0]);
  C_clover_oo ( eo_spinor_field[1], eo_spinor_work[0],  gauge_field_with_phase, eo_spinor_work[1], mzz[0][1], mzzinv[0][0]);

  complex w;
  spinor_scalar_product_co( &w, eo_spinor_field[0], eo_spinor_field[1], Vhalf );

  memcpy ( eo_spinor_work[0], eo_spinor_field[0], sizeof_eo_spinor_field );
  spinor_field_ti_eq_re ( eo_spinor_work[0], w.re, Vhalf );

  // n = | A x - lambda x|^2
  spinor_field_norm_diff ( &norm, eo_spinor_work[0], eo_spinor_field[1], Vhalf );

  // n <- n^1/2 / lambda
  norm  = sqrt ( norm ) / w.re;

  w.re *= 4.*g_kappa*g_kappa;
  w.im *= 4.*g_kappa*g_kappa;

  fprintf ( stdout, "# [hvp_lma] w  %25.16e %25.16e norm diff  %25.16e\n", w.re, w.im, norm );

  // reallocate eo_evecs_block
  double * aux = init_1level_dtable ( (evecs_num+1) * _GSI(Vhalf) );
  if( aux == NULL ) {
    fprintf(stderr, "[hvp_lma] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(8);
  }

  memcpy ( aux, eo_evecs_block, evecs_num*sizeof_eo_spinor_field );
  memcpy ( aux+evecs_num*_GSI(Vhalf), eo_spinor_field[0], sizeof_eo_spinor_field );

  fini_1level_dtable ( &eo_evecs_block );
  eo_evecs_block  = aux;

  // reallocate eo_evecs_field
  free ( eo_evecs_field );
  eo_evecs_field = (double**) malloc ( (evecs_num+1) * sizeof(double*) );
  if ( eo_evecs_field == NULL ) {
    fprintf ( stderr, "[hvp_lma] Error from malloc %s %d\n", __FILE__, __LINE__ );
    EXIT(42);
  }
  eo_evecs_field[0] = eo_evecs_block;
  for( unsigned int i = 1; i <= evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + _GSI(Vhalf);

  // reallocate evecs_eval
  aux = init_1level_dtable ( evecs_num+1 );
  memcpy ( aux, evecs_eval, evecs_num*sizeof(double) );
  aux[evecs_num] = w.re;
  fini_1level_dtable ( &evecs_eval );
  evecs_eval = aux;

  evecs_num++;

  fini_2level_dtable ( &eo_spinor_work );
  fini_2level_dtable ( &eo_spinor_field );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set auxilliary eigenvalue fields
   ***********************************************************/
  double * const evecs_lambdainv           = (double*)malloc(evecs_num*sizeof(double));
  double * const evecs_4kappasqr_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if( evecs_lambdainv == NULL || evecs_4kappasqr_lambdainv == NULL ) {
    fprintf(stderr, "[hvp_lma] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for( unsigned int i = 0; i < evecs_num; i++) {
    evecs_lambdainv[i]           = 2.* g_kappa / evecs_eval[i];
    evecs_4kappasqr_lambdainv[i] = 4.* g_kappa * g_kappa / evecs_eval[i];
  }

  /***********************************************************
   * set eigenvalue and eigenvector fields by
   * eigenvalue if needed
   ***********************************************************/
  if ( sort_eigenvalues ) {
    unsigned int * const sort_map = sort_by_dvalue_mapping ( evecs_eval, evecs_num );
    if( sort_map == NULL  ) {
      fprintf(stderr, "[hvp_lma] Error from sort_by_dvalue_mapping %s %d\n", __FILE__, __LINE__);
      EXIT(43);
    }

    exitstatus = sort_dfield_by_map ( eo_evecs_block, evecs_num, sort_map, _GSI(Vhalf) );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[hvp_lma] Error from sort_dfield_by_map, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(44);
    }

    exitstatus = sort_dfield_by_map ( evecs_eval, evecs_num, sort_map, 1 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[hvp_lma] Error from sort_dfield_by_map, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }

    exitstatus = sort_dfield_by_map ( evecs_lambdainv, evecs_num, sort_map, 1 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[hvp_lma] Error from sort_dfield_by_map, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(46);
    }
    exitstatus = sort_dfield_by_map ( evecs_4kappasqr_lambdainv, evecs_num, sort_map, 1 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[hvp_lma] Error from sort_dfield_by_map, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(47);
    }

    free ( sort_map );
  }   // end of if sort eigenvalues


  /***********************************************************
   * fix eigenvector phase
   ***********************************************************/
  exitstatus = fix_eigenvector_phase ( eo_evecs_field, evecs_num );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma] Error from fix_eigenvector_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(17);
  }

  /***********************************************************
   * check eigenvector equation
   ***********************************************************/
  exitstatus = check_eigenpairs ( eo_evecs_field, &evecs_eval, evecs_num, gauge_field_with_phase, mzz, mzzinv );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma] Error from check_eigenpairs, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(17);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * check evecs_block_length
   ***********************************************************/
  if ( evecs_block_length == 0 ) {
    evecs_block_length = evecs_num;
    if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma] WARNING, reset evecs_block_length to %u\n", evecs_num );
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set io process
   ***********************************************************/
  int const io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hvp_lma] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [hvp_lma] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  sprintf( aff_tag, "/hvp/eval/N%d", evecs_num );
  sprintf( filename, "hvp_lma.%.4d.eval", Nconf );
  exitstatus = gsp_write_eval( evecs_eval, evecs_num, filename, aff_tag);

  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma] Error from gsp_write_eval, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(141);
  }

#ifdef HAVE_LHPC_AFF
  /***********************************************************
   * writer for aff output file
   ***********************************************************/
  struct AffWriter_s *affw = NULL;
  if(io_proc >= 1) {
    sprintf(filename, "%s.%.4d.t%.2d.aff", outfile_prefix, Nconf, g_proc_coords[0]*T );
    fprintf(stdout, "# [hvp_lma] proc%.4d writing data to file %s\n", g_cart_id, filename);
    affw = aff_writer(filename);
    const char * aff_status_str = aff_writer_errstr( affw );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[hvp_lma] Error from aff_writer proc%.4d status was %s %s %d\n", g_cart_id, aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }  // end of if io_proc >= 1
#endif


  /***********************************************************
   * eo lm factors
   ***********************************************************/
  sprintf(aff_tag, "/hvp/lma/N%d/B%d", evecs_num, evecs_block_length );
  if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma] current aff tag = %s\n", aff_tag );

  exitstatus = contract_cvc_tensor_eo_lm_factors ( eo_evecs_field, evecs_num, gauge_field_with_phase, mzz, mzzinv, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc, evecs_block_length );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma] Error from contract_cvc_tensor_eo_lm_factors, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(32);
  }

  /***********************************************************/
  /***********************************************************/
 
  /***********************************************************
   * low-mode contribution to mee part
   ***********************************************************/
  sprintf(aff_tag, "/hvp/lma/N%d/B%d/mee", evecs_num, evecs_block_length );
  if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma] current aff tag = %s\n", aff_tag );

  exitstatus = contract_cvc_tensor_eo_lm_mee ( eo_evecs_field, evecs_num, gauge_field_with_phase,  mzz, mzzinv, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma] Error from contract_cvc_tensor_eo_lm_mee, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(33);
  }

  /***********************************************************/
  /***********************************************************/
 
  /***********************************************************
   * low-mode to mee contact term
   ***********************************************************/
  sprintf(aff_tag, "/hvp/lma/N%d/B%d/mee/ct", evecs_num, evecs_block_length );
  if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma] current aff tag = %s\n", aff_tag );

  exitstatus = contract_cvc_tensor_eo_lm_mee_ct ( eo_evecs_field, evecs_num, gauge_field_with_phase, mzz, mzzinv,
      affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma] Error from contract_cvc_tensor_eo_lm_mee_ct, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(33);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * low-mode contribution to contact term
   ***********************************************************/
  sprintf(aff_tag, "/hvp/lma/N%d/ct", evecs_num );
  if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma] current aff tag = %s\n", aff_tag );

  exitstatus = contract_cvc_tensor_eo_lm_ct ( eo_evecs_field, evecs_num, gauge_field_with_phase, mzz, mzzinv, affw, aff_tag, io_proc);

  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma] Error from contract_cvc_tensor_eo_lm_ct, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(33);
  }

#ifdef HAVE_LHPC_AFF
  if( io_proc >= 1 ) {
    const char * aff_status_str = aff_writer_close ( affw );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[hvp_lma] Error from aff_writer_close proc%.4d status was %s %s %d\n", g_cart_id, aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  // end of if io_proc >= 1
#endif  // of ifdef HAVE_LHPC_AFF

  /***********************************************************/
  /***********************************************************/
 
#if 0

  double ** eo_work = init_2level_dtable ( 3, _GSI( (VOLUME+RAND)/2 ) );
  size_t const sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);
  void *buffer = NULL;


  /***********************************************************
   * check V^+ V with momentum (for contact term)
   ***********************************************************/

  double _Complex **** contr_vv = init_4level_ztable (T, g_sink_momentum_number, evecs_num, evecs_num );
  
  for ( int t = 0; t < T; t++ ) {
    unsigned int const offset = t * VOL3half;
  
    for ( unsigned int i = 0; i < evecs_num; i++ ) {
      for ( unsigned int k = 0; k < evecs_num; k++ ) {

        // contract in position space
        unsigned int ixeo = 0;
        for ( int x1 = 0; x1 < LX; x1++ ) {
        for ( int x2 = 0; x2 < LY; x2++ ) {
        for ( int x3 = 0; x3 < LZ; x3++ ) {

          unsigned int const ix = g_ipt[t][x1][x2][x3];
          if ( g_iseven[ix] ) continue;  // only odd points

          int const r[3] =  { x1, x2, x3 };

          complex w;
          _co_eq_fv_dag_ti_fv ( &w, eo_evecs_field[i]+_GSI(offset+ixeo), eo_evecs_field[k]+_GSI(offset+ixeo) );
         
          for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

            double const p[3] = {
              2.*M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
              2.*M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
              2.*M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global
                                                                  };
            double const phase = ( r[0] + g_proc_coords[1]*LX ) * p[0] + ( r[1] + g_proc_coords[2]*LY ) * p[1] + ( r[2] + g_proc_coords[3]*LZ ) * p[2];
            double _Complex const ephase = cexp ( I*phase );

            contr_vv[t][imom][i][k] += (w.re + I * w.im) * ephase;
          }
          ixeo++;
        }}}
 
      }  // end of loop on timeslices
    }
  }  // end of loop on evecs

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
  buffer = malloc ( T*evecs_num*evecs_num * g_sink_momentum_number * 2 *sizeof (double) );
  memcpy ( buffer,  contr_vv[0][0][0], T*evecs_num*evecs_num*g_sink_momentum_number * 2 * sizeof(double) );
  MPI_Allreduce( buffer, contr_vv[0][0][0], T*evecs_num*evecs_num*2*g_sink_momentum_number, MPI_DOUBLE, MPI_SUM, g_ts_comm );
  free ( buffer );
#  endif
#endif

  if ( io_proc >= 1 ) {

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
    for ( int iproc = 0; iproc < g_tr_nproc; iproc++  ) 
#  else
    for ( int iproc = 0; iproc < g_nproc; iproc++  ) 
#  endif
#endif
    {
#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
      if ( g_tr_id == iproc )
#  else
      if ( g_cart_id == iproc )
#  endif
#endif
      {
        for ( int t = 0; t < T; t++ ) {
          for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
     
            for ( unsigned int ib = 0; ib < ( evecs_num / evecs_block_length ); ib++ ) {
        
              fprintf ( stdout, "/hvp/lma/N%d/B%d/vv/t%.2d/b%.2d/px%.2dpy%.2dpz%.2d\n", evecs_num, evecs_block_length, t+g_proc_coords[0]*T, ib,
                  g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );

              for ( unsigned int i = 0; i < evecs_num; i++ ) {
              for ( unsigned int k = 0; k < evecs_block_length; k++ ) {
                fprintf ( stdout, "    %25.16e %25.16e\n", creal ( contr_vv[t][imom][i][ib*evecs_block_length + k] ), cimag ( contr_vv[t][imom][i][ib*evecs_block_length + k] ) );
              }}
            }  // end of loop on blocks

          }  // end of loop on momenta
        }  // end of loop on timeslices
      }  // end of if g_tr_id == iproc
#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
      MPI_Barrier ( g_tr_comm );
#  else
      MPI_Barrier ( g_cart_id );
#  endif
#endif
    }
  }

  fini_4level_ztable ( &contr_vv );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * check phi_1,4 with momentum (for contact term)
   ***********************************************************/

  double _Complex ***** contr_p = init_5level_ztable (T, 4, g_sink_momentum_number, evecs_num, evecs_num );
  // size_t const sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

  double ** w_field = init_2level_dtable ( 4, _GSI( (VOLUME+RAND)/2 ) );
  double ** v_field = init_2level_dtable ( 2, _GSI( (VOLUME+RAND)/2 ) );
  
  for ( unsigned int k = 0; k < evecs_num; k++ ) {
    memcpy( eo_work[0], eo_evecs_field[k], sizeof_eo_spinor_field );
    C_clover_oo ( w_field[0], eo_work[0], gauge_field_with_phase, eo_work[1], mzz[1][1], mzzinv[1][0] );
    memcpy( eo_work[0], w_field[0], sizeof_eo_spinor_field );
    X_clover_eo ( w_field[1], eo_work[0], gauge_field_with_phase, mzzinv[0][0] );
 
    // double dtmp;
    // spinor_scalar_product_re ( &dtmp, eo_field[1], eo_field[1], Vhalf );
    // if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma] evec eval %4d lambda %25.16e\n", k, dtmp*4*g_kappa*g_kappa  );


    for ( unsigned int i = 0; i < evecs_num; i++ ) {
      memcpy( v_field[0], eo_evecs_field[i], sizeof_eo_spinor_field );
      memcpy( eo_work[0], eo_evecs_field[i], sizeof_eo_spinor_field );

      X_clover_eo ( v_field[1], eo_work[0], gauge_field_with_phase, mzzinv[1][0] );

      for ( int mu = 0; mu < 4; mu++ ) {

        memcpy ( eo_work[0], w_field[0], sizeof_eo_spinor_field );
        memcpy ( eo_work[1], w_field[1], sizeof_eo_spinor_field );
#ifdef HAVE_MPI
        xchange_eo_field( eo_work[0], 1);
        xchange_eo_field( eo_work[1], 0);
#endif
        for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {

          // point on even sublattice
          unsigned int const ixeo = g_lexic2eosub[ix];

          // thus point on odd sublattice
          unsigned int const ixeopm = g_lexic2eosub[ g_iup[ix][mu] ];

          double spinor1[24], spinor2[24];

          if ( g_iseven[ix] ) {
            // contract even points
            _fv_eq_gamma_ti_fv ( spinor1, mu, eo_work[0]+_GSI(ixeopm) );
            _fv_mi_eq_fv ( spinor1, eo_work[0]+_GSI(ixeopm) );
            _fv_eq_gamma_ti_fv ( spinor2, 5, spinor1 );
            _fv_ti_eq_re ( spinor2, 0.5 );
            _fv_eq_cm_ti_fv ( w_field[2]+_GSI(ixeo), gauge_field_with_phase+_GGI(ix,mu), spinor2 );
          } else {
            // contract odd points
            _fv_eq_gamma_ti_fv ( spinor1, mu, eo_work[1]+_GSI(ixeopm) );
            _fv_mi_eq_fv ( spinor1, eo_work[1]+_GSI(ixeopm) );
            _fv_eq_gamma_ti_fv ( spinor2, 5, spinor1 );
            _fv_ti_eq_re ( spinor2, 0.5 );
            _fv_eq_cm_ti_fv ( w_field[3]+_GSI(ixeo), gauge_field_with_phase+_GGI(ix,mu), spinor2 );
          }
        }

        // loop on timeslices
        for ( int t = 0; t < T; t++ ) {
          unsigned int const offset = t * VOL3half;
    
          // contract in position space
          unsigned int ixe = 0, ixo = 0;
          for ( int x1 = 0; x1 < LX; x1++ ) {
          for ( int x2 = 0; x2 < LY; x2++ ) {
          for ( int x3 = 0; x3 < LZ; x3++ ) {

            unsigned int const ix = g_ipt[t][x1][x2][x3];
            int const r[3] =  { x1, x2, x3 };
            complex w = {0., 0.};

            if ( g_iseven[ix] ) {
              _co_eq_fv_dag_ti_fv ( &w, v_field[1]+_GSI(offset+ixe), w_field[2]+_GSI(offset+ixe) );
              ixe++;
            } else {
              _co_eq_fv_dag_ti_fv ( &w, v_field[0]+_GSI(offset+ixo), w_field[3]+_GSI(offset+ixo) );
              ixo++;
            }
         
            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

              double const p[3] = {
                2.*M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
                2.*M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
                2.*M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global
                                                                  };
              double const phase = ( r[0] + g_proc_coords[1]*LX ) * p[0] + ( r[1] + g_proc_coords[2]*LY ) * p[1] + ( r[2] + g_proc_coords[3]*LZ ) * p[2];
              double _Complex const ephase = cexp ( I*phase );

              contr_p[t][mu][imom][i][k] += (w.re + I * w.im) * ephase;
            }

          }}}
 
        }  // end of loop on timeslices
      }  // end of loop on mu
    }  // end of loop on evecs
  }  // end of loop on evecs

#if 0

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
  void *buffer = malloc ( T*4*evecs_num*evecs_num * g_sink_momentum_number * 2 *sizeof (double) );
  memcpy ( buffer,  contr_p[0][0][0][0], T*4*evecs_num*evecs_num*g_sink_momentum_number * 2 * sizeof(double) );
  MPI_Allreduce( buffer, contr_p[0][0][0][0], T*4*evecs_num*evecs_num*2*g_sink_momentum_number, MPI_DOUBLE, MPI_SUM, g_ts_comm );
  free ( buffer );
#  endif
#endif

  if ( io_proc >= 1 ) {

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
    for ( int iproc = 0; iproc < g_tr_nproc; iproc++  ) 
#  else
    for ( int iproc = 0; iproc < g_nproc; iproc++  ) 
#  endif
#endif
    {
#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
      if ( g_tr_id == iproc )
#  else
      if ( g_cart_id == iproc )
#  endif
#endif
      {
        for ( int t = 0; t < T; t++ ) {

          for ( int mu = 0; mu < 4; mu++ ) {

            for ( unsigned int ib = 0; ib < ( evecs_num / evecs_block_length ); ib++ ) {
          
              for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
     
                fprintf ( stdout, "/hvp/lma/N%d/B%d/t%.2d/mu%d/b%.2d/px%.2dpy%.2dpz%.2d\n", evecs_num, evecs_block_length, t+g_proc_coords[0]*T,
                    mu, ib, g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );

                for ( unsigned int i = 0; i < evecs_num; i++ ) {
                for ( unsigned int k = 0; k < evecs_block_length; k++ ) {
                  fprintf ( stdout, "    %25.16e %25.16e\n",
                      creal ( contr_p[t][mu][imom][i][ib*evecs_block_length + k] ), cimag ( contr_p[t][mu][imom][i][ib*evecs_block_length + k] ) );
                }}
              }  // end of loop on momenta
            }  // end of loop on blocks
          }  // end of loop on mu
        }  // end of loop on timeslices
      }  // end of if g_tr_id == iproc
#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
      MPI_Barrier ( g_tr_comm );
#  else
      MPI_Barrier ( g_cart_id );
#  endif
#endif
    }
  }
#endif  // of if 0

#if 0
  fini_5level_ztable ( &contr_p );
  fini_2level_dtable ( &eo_work );
  fini_2level_dtable ( &w_field );
  fini_2level_dtable ( &v_field );
#endif  // of if 0

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * check phi_2,3 with momentum (for contact term)
   ***********************************************************/
#if 0
  double _Complex ***** contr_p = init_5level_ztable (T, 4, g_sink_momentum_number, evecs_num, evecs_num );
  size_t const sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);
  double ** eo_work  = init_2level_dtable ( 2, _GSI( (VOLUME+RAND)/2 ) );
  double ** w_field = init_2level_dtable ( 4, _GSI( (VOLUME+RAND)/2 ) );
  double ** v_field = init_2level_dtable ( 2, _GSI( (VOLUME+RAND)/2 ) );
#endif  // of if 0

  for ( unsigned int k = 0; k < evecs_num; k++ ) {

    memcpy( eo_work[0], eo_evecs_field[k], sizeof_eo_spinor_field );
    C_clover_oo ( w_field[0], eo_work[0], gauge_field_with_phase, eo_work[1], mzz[1][1], mzzinv[1][0] );
    X_clover_eo ( w_field[1], w_field[0], gauge_field_with_phase, mzzinv[0][0] );
 
#ifdef HAVE_MPI
    xchange_eo_field( w_field[0], 1);
    xchange_eo_field( w_field[1], 0);
#endif
    // double dtmp;
    // spinor_scalar_product_re ( &dtmp, w_field[0], w_field[0], Vhalf );
    // if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma] evec eval %4d lambda %25.16e\n", k, dtmp*4*g_kappa*g_kappa  );


    for ( unsigned int i = 0; i < evecs_num; i++ ) {
      memcpy( v_field[0], eo_evecs_field[i], sizeof_eo_spinor_field );

      X_clover_eo ( v_field[1], v_field[0], gauge_field_with_phase, mzzinv[1][0] );
#ifdef HAVE_MPI
      xchange_eo_field( v_field[0], 1);
      xchange_eo_field( v_field[1], 0);
#endif

      for ( int mu = 0; mu < 4; mu++ ) {
        int shift[3] = {0,0,0};
        if ( mu > 0 ) shift[mu-1] = -1;

        for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {

          // point on even sublattice
          unsigned int const ixeo = g_lexic2eosub[ix];

          // thus point on odd sublattice
          unsigned int const ixeomm = g_lexic2eosub[ g_idn[ix][mu] ];

          double spinor1[24], spinor2[24];

          if ( g_iseven[ix] ) {
            // contract even points
            _fv_eq_gamma_ti_fv ( spinor1, mu, w_field[0]+_GSI(ixeomm) );
            _fv_pl_eq_fv ( spinor1, w_field[0]+_GSI(ixeomm) );
            _fv_eq_gamma_ti_fv ( spinor2, 5, spinor1 );
            _fv_ti_eq_re ( spinor2, 0.5 );
            _fv_eq_cm_dag_ti_fv ( w_field[2]+_GSI(ixeo), gauge_field_with_phase+_GGI(g_idn[ix][mu],mu), spinor2 );
          } else {
            // contract odd points
            _fv_eq_gamma_ti_fv ( spinor1, mu, w_field[1]+_GSI(ixeomm) );
            _fv_pl_eq_fv ( spinor1, w_field[1]+_GSI(ixeomm) );
            _fv_eq_gamma_ti_fv ( spinor2, 5, spinor1 );
            _fv_ti_eq_re ( spinor2, 0.5 );
            _fv_eq_cm_dag_ti_fv ( w_field[3]+_GSI(ixeo), gauge_field_with_phase+_GGI(g_idn[ix][mu],mu), spinor2 );
          }
        }

#ifdef HAVE_MPI
        xchange_eo_field( w_field[2], 0);
        xchange_eo_field( w_field[3], 1);
#endif
        // loop on timeslices
        for ( int t = 0; t < T; t++ ) {

          // contract in position space
          for ( int x1 = 0; x1 < LX; x1++ ) {
          for ( int x2 = 0; x2 < LY; x2++ ) {
          for ( int x3 = 0; x3 < LZ; x3++ ) {

            // unsigned int const ix = mu==0 ? g_iup[g_ipt[t][x1][x2][x3]][0] : g_ipt[t][x1][x2][x3];
            unsigned int const ix = mu==0 ? g_iup[g_ipt[t][x1][x2][x3]][0] : g_ipt[t][x1][x2][x3];
            int const r[3] = { x1, x2, x3 };
            complex w = {0., 0.};

            if ( g_iseven[ix] ) {
              _co_eq_fv_dag_ti_fv ( &w, v_field[1]+_GSI(g_lexic2eosub[ix]), w_field[2]+_GSI(g_lexic2eosub[ix]) );
            } else {
              _co_eq_fv_dag_ti_fv ( &w, v_field[0]+_GSI(g_lexic2eosub[ix]), w_field[3]+_GSI(g_lexic2eosub[ix]) );
            }
         
            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

              double const p[3] = {
                2.*M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
                2.*M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
                2.*M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global
                                                                  };
              double const phase = ( r[0] + g_proc_coords[1]*LX + shift[0] ) * p[0] + ( r[1] + g_proc_coords[2]*LY + shift[1] ) * p[1] + ( r[2] + g_proc_coords[3]*LZ + shift[2] ) * p[2];
              double _Complex const ephase = cexp ( I*phase );

              contr_p[t][mu][imom][i][k] += (w.re + I * w.im) * ephase;
            }

          }}}
 
        }  // end of loop on timeslices
      }  // end of loop on mu
    }  // end of loop on evecs
  }  // end of loop on evecs

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
  buffer = malloc ( T*4*evecs_num*evecs_num * g_sink_momentum_number * 2 *sizeof (double) );
  memcpy ( buffer,  contr_p[0][0][0][0], T*4*evecs_num*evecs_num*g_sink_momentum_number * 2 * sizeof(double) );
  MPI_Allreduce( buffer, contr_p[0][0][0][0], T*4*evecs_num*evecs_num*2*g_sink_momentum_number, MPI_DOUBLE, MPI_SUM, g_ts_comm );
  free ( buffer );
#  endif
#endif

  if ( io_proc >= 1 ) {

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
    for ( int iproc = 0; iproc < g_tr_nproc; iproc++  ) 
#  else
    for ( int iproc = 0; iproc < g_nproc; iproc++  ) 
#  endif
#endif
    {
#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
      if ( g_tr_id == iproc )
#  else
      if ( g_cart_id == iproc )
#  endif
#endif
      {
        for ( int t = 0; t < T; t++ ) {

          for ( int mu = 0; mu < 4; mu++ ) {

            for ( unsigned int ib = 0; ib < ( evecs_num / evecs_block_length ); ib++ ) {
          
              for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
     
                fprintf ( stdout, "/hvp/lma/N%d/B%d/t%.2d/mu%d/b%.2d/px%.2dpy%.2dpz%.2d\n", evecs_num, evecs_block_length, t+g_proc_coords[0]*T,
                    mu, ib, g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );

                for ( unsigned int i = 0; i < evecs_num; i++ ) {
                for ( unsigned int k = 0; k < evecs_block_length; k++ ) {
                  fprintf ( stdout, "    %25.16e %25.16e\n",
                      creal ( contr_p[t][mu][imom][i][ib*evecs_block_length + k] ), cimag ( contr_p[t][mu][imom][i][ib*evecs_block_length + k] ) );
                }}
              }  // end of loop on momenta
            }  // end of loop on blocks
          }  // end of loop on mu
        }  // end of loop on timeslices
      }  // end of if g_tr_id == iproc
#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
      MPI_Barrier ( g_tr_comm );
#  else
      MPI_Barrier ( g_cart_id );
#  endif
#endif
    }
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * check W^+ W with momentum (for contact term)
   ***********************************************************/

  double _Complex **** contr_ww = init_4level_ztable (T, g_sink_momentum_number, evecs_num, evecs_num );

  for ( unsigned int i = 0; i < evecs_num; i++ ) {
    memcpy( eo_work[0], eo_evecs_field[i], sizeof_eo_spinor_field );
    C_clover_oo (  eo_evecs_field[i], eo_work[0], gauge_field_with_phase, eo_work[1], mzz[1][1], mzzinv[1][0] );
  }

  for ( int t = 0; t < T; t++ ) {
    unsigned int const offset = t * VOL3half;
  
    for ( unsigned int i = 0; i < evecs_num; i++ ) {
      for ( unsigned int k = 0; k < evecs_num; k++ ) {

        // contract in position space
        unsigned int ixeo = 0;
        for ( int x1 = 0; x1 < LX; x1++ ) {
        for ( int x2 = 0; x2 < LY; x2++ ) {
        for ( int x3 = 0; x3 < LZ; x3++ ) {

          unsigned int const ix = g_ipt[t][x1][x2][x3];
          if ( g_iseven[ix] ) continue;  // only odd points

          int const r[3] =  { x1, x2, x3 };

          complex w;
          _co_eq_fv_dag_ti_fv ( &w, eo_evecs_field[i]+_GSI(offset+ixeo), eo_evecs_field[k]+_GSI(offset+ixeo) );
         
          for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

            double const p[3] = {
              2.*M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
              2.*M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
              2.*M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global
                                                                  };
            double const phase = ( r[0] + g_proc_coords[1]*LX ) * p[0] + ( r[1] + g_proc_coords[2]*LY ) * p[1] + ( r[2] + g_proc_coords[3]*LZ ) * p[2];
            double _Complex const ephase = cexp ( I*phase );

            contr_ww[t][imom][i][k] += (w.re + I * w.im) * ephase;
          }
          ixeo++;
        }}}
 
      }  // end of loop on timeslices
    }
  }  // end of loop on evecs

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
  buffer = malloc ( T*evecs_num*evecs_num * g_sink_momentum_number * 2 *sizeof (double) );
  memcpy ( buffer,  contr_ww[0][0][0], T*evecs_num*evecs_num*g_sink_momentum_number * 2 * sizeof(double) );
  MPI_Allreduce( buffer, contr_ww[0][0][0], T*evecs_num*evecs_num*2*g_sink_momentum_number, MPI_DOUBLE, MPI_SUM, g_ts_comm );
  free ( buffer );
#  endif
#endif

  if ( io_proc >= 1 ) {

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
    for ( int iproc = 0; iproc < g_tr_nproc; iproc++  ) 
#  else
    for ( int iproc = 0; iproc < g_nproc; iproc++  ) 
#  endif
#endif
    {
#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
      if ( g_tr_id == iproc )
#  else
      if ( g_cart_id == iproc )
#  endif
#endif
      {
        for ( int t = 0; t < T; t++ ) {
          for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
     
            for ( unsigned int ib = 0; ib < ( evecs_num / evecs_block_length ); ib++ ) {
        
              fprintf ( stdout, "/hvp/lma/N%d/B%d/ww/t%.2d/b%.2d/px%.2dpy%.2dpz%.2d\n", evecs_num, evecs_block_length, t+g_proc_coords[0]*T, ib,
                  g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );

              for ( unsigned int i = 0; i < evecs_num; i++ ) {
              for ( unsigned int k = 0; k < evecs_block_length; k++ ) {
                fprintf ( stdout, "    %25.16e %25.16e\n", creal ( contr_ww[t][imom][i][ib*evecs_block_length + k] ), cimag ( contr_ww[t][imom][i][ib*evecs_block_length + k] ) );
              }}
            }  // end of loop on blocks

          }  // end of loop on momenta
        }  // end of loop on timeslices
      }  // end of if g_tr_id == iproc
#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ )
      MPI_Barrier ( g_tr_comm );
#  else
      MPI_Barrier ( g_cart_id );
#  endif
#endif
    }
  }

  fini_4level_ztable ( &contr_ww );

  fini_5level_ztable ( &contr_p );
  fini_2level_dtable ( &eo_work );
  fini_2level_dtable ( &w_field );
  fini_2level_dtable ( &v_field );

#endif  // of if 0 around checks by hand

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/

  if ( g_gauge_field             != NULL ) free ( g_gauge_field );
  if ( gauge_field_with_phase    != NULL ) free ( gauge_field_with_phase );
  if ( eo_evecs_field            != NULL ) free ( eo_evecs_field );
  if ( evecs_eval                != NULL ) free ( evecs_eval );
  if ( evecs_lambdainv           != NULL ) free ( evecs_lambdainv );
  if ( evecs_4kappasqr_lambdainv != NULL ) free ( evecs_4kappasqr_lambdainv ); 

  fini_1level_dtable ( &eo_evecs_block );

  /***********************************************************
   * free clover matrix terms
   ***********************************************************/
  fini_clover ();

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_xchange_eo_propagator();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [hvp_lma] %s# [hvp_lma] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [hvp_lma] %s# [hvp_lma] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
