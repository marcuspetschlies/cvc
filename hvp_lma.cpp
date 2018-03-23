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


#ifdef DUMMY_SOLVER 
#  define _TMLQCD_INVERT_EO dummy_eo_solver
#else
#  define _TMLQCD_INVERT_EO tmLQCD_invert_eo
#endif

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "hvp_lma";

  int c;
  int filename_set = 0;
  // int check_position_space_WI=0;
  int exitstatus;
  int sort_eigenvalues = 0;
  int check_eigenpairs = 0;
  char filename[100];
  double ratime, retime;
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

  while ((c = getopt(argc, argv, "scwh?f:b:n:")) != -1) {
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
    case 'c':
      check_eigenpairs = 1;
      break;
    case 'n':
      evecs_num = atoi( optarg );
      break;
    //case 'w':
    //  check_position_space_WI = 1;
    //  break;
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

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [hvp_lma] calling tmLQCD wrapper init functions\n");

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

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  unsigned int const Vhalf = VOLUME / 2;

#ifndef HAVE_TMLQCD_LIBWRAPPER
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
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[hvp_lma] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[hvp_lma] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

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

#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************************/
  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[hvp_lma] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[hvp_lma] Error from tmLQCD_get_deflator_params, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(9);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [hvp_lma] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [hvp_lma] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [hvp_lma] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [hvp_lma] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  double * const eo_evecs_block = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block == NULL) {
    fprintf(stderr, "[hvp_lma] Error, eo_evecs_block is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(10);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[hvp_lma] Error, dimension of eigenspace is zero %s %d\n", __FILE__, __LINE__);
    EXIT(11);
  }

  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[hvp_lma] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }
#else
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

#endif  // of ifdef HAVE_TMLQCD_LIBWRAPPER

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * allocate eigenvalues
   ***********************************************************/
  double * const evecs_eval = (double*)malloc(evecs_num*sizeof(double));
  if( evecs_eval == NULL ) {
    fprintf(stderr, "[hvp_lma] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }

#ifdef HAVE_TMLQCD_LIBWRAPPER
  for( unsigned int i = 0; i < evecs_num; i++) {
    evecs_eval[i] = ((double*)(g_tmLQCD_defl.evals))[2*i];
    if( g_cart_id == 0 ) fprintf(stdout, "# [hvp_lma] eval %4d %16.7e\n", i, evecs_eval[i] );
  }
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set eigenvector field
   ***********************************************************/
  double ** const eo_evecs_field = (double**) malloc ( evecs_num * sizeof(double*) );
  if ( eo_evecs_field == NULL ) {
    fprintf ( stderr, "[hvp_lma] Error from malloc %s %d\n", __FILE__, __LINE__ );
    EXIT(42);
  }
  eo_evecs_field[0] = eo_evecs_block;
  for( unsigned int i = 1; i < evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + _GSI(Vhalf);

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * check eigenpairs; calculate them, if they were read in this
   * program
   ***********************************************************/
#ifdef HAVE_TMLQCD_LIBWRAPPER
  if ( check_eigenpairs ) {
#endif
    double ** eo_field = init_2level_dtable ( 2, _GSI(Vhalf));
    double ** eo_work  = init_2level_dtable ( 3, _GSI( (VOLUME+RAND) / 2 ));
    if( eo_field == NULL  || eo_work == NULL ) {
      fprintf(stderr, "[loops_em] Error from init_2level_dtable was %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    for( unsigned int i = 0; i < evecs_num; i++)
    {
      double norm;
      complex w;

      C_clover_oo ( eo_field[0], eo_evecs_field[i],  gauge_field_with_phase, eo_work[2], g_mzz_dn[1], g_mzzinv_dn[0]);
      C_clover_oo ( eo_field[1], eo_field[0], gauge_field_with_phase, eo_work[2], g_mzz_up[1], g_mzzinv_up[0]);

      spinor_scalar_product_re(&norm, eo_evecs_field[i], eo_evecs_field[i], Vhalf);
      spinor_scalar_product_co(&w, eo_field[1], eo_evecs_field[i], Vhalf);

      w.re *= 4.*g_kappa*g_kappa;
      w.im *= 4.*g_kappa*g_kappa;

#ifdef HAVE_TMLQCD_LIBWRAPPER
      if(g_cart_id == 0) {
        fprintf(stdout, "# [hvp_lma] evec %.4d norm = %25.16e w = %25.16e +I %25.16e lambda %25.16e diff = %25.16e\n", i, norm, w.re, w.im, evecs_eval[i], fabs( w.re-evecs_eval[i]));
      }
#else
      evecs_eval[i] = w.re;
      if(g_cart_id == 0) {
        fprintf(stdout, "# [hvp_lma] evec %.4d norm = %25.16e w = %25.16e +I %25.16e\n", i, norm, w.re, w.im );
      }
#endif
      norm = -evecs_eval[i] / ( 4.*g_kappa*g_kappa );
      spinor_field_eq_spinor_field_pl_spinor_field_ti_re( eo_field[0], eo_field[1], eo_evecs_field[i], norm, Vhalf );
      spinor_scalar_product_re(&norm, eo_field[0], eo_field[0], Vhalf);
      if(g_cart_id == 0) {
        fprintf(stdout, "# [hvp_lma] evec %.4d | Ax - lambda x | / | lambda |= %25.16e\n", i, sqrt( norm ) / fabs(evecs_eval[i]) );
      }

    }  // end of loop on evals

    fini_2level_dtable ( &eo_field );
    fini_2level_dtable ( &eo_work );

#ifdef HAVE_TMLQCD_LIBWRAPPER
  }  // end of if check eigenpairs
#endif

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
   * check evecs_block_length
   ***********************************************************/
  if ( evecs_block_length == 0 ) {
    evecs_block_length = evecs_num;
    if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma] WARNING, reset evecs_block_length to %u\n", evecs_num );
  }

  /*************************************************
   * set eigenvalue and eigenvector fields by
   * eigenvalue if needed
   *************************************************/
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

#ifdef HAVE_TMLQCD_LIBWRAPPER
    exitstatus = sort_dfield_by_map ( (double*)g_tmLQCD_defl.evals, evecs_num, sort_map, 2 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[hvp_lma] Error from sort_dfield_by_map, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(44);
    }
#endif  // of ifdef HAVE_TMLQCD_LIBWRAPPER

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

  /***********************************************
   * set io process
   ***********************************************/
  int const io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hvp_lma] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }

#if 0
  /***********************************************/
  /***********************************************/

#ifdef HAVE_LHPC_AFF
  /***********************************************
   * writer for aff output file
   ***********************************************/
  struct AffWriter_s **affw = NULL;
  if(io_proc >= 1) {
    affw = (struct AffWriter_s **) malloc ( T * sizeof(struct AffWriter_s*) );
    if ( affw == NULL ) {
      fprintf(stderr, "[hvp_lma] Error from malloc %s %d\n", __FILE__, __LINE__);
      EXIT(1);
    }
    for ( int i = 0; i < T; i++ ) {
      sprintf(filename, "%s.%.4d.t%.2d.aff", outfile_prefix, Nconf, i + g_proc_coords[0]*T );
      fprintf(stdout, "# [hvp_lma] proc%.4d writing data to file %s\n", g_cart_id, filename);
      affw[i] = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr(affw[i]);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[hvp_lma] Error from aff_writer proc%.4d (%d), status was %s %s %d\n", g_cart_id, i, aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }
  }  // end of if io_proc >= 1
#endif

  /***********************************************
   * set aff tag
   ***********************************************/
  sprintf(aff_tag, "/hvp/lma/N%d/B%d", evecs_num, evecs_block_length );

  /***********************************************
   * contractions
   ***********************************************/

  ratime = _GET_TIME;

  exitstatus = contract_cvc_tensor_eo_lm_factors ( eo_evecs_field, evecs_num, gauge_field_with_phase, mzz, mzzinv, affw, aff_tag,
      g_sink_momentum_list, g_sink_momentum_number, io_proc, evecs_block_length );


  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma] Error from contract_cvc_tensor_eo_lm_factors, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(32);
  }

  retime = _GET_TIME;
  if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma] time for contract_cvc_tensor_eo_lm_factors = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__ );

#ifdef HAVE_LHPC_AFF
  if( io_proc >= 1 ) {
    for ( int i = 0; i < T; i++ ) {
      const char * aff_status_str = aff_writer_close ( affw[i] );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[hvp_lma] Error from aff_writer_close proc%.4d (%d), status was %s %s %d\n", g_cart_id, i, aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }
  }  // end of if io_proc >= 1
#endif  // of ifdef HAVE_LHPC_AFF

#endif  // of if 0


#if 0
#ifdef HAVE_LHPC_AFF
  /***********************************************
   * writer for aff output file
   ***********************************************/
  struct AffWriter_s *affw2 = NULL;
  if( io_proc == 2 ) {
    sprintf(filename, "%s.mee.%.4d.aff", outfile_prefix, Nconf );
    fprintf(stdout, "# [hvp_lma] writing data to file %s\n", filename);
    affw2 = aff_writer( filename );
    const char * aff_status_str = aff_writer_errstr( affw2 );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[hvp_lma] Error from aff_writer proc%.4d, status was %s %s %d\n", g_cart_id, aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }  // end of if io_proc == 2
#endif 

  /***********************************************
   * set aff tag
   ***********************************************/
  sprintf(aff_tag, "/hvp/lma/N%d/mee", evecs_num );

  ratime = _GET_TIME;

  /***********************************************
   * contractions for Mee part
   ***********************************************/
  exitstatus = contract_cvc_tensor_eo_lm_mee ( eo_evecs_field, evecs_num, gauge_field_with_phase, mzz, mzzinv, affw2, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma] Error from contract_cvc_tensor_eo_lm_mee, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(33);
  }

  retime = _GET_TIME;
  if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma] time for contract_cvc_tensor_eo_lm_mee = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__ );

#ifdef HAVE_LHPC_AFF
  if( io_proc == 2 ) {
    const char * aff_status_str = aff_writer_close ( affw2 );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[hvp_lma] Error from aff_writer_close proc%.4d, status was %s %s %d\n", g_cart_id, aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  // end of if io_proc == 2
#endif  // of ifdef HAVE_LHPC_AFF
#endif  // of if 0



#if 0
#ifndef HAVE_MPI
  /****************************************
   * check by hand
   ****************************************/

  double _Complex ***** sp = init_5level_ztable ( evecs_num, 4, 4, T, 4 );
  
  for ( unsigned int iev = 0; iev < evecs_num; iev++ ) {

    size_t const sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof( double );
    
    double ** work_field = init_2level_dtable ( 4, _GSI( (VOLUME+RAND)/2 ) );

    double ** gfv = init_2level_dtable ( 4, _GSI( Vhalf) );
    double ** gbv = init_2level_dtable ( 4, _GSI( Vhalf) );
    double ** gfw = init_2level_dtable ( 4, _GSI( Vhalf) );
    double ** gbw = init_2level_dtable ( 4, _GSI( Vhalf) );

    memcpy ( work_field[0], eo_evecs_field[iev], sizeof_eo_spinor_field );

    // calculate W = Cbar V
    C_clover_oo ( work_field[1], work_field[0], gauge_field_with_phase, work_field[2], mzz[1][1], mzzinv[1][0] );

    for ( int mu = 0; mu < 4; mu++ ) {
      // g5 Gamma^F_mu V
      apply_cvc_vertex_eo ( gfv[mu], work_field[0], mu, 0, gauge_field_with_phase, 0 );
      g5_phi ( gfv[mu], Vhalf );

      // g5 Gamma^B_mu V
      apply_cvc_vertex_eo ( gbv[mu], work_field[0], mu, 1, gauge_field_with_phase, 0 );
      g5_phi ( gbv[mu], Vhalf );

      // Mee^-1 Gamma^F_mu W
      apply_cvc_vertex_eo ( gfw[mu], work_field[1], mu, 0, gauge_field_with_phase, 0 );
      M_clover_zz_inv_matrix ( gfw[mu], gfw[mu], mzzinv[0][0] );

      // Mee^-1 Gamma^B_mu W
      apply_cvc_vertex_eo ( gbw[mu], work_field[1], mu, 1, gauge_field_with_phase, 0 );
      M_clover_zz_inv_matrix ( gbw[mu], gbw[mu], mzzinv[0][0] );

    }

    for ( int mu = 0; mu < 4; mu++ ) {

      for ( int nu = 0; nu < 4; nu++ ) {


        for ( int x0 = 0; x0 < T; x0++ ) {

          for ( int x1 = 0; x1 < LX; x1++ ) {
          for ( int x2 = 0; x2 < LY; x2++ ) {
          for ( int x3 = 0; x3 < LZ; x3++ ) {

            unsigned int const ix = g_ipt[x0][x1][x2][x3];
     
            if ( ! g_iseven[ix] ) continue;

            unsigned int const ixeo = g_lexic2eosub[ix];

            unsigned int const ixplnueo = g_lexic2eosub[g_iup[ix][nu]];

            complex w1, w2;


            _co_eq_fv_dag_ti_fv ( &w1, gbv[mu]+_GSI(ixeo), gfw[nu]+_GSI(ixeo) );
            _co_eq_fv_dag_ti_fv ( &w2, gfv[nu]+_GSI(ixeo), gbw[mu]+_GSI(ixeo) );

            sp[iev][mu][nu][x0][0] += ( w1.re + w2.re ) + ( w1.im + w2.im ) * I;

            _co_eq_fv_dag_ti_fv ( &w1, gfv[mu]+_GSI(ixeo), gfw[nu]+_GSI(ixeo) );
            _co_eq_fv_dag_ti_fv ( &w2, gfv[nu]+_GSI(ixeo), gfw[mu]+_GSI(ixeo) );

            sp[iev][mu][nu][x0][1] += ( w1.re + w2.re ) + ( w1.im + w2.im ) * I;


            _co_eq_fv_dag_ti_fv ( &w1, gbv[nu]+_GSI(ixplnueo), gfw[mu]+_GSI(ixplnueo) );
            _co_eq_fv_dag_ti_fv ( &w2, gfv[mu]+_GSI(ixplnueo), gbw[nu]+_GSI(ixplnueo) );

            sp[iev][mu][nu][x0][2] += ( w1.re + w2.re ) + ( w1.im + w2.im ) * I;

            _co_eq_fv_dag_ti_fv ( &w1, gbv[nu]+_GSI(ixplnueo), gbw[mu]+_GSI(ixplnueo) );
            _co_eq_fv_dag_ti_fv ( &w2, gbv[mu]+_GSI(ixplnueo), gbw[nu]+_GSI(ixplnueo) );

            sp[iev][mu][nu][x0][3] += ( w1.re + w2.re ) + ( w1.im + w2.im ) * I;


          }}}
        }  // end of loop on t

      }  // end of loop on nu
    }  // end of loop on mu

    fini_2level_dtable ( &work_field );
    fini_2level_dtable ( &gfv );
    fini_2level_dtable ( &gbv );
    fini_2level_dtable ( &gfw );
    fini_2level_dtable ( &gbw );
  }  // end of loop on eigenvectors


  for ( int mu = 0; mu < 4; mu++ ) {

    for ( int nu = 0; nu < 4; nu++ ) {

      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

        double _Complex *** sp2 = init_3level_ztable ( T, 3, evecs_num );

        double const pvec[4] = {
          0., 
          2. * M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
          2. * M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
          2. * M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global
        };

        double _Complex const phase[4] = {
          cos ( pvec[mu] ) + I * sin ( pvec[mu] ) ,
          1.,
          cos ( pvec[nu] ) - I * sin ( pvec[nu] ) ,
          cos ( pvec[mu] - pvec[nu] ) + I * sin ( pvec[mu] - pvec[nu] ) };

        for ( int it = 0; it < T; it++ ) {

          for ( int iev = 0; iev < evecs_num; iev++ ) {

            sp2[it][(mu==0)+1][iev]         += phase[0] * sp[iev][mu][nu][it][0];

            sp2[it][1][iev]                 += phase[1] * sp[iev][mu][nu][it][1];

            sp2[it][-(nu==0)+1][iev]        += phase[2] * sp[iev][mu][nu][it][2];


            sp2[it][(mu==0)-(nu==0)+1][iev] += phase[3] * sp[iev][mu][nu][it][3];

          }
        }

        // print
        for ( int it = 0; it < T; it++ ) {
          for ( int dt = 0; dt < 3; dt++ ) {
            fprintf ( stdout, "# [test mee part] /hvp/lma/N%d/mee/mu%d/nu%d/px%.2dpy%.2dpz%.2d/t%.2d/dt%d\n", evecs_num,  mu, nu,
                g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], it, dt-1 );
            for ( int iev = 0; iev < evecs_num; iev++ ) {
              fprintf ( stdout, "  %25.16e %25.16e\n", creal(sp2[it][dt][iev]), cimag(sp2[it][dt][iev]) );
            }
          }
        }

        fini_3level_ztable ( &sp2 );

      }  // end of loop on momenta

    }  // end of loop on nu
    
  }  // end of loop on mu


  fini_5level_ztable ( &sp );
 
#endif
#endif  // of if 0

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

#ifndef HAVE_TMLQCD_LIBWRAPPER
  fini_1level_dtable ( &eo_evecs_block );
#else
  exitstatus = tmLQCD_fini_deflator(_OP_ID_UP);
#endif
  free(eo_evecs_field);

  if ( evecs_eval                != NULL ) free ( evecs_eval );
  if ( evecs_lambdainv           != NULL ) free ( evecs_lambdainv );
  if ( evecs_4kappasqr_lambdainv != NULL ) free ( evecs_4kappasqr_lambdainv ); 

  /****************************************
   * free clover matrix terms
   ****************************************/
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
    fprintf(stdout, "# [hvp_lma] %s# [hvp_lma] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [hvp_lma] %s# [hvp_lma] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
