/****************************************************
 * ll_lma.c
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
#include "read_input_parser.h"
#include "Q_clover_phi.h"
#include "clover.h"
#include "gsp_utils.h"
#include "gsp.h"
#include "matrix_init.h"

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
  
  const char outfile_prefix[] = "ll_lma";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int evecs_num = 0;
  int do_fix_eigenvector_phase = 0;
  unsigned int Vhalf;
  double *eo_evecs_block=NULL;
  double **eo_evecs_field=NULL;
  double *evecs_eval = NULL, *evecs_lambdainv=NULL, *evecs_4kappasqr_lambdainv = NULL;
  char filename[100];
  /* double ratime, retime; */
  double **mzz[2], **mzzinv[2];
  double *gauge_field_with_phase = NULL;

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
# else
  void *affw = NULL;
#endif
  char aff_tag[400];

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

  while ((c = getopt(argc, argv, "h?f:p:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'p':
      do_fix_eigenvector_phase = 1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [ll_lma] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***********************************************************/
  /***********************************************************/

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [ll_lma] calling tmLQCD wrapper init functions\n");

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

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /***********************************************************/
  /***********************************************************/

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [ll_lma] %s# [ll_lma] start of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [ll_lma] %s# [ll_lma] start of run\n", ctime(&g_the_time));
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set number of openmp threads
   ***********************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [ll_lma] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [ll_lma] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[ll_lma] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * intialize geometry
   ***********************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[ll_lma] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************/
  /***********************************************************/

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***********************************************************/
  /***********************************************************/

  Vhalf    = VOLUME / 2;

  /***********************************************************/
  /***********************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [ll_lma] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [ll_lma] initializing unit matrices\n");
    unit_gauge_field( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[ll_lma] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[ll_lma] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************/
  /***********************************************************/

#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[ll_lma] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[ll_lma] Error from tmLQCD_get_deflator_params, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(9);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [ll_lma] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [ll_lma] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [ll_lma] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [ll_lma] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block == NULL) {
    fprintf(stderr, "[ll_lma] Error, eo_evecs_block is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(10);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[ll_lma] Error, dimension of eigenspace is zero %s %d\n", __FILE__, __LINE__);
    EXIT(11);
  }

  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[ll_lma] Error from tmLQCD_init_deflator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }

  evecs_eval                = (double*)malloc(evecs_num*sizeof(double));
  evecs_lambdainv           = (double*)malloc(evecs_num*sizeof(double));
  evecs_4kappasqr_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if(    evecs_eval                == NULL 
      || evecs_lambdainv           == NULL 
      || evecs_4kappasqr_lambdainv == NULL 
    ) {
    fprintf(stderr, "[ll_lma] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for( int i = 0; i < evecs_num; i++) {
    evecs_eval[i]                = ((double*)(g_tmLQCD_defl.evals))[2*i];
    evecs_lambdainv[i]           = 2.* g_kappa / evecs_eval[i];
    evecs_4kappasqr_lambdainv[i] = 4.* g_kappa * g_kappa / evecs_eval[i];
    if( g_cart_id == 0 ) fprintf(stdout, "# [ll_lma] eval %4d %16.7e\n", i, evecs_eval[i] );
  }

#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   *
   ***********************************************************/
  sprintf ( aff_tag, "%s.%.4d", outfile_prefix, Nconf );
  exitstatus = gsp_write_eval( evecs_eval, evecs_num, aff_tag);
  if(exitstatus != 0) {
    fprintf(stderr, "[ll_lma] Error from gsp_write_eval, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }



  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * allocate memory for the eigenvector fields
   ***********************************************************/
  eo_evecs_field = (double**)calloc(evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block;
  for( int i = 1; i < evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + _GSI(Vhalf);

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[ll_lma] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[ll_lma] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  /***********************************************************/
  /***********************************************************/

#ifdef HAVE_MPI
  /***********************************************************
   * set io process
   ***********************************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [ll_lma] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [ll_lma] proc%.4d is send process\n", g_cart_id);
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
      fprintf(stderr, "[ll_lma] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * fix eigenvector phase
   ***********************************************************/
  if ( do_fix_eigenvector_phase == 1 ) {
    if ( io_proc == 2 ) fprintf(stdout, "# [ll_lma] fixing eigenvector phase\n");
    exitstatus = fix_eigenvector_phase ( eo_evecs_field, evecs_num );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[ll_lma] Error from fix_eigenvector_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
  }

  /***********************************************************/
  /***********************************************************/

#if 0
  /***********************************************************
   * TEST
   ***********************************************************/
  unsigned int VOL3half = LX*LY*LZ/2;
  size_t sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);
  FILE*ofs = NULL;
  if ( io_proc >= 1 ) {
    sprintf( filename, "test.t%.2d", g_proc_coords[0] );
    ofs = fopen( filename, "w");
  }
  
  double **eo_X_field = NULL, **eo_Y_field = NULL;
  exitstatus = init_2level_buffer ( &eo_X_field, evecs_num, _GSI(Vhalf) );
  exitstatus = init_2level_buffer ( &eo_Y_field, evecs_num, _GSI(Vhalf) );

  for ( int i = 0; i < evecs_num; i++ ) {

    /* X= V and Y = V */
    memcpy ( eo_X_field[i], eo_evecs_field[i], sizeof_eo_spinor_field );
    memcpy ( eo_Y_field[i], eo_evecs_field[i], sizeof_eo_spinor_field );
#if 0
#endif

#if 0
    /* X = W, Y = V */
    double **eo_spinor_work = NULL;
    init_2level_buffer ( &eo_spinor_work, 2, _GSI((VOLUME+RAND)/2) );
    memcpy ( eo_spinor_work[0], eo_evecs_field[i], sizeof_eo_spinor_field );
    C_clover_oo ( eo_X_field[i], eo_spinor_work[0], gauge_field_with_phase, eo_spinor_work[1], mzz[1][1], mzzinv[1][0] );
    fini_2level_buffer ( &eo_spinor_work );
    memcpy ( eo_Y_field[i], eo_evecs_field[i], sizeof_eo_spinor_field );
#endif

#if 0
    /* X = W, Y = W */
    double **eo_spinor_work = NULL;
    init_2level_buffer ( &eo_spinor_work, 2, _GSI((VOLUME+RAND)/2) );
    memcpy ( eo_spinor_work[0], eo_evecs_field[i], sizeof_eo_spinor_field );
    C_clover_oo ( eo_X_field[i], eo_spinor_work[0], gauge_field_with_phase, eo_spinor_work[1], mzz[1][1], mzzinv[1][0] );
    fini_2level_buffer ( &eo_spinor_work );
    memcpy ( eo_Y_field[i], eo_X_field[i], sizeof_eo_spinor_field );
#endif

#if 0
    /* X = XV, Y = XV */
    double **eo_spinor_work = NULL;
    init_2level_buffer ( &eo_spinor_work, 1, _GSI((VOLUME+RAND)/2) );
    memcpy ( eo_spinor_work[0], eo_evecs_field[i], sizeof_eo_spinor_field );
    X_clover_eo ( eo_X_field[i], eo_spinor_work[0], gauge_field_with_phase, mzzinv[1][0] );
    fini_2level_buffer ( &eo_spinor_work );
    memcpy ( eo_Y_field[i], eo_X_field[i], sizeof_eo_spinor_field );
#endif

#if 0
    /* X = XW, Y = XV */
    double **eo_spinor_work = NULL;
    init_2level_buffer ( &eo_spinor_work, 3, _GSI((VOLUME+RAND)/2) );
    memcpy ( eo_spinor_work[0], eo_evecs_field[i], sizeof_eo_spinor_field );
    C_clover_oo ( eo_spinor_work[1], eo_spinor_work[0], gauge_field_with_phase, eo_spinor_work[2], mzz[1][1], mzzinv[1][0] );
    X_clover_eo ( eo_X_field[i], eo_spinor_work[1], gauge_field_with_phase, mzzinv[0][0] );
    memcpy ( eo_spinor_work[0], eo_evecs_field[i], sizeof_eo_spinor_field );
    X_clover_eo ( eo_Y_field[i], eo_spinor_work[0], gauge_field_with_phase, mzzinv[1][0] );
    fini_2level_buffer ( &eo_spinor_work );
#endif

#if 0
    /* X = XW, Y = XW */
    double **eo_spinor_work = NULL;
    init_2level_buffer ( &eo_spinor_work, 3, _GSI((VOLUME+RAND)/2) );
    memcpy ( eo_spinor_work[0], eo_evecs_field[i], sizeof_eo_spinor_field );
    C_clover_oo ( eo_spinor_work[1], eo_spinor_work[0], gauge_field_with_phase, eo_spinor_work[2], mzz[1][1], mzzinv[1][0] );
    X_clover_eo ( eo_X_field[i], eo_spinor_work[1], gauge_field_with_phase, mzzinv[0][0] );
    memcpy ( eo_Y_field[i], eo_X_field[i], sizeof_eo_spinor_field );
#endif
  }

  char aff_prefix[] = "v-v";

  for ( int k = 0; k < evecs_num; k++ ) {
    double **eo_spinor_field = NULL;
    init_2level_buffer ( &eo_spinor_field, 4, _GSI(Vhalf) );

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      for ( int x0 = 0; x0 < T; x0++ ) {
        for ( int x1 = 0; x1 < LX; x1++ ) { 
          double px = 2*M_PI * ( g_proc_coords[1]*LX + x1 ) / (double)LX_global * g_sink_momentum_list[imom][0];
        for ( int x2 = 0; x2 < LY; x2++ ) { 
          double py = 2*M_PI * ( g_proc_coords[2]*LY + x2 ) / (double)LY_global * g_sink_momentum_list[imom][1];
        for ( int x3 = 0; x3 < LZ; x3++ ) { 
          double pz = 2*M_PI * ( g_proc_coords[3]*LZ + x3 ) / (double)LZ_global * g_sink_momentum_list[imom][2];
          unsigned int ix = g_ipt[x0][x1][x2][x3];
          if ( g_iseven[ix] ) continue;
          unsigned int ixeo = g_lexic2eosub[ix];
          double phase = px + py + pz;
          complex w = {cos(phase), sin(phase)};
          _fv_eq_fv_ti_co ( eo_spinor_field[0]+_GSI(ixeo), eo_Y_field[k]+_GSI(ixeo), &w );
        }}}
      }
      for ( int ig = 0; ig < g_source_gamma_id_number; ig++ ) {
        for ( unsigned int ix = 0; ix < Vhalf; ix++ ) {
          _fv_eq_gamma_ti_fv ( eo_spinor_field[1]+_GSI(ix), g_source_gamma_id_list[ig], eo_spinor_field[0]+_GSI(ix) );
        }
        for ( int l = 0; l < evecs_num; l++ ) {
          for ( int x0 = 0; x0 < T; x0++ )
          {
            complex w = {0.,0.};
            for ( unsigned int ix = 0; ix < VOL3half; ix++ ) {
              unsigned int iix = x0 * _GSI(VOL3half) + _GSI(ix);
              _co_pl_eq_fv_dag_ti_fv( &w, eo_X_field[l]+iix , eo_spinor_field[1]+iix );
            }
#ifdef HAVE_MPI
            double dtmp[2] = {w.re, w.im};
            MPI_Allreduce( dtmp, &w, 2, MPI_DOUBLE, MPI_SUM, g_ts_comm );
#endif
            if ( io_proc >= 1 )  {
              fprintf ( ofs, "# /ll/lma/N%d/%s/t%.2d/px%.2dpy%.2dpz%.2d/g%.2d\n", evecs_num, aff_prefix, x0+g_proc_coords[0]*T, 
                  g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], g_source_gamma_id_list[ig] );
              fprintf( ofs, "%3d %3d  %25.16e %25.16e\n", l, k, w.re, w.im);
            }
          }
        }
      }
    }
    fini_2level_buffer ( &eo_spinor_field );
  }
  if ( io_proc >= 1 ) {
    sprintf( filename, "test.t%.2d", g_proc_coords[0] );
    fclose ( ofs );
  }
  fini_2level_buffer ( &eo_X_field );
  fini_2level_buffer ( &eo_Y_field );

  /***********************************************************
   * END OF TEST
   ***********************************************************/
#endif  /* of if 0 on TEST */

  /***********************************************************/
  /***********************************************************/
  
  sprintf( aff_tag, "/ll/lma/N%d", evecs_num );
  sprintf( filename_prefix, "%s.%.4d.N%d", outfile_prefix, Nconf, evecs_num );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * contract and write to disk
   ***********************************************************/
  exitstatus = gsp_calculate_v_dag_gamma_p_w_block ( eo_evecs_field, evecs_num, g_sink_momentum_number, g_sink_momentum_list, g_source_gamma_id_number, g_source_gamma_id_list, filename_prefix, aff_tag, io_proc, \
           gauge_field_with_phase, mzz, mzzinv );

  if ( exitstatus != 0 ) {
    fprintf(stderr, "[ll_lma] Error from gsp_calculate_v_dag_gamma_p_w_block, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(32);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

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
    fprintf(stdout, "# [ll_lma] %s# [ll_lma] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [ll_lma] %s# [ll_lma] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
