/****************************************************
 * test_geom3.cpp
 *
 * Di 6. Feb 15:33:51 CET 2018
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
#include "ranlxd.h"

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
#include "cvc_geometry_3d.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "matrix_init.h"
#include "clover.h"
#include "distillation_vertices.h"
#include "distillation_utils.h"
#include "distillation_vertices.h"
#include "laplace_linalg.h"

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

int dummy_solver (double * const propagator, double * const source, const int op_id, int d) {
  memcpy(propagator, source, _GSI(VOLUME)*sizeof(double) );
  return(0);
}


#ifdef DUMMY_SOLVER 
#  define _TMLQCD_INVERT dummy_solver
#else
#  define _TMLQCD_INVERT tmLQCD_invert
#endif

int main(int argc, char **argv) {

  const char outfile_prefix[] = "distvert";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int evecs_num = 0;
  int check_propagator_residual = 0;
  double *evecs_eval = NULL;
  char filename[100];
  double ratime, retime;
  double *gauge_field_with_phase = NULL;
  double **mzz[2], **mzzinv[2];
  unsigned int VOL3;
  char tag[200];

  int laph_evecs_num = 0;
  double ***laph_evecs_field = NULL;
  int laph_read_evecs = 0;
  int laph_write_evecs = 0;


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "rwh?f:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'r':
      laph_read_evecs = 1;
      break;
    case 'w':
      laph_write_evecs = 1;
      break;
    case 'n':
      laph_evecs_num = atoi( optarg);
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
  /* fprintf(stdout, "# [test_geom3] Reading input from file %s\n", filename); */
  read_input_parser(filename);


  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_nvector_3d(6);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_geom3] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_geom3] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_geom3] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_geom3] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif


  /******************************************************
   * initialize geometry fields
   ******************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_geom3] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }
  geometry();

  if(init_geometry_3d() != 0) {
    fprintf(stderr, "[] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }
  geometry_3d();

  VOL3 = LX*LY*LZ;

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [test_geom3] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_geom3] initializing unit matrices\n");
    for( unsigned int ix=0; ix < VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[test_geom3] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_geom3] Error from init_clover, status was %d\n", exitstatus);
    EXIT(1);
  }



#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [test_geom3] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [test_geom3] proc%.4d is send process\n", g_cart_id);
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
      fprintf(stderr, "[test_geom3] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************
   * init rng
   ***********************************************/
  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_geom3] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }


  laph_evecs_num = 3;

  exitstatus = init_3level_buffer ( &laph_evecs_field, T, laph_evecs_num,_GVI(VOL3) );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[test_geom3] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( laph_read_evecs == 1 ) {
    /***********************************************
     * read eigenvectors from file
     ***********************************************/
    for ( int it = 0; it < T; it++ ) {
      sprintf ( filename, "laph_evecs.%.4d.%.3d", Nconf, it+g_proc_coords[0]*T);
      if ( g_cart_id == 0 ) fprintf( stdout, "# [test_geom3] reading evecs from file %s\n", filename );

      exitstatus = read_eigensystem_timeslice ( laph_evecs_field[it], laph_evecs_num, filename);

      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[test_geom3] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }
    }
  } else {

    /***********************************************
     * random eigenvector field
     ***********************************************/
    ranlxd( laph_evecs_field[0][0], T*laph_evecs_num*_GVI(VOL3) );
  }

  if ( laph_write_evecs == 1 ) {

    for ( int it = 0; it < T; it++ ) {
      sprintf ( filename, "laph_evecs.%.4d.%.3d", Nconf, it+g_proc_coords[0]*T);

      exitstatus = write_eigensystem_timeslice ( laph_evecs_field[it], laph_evecs_num, filename);

      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[test_geom3] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }
    }
  }
#if 0
#endif



  /***********************************************
   * 
   ***********************************************/
  // for ( int x0 = 0; x0 < T; x0++ )
  for ( int x0 = 0; x0 < 1; x0++ )
  {

    sprintf ( tag, "t%.2d", x0 );
    exitstatus = distillation_vertex_displacement ( laph_evecs_field[x0], laph_evecs_num, g_sink_momentum_number, g_sink_momentum_list, (char*)outfile_prefix, tag, io_proc, gauge_field_with_phase, x0 );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_geom3] Error from distillation_vertex_displacement, status was %d\n", exitstatus );
      EXIT(1);
    }

  }
#if 0
#endif


#if 0
  double **work = NULL;
  exitstatus = init_2level_buffer ( &work, 4, _GVI(VOL3+RAND3) );

  for ( int x0 = 0; x0 < T; x0++ ) {
    for ( int k = 1; k < 4; k++ ) {
      for ( int fbwd = 0; fbwd < 2; fbwd++ ) {
        apply_displacement_colorvector ( work[1], work[0], k, fbwd, gauge_field_with_phase, x0 );
        for ( int x1 = 0; x1 < LX; x1++ ) {
        for ( int x2 = 0; x2 < LY; x2++ ) {
        for ( int x3 = 0; x3 < LZ; x3++ ) {
          unsigned int ix = g_ipt[0][x1][x2][x3];
          unsigned int ixpk = g_iup[ix][k];
          unsigned int ixmk = g_idn[ix][k];
          double v1[_GVI(1)];
          double v2[_GVI(1)];
          double norm;

          if ( fbwd == 0 ) {
            _cv_eq_cv ( v1, work[0]+_GVI(ixpk) );
            _cv_eq_cm_ti_cv ( v2, gauge_field_with_phase+_GGI(x0*VOL3+ix,k), v1 );
          } else {
            _cv_eq_cv ( v1, work[0]+_GVI(ixmk) );
            _cv_eq_cm_dag_ti_cv ( v2, gauge_field_with_phase+_GGI( g_idn[x0*VOL3+ix][k],k), v1 );
          }

          _cv_mi_eq_cv ( v2, work[1]+_GVI(ix) );

          _re_eq_cv_dag_ti_cv ( norm, v2, v2 );

          fprintf ( stdout, "# [test_geom3] x %3d %3d %3d  norm %25.16e\n", x1, x2, x3, norm );
        }}}
      }
    }
  }
  fini_2level_buffer ( &work );
#endif

#if 0

  /****************************************
   * test gauge covariance
   ****************************************/
  double **work = NULL;
  exitstatus = init_2level_buffer ( &work, 4, _GVI(VOL3+RAND3) );

  double *gt = NULL;
  init_1level_buffer ( &gt, 18*VOLUME );
  for ( unsigned int ix = 0; ix < VOLUME; ix++ ) random_cm( gt+18*ix, 1. );


  memcpy ( g_gauge_field, gauge_field_with_phase, 72*VOLUME*sizeof(double) );
#ifdef HAVE_MPI
  xchange_gauge_field( g_gauge_field );
#endif

  apply_gt_gauge( gt, g_gauge_field );

  double plaq;
  plaquette2( &plaq, g_gauge_field );
  if ( g_cart_id == 0 ) fprintf( stdout, "# [test_geom3] plaquette of gt gauge field = %25.16e\n", plaq );


  for ( int x0 = 0; x0 < T; x0++ ) {

    memset( work[0], 0, 4*_GVI(VOL3) );

    ranlxd ( work[0], _GVI(VOL3));

    for ( unsigned int ix = 0; ix < VOL3; ix++ ) {
      _cv_eq_cm_ti_cv ( work[2]+_GVI(ix), gt+18*(x0*VOL3+ix) , work[0]+_GVI(ix));
    }

    for ( int k = 1; k < 4; k++ ) {

      for ( int fbwd = 0; fbwd < 2; fbwd++ ) {

        apply_displacement_colorvector ( work[1], work[0], k, fbwd, gauge_field_with_phase, x0 );

        for ( unsigned int ix = 0; ix < VOL3; ix++ ) {
          double v1[_GVI(1)];
          _cv_eq_cm_ti_cv ( v1, gt+18*(x0*VOL3+ix) , work[1]+_GVI(ix));
          _cv_eq_cv ( work[1]+_GVI(ix), v1);
        }

        apply_displacement_colorvector ( work[3], work[2], k, fbwd, g_gauge_field, x0 );

        double norm = 0.;
        colorvector_field_norm_diff_timeslice ( &norm, work[3], work[1], 0, VOL3);
        if ( io_proc >= 1 ) {
          fprintf ( stdout, "# [test_geom3] x0 %3d k %d fbwd %d norm diff %16.7e\n", x0+g_proc_coords[0]*T, k, fbwd, norm );
        }


      }
    }

  }
    
  fini_1level_buffer ( &gt );
  fini_2level_buffer ( &work );

#endif




  /****************************************
   * free the allocated memory, finalize
   ****************************************/

  free(g_gauge_field);

  free( gauge_field_with_phase );

  // fini_3level_buffer ( &laph_evecs_field );

  free_geometry();

  free_geometry_3d();

  fini_clover();


#ifdef HAVE_MPI
  mpi_fini_xchange_nvector_3d();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_geom3] %s# [test_geom3] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_geom3] %s# [test_geom3] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
