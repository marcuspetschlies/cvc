/****************************************************
 * test_dp.cpp
 *
 * Sa 3. Feb 14:47:45 CET 2018
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
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "matrix_init.h"
#include "clover.h"
#include "distillation_vertices.h"
#include "distillation_utils.h"
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

  const char outfile_prefix[] = "perambulator";

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

  int laph_evecs_num = 0;
  double ***laph_evecs_field = NULL;
  int laph_read_evecs = 0;
  int laph_write_evecs = 0;

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
#endif



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
  /* fprintf(stdout, "# [test_dp] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_dp] calling tmLQCD wrapper init functions\n");

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
  mpi_init_xchange_nvector_3d(6);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_dp] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_dp] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_dp] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_dp] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif


  /******************************************************
   * initialize geometry fields
   ******************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_dp] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [test_dp] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_dp] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_dp] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[test_dp] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[test_dp] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_dp] Error from init_clover, status was %d\n", exitstatus);
    EXIT(1);
  }



#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [test_dp] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [test_dp] proc%.4d is send process\n", g_cart_id);
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
      fprintf(stderr, "[test_dp] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

#ifdef HAVE_LHPC_AFF
  /***********************************************
   *
   * writer for aff output file
   *
   ***********************************************/
  if(io_proc >= 1) {
    sprintf(filename, "%s.%.4d.tblock%.2d.aff", outfile_prefix, Nconf, g_proc_coords[0] );
    fprintf(stdout, "# [test_dp] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_dp] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }  /* end of if io_proc >= 1 */
#endif



  /***********************************************
   * init rng
   ***********************************************/
  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_dp] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  laph_evecs_num = 3;


  exitstatus = init_3level_buffer ( &laph_evecs_field, T, laph_evecs_num,_GVI(VOL3) );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[test_dp] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  if ( laph_read_evecs == 1 ) {
    /***********************************************
     * read eigenvectors from file
     ***********************************************/
    for ( int it = 0; it < T; it++ ) {
      sprintf ( filename, "laph_evecs.%.4d.%.3d", Nconf, it+g_proc_coords[0]*T);
      if ( g_cart_id == 0 ) fprintf( stdout, "# [test_dp] reading evecs from file %s\n", filename );

      exitstatus = read_eigensystem_timeslice ( laph_evecs_field[it], laph_evecs_num, filename);

      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[test_dp] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
        fprintf ( stderr, "[test_dp] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }
    }
  }

  /***********************************************
   * calculate perambulators
   ***********************************************/

  // for ( int t_src = 0; t_src < T_global; t_src++ )
  for ( int t_src = 0; t_src <= 2; t_src++ )
  {

    for ( int s_src = 0; s_src < 4; s_src++ )
    // for ( int s_src = 1; s_src <=1; s_src++ )
    {

      double ****peram = NULL, **spinor_work = NULL;
      exitstatus = init_4level_buffer ( &peram, laph_evecs_num, T, 4, 2*laph_evecs_num );

      exitstatus = init_2level_buffer ( &spinor_work, 2, _GSI(VOLUME+RAND) );

      for ( int l_src = 0; l_src < laph_evecs_num; l_src++ ) {

        exitstatus = distillation_prepare_source (spinor_work[0],  laph_evecs_field[t_src%T], l_src, s_src, t_src );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[test_dp] Error from distillation_prepare_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(1);
        }

#if 0
        /* TEST */
        for ( int it = 0; it < T_global; it++ ) {
          if ( it / T == g_proc_coords[0] ) {
            double norm;
            spinor_field_norm_timeslice ( &norm, spinor_work[0], it%T, VOL3);

            if ( io_proc >= 1 ) fprintf(stdout, "# [test_dp] spinor field t %2d norm %25.16e\n", it, norm );
          }
#ifdef HAVE_MPI
          MPI_Barrier(g_cart_grid );
#endif

        }
        /* END OF TEST */
#endif

#if 0
        /* TEST */
        double ***cv = NULL;
        exitstatus = init_3level_buffer ( &cv, T, 4, _GVI(VOL3) );

        exitstatus = colorvector_field_from_spinor_field ( cv[0][0], spinor_work[0] );
        if ( exitstatus != 0 ) {
          fprintf ( stdout, "[test_dp] Error from colorvector_field_from_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(1);
        }

        for ( int it = 0; it < T_global; it++ ) {

          if ( it / T == g_proc_coords[0] ) {
            for ( int ispin = 0; ispin < 4; ispin++ ) {
              double norm;
              colorvector_field_norm_diff_timeslice ( &norm, cv[it%T][ispin], laph_evecs_field[it%T][l_src], 0, VOL3 );
              if ( io_proc >= 1 ) fprintf(stdout, "# [test_dp] colorvector field t %2d spin %d norm %25.16e\n", it, ispin, norm );
            }
          }
#ifdef HAVE_MPI
          MPI_Barrier(g_cart_grid );
#endif
        }
        fini_3level_buffer ( &cv );
        /* END OF TEST */
#endif

        /***********************************************
         * invert on source
         ***********************************************/
        exitstatus = _TMLQCD_INVERT( spinor_work[1], spinor_work[0], _OP_ID_UP, 0);
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[test_dp] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(1);
        }


        /***********************************************
         * reduce propagator to perambulator
         ***********************************************/
        exitstatus = distillation_reduce_propagator ( peram[l_src], spinor_work[1], laph_evecs_field, laph_evecs_num );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[test_dp] Error from distillation_reduce_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(1);
        }


      }  /* end of loop on eigenvector at source */


      char tag[200];
      sprintf( tag, "/perambulator/up/tsrc%.2d/ssrc%d", t_src, s_src );

      /***********************************************
       * write to AFF file
       ***********************************************/
      exitstatus = distillation_write_perambulator ( peram, laph_evecs_num, affw, tag, io_proc );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[test_dp] Error from distillation_write_perambulator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }

      fini_4level_buffer ( &peram );
      fini_2level_buffer ( &spinor_work );

    }  /* end of loop on source spin */

  }  /* end of loop on source timeslice */

#ifdef HAVE_LHPC_AFF
  if(io_proc >= 1) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_dp] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  /* end of if io_proc >= 1 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  fini_3level_buffer ( &laph_evecs_field );

  free_geometry();

  free_geometry_3d();

  fini_clover();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_xchange_nvector_3d();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_dp] %s# [test_dp] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_dp] %s# [test_dp] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
