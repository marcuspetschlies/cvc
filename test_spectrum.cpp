/****************************************************
 * test_spectrum.cpp
 *
 * Mi 14. Mär 19:04:42 CET 2018
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
#include "ranlxd.h"

#ifdef __cplusplus
extern "C"
{
#endif

//#  ifdef HAVE_TMLQCD_LIBWRAPPER
//#    include "tmLQCD.h"
//#  endif

#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM

// #include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_geometry_3d.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "gauge_io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_d.h"
#include "clover.h"
#include "distillation_vertices.h"
#include "distillation_utils.h"
#include "laplace_linalg.h"
#include "hyp_smear.h"
#include "laplace.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1


using namespace cvc;

void usage() {
  fprintf(stdout, "# [usage]\n");
  EXIT(0);
}

int main(int argc, char **argv) {

  const char outfile_prefix[] = "perambulator";

  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[100];
  // double ratime, retime;
  double *gauge_field_with_phase = NULL;
#if 0
  double **mzz[2], **mzzinv[2];
#endif
  unsigned int VOL3;

  int laph_evecs_num = 0;
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

  /******************************************************
   * check number of evecs
   ******************************************************/
  if ( laph_evecs_num == 0 ) {
    fprintf ( stderr, "[test_spectrum] Error, number of evecs is NULL\n" );
    EXIT(1);
  }


  /******************************************************
   * set the default values
   ******************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  /* fprintf(stdout, "# [test_spectrum] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /******************************************************
   * initialize MPI parameters for cvc
   ******************************************************/
  mpi_init(argc, argv);
  mpi_init_xchange_nvector_3d(6);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_spectrum] git version = %s\n", g_gitversion);
  }


  /******************************************************
   * set number of openmp threads
   ******************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_spectrum] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_spectrum] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_spectrum] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif


  /******************************************************
   * initialize geometry fields
   ******************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_spectrum] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }
  geometry();

  if(init_geometry_3d() != 0) {
    fprintf(stderr, "[] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }
  geometry_3d();

  VOL3 = LX*LY*LZ;


  /******************************************************
   * allocate the gauge field
   ******************************************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);


  /******************************************************
   * read gauge field
   ******************************************************/
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {

    // read the gauge field
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [test_spectrum] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {

    // initialize unit matrices
    if(g_cart_id==0) fprintf(stdout, "\n# [test_spectrum] initializing unit matrices\n");
    exitstatus = unit_gauge_field( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf(stderr, "[test_spectrum] Error from read / init gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

#if 0
  /***********************************************
   * HYP smear the gauge field
   ***********************************************/
  exitstatus = hyp_smear_3d (g_gauge_field, N_hyp, alpha_hyp, 0, 0);
  if( exitstatus != 0) {
    fprintf(stderr, "[test_spectrum] Error from hyp_smear_3d, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(7);
  }

  /***********************************************
   * write HYP smeared gauge field,
   ***********************************************/
  double plaq = 0.;
  plaquette2( &plaq, g_gauge_field );
  sprintf ( filename, "conf.hyp.%.4d", Nconf );
  exitstatus = write_lime_gauge_field( filename, plaq, Nconf, 64 );
  if( exitstatus != 0) {
    fprintf(stderr, "[test_spectrum] Error from write_lime_gauge_field status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(7);
  }
#endif  // of if 0


  /***********************************************
   * check plaquette of HYP smeared gauge field
   ***********************************************/
  exitstatus = plaquetteria  ( g_gauge_field );
  if( exitstatus != 0) {
    fprintf(stderr, "[test_spectrum] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(8);
  }



#if 0
  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[test_spectrum] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }


  /******************************************************
   * initialize clover, mzz and mzz_inv
   ******************************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_spectrum] Error from init_clover, status was %d\n", exitstatus);
    EXIT(1);
  }
#endif  // of if 0



  /***********************************************************
   * set io proc
   ***********************************************************/
  int const io_proc = get_io_proc ();
  if ( io_proc == -1 ) {
    fprintf ( stderr, "# [test_spectrum] Error from get_io_proc %s %d\n", __FILE__, __LINE__);
    EXIT(1);
  }

  /***********************************************************
   * init rng
   ***********************************************************/
  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_spectrum] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }




  /***********************************************************
   * allocate eigenvector fields
   ***********************************************************/
  double *** laph_evecs_field = init_3level_dtable ( T, laph_evecs_num, _GVI(VOL3) );
  if ( laph_evecs_field == NULL ) {
    fprintf ( stderr, "[test_spectrum] Error from init_3level_dtable, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  double ** laph_evals_field = init_2level_dtable ( T, laph_evecs_num );
  if ( laph_evals_field == NULL ) {
    fprintf ( stderr, "[test_spectrum] Error from init_2level_dtable, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

#if 0
  if ( laph_read_evecs == 1 ) {
    /***********************************************
     * read eigenvectors from file
     ***********************************************/
    for ( int it = 0; it < T; it++ ) 
    {
      sprintf ( filename, "%s/%d/eigenvectors.%.4d.%.3d", filename_prefix, Nconf, Nconf, it+g_proc_coords[0]*T);
      if ( g_cart_id == 0 ) fprintf( stdout, "# [test_spectrum] reading evecs from file %s\n", filename );

      exitstatus = read_eigensystem_timeslice ( laph_evecs_field[it], laph_evecs_num, filename);
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[test_spectrum] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }

      sprintf ( filename, "%s/%d/eigenvalues.%.4d.%.3d", filename_prefix, Nconf, Nconf, it+g_proc_coords[0]*T);
      if ( g_cart_id == 0 ) fprintf( stdout, "# [test_spectrum] reading evals from file %s\n", filename );
      FILE * ifs = fopen ( filename, "rb" );

      if ( fread ( laph_evals_field[it], sizeof(double), laph_evecs_num, ifs ) != laph_evecs_num ) {
        fprintf ( stderr, "[test_spectrum] Error from fread %s %d\n", __FILE__, __LINE__);
        EXIT(2);
      }
      fclose ( ifs );
    }
  } else {

    /***********************************************
     * random eigenvector field
     ***********************************************/
    ranlxd( laph_evecs_field[0][0], T*laph_evecs_num*_GVI(VOL3) );
  }

  if ( laph_write_evecs == 1 ) {

    for ( int it = 0; it < T; it++ ) {
      sprintf ( filename, "%s/%d/laph_evecs.%.4d.%.3d", filename_prefix, Nconf, Nconf, it+g_proc_coords[0]*T);

      exitstatus = write_eigensystem_timeslice ( laph_evecs_field[it], laph_evecs_num, filename);

      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[test_spectrum] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(1);
      }
    }
  }
#endif  // of if 0


#if 0
  /***********************************************
   * check orthogonality
   ***********************************************/
  double *** vv = init_3level_dtable ( T, laph_evecs_num, 2*laph_evecs_num );
  if ( vv == NULL ) {
    fprintf ( stderr, "[test_spectrum] Error from init_3level_dtable %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  for ( int it = 0; it < T; it++ ) {
    exitstatus = distillation_vertex_vdagw ( vv[it], laph_evecs_field[it], laph_evecs_field[it], laph_evecs_num, laph_evecs_num );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_spectrum] Error from distillation_vertex_vdagw, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
  }

  for ( int it = 0; it < T_global; it++ ) {
    if ( it / T == g_proc_coords[0] &&  io_proc >= 1 ) {
      int tloc = it % T;
      for ( int i = 0; i < laph_evecs_num; i++ ) {
      for ( int k = 0; k < laph_evecs_num; k++ ) {
        fprintf( stdout, "p%.4d t %2d n %3d %3d c  %25.16e %25.16e\n", g_cart_id, it, i, k, vv[tloc][i][2*k], vv[tloc][i][2*k+1] );
      }}
    }
    MPI_Barrier ( g_cart_grid );
  }

  fini_3level_dtable ( &vv );
#endif  // end of if 0

#if 0

  /***********************************************
   * check laplacian
   ***********************************************/
  for ( int it = 0; it < T; it++ )
  // for ( int it = 0; it < 2; it++ )
  {
    double ** v = init_2level_dtable ( laph_evecs_num, _GVI(VOL3) );

    for ( int l = 0; l < laph_evecs_num; l++ )
    // for ( int l = 0; l < 2; l++ )
    {
      cv_eq_laplace_cv_3d( v[l], g_gauge_field, laph_evecs_field[it][l], it );
    }

    double ** vv = init_2level_dtable ( laph_evecs_num, 2*laph_evecs_num );
    double ** vw = init_2level_dtable ( laph_evecs_num, 2*laph_evecs_num );

    distillation_vertex_vdagw ( vv, laph_evecs_field[it], laph_evecs_field[it], laph_evecs_num, laph_evecs_num );
    distillation_vertex_vdagw ( vw, laph_evecs_field[it], v, laph_evecs_num, laph_evecs_num );

    if ( io_proc >= 1 ) {
      for ( int k = 0; k < laph_evecs_num; k++ ) {
      for ( int l = 0; l < laph_evecs_num; l++ ) {
        if ( l == k ) {
          fprintf ( stdout, "# [] t %2d l %3d %3d vv %25.16e %25.16e vw %25.16e %25.16e  lambda %25.16e\n", it+g_proc_coords[0]*T, l, k, vv[k][2*l], vv[k][2*l+1], vw[k][2*l], vw[k][2*l+1], laph_evals_field[it][l] );
        } else {
          fprintf ( stdout, "# [] t %2d l %3d %3d vv %25.16e %25.16e vw %25.16e %25.16e  lambda %25.16e\n", it+g_proc_coords[0]*T, l, k, vv[k][2*l], vv[k][2*l+1], vw[k][2*l], vw[k][2*l+1], 0. );
        }
      }}  // end of loop on eigenvectors
    }

    fini_2level_dtable ( &vv );
    fini_2level_dtable ( &vw );


    fini_2level_dtable ( &v );
  }  // end of loop on timeslices
#endif  // of if 0

#if 0
#ifdef HAVE_LHPC_AFF
  /***********************************************************
   * writer for aff output file
   ***********************************************************/
  if( io_proc >= 1 ) {
    sprintf(filename, "%s.%.4d.tblock%.2d.aff", "D", Nconf, g_proc_coords[0] );
    fprintf(stdout, "# [test_spectrum] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_spectrum] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }  // end of if io_proc >= 1
#endif
#endif // of if 0

#if 0
  /***********************************************************
   * single displacement
   ***********************************************************/
  for ( int ts = 0; ts < T; ts++ ) {
    
    int pzero[3] = {0,0,0};
    char aff_tag[] = "/displacement";

    exitstatus = distillation_vertex_displacement ( laph_evecs_field[ts], laph_evecs_num, 1, &pzero, "vdv", aff_tag, io_proc, g_gauge_field, ts );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_spectrum] Error from distillation_vertex_displacement, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(4);
    }

  }
#endif // of if 0


  if ( g_cart_id == 0 ) {

    int const nvec = 66;
    int ts = 0;
    char op_str[] = "d_\<y";
    int p[3] = {0,0,0};

    double ** vdv_operators = init_2level_dtable ( nvec, 2 * nvec );

    sprintf ( filename, "%s/%d/%s.%.4d.p_%d%d%d.%s.t_%.3d", filename_prefix2, Nconf, "operators", Nconf, p[0], p[1], p[2], op_str, ts );
    fprintf ( stdout, "# [test_spectrum] using filename %s\n", filename );

    FILE *ifs = fopen ( filename, "rb" );
    if ( ifs == NULL ) {
      fprintf ( stderr, "[test_spectrum] Error from fopen %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(4);
    }

    fread ( vdv_operators[0], sizeof(double), 2*nvec*nvec, ifs );

    fclose ( ifs );

    fprintf ( stdout, "# [test_spectrum] op %s t %3d\n", op_str, ts );
    for ( int k = 0; k < nvec; k++ ) {
    for ( int l = 0; l < nvec; l++ ) {
      fprintf ( stdout, "  %3d %3d  %25.16e %25.16e\n", k, l, vdv_operators[k][2*l], vdv_operators[k][2*l+1] );
    }}

    fini_2level_dtable ( &vdv_operators );
  }

#if 0
#ifdef HAVE_LHPC_AFF
  if(io_proc >= 1) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_spectrum] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  /* end of if io_proc >= 1 */
#endif  /* of ifdef HAVE_LHPC_AFF */
#endif // of if 0


  /****************************************
   * free the allocated memory, finalize
   ****************************************/

  if ( g_gauge_field          != NULL ) free ( g_gauge_field );
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );

  fini_3level_dtable ( &laph_evecs_field );
  fini_2level_dtable ( &laph_evals_field );

  free_geometry();

  free_geometry_3d();

#ifdef HAVE_MPI
  mpi_fini_xchange_nvector_3d();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_spectrum] %s# [test_spectrum] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_spectrum] %s# [test_spectrum] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
