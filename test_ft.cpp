/****************************************************
 * test_ft.cpp
 *
 * Wed Sep 27 14:06:50 CEST 2017
 *
 * PURPOSE:
 * - originally copied from cvc_exact2_pspace.cpp
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
#include "ifftw.h"
#include <getopt.h>

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "contractions_io.h"
#include "read_input_parser.h"
#include "fft.h"
#include "ranlxd.h"
#include "matrix_init.h"

#define _SQR(_a) ((_a) * (_a))

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to test ft_4dim\n");
  EXIT(0);
}


int main(int argc, char **argv) {
  
  int c;
  int exitstatus;
  int filename_set = 0;
#ifndef HAVE_MPI
  int dims[4]      = {0,0,0,0};
#endif
  double *conn_x = NULL, *conn_p = NULL, *conn_q = NULL;
  char filename[200], contype[200];
  int read_x  = 0, read_q  = 0;
  int write_x = 0, write_q = 0;
  int read_p  = 0, write_p = 0;
  double ratime, retime;
  double norm = 0.;


#ifndef HAVE_MPI
  fftw_complex *ft_in = NULL;
  fftwnd_plan plan_p;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "xXqQpPh?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'x':
      read_x = 1;
      break;
    case 'X':
      write_x = 1;
      break;
    case 'q':
      read_q = 1;
      break;
    case 'Q':
      write_q = 1;
      break;
    case 'p':
      read_p = 1;
      break;
    case 'P':
      write_p = 1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_ft] using global time stamp %s", ctime(&g_the_time));
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_ft] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  omp_set_num_threads( g_num_threads );
#else
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /*******************************
   * initialize fftw
   *******************************/
#if ( defined HAVE_OPENMP ) && ! ( defined HAVE_MPI )
  if ( (exitstatus = fftw_threads_init() ) != 0 ) {
    fprintf(stderr, "[test_ft] Error from fftw_init_threads; status was %d\n", exitstatus);
    EXIT(120);
  }
#endif

  /*******************************
   * initialize geometry
   *******************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_ft] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

#if 0
  for ( unsigned int i = 2; i <= 33554432; i*=2 ) {
    double _Complex **z = NULL;
    double t1, t2, ti=0, tf = 0;

    for ( int k = 0; k < 10; k++ ) {
      t1  = _GET_TIME;
      init_2level_zbuffer ( &z, 2, i );
      t2 = _GET_TIME;
      ti += t2 - t1;

      fini_2level_zbuffer ( &z );

    }
    fprintf(stdout, "# [test_ft] init %2d %8u %16.7e\n", g_cart_id, i, ti/10);
  }
#endif  /* of if 0 */


#ifndef HAVE_MPI
  dims[0] = T; 
  dims[1] = LX;
  dims[2] = LY;
  dims[3] = LZ;
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);

  ft_in  = ( fftw_complex * ) malloc( VOLUME * sizeof(fftw_complex) );
  if( ft_in == NULL ) {
    fprintf( stderr, "[test_ft] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }
#endif

  /***********************************************************
   * allocate memory for the contractions
   ***********************************************************/
  conn_x = (double*) malloc( 2 * VOLUME * sizeof(double) );
  conn_p = (double*) malloc( 2 * VOLUME * sizeof(double) );
  conn_q = (double*) malloc( 2 * VOLUME * sizeof(double) );
  if( conn_x == NULL || conn_p == NULL || conn_q == NULL ) { 
    fprintf ( stderr, "[test_ft] could not allocate memory for contr. fields\n");
    EXIT(3);
  }

  /*******************************
   * initialize rng
   *******************************/
  if( ( exitstatus = init_rng_stat_file ( g_seed, NULL ) ) != 0 ) {
    fprintf( stderr, "[test_ft] Error from init_rng_stat_file, status was %d\n", exitstatus );
    EXIT(2);
  }

  if ( read_x ) {
    sprintf(filename, "conn_x.lime");
    if ( (exitstatus = read_lime_contraction( conn_x, filename, 1, 0 ) ) != 0 ) {
      fprintf( stderr, "[test_ft] Error from read_lime_contraction, status was %d\n", exitstatus );
      EXIT(2);
    }
  } else {

    if ( ( exitstatus = rangauss ( conn_x, 2*(unsigned int)VOLUME ) ) != 0 ) {
      fprintf( stderr, "[test_ft] Error from rangauss, status was %d\n", exitstatus );
      EXIT(2);
    }

    /* TEST */
    /*
    for ( int x0 = 0; x0 < T;  x0++ ) {
    for ( int x1 = 0; x1 < LX; x1++ ) {
    for ( int x2 = 0; x2 < LY; x2++ ) {
    for ( int x3 = 0; x3 < LZ; x3++ ) {
      unsigned int ix = g_ipt[x0][x1][x2][x3];
      conn_x[2*ix  ] = (double) ( ( (x0 + g_proc_coords[0]*T ) * LX_global + ( x1 + g_proc_coords[1]*LX ) ) * LY_global + ( x2 + g_proc_coords[2]*LY ) ) * LZ_global + ( x3 + g_proc_coords[3]*LZ );
      conn_x[2*ix+1] = 1. /  ( (double) ( ( (x0 + g_proc_coords[0]*T ) * LX_global + ( x1 + g_proc_coords[1]*LX ) ) * LY_global + ( x2 + g_proc_coords[2]*LY ) ) * LZ_global + ( x3 + g_proc_coords[3]*LZ ) + 1. );
    }}}}
    */
    /* END OF TEST */
  
  }  /* end of if read x else */

  if ( write_x ) {
    sprintf(filename, "conn_x.lime");
    sprintf(contype, "complex lattice field x space" );
    if ( ( exitstatus = write_lime_contraction( conn_x, filename, 64, 1, contype, Nconf, 0 ) ) != 0 ) {
      fprintf( stderr, "[test_ft] Error from write_lime_contraction, status was %d\n", exitstatus );
      EXIT(2);
    }
  }

  /* TEST */
  memset ( conn_q, 0, 2*VOLUME*sizeof(double ) );
  complex_field_norm_diff ( &norm, conn_x, conn_q, VOLUME );
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_ft] x norm %25.16e\n", norm);
  }
  /* END OF TEST */
  


#if 0
  /*********************************************
   * FFTW Fourier transformation 
   *********************************************/
  ratime = _GET_TIME;
  if ( read_p ) {
    sprintf(filename, "conn_p.lime");
    if ( (exitstatus = read_lime_contraction( conn_p, filename, 1, 0 ) ) != 0 ) {
      fprintf( stderr, "[test_ft] Error from read_lime_contraction, status was %d\n", exitstatus );
      EXIT(2);
    }
  }
#ifndef HAVE_MPI
  else {
    memcpy( ft_in, conn_x, 2*VOLUME*sizeof(double) );
#ifdef HAVE_OPENMP
    fftwnd_threads_one( g_num_threads, plan_p, ft_in, NULL);
#else
    fftwnd_one( plan_p, ft_in, NULL);
#endif
    memcpy( conn_p, ft_in, 2*VOLUME*sizeof(double) );

  }  /* end of if read p */
#endif

  if ( write_p ) {
    sprintf(filename, "conn_p.lime");
    sprintf(contype, "complex lattice field p space" );
    if ( ( exitstatus = write_lime_contraction( conn_p, filename, 64, 1, contype, Nconf, 0 ) ) != 0 ) {
      fprintf( stderr, "[test_ft] Error from write_lime_contraction, status was %d\n", exitstatus );
      EXIT(2);
    }
  }

  retime = _GET_TIME;
  if ( g_cart_id == 0 ) fprintf(stdout, "# [test_ft] time for x -> p = %e seconds\n", retime-ratime );

#endif  /* of if 0 */

  /*********************************************
   * ft_4dim Fourier transformation 
   *********************************************/
  ratime = _GET_TIME;
  if ( read_q ) {
    sprintf(filename, "conn_q.lime");
    if ( (exitstatus = read_lime_contraction( conn_q, filename, 1, 0 ) ) != 0 ) {
      fprintf( stderr, "[test_ft] Error from read_lime_contraction, status was %d\n", exitstatus );
      EXIT(2);
    }
  } else {
    if ( (exitstatus = ft_4dim ( conn_q, conn_x,  1, 3 ) ) != 0 ) {
      fprintf( stderr, "[test_ft] Error from ft_4dim, status was %d\n", exitstatus );
      EXIT(2);
    }
    /* TEST */
    /*
    if ( (exitstatus = ft_4dim ( conn_p, conn_q, -1, 3 ) ) != 0 ) {
      fprintf( stderr, "[test_ft] Error from ft_4dim, status was %d\n", exitstatus );
      EXIT(2);
    }
    double norm = 1. / ( VOLUME * g_nproc );
    complex_field_ti_eq_re ( conn_p, norm, VOLUME );
    */
    /* END OF TEST */

  }

  if ( write_q ) {
    sprintf(filename, "conn_q.lime");
    sprintf(contype, "complex lattice field p space" );
    if ( ( exitstatus = write_lime_contraction( conn_q, filename, 64, 1, contype, Nconf, 0 ) ) != 0 ) {
      fprintf( stderr, "[test_ft] Error from write_lime_contraction, status was %d\n", exitstatus );
      EXIT(2);
    }
  }

  retime = _GET_TIME;
  if ( g_cart_id == 0 ) fprintf(stdout, "# [test_ft] time for x -> q = %e seconds\n", retime-ratime );

  /*  
  for ( unsigned int ix = 0; ix < VOLUME; ix++ )  norm += _SQR( conn_p[2*ix] - conn_q[2*ix] ) + _SQR( conn_p[2*ix+1] - conn_q[2*ix+1] ) ;
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_ft] p - q norm %25.16e\n", norm);
  }
  */

  complex_field_norm_diff ( &norm, conn_p, conn_q, VOLUME );
  /* complex_field_norm_diff ( &norm, conn_x, conn_p, VOLUME ); */
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_ft] p - q norm %25.16e\n", norm);
    /* fprintf(stdout, "# [test_ft] x - Finv F x norm %25.16e\n", norm); */
  }
#if 0
#endif  /* of if 0 */

#if 0
  /****************************************
   * write position and momentum space
   * fields to stdout
   ****************************************/
  for ( int iproc = 0; iproc < g_nproc; iproc++ ) {
    char *name = "conn_x";
    if ( iproc == g_cart_id ) {
      FILE *ofs = iproc == 0 ? fopen(name, "w") : fopen( name, "a" );
      if ( iproc == 0 ) {
        fprintf(ofs, "%s <- array( dim=c(%d,%d,%d,%d) )\n", name, T_global, LX_global, LY_global, LZ_global );
      }
      for ( int x0 = 0; x0 < T;  x0++ ) {
      for ( int x1 = 0; x1 < LX; x1++ ) {
      for ( int x2 = 0; x2 < LY; x2++ ) {
      for ( int x3 = 0; x3 < LZ; x3++ ) {
        unsigned int ix = g_ipt[x0][x1][x2][x3];
        fprintf(ofs, "%s[%2d, %2d, %2d, %2d] <- %25.16e + %25.16e*1.i\n", name,
            x0 + g_proc_coords[0]*T+1,
            x1 + g_proc_coords[1]*LX+1,
            x2 + g_proc_coords[2]*LY+1,
            x3 + g_proc_coords[3]*LZ+1,
            conn_x[2*ix], conn_x[2*ix+1] );
      }}}}
      fclose ( ofs );
    }
#ifdef HAVE_MPI
    MPI_Barrier ( g_cart_grid );
#endif
  }
#endif  /* of if 0 */

#if 0
#endif  /* of if 0 */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  free_geometry();
#ifndef HAVE_MPI
  fftw_free ( ft_in );
  fftwnd_destroy_plan(plan_p);
#endif

  free(conn_x);
  free(conn_p);
  free(conn_q);

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [test_ft] %s# [test_ft] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [test_ft] %s# [test_ft] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
