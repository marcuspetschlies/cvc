/****************************************************
 * test_gm
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
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

#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "read_input_parser.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "ranlxd.h"
#include "gluon_operators.h"
#include "su3.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "cpff";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  struct timeval ta, tb;
  struct timeval start_time, end_time;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  gettimeofday ( &start_time, (struct timezone *)NULL );

  /* set the default values */
  if(filename_set==0) sprintf ( filename, "%s.input", outfile_prefix );
  /* fprintf(stdout, "# [test_gm] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_gm] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[test_gm] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[test_gm] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [test_gm] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  init_lambda_gm();

#if 0
  double _Complex ** A   = init_2level_ztable ( 3, 3);
  ranlxd ( (double*)A[0], 18 );
  double * p = init_1level_dtable ( 9 );
  project_to_generators ( p, (double*)A[0] );

  double _Complex * bs = init_1level_ztable ( 9);
  double _Complex * ba = init_1level_ztable ( 9);
  for ( int i = 0; i < 9; i++ ) {
    double _Complex xa = 0., xs=0., y=0.;

    for ( int l1 = 0; l1 < 3; l1++) {
    for ( int l2 = 0; l2 < 3; l2++) {

      xa += 0.5 * ( A[l1][l2] - conj( A[l2][l1] )  ) * conj( lambda_gm[i][l1][l2] );
      xs += 0.5 * ( A[l1][l2] + conj( A[l2][l1] )  ) * conj( lambda_gm[i][l1][l2] );
      y  += lambda_gm[i][l1][l2] * conj( lambda_gm[i][l1][l2] );
    }}

    /* bs[i] = xs / creal ( y );
    ba[i] = xa / creal ( y ); */
    bs[i] = xs;
    ba[i] = xa;
  }
  for ( int i = 0; i < 9; i++ ) {
    fprintf ( stdout, "# [test_gm] i = %d   p = %25.16e     ba = %25.16e %25.16e     bs = %25.16e %25.16e\n", i, 
        p[i], creal( ba[i]), cimag( ba[i]), creal( bs[i] ), cimag( bs[i] ) );

  }
  fini_1level_ztable ( &bs );
  fini_1level_ztable ( &ba );

#endif

  double A[18], Aa[18], As[18], Aas[18];
  double ps[9], pa[9];

  ranlxd ( A, 18 );

  project_to_generators_hermitean ( ps, A );
  project_to_generators ( pa, A );

  for ( int i = 0; i< 9; i++ ) {
    fprintf ( stdout, "# [test_gm] %d   ps  %25.16e    pa %25.16e\n", i, ps[i], pa[i] );
  }

  restore_from_generators_hermitean ( As, ps );
  restore_from_generators ( Aa, pa );

  for ( int i = 0; i < 18; i++ ) Aas[i] = As[i] + Aa[i];

  for ( int i = 0; i< 9; i++ ) {
    fprintf ( stdout, " %d %d   As %25.16e %25.16e    Aa %25.16e %25.16e    Aas %25.16e %25.16e    A %25.16e %25.16e    dA %e %e\n",
       i/3, i%3, 
       As[2*i],  As[2*i+1],
       Aa[2*i],  Aa[2*i+1],
       Aas[2*i], Aas[2*i+1],
       A[2*i],   A[2*i+1],
      fabs( Aas[2*i]-A[2*i] ),
      fabs( Aas[2*i+1]-A[2*i+1] ));
  }

  fini_lambda_gm();

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "test_gm", "runtime", io_proc==2 );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_gm] %s# [test_gm] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_gm] %s# [test_gm] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
