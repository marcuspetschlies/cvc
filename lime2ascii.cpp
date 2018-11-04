/****************************************************
 * lime2ascii.cpp
 *
 * So 4. Nov 12:11:21 CET 2018
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
#include "table_init_z.h"
#include "table_init_d.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code for lime -> ascii \n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cvc.input]\n");
  fprintf(stdout, "          -l                  : lime filename [default \"NA\"]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[100];
  char limefile_name[100] = "NA";
  char limefile_type[100] = "NA";
  int limefile_pos = 0;
  // double ratime, retime;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:l:t:p:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'l':
      strcpy ( limefile_name, optarg );
      break;
    case 't':
      strcpy ( limefile_type, optarg );
      break;
    case 'p':
      limefile_pos = atoi ( optarg );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

#ifdef HAVE_MPI
  fprintf ( stderr, "[lime2ascii] Not to be used in MPI mode\n");
  exit(1);
#endif

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) sprintf ( filename, "cvc.input");
  /* fprintf(stdout, "# [lime2ascii] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [lime2ascii] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[lime2ascii] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  if ( strcmp ( limefile_type, "DiracFermion" ) == 0 ) {
    double * spinor_field = init_1level_dtable ( _GSI(VOLUME) );
    exitstatus = read_lime_spinor ( spinor_field, limefile_name, limefile_pos);

    if ( exitstatus != 0 ) {
      fprintf(stderr, "[lime2ascii] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    sprintf ( filename,"%s.ascii", limefile_name );
    FILE * ofs = fopen ( filename, "w" );
    exitstatus = printf_spinor_field ( spinor_field, 0, ofs );
    fclose ( ofs );

    fini_1level_dtable ( &spinor_field );
  }

  free_geometry();

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [lime2ascii] %s# [lime2ascii] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [lime2ascii] %s# [lime2ascii] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
