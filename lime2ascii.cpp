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
  fprintf(stdout, "          -t                  : type of field [default \"DiracFermion\"]\n");
  fprintf(stdout, "          -p                  : pos in lime file [default 0]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[100];
  char limefile_name[100] = "NA";
  char limefile_suffix[100] = "inverted";
  char limefile_type[100] = "DiracFermion";
  int limefile_pos = 0;
  int tsize = 0, lsize = 0;
  // double ratime, retime;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:l:t:p:T:L:")) != -1) {
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
    case 'T':
      tsize = atoi ( optarg );
      break;
    case 'L':
      lsize = atoi ( optarg );
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

  /***************************************************************************
   * set the default values
   ***************************************************************************/
  if ( filename_set ) {
    /* fprintf(stdout, "# [lime2ascii] Reading input from file %s\n", filename); */
    read_input_parser( filename );
  } else {
    set_default_input_values();

    T  = tsize;
    T_global = tsize;

    L  = lsize;
    LX = lsize;
    LY = lsize;
    LZ = lsize;

  }

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
    /***************************************************************************
     * write as Dirac fermion
     ***************************************************************************/

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

  } else if ( strcmp ( limefile_type, "GaugeField" ) == 0 ) {

    /***************************************************************************
     * write as gauge field
     ***************************************************************************/

    g_gauge_field = init_1level_dtable ( 72*VOLUME );
    exitstatus = read_lime_gauge_field_doubleprec ( limefile_name );

    if ( exitstatus != 0 ) {
      fprintf(stderr, "[lime2ascii] Error from read_lime_gauge_field_doubleprec, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

    sprintf ( filename,"%s.ascii", limefile_name );
    FILE * ofs = fopen ( filename, "w" );
/*
    exitstatus = printf_gauge_field( g_gauge_field, ofs );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[lime2ascii] Error from printf_gauge_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }
*/
    for ( int x0 = 0; x0 < T; x0++ ) {
    for ( int x1 = 0; x1 < LX; x1++ ) {
    for ( int x2 = 0; x2 < LY; x2++ ) {
    for ( int x3 = 0; x3 < LZ; x3++ ) {
      unsigned int const ix = g_ipt[x0][x1][x2][x3];

      for ( int mu = 0; mu < 4; mu++ ) {
        fprintf ( ofs, "# [lime2ascii] x %3d %3d %3d %3d mu %2d\n", x0, x1, x2, x3, mu );

        unsigned int const iy = _GGI(ix,mu);
        for ( int a = 0; a < 3; a++ ) {
        for ( int b = 0; b < 3; b++ ) {
          fprintf ( ofs, "  %3d %3d   %25.16e  %25.16e\n", a, b, g_gauge_field[iy+2*(3*a+b)], g_gauge_field[iy+2*(3*a+b)+1] );
        }}
      }
    }}}}

    fclose ( ofs );

    fini_1level_dtable ( &g_gauge_field );

  } else if ( strcmp ( limefile_type, "DiracPropagator" ) == 0 ) {
    /***************************************************************************
     * write as Dirac propagator
     ***************************************************************************/

    double ** propagator_field = init_2level_dtable ( 12, _GSI(VOLUME) );
    for ( int i = 0; i < 12; i++ ) {
      sprintf ( filename, "%s.%d.%s", limefile_name, i, limefile_suffix );
      exitstatus = read_lime_spinor ( propagator_field[i], filename, limefile_pos);

      if ( exitstatus != 0 ) {
        fprintf(stderr, "[lime2ascii] Error from read_lime_spinor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }
    }  /* of loop on i */

    sprintf ( filename,"%s.%s.ascii", limefile_name, limefile_suffix );
    FILE * ofs = fopen ( filename, "w" );

    fprintf ( ofs, "s <- array(dim=c(4,3,4,3))\n" );
    for ( int x0 = 0; x0 < T; x0++ ) {
    for ( int x1 = 0; x1 < LX; x1++ ) {
    for ( int x2 = 0; x2 < LY; x2++ ) {
    for ( int x3 = 0; x3 < LZ; x3++ ) {
      unsigned int const ix = g_ipt[x0][x1][x2][x3];
      fprintf ( ofs, "# [lime2ascii] x %3d %3d %3d %3d\n", x0, x1, x2, x3 );


      for ( int i = 0; i < 12; i++ ) {
      for ( int k = 0; k < 12; k++ ) {

        fprintf ( ofs, "s[%d,%d,%d,%d] <- (%25.16e) + (%25.16e)*1.i\n", i/3+1, i%3+1, k/3+1, k%3+1, 
            propagator_field[k][_GSI(ix)+2*i], propagator_field[k][_GSI(ix)+2*i+1] );
      }}
#if 0
#endif
    }}}}

    fclose ( ofs );

    fini_2level_dtable ( &propagator_field );
  }

  free_geometry();

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [lime2ascii] %s# [lime2ascii] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [lime2ascii] %s# [lime2ascii] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
