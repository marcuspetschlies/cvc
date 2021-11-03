/****************************************************
 * stout_smearing_radius
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
#include "contractions_io.h"
#include "su3.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_su3.h"
#include "clover.h"
#include "ranlxd.h"
#include "Q_clover_phi.h"
#include "smearing_techniques.h"
#include "gluon_operators.h"
#include "gauge_io.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  EXIT(0);
}

#define MAX_SMEARING_LEVELS 40

#define _GLUONIC_OPERATORS_PLAQ 0
#define _GLUONIC_OPERATORS_CLOV 1
#define _GLUONIC_OPERATORS_RECT 1

#define _QTOP 0

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "cpff";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  char output_filename[400];
  unsigned int stout_level_iter[MAX_SMEARING_LEVELS];
  unsigned int stout_level_num = 0;
  double stout_rho = 0.;
  struct timeval ta, tb;
  struct timeval start_time, end_time;
  int write_checkpoint = 0;
  int read_checkpoint = 0;

  char data_tag[400];
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:s:r:w:b:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'r':
      stout_rho = atof ( optarg );
      fprintf ( stdout, "# [stout_smearing_radius] stout_rho set to %f\n", stout_rho );
      break;
    case 's':
      stout_level_iter[stout_level_num] = atoi ( optarg );
      fprintf ( stdout, "# [stout_smearing_radius] stout_level_iter %2d set to %2d\n", stout_level_num, stout_level_iter[stout_level_num] );
      stout_level_num++;
      break;
    case 'w':
      write_checkpoint = atoi( optarg );
      fprintf ( stdout, "# [stout_smearing_radius] write checkpoint every %d iter\n", write_checkpoint );
      break;
    case 'b':
      read_checkpoint = atoi( optarg );
      fprintf ( stdout, "# [stout_smearing_radius] read checkpoint %d iter\n", read_checkpoint );
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
  /* fprintf(stdout, "# [stout_smearing_radius] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [stout_smearing_radius] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[stout_smearing_radius] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***************************************************************************
   * initialize own gauge field or get from tmLQCD wrapper
   ***************************************************************************/

  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  if(exitstatus != 0) {
    fprintf ( stderr, "[stout_smearing_radius] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }

#ifdef HAVE_MPI
  xchange_gauge();
#endif

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( g_gauge_field );
  if(exitstatus != 0) {
    fprintf(stderr, "[stout_smearing_radius] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[stout_smearing_radius] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [stout_smearing_radius] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * set output filename
   ***************************************************************************/
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  if ( io_proc == 2 ) {
    sprintf( output_filename, "%s.%d.aff", g_outfile_prefix, Nconf );
    affw = aff_writer (output_filename);
    if( const char * aff_status_str = aff_writer_errstr(affw) ) {
      fprintf(stderr, "[stout_smearing_radius] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT( 4 );
    }
  }
#elif (defined HAVE_HDF5 )
  sprintf( output_filename, "%s.%d.h5", g_outfile_prefix, Nconf );
#endif

  /***************************************************************************
   * set a source at link x=0, mu=0
   ***************************************************************************/
  double heat = 0.1;
  random_gauge_point ( &( g_gauge_field[_GGI(0,0)] ), heat);

  double plaq = 0.;
  plaquette2( &plaq, g_gauge_field );

  /***************************************************************************
   *
   * smear and calculate operators
   *
   ***************************************************************************/
  for ( int istout = 0; istout < stout_level_num; istout++ ) {

    char stout_tag[60];
    sprintf ( stout_tag, "/StoutN%u/StoutRho%6.4f", stout_level_iter[istout], stout_rho );


    /* int const stout_iter = ( istout == 0 ) ? stout_level_iter[0] : stout_level_iter[istout] - stout_level_iter[istout - 1]; */
    int const stout_iter = ( istout == 0 ) ? stout_level_iter[0] - read_checkpoint : stout_level_iter[istout] - stout_level_iter[istout - 1];

    if ( io_proc == 2 ) fprintf ( stdout, "# [stout_smearing_radius] stout level %2d iter %2d\n", istout, stout_iter );

    gettimeofday ( &ta, (struct timezone *)NULL );

    double * gbuffer = init_1level_dtable ( 72*VOLUMEPLUSRAND );
    if ( gbuffer == NULL ) {
      fprintf ( stderr, "[stout_smearing_radius] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(123);
    }

    exitstatus = stout_smear_inplace ( g_gauge_field, stout_iter , stout_rho, gbuffer );
    if(exitstatus != 0) {
      fprintf(stderr, "[stout_smearing_radius] Error from stout smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(47);
    }

    fini_1level_dtable ( &gbuffer );

    gettimeofday ( &tb, (struct timezone *)NULL );
    char timer_tag[100];
    sprintf( timer_tag, "stout-smear-%u", stout_iter );
    show_time ( &ta, &tb, "stout_smearing_radius", timer_tag, io_proc==2 );

    /***************************************************************************
     * optionally write checkpoint configuration
     ***************************************************************************/
    if ( write_checkpoint && ( stout_level_iter[istout] % write_checkpoint == 0 ) && ( stout_level_iter[istout] > 0 ) ) {
      sprintf ( filename, "%s.%.4d.stoutn%d.stoutr%6.4f", gaugefilename_prefix, Nconf,  stout_level_iter[istout] , stout_rho );
      if(g_cart_id==0) fprintf(stdout, "# [stout_smearing_radius] writing gauge field to file %s\n", filename);

      double plaq = 0.;
      plaquette2 ( &plaq, g_gauge_field );

      exitstatus = write_lime_gauge_field ( filename, plaq, Nconf, 64 );
      if(exitstatus != 0) {
        fprintf ( stderr, "[stout_smearing_radius] Error from write_lime_gauge_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(8);
      }

    }


#ifdef HAVE_MPI
    xchange_gauge_field ( g_gauge_field );
#endif

    /***************************************************************************
     * operator output field
     ***************************************************************************/
    double ** pl = init_2level_dtable ( T_global, 2 );
    if( pl == NULL ) {
      fprintf(stderr, "[stout_smearing_radius] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }

    /***************************************************************************/
    /***************************************************************************/

#if _GLUONIC_OPERATORS_CLOV

    /***************************************************************************
     ***************************************************************************
     **
     ** Measurement with gluonic operators from field strength tensor
     **
     ** CLOVER DEFINITION
     **
     ***************************************************************************
     ***************************************************************************/
    double *** Gp = init_3level_dtable ( VOLUME, 6, 9 );
    if ( Gp == NULL ) {
      fprintf ( stderr, "[stout_smearing_radius] Error from  init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(8);
    }

    /***********************************************************
     * (1) plaquette clover field strength tensors
     ***********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    exitstatus = G_plaq ( Gp, g_gauge_field, 1);
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[stout_smearing_radius] Error from G_plaq, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(8);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "stout_smearing_radius", "G_plaq", io_proc==2 );

    /***********************************************************
     * (2) gluon field strength tensor components
     ***********************************************************/

    /***********************************************************
     * measurement of all non-zero tensor components
     * 
     * WITHOUT TRACE of G FST
     ***********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    double ** p_tc = init_2level_dtable ( T_global, 21 );
    if ( p_tc == NULL ) {
      fprintf ( stderr, "[stout_smearing_radius] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(8);
    }

    exitstatus = gluonic_operators_gg_from_fst_projected ( p_tc, Gp, 0 );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[stout_smearing_radius] Error from gluonic_operators_gg_from_fst_projected, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(8);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "stout_smearing_radius", "gluonic_operators_gg_from_fst_projected", io_proc==2 );

    if ( io_proc == 2 ) {
      sprintf ( data_tag, "%s/clover/GG", stout_tag );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
      exitstatus = write_aff_contraction ( p_tc[0], affw, NULL, data_tag, 21 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
      int const dims = 21 * T_global;
      exitstatus = write_h5_contraction ( p_tc[0], NULL, output_filename, data_tag, "double", 1, &dims );
#else
      exitstatus = 1;
#endif
      if ( exitstatus != 0) {
        fprintf(stderr, "[stout_smearing_radius] Error from write_contraction %s %d\n", __FILE__, __LINE__ );
        EXIT(48);
      }
    }  /* end of if io_proc == 2  */

    fini_2level_dtable ( &p_tc );

    /***********************************************************/
    /***********************************************************/

    fini_3level_dtable ( &Gp );

#endif  /* of _GLUONIC_OPERATORS_CLOV */

    /***************************************************************************/
    /***************************************************************************/

    fini_2level_dtable ( &pl );

  }  /* end of loop on stout smearing steps */

  /***************************************************************************
   * close writer
   ***************************************************************************/
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  if ( io_proc == 2 ) {
    const char * aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[stout_smearing_radius] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }
#endif

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
  free(g_gauge_field);

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "stout_smearing_radius", "runtime", io_proc==2 );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [stout_smearing_radius] %s# [stout_smearing_radius] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [stout_smearing_radius] %s# [stout_smearing_radius] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
