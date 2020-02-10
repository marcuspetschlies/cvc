/****************************************************
 * cpff_xg_contract_lowmem
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

using namespace cvc;

void usage() {
  fprintf(stdout, "Code\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  EXIT(0);
}

#define MAX_SMEARING_LEVELS 40

#define _GLUONIC_OPERATORS_CLOV
#define _GLUONIC_OPERATORS_RECT

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

  char data_tag[400];
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:s:r:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'r':
      stout_rho = atof ( optarg );
      fprintf ( stdout, "# [cpff_xg_contract_lowmem] stout_rho set to %f\n", stout_rho );
      break;
    case 's':
      stout_level_iter[stout_level_num] = atoi ( optarg );
      fprintf ( stdout, "# [cpff_xg_contract_lowmem] stout_level_iter %2d set to %2d\n", stout_level_num, stout_level_iter[stout_level_num] );
      stout_level_num++;
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
  /* fprintf(stdout, "# [cpff_xg_contract_lowmem] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [cpff_xg_contract_lowmem] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[cpff_xg_contract_lowmem] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***************************************************************************
   * initialize own gauge field or get from tmLQCD wrapper
   ***************************************************************************/

  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [cpff_xg_contract_lowmem] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [cpff_xg_contract_lowmem] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[cpff_xg_contract_lowmem] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
    fprintf(stderr, "[cpff_xg_contract_lowmem] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[cpff_xg_contract_lowmem] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [cpff_xg_contract_lowmem] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * set output filename
   ***************************************************************************/
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  if ( io_proc == 2 ) {
    sprintf( output_filename, "%s-lowmem.xg.%d.aff", g_outfile_prefix, Nconf );
    affw = aff_writer (output_filename);
    if( const char * aff_status_str = aff_writer_errstr(affw) ) {
      fprintf(stderr, "[cpff_xg_contract_lowmem] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT( 4 );
    }
  }
#elif (defined HAVE_HDF5 )
  sprintf( output_filename, "%s.xg.%d.h5", g_outfile_prefix, Nconf );
#endif

  /***********************************************
   *
   * smear and calculate operators
   *
   ***********************************************/
  for ( int istout = 0; istout < stout_level_num; istout++ ) {

    char stout_tag[60];
    sprintf ( stout_tag, "/StoutN%u/StoutRho%6.4f", stout_level_iter[istout], stout_rho );


    int const stout_iter = ( istout == 0 ) ? stout_level_iter[0] : stout_level_iter[istout] - stout_level_iter[istout - 1];

    if ( io_proc == 2 ) fprintf ( stdout, "# [cpff_xg_contract_lowmem] stout level %2d iter %2d\n", istout, stout_iter );

    gettimeofday ( &ta, (struct timezone *)NULL );

    double * gbuffer = init_1level_dtable ( 72*VOLUMEPLUSRAND );
    if ( gbuffer == NULL ) {
      fprintf ( stderr, "[cpff_xg_contract_lowmem] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(123);
    }

    exitstatus = stout_smear_inplace ( g_gauge_field, stout_iter , stout_rho, gbuffer );
    if(exitstatus != 0) {
      fprintf(stderr, "[cpff_xg_contract_lowmem] Error from stout smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(47);
    }

    fini_1level_dtable ( &gbuffer );

    gettimeofday ( &tb, (struct timezone *)NULL );
    char timer_tag[100];
    sprintf( timer_tag, "stout-smear-%u", stout_iter );
    show_time ( &ta, &tb, "cpff_xg_contract_lowmem", timer_tag, io_proc==2 );

#ifdef HAVE_MPI
    xchange_gauge_field ( g_gauge_field );
#endif

    /***************************************************************************
     * operator output field
     ***************************************************************************/
    double ** pl = init_2level_dtable ( T_global, 2 );
    if( pl == NULL ) {
      fprintf(stderr, "[cpff_xg_contract_lowmem] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }

    /***************************************************************************
     *
     * Measurement from plaqettes
     *
     ***************************************************************************/
    exitstatus = gluonic_operators ( pl, g_gauge_field );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[cpff_xg_contract_lowmem] Error from gluonic_operators, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(14);
    }

    if ( io_proc == 2 ) {
      sprintf ( data_tag, "%s/plaquette", stout_tag );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
      exitstatus = write_aff_contraction ( pl[0], affw, NULL, data_tag, 2 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
      exitstatus = write_h5_contraction ( pl[0], NULL, output_filename, data_tag, 2 * T_global , "double" );
#else
      exitstatus = 1;
#endif
      if ( exitstatus != 0) {
        fprintf(stderr, "[cpff_xg_contract_lowmem] Error from write_h5_contraction %s %d\n", __FILE__, __LINE__ );
        EXIT(48);
      }
    }  /* end of if io_proc == 2 */

#ifdef _GLUONIC_OPERATORS_CLOV

    /***********************************************************
     * 
     * Measurement with gluonic operators from field strength tensor
     *
     * CLOVER DEFINITION
     *
     ***********************************************************/
    double *** Gp = init_3level_dtable ( VOLUME, 6, 9 );
    if ( Gp == NULL ) {
      fprintf ( stderr, "[cpff_xg_contract_lowmem] Error from  init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(8);
    }

    /***********************************************************
     * plaquette clover field strength tensors
     ***********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    exitstatus = G_plaq ( Gp, g_gauge_field);
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[cpff_xg_contract_lowmem] Error from G_plaq, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(8);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "cpff_xg_contract_lowmem", "G_plaq", io_proc==2 );

    /***********************************************************
     * measurement INCLUDING fst trace
     ***********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    exitstatus = gluonic_operators_eo_from_fst_projected ( pl, Gp, 0 );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[cpff_xg_contract_lowmem] Error from gluonic_operators_eo_from_fst_projected, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(8);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "cpff_xg_contract_lowmem", "gluonic_operators_eo_from_fst_projected", io_proc==2 );

    if ( io_proc == 2 ) {
      sprintf ( data_tag, "%s/clover", stout_tag );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
      exitstatus = write_aff_contraction ( pl[0], affw, NULL, data_tag, 2 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
      exitstatus = write_h5_contraction ( pl[0], NULL, output_filename, data_tag, 2 * T_global , "double" );
#else
      exitstatus = 1;
#endif
      if ( exitstatus != 0) {
        fprintf(stderr, "[cpff_xg_contract_lowmem] Error from write_contraction %s %d\n", __FILE__, __LINE__ );
        EXIT(48);
      }
    }  /* end of if io_proc == 2  */

    /***********************************************************
     * measurement EXCLUDING fst trace
     ***********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    exitstatus = gluonic_operators_eo_from_fst_projected ( pl, Gp, 1 );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[cpff_xg_contract_lowmem] Error from gluonic_operators_eo_from_fst_projected, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(8);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "cpff_xg_contract_lowmem", "gluonic_operators_eo_from_fst_projected-tl", io_proc==2 );

    if ( io_proc == 2 ) {
      sprintf ( data_tag, "%s/clover-tl", stout_tag );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
      exitstatus = write_aff_contraction ( pl[0], affw, NULL, data_tag, 2 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
      exitstatus = write_h5_contraction ( pl[0], NULL, output_filename, data_tag, 2 * T_global , "double" );
#else
      exitstatus = 1;
#endif
      if ( exitstatus != 0) {
        fprintf(stderr, "[cpff_xg_contract_lowmem] Error from write_contraction %s %d\n", __FILE__, __LINE__ );
        EXIT(48);
      }
    }  /* end of if io_proc == 2  */

    fini_3level_dtable ( &Gp );

#endif  /* of _GLUONIC_OPERATORS_CLOV */

#ifdef _GLUONIC_OPERATORS_RECT

    /***********************************************************
     *
     * Measurement with gluonic operators from field strength tensor
     *
     * RECTANGLE DEFINITION
     *
     ***********************************************************/
    double *** Gr = init_3level_dtable ( VOLUME, 6, 9 );
    if ( Gr == NULL ) {
      fprintf ( stderr, "[cpff_xg_contract_lowmem] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(8);
    }

    /***********************************************************
     * rectangle clover field strength tensors
     ***********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    exitstatus = G_rect ( Gr, g_gauge_field);
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[cpff_xg_contract_lowmem] Error from G_rect, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(8);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "cpff_xg_contract_lowmem", "G_rect", io_proc==2 );

    /***********************************************************
     *
     * measurement INCLUDING fst trace
     *
     ***********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    exitstatus = gluonic_operators_eo_from_fst_projected ( pl, Gr, 0 );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[cpff_xg_contract_lowmem] Error from gluonic_operators_eo_from_fst_projected, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(8);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "cpff_xg_contract_lowmem", "gluonic_operators_eo_from_fst_projected-tl", io_proc==2 );

    if ( io_proc == 2 ) {
      sprintf ( data_tag, "%s/rectangle", stout_tag );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
      exitstatus = write_aff_contraction ( pl[0], affw, NULL, data_tag, 2 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
      exitstatus = write_h5_contraction ( pl[0], NULL, output_filename, data_tag, 2 * T_global , "double" );
#else
      exitstatus = 1;
#endif
      if ( exitstatus != 0) {
        fprintf(stderr, "[cpff_xg_contract_lowmem] Error from write_contraction %s %d\n", __FILE__, __LINE__ );
        EXIT(48);
      }
    }  /* end of if io_proc == 2  */

    /***********************************************************
     *
     * measurement EXCLUDING fst trace
     *
     ***********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    exitstatus = gluonic_operators_eo_from_fst_projected ( pl, Gr, 1 );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[cpff_xg_contract_lowmem] Error from gluonic_operators_eo_from_fst_projected, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(8);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "cpff_xg_contract_lowmem", "gluonic_operators_eo_from_fst_projected", io_proc==2 );

    if ( io_proc == 2 ) {
      sprintf ( data_tag, "%s/rectangle-tl", stout_tag );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
      exitstatus = write_aff_contraction ( pl[0], affw, NULL, data_tag, 2 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
      exitstatus = write_h5_contraction ( pl[0], NULL, output_filename, data_tag, 2 * T_global , "double" );
#else
      exitstatus = 1;
#endif
      if ( exitstatus != 0) {
        fprintf(stderr, "[cpff_xg_contract_lowmem] Error from write_contraction %s %d\n", __FILE__, __LINE__ );
        EXIT(48);
      }
    }  /* end of if io_proc == 2  */


    fini_3level_dtable ( &Gr );

#endif  /*  of _GLUONIC_OPERATORS_RECT */

    fini_2level_dtable ( &pl );

  }  /* end of loop on stout smearing steps */

  /***************************************************************************
   * close writer
   ***************************************************************************/
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  if ( io_proc == 2 ) {
    const char * aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[cpff_xg_contract_lowmem] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
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
  show_time ( &start_time, &end_time, "cpff_xg_contract_lowmem", "runtime", io_proc==2 );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [cpff_xg_contract_lowmem] %s# [cpff_xg_contract_lowmem] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [cpff_xg_contract_lowmem] %s# [cpff_xg_contract_lowmem] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
