/****************************************************
 * cpff_xg_contract
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

using namespace cvc;

void usage() {
  fprintf(stdout, "Code\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  EXIT(0);
}

#define MAX_SMEARING_LEVELS 12

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
      fprintf ( stdout, "# [cpff_xg_contract] stout_rho set to %f\n", stout_rho );
      break;
    case 's':
      stout_level_iter[stout_level_num] = atoi ( optarg );
      fprintf ( stdout, "# [cpff_xg_contract] stout_level_iter %2d set to %2d\n", stout_level_num, stout_level_iter[stout_level_num] );
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
  /* fprintf(stdout, "# [cpff_xg_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [cpff_xg_contract] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
  /* exitstatus = tmLQCD_invert_init(argc, argv, 1); */
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

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [cpff_xg_contract] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[cpff_xg_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***************************************************************************
   * initialize own gauge field or get from tmLQCD wrapper
   ***************************************************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [cpff_xg_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [cpff_xg_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[cpff_xg_contract] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#ifdef HAVE_MPI
  xchange_gauge();
#endif

#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[cpff_xg_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[cpff_xg_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( g_gauge_field );
  if(exitstatus != 0) {
    fprintf(stderr, "[cpff_xg_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[cpff_xg_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [cpff_xg_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * set output filename
   ***************************************************************************/
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  sprintf( output_filename, "%s.xg.%d.aff", g_outfile_prefix, Nconf );
  affw = aff_writer (output_filename);
  if( const char * aff_status_str = aff_writer_errstr(affw) ) {
    fprintf(stderr, "[cpff_xg_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    EXIT( 4 );
  }

#elif (defined HAVE_HDF5 )
  sprintf( output_filename, "%s.xg.%d.h5", g_outfile_prefix, Nconf );
#endif

  /***************************************************************************
   * operator output field
   ***************************************************************************/
  double ** pl = NULL;
  //if ( io_proc > 0  ) {
    pl = init_2level_dtable ( T_global, 2 );
 // } else {
  //  pl = init_2level_dtable ( 1, 1 );
  //}

  /***************************************************************************
   *
   * Measurement on original, UNSMEARED gauge field
   *
   ***************************************************************************/

  /***************************************************************************
   * gluonic operators from plaquettes
   ***************************************************************************/
  exitstatus = gluonic_operators ( pl, g_gauge_field );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[cpff_xg_contract] Error from gluonic_operators, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(14);
  }

  if ( io_proc == 2 && g_verbose > 2 ) {
    fprintf( stdout, "\n\n# [cpff_xg_contract] Original config\n" );
    for ( int it = 0; it < T_global; it++ ) {
      fprintf ( stdout, "x %3d %25.16e %25.16e\n", it, pl[it][0], pl[it][1] );
    }
  }

  sprintf ( data_tag, "/StoutN%u/StoutRho%6.4f/plaquette", 0, 0. );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  exitstatus = write_aff_contraction ( pl[0], affw, NULL, data_tag, 2 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
  exitstatus = write_h5_contraction ( pl[0], NULL, filename, data_tag, 2 * T_global );
#else
  exitstatus = 1;
#endif
  if ( exitstatus != 0) {
    fprintf(stderr, "[cpff_xg_contract] Error from write_h5_contraction %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

#if 0

  /***************************************************************************
   * gluonic operators from field strength tensor
   ***************************************************************************/
  double *** Gp = init_3level_dtable ( VOLUME, 6, 18 );
  double *** Gr = init_3level_dtable ( VOLUME, 6, 18 );
  if ( Gr == NULL || Gp == NULL ) {
    fprintf ( stderr, "[cpff_xg_contract] Error from G_plaq_rect, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }

  exitstatus = G_plaq_rect ( Gp, Gr, g_gauge_field);
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[cpff_xg_contract] Error from G_plaq_rect, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }

  exitstatus = gluonic_operators_eo_from_fst ( pl, Gp );
  if ( exitstatus != 0) {
    fprintf(stderr, "[cpff_xg_contract] Error from gluonic_operators_eo_from_fst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(48);
  }

  sprintf ( data_tag, "/StoutN%u/StoutRho%6.4f/clover/plaquette", 0, 0. );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  exitstatus = write_aff_contraction ( pl[0], affw, NULL, data_tag, 2 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
  exitstatus = write_h5_contraction ( pl[0], NULL, filename, data_tag, 2 * T_global );
#else
  exitstatus = 1;
#endif
  if ( exitstatus != 0) {
    fprintf(stderr, "[cpff_xg_contract] Error from write_h5_contraction %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }


  exitstatus = gluonic_operators_eo_from_fst ( pl, Gr );
  if ( exitstatus != 0) {
    fprintf(stderr, "[cpff_xg_contract] Error from gluonic_operators_eo_from_fst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(48);
  }
  sprintf ( data_tag, "/StoutN%u/StoutRho%6.4f/clover/rectangle", 0, 0. );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  exitstatus = write_aff_contraction ( pl[0], affw, NULL, data_tag, 2 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
  exitstatus = write_h5_contraction ( pl[0], NULL, filename, data_tag, 2 * T_global );
#else
  exitstatus = 1;
#endif
  if ( exitstatus != 0) {
    fprintf(stderr, "[cpff_xg_contract] Error from write_h5_contraction %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  fini_3level_dtable ( &Gp );
  fini_3level_dtable ( &Gr );
#endif
  fini_2level_dtable ( &pl );

  /***********************************************
   * smear and calculate operators
   ***********************************************/

  su3_tuple * gauge_field_smeared_ptr = (su3_tuple*) malloc ( VOLUMEPLUSRAND * sizeof( su3_tuple ) );
  if ( gauge_field_smeared_ptr == NULL) {
    fprintf(stderr, "[cpff_xg_contract] Error from init_2level_su3table %s %d\n", __FILE__, __LINE__ );
    EXIT(47);
  }

#pragma omp parallel for
  for ( unsigned int i = 0; i < VOLUMEPLUSRAND; i++ ) {
    for ( int mu = 0; mu < 4; mu++ ) {
      gauge_field_smeared_ptr[i][mu].c00 = g_gauge_field[_GGI(i,mu)+ 0] + I * g_gauge_field[_GGI(i,mu)+ 1];
      gauge_field_smeared_ptr[i][mu].c01 = g_gauge_field[_GGI(i,mu)+ 2] + I * g_gauge_field[_GGI(i,mu)+ 3];
      gauge_field_smeared_ptr[i][mu].c02 = g_gauge_field[_GGI(i,mu)+ 4] + I * g_gauge_field[_GGI(i,mu)+ 5];
      gauge_field_smeared_ptr[i][mu].c10 = g_gauge_field[_GGI(i,mu)+ 6] + I * g_gauge_field[_GGI(i,mu)+ 7];
      gauge_field_smeared_ptr[i][mu].c11 = g_gauge_field[_GGI(i,mu)+ 8] + I * g_gauge_field[_GGI(i,mu)+ 9];
      gauge_field_smeared_ptr[i][mu].c12 = g_gauge_field[_GGI(i,mu)+10] + I * g_gauge_field[_GGI(i,mu)+11];
      gauge_field_smeared_ptr[i][mu].c20 = g_gauge_field[_GGI(i,mu)+12] + I * g_gauge_field[_GGI(i,mu)+13];
      gauge_field_smeared_ptr[i][mu].c21 = g_gauge_field[_GGI(i,mu)+14] + I * g_gauge_field[_GGI(i,mu)+15];
      gauge_field_smeared_ptr[i][mu].c22 = g_gauge_field[_GGI(i,mu)+16] + I * g_gauge_field[_GGI(i,mu)+17];
    }
  }

  /***************************************************************************
   *
   * Measurement on step-wise smeared gauge field
   *
   ***************************************************************************/

  for ( int istout = 0; istout < stout_level_num; istout++ ) {

    int const stout_iter = ( istout == 0 ) ? stout_level_iter[0] : stout_level_iter[istout] - stout_level_iter[istout - 1];

    if ( io_proc == 2 ) fprintf ( stdout, "# [cpff_xg_contract] stout level %2d iter %2d\n", istout, stout_iter );

    gettimeofday ( &ta, (struct timezone *)NULL );
#ifdef HAVE_TMLQCD_LIBWRAPPER
    exitstatus = tmLQCD_stout_smear_gauge_field ( gauge_field_smeared_ptr, stout_iter , stout_rho );
    /* exitstatus = tmLQCD_stout_smear_3d_gauge_field ( gauge_field_smeared_ptr, stout_iter , stout_rho ); */
#else
    exitstatus = 1;
#endif
    if(exitstatus != 0) {
      fprintf(stderr, "[cpff_xg_contract] Error from stout smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(47);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    char timer_tag[100];
    sprintf( timer_tag, "stout-smear-%u", stout_iter );
    show_time ( &ta, &tb, "cpff_xg_contract", timer_tag, io_proc==2 );


    double * gf = (double*)( gauge_field_smeared_ptr[0] );
#ifdef HAVE_MPI
    xchange_gauge_field ( gf );
#endif

    double ** pl2 = init_2level_dtable ( T_global, 2 );

    /***************************************************************************
     * gluonic operators from plaquettes
     ***************************************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    exitstatus = gluonic_operators ( pl2, gf );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[cpff_xg_contract] Error from gluonic_operators, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(14);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "cpff_xg_contract", "gluonic_operators", io_proc==2 );

    if ( io_proc == 2 && g_verbose > 2) {
      fprintf( stdout, "\n\n# [cpff_xg_contract] After Stout %d %f\n", stout_level_iter[istout], stout_rho );
      for ( int it = 0; it < T_global; it++ ) {
        fprintf ( stdout, "x %3d %25.16e %25.16e\n", it, pl2[it][0], pl2[it][1] );
      }
    }

    gettimeofday ( &ta, (struct timezone *)NULL );

    sprintf ( data_tag, "/StoutN%u/StoutRho%6.4f", stout_level_iter[istout], stout_rho );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
    exitstatus = write_aff_contraction ( pl2[0], affw, NULL, data_tag, 2*T_global, "double" );
#elif ( defined HAVE_HDF5 )
    exitstatus = write_h5_contraction ( pl2[0], NULL, output_filename, data_tag, 2 * T_global );
#else
    exitstatus = 1;
#endif
    if ( exitstatus != 0) {
      fprintf(stderr, "[cpff_xg_contract] Error from write_h5_contraction %s %d\n", __FILE__, __LINE__ );
      EXIT(48);
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "cpff_xg_contract", "write-to-file", io_proc==2 );

#if 0
    /***************************************************************************
     * gluonic operators from elements of field strength tensor
     ***************************************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );
    double *** Gp = init_3level_dtable ( VOLUME, 6, 18 );
    double *** Gr = init_3level_dtable ( VOLUME, 6, 18 );
    if ( Gr == NULL || Gp == NULL ) {
      fprintf ( stderr, "[cpff_xg_contract] Error from G_plaq_rect, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(8);
    }

    exitstatus = G_plaq_rect ( Gp, Gr, g_gauge_field);
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[cpff_xg_contract] Error from G_plaq_rect, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(8);
    }

    exitstatus = gluonic_operators_eo_from_fst ( pl, Gp );
    if ( exitstatus != 0) {
      fprintf(stderr, "[cpff_xg_contract] Error from gluonic_operators_eo_from_fst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(48);
    }

    sprintf ( data_tag, "/StoutN%u/StoutRho%6.4f/clover/plaquette", stout_level_iter[istout], stout_rho );
#  if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
    exitstatus = write_aff_contraction ( pl[0], affw, NULL, data_tag, 2 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
    exitstatus = write_h5_contraction ( pl[0], NULL, filename, data_tag, 2 * T_global );
#else
    exitstatus = 1;
#endif
    if ( exitstatus != 0) {
      fprintf(stderr, "[cpff_xg_contract] Error from write_h5_contraction %s %d\n", __FILE__, __LINE__ );
      EXIT(48);
    }

    exitstatus = gluonic_operators_eo_from_fst ( pl2, Gr );
    if ( exitstatus != 0) {
      fprintf(stderr, "[cpff_xg_contract] Error from gluonic_operators_eo_from_fst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(48);
    }
    sprintf ( data_tag, "/StoutN%u/StoutRho%6.4f/clover/rectangle", stout_level_iter[istout], stout_rho );
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
    exitstatus = write_aff_contraction ( pl[0], affw, NULL, data_tag, 2 * T_global, "double" );
#elif ( defined HAVE_HDF5 )
    exitstatus = write_h5_contraction ( pl[0], NULL, filename, data_tag, 2 * T_global );
#else
    exitstatus = 1;
#endif
    if ( exitstatus != 0) {
      fprintf(stderr, "[cpff_xg_contract] Error from write_h5_contraction %s %d\n", __FILE__, __LINE__ );
      EXIT(48);
    }

    fini_3level_dtable ( &Gp );
    fini_3level_dtable ( &Gr );
#endif

    fini_2level_dtable ( &pl2 );
    
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "cpff_xg_contract", "xg-G-contract-write", io_proc==2 );

  }  /* end of if n_stout > 0 */

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
  free ( gauge_field_smeared_ptr );

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  const char * aff_status_str = (char*)aff_writer_close (affw);
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[cpff_xg_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    return(32);
  }
#endif

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "cpff_xg_contract", "runtime", io_proc==2 );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [cpff_xg_contract] %s# [cpff_xg_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [cpff_xg_contract] %s# [cpff_xg_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
