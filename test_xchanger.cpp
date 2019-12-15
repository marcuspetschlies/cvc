/****************************************************
 * test_xchanger
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
  fprintf(stdout, "# [test_xchanger] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_xchanger] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[test_xchanger] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [test_xchanger] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_xchanger] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[test_xchanger] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
    fprintf(stderr, "[test_xchanger] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }


  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[test_xchanger] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [test_xchanger] proc%.4d has io proc id %d\n", g_cart_id, io_proc );



  double * gauge_field_copy = init_1level_dtable ( 72 * VOLUMEPLUSRAND );

  memcpy ( gauge_field_copy, g_gauge_field, 72*VOLUME*sizeof(double) );

  plaquetteria ( gauge_field_copy );

  xchanger_type xg;
  mpi_init_xchanger( &xg, 72 );

  mpi_xchanger ( gauge_field_copy, &xg );

  plaquetteria ( gauge_field_copy );

  mpi_fini_xchanger ( &xg );
  fini_1level_dtable ( &gauge_field_copy );


  /***************************************************************************
   * test field strength tensor function
   ***************************************************************************/
  double *** Gp = init_3level_dtable ( VOLUME, 6, 18 );
  double *** Gr = init_3level_dtable ( VOLUME, 6, 18 );
  
  if ( Gr == NULL || Gp == NULL ) {
    fprintf ( stderr, "[test_xchanger] Error from G_plaq_rect, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }

  exitstatus = G_plaq_rect ( Gp, Gr, g_gauge_field);
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[test_xchanger] Error from G_plaq_rect, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }


  for ( int it = 0; it  < T; it++ ) {
  for ( int ix = 0; ix  < LX; ix++ ) {
  for ( int iy = 0; iy  < LY; iy++ ) {
  for ( int iz = 0; iz  < LZ; iz++ ) {
    unsigned int const iix = g_ipt[it][ix][iy][iz];
    for ( int i = 0; i < 6; i++ ) {
      for ( int k = 0; k < 9; k++ ) {
        fprintf ( stdout, "G %3d%3d%3d%3d  %d    %d %d   %25.16e %25.16e   %25.16e %25.16e\n",
            it + g_proc_coords[0] * T,
            ix + g_proc_coords[1] * LX,
            iy + g_proc_coords[2] * LY,
            iz + g_proc_coords[3] * LZ,
            i, k/3, k%3,
            Gp[iix][i][2*k],
            Gp[iix][i][2*k+1],
            Gr[iix][i][2*k],
            Gr[iix][i][2*k+1] );
      }
    }
  }}}}


  sprintf( filename, "Gp.%d_%d_%d_%d", g_nproc_t, g_nproc_x, g_nproc_y, g_nproc_z );

  exitstatus = write_lime_contraction ( Gp[0][0], filename, 64, 54, "field strength tensor clover plaquettes", Nconf, 0 );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[test_xchanger] Error from write_lime_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }

  sprintf( filename, "Gr.%d_%d_%d_%d", g_nproc_t, g_nproc_x, g_nproc_y, g_nproc_z );

  exitstatus = write_lime_contraction ( Gr[0][0], filename, 64, 54, "field strength tensor clover rectangles", Nconf, 0 );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[test_xchanger] Error from write_lime_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }

#if 0
  double psum[2] = { 0., 0. };

  for ( unsigned int ix = 0; ix < VOLUME ; ix++ ) {
    for ( int i = 0; i < 6; i++ ) {
      psum[0] += Gp[ix][i][0] + Gp[ix][i][8] + Gp[ix][i][16];
      psum[1] += Gp[ix][i][1] + Gp[ix][i][9] + Gp[ix][i][17];
    }
  }

  double psumg[2] = {0., 0.};
#ifdef HAVE_MPI
  MPI_Allreduce ( psum, psumg, 2, MPI_DOUBLE, MPI_SUM, g_cart_grid );
#else
  psumg[0] = psum[0];
  psumg[1] = psum[1];
#endif
  psumg[0] /= T_global * LX_global * LY_global * LZ_global * 18;
  psumg[1] /= T_global * LX_global * LY_global * LZ_global * 18;

  if ( g_cart_id  == 0 ) {
    fprintf ( stdout, "# [test_xchanger] psumg  %25.16e  %25.16e\n", psumg[0], psumg[1] );
  }
#endif  /* of if 0  */
  fini_3level_dtable ( &Gp );
  fini_3level_dtable ( &Gr );

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

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
  show_time ( &start_time, &end_time, "test_xchanger", "runtime", io_proc==2 );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_xchanger] %s# [test_xchanger] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_xchanger] %s# [test_xchanger] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
