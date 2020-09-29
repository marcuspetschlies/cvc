/***************************************************************************

 * test_smearing
 *
 ***************************************************************************/
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
#include <omp.h>
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
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "contractions_io.h"
#include "table_init_d.h"
#include "project.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "contract_factorized.h"
#include "ranlxd.h"

using namespace cvc;


/***********************************************************
 * usage function
 ***********************************************************/
void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}
  
  
/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[200];
  double *gauge_field_smeared = NULL;
  double plaq_r;


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

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  /* exitstatus = tmLQCD_invert_init(argc, argv, 1, 0); */
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    EXIT(14);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(15);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    EXIT(16);
  }
#endif

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[test_smearing] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_smearing] Error from init_geometry\n");
    EXIT(1);
  }
  geometry();

  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );
  size_t const sizeof_gauge_field  = 72 * VOLUME * sizeof( double );

  /***************************************************************************
   * read the gauge field
   ***************************************************************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
#ifndef HAVE_TMLQCD_LIBWRAPPER
  switch(g_gauge_file_format) {
    case 0:
      sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "# [test_smearing] reading ILDG gauge field from file %s\n", filename);
      exitstatus = read_lime_gauge_field_doubleprec(filename);
      break;
    case 1:
      sprintf(filename, "%s.%.5d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "\n# [test_smearing] reading gauge field from file %s\n", filename);
      exitstatus = read_nersc_gauge_field(g_gauge_field, filename, &plaq_r);
      break;
  }
  if(exitstatus != 0) {
    fprintf(stderr, "[test_smearing] Error, could not read gauge field\n");
    EXIT(21);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_smearing] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&tmLQCD_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if( tmLQCD_gauge_field == NULL) {
    fprintf(stderr, "[test_smearing] Error, tmLQCD_gauge_field is NULL\n");
    EXIT(5);
  }
  memcpy( g_gauge_field, tmLQCD_gauge_field, 72*VOLUME*sizeof(double));
#endif

#ifdef HAVE_MPI
  xchange_gauge_field ( g_gauge_field );
#endif
  
  /* unit_gauge_field( g_gauge_field, VOLUME); */

  /***************************************************************************
   * measure the plaquette
   ***************************************************************************/
  exitstatus = plaquetteria  ( g_gauge_field );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_smearing] Error from plaquetteria, status was %d\n", exitstatus);
    EXIT(2);
  }
 
  /***************************************************************************
   * smeared gauge field
   ***************************************************************************/
  alloc_gauge_field( &gauge_field_smeared, VOLUMEPLUSRAND);

  if ( g_cart_id == 0 ) { fprintf(stdout, "# [test_smearing] smearing gauge field \n"); }

  memcpy(gauge_field_smeared, g_gauge_field, sizeof_gauge_field);
#ifdef HAVE_MPI
  xchange_gauge_field ( gauge_field_smeared );
#endif

  if ( N_ape > 0 ) {
    if ( g_cart_id == 0 ) fprintf ( stdout, "# [test_smearing] start APE-smearing gauge field\n" );

    exitstatus = APE_Smearing ( gauge_field_smeared, alpha_ape, N_ape);
    if(exitstatus != 0) {
      fprintf(stderr, "[test_smearing] Error from APE_Smearing, status was %d\n", exitstatus);
      EXIT(47);
    }
  }  /* end of if N_ape > 0 */

  /***************************************************************************
   * plaquette of APE smeared gauge field
   ***************************************************************************/
  exitstatus = plaquetteria  ( gauge_field_smeared );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_smearing] Error from plaquetteria, status was %d\n", exitstatus);
    EXIT(2);
  }

  if ( N_ape > 0 ) {
    memcpy ( g_gauge_field, gauge_field_smeared, sizeof_gauge_field);
    double plaq = 0;
    plaquette2 ( &plaq,  g_gauge_field );

    sprintf ( filename, "%s.N%d_a%4.2f.%.4d", gaugefilename_prefix, N_ape, alpha_ape, Nconf );
    exitstatus = write_lime_gauge_field ( filename, plaq,  Nconf,  64 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_smearing] Error from write_lime_gauge_field, status was %d\n", exitstatus);
      EXIT(2);
    }

    if ( g_cart_id == 0 ) fprintf ( stdout, "# [test_smearing] finished APE-smearing gauge field\n" );
  }


  /***************************************************************************
   * spinor fields
   ***************************************************************************/
  double ** spinor_work = init_2level_dtable ( 2, _GSI(VOLUME+RAND) );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[test_smearing] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(2);
  }
#if 0
#endif  /* of if 0 */

#if 0
  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_smearing] Error from init_rng_stat_file status was %d\n", exitstatus);
    EXIT(38);
  }


  if ( read_stochastic_source ) {
    sprintf(filename, "%s.%.4d.%.5d", filename_prefix, Nconf, 0);
    if ( ( exitstatus = read_lime_spinor( spinor_work[0], filename, 0) ) != 0 ) {
      fprintf(stderr, "[test_smearing] Error from read_lime_spinor, status was %d\n", exitstatus);
      EXIT(2);
    }
  } else {
    /* set a stochstic volume source */
    exitstatus = prepare_volume_source(spinor_work[0], VOLUME);
    if(exitstatus != 0) {
      fprintf(stderr, "[test_smearing] Error from prepare_volume_source, status was %d\n", exitstatus);
      EXIT(39);
    }
     if ( write_stochastic_source ) {
       sprintf( filename, "%s.%.4d.%.5d", filename_prefix, Nconf, 0);
       exitstatus = write_propagator( spinor_work[0], filename, 0, 64 );
       if ( exitstatus != 0 ) {
         fprintf(stderr, "[piN2piN_factorized] Error from write_propagator, status was %d\n", exitstatus);
       }
     }

  }  /* end of if read stochastic source */

  double checksum[2] = {0., 0.};
#ifdef HAVE_MPI
  double buffer[2];
#endif

  for ( unsigned int ix = 0; ix < 12*VOLUME; ix++ ) {
    checksum[0] += spinor_work[0][2*ix  ];
    checksum[1] += spinor_work[0][2*ix+1];
  }
#ifdef HAVE_MPI
  exitstatus = MPI_Allreduce(checksum, buffer, 2, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  if( exitstatus != MPI_SUCCESS) {
    fprintf(stderr, "[test_smearing] Error from MPI_Allreduce, status was %d\n", exitstatus);
    return(1);
  }
  checksum[0] = buffer[0];
  checksum[1] = buffer[1];
#endif

  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [] checksum before smearing = %25.16e %25.16e\n", checksum[0], checksum[1]);
  }
  
  memcpy( spinor_work[1], spinor_work[0], _GSI(VOLUME)*sizeof(double));
  exitstatus = Jacobi_Smearing( gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);

  checksum[0] = 0.;
  checksum[1] = 0.;
  for ( unsigned int ix = 0; ix < 12*VOLUME; ix++ ) {
    checksum[0] += spinor_work[1][2*ix  ];
    checksum[1] += spinor_work[1][2*ix+1];
  }
#ifdef HAVE_MPI
  exitstatus = MPI_Allreduce(checksum, buffer, 2, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  if( exitstatus != MPI_SUCCESS) {
    fprintf(stderr, "[test_smearing] Error from MPI_Allreduce, status was %d\n", exitstatus);
    return(1);
  }
  checksum[0] = buffer[0];
  checksum[1] = buffer[1];
#endif

  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test_smearing] checksum after smearing = %25.16e %25.16e\n", checksum[0], checksum[1]);
  }
#endif  /* of if 0 */
  
  memset( spinor_work[0], 0, sizeof_spinor_field );

#if 0
  /* for ( int isrc = 0; isrc < g_source_location_number; isrc++ ) */
  for ( int isrc = 0; isrc <= 0; isrc++ )
  {
    int source_proc_id = -1, sx[4] = {0,0,0,0};
    int gsx[4] = { g_source_coords_list[isrc][0], g_source_coords_list[isrc][1], g_source_coords_list[isrc][2], g_source_coords_list[isrc][3] };
 
    get_point_source_info (gsx, sx, &source_proc_id);

    if ( source_proc_id == g_cart_id ) {
      fprintf(stdout, "# [test_smearing] proc%.4d has the source %3d %3d %3d %3d / %3d %3d %3d %3d\n", source_proc_id,
          gsx[0], gsx[1], gsx[2], gsx[3], sx[0], sx[1], sx[2], sx[3]);
      spinor_work[0][ _GSI( g_ipt[sx[0]][sx[1]][sx[2]][sx[3]]) ] = 1.;
    }
  }
#endif

  for ( int it = 0; it < T_global; it++ )
  {
    int source_proc_id = -1, sx[4] = {0,0,0,0};
    int gsx[4] = { it, g_source_coords_list[0][1], g_source_coords_list[0][2], g_source_coords_list[0][3] };
 
    get_point_source_info (gsx, sx, &source_proc_id);

    if ( source_proc_id == g_cart_id ) {
      fprintf(stdout, "# [test_smearing] proc%.4d has the source %3d %3d %3d %3d / %3d %3d %3d %3d\n", source_proc_id,
          gsx[0], gsx[1], gsx[2], gsx[3], sx[0], sx[1], sx[2], sx[3]);
      spinor_work[0][ _GSI( g_ipt[sx[0]][sx[1]][sx[2]][sx[3]]) ] = 1.;
    }
  }
  int gsx[4] = { g_source_coords_list[0][0], g_source_coords_list[0][1], g_source_coords_list[0][2], g_source_coords_list[0][3] };

  /*******************************************
   * smearing rms source radius
   *******************************************/
  memcpy( spinor_work[1], spinor_work[0], sizeof_spinor_field );

  double ** r2 = init_2level_dtable ( T_global, 4 );
  double ** w2 = init_2level_dtable ( T_global, 4 );

  sprintf ( filename, "r2_w2.N%d_a%4.2f.%.4d", N_Jacobi, kappa_Jacobi, Nconf );
  FILE * ofs = NULL; 
  if ( g_cart_id == 0 ) {
    ofs = fopen ( filename, "w" );
  }

  for ( int Nsmear = 0; Nsmear <= N_Jacobi; Nsmear += 5 ) { 

    if ( g_cart_id == 0 ) fprintf ( stdout, "# [test_smearing] N = %3d  kappa = %f\n", Nsmear, kappa_Jacobi );

    exitstatus = rms_radius ( r2, w2, spinor_work[1], gsx );

    if (g_cart_id == 0 ) {
      for ( int it = 0; it < T_global; it++ ) {
        for ( int ispin = 0; ispin < 4; ispin ++ ) {
          fprintf ( ofs, "%4d   %4d %2d    %25.16e %25.16e\n", Nsmear, it, ispin, r2[it][ispin], w2[it][ispin] );
        }
      }
    }

    exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], 5, kappa_Jacobi);
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[test_smearing] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }
  }
  if ( g_cart_id == 0 ) fclose ( ofs );
  
  fini_2level_dtable ( &r2 );
  fini_2level_dtable ( &w2 );

  /***********************************************
   * source profile
   ***********************************************/
  sprintf( filename, "source_profile.N%d.k%4.2f", N_Jacobi, kappa_Jacobi );
  exitstatus = source_profile ( spinor_work[1], gsx, filename );

#if 0
#endif  /* of if 0 */

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();

  fini_2level_dtable ( &spinor_work );

  if( g_gauge_field != NULL ) free( g_gauge_field );

  if( gauge_field_smeared != NULL ) free( gauge_field_smeared );
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_smearing] %s# [test_smearing] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_smearing] %s# [test_smearing] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
