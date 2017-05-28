/****************************************************
 * test2.cpp
 * 
 * Tue May 23 23:45:32 CEST 2017
 *
 * PURPOSE:
 * TODO:
 * DONE:
 *
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
#include "matrix_init.h"
#include "project.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "contract_factorized.h"
#include "ranlxd.h"

#include <string>
#include <iostream>
#include <iomanip>

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
  int io_proc = -1;
  char filename[200];
  double *tmLQCD_gauge_field = NULL;

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
  char aff_tag[200];
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
  fprintf(stdout, "[test2] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[test2] Error from init_geometry\n");
    EXIT(1);
  }
  geometry();


  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
#ifndef HAVE_TMLQCD_LIBWRAPPER
  switch(g_gauge_file_format) {
    case 0:
      sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
      exitstatus = read_lime_gauge_field_doubleprec(filename);
      break;
    case 1:
      sprintf(filename, "%s.%.5d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "\n# [test2] reading gauge field from file %s\n", filename);
      exitstatus = read_nersc_gauge_field(g_gauge_field, filename, &plaq_r);
      break;
  }
  if(exitstatus != 0) {
    fprintf(stderr, "[test2] Error, could not read gauge field\n");
    EXIT(21);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test2] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&tmLQCD_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if( tmLQCD_gauge_field == NULL) {
    fprintf(stderr, "[test2] Error, tmLQCD_gauge_field is NULL\n");
    EXIT(5);
  }
  memcpy( g_gauge_field, tmLQCD_gauge_field, 72*VOLUME*sizeof(double));
#endif

#ifdef HAVE_MPI
  xchange_gauge_field ( g_gauge_field );
#endif
  /* measure the plaquette */
  
  if ( ( exitstatus = plaquetteria  ( g_gauge_field ) ) != 0 ) {
    fprintf(stderr, "[test2] Error from plaquetteria, status was %d\n", exitstatus);
    EXIT(2);
  }
 
#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [test2] proc%.4d tr%.4d is io process\n", g_cart_id, g_tr_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [test2] proc%.4d tr%.4d is send process\n", g_cart_id, g_tr_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif

#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [test2] using aff version %s\n", aff_status_str);

    sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "B_B", Nconf, 0 );
    fprintf(stdout, "# [test2] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test2] Error from aff_writer, status was %s\n", aff_status_str);
      EXIT(4);
    }
  }  /* end of if io_proc == 2 */
#endif

  rlxd_init(2, g_seed);

  double **sf = NULL;
  init_2level_buffer ( &sf, 12, 24*VOLUME );

  fermion_propagator_type *fp  = create_fp_field( VOLUME );
  fermion_propagator_type *fp2 = create_fp_field( VOLUME );
  double *fv = (double*)malloc(24*VOLUME*sizeof(double) );
  double **v3 = NULL, ***vp = NULL, **v1 = NULL, **v2 = NULL;

  exitstatus = init_2level_buffer ( &v3, VOLUME, 24 );
  exitstatus = init_3level_buffer ( &vp, T_global, g_sink_momentum_number, 24 );

  // ranlxd( sf[0], VOLUME*288);
  for( int i=0; i<12; i++) {
    sprintf(filename, "sf1.%.2d", i);
    // write_propagator( sf[i], filename, 0, 64);
    read_lime_spinor ( sf[i], filename, 0);
  }
  assign_fermion_propagator_from_spinor_field ( fp, sf, VOLUME);

  // ranlxd( sf[0], VOLUME*288);
  for( int i=0; i<12; i++) {
    sprintf(filename, "sf2.%.2d", i);
    // write_propagator( sf[i], filename, 0, 64);
    read_lime_spinor ( sf[i], filename, 0);
  }
  assign_fermion_propagator_from_spinor_field ( fp2, sf, VOLUME);
  fini_2level_buffer ( &sf );

  // ranlxd( fv, 24*VOLUME);
  sprintf(filename, "sf3.%.2d", 0);
  // write_propagator( fv, filename, 0, 64);
  read_lime_spinor ( fv, filename, 0);


  exitstatus = contract_v3  ( v3, fv, fp, VOLUME );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test2] Error from contract_v3, status was %d\n", exitstatus);
    EXIT(12);
  }

  FILE *ofs = fopen("v3_x", "w");
  for( int x0 = 0; x0 < T; x0++ ) {
  for( int x1 = 0; x1 < LX; x1++ ) {
  for( int x2 = 0; x2 < LY; x2++ ) {
  for( int x3 = 0; x3 < LZ; x3++ ) {
    unsigned int ix = g_ipt[x0][x1][x2][x3];
    for ( int c = 0; c < 12; c++ ) {
      fprintf(ofs, "%25.16e %25.16e\n", v3[ix][2*c], v3[ix][2*c+1]);
    }
  }}}}
  fclose( ofs );

  exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test2] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
    EXIT(12);
  }

  ofs = fopen("v3_p", "w");
  for( int x0 = 0; x0 < T; x0++ ) {
    for( int ip = 0; ip < g_sink_momentum_number; ip++ ) {
      for ( int c = 0; c < 12; c++ ) {
        fprintf(ofs, "t %2d p %3d %3d %3d c %3d vp %25.16e %25.16e\n", x0, 
            g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2],
            c, vp[x0][ip][2*c], vp[x0][ip][2*c+1] );
      }
    }
  }
  fclose( ofs );

  sprintf(aff_tag, "/v3/test");
  exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test2] Error from contract_vn_write_aff, status was %d\n", exitstatus);
    EXIT(12);
  }

  fini_2level_buffer ( &v3 );
  fini_3level_buffer ( &vp );

  /*******************************************
   *  v1 test
   *******************************************/

  exitstatus = init_2level_buffer ( &v1, VOLUME, 72 );
  exitstatus = init_3level_buffer ( &vp, T_global, g_sink_momentum_number, 72 );

  exitstatus = contract_v1  ( v1, fv, fp, VOLUME );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test2] Error from contract_v1, status was %d\n", exitstatus);
    EXIT(12);
  }

  ofs = fopen("v1_x", "w");
  for( int x0 = 0; x0 < T; x0++ ) {
  for( int x1 = 0; x1 < LX; x1++ ) {
  for( int x2 = 0; x2 < LY; x2++ ) {
  for( int x3 = 0; x3 < LZ; x3++ ) {
    unsigned int ix = g_ipt[x0][x1][x2][x3];
    for ( int c = 0; c < 36; c++ ) {
      fprintf(ofs, "%25.16e %25.16e\n", v1[ix][2*c], v1[ix][2*c+1]);
    }
  }}}}
  fclose( ofs );

  exitstatus = contract_vn_momentum_projection ( vp, v1, 36, g_sink_momentum_list, g_sink_momentum_number);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test2] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
    EXIT(12);
  }

  ofs = fopen("v1_p", "w");
  for( int x0 = 0; x0 < T; x0++ ) {
    for( int ip = 0; ip < g_sink_momentum_number; ip++ ) {
      for ( int c = 0; c < 36; c++ ) {
        fprintf(ofs, "t %2d p %3d %3d %3d c %3d vp %25.16e %25.16e\n", x0, 
            g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2],
            c, vp[x0][ip][2*c], vp[x0][ip][2*c+1] );
      }
    }
  }
  fclose( ofs );

  sprintf(aff_tag, "/v1/test");
  exitstatus = contract_vn_write_aff ( vp, 36, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test2] Error from contract_vn_write_aff, status was %d\n", exitstatus);
    EXIT(12);
  }

  fini_2level_buffer ( &v1 );
  fini_3level_buffer ( &vp );

  /*******************************************
   * v2 test
   *******************************************/

  exitstatus = init_2level_buffer ( &v2, VOLUME, 384 );
  exitstatus = init_3level_buffer ( &vp, T_global, g_sink_momentum_number, 384 );

  exitstatus = contract_v2  ( v2, fv, fp, fp2, VOLUME );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test2] Error from contract_v2, status was %d\n", exitstatus);
    EXIT(12);
  }

  ofs = fopen("v2_x", "w");
  for( int x0 = 0; x0 < T; x0++ ) {
  for( int x1 = 0; x1 < LX; x1++ ) {
  for( int x2 = 0; x2 < LY; x2++ ) {
  for( int x3 = 0; x3 < LZ; x3++ ) {
    unsigned int ix = g_ipt[x0][x1][x2][x3];
    for ( int c = 0; c < 192; c++ ) {
      fprintf(ofs, "%25.16e %25.16e\n", v2[ix][2*c], v2[ix][2*c+1]);
    }
  }}}}
  fclose( ofs );

  exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test2] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
    EXIT(12);
  }

  ofs = fopen("v2_p", "w");
  for( int x0 = 0; x0 < T; x0++ ) {
    for( int ip = 0; ip < g_sink_momentum_number; ip++ ) {
      for ( int c = 0; c < 192; c++ ) {
        fprintf(ofs, "t %2d p %3d %3d %3d c %3d vp %25.16e %25.16e\n", x0, 
            g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2],
            c, vp[x0][ip][2*c], vp[x0][ip][2*c+1] );
      }
    }
  }
  fclose( ofs );

  sprintf(aff_tag, "/v2/test");
  exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test2] Error from contract_vn_write_aff, status was %d\n", exitstatus);
    EXIT(12);
  }

  fini_2level_buffer ( &v2 );
  fini_3level_buffer ( &vp );

  /*******************************************
   * finalize
   *******************************************/

  free_fp_field( &fp );
  free_fp_field( &fp2 );
  free( fv );
  fini_2level_buffer ( &v3 );
  fini_3level_buffer ( &vp );

#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test2] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(111);
    }
  }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */


  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();

  if( g_gauge_field != NULL ) free( g_gauge_field );
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test2] %s# [test2] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test2] %s# [test2] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
