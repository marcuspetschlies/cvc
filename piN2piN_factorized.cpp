/****************************************************
 * piN2piN_factorized.c
 * 
 * Tue May 30 10:40:59 CEST 2017
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
#include "clover.h"
#include "dummy_solver.h"
#include "scalar_products.h"

using namespace cvc;

/************************************************************************************
 * determine all stochastic source timeslices needed; make a source timeslice list
 ************************************************************************************/
int **stochastic_source_timeslice_lookup_table;
int *stochastic_source_timeslice_list;
int stochastic_source_timeslice_number;

int get_stochastic_source_timeslices (void) {
  int tmp_list[T_global];
  int i_snk;

  for( int t = 0; t<T_global; t++) { tmp_list[t] = -1; }

  i_snk = 0;
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];
    for( int i_coherent = 0; i_coherent < g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + i_coherent * ( T_global / g_coherent_source_number) ) % T_global;
      for( int t = 0; t<=g_src_snk_time_separation; t++) {
        int t_snk = ( t_coherent + t ) % T_global;
        if( tmp_list[t_snk] == -1 ) {
          tmp_list[t_snk] = i_snk;
          i_snk++;
        }
      }
    }  /* of loop on coherent source timeslices */
  }    /* of loop on base source timeslices */
  if(g_cart_id == 0) { fprintf(stdout, "# [get_stochastic_source_timeslices] number of stochastic timeslices = %2d\n", i_snk); }

  stochastic_source_timeslice_number = i_snk;
  if(stochastic_source_timeslice_number == 0) {
    fprintf(stderr, "# [get_stochastic_source_timeslices] Error, stochastic_source_timeslice_number = 0\n");
    return(4);
  }

  stochastic_source_timeslice_list = (int*)malloc(i_snk*sizeof(int));
  if(stochastic_source_timeslice_list == NULL) {
    fprintf(stderr, "[get_stochastic_source_timeslices] Error from malloc\n");
    return(1);
  }

  stochastic_source_timeslice_lookup_table = (int**)malloc( g_source_location_number * g_coherent_source_number * sizeof(int*));
  if(stochastic_source_timeslice_lookup_table == NULL) {
    fprintf(stderr, "[get_stochastic_source_timeslices] Error from malloc\n");
    return(2);
  }

  stochastic_source_timeslice_lookup_table[0] = (int*)malloc( (g_src_snk_time_separation+1) * g_source_location_number * g_coherent_source_number * sizeof(int));
  if(stochastic_source_timeslice_lookup_table[0] == NULL) {
    fprintf(stderr, "[get_stochastic_source_timeslices] Error from malloc\n");
    return(3);
  }
  for( int i_src=1; i_src<g_source_location_number*g_coherent_source_number; i_src++) {
    stochastic_source_timeslice_lookup_table[i_src] = stochastic_source_timeslice_lookup_table[i_src-1] + (g_src_snk_time_separation+1);
  }

  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];
    for( int i_coherent = 0; i_coherent < g_coherent_source_number; i_coherent++) {
      int i_prop = i_src * g_coherent_source_number + i_coherent;
      int t_coherent = ( t_base + i_coherent * ( T_global / g_coherent_source_number) ) % T_global;
      for( int t = 0; t<=g_src_snk_time_separation; t++) {
        int t_snk = ( t_coherent + t ) % T_global;
        if( tmp_list[t_snk] != -1 ) {
          stochastic_source_timeslice_list[ tmp_list[t_snk] ] = t_snk;
          stochastic_source_timeslice_lookup_table[i_prop][t] = tmp_list[t_snk];
        }
      }
    }  /* of loop on coherent source timeslices */
  }    /* of loop on base source timeslices */

  if(g_cart_id == 0) {
    /* TEST */
    for( int i_src = 0; i_src<g_source_location_number; i_src++) {
      int t_base = g_source_coords_list[i_src][0];
      for( int i_coherent = 0; i_coherent < g_coherent_source_number; i_coherent++) {
        int i_prop = i_src * g_coherent_source_number + i_coherent;
        int t_coherent = ( t_base + i_coherent * ( T_global / g_coherent_source_number) ) % T_global;

        for( int t = 0; t <= g_src_snk_time_separation; t++) {
          fprintf(stdout, "# [get_stochastic_source_timeslices] i_src = %d, i_prop = %d, t_src = %d, dt = %d, t_snk = %d, lookup table = %d\n",
              i_src, i_prop, t_coherent, t,
              stochastic_source_timeslice_list[ stochastic_source_timeslice_lookup_table[i_prop][t] ],
              stochastic_source_timeslice_lookup_table[i_prop][t]);
        }
      }
    }

    /* TEST */
    for( int t=0; t<stochastic_source_timeslice_number; t++) {
      fprintf(stdout, "# [get_stochastic_source_timeslices] stochastic source timeslice no. %d is t = %d\n", t, stochastic_source_timeslice_list[t]);
    }
  }  /* end of if g_cart_id == 0 */
  return(0);
}  /* end of get_stochastic_source_timeslices */

void fini_stochastic_source_timeslices (void) {
  free ( stochastic_source_timeslice_list );
  free ( stochastic_source_timeslice_lookup_table[0] );
  free ( stochastic_source_timeslice_lookup_table );
}  /* end of fini_stochastic_source_timeslices */



/***********************************************************
 * usage function
 ***********************************************************/
void usage() {
  fprintf(stdout, "Code to perform contractions for piN 2-pt. function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -h? this help\n");
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
  
  const int n_c=3;
  const int n_s=4;
  const int max_num_diagram = 6;


  int c;
  int filename_set = 0;
  int exitstatus;
  int op_id_up= -1, op_id_dn = -1;
  int gsx[4], sx[4];
  int source_proc_id = 0;
  int read_stochastic_source      = 0;
  int read_stochastic_source_oet  = 0;
  int read_stochastic_propagator  = 0;
  int write_stochastic_source     = 0;
  int write_stochastic_source_oet = 0;
  int write_stochastic_propagator = 0;
  int check_propagator_residual   = 0;
/***********************************************************/
  int mode_of_operation           = 0;

/* mode of operation
 * 000 = 0 nothing
 * 100 = 1 stoch
 * 010 = 2 contr std
 * 001 = 4 contr oet
 * or combinations thereof
 * per default set to minimal value 0                      */
  int do_stochastic      = 0;
  int do_contraction_std = 0;
  int do_contraction_oet = 0;
/***********************************************************/

  char filename[200];
  double ratime, retime;
#ifndef HAVE_TMLQCD_LIBWRAPPER
  double plaq_r = 0.;
#endif
  double *spinor_work[2];
  unsigned int VOL3;
  size_t sizeof_spinor_field = 0, sizeof_spinor_field_timeslice = 0;
  int io_proc = -1;
  double **propagator_list_up = NULL, **propagator_list_dn = NULL, **sequential_propagator_list = NULL, **stochastic_propagator_list = NULL, **stochastic_propagator_zero_list = NULL,
         **stochastic_source_list = NULL;
  double *gauge_field_smeared = NULL, *tmLQCD_gauge_field = NULL, *gauge_field_with_phase = NULL;
  double **mzz[2], **mzzinv[2];

/*******************************************************************
 * Gamma components for the piN and Delta:
 *                                                                 */
  /* vertex i2, gamma_5 only */
  const int gamma_i2_number             = 1;
  int gamma_i2_list[gamma_i2_number]    = {  5 };
  double gamma_i2_sign[gamma_i2_number] = { +1 };

  /* vertex f2, gamma_5 and id,  vector indices and pseudo-vector */
  const int gamma_f2_number                        = 1;
  int gamma_f2_list[gamma_f2_number]               = {  5 };
  double gamma_f2_sign[gamma_f2_number]            = { +1 };
  double gamma_f2_adjoint_sign[gamma_f2_number]    = { +1 };
  double gamma_f2_g5_adjoint_sign[gamma_f2_number] = { +1 };

  /* vertex c, vector indices and pseudo-vector */
  const int gamma_c_number            = 6;
  int gamma_c_list[gamma_c_number]    = {  1,  2,  3,  7,  8,  9 };
  double gamma_c_sign[gamma_c_number] = { +1, +1, +1, +1, +1, +1 };


  /* vertex f1 for nucleon-type, C g5, C, C g0 g5, C g0 */
  const int gamma_f1_nucleon_number                                = 4;
  int gamma_f1_nucleon_list[gamma_f1_nucleon_number]               = { 14, 11,  8,  2 };
  double gamma_f1_nucleon_sign[gamma_f1_nucleon_number]            = { +1, +1, -1, -1 };
  double gamma_f1_nucleon_transposed_sign[gamma_f1_nucleon_number] = { -1, -1, +1, -1 };

  /* vertex f1 for Delta-type operators, C gi, C gi g0 */
  const int gamma_f1_delta_number                       = 6;
  int gamma_f1_delta_list[gamma_f1_delta_number]        = { 9,  0,  7, 13,  4, 15 };
  double gamma_f1_delta_src_sign[gamma_f1_delta_number] = {-1, +1, +1, +1, +1, -1 };
  double gamma_f1_delta_snk_sign[gamma_f1_delta_number] = {+1, +1, -1, -1, +1, +1 };

  /* vertex for the rho */
  const int gamma_rho_number = 6;
                                          /* g1, g2, g3, g0 g1, g0 g2, g0 g3 */
  int gamma_rho_list[gamma_rho_number] = {    1,  2,  3,    10,    11,    12 };


/*
 *******************************************************************/

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
  char aff_tag[200];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "SscrRwWh?f:m:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'r':
      read_stochastic_source = 1;
      fprintf(stdout, "# [piN2piN_factorized] will read stochastic source\n");
      break;
    case 's':
      read_stochastic_source_oet = 1;
      fprintf(stdout, "# [piN2piN_factorized] will read stochastic oet source\n");
      break;
    case 'S':
      write_stochastic_source_oet = 1;
      fprintf(stdout, "# [piN2piN_factorized] will write stochastic oet source\n");
      break;
    case 'R':
      read_stochastic_propagator = 1;
      fprintf(stdout, "# [piN2piN_factorized] will read stochastic propagator\n");
      break;
    case 'w':
      write_stochastic_source = 1;
      fprintf(stdout, "# [piN2piN_factorized] will write stochastic source\n");
      break;
    case 'W':
      write_stochastic_propagator = 1;
      fprintf(stdout, "# [piN2piN_factorized] will write stochastic propagator\n");
      break;
    case 'c':
      check_propagator_residual = 1;
      fprintf(stdout, "# [piN2piN_factorized] will check_propagator_residual\n");
      break;
    case 'm':
      mode_of_operation = atoi( optarg );
      fprintf(stdout, "# [piN2piN_factorized] will use mdoe of operation %d\n", mode_of_operation );
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
  fprintf(stdout, "# reading input from file %s\n", filename);
  read_input_parser(filename);

  if(g_fermion_type == -1 ) {
    fprintf(stderr, "# [piN2piN_factorized] fermion_type must be set\n");
    exit(1);
  } else {
    fprintf(stdout, "# [piN2piN_factorized] using fermion type %d\n", g_fermion_type);
  }

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [piN2piN_factorized] calling tmLQCD wrapper init functions\n");

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
  fprintf(stdout, "[piN2piN_factorized] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [piN2piN_factorized] git version = %s\n", g_gitversion);
  }

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_geometry\n");
    EXIT(1);
  }
  geometry();

#ifdef HAVE_MPI
  if ( check_propagator_residual ) {
    mpi_init_xchange_eo_spinor();
  }
#endif

  VOL3 = LX*LY*LZ;
  sizeof_spinor_field           = _GSI(VOLUME) * sizeof(double);
  sizeof_spinor_field_timeslice = _GSI(VOL3)   * sizeof(double);


#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [piN2piN_factorized] proc%.4d tr%.4d is io process\n", g_cart_id, g_tr_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [piN2piN_factorized] proc%.4d tr%.4d is send process\n", g_cart_id, g_tr_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif


  /***********************************************
   * read the gauge field or obtain from tmLQCD
   ***********************************************/
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
      if(g_cart_id==0) fprintf(stdout, "\n# [piN2piN_factorized] reading gauge field from file %s\n", filename);
      exitstatus = read_nersc_gauge_field(g_gauge_field, filename, &plaq_r);
      break;
  }
  if(exitstatus != 0) {
    fprintf(stderr, "[piN2piN_factorized] Error, could not read gauge field\n");
    EXIT(21);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[piN2piN_factorized] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&tmLQCD_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if( tmLQCD_gauge_field == NULL) {
    fprintf(stderr, "[piN2piN_factorized] Error, tmLQCD_gauge_field is NULL\n");
    EXIT(5);
  }
  memcpy( g_gauge_field, tmLQCD_gauge_field, 72*VOLUME*sizeof(double));
#endif

#ifdef HAVE_MPI
  xchange_gauge_field ( g_gauge_field );
#endif
  /* measure the plaquette */
  
  if ( ( exitstatus = plaquetteria  ( g_gauge_field ) ) != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from plaquetteria, status was %d\n", exitstatus);
    EXIT(2);
  }
  free ( g_gauge_field ); g_gauge_field = NULL;

  /***********************************************
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {

    alloc_gauge_field(&gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy(gauge_field_smeared, tmLQCD_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_factorized] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
    }  /* end of if N_aoe > 0 */
  }  /* end of if N_Jacobi > 0 */

  if ( check_propagator_residual ) {
    /***********************************************************
     * multiply the phase to the gauge field
     ***********************************************************/
    exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, tmLQCD_gauge_field, co_phase_up );
    if(exitstatus != 0) {
      fprintf(stderr, "[piN2piN_factorized] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    }
  
    /***********************************************
     * initialize clover, mzz and mzz_inv
     ***********************************************/
    exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[piN2piN_factorized] Error from init_clover, status was %d\n", exitstatus );
      EXIT(1);
    }
  
  }  /* end of if check propagator residual */

  /***********************************************************
   * determine the stochastic source timeslices
   ***********************************************************/
  exitstatus = get_stochastic_source_timeslices();
  if(exitstatus != 0) {
    fprintf(stderr, "[piN2piN_factorized] Error from get_stochastic_source_timeslices, status was %d\n", exitstatus);
    EXIT(19);
  }

  /***********************************************************
   * allocate work spaces with halo
   ***********************************************************/
  alloc_spinor_field(&spinor_work[0], VOLUMEPLUSRAND);
  alloc_spinor_field(&spinor_work[1], VOLUMEPLUSRAND);


  /***********************************************************
   * set operator ids depending on fermion type
   ***********************************************************/
  if(g_fermion_type == _TM_FERMION) {
    op_id_up = 0;
    op_id_dn = 1;
  } else if(g_fermion_type == _WILSON_FERMION) {
    op_id_up = 0;
    op_id_dn = 0;
  }

  /***********************************************************
   * interprete mode of operation
   ***********************************************************/
  do_contraction_oet = mode_of_operation / 4;
  do_contraction_std = ( mode_of_operation - do_contraction_oet * 4 ) / 2;
  do_stochastic      = mode_of_operation % 2;
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [piN2piN_factorized] mode of operation = %d / %d / %d\n", do_stochastic, do_contraction_std, do_contraction_oet );
  }




  /******************************************************
   ******************************************************
   **
   ** stochastic inversions
   **  
   **  dn-type inversions
   ******************************************************
   ******************************************************/

  /******************************************************
   * check source coords list
   ******************************************************/
  for ( int i = 0; i < g_source_location_number; i++ ) {
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] + T_global ) % T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /******************************************************
   * determine, which samples to run
   ******************************************************/
  if ( g_sourceid == 0 && g_sourceid2 == 0 ) {
    g_sourceid  = 0;
    g_sourceid2 = g_nsample -1;
  } else if ( g_sourceid2 < g_sourceid ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [piN2piN_factorized] Warning, exchanging source ids\n");
    int i = g_sourceid;
    g_sourceid  = g_sourceid2;
    g_sourceid2 = i;
  }
  g_nsample = g_sourceid2 - g_sourceid + 1;
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [piN2piN_factorized] nample = %2d sourceid %2d - %2d\n", g_nsample, g_sourceid, g_sourceid2 );
    fflush( stdout );
  }

  /******************************************************
   * allocate memory for stochastic sources
   *   and propagators
   ******************************************************/
  exitstatus = init_2level_buffer ( &stochastic_propagator_list, g_nsample, _GSI(VOLUME) );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(44);
  }

  exitstatus = init_2level_buffer ( &stochastic_source_list, g_nsample, _GSI(VOLUME) );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(44);
  }

  /******************************************************/
  /******************************************************/

  if ( do_stochastic % 2 == 1 ) {

  /******************************************************/
  /******************************************************/

  /******************************************************
   * initialize random number generator
   ******************************************************/
  sprintf(filename, "rng_stat.%.4d.stochastic.out", Nconf);
  if( read_stochastic_source || ( ( ! read_stochastic_source ) && ( g_sourceid == 0 ) ) ) {
    /******************************************************
     * if we read stochastic sources or if we do not read
     * stochastic sources but start from sample 0, then
     * initialize random number generator
     ******************************************************/
    exitstatus = init_rng_stat_file ( g_seed, filename );
    if(exitstatus != 0) {
      fprintf(stderr, "[piN2piN_factorized] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    } 
  } else if ( (! read_stochastic_source ) && ( g_sourceid > 0 ) ) {
    /******************************************************
     * if we do not read
     * stochastic sources and start from sample larger than 0,
     * then we read the last rng state
     ******************************************************/
    exitstatus = read_set_rng_stat_file ( filename );
    if(exitstatus != 0) {
      fprintf(stderr, "[piN2piN_factorized] Error from read_set_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(38);
    } 
  }

  /* loop on stochastic samples */
  for(int isample = 0; isample < g_nsample; isample++)
  {

    if ( read_stochastic_source ) {
      sprintf(filename, "%s.%.4d.%.5d", filename_prefix, Nconf, isample + g_sourceid );
      if ( ( exitstatus = read_lime_spinor( stochastic_source_list[isample], filename, 0) ) != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from read_lime_spinor, status was %d\n", exitstatus);
        EXIT(2);
      }
    } else {


      /* set a stochstic volume source */
      exitstatus = prepare_volume_source(stochastic_source_list[isample], VOLUME);
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_factorized] Error from prepare_volume_source, status was %d\n", exitstatus);
        EXIT(39);
      }
    }  /* end of if read stochastic source */


    /******************************************************
     * dummy inversion to start the deflator
     ******************************************************/
    if ( isample == 0 ) {
      memset(spinor_work[1], 0, sizeof_spinor_field);
      exitstatus = _TMLQCD_INVERT(spinor_work[1], stochastic_source_list[0], op_id_up, 0);
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_factorized] Error from tmLQCD_invert, status was %d\n", exitstatus);
        EXIT(12);
      }
    }

    if ( read_stochastic_propagator ) {
      sprintf(filename, "%s.%.4d.%.5d", filename_prefix2, Nconf, isample + g_sourceid );
      if ( ( exitstatus = read_lime_spinor( stochastic_propagator_list[isample], filename, 0) ) != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from read_lime_spinor, status was %d\n", exitstatus);
        EXIT(2);
      }
    } else {

      memset( stochastic_propagator_list[isample], 0, sizeof_spinor_field);
  
      /* project to timeslices, invert */
      for( int i_src = 0; i_src < stochastic_source_timeslice_number; i_src++) {
  
        /******************************************************
         * i_src is just a counter; we take the timeslices from
         * the list stochastic_source_timeslice_list, which are
         * in some order;
         * t_src should be used to address the fields
         ******************************************************/
        int t_src = stochastic_source_timeslice_list[i_src];
        memset(spinor_work[0], 0, sizeof_spinor_field);
  
        int have_source = ( g_proc_coords[0] == t_src / T );
        if( have_source ) {
          fprintf(stdout, "# [piN2piN_factorized] proc %4d = ( %d, %d, %d, %d) has t_src = %3d \n", g_cart_id, 
              g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3], t_src);
          /* this process copies timeslice t_src%T from source */
          unsigned int shift = _GSI(g_ipt[t_src%T][0][0][0]);
          memcpy(spinor_work[0]+shift, stochastic_source_list[isample]+shift, sizeof_spinor_field_timeslice );
        }
  
        /* tm-rotate stochastic source */
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], -1, g_fermion_type, VOLUME);
        }
  
        memset(spinor_work[1], 0, sizeof_spinor_field);
        exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], op_id_dn, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN_factorized] Error from tmLQCD_invert, status was %d\n", exitstatus);
          EXIT(12);
        }
  
        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_dn], 1 );
        }


        /* tm-rotate stochastic propagator at sink */
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[1], spinor_work[1], -1, g_fermion_type, VOLUME);
        }
  
        /* copy only source timeslice from propagator */
        if(have_source) {
          unsigned int shift = _GSI(g_ipt[t_src%T][0][0][0]);
          memcpy( stochastic_propagator_list[isample]+shift, spinor_work[1]+shift, sizeof_spinor_field_timeslice);
        }
  
      }  /* end of loop on stochastic source timeslices */
   
      /* source-smear the stochastic source */
      if ( ( exitstatus = Jacobi_Smearing(gauge_field_smeared, stochastic_source_list[isample], N_Jacobi, kappa_Jacobi) ) != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(72);
      }
  
      /* sink-smear the stochastic propagator */
      if ( ( exitstatus = Jacobi_Smearing(gauge_field_smeared, stochastic_propagator_list[isample], N_Jacobi, kappa_Jacobi) ) != ) {
        fprintf(stderr, "[piN2piN_factorized] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(72);
      }
  
      if ( write_stochastic_source ) {
        /* write to file */
        sprintf( filename, "%s.%.4d.%.5d", filename_prefix, Nconf, isample + g_sourceid ); 
        exitstatus = write_propagator( stochastic_source_list[isample], filename, 0, 64 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from write_propagator, status was %d\n", exitstatus);
        }
      }
      if ( write_stochastic_propagator ) {
        sprintf( filename, "%s.%.4d.%.5d", filename_prefix2, Nconf, isample + g_sourceid ); 
        exitstatus = write_propagator( stochastic_propagator_list[isample], filename, 0, 64 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from write_propagator, status was %d\n", exitstatus);
        }
      }

      /***********************************************************
       * after the inversion is complete and written if necessary,
       * write the current rng state to file
       ***********************************************************/
      sprintf(filename, "rng_stat.%.4d.stochastic.out", Nconf);
      exitstatus = write_rng_stat_file ( filename );
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_factorized] Error from write_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      } 

    }  /* end of if read stochastic propagator else */

  }  /* end of loop on samples */

  /***********************************************************/
  /***********************************************************/

  }  /* end of if do_stochastic */

  /***********************************************************/
  /***********************************************************/

  if ( do_contraction_std ) {

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * up-type, dn-type and sequential propagator
   ***********************************************************/
  no_fields = g_coherent_source_number * n_s*n_c;
  exitstatus = init_2level_buffer ( &propagator_list_up, no_fields, _GSI(VOLUME) );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(44);
  }
  exitstatus = init_2level_buffer ( &propagator_list_dn, no_fields, _GSI(VOLUME) );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(45);
  }

  no_fields = n_s*n_c;
  exitstatus = init_2level_buffer ( &sequential_propagator_list, no_fields, _GSI(VOLUME) );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(45);
  }

  /***********************************************************/
  /***********************************************************/

  /* loop on source locations */
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];
 
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN", Nconf, t_base );
      fprintf(stdout, "# [piN2piN_factorized] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_factorized] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      }
    }  /* end of if io_proc == 2 */

    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {

      /* coherent source timeslice */
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global; 
 
      gsx[0] = t_coherent;
      gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
      gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
      gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

      ratime = _GET_TIME;
      get_point_source_info (gsx, sx, &source_proc_id);

      /***********************************************************
       * up-type propagator
       ***********************************************************/
      exitstatus = point_source_propagator ( &(propagator_list_up[12*i_coherent]), gsx, op_id_up, 1, 1, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, mzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_factorized] Error from point_source_propagator, status was %d\n", exitstatus);
        EXIT(12);
      }

      /***********************************************************
       * dn-type propagator
       ***********************************************************/
      exitstatus = point_source_propagator ( &(propagator_list_dn[12*i_coherent]), gsx, op_id_dn, 1, 1, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, mzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_factorized] Error from point_source_propagator, status was %d\n", exitstatus);
        EXIT(12);
      }

      /***********************************************************/
      /***********************************************************/

      fermion_propagator_type *fp = NULL, *fp2 = NULL, *fp3=NULL;
      double **v1 = NULL, **v2 = NULL,  **v3 = NULL, ***vp = NULL;
      fp  = create_fp_field ( VOLUME );
      fp2 = create_fp_field ( VOLUME );
      fp3 = create_fp_field ( VOLUME );

      /* up propagator as propagator type field */
      assign_fermion_propagator_from_spinor_field ( fp,  &(propagator_list_up[i_coherent * n_s*n_c]), VOLUME);

      /* dn propagator as propagator type field */
      assign_fermion_propagator_from_spinor_field ( fp2, &(propagator_list_dn[i_coherent * n_s*n_c]), VOLUME);

      exitstatus= init_2level_buffer ( &v2, VOLUME, 32 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 32 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      /***********************************************************
       * contractions for N - N with up and dn propagagor
       ***********************************************************/
      for ( int if1 = 0; if1 < gamma_f1_nucleon_number; if1++ ) {
      for ( int if2 = 0; if2 < gamma_f1_nucleon_number; if2++ ) {

        fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_nucleon_list[if2], fp2, VOLUME );

        fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_nucleon_list[if1], fp3, VOLUME );

        fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp3, fp3, -gamma_f1_nucleon_sign[if1]*gamma_f1_nucleon_sign[if2], VOLUME );


        sprintf(aff_tag, "/N-N/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/n1",
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_nucleon_list[if1], gamma_f1_nucleon_list[if2]);

        exitstatus = contract_v5 ( v2, fp, fp3, fp, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_v5, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
          EXIT(49);
        }

        /***********************************************************/
        /***********************************************************/

        sprintf(aff_tag, "/N-N/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/n2",
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_nucleon_list[if1], gamma_f1_nucleon_list[if2]);

        exitstatus = contract_v6 ( v2, fp, fp3, fp, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_v6, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
          EXIT(49);
        }

      }}

      /***********************************************************/
      /***********************************************************/

      /***********************************************************
       * contractions for Delta++ - Delta++
       *   with up ( and dn ) propagagor
       ***********************************************************/
      for ( int if1 = 0; if1 < gamma_f1_delta_number; if1++ ) {

        /* fp2 <- fp x Gamma_i1 */
        fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp2, gamma_f1_delta_list[if1], fp, VOLUME );
        fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp2, fp2, gamma_f1_delta_src_sign[if1], VOLUME );

      for ( int if2 = 0; if2 < gamma_f1_delta_number; if2++ ) {

        /* fp3 <- Gamma_f1 x fp */
        fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_delta_list[if2], fp, VOLUME );
        fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp3, fp3, gamma_f1_delta_snk_sign[if2], VOLUME );

        /***********************************************************/
        /***********************************************************/

        sprintf(aff_tag, "/D-D/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/d1",
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_delta_list[if1], gamma_f1_delta_list[if2]);

        exitstatus = contract_v5 ( v2, fp2, fp3, fp, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_v5, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
          EXIT(49);
        }

        /***********************************************************/
        /***********************************************************/

        sprintf(aff_tag, "/D-D/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/d2",
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_delta_list[if1], gamma_f1_delta_list[if2]);

        exitstatus = contract_v5 ( v2, fp2, fp, fp3, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_v5, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
          EXIT(49);
        }

        /***********************************************************/
        /***********************************************************/

        sprintf(aff_tag, "/D-D/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/d4",
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_delta_list[if1], gamma_f1_delta_list[if2]);

        exitstatus = contract_v5 ( v2, fp, fp2, fp3, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_v5, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
          EXIT(49);
        }

        /***********************************************************/
        /***********************************************************/

        sprintf(aff_tag, "/D-D/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/d6",
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_delta_list[if1], gamma_f1_delta_list[if2]);

        exitstatus = contract_v6 ( v2, fp, fp2, fp3, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_v6, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
          EXIT(49);
        }

        /***********************************************************/
        /***********************************************************/

        /* fp3 <- fp3 x Gamma_i1 = Gamma_f1 x fp x Gamma_i1 */
        fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_delta_list[if1], fp3, VOLUME );
        fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp3, fp3, gamma_f1_delta_src_sign[if1], VOLUME );

        /***********************************************************/
        /***********************************************************/

        sprintf(aff_tag, "/D-D/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/d3",
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_delta_list[if1], gamma_f1_delta_list[if2]);

        exitstatus = contract_v5 ( v2, fp, fp3, fp, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_v5, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
          EXIT(49);
        }

        /***********************************************************/
        /***********************************************************/

        sprintf(aff_tag, "/D-D/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/d5",
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_delta_list[if1], gamma_f1_delta_list[if2]);

        exitstatus = contract_v6 ( v2, fp, fp3, fp, VOLUME );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_v6, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
          EXIT(48);
        }

        exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
          EXIT(49);
        }
      }}  /* end of loop of g_f1_delta, g_i1_delta */

      free_fp_field ( &fp3 );
      fini_2level_buffer ( &v2 );
      fini_3level_buffer ( &vp );

      /*****************************************************************/
      /*****************************************************************/

      /*****************************************************************
       * contraction for pion - pion and rho - rho 2-point function
       *****************************************************************/

      exitstatus= init_2level_buffer ( &v3, VOLUME, 2 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 2 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      /*****************************************************************/
      /*****************************************************************/

      /*****************************************************************
       * contractions for the charged pion - pion correlator
       *****************************************************************/
      sprintf(aff_tag, "/m-m/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d", gsx[0], gsx[1], gsx[2], gsx[3], 5, 5 );

      contract_twopoint_xdep (v3[0], 5, 5, &(propagator_list_up[i_coherent * n_s*n_c]), &(propagator_list_up[i_coherent * n_s*n_c]), n_c, 1, 1., 64);

      exitstatus = contract_vn_momentum_projection ( vp, v3, 1, g_sink_momentum_list, g_sink_momentum_number);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
        EXIT(48);
      }

      exitstatus = contract_vn_write_aff ( vp, 1, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
        EXIT(49);
      }

      /*****************************************************************/
      /*****************************************************************/

      /*****************************************************************
       * contractions for the neutral rho - rho correlator
       *****************************************************************/
      for ( int igi = 0; igi < gamma_rho_number; igi++ ) {
        for ( int igf = 0; igf < gamma_rho_number; igf++ ) {

          sprintf(aff_tag, "/m-m/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d", gsx[0], gsx[1], gsx[2], gsx[3], gamma_rho_list[igi],  gamma_rho_list[igf] );

          contract_twopoint_xdep (v3[0], gamma_rho_list[igi], gamma_rho_list[igf], &(propagator_list_up[i_coherent * n_s*n_c]), &(propagator_list_dn[i_coherent * n_s*n_c]), n_c, 1, 1., 64);

          exitstatus = contract_vn_momentum_projection ( vp, v3, 1, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 1, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
        }  /* end of loop on gamma rho at sink */
      }  /* end of loop on gamma rho at source */

      fini_2level_buffer ( &v3 );
      fini_3level_buffer ( &vp );


      /***********************************************************/
      /***********************************************************/


      /***********************************************************
       * contractions with up and dn propagator and stochastic source
       ***********************************************************/
 
      exitstatus= init_2level_buffer ( &v3, VOLUME, 24 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 24 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      /* up propagator as propagator type field */
      assign_fermion_propagator_from_spinor_field ( fp,  &(propagator_list_up[i_coherent * n_s*n_c]), VOLUME);

      /* dn propagator as propagator type field */
      assign_fermion_propagator_from_spinor_field ( fp2, &(propagator_list_dn[i_coherent * n_s*n_c]), VOLUME);

      /* loop on gamma structures at vertex f2 */
      for ( int i = 0; i < gamma_f2_number; i++ ) {

        /* loop on samples */
        for ( int i_sample = 0; i_sample < g_nsample; i_sample++ ) {

          /* multiply with Dirac structure at vertex f2 */
          spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f2_list[i], stochastic_source_list[i_sample], VOLUME );
          spinor_field_ti_eq_re ( spinor_work[0], gamma_f2_adjoint_sign[i], VOLUME);

          /*****************************************************************
           * xi - gf2 - u
           *****************************************************************/
          sprintf(aff_tag, "/v3/t%.2dx%.2dy%.2dz%.2d/xi-g%.2d-u/sample%.2d", 
              gsx[0], gsx[1], gsx[2], gsx[3],
              gamma_f2_list[i], i_sample);

          exitstatus = contract_v3  ( v3, spinor_work[0], fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

#if 0
          /*****************************************************************
           * xi - gf2 - d
           *****************************************************************/
          sprintf(aff_tag, "/v3/t%.2dx%.2dy%.2dz%.2d/xi-g%.2d-d/sample%.2d", 
              gsx[0], gsx[1], gsx[2], gsx[3],
              gamma_f2_list[i], i_sample);

          exitstatus = contract_v3  ( v3, stochastic_source_list[i_sample], fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#endif  /* of if 0 */

          /*****************************************************************
           * phi - gf2 - u
           *****************************************************************/
          /* multiply with Dirac structure at vertex f2 */
          spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f2_list[i], stochastic_propagator_list[i_sample], VOLUME );
          spinor_field_ti_eq_re ( spinor_work[0], gamma_f2_adjoint_sign[i], VOLUME);

#if 0
          /*****************************************************************/
          /*****************************************************************/

          sprintf(aff_tag, "/v3/t%.2dx%.2dy%.2dz%.2d/phi-g%.2d-u/sample%.2d", 
              gsx[0], gsx[1], gsx[2], gsx[3],
              gamma_f2_list[i], i_sample);

          exitstatus = contract_v3  ( v3, spinor_work[0], fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

#endif  /* of if 0 */


          /*****************************************************************
           * phi - gf2 - d
           *****************************************************************/
          sprintf(aff_tag, "/v3/t%.2dx%.2dy%.2dz%.2d/phi-g%.2d-d/sample%.2d", 
              gsx[0], gsx[1], gsx[2], gsx[3],
              gamma_f2_list[i], i_sample);

          exitstatus = contract_v3  ( v3, spinor_work[0], fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#if 0
#endif  /* of if 0 */

        }   /* end of loop on samples */
      }  /* end of loop on gf2 */

      fini_2level_buffer ( &v3 );
      fini_3level_buffer ( &vp );

      /*****************************************************************/
      /*****************************************************************/

      exitstatus= init_2level_buffer ( &v1, VOLUME, 72 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_2level_buffer ( &v2, VOLUME, 384 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 384 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      for ( int i = 0; i < gamma_f1_nucleon_number; i++ ) {

        for ( int i_sample = 0; i_sample < g_nsample; i_sample++ ) {

#if 0
          /* multiply with Dirac structure at vertex f2 */
          spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f1_nucleon_list[i], stochastic_source_list[i_sample], VOLUME );
          spinor_field_ti_eq_re ( spinor_work[0], gamma_f1_nucleon_sign[i], VOLUME);

          /*****************************************************************
           * xi - gf1 - u
           *****************************************************************/
          exitstatus = contract_v1 ( v1, spinor_work[0], fp, VOLUME  );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
            EXIT(47);
          }
#endif  /* of if 0 */
#if 0
          /*****************************************************************
           * xi - gf1 - u - u
           *****************************************************************/
          sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/xi-g%.2d-u-u/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_nucleon_list[i], i_sample);

          exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /*****************************************************************
           * xi - gf1 - u - d
           *****************************************************************/
          sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/xi-g%.2d-u-d/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_nucleon_list[i], i_sample);

          exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /*****************************************************************/
          /*****************************************************************/

          /*****************************************************************
           * xi - gf1 - d
           *****************************************************************/
          exitstatus = contract_v1 ( v1, spinor_work[0], fp2, VOLUME  );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
            EXIT(47);
          }

          /*****************************************************************
           * xi - gf1 - d - u
           *****************************************************************/
          sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/xi-g%.2d-d-u/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_nucleon_list[i], i_sample);

          exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /*****************************************************************
           * xi - gf1 - d - d
           *****************************************************************/
          sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/xi-g%.2d-d-d/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_nucleon_list[i], i_sample);

          exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#endif  /* of if 0 */

          /*****************************************************************/
          /*****************************************************************/

          /* multiply with Dirac structure at vertex f1 */
          spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f1_nucleon_list[i], stochastic_propagator_list[i_sample], VOLUME );
          spinor_field_ti_eq_re ( spinor_work[0], gamma_f1_nucleon_sign[i], VOLUME);


          /*****************************************************************
           * phi - gf1 - u
           *****************************************************************/
          exitstatus = contract_v1 ( v1, spinor_work[0], fp, VOLUME  );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
            EXIT(47);
          }

          /*****************************************************************
           * phi - gf1 - u - u
           *****************************************************************/
          sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/phi-g%.2d-u-u/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_nucleon_list[i], i_sample);

          exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

#if 0
          /*****************************************************************
           * phi - gf1 - u - d
           *****************************************************************/
          sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/phi-g%.2d-u-d/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_nucleon_list[i], i_sample);

          exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#endif  /* of if 0 */

          /*****************************************************************/
          /*****************************************************************/
#if 0
          /*****************************************************************
           * phi - gf1 - d
           *****************************************************************/
          exitstatus = contract_v1 ( v1, spinor_work[0], fp2, VOLUME  );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
            EXIT(47);
          }
#endif  /* of if 0 */

#if 0
          /*****************************************************************
           * phi - gf1 - d - u
           *****************************************************************/
          sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/phi-g%.2d-d-u/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_nucleon_list[i], i_sample);

          exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#endif  /* of if 0 */

#if 0
          /*****************************************************************
           * phi - gf1 - d - d
           *****************************************************************/
          sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/phi-g%.2d-d-d/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_nucleon_list[i], i_sample);

          exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#endif  /* of if 0 */

        }  /* end of loop on samples */
      }  /* end of loop on gf1  */

      fini_2level_buffer ( &v1 );
      fini_2level_buffer ( &v2 );
      fini_3level_buffer ( &vp );
      free_fp_field ( &fp  );
      free_fp_field ( &fp2 );

    } /* end of loop on coherent source timeslices */

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * sequential propagator part
     ***********************************************************/

    /* loop on sequential source momenta */
    for( int iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {

      /***********************************************************
       * sequential propagator U^{-1} g5 exp(ip) D^{-1}: tfii
       ***********************************************************/
      if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_factorized] sequential inversion fpr pi2 = (%d, %d, %d)\n", 
      g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);

      double **prop_list = (double**)malloc(g_coherent_source_number * sizeof(double*));
      if(prop_list == NULL) {
        fprintf(stderr, "[piN2piN_factorized] Error from malloc\n");
        EXIT(43);
      }

      ratime = _GET_TIME;
      for( int is=0;is<n_s*n_c;is++) {

        /* extract spin-color source-component is from coherent source dn propagators */
        for( int i=0; i<g_coherent_source_number; i++) {
          if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_factorized] using dn prop id %d / %d\n", (i_src * g_coherent_source_number + i), (i_src * g_coherent_source_number + i)*n_s*n_c + is);
          prop_list[i] = propagator_list_dn[i * n_s*n_c + is];
        }

        /* build sequential source */
        exitstatus = init_coherent_sequential_source(spinor_work[0], prop_list, g_source_coords_list[i_src][0], g_coherent_source_number, g_seq_source_momentum_list[iseq_mom], 5);
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_coherent_sequential_source, status was %d\n", exitstatus);
          EXIT(14);
        }

        /* source-smear the coherent source */
        exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);

        /* tm-rotate sequential source */
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
        }

        memset(spinor_work[1], 0, sizeof_spinor_field);
        /* invert */
        exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], op_id_up, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN_factorized] Error from tmLQCD_invert, status was %d\n", exitstatus);
          EXIT(12);
        }

        if ( check_propagator_residual ) {
          check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_up], 1 );
        }

        /* tm-rotate at sink */
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
        }

        /* sink-smear the coherent-source propagator */
        exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);

        memcpy( sequential_propagator_list[is], spinor_work[1], sizeof_spinor_field);

      }  /* end of loop on spin-color component */
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_factorized] time for seq propagator = %e seconds\n", retime-ratime);

      free(prop_list);
 
      /***********************************************/
      /***********************************************/

      /***********************************************
       * contractions involving sequential propagator
       ***********************************************/

      double **v1 = NULL, **v2 = NULL, **v3 = NULL, ***vp = NULL;
      fermion_propagator_type *fp = NULL;
      fp  = create_fp_field ( VOLUME );

      exitstatus= init_2level_buffer ( &v3, VOLUME, 24 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 24 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      /* sequential propagator as propagator type field */
      assign_fermion_propagator_from_spinor_field ( fp, sequential_propagator_list, VOLUME);

      for ( int i = 0; i < gamma_f2_number; i++ ) {

        for ( int i_sample = 0; i_sample < g_nsample; i_sample++ ) {


          /*****************************************************************
           * xi^+ - gf2 - ud
           *****************************************************************/

          /* spinor_work <- Gamma_f2^+ stochastic source */
          spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f2_list[i], stochastic_source_list[i_sample], VOLUME );
          spinor_field_ti_eq_re ( spinor_work[0], gamma_f2_adjoint_sign[i], VOLUME);

          sprintf(aff_tag, "/v3/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-ud/sample%.2d", 
              g_source_coords_list[i_src][0],
              g_source_coords_list[i_src][1],
              g_source_coords_list[i_src][2],
              g_source_coords_list[i_src][3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f2_list[i], i_sample);

          exitstatus = contract_v3  ( v3, spinor_work[0], fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v3, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#if 0
          /*****************************************************************
           * phi^+ - gf2 - ud
           *****************************************************************/
          /* spinor_work <- Gamma_f2^+ stochastic propagator */
          spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f2_list[i], stochastic_propagator_list[i_sample], VOLUME );
          spinor_field_ti_eq_re ( spinor_work[0], gamma_f2_adjoint_sign[i], VOLUME);

          sprintf(aff_tag, "/v3/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-ud/sample%.2d", 
              g_source_coords_list[i_src][0],
              g_source_coords_list[i_src][1],
              g_source_coords_list[i_src][2],
              g_source_coords_list[i_src][3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f2_list[i], i_sample);

          exitstatus = contract_v3  ( v3, spinor_work[0], fp, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v3, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
#endif

        }  /* end of loop on samples */
      }  /* end of loop on gf2  */

      fini_2level_buffer ( &v3 );
      fini_3level_buffer ( &vp );
      free_fp_field ( &fp );

      /***********************************************/
      /***********************************************/

      /* loop on coherent source locations */
      for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
        int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

        gsx[0] = t_coherent;
        gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
        gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
        gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

        get_point_source_info (gsx, sx, &source_proc_id);


        /***********************************************/
        /***********************************************/

        double **v1 = NULL, **v2 = NULL, **v3 = NULL, ***vp = NULL;
        fermion_propagator_type *fp = NULL, *fp2 = NULL, *fp3 = NULL, *fp4 = NULL;
        fp  = create_fp_field ( VOLUME );
        fp2 = create_fp_field ( VOLUME );
        fp3 = create_fp_field ( VOLUME );
        fp4 = create_fp_field ( VOLUME );

        /* sequential propagator as propagator type field */
        assign_fermion_propagator_from_spinor_field ( fp, sequential_propagator_list, VOLUME);

        /* up propagator as propagator type field */
        assign_fermion_propagator_from_spinor_field ( fp2,  &(propagator_list_up[i_coherent * n_s*n_c]), VOLUME);

        /***********************************************************/
        /***********************************************************/

        exitstatus= init_2level_buffer ( &v2, VOLUME, 32 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 32 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        /***********************************************************
         * contractions for pi N - D with up sequential propagagor
         ***********************************************************/
        for ( int if1 = 0; if1 < gamma_f1_nucleon_number; if1++ ) {

          /* fp3 <- fp x Gamma_i1 */
          fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_nucleon_list[if1], fp, VOLUME );
          fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp3, fp3, -gamma_f1_nucleon_sign[if1], VOLUME );

        for ( int if2 = 0; if2 < gamma_f1_delta_number; if2++ ) {

          /* fp4 <- Gamma_f2 x fp2 */
          fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp4, gamma_f1_delta_list[if2], fp2, VOLUME );
          fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp4, fp4, gamma_f1_delta_snk_sign[if2], VOLUME );

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t1",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v5 ( v2, fp3, fp4, fp2, VOLUME );
          if ( exitstatus != 0 ) {
           fprintf(stderr, "[piN2piN_factorized] Error from contract_v5, status was %d\n", exitstatus);
             EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t2",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v5 ( v2, fp3, fp2, fp4, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t4",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v5 ( v2, fp2, fp3, fp4, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t6",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v6 ( v2, fp2, fp4, fp3, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v6, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***********************************************************/
          /***********************************************************/

          fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp4, gamma_f1_delta_list[if2], fp3, VOLUME );
          fermion_propagator_field_eq_fermion_propagator_field_ti_re (fp4, fp4, gamma_f1_delta_snk_sign[if2], VOLUME );

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t3",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v5 ( v2, fp2, fp4, fp2, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v5, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          /***********************************************************/
          /***********************************************************/

          sprintf(aff_tag, "/piN-D/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/t5",
              gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              gamma_f1_nucleon_list[if1], gamma_f1_delta_list[if2]);

          exitstatus = contract_v6 ( v2, fp2, fp2, fp4, VOLUME );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_v6, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_momentum_projection ( vp, v2, 16, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 16, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

        }}  /* end of gf1_delta, gf1_nucleon */

        fini_2level_buffer ( &v2 );
        fini_3level_buffer ( &vp );
        free_fp_field ( &fp4 );

        /*****************************************************************/
        /*****************************************************************/

        /*****************************************************************
         * contraction for pi x pi - rho 2-point function
         *****************************************************************/

        exitstatus= init_2level_buffer ( &v3, VOLUME, 2 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 2 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        /*****************************************************************/
        /*****************************************************************/

        /*****************************************************************
         * contractions for the pi^+ pi^- rho^0 correlator
         *****************************************************************/
        for ( int igf = 0; igf < gamma_rho_number; igf++ ) {

          sprintf(aff_tag, "/mxm-m/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              5,  gamma_rho_list[igf] );

          contract_twopoint_xdep (v3[0], 5, gamma_rho_list[igf], sequential_propagator_list, &(propagator_list_dn[i_coherent * n_s*n_c]), n_c, 1, 1., 64);

          exitstatus = contract_vn_momentum_projection ( vp, v3, 1, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 1, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }
        }  /* end of loop on gamma rho at sink */

        fini_2level_buffer ( &v3 );
        fini_3level_buffer ( &vp );

        /*****************************************************************/
        /*****************************************************************/

        exitstatus= init_2level_buffer ( &v1, VOLUME, 72 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_2level_buffer ( &v2, VOLUME, 384 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 384 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        assign_fermion_propagator_from_spinor_field ( fp,  sequential_propagator_list, VOLUME);
        assign_fermion_propagator_from_spinor_field ( fp2,  &(propagator_list_up[i_coherent * n_s*n_c]), VOLUME);
        assign_fermion_propagator_from_spinor_field ( fp3,  &(propagator_list_dn[i_coherent * n_s*n_c]), VOLUME);

        for ( int i = 0; i < gamma_f1_nucleon_number; i++ ) {
          for ( int i_sample = 0; i_sample < g_nsample; i_sample++ ) {
  
#if 0
            /* multiply with Dirac structure at vertex f1 */
            spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f1_nucleon_list[i], stochastic_source_list[i_sample], VOLUME );
            spinor_field_ti_eq_re ( spinor_work[0], gamma_f1_nucleon_sign[i], VOLUME);


            /*****************************************************************
             * xi - gf1 - ud 
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (1) xi - gf1 - ud - u
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-ud-u/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3], 
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

            /*****************************************************************
             * (2) xi - gf1 - ud - d
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-ud-d/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp3, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

            /*****************************************************************/
            /*****************************************************************/

            /*****************************************************************
             * xi - gf1 - u
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp2, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (3) xi - gf1 - u - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-u-ud/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

            /*****************************************************************/
            /*****************************************************************/
  
            /*****************************************************************
             * xi - gf1 - d
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp3, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (4) xi - gf1 - d - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-d-ud/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
#endif  /* of if 0 */

            /*****************************************************************/
            /*****************************************************************/

            /* multiply with Dirac structure at vertex f1 */
            spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f1_nucleon_list[i], stochastic_propagator_list[i_sample], VOLUME );
            spinor_field_ti_eq_re ( spinor_work[0], gamma_f1_nucleon_sign[i], VOLUME);
  
            /*****************************************************************
             * phi - gf1 - ud
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (1) phi - gf1 - ud - u
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-ud-u/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp2, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

#if 0
            /*****************************************************************
             * (2 )phi - gf1 - ud - d
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-ud-d/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp3, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
  
#endif  /* end of if 0  */

            /*****************************************************************/
            /*****************************************************************/
  
            /*****************************************************************
             * phi - gf1 - u
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp2, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (3) phi - gf1 - u - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-u-ud/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
  
            /*****************************************************************/
            /*****************************************************************/
  
#if 0
            /*****************************************************************
             * phi - gf1 - d
             *****************************************************************/
            exitstatus = contract_v1 ( v1, spinor_work[0], fp3, VOLUME  );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            /*****************************************************************
             * (4) phi - gf1 - d - ud
             *****************************************************************/
            sprintf(aff_tag, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-d-ud/sample%.2d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                gamma_f1_nucleon_list[i], i_sample);
  
            exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v2_from_v1, status was %d\n", exitstatus);
              EXIT(47);
            }
  
            exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }
  
            exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
#endif  /* end of if 0  */

          }  /* end of loop on samples */
        }  /* end of loop on gf1  */

        fini_2level_buffer ( &v1 );
        fini_2level_buffer ( &v2 );
        fini_3level_buffer ( &vp );


        free_fp_field ( &fp  );
        free_fp_field ( &fp2 );
        free_fp_field ( &fp3 );

      }  /* end of loop on coherent source timeslices */

    }  /* end of loop on sequential momentum list */

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_factorized] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(111);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */
  }  /* end of loop on base source locations */

  fini_2level_buffer ( &sequential_propagator_list );
  fini_2level_buffer ( &propagator_list_up );
  fini_2level_buffer ( &propagator_list_dn );

  /***********************************************************/
  /***********************************************************/

  }  /* end of if do_contraction_std */

  /***********************************************************/
  /***********************************************************/

  fini_2level_buffer ( &stochastic_propagator_list );
  fini_2level_buffer ( &stochastic_source_list );

  /***********************************************************/
  /***********************************************************/

  if ( do_contraction_oet ) {

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   ***********************************************
   **
   ** stochastic contractions using the 
   **   one-end-trick
   **
   ***********************************************
   ***********************************************/
  exitstatus = init_2level_buffer ( &stochastic_propagator_list, 4, _GSI(VOLUME) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &stochastic_propagator_zero_list, 4, _GSI(VOLUME) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &stochastic_source_list, 4, _GSI(VOLUME) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &propagator_list_up, 12, _GSI(VOLUME) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }

  exitstatus = init_2level_buffer ( &propagator_list_dn, 12, _GSI(VOLUME) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
    EXIT(48);
  }


  /* loop on base source locations */
  for( int i_src=0; i_src < g_source_location_number; i_src++) {

    /* base source timeslice */
    int t_base = g_source_coords_list[i_src][0];

#ifdef HAVE_LHPC_AFF
    /***********************************************
     * open aff output file
     ***********************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN_oet", Nconf, t_base );
      fprintf(stdout, "# [piN2piN_factorized] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_factorized] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      }
    }  /* end of if io_proc == 2 */
#endif

    /* loop on coherent source locations */
    for(int i_coherent = 0; i_coherent < g_coherent_source_number; i_coherent++) {

      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;
      gsx[0] = t_coherent;
      gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
      gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
      gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

      exitstatus = point_source_propagator ( propagator_list_up, gsx, op_id_up, 1, 1, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, mzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_factorized] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(12);
      }

      exitstatus = point_source_propagator ( propagator_list_dn, gsx, op_id_dn, 1, 1, gauge_field_smeared, check_propagator_residual, gauge_field_with_phase, mzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_factorized] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(12);
      }

      double **v1 = NULL, **v2 = NULL;
      double **v3 = NULL, ***vp = NULL;
      fermion_propagator_type *fp = NULL, *fp2 = NULL, *fp3 = NULL;
      fp  = create_fp_field ( VOLUME );
      fp2 = create_fp_field ( VOLUME );
      fp3 = create_fp_field ( VOLUME );

      /* fp  <- up propagator as fermion_propagator_type */
      assign_fermion_propagator_from_spinor_field ( fp,  propagator_list_up, VOLUME);
      /* fp2 <- dn propagator as fermion_propagator_type */
      assign_fermion_propagator_from_spinor_field ( fp2, propagator_list_dn, VOLUME);

      /******************************************************
       * re-initialize random number generator
       ******************************************************/
      if ( ! read_stochastic_source_oet ) {
        sprintf(filename, "rng_stat.%.4d.tsrc%.3d.stochastic-oet.out", Nconf, gsx[0]);
        exitstatus = init_rng_stat_file ( ( ( gsx[0] + 1 ) * 10000 + g_seed ), filename );
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_rng_stat_file status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(38);
        }
      }

      /* loop on oet samples */
      for( int isample=0; isample < g_nsample_oet; isample++) {

        /*****************************************************************
         * read stochastic oet source from file
         *****************************************************************/
        if ( read_stochastic_source_oet ) {
          for ( int ispin = 0; ispin < 4; ispin++ ) {
            sprintf(filename, "%s-oet.%.4d.t%.2d.%.2d.%.5d", filename_prefix, Nconf, gsx[0], ispin, isample);
            if ( ( exitstatus = read_lime_spinor( stochastic_source_list[ispin], filename, 0) ) != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from read_lime_spinor, status was %d\n", exitstatus);
              EXIT(2);
            }
          }
          /* recover the random field */
          if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gsx[0], NULL, -1 ) ) != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(64);
          }

        /*****************************************************************
         * generate stochastic oet source
         *****************************************************************/
        } else {
          /* dummy call to initialize the ran field, we do not use the resulting stochastic_source_list */
          if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gsx[0], NULL, 1 ) ) != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(64);
          }
          if ( write_stochastic_source_oet ) {
            for ( int ispin = 0; ispin < 4; ispin++ ) {
              sprintf(filename, "%s-oet.%.4d.t%.2d.%.2d.%.5d", filename_prefix, Nconf, gsx[0], ispin, isample);
              if ( ( exitstatus = write_propagator( stochastic_source_list[ispin], filename, 0, 64) ) != 0 ) {
                fprintf(stderr, "[piN2piN_factorized] Error from write_propagator, status was %d\n", exitstatus);
                EXIT(2);
              }
            }
          }
        }  /* end of if read stochastic source - else */

        /*****************************************************************
         * invert for stochastic timeslice propagator at zero momentum
         *****************************************************************/
        for( int i = 0; i < 4; i++) {
          memcpy(spinor_work[0], stochastic_source_list[i], sizeof_spinor_field);

          /* source-smearing stochastic momentum source */
          if ( ( exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi) ) != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(5);
          }

          /* tm-rotate stochastic source */
          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
          }

          memset(spinor_work[1], 0, sizeof_spinor_field);

          exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], op_id_up, 0);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN_factorized] Error from tmLQCD_invert, status was %d\n", exitstatus);
            EXIT(44);
          }

          if ( check_propagator_residual ) {
            check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_up], 1 );
          }

          /* tm-rotate stochastic propagator at sink */
          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
          }

          /* sink smearing stochastic propagator */
          if ( ( exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi) ) != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(5);
          }

          memcpy( stochastic_propagator_zero_list[i], spinor_work[1], sizeof_spinor_field);
        }


        /*****************************************************************
         * calculate V3
         *
         * phi^+ g5 Gamma_f2 ( pf2 ) U ( z_1xi )
         * phi^+ g5 Gamma_f2 ( pf2 ) D
         *****************************************************************/

        exitstatus= init_2level_buffer ( &v3, VOLUME, 24 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 24 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        for ( int if2 = 0; if2 < gamma_f2_number; if2++ ) {

          for( int ispin = 0; ispin < 4; ispin++ ) {

            /*****************************************************************
             * (1) phi - gf2 - u
             *****************************************************************/

            /* spinor_work <- Gamma_f2^+ x stochastic_propagator, up to sign */
            spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f2_list[if2], stochastic_propagator_zero_list[ispin], VOLUME );
            /* spinor_work <- g5 spinor_work */
            g5_phi( spinor_work[0], VOLUME );
            /* spinor_work <- spinor_work x sign for Gamma_f2^+ */
            spinor_field_ti_eq_re ( spinor_work[0], gamma_f2_g5_adjoint_sign[if2], VOLUME);

            sprintf(aff_tag, "/v3-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-u/sample%.2d/d%d", gsx[0], gsx[1], gsx[2], gsx[3],
                0, 0, 0, gamma_f2_list[if2], isample, ispin);

            exitstatus = contract_v3 ( v3, spinor_work[0], fp, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v3, status was %d\n", exitstatus);
              EXIT(47);
            }
 
            exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }

            exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }

#if 0
            /*****************************************************************/
            /*****************************************************************/

            /*****************************************************************
             * (2) phi - gf2 - d
             *****************************************************************/
            sprintf(aff_tag, "/v3-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-d/sample%.2d/d%d", gsx[0], gsx[1], gsx[2], gsx[3],
                0, 0, 0,
                gamma_f2_list[if2], isample, ispin);

            exitstatus = contract_v3 ( v3, spinor_work[0], fp2, VOLUME );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_v3, status was %d\n", exitstatus);
              EXIT(47);
            }
 
            exitstatus = contract_vn_momentum_projection ( vp, v3, 12, g_sink_momentum_list, g_sink_momentum_number);
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
              EXIT(48);
            }

            exitstatus = contract_vn_write_aff ( vp, 12, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
              EXIT(49);
            }
#endif  /* of if 0 */
          }  /* end of loop on spin components ispin */

        }  /* end of loop on gamma_f2_list */

        fini_2level_buffer ( &v3 );
        fini_3level_buffer ( &vp );

        /*****************************************************************/
        /*****************************************************************/

        /*****************************************************************
         * loop on sequential source momenta p_i2
         *****************************************************************/
        for( int iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {

          int seq_source_momentum[3] = { g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2] };

          if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gsx[0], seq_source_momentum, 0 ) ) != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from init_timeslice_source_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(64);
          }

          /*****************************************************************/
          /*****************************************************************/

          /*****************************************************************
           * invert for stochastic timeslice propagator 
           *   with sequential * momentum p_i2
           *****************************************************************/
          for( int i = 0; i < 4; i++) {
            memcpy(spinor_work[0], stochastic_source_list[i], sizeof_spinor_field);

            /* source-smearing stochastic momentum source */
            if ( ( exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi) ) != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(12);
            }

            /* tm-rotate stochastic source */
            if( g_fermion_type == _TM_FERMION ) {
              spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
            }

            memset(spinor_work[1], 0, sizeof_spinor_field);

            exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], op_id_up, 0);
            if(exitstatus != 0) {
              fprintf(stderr, "[piN2piN_factorized] Error from tmLQCD_invert, status was %d\n", exitstatus);
              EXIT(44);
            }

            if ( check_propagator_residual ) {
              check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_up], 1 );
            }

            /* tm-rotate stochastic propagator at sink */
            if( g_fermion_type == _TM_FERMION ) {
              spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
            }

            /* sink smearing stochastic propagator */
            exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);

            memcpy( stochastic_propagator_list[i], spinor_work[1], sizeof_spinor_field);
          }

          /*****************************************************************/
          /*****************************************************************/

          /*****************************************************************
           * contraction for pion - pion 2-point function
           *****************************************************************/

          exitstatus= init_2level_buffer ( &v3, VOLUME, 2 );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
            EXIT(47);
          }

          exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 2 );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
            EXIT(47);
          }

          sprintf(aff_tag, "/m-m/t%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/sample%.2d/gi%.2d/gf%.2d", gsx[0],
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
              isample, 5, 5);

          contract_twopoint_xdep (v3[0], 5, 5, stochastic_propagator_zero_list, stochastic_propagator_list, 1, 1, 1., 64);

          exitstatus = contract_vn_momentum_projection ( vp, v3, 1, g_sink_momentum_list, g_sink_momentum_number);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
            EXIT(48);
          }

          exitstatus = contract_vn_write_aff ( vp, 1, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
            EXIT(49);
          }

          fini_2level_buffer ( &v3 );
          fini_3level_buffer ( &vp );


          /*****************************************************************/
          /*****************************************************************/

          /*****************************************************************
           * calculate V2 and V4
           *
           * z_1phi = V4
           * z_3phi = V2
           *****************************************************************/

            exitstatus= init_2level_buffer ( &v1, VOLUME, 72 );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
              EXIT(47);
            }

            exitstatus= init_2level_buffer ( &v2, VOLUME, 384 );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from init_2level_buffer, status was %d\n", exitstatus);
              EXIT(47);
            }

            exitstatus= init_3level_buffer ( &vp, T, g_sink_momentum_number, 384 );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_factorized] Error from init_3level_buffer, status was %d\n", exitstatus);
              EXIT(47);
            }


            for( int if1 = 0; if1 < gamma_f1_nucleon_number; if1++ ) {

              for( int ispin = 0; ispin < 4; ispin++ ) {

                /* spinor_work <- Gamma_f1^t x stochastic propagator */
                spinor_field_eq_gamma_ti_spinor_field (spinor_work[0], gamma_f1_nucleon_list[if1], stochastic_propagator_list[ispin], VOLUME );
                spinor_field_ti_eq_re ( spinor_work[0], gamma_f1_nucleon_transposed_sign[if1], VOLUME);

                /*****************************************************************/
                /*****************************************************************/

                /*****************************************************************
                 * phi - gf1 - d
                 *****************************************************************/
                exitstatus = contract_v1 ( v1, spinor_work[0], fp2, VOLUME  );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_factorized] Error from contract_v1, status was %d\n", exitstatus);
                  EXIT(47);
                }
  
                /*****************************************************************
                 * (3) phi - gf1 - d - u
                 *****************************************************************/
                sprintf(aff_tag, "/v2-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-d-u/sample%.2d/d%d", gsx[0], gsx[1], gsx[2], gsx[3],
                    g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                    gamma_f1_nucleon_list[if1], isample, ispin);

                exitstatus = contract_v2_from_v1 ( v2, v1, fp, VOLUME );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_factorized] Error from contract_v4, status was %d\n", exitstatus);
                  EXIT(47);
                }

                exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
                  EXIT(48);
                }

                exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
                  EXIT(49);
                }

              }  /* end of loop on ispin */

              /*****************************************************************/
              /*****************************************************************/

              /* fp3 <- Gamma_f1 x fp2 = Gamma_i1 x D */
              fermion_propagator_field_eq_gamma_ti_fermion_propagator_field (fp3, gamma_f1_nucleon_list[if1], fp2, VOLUME );
              fermion_propagator_field_eq_fermion_propagator_field_ti_re ( fp3, fp3, gamma_f1_nucleon_sign[if1], VOLUME );


              for( int ispin = 0; ispin < 4; ispin++ ) {
                /*****************************************************************
                 * (4) phi - gf1 - d - u
                 *****************************************************************/
                sprintf(aff_tag, "/v4-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-d-u/sample%.2d/d%d", gsx[0], gsx[1], gsx[2], gsx[3],
                    g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                    gamma_f1_nucleon_list[if1], isample, ispin);

                exitstatus = contract_v4 ( v2, stochastic_propagator_list[ispin], fp3, fp, VOLUME );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_factorized] Error from contract_v4, status was %d\n", exitstatus);
                  EXIT(47);
                }

                exitstatus = contract_vn_momentum_projection ( vp, v2, 192, g_sink_momentum_list, g_sink_momentum_number);
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
                  EXIT(48);
                }

                exitstatus = contract_vn_write_aff ( vp, 192, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_factorized] Error from contract_vn_write_aff, status was %d\n", exitstatus);
                  EXIT(49);
                }

              }  /* end of loop on ispin */
            }  /* end of loop on gamma at f1 */


            fini_2level_buffer ( &v1 );
            fini_2level_buffer ( &v2 );
            fini_3level_buffer ( &vp );

        }  /* end of loop on sequential source momenta pi2 */

      } /* end of loop on oet samples */

      free_fp_field( &fp  );
      free_fp_field( &fp2 );
      free_fp_field( &fp3 );

    }  /* end of loop on coherent sources */

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_factorized] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(111);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */
  } /* end of loop on base sources */ 


  fini_2level_buffer ( &stochastic_propagator_list );
  fini_2level_buffer ( &stochastic_propagator_zero_list );
  fini_2level_buffer ( &stochastic_source_list );
  fini_2level_buffer ( &propagator_list_up );
  fini_2level_buffer ( &propagator_list_dn );

  /***********************************************************/
  /***********************************************************/

  }  /* end of if do_contraction_oet */

  /***********************************************************/
  /***********************************************************/


  /***********************************************
   * free gauge fields and spinor fields
   ***********************************************/

  fini_stochastic_source_timeslices ();

#ifndef HAVE_TMLQCD_LIBWRAPPER
  if(g_gauge_field != NULL) free(g_gauge_field);
#endif
  free ( spinor_work[0] );
  free ( spinor_work[1] );

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  if ( check_propagator_residual ) {
    fini_clover ();
    if( gauge_field_with_phase != NULL) free( gauge_field_with_phase );
  }

  free_geometry();

  if( gauge_field_smeared != NULL ) free(gauge_field_smeared);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  if ( check_propagator_residual ) mpi_fini_xchange_eo_spinor();
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_factorized] %s# [piN2piN_factorized] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_factorized] %s# [piN2piN_factorized] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
