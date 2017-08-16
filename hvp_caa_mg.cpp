/****************************************************
 * hvp_caa_mg.c
 *
 * Tue Aug 15 15:35:23 CEST 2017
 *
 * - originally copied from hvp_caa_lma.cpp
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
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "matrix_init.h"
#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -w                  : check position space WI [default false]\n");
  EXIT(0);
}

int dummy_eo_solver (double * const propagator, double * const source, const int op_id) {
  memcpy(propagator, source, _GSI(VOLUME)/2*sizeof(double) );
  return(0);
}


#ifdef DUMMY_SOLVER 
#  define _TMLQCD_INVERT_EO dummy_eo_solver
#else
#  define _TMLQCD_INVERT_EO tmLQCD_invert_eo
#endif

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "hvp_caa";

  int c;
  int iflavor;
  int filename_set = 0;
  int isource_location;
  unsigned int ix;
  int gsx[4], sx[4];
  int check_position_space_WI=0;
  int exitstatus;
  int source_proc_id = 0;
  int iseq_source_momentum;
  int isequential_source_gamma_id, isequential_source_timeslice;
  int no_eo_fields = 0;
  int io_proc = -1;
  int evecs_num = 0;
  int check_propagator_residual = 0;
  unsigned int Vhalf;
  size_t sizeof_eo_spinor_field;
  double **eo_spinor_field=NULL;
  double **eo_stochastic_source = NULL, ***eo_stochastic_propagator = NULL;
  double **cvc_tensor_eo = NULL, contact_term[2][8], *cvc_tensor_lexic=NULL;
  double ***cvc_tp = NULL;
  char filename[100];
  double ratime, retime;
  double plaq;
  double **mzz[2], **mzzinv[2];
  double *gauge_field_with_phase = NULL;
  double ***eo_source_buffer = NULL;



#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
  char aff_tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "cwh?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_position_space_WI = 1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) strcpy(filename, "hvp.input");
  /* fprintf(stdout, "# [hvp_caa_mg] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [hvp_caa_mg] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
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

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hvp_caa_mg] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hvp_caa_mg] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hvp_caa_mg] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[hvp_caa_mg] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  Vhalf = VOLUME / 2;
  sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [hvp_caa_mg] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [hvp_caa_mg] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[hvp_caa_mg] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[hvp_caa_mg] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[hvp_caa_mg] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[] Error from init_clover, status was %d\n");
    EXIT(1);
  }

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [hvp_caa_mg] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [hvp_caa_mg] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  if(io_proc == 2) {
    if(g_tr_id != 0) {
      fprintf(stderr, "[hvp_caa_mg] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************************
   * allocate eo_spinor_field
   ***********************************************************/
  exitstatus = init_2level_buffer ( &eo_spinor_field, 240, _GSI(Vhalf));
  if( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_caa_mg] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(123);
  }
  
  /***********************************************************
   ***********************************************************
   **
   ** loop on source locations
   **
   ***********************************************************
   ***********************************************************/
  for(isource_location=0; isource_location < g_source_location_number; isource_location++) {

    /***********************************************************
     * determine source coordinates, find out, if source_location is in this process
     ***********************************************************/
    gsx[0] = g_source_coords_list[isource_location][0];
    gsx[1] = g_source_coords_list[isource_location][1];
    gsx[2] = g_source_coords_list[isource_location][2];
    gsx[3] = g_source_coords_list[isource_location][3];

    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[hvp_caa_mg] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***********************************************************
     * init Usource and source_proc_id
     *
     * NOTE: here it must be g_gauge_field, 
     *       NOT gauge_field_with_phase,
     *       because the boundary phase is multiplied at
     *       source inside init_contract_cvc_tensor_usource
     ***********************************************************/
    init_contract_cvc_tensor_usource( gauge_field_with_phase, gsx, NULL);

#ifdef HAVE_LHPC_AFF
    /***********************************************
     ***********************************************
     **
     ** writer for aff output file
     **
     ***********************************************
     ***********************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [hvp_caa_mg] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[hvp_caa_mg] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#endif



    /**********************************************************
     **********************************************************
     **
     ** propagators with source at gsx and gsx + mu
     **
     **********************************************************
     **********************************************************/

    for( int mu = 0; mu < 5; mu++)
    {  /* loop on shifts in direction mu */
      /**********************************************************
       * shifted source coords and source proc coords
       **********************************************************/
      int g_shifted_source_coords[4], shifted_source_coords[4], shifted_source_proc_id = 0;
      memcpy( g_shifted_source_coords, gsx, 4*sizeof(int));
      switch(mu) {
        case 0: g_shifted_source_coords[0] = ( g_shifted_source_coords[0] + 1 ) % T_global; break;;
        case 1: g_shifted_source_coords[1] = ( g_shifted_source_coords[1] + 1 ) % LX_global; break;;
        case 2: g_shifted_source_coords[2] = ( g_shifted_source_coords[2] + 1 ) % LY_global; break;;
        case 3: g_shifted_source_coords[3] = ( g_shifted_source_coords[3] + 1 ) % LZ_global; break;;
      }

      /* source info for shifted source location */
      if( (exitstatus = get_point_source_info (g_shifted_source_coords, shifted_source_coords, &shifted_source_proc_id) ) != 0 ) {
        fprintf(stderr, "[hvp_caa_mg] Error from get_point_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(15);
      }

      /**********************************************************
       * up-type propagators
       **********************************************************/
      exitstatus = point_to_all_fermion_propagator_clover_full2eo ( &(eo_spinor_field[mu*12]), &(eo_spinor_field[60+12*mu]),  _OP_ID_UP,
          g_shifted_source_coords, gauge_field_with_phase, g_mzz_up, g_mzzinv_up, check_propagator_residual );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_mixed] Error from point_to_all_fermion_propagator_clover_eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(21);
      }

      /**********************************************************
       * dn-type propagators
       **********************************************************/
      exitstatus = point_to_all_fermion_propagator_clover_full2eo ( &(eo_spinor_field[120+mu*12]), &(eo_spinor_field[180+12*mu]),  _OP_ID_DN,
          g_shifted_source_coords, gauge_field_with_phase, g_mzz_dn, g_mzzinv_dn, check_propagator_residual );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_mixed] Error from point_to_all_fermion_propagator_clover_eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(21);
      }

    }  /* end of loop on shift direction mu */

    /* allocate memory for contractions, initialize */
    exitstatus = init_2level_buffer( &cvc_tensor_eo, 2, 32*Vhalf);
    if( exitstatus != 0) {
      fprintf(stderr, "[hvp_caa_mg] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(24);
    }

    /***************************************************************************
     * full tensor
     ***************************************************************************/
    memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
    memset(contact_term[0], 0, 8*sizeof(double));
    /* contraction */
    contract_cvc_tensor_eo ( cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[0], &(eo_spinor_field[120]), &(eo_spinor_field[180]),
       &(eo_spinor_field[0]), &(eo_spinor_field[60]), gauge_field_with_phase );

    /* subtract contact term */
    cvc_tensor_eo_subtract_contact_term (cvc_tensor_eo, contact_term[0], gsx, (int)( source_proc_id == g_cart_id ) );

    /* momentum projections */
    exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cvc_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0) {
      fprintf(stderr, "[hvp_caa_mg] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(26);
    }
    /* write results to file */
    sprintf(aff_tag, "/hvp/full/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = cvc_tensor_tp_write_to_aff_file ( cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[hvp_caa_mg] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }
    fini_3level_buffer(&cvc_tp);

    /* check position space WI */
    if(check_position_space_WI) {
      exitstatus = cvc_tensor_eo_check_wi_position_space ( cvc_tensor_eo );
      if(exitstatus != 0) {
        fprintf(stderr, "[hvp_caa_mg] Error from cvc_tensor_eo_check_wi_position_space for full, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }
    }

    fini_2level_buffer( &cvc_tensor_eo );

    /***************************************************************************/
    /***************************************************************************/

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[hvp_caa_mg] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */


  }  /* end of loop on source locations */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  fini_2level_buffer ( &eo_spinor_field );

  /* free clover matrix terms */
  fini_clover ();

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [hvp_caa_mg] %s# [hvp_caa_mg] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [hvp_caa_mg] %s# [hvp_caa_mg] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
