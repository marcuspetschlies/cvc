/****************************************************
 * p2gg.c
 *
 * Thu Nov 17 13:10:32 CET 2016
 *
 * - originally copied from p2gg_xspace.c
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
#include "project.h"
#include "matrix_init.h"

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


int main(int argc, char **argv) {
  
  /*
   * sign for g5 Gamma^\dagger g5
   *                                                0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   * */
  const int sequential_source_gamma_id_sign[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };

  const char outfile_prefix[] = "p2gg";

  int c, i, mu, nu;
  int iflavor;
  int filename_set = 0;
  int isource_location;
  int x0, x1, x2, x3;
  unsigned int ix;
  int gsx[4], sx[4];
  int check_position_space_WI=0;
  int exitstatus;
  int source_proc_coords[4], source_proc_id = 0;
  int shifted_source_coords[4], shifted_source_proc_coords[4], shifted_source_proc_id=0, shifted_sequential_source_timeslice;
  int seq_source_momentum[3], iseq_source_momentum;
  int sequential_source_gamma_id, isequential_source_gamma_id, isequential_source_timeslice;
  int no_eo_fields = 0, no_full_fields = 0;
  int io_proc = -1;
  int evecs_num = 0;
  int eo_spinor_field_id_e, eo_spinor_field_id_o, eo_seq_spinor_field_id_e, eo_seq_spinor_field_id_o;
  unsigned int Vhalf;
  size_t sizeof_eo_spinor_field;
  double **eo_spinor_field=NULL, **eo_spinor_work=NULL, *eo_evecs_block=NULL, *eo_sample_block=NULL,   **eo_sample_field=NULL;
  double **full_spinor_field=NULL, **eo_evecs_field=NULL;
  double *cvc_tensor[2] = {NULL, NULL}, contact_term[2][8], ***cvc_tp;
  double *evecs_eval = NULL;
  double *fwd_list_eo[2][5][12], *bwd_list_eo[2][5][12];
  char filename[100];
  double ratime, retime;
  double plaq;
  double **mzz[2], **mzzinv[2];


#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex *aff_buffer = NULL;
  char aff_buffer_path[200];
  /*  uint32_t aff_buffer_size; */
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, "wh?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_position_space_WI = 1;
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
  if(filename_set==0) strcpy(filename, "p2gg.input");
  fprintf(stdout, "# [p2gg] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
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
  mpi_init_xchange_contraction(32);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg] Error from init_geometry\n");
    EXIT(4);
  }

  geometry();

  Vhalf = VOLUME / 2;
  sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [p2gg] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(6);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg] Error, &g_gauge_field is NULL\n");
    EXIT(7);
  }
#endif

#ifdef HAVE_MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [p2gg] measured plaquette value: %25.16e\n", plaq);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[p2gg] Error from tmLQCD_init_deflator, status was %d\n", exitstatus);
    EXIT(8);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg] Error from tmLQCD_get_deflator_params, status was %d\n", exitstatus);
    EXIT(9);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [p2gg] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [p2gg] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [p2gg] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [p2gg] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block == NULL) {
    fprintf(stderr, "[p2gg] Error, eo_evecs_block is NULL\n");
    EXIT(10);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[p2gg] Error, dimension of eigenspace is zero\n");
    EXIT(11);
  }

  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[p2gg] Error from tmLQCD_init_deflator, status was %d\n", exitstatus);
    EXIT(8);
  }

  evecs_eval = (double*)malloc(evecs_num*sizeof(double));
  if(evecs_eval == NULL) {
    fprintf(stderr, "[p2gg] Error from malloc\n");
    EXIT(39);
  }
  for(i=0; i<evecs_num; i++) {
    evecs_eval[i] = ((double*)(g_tmLQCD_defl.evals))[2*i];
  }

#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */

  /*************************************************
   * allocate memory for the eigenvector fields
   *************************************************/
  eo_evecs_field = (double**)calloc(evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block;
  for(i=1; i<evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + _GSI(Vhalf);

  /*************************************************
   * allocate memory for samples
   *************************************************/
  eo_sample_block = (double*)malloc(2*g_nsample * sizeof_eo_spinor_field);
  if(eo_sample_block == NULL) {
    fprintf(stderr, "[p2gg] Error from malloc\n");
    EXIT(12);
  }
  eo_sample_field    = (double**)calloc(2*g_nsample, sizeof(double*));
  eo_sample_field[0] = eo_sample_block;
  for(i=1; i<2*g_nsample; i++) eo_sample_field[i] = eo_sample_field[i-1] + _GSI(Vhalf);

  /*************************************************
   * allocate memory for eo spinor fields 
   * WITHOUT HALO
   *************************************************/
  no_eo_fields = 360;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));

  eo_spinor_field[0] = (double*)malloc( no_eo_fields * sizeof_eo_spinor_field);
  if(eo_spinor_field[0] == NULL) {
    fprintf(stderr, "[p2gg] Error from calloc\n");
    EXIT(13);
  }
  for(i=1; i<no_eo_fields; i++) eo_spinor_field[i] = eo_spinor_field[i-1] + _GSI(Vhalf);

  /*************************************************
   * allocate memory for eo spinor fields 
   * WITH HALO
   *************************************************/
  no_full_fields = 2;
  full_spinor_field = (double**)calloc(no_full_fields, sizeof(double*));
  for(i=0; i<no_full_fields; i++) alloc_spinor_field(&full_spinor_field[i], VOLUME+RAND);
  eo_spinor_work = (double**)calloc(2*no_full_fields, sizeof(double*));
  for(i=0; i<no_full_fields; i++) {
    eo_spinor_work[2*i  ] = full_spinor_field[i];
    eo_spinor_work[2*i+1] = full_spinor_field[i] + _GSI(VOLUME+RAND) / 2;
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  clover_term_init(&g_clover, 6);

  clover_term_init(&g_mzz_up, 6);
  clover_term_init(&g_mzz_dn, 6);
  clover_term_init(&g_mzzinv_up, 8);
  clover_term_init(&g_mzzinv_dn, 8);

  ratime = _GET_TIME;
  clover_term_eo (g_clover, g_gauge_field);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_term_eo = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (g_mzz_up, g_clover, g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (g_mzz_dn, g_clover, -g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (g_mzzinv_up, g_mzz_up);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (g_mzzinv_dn, g_mzz_dn);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  mzz[0] = g_mzz_up;
  mzz[1] = g_mzz_dn;
  mzzinv[0] = g_mzzinv_up;
  mzzinv[1] = g_mzzinv_dn;

  /***********************************************************
   * init gamma permutations
   ***********************************************************/
  init_contract_cvc_tensor_gperm();

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [p2gg] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [p2gg] proc%.4d is send process\n", g_cart_id);
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
      fprintf(stderr, "[p2gg] Error, io proc must be id 0 in g_tr_comm\n");
      EXIT(14);
    }
  }
#endif


  /***********************************************
   * stochastic propagators
   ***********************************************/
  /* make a source */
  exitstatus = prepare_volume_source(eo_sample_block, g_nsample*VOLUME/2);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg] Error from prepare_volume_source, status was %d\n", exitstatus);
    EXIT(33);
  }

  /* orthogonal projection */
  exitstatus = project_propagator_field ( eo_sample_block, eo_sample_block, 0, eo_evecs_block, g_nsample, evecs_num, Vhalf);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg] Error from project_propagator_field, status was %d\n", exitstatus);
    EXIT(35);
  }

  for(i = 0; i < g_nsample; i++) {
    /* invert */
    memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
    memcpy(eo_spinor_work[0], eo_sample_field[i], sizeof_eo_spinor_field);
    exitstatus = tmLQCD_invert_eo(eo_spinor_work[1], eo_spinor_work[0], _OP_ID_UP);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
      EXIT(19);
    }
    memcpy(eo_sample_field[g_nsample+i], eo_spinor_work[1], sizeof_eo_spinor_field);
  }  /* end of loop on samples */



#ifdef HAVE_LHPC_AFF
  /***********************************************
   * writer for aff output file
   ***********************************************/
  if(io_proc == 2) {
    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [p2gg] using aff version %s\n", aff_status_str);

    sprintf(filename, "%s.%.4d.aff", outfile_prefix, Nconf);
    fprintf(stdout, "# [p2gg] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[p2gg] Error from aff_writer, status was %s\n", aff_status_str);
      EXIT(15);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[p2gg] Error, aff writer is not initialized\n");
      EXIT(16);
    }

  }  /* end of if io_proc == 2 */
#endif


  /***********************************************************
   * loop on source locations
   ***********************************************************/
  for(isource_location=0; isource_location < g_source_location_number; isource_location++) {
    /***********************************************************
     * determine source coordinates, find out, if source_location is in this process
     ***********************************************************/
    gsx[0] = g_source_coords_list[isource_location][0];
    gsx[1] = g_source_coords_list[isource_location][1];
    gsx[2] = g_source_coords_list[isource_location][2];
    gsx[3] = g_source_coords_list[isource_location][3];
#if HAVE_MPI
    source_proc_coords[0] = gsx[0] / T;
    source_proc_coords[1] = gsx[1] / LX;
    source_proc_coords[2] = gsx[2] / LY;
    source_proc_coords[3] = gsx[3] / LZ;

    if(g_cart_id == 0) {
      fprintf(stdout, "# [p2gg] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [p2gg] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
    }

    MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
    if(source_proc_id == g_cart_id) {
      fprintf(stdout, "# [p2gg] process %2d has source location\n", source_proc_id);
    }
#else
    source_proc_id = 0;
#endif
    sx[0] = gsx[0] % T;
    sx[1] = gsx[1] % LX;
    sx[2] = gsx[2] % LY;
    sx[3] = gsx[3] % LZ;

    /* init Usource and source_proc_id */
    init_contract_cvc_tensor_usource(g_gauge_field, gsx);


    for(mu=0; mu<5; mu++) {  /* loop on shifts in direction */
      /**********************************************************
       * shifted source coords and source proc coords
       **********************************************************/
      memcpy( shifted_source_coords, gsx, 4*sizeof(int));
      switch(mu) {
        case 0: shifted_source_coords[0] = ( shifted_source_coords[0] + 1 ) % T_global; break;;
        case 1: shifted_source_coords[1] = ( shifted_source_coords[1] + 1 ) % LX_global; break;;
        case 2: shifted_source_coords[2] = ( shifted_source_coords[2] + 1 ) % LY_global; break;;
        case 3: shifted_source_coords[3] = ( shifted_source_coords[3] + 1 ) % LZ_global; break;;
      }
#if HAVE_MPI
      shifted_source_proc_coords[0] = shifted_source_coords[0] / T;
      shifted_source_proc_coords[1] = shifted_source_coords[1] / LX;
      shifted_source_proc_coords[2] = shifted_source_coords[2] / LY;
      shifted_source_proc_coords[3] = shifted_source_coords[3] / LZ;

      if(g_cart_id == 0) {
        fprintf(stdout, "# [p2gg] global shifted source coordinates: (%3d,%3d,%3d,%3d)\n",  shifted_source_coords[0], shifted_source_coords[1],  shifted_source_coords[2], shifted_source_coords[3]);
        fprintf(stdout, "# [p2gg] shifted source proc coordinates: (%3d,%3d,%3d,%3d)\n",  shifted_source_proc_coords[0], shifted_source_proc_coords[1], shifted_source_proc_coords[2], shifted_source_proc_coords[3]);
      }

      exitstatus = MPI_Cart_rank(g_cart_grid, shifted_source_proc_coords, &shifted_source_proc_id);
      if(exitstatus != MPI_SUCCESS ) {
        fprintf(stderr, "[p2gg] Error from MPI_Cart_rank; status was %d\n", exitstatus);
        EXIT(17);
      }
      if(shifted_source_proc_id == g_cart_id) {
        fprintf(stdout, "# [p2gg] process %2d has shifted source location\n", shifted_source_proc_id);
      }
#else
      shifted_source_proc_id = 0;
#endif

      /**********************************************************
       * up-type propagators
       **********************************************************/
      for(i=0; i<12; i++) {

        eo_spinor_field_id_e =      mu * 12 + i;
        eo_spinor_field_id_o = 60 + mu * 12 + i;

        /* A^-1 g5 source */
        exitstatus = init_clover_eo_spincolor_pointsource_propagator (eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
            shifted_source_coords, i, g_mzzinv_up[0], shifted_source_proc_id==g_cart_id, eo_spinor_work[0]);
        if(exitstatus != 0 ) {
          fprintf(stderr, "[p2gg] Error from init_eo_spincolor_pointsource_propagator; status was %d\n", exitstatus);
          EXIT(18);
        }

        /* C^-1 */
        if(g_cart_id == 0) fprintf(stdout, "# [p2gg] calling tmLQCD_invert_eo\n");
        memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
        memcpy(eo_spinor_work[0], eo_spinor_field[eo_spinor_field_id_o], sizeof_eo_spinor_field);
        exitstatus = tmLQCD_invert_eo(eo_spinor_work[1], eo_spinor_work[0], _OP_ID_UP);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
          EXIT(19);
        }
        memcpy(eo_spinor_field[eo_spinor_field_id_o], eo_spinor_work[1], sizeof_eo_spinor_field);

        /* B^-1 excl. C^-1 */
        exitstatus = fini_clover_eo_propagator (eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o], eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
            g_mzzinv_up[0], eo_spinor_work[0]);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg] Error from fini_eo_propagator, status was %d\n", exitstatus);
          EXIT(20);
        }
      }  /* of loop on i */

      /**********************************************************
       * dn-type propagators
       **********************************************************/
      for(i=0; i<12; i++) {
        /* A^-1 g5 source */
        eo_spinor_field_id_e = 120 + mu * 12 + i;
        eo_spinor_field_id_o = 180 + mu * 12 + i;

        exitstatus = init_clover_eo_spincolor_pointsource_propagator (eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
            shifted_source_coords, i, g_mzzinv_dn[0], shifted_source_proc_id==g_cart_id, eo_spinor_work[0]);
        if(exitstatus != 0 ) {
          fprintf(stderr, "[p2gg] Error from init_eo_spincolor_pointsource_propagator; status was %d\n", exitstatus);
          EXIT(21);
        }

        /* C^-1 */
        if(g_cart_id == 0) fprintf(stdout, "# [p2gg] calling tmLQCD_invert_eo\n");
        memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
        memcpy(eo_spinor_work[0], eo_spinor_field[eo_spinor_field_id_o], sizeof_eo_spinor_field);
        exitstatus = tmLQCD_invert_eo(eo_spinor_work[1], eo_spinor_work[0], _OP_ID_DN);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
          EXIT(22);
        }
        memcpy(eo_spinor_field[eo_spinor_field_id_o], eo_spinor_work[1], sizeof_eo_spinor_field);

        /* B^-1 excl. C^-1 */
        exitstatus = fini_clover_eo_propagator (eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o], eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
            g_mzzinv_dn[0], eo_spinor_work[0]);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg] Error from fini_eo_propagator, status was %d\n", exitstatus);
          EXIT(23);
        }
      }  /* of loop on i */

    }    /* of loop on mu */

  
    /***************************************************************************
     * loop on sequential source gamma matrices
     ***************************************************************************/
    for(iseq_source_momentum=0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) {

      g_seq_source_momentum[0] = g_seq_source_momentum_list[iseq_source_momentum][0];
      g_seq_source_momentum[1] = g_seq_source_momentum_list[iseq_source_momentum][1];
      g_seq_source_momentum[2] = g_seq_source_momentum_list[iseq_source_momentum][2];
      if(g_cart_id == 0) fprintf(stdout, "# [p2gg] using sequential source momentum = (%d, %d, %d)\n",
        g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);

      /***************************************************************************
       * loop on sequential source gamma matrices
       ***************************************************************************/
      for(isequential_source_gamma_id=0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++) {
  
        sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];
        if(g_cart_id == 0) fprintf(stdout, "# [p2gg] using sequential source gamma id = %d\n", sequential_source_gamma_id);

        /***************************************************************************
         * loop on sequential source time slices
         ***************************************************************************/
        for(isequential_source_timeslice=0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++) {

          g_sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];
          shifted_sequential_source_timeslice = ( shifted_source_coords [0] + g_sequential_source_timeslice ) % T_global;

          if(g_cart_id == 0) fprintf(stdout, "# [p2gg] using sequential source timeslice %d\n", g_sequential_source_timeslice);

          /*************************************************
           * allocate memory for the contractions
           *************************************************/
          unsigned int items = 32 * VOLUME;
          size_t bytes = items * sizeof(double);
          cvc_tensor[0] = (double*)malloc(bytes);
          cvc_tensor[1] = (double*)malloc(bytes);
          if( cvc_tensor[0] == NULL || cvc_tensor[1] == NULL ) {
            fprintf(stderr, "[p2gg] could not allocate memory for contr. fields\n");
            EXIT(24);
          }

          memset(cvc_tensor[0], 0, 32*VOLUME*sizeof(double));
          memset(cvc_tensor[1], 0, 32*VOLUME*sizeof(double));
          memset(contact_term[0], 0, 8*sizeof(double));
          memset(contact_term[1], 0, 8*sizeof(double));

          /* loop on up-type and down-type flavor */
          for(iflavor=0; iflavor<2; iflavor++) {

            ratime = _GET_TIME;

             /* flavor-dependent sequential source momentum */
            seq_source_momentum[0] = (1 - 2*iflavor) * g_seq_source_momentum[0];
            seq_source_momentum[1] = (1 - 2*iflavor) * g_seq_source_momentum[1];
            seq_source_momentum[2] = (1 - 2*iflavor) * g_seq_source_momentum[2];

            if(g_cart_id == 0) fprintf(stdout, "# [p2gg] using flavor-dependent sequential source momentum (%d, %d, %d)\n", 
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2]);

            for(mu=0; mu<5; mu++) {
              for(i=0; i<12; i++) {
                eo_spinor_field_id_e = iflavor * 120 +      mu*12 + i;
                eo_spinor_field_id_o = iflavor * 120 + 60 + mu*12 + i;
                eo_seq_spinor_field_id_e = 240 + mu*12 + i;
                eo_seq_spinor_field_id_o = 300 + mu*12 + i;

                exitstatus = init_clover_eo_sequential_source(eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_field[eo_seq_spinor_field_id_o], eo_spinor_field[eo_spinor_field_id_e], eo_spinor_field[eo_spinor_field_id_o],
                    shifted_sequential_source_timeslice, mzzinv[iflavor][0], g_seq_source_momentum, sequential_source_gamma_id, eo_spinor_work[0]);
                if(exitstatus != 0) {
                  fprintf(stderr, "[p2gg] Error from init_clover_eo_sequential_source, status was %d\n", exitstatus);
                  EXIT(25);
                }
              }

              double **pcoeff = NULL;
              init_2level_buffer(&pcoeff, 12, 2*evecs_num);

              /* odd projection coefficients v^+ sp_o */
              exitstatus = project_reduce_from_propagator_field (pcoeff[0], eo_spinor_field[eo_seq_spinor_field_id_o], eo_evecs_block, 12, evecs_num, Vhalf);

              for(i=0; i<12; i++) {
                for(i=0; i<evecs_num; i++) {
                  double norm = 2.*g_kappa / evecs_eval[i];
                  pcoeff[i][2*i  ] *= norm;
                  pcoeff[i][2*i+1] *= norm;
                }
              }

              exitstatus = project_expand_to_propagator_field(eo_spinor_field[300+12*mu], pcoeff[0], eo_evecs_block, 12, evecs_num, Vhalf);

              for(i=0; i<12; i++) {
                eo_seq_spinor_field_id_e = 240 + mu*12 + i;
                eo_seq_spinor_field_id_o = 300 + mu*12 + i;
                memcpy(eo_spinor_work[0], eo_spinor_field[eo_seq_spinor_field_id_o], sizeof_eo_spinor_field);
                C_clover_oo (eo_spinor_field[eo_seq_spinor_field_id_o], eo_spinor_work[0], g_gauge_field, eo_spinor_work[1], mzz[1-iflavor][1], mzzinv[1-iflavor][0]);

                Q_clover_eo_SchurDecomp_Binv (eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_field[eo_seq_spinor_field_id_o],
                                              eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_field[eo_seq_spinor_field_id_o], g_gauge_field, mzzinv[iflavor][0], eo_spinor_work[0]);
              }
              fini_2level_buffer(&pcoeff);
            }
            retime = _GET_TIME;
            if(g_cart_id==0) fprintf(stdout, "# [p2gg] time for preparing sequential propagator = %e seconds\n", retime-ratime);

            for(mu=0; mu<5; mu++) {
              for(i=0; i<12; i++) {
                /* the sequential propagator */
                fwd_list_eo[0][mu][i] = eo_spinor_field[240 + mu*12 + i];
                fwd_list_eo[1][mu][i] = eo_spinor_field[300 + mu*12 + i];

                bwd_list_eo[0][mu][i] = eo_spinor_field[120*(1-iflavor) +      mu*12 + i];
                bwd_list_eo[1][mu][i] = eo_spinor_field[120*(1-iflavor) + 60 + mu*12 + i];
              }
            }

            ratime = _GET_TIME;
            contract_cvc_tensor(cvc_tensor[iflavor], contact_term[iflavor], NULL, NULL, fwd_list_eo, bwd_list_eo);

            retime = _GET_TIME;
            if(g_cart_id==0) fprintf(stdout, "# [p2gg] time for cvc_tensor contraction = %e seconds\n", retime-ratime);

          }  /* end of loop on iflavor */
  
          /***************************************************************************
           * combine up-type and dn-type part, normalisation of contractions
           ***************************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for(ix=0; ix<16*VOLUME; ix++) {
            /* real part */
            cvc_tensor[0][2*ix  ] += sequential_source_gamma_id_sign[ sequential_source_gamma_id ] * cvc_tensor[1][2*ix  ];
            cvc_tensor[0][2*ix  ] *= -0.25;
            /* imaginary part */
            cvc_tensor[0][2*ix+1] -= sequential_source_gamma_id_sign[ sequential_source_gamma_id ] * cvc_tensor[1][2*ix+1];
            cvc_tensor[0][2*ix+1] *= -0.25;
          }
          free(cvc_tensor[1]);

          /***************************************************************************
           * subtract contact term
           ***************************************************************************/
          if(source_proc_id == g_cart_id) {
            ix = g_ipt[sx[0]][sx[1]][sx[2]][sx[3]];
            cvc_tensor[0][_GWI( 0,ix, VOLUME)  ] -= contact_term[0][ 0];
            cvc_tensor[0][_GWI( 0,ix, VOLUME)+1] -= contact_term[0][ 1];
            cvc_tensor[0][_GWI( 5,ix, VOLUME)  ] -= contact_term[0][ 2];
            cvc_tensor[0][_GWI( 5,ix, VOLUME)+1] -= contact_term[0][ 3];
            cvc_tensor[0][_GWI(10,ix, VOLUME)  ] -= contact_term[0][ 4];
            cvc_tensor[0][_GWI(10,ix, VOLUME)+1] -= contact_term[0][ 5];
            cvc_tensor[0][_GWI(15,ix, VOLUME)  ] -= contact_term[0][ 6];
            cvc_tensor[0][_GWI(15,ix, VOLUME)+1] -= contact_term[0][ 7];
          }

          /***************************************************************************
           * momentum projections
           ***************************************************************************/
          ratime = _GET_TIME;
          init_3level_buffer(&cvc_tp, g_sink_momentum_number, 16, 2*T);
          exitstatus = momentum_projection (cvc_tensor[0], cvc_tp[0][0], T*16, g_sink_momentum_number, g_sink_momentum_list);
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg] Error from init_clover_eo_sequential_source, status was %d\n", exitstatus);
            EXIT(26);
          }
          retime = _GET_TIME;
          if(g_cart_id==0) fprintf(stdout, "# [p2gg] time for momentum projection = %e seconds\n", retime-ratime);

          /***************************************************************************
           * write results to file
           ***************************************************************************/
          ratime = _GET_TIME;

          items = g_sink_momentum_number * 16 * T_global;
          bytes = items * sizeof(double _Complex);
          if(io_proc == 2) {
            aff_buffer = (double _Complex*)malloc(bytes);
            if(aff_buffer == NULL) {
              fprintf(stderr, "[p2gg] Error from malloc\n");
              EXIT(27);
            }
          }

#ifdef HAVE_MPI
          i = g_sink_momentum_number * 32 * T;
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
          if(io_proc>0) {
            exitstatus = MPI_Gather(cvc_tp[0][0], i, MPI_DOUBLE, aff_buffer, i, MPI_DOUBLE, 0, g_tr_comm);
          }
#else
          exitstatus = MPI_Gather(cvc_tp[0][0], i, MPI_DOUBLE, aff_buffer, i, MPI_DOUBLE, 0, g_cart_grid);
#endif
          if(exitstatus != MPI_SUCCESS) {
            fprintf(stderr, "[p2gg] Error from MPI_Gather, status was %d\n", exitstatus);
            EXIT(28);
          }
#else
          memcpy(aff_buffer, cvc_tp[0][0], bytes);
#endif
          if(io_proc == 2) {
            for(i=0; i<g_sink_momentum_number; i++) {
              sprintf(aff_buffer_path, "/PJJ/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/px%.2dpy%.2dpz%.2d", 
                  gsx[0], gsx[1], gsx[2], gsx[3],
                  g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
                  g_sequential_source_gamma_id, g_sequential_source_timeslice, 
                  g_sink_momentum_list[i][0], g_sink_momentum_list[i][1], g_sink_momentum_list[i][2]);
              fprintf(stdout, "# [p2gg] current aff path = %s\n", aff_buffer_path);
              affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
              exitstatus = aff_node_put_complex (affw, affdir, aff_buffer+16*T_global*i, (uint32_t)T_global*16);
              if(exitstatus != 0) {
                fprintf(stderr, "[p2gg] Error from aff_node_put_double, status was %d\n", exitstatus);
                EXIT(29);
              }
            }
            free(aff_buffer);
          }  /* if io_proc == 2 */

          retime = _GET_TIME;
          if(g_cart_id==0) fprintf(stdout, "# [p2gg] time for saving momentum space results = %e seconds\n", retime-ratime);

          /********************************************
           * check the Ward identity in position space 
           ********************************************/
          if(check_position_space_WI) {
            ratime = _GET_TIME;
            double *conn_buffer = (double*)malloc(32*(VOLUME+RAND)*sizeof(double));
            if(conn_buffer == NULL)  {
              fprintf(stderr, "# [p2gg] Error from malloc\n");
              EXIT(30);
            }
            memcpy(conn_buffer, cvc_tensor[0], 32*VOLUME*sizeof(double));
            xchange_contraction(conn_buffer, 32);
            if(g_cart_id == 0) fprintf(stdout, "# [p2gg] checking Ward identity in position space\n");
            for(nu=0; nu<4; nu++) {
              double norm = 0.;
              complex w;
              for(x0=0; x0<T;  x0++) {
              for(x1=0; x1<LX; x1++) {
              for(x2=0; x2<LY; x2++) {
              for(x3=0; x3<LZ; x3++) {
                ix=g_ipt[x0][x1][x2][x3];
                w.re = conn_buffer[_GWI(4*0+nu,ix          ,VOLUME)  ] + conn_buffer[_GWI(4*1+nu,ix          ,VOLUME)  ]
                     + conn_buffer[_GWI(4*2+nu,ix          ,VOLUME)  ] + conn_buffer[_GWI(4*3+nu,ix          ,VOLUME)  ]
                     - conn_buffer[_GWI(4*0+nu,g_idn[ix][0],VOLUME)  ] - conn_buffer[_GWI(4*1+nu,g_idn[ix][1],VOLUME)  ]
                     - conn_buffer[_GWI(4*2+nu,g_idn[ix][2],VOLUME)  ] - conn_buffer[_GWI(4*3+nu,g_idn[ix][3],VOLUME)  ];

                w.im = conn_buffer[_GWI(4*0+nu,ix          ,VOLUME)+1] + conn_buffer[_GWI(4*1+nu,ix          ,VOLUME)+1]
                     + conn_buffer[_GWI(4*2+nu,ix          ,VOLUME)+1] + conn_buffer[_GWI(4*3+nu,ix          ,VOLUME)+1]
                     - conn_buffer[_GWI(4*0+nu,g_idn[ix][0],VOLUME)+1] - conn_buffer[_GWI(4*1+nu,g_idn[ix][1],VOLUME)+1]
                     - conn_buffer[_GWI(4*2+nu,g_idn[ix][2],VOLUME)+1] - conn_buffer[_GWI(4*3+nu,g_idn[ix][3],VOLUME)+1];
      
                norm += w.re*w.re + w.im*w.im;
              }}}}
#ifdef HAVE_MPI
              double dtmp = norm;
              exitstatus = MPI_Allreduce(&dtmp, &norm, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid);
              if(exitstatus != MPI_SUCCESS) {
                fprintf(stderr, "[p2gg] Error from MPI_Allreduce, status was %d\n", exitstatus);
                EXIT(31);
              }
#endif
              if(g_cart_id == 0) fprintf(stdout, "# [p2gg] WI nu = %2d norm = %25.16e\n", nu, norm);
            }  /* end of loop on nu */
            retime = _GET_TIME;
            if(g_cart_id==0) fprintf(stdout, "# [p2gg] time for saving momentum space results = %e seconds\n", retime-ratime);
          }  /* end of if check_position_space_WI */

          free(cvc_tensor[0]);
        }  /* end of loop on sequential gamma id */

      }  /* end of loop on sequential source momentum */

    }  /* end of loop on sequential source timeslices */

  }  /* end of loop on source locations */

#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[p2gg] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(32);
    }
  }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */


  /****************************************
   * free the allocated memory, finalize
   ****************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  for(i=0; i<no_full_fields; i++) free(full_spinor_field[i]);
  free(full_spinor_field);
  free(eo_spinor_work);

  free(eo_spinor_field[0]);
  free(eo_spinor_field);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(eo_evecs_block);
#endif
  free(eo_evecs_field);

  free(eo_sample_block);
  free(eo_sample_field);

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg] %s# [p2gg] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg] %s# [p2gg] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
