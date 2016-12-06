/****************************************************
 * piN2piN_exact.c
 * 
 * Thu Dec  1 15:14:13 CET 2016
 *
 * PURPOSE:
 * pi N - pi N 2-point function contractions
 * with only exact propagators (point-source propagators,
 * seq and seq2 propagators)
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
/* #include "smearing_techniques.h" */
#include "contractions_io.h"
#include "matrix_init.h"
#include "project.h"
#include "prepare_source.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform contractions for proton 2-pt. function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  fprintf(stdout, "         -a write ascii output too [default no ascii output]\n");
  fprintf(stdout, "         -F fermion type, must be set [default -1, no type]\n");
  fprintf(stdout, "         -q/Q/p/P p[i,f][1,2] source and sink momenta [default 0]\n");
  fprintf(stdout, "         -h? this help\n");
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  const int n_c=3;
  const int n_s=4;
  const char outfile_prefix[] = "piN2piN_exact";


  int c, i, k;
  int filename_set = 0;
  int exitstatus;
  int it, ir, is;
  int op_id_up= -1, op_id_dn = -1;
  int gsx[4], sx[4];
  int write_ascii=0;
  int write_xspace = 0;
  int source_proc_id = 0, source_proc_coords[4];
  int sink_proc_id = 0, sink_timeslice, isink_timeslice;
  char filename[200], contype[1200];
  double ratime, retime;
  double plaq_m, plaq_r;
  double *spinor_work[2];
  unsigned int ix;
  unsigned int VOL3;
  size_t sizeof_spinor_field = 0, sizeof_spinor_field_timeslice = 0;
  spinor_propagator_type *connq=NULL;
  double ****connt = NULL, ***connt_p=NULL, ***connt_n=NULL;
  double ***buffer=NULL;
  int io_proc = -1;
  int icomp, iseq_mom, iseq2_mom;

/*******************************************************************
 * Gamma components for the Delta:
 *                                                                 */
  const int num_component = 1;
  int gamma_component[2][1] = { {5},
                                {5} };
  double gamma_component_sign[1] = {+1.};
/*
 *******************************************************************/

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

  while ((c = getopt(argc, argv, "xah?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'x':
      write_xspace = 1;
      fprintf(stdout, "# [piN2piN_exact] will write x-space correlator\n");
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "# [piN2piN_exact] will write x-space correlator in ASCII format\n");
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
    fprintf(stderr, "# [piN2piN_exact] fermion_type must be set\n");
    exit(1);
  } else {
    fprintf(stdout, "# [piN2piN_exact] using fermion type %d\n", g_fermion_type);
  }

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [piN2piN_exact] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
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
  fprintf(stdout, "[piN2piN_exact] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[piN2piN_exact] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  VOL3 = LX*LY*LZ;
  sizeof_spinor_field = _GSI(VOLUME)*sizeof(double);
  sizeof_spinor_field_timeslice = _GSI(VOL3)*sizeof(double);

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [piN2piN_exact] proc%.4d tr%.4d is io process\n", g_cart_id, g_tr_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [piN2piN_exact] proc%.4d tr%.4d is send process\n", g_cart_id, g_tr_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif



#ifndef HAVE_TMLQCD_LIBWRAPPER
  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  switch(g_gauge_file_format) {
    case 0:
      sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
      exitstatus = read_lime_gauge_field_doubleprec(filename);
      break;
    case 1:
      sprintf(filename, "%s.%.5d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "\n# [piN2piN_exact] reading gauge field from file %s\n", filename);
      exitstatus = read_nersc_gauge_field(g_gauge_field, filename, &plaq_r);
      break;
  }
  if(exitstatus != 0) {
    fprintf(stderr, "[piN2piN_exact] Error, could not read gauge field\n");
    EXIT(21);
  }
#  ifdef HAVE_MPI
  xchange_gauge();
#  endif
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[piN2piN_exact] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[piN2piN_exact] Error, &g_gauge_field is NULL\n");
    EXIT(5);
  }
#endif

  /* measure the plaquette */
  plaquette(&plaq_m);
  if(g_cart_id==0) fprintf(stdout, "# [piN2piN_exact] read plaquette value    : %25.16e\n", plaq_r);
  if(g_cart_id==0) fprintf(stdout, "# [piN2piN_exact] measured plaquette value: %25.16e\n", plaq_m);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/
#if 0
  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[piN2piN_exact] Error from tmLQCD_init_deflator, status was %d\n", exitstatus);
    EXIT(8);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[piN2piN_exact] Error from tmLQCD_get_deflator_params, status was %d\n", exitstatus);
    EXIT(9);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [piN2piN_exact] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [piN2piN_exact] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [piN2piN_exact] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [piN2piN_exact] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block == NULL) {
    fprintf(stderr, "[piN2piN_exact] Error, eo_evecs_block is NULL\n");
    EXIT(10);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[piN2piN_exact] Error, dimension of eigenspace is zero\n");
    EXIT(11);
  }

  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[piN2piN_exact] Error from tmLQCD_init_deflator, status was %d\n", exitstatus);
    EXIT(8);
  }

  evecs_eval = (double*)malloc(evecs_num*sizeof(double));
  if(evecs_eval == NULL) {
    fprintf(stderr, "[piN2piN_exact] Error from malloc\n");
    EXIT(39);
  }
  for(i=0; i<evecs_num; i++) {
    evecs_eval[i] = ((double*)(g_tmLQCD_defl.evals))[2*i];
  }
#endif
#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */


  /***********************************************************
   * allocate memory for the spinor fields
   ***********************************************************/
  g_spinor_field = NULL;
  no_fields = n_s*n_c;
  if(g_fermion_type == _TM_FERMION) {
    no_fields *= 6;
  } else {
    no_fields *= 5;
  }
  no_fields += 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields-2; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME);
  alloc_spinor_field(&g_spinor_field[no_fields-2], VOLUMEPLUSRAND);
  alloc_spinor_field(&g_spinor_field[no_fields-1], VOLUMEPLUSRAND);
  spinor_work[0] = g_spinor_field[no_fields-2];
  spinor_work[1] = g_spinor_field[no_fields-1];

  /***********************************************************
   * allocate memory for the contractions
   **********************************************************/
  connq = create_sp_field( (size_t)VOLUME * num_component );
  if(connq == NULL) {
    fprintf(stderr, "[piN2piN_exact] Error, could not alloc connq\n");
    EXIT(2);
  }

  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/
  /* global source coordinates */
  gsx[0] = g_source_location / ( LX_global * LY_global * LZ_global);
  gsx[1] = (g_source_location % ( LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  gsx[2] = (g_source_location % ( LY_global * LZ_global)) / LZ_global;
  gsx[3] = (g_source_location % LZ_global);
  /* local source coordinates */
  sx[0] = gsx[0] % T;
  sx[1] = gsx[1] % LX;
  sx[2] = gsx[2] % LY;
  sx[3] = gsx[3] % LZ;
  source_proc_id = 0;
#ifdef HAVE_MPI
  source_proc_coords[0] = gsx[0] / T;
  source_proc_coords[1] = gsx[1] / LX;
  source_proc_coords[2] = gsx[2] / LY;
  source_proc_coords[3] = gsx[3] / LZ;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [piN2piN_exact] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx[0], gsx[1], gsx[2], gsx[3]);
    fprintf(stdout, "# [piN2piN_exact] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  exitstatus = MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  if(exitstatus !=  MPI_SUCCESS ) {
    fprintf(stderr, "[piN2piN_exact] Error from MPI_Cart_rank, status was %d\n", exitstatus);
    EXIT(9);
  }
#endif
  if( source_proc_id == g_cart_id) {
    fprintf(stdout, "# [piN2piN_exact] process %2d has source location\n", source_proc_id);
  }


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
   * up-type propagator
   ***********************************************************/
  ratime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] up-type inversion\n");
  for(is=0;is<n_s*n_c;is++) {
    memset(spinor_work[0], 0, sizeof_spinor_field);
    memset(spinor_work[1], 0, sizeof_spinor_field);
    if(source_proc_id == g_cart_id)  {
      spinor_work[0][_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]])+2*is] = 1.;
    }

    exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_up, 0);
    if(exitstatus != 0) {
      fprintf(stderr, "[piN2piN_exact] Error from tmLQCD_invert, status was %d\n", exitstatus);
      EXIT(12);
    }
    memcpy( g_spinor_field[is], spinor_work[1], sizeof_spinor_field);
  }
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stderr, "# [piN2piN_exact] time for up propagator = %e seconds\n", retime-ratime);


  /***********************************************************
   * dn-type propagator
   ***********************************************************/
  if(g_fermion_type == _TM_FERMION) {
    if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] dn-type inversion\n");
    ratime = _GET_TIME;
    for(is=0;is<n_s*n_c;is++) {

      memset(spinor_work[0], 0, sizeof_spinor_field);
      memset(spinor_work[1], 0, sizeof_spinor_field);
      if(source_proc_id == g_cart_id)  {
        spinor_work[0][_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]])+2*is] = 1.;
      }

      exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_dn, 0);
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_exact] Error from tmLQCD_invert, status was %d\n", exitstatus);
        EXIT(12);
      }
      memcpy( g_spinor_field[n_s*n_c+is], spinor_work[1], sizeof_spinor_field);
    }
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] time for dn propagator = %e seconds\n", retime-ratime);
  }

  for(iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {

    /***********************************************************
     * sequential propagator U^{-1} g5 exp(ip) D^{-1}: tfii
     ***********************************************************/
    if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] sequential inversion fpr pi2 = (%d, %d, %d)\n", 
        g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);
    ratime = _GET_TIME;
    for(is=0;is<n_s*n_c;is++) {
      int idprop = (int)( g_fermion_type == _TM_FERMION ) * n_s*n_c + is;
      memset(spinor_work[0], 0, sizeof_spinor_field);
      exitstatus = init_sequential_source(spinor_work[0], g_spinor_field[idprop], gsx[0], g_seq_source_momentum_list[iseq_mom], 5);
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_exact] Error from init_sequential_source, status was %d\n", exitstatus);
        EXIT(14);
      }
      memset(spinor_work[1], 0, sizeof_spinor_field);

      exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_up, 0);
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN_exact] Error from tmLQCD_invert, status was %d\n", exitstatus);
        EXIT(12);
      }
      memcpy( g_spinor_field[idprop + n_s*n_c], spinor_work[1], sizeof_spinor_field);
      if(g_write_sequential_propagator) {
        sprintf(filename, "seq_%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d.inverted",
            filename_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3], is,
            g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);
        if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] writing propagator to file %s\n", filename);
        exitstatus = write_propagator(spinor_work[1], filename, 0, 64);
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN_exact] Error from write_propagator, status was %d\n", exitstatus);
          EXIT(15);
        }
      }  /* end of if write sequential propagator */
    }  /* end of loop on spin-color component */
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stderr, "# [piN2piN_exact] time for seq propagator = %e seconds\n", retime-ratime);
  
    for(iseq2_mom=0; iseq2_mom < g_seq2_source_momentum_number; iseq2_mom++) {
      if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] seq2 inversion for pf2 = (%d, %d, %d)\n",
        g_seq2_source_momentum_list[iseq2_mom][0], g_seq2_source_momentum_list[iseq2_mom][1], g_seq2_source_momentum_list[iseq2_mom][2]);
      /******************************************************
       * seq2 propagators for B diagrams: pffii
       *   D-after-U-after-D
       ******************************************************/
      for(is=0; is<n_s*n_c; is++) memset(g_spinor_field[( (int)( g_fermion_type == _TM_FERMION ) + 2 ) * n_s*n_c + is], 0, sizeof_spinor_field);

      for(isink_timeslice=0; isink_timeslice <= g_src_snk_time_separation; isink_timeslice++) {
        /******************************************************
         * seq2 source
         ******************************************************/
        sink_timeslice = ( gsx[0] + isink_timeslice ) % T_global;
        sink_proc_id = sink_timeslice / T == g_proc_coords[0] ? g_cart_id : -1;
        if(sink_proc_id == g_cart_id ) fprintf(stdout, "# [piN2piN_exact] proc%.4d pffii t_sink = %2d\n", g_cart_id, sink_timeslice);
        for(is=0; is<n_s*n_c; is++) {
          int idprop = ( (int)( g_fermion_type == _TM_FERMION ) + 1 ) * n_s*n_c + is;
          if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] pffii source field id = %d\n", idprop);

          exitstatus = init_sequential_source(spinor_work[0], g_spinor_field[idprop], sink_timeslice, g_seq2_source_momentum_list[iseq2_mom], 5);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN_exact] Error from init_sequential_source, status was %d\n", exitstatus);
            EXIT(14);
          }
          memset(spinor_work[1], 0, sizeof_spinor_field);

          exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_dn, 0);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN_exact] Error from tmLQCD_invert, status was %d\n", exitstatus);
            EXIT(12);
          }
          /******************************************************
           * processes, which have the sink timeslice, copy this
           * timeslice
           ******************************************************/
          if(sink_proc_id == g_cart_id) {
            size_t offset = (size_t) (sink_timeslice%T) * _GSI(VOL3);
            idprop = ( (int)( g_fermion_type == _TM_FERMION ) + 2 ) * n_s*n_c + is;
            if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] pffii propagator field id = %d\n", idprop);
            memcpy( g_spinor_field[idprop]+offset, spinor_work[1]+offset, sizeof_spinor_field_timeslice);
          }
        }  /* end of loop on spin-color */
      }  /* end of loop on sink timeslices */

      /******************************************************
       * seq propagators for W diagrams: tffi
       *   D-after-U
       ******************************************************/
      for(is=0; is<n_s*n_c; is++) memset(g_spinor_field[( (int)( g_fermion_type == _TM_FERMION ) + 3 ) * n_s*n_c + is], 0, sizeof_spinor_field);

      for(isink_timeslice=0; isink_timeslice <= g_src_snk_time_separation; isink_timeslice++) {
        /******************************************************
         * seq source
         ******************************************************/
        sink_timeslice = ( gsx[0] + isink_timeslice ) % T_global;
        sink_proc_id = sink_timeslice / T == g_proc_coords[0] ? g_cart_id : -1;

        if(sink_proc_id == g_cart_id ) fprintf(stdout, "# [piN2piN_exact] proc%.4d tffi t_sink = %2d\n", g_cart_id, sink_timeslice);

        for(is=0; is<n_s*n_c; is++) {
          int idprop = is;  /* source from up-propagator */
          if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] tffi source field id = %d\n", idprop);

          exitstatus = init_sequential_source(spinor_work[0], g_spinor_field[idprop], sink_timeslice, g_seq2_source_momentum_list[iseq2_mom], 5);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN_exact] Error from init_sequential_source, status was %d\n", exitstatus);
            EXIT(14);
          }
          memset(spinor_work[1], 0, sizeof_spinor_field);

          exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_dn, 0);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN_exact] Error from tmLQCD_invert, status was %d\n", exitstatus);
            EXIT(12);
          }
          /******************************************************
           * processes, which have the sink timeslice, copy this
           * timeslice
           ******************************************************/
          if(sink_proc_id == g_cart_id) {
            size_t offset = (size_t) (sink_timeslice%T) * _GSI(VOL3);
            idprop = ( (int)( g_fermion_type == _TM_FERMION ) + 3 ) * n_s*n_c + is;
            if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] tffi propagator field id = %d\n", idprop);
            memcpy( g_spinor_field[idprop]+offset, spinor_work[1]+offset, sizeof_spinor_field_timeslice);
          }
        }  /* end of loop on spin-color */
      }  /* end of loop on sink timeslices */

      /******************************************************
       * seq2 propagators for Z diagrams: pfifi
       *   U-after-D-after-U
       ******************************************************/
      for(is=0; is<n_s*n_c; is++) memset(g_spinor_field[( (int)( g_fermion_type == _TM_FERMION ) + 4 ) * n_s*n_c + is], 0, sizeof_spinor_field);

      for(isink_timeslice=0; isink_timeslice <= g_src_snk_time_separation; isink_timeslice++) {
        /******************************************************
         * seq2 source
         ******************************************************/
        sink_timeslice = ( gsx[0] + isink_timeslice ) % T_global;
        sink_proc_id = sink_timeslice / T == g_proc_coords[0] ? g_cart_id : -1;
        if(sink_proc_id == g_cart_id ) fprintf(stdout, "# [piN2piN_exact] proc%.4d pfifi t_sink = %2d\n", g_cart_id, sink_timeslice);
        for(is=0; is<n_s*n_c; is++) {
          int idprop = ( (int)( g_fermion_type == _TM_FERMION ) + 3 ) * n_s*n_c + is; /* source from tffi just calculated */
          if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] pfifi source field id = %d\n", idprop);

          exitstatus = init_sequential_source(spinor_work[0], g_spinor_field[idprop], sink_timeslice, g_seq2_source_momentum_list[iseq2_mom], 5);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN_exact] Error from init_sequential_source, status was %d\n", exitstatus);
            EXIT(14);
          }
          memset(spinor_work[1], 0, sizeof_spinor_field);

          exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_up, 0);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN_exact] Error from tmLQCD_invert, status was %d\n", exitstatus);
            EXIT(12);
          }
          /******************************************************
           * processes, which have the sink timeslice, copy this
           * timeslice
           ******************************************************/
          if(sink_proc_id == g_cart_id) {
            size_t offset = (size_t) (sink_timeslice%T) * _GSI(VOL3);
            idprop = ( (int)( g_fermion_type == _TM_FERMION ) + 4 ) * n_s*n_c + is;
            if(g_cart_id == 0) fprintf(stdout, "# [piN2piN_exact] pfifi propagator field id = %d\n", idprop);
            memcpy( g_spinor_field[idprop]+offset, spinor_work[1]+offset, sizeof_spinor_field_timeslice);
          }
        }  /* end of loop on spin-color */
      }  /* end of loop on sink timeslices */


      /******************************************************
       * contractions
       *
       * REMEMBER:
       *
       *   uprop = S_u
       *   dprop = S_d  // not needed for Wilson fermions (u and d degenerate)
       *   tfii  = sequential propagator t_f <- t_i <- t_i
       *   tffi  = sequential propagator t_f <- t_f <- t_i
       *   pffii = sequential^2 propagator t_f <- t_f <- t_i <- t_i
       *   pfifi = sequential^2 propagator t_f <- t_i <- t_f <- t_i
       *   fp1   = C Gamma_1 S_u
       *   fp2   = C Gamma_1 S_u C Gamma_2
       *   fp3   =           S_u C Gamma_2
       *   fp4   = C Gamma_1 tffi
       *   fp5   = C Gamma_1 S_d C Gamma_2
       *   fp6   = C Gamma_1 tffi C Gamma_2
       ******************************************************/
      ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel private (ix,icomp) shared(connq)
{
#endif
      fermion_propagator_type fp1, fp2, fp3, fpaux, fp4, fp5, fp6, uprop, dprop;
      fermion_propagator_type tfii, tffi, pffii, pfifi;
      spinor_propagator_type sp_c1, sp_c2, sp_c3, sp1, sp2;

      create_fp(&uprop);
      create_fp(&dprop);
      create_fp(&tfii);
      create_fp(&tffi);
      create_fp(&pfifi);
      create_fp(&pffii);
      create_fp(&fp1);
      create_fp(&fp2);
      create_fp(&fp3);
      create_fp(&fp4);
      create_fp(&fp5);
      create_fp(&fp6);
      create_fp(&fpaux);

      create_sp(&sp_c1);
      create_sp(&sp_c2);
      create_sp(&sp_c3);
      create_sp(&sp1);
      create_sp(&sp2);

      const int _UU    = 0;
      const int _DD    =  (int)(g_fermion_type == _TM_FERMION)      * n_s*n_c;
      const int _TFII  = ((int)(g_fermion_type == _TM_FERMION) +1 ) * n_s*n_c;
      const int _PFFII = ((int)(g_fermion_type == _TM_FERMION) +2 ) * n_s*n_c;
      const int _TFFI  = ((int)(g_fermion_type == _TM_FERMION) +3 ) * n_s*n_c;
      const int _PFIFI = ((int)(g_fermion_type == _TM_FERMION) +4 ) * n_s*n_c;

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for(ix=0; ix<VOLUME; ix++)
      {
        /* assign the propagator points */

        _assign_fp_point_from_field(uprop, g_spinor_field + _UU,    ix);

        _assign_fp_point_from_field(tfii,  g_spinor_field + _TFII,  ix);

        _assign_fp_point_from_field(tffi,  g_spinor_field + _TFFI,  ix);

        _assign_fp_point_from_field(pffii, g_spinor_field + _PFFII, ix);

        _assign_fp_point_from_field(pfifi, g_spinor_field + _PFIFI, ix);

        _assign_fp_point_from_field(dprop, g_spinor_field + _DD,    ix);


        /* flavor rotation for twisted mass fermions */
        if(g_fermion_type == _TM_FERMION) {
          /* S_u */
          _fp_eq_rot_ti_fp(fp1, uprop, +1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(uprop, fp1, +1, g_fermion_type, fp2);

          /* S_d */
          _fp_eq_rot_ti_fp(fp1, dprop, -1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(dprop, fp1, -1, g_fermion_type, fp2);

          /* T_fii^ud */
          _fp_eq_rot_ti_fp(fp1, tfii, +1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(tfii, fp1, -1, g_fermion_type, fp2);

          /* T_ffi^du */
          _fp_eq_rot_ti_fp(fp1, tffi, -1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(tffi, fp1, +1, g_fermion_type, fp2);

          /* P_ffii^dud */
          _fp_eq_rot_ti_fp(fp1, pffii, -1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(pffii, fp1, -1, g_fermion_type, fp2);

          /* P_fifi^udu */
          _fp_eq_rot_ti_fp(fp1, pfifi, +1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(pfifi, fp1, +1, g_fermion_type, fp2);
        }

        for(icomp=0; icomp<num_component; icomp++) {

          _sp_eq_zero( connq[ix*num_component+icomp]);

          /******************************************************
           * prepare fermion propagators
           ******************************************************/
          _fp_eq_zero(fp1);
          _fp_eq_zero(fp2);
          _fp_eq_zero(fp3);
          _fp_eq_zero(fp4);
          _fp_eq_zero(fp5);
          _fp_eq_zero(fp6);
          _fp_eq_zero(fpaux);

          /* fp1 = C Gamma_1 x S_u = g0 g2 Gamma_1 S_u */
          _fp_eq_gamma_ti_fp(fp1, gamma_component[0][icomp], uprop);
          _fp_eq_gamma_ti_fp(fpaux, 2, fp1);
          _fp_eq_gamma_ti_fp(fp1,   0, fpaux);

          /* fp2 = C Gamma_1 x S_u x C Gamma_2 = fp1 x g0 g2 Gamma_2 */
          _fp_eq_fp_ti_gamma(fp2, 0, fp1);
          _fp_eq_fp_ti_gamma(fpaux, 2, fp2);
          _fp_eq_fp_ti_gamma(fp2, gamma_component[1][icomp], fpaux);

          /* fp5 = C Gamma_1 x S_d x C Gamma_2 */
          if(g_fermion_type == _TM_FERMION) {
            _fp_eq_gamma_ti_fp(fp5, gamma_component[0][icomp], dprop);
            _fp_eq_gamma_ti_fp(fpaux, 2, fp5);
            _fp_eq_gamma_ti_fp(fp5,   0, fpaux);
            _fp_eq_fp_ti_gamma(fpaux, 0, fp5);
            _fp_eq_fp_ti_gamma(fp5, 2, fpaux);
            _fp_eq_fp_ti_gamma(fpaux, gamma_component[1][icomp], fp5);
            _fp_eq_fp(fp5, fpaux);
          } else {
            _fp_eq_fp(fp5, fp2);
          }

          /* fp3 = S_u x C Gamma_2 = uprop x g0 g2 Gamma_2 */
          _fp_eq_fp_ti_gamma(fp3,   0, uprop);
          _fp_eq_fp_ti_gamma(fpaux, 2, fp3);
          _fp_eq_fp_ti_gamma(fp3, gamma_component[1][icomp], fpaux);

          /* fp4 = [ C Gamma_1 x tffi ] x C Gamma_2 */
          _fp_eq_gamma_ti_fp(fp4, gamma_component[0][icomp], tffi);
          _fp_eq_gamma_ti_fp(fpaux, 2, fp4);
          _fp_eq_gamma_ti_fp(fp4,   0, fpaux);

          _fp_eq_fp_ti_gamma(fp6, 0, fp4);
          _fp_eq_fp_ti_gamma(fpaux, 2, fp6);
          _fp_eq_fp_ti_gamma(fp6, gamma_component[1][icomp], fpaux);

          /*********************
           * C_1
           *********************/
          _sp_eq_zero( sp_c1 );
//#if 0  /* pffii */
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, pffii, fp1);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, fp3, fpaux);
          _sp_pl_eq_sp(sp_c1, sp1);
//#endif
//#if 0  /* tr */
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, pffii, fp2);
          /* reduce to spin propagator */
          /* _sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp(sp_c1, sp1);
//#endif
          /* add and assign */
          _sp_pl_eq_sp_ti_re( connq[ix*num_component+icomp], sp_c1, gamma_component_sign[icomp]);

          /*********************
           * C_2
           *********************/
          _sp_eq_zero( sp_c2 );
//#if 0
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp3, fp4);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, tfii, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
//#if 0
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp6, uprop);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, tfii, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
//#if 0
          /* (3) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, tfii, fp4);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, fp3, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
//#if 0  /* tr */
          /* (4) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, tfii, fp6);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
          /* add and assign */
          _sp_pl_eq_sp_ti_re( connq[ix*num_component+icomp], sp_c2, -gamma_component_sign[icomp]);

          /*********************
           * C_3
           *********************/
          _sp_eq_zero( sp_c3 );
//#if 0  /* tr */
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, uprop);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract34_fp(sp1, pfifi, fpaux);
          _sp_pl_eq_sp(sp_c3, sp1);
//#endif
//#if 0
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, uprop);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 );*/
          _sp_eq_fp_del_contract23_fp(sp1, pfifi, fpaux);
          _sp_pl_eq_sp(sp_c3, sp1);
//#endif
//#if 0
          /* (3) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, pfifi);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp(sp_c3, sp1);
//#endif
//#if 0  /* tr */
          /* (4) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, pfifi);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp(sp_c3, sp1);
//#endif
          _sp_pl_eq_sp_ti_re( connq[ix*num_component+icomp], sp_c3, gamma_component_sign[icomp]);
        }  /* of icomp */

      }    /* of ix */

      free_fp(&uprop);
      free_fp(&dprop);
      free_fp(&tfii);
      free_fp(&tffi);
      free_fp(&pfifi);
      free_fp(&pffii);
      free_fp(&fp1);
      free_fp(&fp2);
      free_fp(&fp3);
      free_fp(&fp4);
      free_fp(&fp5);
      free_fp(&fp6);
      free_fp(&fpaux);

      free_sp(&sp_c1);
      free_sp(&sp_c2);
      free_sp(&sp_c3);
      free_sp(&sp1);
      free_sp(&sp2);

#ifdef OPENMP
}
#endif
      retime = _GET_TIME;
      if(g_cart_id == 0)  fprintf(stdout, "# [piN2piN_exact] time for contractions = %e seconds\n", retime-ratime);
  
      /***********************************************
       * finish calculation of connq
       ***********************************************/
      ratime = _GET_TIME;
      if(g_propagator_bc_type == 0) {
        // multiply with phase factor
        fprintf(stdout, "# [piN2piN_exact] multiplying with boundary phase factor\n");
        for(it=0;it<T;it++) {
          ir = (it + g_proc_coords[0] * T - gsx[0] + T_global) % T_global;
          const complex w1 = { cos( 3. * M_PI*(double)ir / (double)T_global ), sin( 3. * M_PI*(double)ir / (double)T_global ) };
#ifdef HAVE_OPENMP
#pragma omp parallel private(ix,icomp) shared(connq,it)
{
#endif
          spinor_propagator_type sp1;
          create_sp(&sp1);
#ifdef HAVE_OPENMP
#pragma omp for
#endif
          for(ix=0;ix<VOL3;ix++) {
            unsigned int iix = (it * VOL3 + ix) * num_component;
            for(icomp=0; icomp<num_component; icomp++) {
              _sp_eq_sp(sp1, connq[iix] );
              _sp_eq_sp_ti_co( connq[iix], sp1, w1);
              iix++;
            }
          }
          free_sp(&sp1);
#ifdef HAVE_OPENMP
}  
#endif
        }
      } else if (g_propagator_bc_type == 1) {
        // multiply with step function
        fprintf(stdout, "# [piN2piN_exact] multiplying with boundary step function\n");
        for(ir=0; ir<T; ir++) {
          it = ir + g_proc_coords[0] * T;  // global t-value, 0 <= t < T_global
          if(it < gsx[0]) {
#ifdef HAVE_OPENMP
#pragma omp parallel private(ix,icomp) shared(it,connq)
{
#endif
            spinor_propagator_type sp1;
            create_sp(&sp1);
#ifdef HAVE_OPENMP
#pragma omp for
#endif
            for(ix=0;ix<VOL3;ix++) {
              unsigned int iix = (it * VOL3 + ix) * num_component;
              for(icomp=0; icomp<num_component; icomp++) {
                _sp_eq_sp(sp1, connq[iix] );
                _sp_eq_sp_ti_re( connq[iix], sp1, -1.);
                iix++;
              }
            }
  
            free_sp(&sp1);
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
          }
        }  /* end of if it < gsx[0] */
      }
      retime = _GET_TIME;
      if(g_cart_id == 0)  fprintf(stdout, "# [piN2piN_exact] time for boundary phase = %e seconds\n", retime-ratime);
  
  
      if(write_ascii) {
        /***********************************************
         * each MPI process dump his part in ascii format
         ***********************************************/
        int x0, x1, x2, x3;
        ratime = _GET_TIME;
        sprintf(filename, "%s_x.%.4d.t%.2dx%.2dy%.2dz%.2d.px%.2dpy%.2dpz%.2d.proct%.2dprocx%.2dprocy%.2dprocz%.2d.ascii", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3],
            g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
            g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);
        FILE *ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[piN2piN_exact] Error opening file %s\n", filename);
          EXIT(56);
        }
        for(x0=0; x0 < T; x0++) {
        for(x1=0; x1 < LX; x1++) {
        for(x2=0; x2 < LY; x2++) {
        for(x3=0; x3 < LZ; x3++) {
          ix = g_ipt[x0][x1][x2][x3];
          for(icomp=0; icomp<num_component; icomp++) {
            unsigned int iix = num_component * ix + icomp;
            sprintf(contype, "# t= %2d, x= %2d, y= %2d, z= %2d comp = %2d %2d", x0 + g_proc_coords[0]*T, x1 + g_proc_coords[1]*LX, x2 + g_proc_coords[2]*LY, x3 + g_proc_coords[3]*LZ,
                gamma_component[0][icomp], gamma_component[1][icomp]);
            printf_sp(connq[iix], contype, ofs);
          }
        }}}}
        fclose(ofs);
        retime = _GET_TIME;
        if(g_cart_id == 0)  fprintf(stdout, "# [piN2piN_exact] time for writing ascii = %e seconds\n", retime-ratime);
      }  /* end of if write ascii */
  
  
      /***********************************************
       * write to file
       ***********************************************/
      if(write_xspace) {
        ratime = _GET_TIME;
        char xml_msg[200];
        sprintf(contype, "\n<description> proton 2pt spinor propagator position space\n"\
          "<components>%dx%d</components>\n"\
          "<data_type>%s</data_type>\n"\
          "<precision>%d</precision>\n"\
          "<source_coords_t>%2d</source_coords_t>\n"\
          "<source_coords_x>%2d</source_coords_x>\n"\
          "<source_coords_y>%2d</source_coords_y>\n"\
          "<source_coords_z>%2d</source_coords_z>\n"\
          "<sequential_source_momentum_x>%2d</sequential_source_momentum_x>\n"\
          "<sequential_source_momentum_y>%2d</sequential_source_momentum_y>\n"\
          "<sequential_source_momentum_z>%2d</sequential_source_momentum_z>\n",\
          g_sv_dim, g_sv_dim, "complex", 64, gsx[0], gsx[1], gsx[2], gsx[3],
          g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);

        for(icomp=0; icomp<num_component; icomp++) {
          sprintf(xml_msg, "<spin_structure>Cg%.2d-Cg%.2d</spin_structure>\n",\
              gamma_component[0][icomp], gamma_component[1][icomp]);
          sprintf(contype, "%s\n%s", contype, xml_msg);
        }
        sprintf(filename, "%s_x.%.4d.t%.2dx%.2dy%.2dz%.2d.px%.2dpy%.2dpz%.2d", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3],
            g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);
        write_lime_contraction(connq[0][0], filename, 64, num_component*g_sv_dim*g_sv_dim, contype, Nconf, 0);
        retime = _GET_TIME;
        if(g_cart_id == 0) {
          fprintf(stdout, "# [piN2piN_exact] time for writing xspace = %e seconds\n", retime-ratime);
        }
      }  /* end of if write x-space */
  
      /***********************************************
       * momentum projections
       ***********************************************/
      init_4level_buffer(&connt, T, g_sink_momentum_number, num_component*g_sv_dim, 2*g_sv_dim);
      for(it=0; it<T; it++) {
        fprintf(stdout, "# [piN2piN_exact] proc%.4d momentum projection for t = %2d\n", g_cart_id, it); fflush(stdout);
        /* exitstatus = momentum_projection2 (connq[it*VOL3*num_component][0], connt[it][0][0], num_component*g_sv_dim*g_sv_dim, g_sink_momentum_number, g_sink_momentum_list, &(gsx[1]) ); */
        exitstatus = momentum_projection2 (connq[it*VOL3*num_component][0], connt[it][0][0], num_component*g_sv_dim*g_sv_dim, g_sink_momentum_number, g_sink_momentum_list, NULL );
      }

      /***********************************************
       * multiply with phase from source location
       * - using pi1 + pi2 = - ( pf1 + pf2 ), so
       *   pi1 = - ( pi2 + pf1 + pf2 )
       ***********************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for private(icomp)
#endif
      for(it=0; it<T; it++) {
        double phase;
        complex w;
        spinor_propagator_type sp1;
        create_sp(&sp1);
        for(k=0; k<g_sink_momentum_number; k++) {
          phase = -2 * M_PI * (
            (double)(g_sink_momentum_list[k][0] + g_seq_source_momentum_list[iseq_mom][0] + g_seq2_source_momentum_list[iseq2_mom][0] ) / (double)LX_global * gsx[1]
          + (double)(g_sink_momentum_list[k][1] + g_seq_source_momentum_list[iseq_mom][1] + g_seq2_source_momentum_list[iseq2_mom][1] ) / (double)LY_global * gsx[2]
          + (double)(g_sink_momentum_list[k][2] + g_seq_source_momentum_list[iseq_mom][2] + g_seq2_source_momentum_list[iseq2_mom][2] ) / (double)LZ_global * gsx[3]
          );
          w.re = cos(phase);
          w.im = sin(phase);
          for(icomp=0; icomp<num_component; icomp++) {
            spinor_propagator_type connt_sp = &(connt[it][k][icomp*g_sv_dim]);
            _sp_eq_sp(sp1, connt_sp );
            _sp_eq_sp_ti_co(connt_sp, sp1, w);
          }  /* end of loop on components */
        }  /* end of loop on sink momenta */
        free_sp(&sp1);
      }  /* end of loop on T */

      /***********************************************
       * init connt_p/n for positive/negative parity
       * spin-projection
       ***********************************************/
      init_3level_buffer(&connt_p, T, g_sink_momentum_number, num_component * 2);
      init_3level_buffer(&connt_n, T, g_sink_momentum_number, num_component * 2);
  
  
      if(write_ascii) {
        sprintf(filename, "%s_tq.%.4d.t%.2dx%.2dy%.2dz%.2d.px%.2dpy%.2dpz%.2d.proct%.2dprocx%.2dprocy%.2dprocz%.2d.ascii", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3],
            g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
            g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);
  
        FILE *ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[piN2piN_exact] Error opening file %s\n", filename);
          EXIT(56);
        }
        for(it=0; it<T; it++) {
          for(k=0; k<g_sink_momentum_number; k++) {
            for(icomp=0; icomp<num_component; icomp++) {
              fprintf(ofs, "# t = %2d p = (%d, %d, %d) comp = (%d, %d)\n", it+g_proc_coords[0]*T, g_sink_momentum_list[k][0], g_sink_momentum_list[k][1], g_sink_momentum_list[k][2],
                  gamma_component[0][icomp], gamma_component[1][icomp]);
              int j;
              for(i=0; i<g_sv_dim; i++) {
                for(j=0; j<g_sv_dim; j++) {
                  fprintf(ofs, "%3d%3d%25.16e%25.16e\n", i, j, connt[it][k][icomp*g_sv_dim+i][2*j], connt[it][k][icomp*g_sv_dim+i][2*j+1] );
                }
              }
            }
          }
        }
        fclose(ofs);
      }  /* end of if write ascii */
  
  
#ifdef HAVE_OPENMP
#pragma omp parallel private(k,icomp, it)
{
#endif
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(it=0; it<T; it++) {
        spinor_propagator_type sp1, sp2;
        create_sp(&sp1);
        create_sp(&sp2);
        complex w;
        for(k=0; k<g_sink_momentum_number; k++) {
          for(icomp=0; icomp<num_component; icomp++) {
            _sp_eq_sp(sp1, &(connt[it][k][icomp*g_sv_dim]) );
            _sp_eq_gamma_ti_sp(sp2, 0, sp1);
            _sp_pl_eq_sp(sp1, sp2);
            _co_eq_tr_sp(&w, sp1);
            connt_p[it][k][2*icomp  ] = w.re * 0.25;
            connt_p[it][k][2*icomp+1] = w.im * 0.25;
            /* printf("# [piN2piN_exact] proc%.4d it=%d k=%d icomp=%d w= %25.16e %25.16e\n", g_cart_id, it, k, icomp, connt_p[it][k][2*icomp], connt_p[it][k][2*icomp+1]); */
            _sp_eq_sp(sp1, &(connt[it][k][icomp*g_sv_dim]) );
            _sp_eq_gamma_ti_sp(sp2, 0, sp1);
            _sp_mi_eq_sp(sp1, sp2);
            _co_eq_tr_sp(&w, sp1);
            connt_n[it][k][2*icomp  ] = w.re * 0.25;
            connt_n[it][k][2*icomp+1] = w.im * 0.25;
         }  /* end of loop on components */
         }  /* end of loop on sink momenta */
        free_sp(&sp1);
        free_sp(&sp2);
      }  /* end of loop on T */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  
      fini_4level_buffer(&connt);
  
      if(write_ascii) {
        sprintf(filename, "%s_fw.%.4d.t%.2dx%.2dy%.2dz%.2d.px%.2dpy%.2dpz%.2d.proct%.2dprocx%.2dprocy%.2dprocz%.2d.ascii", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3],
            g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
            g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);

        FILE *ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[piN2piN_exact] Error opening file %s\n", filename);
          EXIT(56);
        }
        for(k=0; k<g_sink_momentum_number; k++) {
          for(icomp=0; icomp<num_component; icomp++) {
            fprintf(ofs, "# p = (%d, %d, %d) comp = (%d, %d)\n", g_sink_momentum_list[k][0], g_sink_momentum_list[k][1], g_sink_momentum_list[k][2],
                gamma_component[0][icomp], gamma_component[1][icomp]);
            for(it=0; it<T; it++) {
                fprintf(ofs, "%3d%25.16e%25.16e\n", it+g_proc_coords[0]*T, connt_p[it][k][2*icomp], connt_p[it][k][2*icomp+1]);
            }
          }
        }
        fclose(ofs);
  
        sprintf(filename, "%s_bw.%.4d.t%.2dx%.2dy%.2dz%.2d.px%.2dpy%.2dpz%.2d.proct%.2dprocx%.2dprocy%.2dprocz%.2d.ascii", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3],
            g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
            g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);
  
        ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[piN2piN_exact] Error opening file %s\n", filename);
          EXIT(56);
        }
        for(k=0; k<g_sink_momentum_number; k++) {
          for(icomp=0; icomp<num_component; icomp++) {
            fprintf(ofs, "# p = (%d, %d, %d) comp = (%d, %d)\n", g_sink_momentum_list[k][0], g_sink_momentum_list[k][1], g_sink_momentum_list[k][2],
                gamma_component[0][icomp], gamma_component[1][icomp]);
            for(it=0; it<T; it++) {
              fprintf(ofs, "%3d%25.16e%25.16e\n", it+g_proc_coords[0]*T, connt_n[it][k][2*icomp], connt_n[it][k][2*icomp+1]);
            }
          }
        }
        fclose(ofs);
      }  /* end of if write ascii */
  
#ifdef HAVE_LHPC_AFF
      /***********************************************
       * open aff output file
       ***********************************************/
    
      if(io_proc == 2) {
        aff_status_str = (char*)aff_version();
        fprintf(stdout, "# [piN2piN_exact] using aff version %s\n", aff_status_str);
    
        sprintf(filename, "%s.%.4d.px%.2dpy%.2dpz%.2d.aff", outfile_prefix, Nconf,
            g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);
        fprintf(stdout, "# [piN2piN_exact] writing data to file %s\n", filename);
        affw = aff_writer(filename);
        aff_status_str = (char*)aff_writer_errstr(affw);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[piN2piN_exact] Error from aff_writer, status was %s\n", aff_status_str);
          EXIT(4);
        }
    
        if( (affn = aff_writer_root(affw)) == NULL ) {
          fprintf(stderr, "[piN2piN_exact] Error, aff writer is not initialized\n");
          EXIT(5);
        }
    
        aff_buffer = (double _Complex*)malloc(T_global*sizeof(double _Complex));
        if(aff_buffer == NULL) {
          fprintf(stderr, "[piN2piN_exact] Error from malloc\n");
          EXIT(6);
        }
      }  /* end of if io_proc == 2 */
#endif
    
    
      /***********************************************
       * output for positive parity spin-projection
       ***********************************************/
      ratime = _GET_TIME;
#ifdef HAVE_MPI
      if(io_proc>0) {
        fprintf(stdout, "# [piN2piN_exact] proc%.4d taking part in Gather\n", g_cart_id);
        init_3level_buffer(&buffer, T_global, g_sink_momentum_number, 2*num_component);
        k = 2 * g_sink_momentum_number * T * num_component;
        exitstatus = MPI_Allgather(connt_p[0][0], k, MPI_DOUBLE, buffer[0][0], k, MPI_DOUBLE, g_tr_comm);
        if(exitstatus != MPI_SUCCESS) {
          EXIT(124);
        }
      }
#else
      buffer = connt_p;
#endif
    
      if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF
        for(k=0; k<g_sink_momentum_number; k++) {
          for(icomp=0; icomp<num_component; icomp++) {
            sprintf(aff_buffer_path, "/%s/P+/qx%.2dqy%.2dqz%.2d/px%.2dpy%.2dpz%.2d/t%.2dx%.2dy%.2dz%.2d/mu%dnu%d", outfile_prefix, 
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                g_sink_momentum_list[k][0], g_sink_momentum_list[k][1], g_sink_momentum_list[k][2],
                gsx[0], gsx[1], gsx[2], gsx[3], gamma_component[0][icomp], gamma_component[1][icomp]);
            fprintf(stdout, "# [piN2piN_exact] current aff path = %s\n", aff_buffer_path);
            affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
            for(it=0; it<T_global; it++) {
              ir = ( it - gsx[0] + T_global ) % T_global;
              aff_buffer[ir] = buffer[it][k][2*icomp] + buffer[it][k][2*icomp+1] * I;
            }
            /* memcpy(aff_buffer, buffer[k], 2*T_global*sizeof(double)); */
            int status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)T_global);
            if(status != 0) {
              fprintf(stderr, "[piN2piN_exact] Error from aff_node_put_double, status was %d\n", status);
              EXIT(8);
            }
          }
        }
#endif
      }
    
      /***********************************************
       * output for negative parity spin-projection
       ***********************************************/
#ifdef HAVE_MPI
      if(io_proc>0) {
        k = 2 * g_sink_momentum_number * T * num_component;
        exitstatus = MPI_Allgather(connt_n[0][0], k, MPI_DOUBLE, buffer[0][0], k, MPI_DOUBLE, g_tr_comm);
        if(exitstatus != MPI_SUCCESS) {
          EXIT(124);
        }
      }
#else
      buffer = connt_n;
#endif
    
      if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF
        for(k=0; k<g_sink_momentum_number; k++) {
          for(icomp=0; icomp<num_component; icomp++) {
            sprintf(aff_buffer_path, "/%s/P-/qx%.2dqy%.2dqz%.2d/px%.2dpy%.2dpz%.2d/t%.2dx%.2dy%.2dz%.2d/m%dn%d", outfile_prefix, 
                g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                g_sink_momentum_list[k][0], g_sink_momentum_list[k][1], g_sink_momentum_list[k][2],
                gsx[0], gsx[1], gsx[2], gsx[3], gamma_component[0][icomp], gamma_component[1][icomp]);
            fprintf(stdout, "# [piN2piN_exact] current aff path = %s\n", aff_buffer_path);
            affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
            for(it=0; it<T_global; it++) {
              ir = ( it - gsx[0] + T_global ) % T_global;
              aff_buffer[ir] = buffer[it][k][2*icomp] + buffer[it][k][icomp] * I;
            }
            /* memcpy(aff_buffer, buffer[k], 2*T_global*sizeof(double)); */
            int status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)T_global);
            if(status != 0) {
              fprintf(stderr, "[piN2piN_exact] Error from aff_node_put_double, status was %d\n", status);
              EXIT(8);
            }
          }
        }  /* end of loop on sink momenta */
#endif
      }  /* end of if io_proc == 2 */
    
      retime = _GET_TIME;
      if(io_proc == 2) fprintf(stdout, "# [piN2piN_exact] time for writing = %e seconds\n", retime - ratime);
    
#ifdef HAVE_MPI
      if(io_proc > 0) {
        fini_3level_buffer(&buffer);
      }
#endif
    
#ifdef HAVE_LHPC_AFF
      if(io_proc == 2) {
        aff_status_str = (char*)aff_writer_close (affw);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[piN2piN_exact] Error from aff_writer_close, status was %s\n", aff_status_str);
          EXIT(11);
        }
        if(aff_buffer != NULL) free(aff_buffer);
      }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */
    
    
      fini_3level_buffer(&connt_p);
      fini_3level_buffer(&connt_n);
  
    }  /* end of loop on seq2 source momenta */

  }  /* end of loop on sequential source momentum */

  /***********************************************
   * free gauge fields and spinor fields
   ***********************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  if(g_gauge_field != NULL) {
    free(g_gauge_field);
    g_gauge_field=(double*)NULL;
  }
#endif
  if(g_spinor_field!=NULL) {
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field); g_spinor_field=(double**)NULL;
  }

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();
  free_sp_field(&connq);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_exact] %s# [piN2piN_exact] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_exact] %s# [piN2piN_exact] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
