/****************************************************
 * test_contractions.cpp
 *
 * Mon Dec 19 08:25:48 CET 2016
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
#include "ranlxd.h"

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
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "matrix_init.h"
#include "contract_cvc_tensor.h"

using namespace cvc;

void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
 
  const int n_s = 4;
  const int n_c = 3;

  int c, i, k, no_fields=0;
  int filename_set = 0;
  /* int have_source_flag = 0; */
  // int gsx[4];
  int x0, x1, x2, x3;
  /* int sx[4]; */
  int exitstatus;
  /* int source_proc_coords[4], source_proc_id = -1; */
  unsigned int ix;
  char filename[100];
  /* double ratime, retime; */
  // double plaq;
  FILE *ofs;

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

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    exit(557);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    exit(558);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    exit(559);
  }
#endif

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(560);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(561);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[] Error, &g_gauge_field is NULL\n");
    EXIT(563);
  }
#else
  EXIT(44);
#endif
 
  int gsx[4], sx[4];
  int source_proc_id = 0;
  gsx[0] = g_source_location / ( LX_global * LY_global * LZ_global);
  gsx[1] = (g_source_location % ( LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  gsx[2] = (g_source_location % ( LY_global * LZ_global)) / LZ_global;
  gsx[3] = (g_source_location % LZ_global);
  if(g_cart_id == 0)  fprintf(stdout, "# [] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx[0], gsx[1], gsx[2], gsx[3]);
#ifdef HAVE_MPI
  int source_proc_coords[4];
  source_proc_coords[0] = gsx[0] / T;
  source_proc_coords[1] = gsx[1] / LX;
  source_proc_coords[2] = gsx[2] / LY;
  source_proc_coords[3] = gsx[3] / LZ;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
#endif
  if(source_proc_id == g_cart_id) {
    fprintf(stdout, "# [] process %2d has source location\n", source_proc_id);
  }
  sx[0] = gsx[0] % T;
  sx[1] = gsx[1] % LX;
  sx[2] = gsx[2] % LY;
  sx[3] = gsx[3] % LZ;


  const unsigned int Vhalf = VOLUME / 2;
  const size_t sizeof_eo_spinor_field = _GSI(Vhalf)  * sizeof(double);
  const size_t sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);
  const size_t sizeof_halo_spinor_field    = _GSI( (VOLUME+RAND) ) * sizeof(double);

  double *fwd_list_eo[2][5][12];
  double *bwd_list_eo[2][5][12];
  /* eo spinor field without halo */
  int no_eo_fields = 240;  /** = 5 * 12 * 2 + 2 * 12 */
  double **eo_spinor_field = (double**)malloc(no_eo_fields * sizeof(double));
  eo_spinor_field[0] = (double*)malloc(no_eo_fields * sizeof_eo_spinor_field);
  for(i=1; i<no_eo_fields; i++) {
    eo_spinor_field[i] = eo_spinor_field[i-1] + _GSI(Vhalf);
  }

  /* spinor fields with halo */
  no_fields = 120;
  g_spinor_field = (double**)malloc(no_fields * sizeof(double*));
  g_spinor_field[0] = (double*)malloc(no_fields * sizeof_halo_spinor_field);
  for(i=1; i<no_fields; i++) g_spinor_field[i] = g_spinor_field[i-1] + _GSI((VOLUME+RAND));

  /* x-space contraction fields */
  double *conn  = (double*)malloc( 4 * VOLUME * 2 * sizeof(double));


  /* init rng file */
  init_rng_stat_file (g_seed, NULL);
  int mu, ia, ib;
  int gamma_id = 14;

  /* fill spinor fields with random noise */
  for(i=0; i<no_fields; i++) ranlxd(g_spinor_field[i], _GSI(VOLUME));

  /* copy spinor fields to eo spinor fields, fwd type */
  for(mu=0; mu<5; mu++) {
    for(i=0; i<no_fields; i++) {
      int k  = mu * 12 + i;
      int ke = k; 
      int ko = k + 60;
      spinor_field_lexic2eo (g_spinor_field[k], eo_spinor_field[ke], eo_spinor_field[ko]);
    }
  }
 
  /* copy lexic spinor fields to eo spinor fields, bwd type */
  for(mu=0; mu<5; mu++) {
    for(i=0; i<no_fields; i++) {
      int k  = 60 + mu * 12 + i;
      int ke = k +  60;
      int ko = k + 120;
      spinor_field_lexic2eo (g_spinor_field[k], eo_spinor_field[ke], eo_spinor_field[ko]);
    }
  }
 
  for(mu=0; mu<5; mu++) {
    for(i=0; i<12; i++) {
      /* fwd list */
      fwd_list_eo[0][mu][i] = eo_spinor_field[      mu*12 + i];
      fwd_list_eo[1][mu][i] = eo_spinor_field[ 60 + mu*12 + i];
      /* bwd list */
      bwd_list_eo[0][mu][i] = eo_spinor_field[120 + mu*12 + i];
      bwd_list_eo[1][mu][i] = eo_spinor_field[180 + mu*12 + i];
    }
  }

  init_contract_cvc_tensor_gperm();
  init_contract_cvc_tensor_usource(g_gauge_field, gsx, co_phase_up);


  /* cvc at source, gamma_id at sink */
  contract_cvc_m(conn, gamma_id, NULL, NULL, fwd_list_eo, bwd_list_eo);

  /* calculation by hand */
  double ****conn_aux = NULL;
  init_4level_buffer(&conn_aux, 8, 12, 12, 2*VOLUME);

  for(i=0; i<no_fields; i++) xchange_field(g_spinor_field[i]);   

  for(mu=0; mu<4; mu++) {
    double sp1[24], sp2[24];

    for(ia=0; ia<12; ia++) {
      double *fwd = g_spinor_field[48 + ia];

    for(ib=0; ib<12; ib++) {
      double *bwd = g_spinor_field[60 + mu*12+ib];
      complex *zconn_ = NULL;

      for(x0 = 0; x0 < T; x0++) {
      for(x1 = 0; x1 < LX; x1++) {
      for(x2 = 0; x2 < LY; x2++) {
      for(x3 = 0; x3 < LZ; x3++) {
        ix = g_ipt[x0][x1][x2][x3];
        zconn_ = (complex*)(&(conn_aux[mu][ib][ia][2*ix]));
        /* bwd(ix)^+ g_5 g_gamma_id fwd(ix) */
        _fv_eq_gamma_ti_fv(sp1, gamma_id, fwd+_GSI(ix));
        _fv_eq_gamma_ti_fv(sp2, 5, sp1);
        _co_eq_fv_dag_ti_fv( zconn_, bwd+_GSI(ix), sp2);
      }}}}

    }
    }

    for(ia=0; ia<12; ia++) {
      double *fwd = g_spinor_field[mu*12 + ia];

    for(ib=0; ib<12; ib++) {
      double *bwd = g_spinor_field[60 + 48 + ib];
      complex *zconn_ = NULL;

      for(x0 = 0; x0 < T; x0++) {
      for(x1 = 0; x1 < LX; x1++) {
      for(x2 = 0; x2 < LY; x2++) {
      for(x3 = 0; x3 < LZ; x3++) {
        ix = g_ipt[x0][x1][x2][x3];
        zconn_ = (complex*)(&(conn_aux[4+mu][ib][ia][2*ix]));
        /* bwd(ix)^+ g_5 g_gamma_id fwd(ix) */
        _fv_eq_gamma_ti_fv(sp1, gamma_id, fwd+_GSI(ix));
        _fv_eq_gamma_ti_fv(sp2, 5, sp1);
        _co_eq_fv_dag_ti_fv( zconn_, bwd+_GSI(ix), sp2);
      }}}}

    }
    }

  }  /* end of loop on mu */

  /* reduce 4-level buffer conn_aux */


  double Ubuffer[72];
  if(source_proc_id == g_cart_id) {
    ix = g_ipt[sx[0]][sx[1]][sx[2]][sx[3]];
    memcpy(Ubuffer, g_gauge_field+_GGI(ix,0), 72*sizeof(double));
  }
#ifdef HAVE_MPI
  MPI_Bcast(Ubuffer, 72, MPI_DOUBLE, source_proc_id, g_cart_grid );
#endif
  double **conn2 = NULL;
  init_2level_buffer(&conn2, 4, 2*VOLUME);

  for(mu=0; mu<4; mu++) {
    double U_[18];
    double **fp1;
    init_2level_buffer(&fp1, 12, 24);

    for(ix=0; ix<VOLUME; ix++) {
      _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]));
 
      for(ir=0; ir<4; ir++) {
        for(ia=0; ia<3; ia++) {
          int ira = 3*ir + ia;
          for(ib=0; ib<3; ib++) {
            irb = 
          }
        }
      }
    }
    fini_2level_buffer(&fp1);
  }


  fini_4level_buffer(&conn_aux);
  fini_2level_buffer(&conn2);
  free( conn );
  free(eo_spinor_field[0]);
  free(eo_spinor_field);
  free(g_spinor_field[0]);
  free(g_spinor_field);

  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  if(no_fields > 0 && g_spinor_field != NULL) { 
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field);
  }

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test] %s# [test] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test] %s# [test] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
