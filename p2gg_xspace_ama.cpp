/****************************************************
 * p2gg_xspace_ama.c
 *
 * Fri Oct  7 16:34:24 CEST 2016
 *
 * - originally copied from p2gg_xspace.cpp
 *
 * PURPOSE:
 * - contractions for P -> gamma gamma in position space
 * - contractions AMA style: low-mode contribution from eo-precon eigenvectors
 *   and stochastic high-mode contribution
 * DONE:
 * TODO:
 * - add phases from source location
 * - test and verify
 * - high-mode part
 * - loop on source locations
 * - add disconnected with vector loop and P-V-2-point
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
#include "matrix_init.h"

#define EO_FLAG_EVEN 0
#define EO_FLAG_ODD  1

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  EXIT(0);
}

/************************************************************************************
 * main program
 ************************************************************************************/

int main(int argc, char **argv) {
  
  /*
   * sign for g5 Gamma^\dagger g5
   *                                                0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   * */
  const int sequential_source_gamma_id_sign[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };

  char outfile_name[] = "p2gg";

  const double PI2 =  2. * M_PI;

  int c, i, j, mu, nu, ir, is, ia, ib, imunu;
  int op_id_up=-1, op_id_dn=-1; iflavor;
  int filename_set = 0;
  int source_location, have_source_flag = 0, have_shifted_source_flag = 0;
  int x0, x1, x2, x3
  unsigned int ix;
  int gsx[4];
  int sx0, sx1, sx2, sx3;
  int isimag[4];
  int gperm[5][4], gperm2[4][4];
  int check_position_space_WI=0;
  int nthreads=-1, threadid=-1;
  int exitstatus;
  int write_ascii=0;
  int source_proc_coords[4], source_proc_id = -1;
  int shifted_source_coords[4], shifted_source_proc_coords[4];
  int seq_source_momentum[3], iseq_source_momentum, mseq_source_momentum[3];
  int ud_one_file = 0;
  int sequential_source_gamma_id, isequential_source_gamma_id, i_sequential_source_timeslice;
  double gperm_sign[5][4], gperm2_sign[4][4];
  double *conn  = NULL,  **conn_buffer[2] = {NULL, NULL};
  double **contact_term_buffer[2], *contact_term = NULL;
  double ***gsp_u=NULL, ***gsp_d=NULL;
  /* double *gsp_buffer=NULL; */
  int verbose = 0;
  char filename[100], contype[1200], sequential_filename_prefix[200];
  char outfile_tag[200];
  double ratime, retime;
  double plaq;
#ifndef HAVE_OPENMP
  double spinor1[24], spinor2[24], U_[18];
#endif
/*  double *gauge_trafo = NULL; */
  double *phi=NULL, *chi=NULL, *source=NULL, *propagator=NULL;
  complex w, w1;
  double Usourcebuff[72], *Usource[4];
  FILE *ofs;

  int evecs_num=0, evecs_eval_set = 0;
  double *evecs_lambdaOneHalf=NULL, *evecs_eval = NULL;

#ifdef HAVE_MPI
  int *mstatus;
#endif
  
  // in order to be able to initialise QMP if QPhiX is used, we need
  // to allow tmLQCD to intialise MPI via QMP
  // the input file has to be read 
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_init_parallel_and_read_input(argc, argv, 1, "invert.input");
#else
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
#endif


  while ((c = getopt(argc, argv, "uwah?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_position_space_WI = 1;
      fprintf(stdout, "\n# [p2gg_xspace_ama] will check Ward identity in position space\n");
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "\n# [p2gg_xspace_ama] will write data in ASCII format too\n");
      break;
    case 'u':
      ud_one_file = 1;
      fprintf(stdout, "\n# [p2gg_xspace_ama] will read u/d propagator from same file\n");
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
  fprintf(stdout, "# [p2gg_xspace_ama] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_xspace_ama] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
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

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(32);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#endif

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[p2gg_xspace_ama] T and L's must be set\n");
    usage();
  }


#ifdef HAVE_MPI
  if((mstatus = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    EXIT(1);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg_xspace_ama] Error from init_geometry\n");
    EXIT(2);
  }

  geometry();



#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace_ama] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg_xspace_ama] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg_xspace_ama] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg_xspace_ama] Error, &g_gauge_field is NULL\n");
    EXIT(5);
  }
#endif

#ifdef HAVE_MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace_ama] measured plaquette value: %25.16e\n", plaq);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  exitstatus = tmLQCD_init_deflator(op_id);
  if( exitstatus > 0) {
    fprintf(stderr, "[p2gg_xspace_ama] Error from tmLQCD_init_deflator, status was %d\n", exitstatus);
    EXIT(9);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, op_id);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_xspace_ama] Error from tmLQCD_get_deflator_params, status was %d\n", exitstatus);
    EXIT(30);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [p2gg_xspace_ama] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [p2gg_xspace_ama] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [p2gg_xspace_ama] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [p2gg_xspace_ama] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block[0] = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block[0] == NULL) {
    fprintf(stderr, "[p2gg_xspace_ama] Error, eo_evecs_block is NULL\n");
    EXIT(32);
  }
  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[p2gg_xspace_ama] Error, dimension of eigenspace is zero\n");
    EXIT(33);
  }

#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */


  /***********************************************
   * allocate memory for the spinor fields
   ***********************************************/
  /* (1) eigenvector blocks */
#ifndef HAVE_TMLQCD_LIBWRAPPER
  eo_evecs_block[0] = (double*)malloc(evecs_num*24*Vhalf*sizeof(double));
  if(eo_evecs_block[0] == NULL) {
    fprintf(stderr, "[] Error from malloc\n");
    EXIT(25);
  }
#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */
  eo_evecs_block[1] = (double*)malloc(evecs_num*24*Vhalf*sizeof(double));
  if(eo_evecs_block[1] == NULL) {
    fprintf(stderr, "[] Error from malloc\n");
    EXIT(26);
  }

  eo_evecs_field = (double**)calloc(2*evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block[0];
  for(i=1; i<evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + 24*Vhalf;
  eo_evecs_field[evecs_num] = eo_evecs_block[1];
  for(i=evecs_num+1; i<2*evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + 24*Vhalf;

  /* (2) fermion fields without halo sites */
  no_eo_fields = 360; /* ( 12 + 4 x 12 ) for up, down and sequential each */
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));

  eo_spinor_field[0] = (double*)calloc( no_eo_fields*24*Vhalf, sizeof(double) );
  if(eo_spinor_field[0] == NULL) {
    fprintf(stderr, "[] Error from calloc\n");
    EXIT(35);
  }
  for(i=1; i<no_eo_fields; i++) eo_spinor_field[i] = eo_spinor_field[i-1] + Vhalf*24;

  /* (3) fermion fields with halo sites */
  eo_spinor_work = (double**)calloc(4, sizeof(double*));
  eo_spinor_work[0] = (double*)calloc( 4*12*(VOLUME+RAND), sizeof(double) );
  if(eo_spinor_work[0] == NULL) {
    fprintf(stderr, "[] Error from calloc\n");
    EXIT(36);
  }
  for(i=1; i<4; i++) {
    eo_spinor_work[i] = eo_spinor_work[i-1] + 12*(VOLUME+RAND);
  }
  full_spinor_work_halo = (double**)calloc(2, sizeof(double*));
  for(i=0; i<2; i++) {
    full_spinor_work_halo[i] = eo_spinor_work[2*i];
  }

  evecs_eval = (double*)malloc(evecs_num * sizeof(double));
  if(evecs_eval == NULL) {
    fprintf(stderr, "[] Error from calloc\n");
    EXIT(63);
  }

  evecs_lambdaOneHalf = (double*)malloc(evecs_num * sizeof(double));
  if(evecs_lambdaOneHalf == NULL) {
    fprintf(stderr, "[] Error from calloc\n");
    EXIT(64);
  }

  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  gsx[0] = g_source_location / ( LX_global * LY_global * LZ_global);
  gsx[1] = (g_source_location % ( LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  gsx[2] = (g_source_location % ( LY_global * LZ_global)) / LZ_global;
  gsx[3] = (g_source_location % LZ_global);
  source_proc_coords[0] = gsx[0] / T;
  source_proc_coords[1] = gsx[1] / LX;
  source_proc_coords[2] = gsx[2] / LY;
  source_proc_coords[3] = gsx[3] / LZ;


  if(g_cart_id == 0) {
    fprintf(stdout, "# [p2gg_xspace_ama] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx[0], gsx[1], gsx[2], gsx[3]);
    fprintf(stdout, "# [p2gg_xspace_ama] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  have_source_flag = (int)(g_cart_id == source_proc_id);
  if(have_source_flag==1) {
    fprintf(stdout, "# [p2gg_xspace_ama] process %2d has source location\n", source_proc_id);
  }
  sx0 = gsx[0] % T;
  sx1 = gsx[1] % LX;
  sx2 = gsx[2] % LY;
  sx3 = gsx[3] % LZ;
# else
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "[p2gg_xspace_ama] process %2d has source location\n", g_cart_id);
  gsx[0] = g_source_location/(LX*LY*LZ)-Tstart;
  gsx[1] = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  gsx[2] = (g_source_location%(LY*LZ)) / LZ;
  gsx[3] = (g_source_location%LZ);

  sx0 = gsx[0] - Tstart;
  sx1 = gsx[1];
  sx2 = gsx[2];
  sx3 = gsx[3];
#endif
  Usource[0] = Usourcebuff;
  Usource[1] = Usourcebuff+18;
  Usource[2] = Usourcebuff+36;
  Usource[3] = Usourcebuff+54;
  if(have_source_flag==1) { 
    fprintf(stdout, "# [p2gg_xspace_ama] local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
    _cm_eq_cm_ti_co(Usource[0], &g_gauge_field[_GGI(source_location,0)], &co_phase_up[0]);
    _cm_eq_cm_ti_co(Usource[1], &g_gauge_field[_GGI(source_location,1)], &co_phase_up[1]);
    _cm_eq_cm_ti_co(Usource[2], &g_gauge_field[_GGI(source_location,2)], &co_phase_up[2]);
    _cm_eq_cm_ti_co(Usource[3], &g_gauge_field[_GGI(source_location,3)], &co_phase_up[3]);
  }
#ifdef HAVE_MPI
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  have_source_flag = source_proc_id;
  MPI_Bcast(Usourcebuff, 72, MPI_DOUBLE, have_source_flag, g_cart_grid);
#  else
  MPI_Gather(&have_source_flag, 1, MPI_INT, mstatus, 1, MPI_INT, 0, g_cart_grid);
  if(g_cart_id==0) {
    for(mu=0; mu<g_nproc; mu++) fprintf(stdout, "# [p2gg_xspace_ama] status[%1d]=%d\n", mu, mstatus[mu]);
  }
  if(g_cart_id==0) {
    for(have_source_flag=0; mstatus[have_source_flag]!=1; have_source_flag++);
    fprintf(stdout, "# [p2gg_xspace_ama] have_source_flag= %d\n", have_source_flag);
  }
  MPI_Bcast(&have_source_flag, 1, MPI_INT, 0, g_cart_grid);
  MPI_Bcast(Usourcebuff, 72, MPI_DOUBLE, have_source_flag, g_cart_grid);
#  endif
  /* fprintf(stdout, "# [p2gg_xspace_ama] proc%.4d have_source_flag = %d\n", g_cart_id, have_source_flag); */
#else
  /* HAVE_MPI not defined */
  have_source_flag = 0;
#endif

  if( have_source_flag == g_cart_id) {
    source_location_iseven = g_iseven[g_ipt[sx0][sx1][sx2][sx3]];
    fprintf(stdout, "# [] source site (%d, %d, %d, %d) is even = %d\n", gsx[0], gsx[1], gsx[2], gsx[3], source_location_iseven);
  }

  /***********************************************************
   *  initialize the Gamma matrices
   ***********************************************************/
  /*  gamma_5: */
  gperm[4][0] = gamma_permutation[5][ 0] / 6;
  gperm[4][1] = gamma_permutation[5][ 6] / 6;
  gperm[4][2] = gamma_permutation[5][12] / 6;
  gperm[4][3] = gamma_permutation[5][18] / 6;
  gperm_sign[4][0] = gamma_sign[5][ 0];
  gperm_sign[4][1] = gamma_sign[5][ 6];
  gperm_sign[4][2] = gamma_sign[5][12];
  gperm_sign[4][3] = gamma_sign[5][18];
  /* gamma_nu gamma_5 */
  for(nu=0;nu<4;nu++) {
    /* permutation */
    gperm[nu][0] = gamma_permutation[6+nu][ 0] / 6;
    gperm[nu][1] = gamma_permutation[6+nu][ 6] / 6;
    gperm[nu][2] = gamma_permutation[6+nu][12] / 6;
    gperm[nu][3] = gamma_permutation[6+nu][18] / 6;
    /* is imaginary ? */
    isimag[nu] = gamma_permutation[6+nu][0] % 2;
    /* (overall) sign */
    gperm_sign[nu][0] = gamma_sign[6+nu][ 0];
    gperm_sign[nu][1] = gamma_sign[6+nu][ 6];
    gperm_sign[nu][2] = gamma_sign[6+nu][12];
    gperm_sign[nu][3] = gamma_sign[6+nu][18];
    /* write to stdout */
    if(g_cart_id == 0) {
      fprintf(stdout, "# gamma_%d5 = (%f %d, %f %d, %f %d, %f %d)\n", nu,
          gperm_sign[nu][0], gperm[nu][0], gperm_sign[nu][1], gperm[nu][1], 
          gperm_sign[nu][2], gperm[nu][2], gperm_sign[nu][3], gperm[nu][3]);
    }
  }
  /* gamma_nu */
  for(nu=0;nu<4;nu++) {
    /* permutation */
    gperm2[nu][0] = gamma_permutation[nu][ 0] / 6;
    gperm2[nu][1] = gamma_permutation[nu][ 6] / 6;
    gperm2[nu][2] = gamma_permutation[nu][12] / 6;
    gperm2[nu][3] = gamma_permutation[nu][18] / 6;
    /* (overall) sign */
    gperm2_sign[nu][0] = gamma_sign[nu][ 0];
    gperm2_sign[nu][1] = gamma_sign[nu][ 6];
    gperm2_sign[nu][2] = gamma_sign[nu][12];
    gperm2_sign[nu][3] = gamma_sign[nu][18];
    /* write to stdout */
    if(g_cart_id == 0) {
    	fprintf(stdout, "# gamma_%d = (%f %d, %f %d, %f %d, %f %d)\n", nu,
        	gperm2_sign[nu][0], gperm2[nu][0], gperm2_sign[nu][1], gperm2[nu][1], 
        	gperm2_sign[nu][2], gperm2[nu][2], gperm2_sign[nu][3], gperm2[nu][3]);
    }
  }

  /***********************************************************
   * invert using tmLQCD invert
   ***********************************************************/
#ifdef HAVE_TMLQCD_LIBWRAPPER
  if(g_tmLQCD_lat.no_operators > 2) {
    if(g_cart_id == 0) fprintf(stderr, "[p2gg_xspace_ama] Error, confused about number of operators, expected 2 operator (up-type, dn-type)\n");
    EXIT(9);
  }
#endif
  

  /**********************************************************
   * up-type and dn-type spinor fields
   **********************************************************/

  flavor_sign = 1;
  op_id_up    = 0;
  op_id_dn    = 1;

  /* loop on (un)shifted source locations */
  for(mu=0; mu<5; mu++) {

    shifted_source_coords[0] = gsx[0];
    shifted_source_coords[1] = gsx[1];
    shifted_source_coords[2] = gsx[2];
    shifted_source_coords[3] = gsx[3];

    switch (mu) {
      case 0: shifted_source_coords[mu] = ( shifted_source_coords[mu] + 1 ) % T_global; break;
      case 1: shifted_source_coords[mu] = ( shifted_source_coords[mu] + 1 ) % LX_global; break;
      case 2: shifted_source_coords[mu] = ( shifted_source_coords[mu] + 1 ) % LY_global; break;
      case 3: shifted_source_coords[mu] = ( shifted_source_coords[mu] + 1 ) % LZ_global; break;
    }

    shifted_source_proc_coords[0] = shifted_source_coords[0] / T;
    shifted_source_proc_coords[1] = shifted_source_coords[1] / LX;
    shifted_source_proc_coords[2] = shifted_source_coords[2] / LY;
    shifted_source_proc_coords[3] = shifted_source_coords[3] / LZ;


    MPI_Cart_rank(g_cart_grid, shifted_source_proc_coords, &have_shifted_source_flag);

    if(have_shifted_source_flag == g_cart_id) {
      fprintf(stdout, "# [p2gg_xspace_ama] process %4d has %d-shifted source location (%d,%d,%d,%d) -> (%d,%d,%d,%d)\n",
          g_cart_id, mu, gsx[0], g_cart_id, gsx[1], g_cart_id, gsx[2], g_cart_id, gsx[3],
          shifted_source_coords[0], shifted_source_coords[1], shifted_source_coords[2], shifted_source_coords[3]);
    }
    
    /* loop on spin-color components of point source */
    for(ia=0; ia<12; ia++) {

      /* up-type propagator */

      /* prepare source */
      exitstatus = init_eo_spincolor_pointsource_propagator (eo_spinor_work[0], eo_spinor_work[1], shifted_source_coords, ia, flavor_sign, have_source_flag==g_cart_id, eo_spinor_work[2]);
      if(exitstatus != 0 ) {
        fprintf(stderr, "[] Error from init_eo_spincolor_pointsource_propagator; status was %d\n", exitstatus);
        EXIT(36);
      }
  
      /* invert */
      if(g_cart_id == 0) fprintf(stdout, "# [] calling tmLQCD_invert_eo\n");
      exitstatus = tmLQCD_invert_eo(eo_spinor_work[2], eo_spinor_work[1], op_id_up);
      if(exitstatus != 0) {
        fprintf(stderr, "[] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
        EXIT(35);
      }
  
      exitstatus = fini_eo_propagator (eo_spinor_field[12*mu+ia], eo_spinor_field[60+12*mu+ia],  eo_spinor_work[0], eo_spinor_work[2], flavor_sign, eo_spinor_work[1]);
      if(exitstatus != 0 ) {
        fprintf(stderr, "[] Error from fini_eo_propagator; status was %d\n", exitstatus);
        EXIT(37);
      }

      /* down-type propagator */

      /* prepare source */
      exitstatus = init_eo_spincolor_pointsource_propagator (eo_spinor_work[0], eo_spinor_work[1], shifted_source_coords, ia, -flavor_sign, have_source_flag==g_cart_id, eo_spinor_work[2]);
      if(exitstatus != 0 ) {
        fprintf(stderr, "[] Error from init_eo_spincolor_pointsource_propagator; status was %d\n", exitstatus);
        EXIT(36);
      }

      /* invert */
      if(g_cart_id == 0) fprintf(stdout, "# [] calling tmLQCD_invert_eo\n");
      exitstatus = tmLQCD_invert_eo(eo_spinor_work[2], eo_spinor_work[1], op_id_dn);
      if(exitstatus != 0) {
        fprintf(stderr, "[] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
        EXIT(35);
      }

      exitstatus = fini_eo_propagator (eo_spinor_field[120+12*mu+ia], eo_spinor_field[180+12*mu+ia],  eo_spinor_work[0], eo_spinor_work[2], -flavor_sign, eo_spinor_work[1]);
      if(exitstatus != 0 ) {
        fprintf(stderr, "[] Error from fini_eo_propagator; status was %d\n", exitstatus);
        EXIT(37);
      }
    
    }  /* of loop on spin-color component ia */
  }    /* of loop on source location offset mu */

  /***************************************************************************
   *  build the sequential propagator for each sequential gamma id and for
   *  each sequential source momentum and contract; save
   *  pf-dependent tensor for all sequential ti
   *
   *  up-type contractions are done with momentum  pi =  seq_source_momentum,
   *  dn-type contractions are done with momentum -pi = mseq_source_momentum
   *
   *
   * loop on sequential source gamma matrices
   ***************************************************************************/
  for(iseq_source_momentum=0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) {
 
    g_seq_source_momentum[0] = g_seq_source_momentum_list[iseq_source_momentum][0];
    g_seq_source_momentum[1] = g_seq_source_momentum_list[iseq_source_momentum][1];
    g_seq_source_momentum[2] = g_seq_source_momentum_list[iseq_source_momentum][2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [p2gg_xspace_ama] using sequential source momentum = (%d, %d, %d)\n", g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);
    }

    /* flavor-dependent sequential source momentum */
    seq_source_momentum[0]  =  g_seq_source_momentum[0];
    seq_source_momentum[1]  =  g_seq_source_momentum[1];
    seq_source_momentum[2]  =  g_seq_source_momentum[2];

    mseq_source_momentum[0] = -g_seq_source_momentum[0];
    mseq_source_momentum[1] = -g_seq_source_momentum[1];
    mseq_source_momentum[2] = -g_seq_source_momentum[2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [] using flavor-dependent sequential source momentum (%d, %d, %d) and its negative (%d, %d, %d)\n", 
           seq_source_momentum[0],  seq_source_momentum[1],  seq_source_momentum[2],
          mseq_source_momentum[0], mseq_source_momentum[1], mseq_source_momentum[2]);
    }

    sequential_source_momentum_ratime = _GET_TIME;

    /***************************************************************************
     * loop on sequential source gamma matrices
     ***************************************************************************/
    for(isequential_source_gamma_id=0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++) {
 
      sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];
      if(g_cart_id == 0) {
        fprintf(stdout, "# [p2gg_xspace_ama] using sequential source gamma id = %d\n", sequential_source_gamma_id);
      }

      sequential_source_gammaid_ratime = _GET_TIME;

      /***********************************************************
       * allocate memory for gsps
       ***********************************************************/
      init_3level_buffer(&gsp_u, T, 60, 2*evecs_num);
      init_3level_buffer(&gsp_d, T, 60, 2*evecs_num);

      bytes = T * evecs_num * 120 * sizeof(double);
      double *gsp_buffer = (double*)malloc(bytes);
      if(gsp_buffer == NULL) {
        fprintf(stderr, "[p2gg_xspace_ama] error from malloc for gsp_buffer\n");
        EXIT(130);
      }

      /***************************************************************
       * gsp calculation:
       * V and odd part of up-type propagators
       ***************************************************************/
      exitstatus = gsp_calculate_v_dag_gamma_p_w_block_asym(gsp_u[0][0], &(eo_evecs_block[0]), &(eo_spinor_field[60]), evecs_num, 60, 1, &(seq_source_momentum),
          1, &(g_sequential_source_gamma_id_list[isequential_source_gamma_id]), NULL, EO_FLAG_ODD);
      if(exitstatus != 0) {
        fprintf(stderr, "[] error from gsp_calculate_v_dag_gamma_p_w_block_asym, status was %d\n", exitstatus);
        EXIT(129);
      }
    
      /***************************************************************
       * gsp calculation:
       * Xbar V and even part of up-type propagators
       ***************************************************************/
      for(ievecs = 0; ievecs<evecs_num; ievecs++) {
        ratime = _GET_TIME;
        /* 0 <- V */
        memcpy(eo_spinor_work[0], eo_evecs_field[ievecs], sizeof_eo_spinor_field);
        /* 1 <- 0 */
        X_eo (eo_spinor_work[1], eo_spinor_work0, -g_mu, g_gauge_field);
        /* XV <- 1 */
        memcpy(eo_evecs_field[evecs_num+ievecs], eo_spinor_work[1], sizeof_eo_spinor_field);
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [] time for X_eo = %e seconds\n", retime-ratime);
      }
    
      exitstatus = gsp_calculate_v_dag_gamma_p_w_block_asym(gsp_buffer, &(eo_evecs_block[1]), &(eo_spinor_field[0]), evecs_num, 60, 1, &(seq_source_momentum),
          1, &(g_sequential_source_gamma_id_list[isequential_source_gamma_id]), NULL, EO_FLAG_EVEN);
      if(exitstatus != 0) {
        fprintf(stderr, "[] error from gsp_calculate_v_dag_gamma_p_w_block_asym, status was %d\n", exitstatus);
        EXIT(129);
      }
    
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix,ievecs,iix)
#endif
      for(ix=0; ix < T*60; ix++) {
        for(ievecs=0; ievecs < evecs_num; ievecs++) {
          iix = 2 * (ix * evecs_num + ievecs);
          gsp_u[0][0][iix  ] = ( gsp_u[0][0][iix  ] + gsp_buffer[iix  ] ) * evecs_lambdaOneHalf[ievecs];
          gsp_u[0][0][iix+1] = ( gsp_u[0][0][iix+1] + gsp_buffer[iix+1] ) * evecs_lambdaOneHalf[ievecs];
        }
      }

      /***************************************************************
       * gsp calculation:
       * W and odd part of dn-type propagators
       ***************************************************************/
      /* calculate W */
      for(ievecs = 0; ievecs<evecs_num; ievecs++) {
        ratime = _GET_TIME;
        /* 0 <- V */
        memcpy(eo_spinor_work[0], eo_evecs_field[          ievecs], sizeof_eo_spinor_field);
        /* 1 <- XV */
        memcpy(eo_spinor_work[1], eo_evecs_field[evecs_num+ievecs], sizeof_eo_spinor_field);
        /*          input+output     Xeobar V         auxilliary */
        /* 0 <- 0,1 | 2 */
        C_from_Xeo (eo_spinor_work[0], eo_spinor_work[1], eo_spinor_work[2], g_gauge_field, -g_mu);
        /*square norm  <0|0> */
        spinor_scalar_product_re(&norm, eo_spinor_work[0], eo_spinor_work[0], Vhalf);
        if(!evecs_eval_set) {
          evecs_eval[ievecs] = norm  * 4.*g_kappa*g_kappa;
          evecs_lambdaOneHalf = 1. / sqrt(evecs_eval[ievecs]);
          evecs_eval_set = 1;
        }
        norm = 1./sqrt( norm );
        /* W <- 0 */
        spinor_field_eq_spinor_field_ti_re (eo_evecs_field[evecs_num + ievecs],  eo_spinor_work[0], norm, Vhalf);
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [] time for C_from_Xeo = %e seconds\n", retime-ratime);
        /* TEST */
        if(g_cart_id == 0) fprintf(stdout, "# [] eval %4d %25.16e\n", ievecs, evecs_eval[ievecs]);
      }
    
      exitstatus = gsp_calculate_v_dag_gamma_p_w_block_asym(gsp_d[0][0], &(eo_evecs_block[1]), &(eo_spinor_field[180]), evecs_num, 60, 1, &(mseq_source_momentum),
            1, &(g_sequential_source_gamma_id_list[isequential_source_gamma_id]), NULL, EO_FLAG_ODD);
      if(exitstatus != 0) {
        fprintf(stderr, "[] error from gsp_calculate_v_dag_gamma_p_w_block_asym, status was %d\n", exitstatus);
        EXIT(129);
      }
    
      /***************************************************************
       * gsp calculation:
       * XW and even part of dn-type propagators
       ***************************************************************/
      /* calculate Xeo W */
      for(ievecs = 0; ievecs<evecs_num; ievecs++) {
        ratime = _GET_TIME;
        /* 0 <- W */
        memcpy(eo_spinor_work[0], eo_evecs_field[evecs_num+ievecs], sizeof_eo_spinor_field);
        /* 1 <- 0 */
        X_eo (eo_spinor_work[1], eo_spinor_work[0], g_mu, g_gauge_field);
        /* XW <- 1 */
        memcpy(eo_evecs_field[evecs_num+ievecs], eo_spinor_work[1], sizeof_eo_spinor_field);
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [] time for X_eo = %e seconds\n", retime-ratime);
      }
    
      exitstatus = gsp_calculate_v_dag_gamma_p_w_block_asym( gsp_buffer, &(eo_evecs_block[1]), &(eo_spinor_field[120]), evecs_num, 60, 1, &(mseq_source_momentum),
          1, &(g_sequential_source_gamma_id_list[isequential_source_gamma_id]), NULL, EO_FLAG_EVEN);
      if(exitstatus != 0) {
        fprintf(stderr, "[] error from gsp_calculate_v_dag_gamma_p_w_block_asym, status was %d\n", exitstatus);
        EXIT(129);
      }


#pragma omp parallel for private(ix,ievecs,iix)
      for(ix=0; ix < T*60; ix++) {
        for(ievecs=0; ievecs < evecs_num; ievecs++) {
          iix = 2 * (ix * evecs_num + ievecs);
          gsp_d[0][0][iix  ] = ( gsp_d[0][0][iix  ] + gsp_buffer[iix  ] ) * evecs_lambdaOneHalf[ievecs];
          gsp_d[0][0][iix+1] = ( gsp_d[0][0][iix+1] + gsp_buffer[iix+1] ) * evecs_lambdaOneHalf[ievecs];
        }
      }
     
      free(gsp_buffer); gsp_buffer = NULL;

      /* output file tag */
      sprintf( outfile_tag, "tc%.2dxc%.2dyc%.2dzc%.2d.gi%.2d.pix%.2dpiy%.2dpiz%.2d", 
          gsx[0], gsx[1], gsx[2], gsx[3], sequential_source_gamma_id, g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);
      if(g_cart_id == 0) {
        fprintf(stdout, "# [p2gg_xspace_ama] output file flag set to %s\n", outfile_tag);
      }

      /* allocate conn and contact term */
      init_2level_buffer ( &(conn_buffer[0]), T_global, 32*VOLUME);
      init_2level_buffer ( &(conn_buffer[1]), T_global, 32*VOLUME);
      
      init_2level_buffer ( &(contact_term_buffer[0]), T_global, 8);
      init_2level_buffer ( &(contact_term_buffer[1]), T_global, 8);


      /***************************************************************************
       * loop on sequential source timeslices
       ***************************************************************************/
      for(i_sequential_source_timeslice = 0; i_sequential_source_timeslice < T_global; i_sequential_source_timeslice++) {

        g_sequential_source_timeslice = ( gsx[0] + i_sequential_source_timeslice ) % T_global;
        sequential_source_timeslice  = g_sequential_source_timeslice % T;
        
        k = g_sequential_source_timeslice / T;
        MPI_Cart_rank(g_tr_comm, k, &have_sequential_source_timeslice);
        if(have_sequential_source_timeslice == g_tr_id) {
          fprintf(stdout, "# [] proc %d / %d = (%d, %d, %d, %d) has the sequential source timeslice at %d\n", g_tr_id, g_cart_id, 
             g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3], g_sequential_source_timeslice);
        }

        /***************************************************************************
         * loop on up-type, down-type
         ***************************************************************************/
        for(iflavor=0; iflavor<2; iflavor++) {

          flavor_sign = 2 * iflavor - 1;
          conn         = conn_buffer[iflavor][g_sequential_source_timeslice];
          contact_term = contact_term_buffer[iflavor][g_sequential_source_timeslice];


          ratime = _GET_TIME;

          /***************************************************************************
           * build the sequential propagator for each sequential gamma id and each
           * sequential source momentum for the given flavor
           *
           * read propagators from field indices iflavor*120 + 2*(nu*12 + ia) / +1
           * save sequential propagators at field indices 240 + 2*(nu*12 + ia) / +1
           *
           ***************************************************************************/
    
          double **gsp_buffer = NULL;
          init_2level_buffer( &gsp_buffer, 60, 2*evecs_num );
          if( status != 0 ) {
            fprintf(stderr, "[] Error from init_2level_buffer, status was %d\n", exitstatus);
            EXIT(133);
          }

          /* piece together sequential propagator */
          if(iflavor == 0) {
            bytes = evecs_num * 120 * sizeof(double);
            if(have_sequential_source_flag == g_tr_id) {
              memcpy(gsp_buffer[0], gsp_d[sequential_source_timeslice][0], bytes);
            }

            k = evecs_num * 120;
            exitstatus = MPI_Bcast( gsp_buffer[0], k, MPI_DOUBLE, have_sequential_source_flag, g_tr_comm );
            if(exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[] Error from MPI_Bcast, status was %d\n", exitstatus);
              EXIT(140);
            }
            /* odd part: V x Lambda^-1/2 x gsp_d */
            exitstatus = gsp_calculate_v_w_block_asym( eo_spinor_field[300], &(eo_evecs_block[0]), gsp_buffer, evecs_num, 60);

            /* calculate Xbar V from V */
            for(ievecs = 0; ievecs<evecs_num; ievecs++) {
              /* 0 <- V */
              memcpy(eo_spinor_work[0], eo_evecs_field[ievecs], sizeof_eo_spinor_field);
              /* 1 <- 0 */
              X_eo (eo_spinor_work[1], eo_spinor_work0, -g_mu, g_gauge_field);
              /* XV <- 1 */
              memcpy(eo_evecs_field[evecs_num+ievecs], eo_spinor_work[1], sizeof_eo_spinor_field);
            }
            /* even part: Xbar V x Lambda^-1/2 gsp_d */
            exitstatus = gsp_calculate_v_w_block_asym( eo_spinor_field[240], &(eo_evecs_block[1]), gsp_buffer, evecs_num, 60);

          } else {
            /*distribute gsp_u into gsp_buffer */

            bytes = evecs_num * 120 * sizeof(double);
            if(have_sequential_source_flag == g_tr_id) {
              memcpy(gsp_buffer[0], gsp_u[sequential_source_timeslice][0], bytes);
            }

            k = evecs_num * 120;
            exitstatus = MPI_Bcast( gsp_buffer[0], k, MPI_DOUBLE, have_sequential_source_flag, g_tr_comm );
            if(exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[] Error from MPI_Bcast, status was %d\n", exitstatus);
              EXIT(140);
            }

            /* calculate W from Xbar V 
             * Xbar V is current set in eo_evecs_block[1] from iflavor = 1
             * */
            for(ievecs = 0; ievecs<evecs_num; ievecs++) {
              /* 0 <- V */
              memcpy(eo_spinor_work[0], eo_evecs_field[          ievecs], sizeof_eo_spinor_field);
              /* 1 <- XV */
              memcpy(eo_spinor_work[1], eo_evecs_field[evecs_num+ievecs], sizeof_eo_spinor_field);
              /*          input+output     Xeobar V         auxilliary */
              /* 0 <- 0,1 | 2 */
              C_from_Xeo (eo_spinor_work[0], eo_spinor_work[1], eo_spinor_work[2], g_gauge_field, -g_mu);
              /*square norm  <0|0> */
              spinor_scalar_product_re(&norm, eo_spinor_work[0], eo_spinor_work[0], Vhalf);
              norm = 1./sqrt( norm );
              /* W <- 0 */
              spinor_field_eq_spinor_field_ti_re (eo_evecs_field[evecs_num + ievecs],  eo_spinor_work[0], norm, Vhalf);
            }
            /* odd part: W x Lambda^-1/2 gsp_u */
            exitstatus = gsp_calculate_v_w_block_asym( eo_spinor_field[300], &(eo_evecs_block[1]), gsp_buffer, evecs_num, 60);

            /* calculate XW  from W */
            for(ievecs = 0; ievecs<evecs_num; ievecs++) {
             /* 0 <- W */
             memcpy(eo_spinor_work[0], eo_evecs_field[evecs_num+ievecs], sizeof_eo_spinor_field);
             /* 1 <- 0 */
             X_eo (eo_spinor_work[1], eo_spinor_work[0], g_mu, g_gauge_field);
             /* XW <- 1 */
             memcpy(eo_evecs_field[evecs_num+ievecs], eo_spinor_work[1], sizeof_eo_spinor_field);
            }

            /* even part: XW x Lambda^-1/2 gsp_u */
            exitstatus = gsp_calculate_v_w_block_asym( eo_spinor_field[240], &(eo_evecs_block[1]), gsp_buffer, evecs_num, 60);

          }  /* end of if iflavor = 0, 1 */

          fini_2level_buffer(&gsp_buffer);

          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [] time to prepare sequential propagators = %e seconds\n", retime-ratime);

          /**********************************************************
           **********************************************************
           **
           ** contractions
           **
           **********************************************************
           **********************************************************/  
        
          ratime = _GET_TIME;
        
          /**********************************************************
           * first contribution
           **********************************************************/  
          
          /* loop on the Lorentz index nu at source */
          for(nu=0; nu<4; nu++) 
          {
        
            for(ir=0; ir<4; ir++) {
        
              for(ia=0; ia<3; ia++) {
      
                /* sequential from type 1-iflavor */
                phi = full_spinor_work_halo[0];
 
                spinor_field_eo2lexic (phi, eo_spinor_field[240 + 48 + 3*ir + ia], eo_spinor_field[300 + 48 + 3*ir + ia]);
                xchange_field(phi);

              for(ib=0; ib<3; ib++) {
      
                /* propagator from type iflavor */
                /* chi = g_spinor_field[(1 - g_propagator_position) * 60 + nu*12 + 3*gperm[nu][ir] + ib]; */
                chi = full_spinor_worka_halo[1];
      
                spinor_field_eo2lexic (chi, eo_spinor_field[iflavor*120 + 12*nu + 3*gperm[nu][ir] + ib], eo_spinor_field[iflavor*120 + 60 + 12*nu + 3*gperm[nu][ir] + ib]);
                xchange_field(chi);
        
                /* 1) gamma_nu gamma_5 x U */
                for(mu=0; mu<4; mu++) 
                {
        
                  imunu = 4*mu+nu;
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) private(ix, w, w1)  shared(imunu, ia, ib, nu, mu)
{
#endif
                  double spinor1[24], spinor2[24], U_[18];
#ifdef HAVE_OPENMP
#pragma omp for 
#endif
                  for(ix=0; ix<VOLUME; ix++) {
        
                    _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
        
                    _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
                    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        	    _fv_mi_eq_fv(spinor2, spinor1);
        	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
        	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
                    _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
                    if(!isimag[nu]) {
                      conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
                      conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
                    } else {
                      conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
                      conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
                    }
                
                  }  /* of ix */
        
#ifdef HAVE_OPENMP
#pragma omp for 
#endif
                  for(ix=0; ix<VOLUME; ix++) {
                    _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
        
                    _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
                    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        	    _fv_pl_eq_fv(spinor2, spinor1);
        	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
        	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
                    _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
                    if(!isimag[nu]) {
                      conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
                      conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
                    } else {
                      conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
                      conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
                    }
        
                  }  /* of ix */
#ifdef HAVE_OPENMP
}
#endif
        	}    /* of mu */
              }      /* of ib */
              }      /* of ia */
        
              for(ia=0; ia<3; ia++) {
                /* phi = g_spinor_field[      4*12 + 3*ir            + ia]; */
                /* field number sequential propagator e/o + 4*12 + 3*ir            + ia */

                phi = full_spinor_work_halo[0];
                spinor_field_eo2lexic (phi, eo_spinor_field[240 + 48 + 3*ir + ia], eo_spinor_field[300 + 48 + 3*ir + ia]);
                xchange_field(phi);

        
              for(ib=0; ib<3; ib++) {
                /* chi = g_spinor_field[60 + nu*12 + 3*gperm[ 4][ir] + ib]; */
                /* field number iflavor-type propagator e/o + nu*12 + 3*gperm[ 4][ir] + ib */

                chi = full_spinor_work_halo[1];
                spinor_field_eo2lexic (chi, eo_spinor_field[iflavor*120 + nu*12 + 3*gperm[ 4][ir] + ib], eo_spinor_field[iflavor*120+60 + nu*12 + 3*gperm[ 4][ir] + ib]);
                xchange_field(chi);
                
        
                /* -gamma_5 x U */
                for(mu=0; mu<4; mu++) 
                {
        
                  imunu = 4*mu+nu;
        
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) private(ix, w, w1)  shared(imunu, ia, ib, nu, mu)
{
#endif
                  double spinor1[24], spinor2[24], U_[18];
#ifdef HAVE_OPENMP
#pragma omp for
#endif
                  for(ix=0; ix<VOLUME; ix++) {
                    _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
        
                    _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
                    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        	    _fv_mi_eq_fv(spinor2, spinor1);
        	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
        	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
                    _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
                    conn[_GWI(imunu,ix,VOLUME)  ] -= gperm_sign[4][ir] * w1.re;
                    conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[4][ir] * w1.im;
        
                  }  /* of ix */
        

#pragma omp for
                  for(ix=0; ix<VOLUME; ix++) {
                    _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
        
                    _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
                    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        	    _fv_pl_eq_fv(spinor2, spinor1);
        	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
        	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
                    _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
                    conn[_GWI(imunu,ix,VOLUME)  ] -= gperm_sign[4][ir] * w1.re;
                    conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[4][ir] * w1.im;
        
                  }  /* of ix */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
        	}    /* of mu */
              }      /* of ib */
        
              if(iflavor == 0) {
                /* contribution to contact term */
                if(have_source_flag == g_cart_id) {
                  double spinor1[24], spinor2[24];
                  _fv_eq_cm_ti_fv(spinor1, Usource[nu], phi+_GSI(g_iup[source_location][nu]));
                  _fv_eq_gamma_ti_fv(spinor2, nu, spinor1);
                  _fv_mi_eq_fv(spinor2, spinor1);
                  contact_term[2*nu  ] += -0.5 * spinor2[2*(3*ir+ia)  ];
                  contact_term[2*nu+1] += -0.5 * spinor2[2*(3*ir+ia)+1];
                }
              }
      
              }  /* of ia */
            }    /* of ir */
        
          }  /* of nu */
        
      /*
          if(have_source_flag == g_cart_id) {
            fprintf(stdout, "# [p2gg_xspace_ama] contact term after 1st part:\n");
            fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 0, contact_term[0], contact_term[1]);
            fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 1, contact_term[2], contact_term[3]);
            fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 2, contact_term[4], contact_term[5]);
            fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 3, contact_term[6], contact_term[7]);
          }
      */
      
          /**********************************************************
           * second contribution
           **********************************************************/  
        
          /* loop on the Lorentz index nu at source */
          for(nu=0; nu<4; nu++) 
          {
        
            for(ir=0; ir<4; ir++) {
        
              for(ia=0; ia<3; ia++) {
                /* phi = g_spinor_field[     nu*12 + 3*ir            + ia]; */
                /* field number sequ e/o + nu*12 + 3*ir + ia */
        
                phi = full_spinor_work_halo[0];
                spinor_field_eo2lexic (phi, eo_spinor_field[240 + nu*12 + 3*ir + ia], eo_spinor_field[300 + nu*12 + 3*ir + ia]);
                xchange_field(phi);


              for(ib=0; ib<3; ib++) {
                /* chi = g_spinor_field[60 +  4*12 + 3*gperm[nu][ir] + ib]; */
                /* field number iflavor-type prop e/o + 4*12 + 3*gperm[nu][ir] + ib */
        
                chi = full_spinor_work_halo[1];
                spinor_field_eo2lexic (chi, eo_spinor_field[iflavor*120 + 4*12 + 3*gperm[nu][ir] + ib], eo_spinor_field[iflavor*120+60 + 4*12 + 3*gperm[nu][ir] + ib]);
                xchange_field(chi);
            
                /* 1) gamma_nu gamma_5 x U^dagger */
                for(mu=0; mu<4; mu++)
                {
        
                  imunu = 4*mu+nu;
        
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) private(ix, w, w1)  shared(imunu, ia, ib, nu, mu)
{
#endif
                  double spinor1[24], spinor2[24], U_[18];
#ifdef HAVE_OPENMP
#pragma omp for
#endif
                  for(ix=0; ix<VOLUME; ix++) {
                    _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
        
                    _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
                    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        	    _fv_mi_eq_fv(spinor2, spinor1);
        	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
        	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
                    _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
                    if(!isimag[nu]) {
                      conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
                      conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
                    } else {
                      conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
                      conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
                    }
                
                  }  /* of ix */
        
#ifdef HAVE_OPENMP
#pragma omp for
#endif
                  for(ix=0; ix<VOLUME; ix++) {
                    _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
        
                    _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
                    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        	    _fv_pl_eq_fv(spinor2, spinor1);
        	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
        	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
                    _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
                    if(!isimag[nu]) {
                      conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
                      conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
                    } else {
                      conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
                      conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
                    }
        
                  }  /* of ix */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

        	}    /* of mu */
        
              } /* of ib */
              } /* of ia */
        
              for(ia=0; ia<3; ia++) {
                /* phi = g_spinor_field[     nu*12 + 3*ir            + ia]; */
                /* field number seq. e/o + nu*12 + 3*ir            + ia */
        
                phi = full_spinor_work_halo[0];
                spinor_field_eo2lexic (phi, eo_spinor_field[240 + nu*12 + 3*ir + ia], eo_spinor_field[300 + nu*12 + 3*ir + ia]);
                xchange_field(phi);


              for(ib=0; ib<3; ib++) {
                /* chi = g_spinor_field[60 +  4*12 + 3*gperm[ 4][ir] + ib]; */
                /* field number iflavor-type propagator e/o + 4*12 + 3*gperm[ 4][ir] + ib */

                chi = full_spinor_work_halo[1];
                spinor_field_eo2lexic (chi, eo_spinor_field[iflavor*120 + 4*12 + 3*gperm[ 4][ir] + ib], eo_spinor_field[iflavor*120+60 + 4*12 + 3*gperm[ 4][ir] + ib]);
                xchange_field(chi);

                /* -gamma_5 x U */
                for(mu=0; mu<4; mu++)
                {
        
                  imunu = 4*mu+nu;
        
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) private(ix, w, w1)  shared(imunu, ia, ib, nu, mu)
{
#endif
                  double spinor1[24], spinor2[24], U_[18];
#ifdef HAVE_OPENMP
#pragma omp for
#endif
                  for(ix=0; ix<VOLUME; ix++) {
                    _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
        
                    _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
                    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        	    _fv_mi_eq_fv(spinor2, spinor1);
        	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
        	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
                    _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
                    conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[4][ir] * w1.re;
                    conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[4][ir] * w1.im;
                
                  }  /* of ix */
        
#ifdef HAVE_OPENMP
#pragma omp for
#endif
                  for(ix=0; ix<VOLUME; ix++) {
                    _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
        
                    _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
                    _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        	    _fv_pl_eq_fv(spinor2, spinor1);
        	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
        	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
                    _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
                    conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[4][ir] * w1.re;
                    conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[4][ir] * w1.im;
        
                  }  /* of ix */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

        	}    /* of mu */
              }      /* of ib */
        
              /* if(iflavor == 0) { */
                /* contribution to contact term */
                if(have_source_flag == g_cart_id)  {
                  _fv_eq_cm_dag_ti_fv(spinor1, Usource[nu], phi+_GSI(source_location));
                  _fv_eq_gamma_ti_fv(spinor2, nu, spinor1);
                  _fv_pl_eq_fv(spinor2, spinor1);
                  contact_term[2*nu  ] += 0.5 * spinor2[2*(3*ir+ia)  ];
                  contact_term[2*nu+1] += 0.5 * spinor2[2*(3*ir+ia)+1];
                }
              /* } */
      
      
              }  /* of ia */
            }    /* of ir */
          }      /* of nu */


        }  /* end of loop on iflavor */
  
        /* print contact term */
        if(g_cart_id == have_source_flag) {
          fprintf(stdout, "# [p2gg_xspace_ama] contact term\n");
          for(i=0;i<4;i++) {
            fprintf(stdout, "\t%d%25.16e%25.16e\n", i, contact_term[2*i], contact_term[2*i+1]);
          }
        }

      }  /* end of loop on sequential source timeslice */

      finit_3level_buffer(&gsp_u);
      finit_3level_buffer(&gsp_d);


      /* combine up-type and dn-type part, normalisation of contractions */

      items = 16 * VOLUME * T_global;
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(ix) firstprivate(items)
#endif
      for(ix=0; ix<items; ix++) {
        /* real part */
        conn_buffer[0][0][2*ix  ] += sequential_source_gamma_id_sign[ sequential_source_gamma_id ] * conn_buffer[1][0][2*ix  ];
        conn_buffer[0][0][2*ix  ] *= -0.25;

        /* imaginary part */
        conn_buffer[0][0][2*ix+1] -= sequential_source_gamma_id_sign[ sequential_source_gamma_id ] * conn_buffer[1][0][2*ix+1];
        conn_buffer[0][0][2*ix+1] *= -0.25;
      }
      fini_2level_buffer ( &(conn_buffer[1]) );
      fini_2level_buffer ( &(contact_term_buffer[1]) );

      retime = _GET_TIME;
      if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace_ama] time for contractions %e seconds\n", retime-ratime);

#ifdef HAVE_MPI
      if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace_ama] broadcasing contact term ...\n");
      MPI_Bcast(contact_term[0][0], 8*T_global, MPI_DOUBLE, have_source_flag, g_cart_grid);
#endif

      /**********************************************************
       * subtract contact term
       **********************************************************/
      if(have_source_flag == g_cart_id) {
        ix = g_ipt[sx0][sx1][sx2][sx3];
        for(i_sequential_source_timeslice=0; i_sequential_source_timeslice<T_global; i_sequential_source_timeslice++) {
          conn         = conn_buffer[0][i_sequential_source_timeslice];
          contact_term = contact_term_buffer[0][i_sequential_source_timeslice];

          conn[_GWI( 0,ix, VOLUME)  ] -= contact_term[ 0];
          conn[_GWI( 0,ix, VOLUME)+1] -= contact_term[ 1];
          conn[_GWI( 5,ix, VOLUME)  ] -= contact_term[ 2];
          conn[_GWI( 5,ix, VOLUME)+1] -= contact_term[ 3];
          conn[_GWI(10,ix, VOLUME)  ] -= contact_term[ 4];
          conn[_GWI(10,ix, VOLUME)+1] -= contact_term[ 5];
          conn[_GWI(15,ix, VOLUME)  ] -= contact_term[ 6];
          conn[_GWI(15,ix, VOLUME)+1] -= contact_term[ 7];
        }
      }
        
      /**********************************************
       * save position space results
       **********************************************/
      if(write_position_space_contractions) {
        ratime = _GET_TIME;
 
        for(i_sequential_source_timeslice=0; i_sequential_source_timeslice<T_global; i_sequential_source_timeslice++) {
          g_sequential_source_timeslice = (gsx[0] + i_sequential_source_timeslice )%T_global;

          conn = conn_buffer[0][i_sequential_source_timeslice];
          contact_term = contact_term_buffer[0][i_sequential_source_timeslice];

          if(strcmp(g_outfile_prefix, "NA") == 0) {
            sprintf(filename, "%s_x.%s.tseq%.2d.%.4d", outfile_name, outfile_tag, g_sequential_source_timeslice, Nconf);
          } else {
            sprintf(filename, "%s/%s_x.%s.tseq%.2d.%.4d", g_outfile_prefix, outfile_name, outfile_tag, g_sequential_source_timeslice, Nconf);
          }
          sprintf(contype, "\n<description>P - cvc - cvc in position space, 4x4 components</description>\n"\
                "<source_coords_t>%2d</source_coords_t>\n"\
                "<source_coords_x>%2d</source_coords_x>\n"\
                "<source_coords_y>%2d</source_coords_y>\n"\
                "<source_coords_z>%2d</source_coords_z>\n"\
                "<seq_source_momentum_x>%2d</seq_source_momentum_x>\n"\
                "<seq_source_momentum_y>%2d</seq_source_momentum_y>\n"\
                "<seq_source_momentum_z>%2d</seq_source_momentum_z>\n"\
                "<seq_source_gamma>g%.2d</seq_source_gamma>\n"\
                "<seq_source_timeslice>%.2d</seq_source_timeslice>\n"\
                "<contact_term_t>%25.16e%25.16e</contact_term_t>\n"\
                "<contact_term_x>%25.16e%25.16e</contact_term_x>\n"\
                "<contact_term_y>%25.16e%25.16e</contact_term_y>\n"\
                "<contact_term_z>%25.16e%25.16e</contact_term_z>\n",
                gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
                sequential_source_gamma_id, g_sequential_source_timeslice, 
                contact_term[0], contact_term[1],
                contact_term[2], contact_term[3],
                contact_term[4], contact_term[5],
                contact_term[6], contact_term[7]);
    
          write_lime_contraction(conn, filename, 64, 16, contype, Nconf, 0);
        }  /* end of loop on sequential source timeslices */

        retime = _GET_TIME;
        if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace_ama] time for saving position space results = %e seconds\n", retime-ratime);

      }  /* end of if write position space contractions */
/*
      for(i_sequential_source_timeslice=0; i_sequential_source_timeslice<T_global; i_sequential_source_timeslice++) {
        for(ix=0;ix<VOLUME;ix++) {
          for(mu=0;mu<16;mu++) {
            fprintf(stdout, "%2d%3d%6d%3d%25.16e%25.16e\n", g_cart_id, i_sequential_source_timeslice, ix, mu,
                conn_buffer[0][i_sequential_source_timeslice][_GWI(mu,ix,VOLUME)], conn_buffer[0][i_sequential_source_timeslice][_GWI(mu,ix,VOLUME)+1]);
          }
        }
      }
*/
      /********************************************
       * check the Ward identity in position space 
       ********************************************/
      if(check_position_space_WI) {
        bytes  = 32*(VOLUME+RAND)*sizeof(double);
        conn = (double*)malloc( bytes );
        if(conn == NULL ) {
          fprintf(stderr, "[] Error from malloc\n");
          EXIT(146);
        }

        for(i_sequential_source_timeslice=0; i_sequential_source_timeslice<T_global; i_sequential_source_timeslice++) {

          ratime = _GET_TIME;
          memcpy( conn, conn_buffer[0][i_sequential_source_timeslice], bytes );
#ifdef HAVE_MPI
          xchange_contraction(conn, 32);
#endif
          if(g_cart_id == 0) fprintf(stdout, "# [p2gg_xspace_ama] checking Ward identity in position space\n");
          for(x0=0; x0<T;  x0++) {
          for(x1=0; x1<LX; x1++) {
          for(x2=0; x2<LY; x2++) {
          for(x3=0; x3<LZ; x3++) {
            /* fprintf(stdout, "# t=%2d x=%2d y=%2d z=%2d\n", x0+g_proc_coords[0] * T, x1+g_proc_coords[1]*LX, x2+g_proc_coords[2]*LY, x3+g_proc_coords[3]*LZ); */
            ix=g_ipt[x0][x1][x2][x3];
            for(nu=0; nu<4; nu++) {
              w.re = conn[_GWI(4*0+nu,ix          ,VOLUME)  ] + conn[_GWI(4*1+nu,ix          ,VOLUME)  ]
                   + conn[_GWI(4*2+nu,ix          ,VOLUME)  ] + conn[_GWI(4*3+nu,ix          ,VOLUME)  ]
      	     - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)  ] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)  ]
      	     - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)  ] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)  ];
      
              w.im = conn[_GWI(4*0+nu,ix          ,VOLUME)+1] + conn[_GWI(4*1+nu,ix          ,VOLUME)+1]
                   + conn[_GWI(4*2+nu,ix          ,VOLUME)+1] + conn[_GWI(4*3+nu,ix          ,VOLUME)+1]
                   - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)+1] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)+1]
                   - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)+1] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)+1];
            
              dtmp += w.re*w.re + w.im*w.im;
            }
          }}}}
 #ifdef HAVE_MPI
          double dtmp2;
          MPI_Allreduce(&dtmp, &dtmp2, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid);
          dtmp = dtmp2;
#endif
          dtmp = sqrt(dtmp);
          if(g_cart_id == 0) fprintf(stdout, "# [] WI %2d norm = %e\n", i_sequential_source_timeslice, dtmp);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [] time to check Ward identity in position space = %e seconds\n", retime-ratime);

        }  /* end of loop on sequential source timeslices */
        free(conn); conn = NULL;
      }  /* end of if check_position_space_WI */
 
      /********************************************
       * momentum projections
       ********************************************/
      ratime = _GET_TIME;
      /* re-order */
      init_2level_buffer(&(conn_buffer[1]), 16*T_global*T, VOL3);
      for(i_sequential_source_timeslice=0; i_sequential_source_timeslice<T_global; i_sequential_source_timeslice++) {

        for(x0=0; x0<T;  x0++) {

          for(ix=0; ix<VOL3; ix++) {
            iix = x0 * VOL3 + ix;

            for(mu=0; mu<16; mu++) {
              conn_buffer[1][ (mu*T_global + i_sequential_source_timeslice) * T + x0][2*ix  ] = conn_buffer[0][i_sequential_source_timeslice][_GWI(mu,iix,VOLUME)  ];
              conn_buffer[1][ (mu*T_global + i_sequential_source_timeslice) * T + x0][2*ix+1] = conn_buffer[0][i_sequential_source_timeslice][_GWI(mu,iix,VOLUME)+1];
            }
          }  /* end of loop on VOL3 */
        }  /* end of loop on T */
      }  /* end of loop on sequential source timeslice */
      fini_2level_buffer(&(conn_buffer[0]));
      init_2level_buffer(&(conn_buffer[0]), g_sink_momentum_number, 16*T_global*T);
                               
      exitstatus = momentum_projection (conn_buffer[1], conn_buffer[0], 16*T_global*T, g_sink_momentum_number, g_sink_momentum_list);
      if( exitstatus != 0) {
        fprintf(stderr, "[] Error from momentum_projection, status was %d\n", exitstatus);
        EXIT(147);
      }
      fini_2level_buffer(&(conn_buffer[1]));
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [] time for momentum projections = %e seconds\n", retime-ratime);

      /********************************************
       * write momentum space contractions to file
       ********************************************/
      if(io_proc == 2) {
        if(strcmp(g_outfile_prefix, "NA") == 0) {
          sprintf(filename, "%s_p.%s.%.4d", outfile_name, outfile_tag, Nconf);
        } else {
          sprintf(filename, "%s/%s_p.%s.%.4d", g_outfile_prefix, outfile_name, outfile_tag, Nconf);
        }
        ofs = fopen(filename, "w");
      }

      for(iproc = 0; iproc<g_nproc_t; iproc++) {
#ifdef HAVE_MPI
        if(iproc > 0) {
          int mcoords[4], mrank, mitems = g_sink_momentum_number * T * T_global * 32;
          if(io_proc == 2)  {
            mcoords[0] = iproc; mcoords[1] = 0; mcoords[2] = 0; mcoords[3] = 0;
            MPI_Cart_rank(g_cart_grid, mcoords, &mrank);
            fprintf(stdout, "# [] proc%.2d receiving from proc%.2d\n", g_cart_id, mrank);
            MPI_Recv(conn_buffer[1][0],  mitems, MPI_DOUBLE, mrank, iproc,   g_cart_grid, &mstatus);
          } else {
            if(g_proc_coords[0] == iproc && io_proc == 1 ) {
              mcoords[0] = 0; mcoords[1] = 0; mcoords[2] = 0; mcoords[3] = 0;
              MPI_Cart_rank(g_cart_grid, mcoords, &mrank);
              fprintf(stdout, "# [] proc%.2d sending to proc%.2d\n", g_cart_id, mrank);
              MPI_Send(conn_buffer[1][0],  mitems, MPI_DOUBLE, mrank, iproc,   g_cart_grid);
            }
          }
        }
#endif
        if(io_proc == 2) {
          items = g_sink_momentum_number * T * T_global * 32;
          if( fwrite(conn_buffer[1][0], sizeof(double), items) != items ) {
            fprintf(stderr, "[] Error from fwrite, could not write %lu items\n", items);
            EXIT(142);
          }
        }
      }  /* end of loop on g_nproc_t */

      fini_2level_buffer(&(conn_buffer[0]));
      fini_2level_buffer ( &(contact_term_buffer[0]) );

      sequential_source_gammaid_retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [] time for sequential source gammaid = %e\n", 
        sequential_source_gammaid_retime - sequential_source_gammaid_ratime);

    }  /* end of loop on sequential gamma id */

    sequential_source_momentum_retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [] time for sequential source momentum = %e\n", 
        sequential_source_momentum_retime - sequential_source_momentum_ratime);
  }  /* end of loop on sequential source momentum */



  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  for(i=0; i<no_eo_fields; i++) free(eo_spinor_field[i]);
  free(eo_spinor_field);
  free(eo_evecs_block[0]);
  free(eo_evecs_block[1]);
  for(i=0; i<4; i++) free(eo_spinor_work[i]);
  free_geometry();

  if(full_spinor_work_halo != NULL) free(full_spinor_work_halo);

  free( evecs_lambdaOneHalf );
  free( evecs_eval );

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  free(status);
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_xspace_ama] %s# [p2gg_xspace_ama] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_xspace_ama] %s# [p2gg_xspace_ama] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
