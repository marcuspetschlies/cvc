/****************************************************
 * test_locality.cpp
 *
 * Wed Oct 26 18:42:34 CEST 2016
 *
 * PURPOSE:
 * - test low-mode construction of propagators and projections
 *   of propagators
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
#include "prepare_source.h"
#include "project.h"
#include "Q_phi.h"
#include "invert_Qtm.h"
#include "make_x_orbits.h"

#define EO_FLAG_EVEN 0
#define EO_FLAG_ODD  1

#define OP_ID_UP 0
#define OP_ID_DN 1

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
  
  char outfile_name[] = "test_locality";

  const double PI2 =  2. * M_PI;

  int c, i, k, mu, ia;
  int no_eo_fields = 0, no_eo_work_fields=0;
  int op_id = -1, flavor_sign=0;
  int filename_set = 0;
  int source_location, source_location_iseven;
  int x0, x1, x2, x3;
  unsigned int ix, Vhalf;
  int gsx[4];
  int sx0, sx1, sx2, sx3;
  int isimag[4];
  int gperm[5][4], gperm2[4][4];
  int check_position_space_WI=0;
  int nthreads=-1, threadid=-1;
  int exitstatus;
  int write_ascii=0;
  int source_proc_coords[4], source_proc_id = -1;
  size_t sizeof_eo_spinor_field;
  double ***gsp_u=NULL;
  /* double ***gsp_d=NULL; */
  int verbose = 0;
  char filename[100];
  char outfile_tag[200];
  double ratime, retime;
  double plaq, norm;
#ifndef HAVE_OPENMP
  double spinor1[24], spinor2[24], U_[18];
#endif
/*  double *gauge_trafo = NULL; */
  double **pcoeff=NULL;
  double ***p3coeff=NULL;
  complex w, w1;
  FILE *ofs;

  int evecs_num=0, evecs_eval_set = 0;
  double *evecs_lambdaOneHalf=NULL, *evecs_eval = NULL;
  double *eo_evecs_block[2], **eo_spinor_field=NULL, **eo_spinor_work=NULL, **full_spinor_work_halo=NULL;
  double **eo_evecs_field=NULL;

  unsigned int *xid=NULL, *xid_count=NULL, xid_nc=0, **xid_member=NULL;
  double *xid_val = NULL;


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
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
  if(filename_set==0) strcpy(filename, "p2gg.input");
  fprintf(stdout, "# [test_lm_propagators] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_lm_propagators] calling tmLQCD wrapper init functions\n");

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

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
/*  mpi_init_xchange_contraction(32); */

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_lm_propagators] Error from init_geometry\n");
    EXIT(2);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;
  sizeof_eo_spinor_field = 24*Vhalf*sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [test_lm_propagators] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_lm_propagators] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_lm_propagators] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error, &g_gauge_field is NULL\n");
    EXIT(5);
  }
#endif

#ifdef HAVE_MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_lm_propagators] measured plaquette value: %25.16e\n", plaq);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/
  op_id = OP_ID_UP;
  flavor_sign = 1;

  exitstatus = tmLQCD_init_deflator(op_id);
  if( exitstatus > 0) {
    fprintf(stderr, "[test_lm_propagators] Error from tmLQCD_init_deflator, status was %d\n", exitstatus);
    EXIT(9);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, op_id);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagators] Error from tmLQCD_get_deflator_params, status was %d\n", exitstatus);
    EXIT(30);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [test_lm_propagators] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [test_lm_propagators] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [test_lm_propagators] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [test_lm_propagators] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block[0] = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block[0] == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error, eo_evecs_block is NULL\n");
    EXIT(32);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[test_lm_propagators] Error, dimension of eigenspace is zero\n");
    EXIT(33);
  }

#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */


  /***********************************************
   * allocate memory for the spinor fields
   ***********************************************/
  /* (1) eigenvector blocks */
#ifndef HAVE_TMLQCD_LIBWRAPPER
  eo_evecs_block[0] = (double*)malloc(evecs_num * sizeof_eo_spinor_field);
  if(eo_evecs_block[0] == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error from malloc\n");
    EXIT(25);
  }
#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */

  eo_evecs_field = (double**)calloc(2*evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block[0];
  for(i=1; i<evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + 24*Vhalf;

  /* (2) fermion fields without halo sites */
  no_eo_fields = 12; /* two full propagators */
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));

  eo_spinor_field[0] = (double*)malloc( no_eo_fields * sizeof_eo_spinor_field);
  if(eo_spinor_field[0] == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error from calloc\n");
    EXIT(35);
  }
  for(i=1; i<no_eo_fields; i++) eo_spinor_field[i] = eo_spinor_field[i-1] + Vhalf*24;

  evecs_eval = (double*)malloc(evecs_num * sizeof(double));
  if(evecs_eval == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error from calloc\n");
    EXIT(63);
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
  sx0 = gsx[0] % T;
  sx1 = gsx[1] % LX;
  sx2 = gsx[2] % LY;
  sx3 = gsx[3] % LZ;
  source_proc_id = 0;
#ifdef HAVE_MPI
  source_proc_coords[0] = gsx[0] / T;
  source_proc_coords[1] = gsx[1] / LX;
  source_proc_coords[2] = gsx[2] / LY;
  source_proc_coords[3] = gsx[3] / LZ;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [test_lm_propagators] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx[0], gsx[1], gsx[2], gsx[3]);
    fprintf(stdout, "# [test_lm_propagators] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  exitstatus = MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  if(exitstatus !=  MPI_SUCCESS ) {
    fprintf(stderr, "[test_lm_propagators] Error from MPI_Cart_rank, status was %d\n", exitstatus);
    EXIT(9);
  }
#endif
  if( source_proc_id == g_cart_id) {
    fprintf(stdout, "# [test_lm_propagators] process %2d has source location\n", source_proc_id);
  }

  if( source_proc_id == g_cart_id) {
    source_location_iseven = g_iseven[g_ipt[sx0][sx1][sx2][sx3]];
    fprintf(stdout, "# [test_lm_propagators] source site (%d, %d, %d, %d) is even = %d\n", gsx[0], gsx[1], gsx[2], gsx[3], source_location_iseven);
  }

  /***********************************************************
   * check number of operators, maximally 2 for now
   ***********************************************************/
#ifdef HAVE_TMLQCD_LIBWRAPPER
  if(g_tmLQCD_lat.no_operators > 2) {
    if(g_cart_id == 0) fprintf(stderr, "[test_lm_propagators] Error, confused about number of operators, expected 2 operator (up-type, dn-type)\n");
    EXIT(9);
  }
#endif
  
#ifdef HAVE_TMLQCD_LIBWRAPPER
  for(i=0; i<evecs_num; i++) {
    evecs_eval[i] = ((double*)(g_tmLQCD_defl.evals))[2*i];
  }
#endif


  ratime = _GET_TIME;
  sprintf(outfile_tag, "%s", outfile_name);
  exitstatus = check_vvdagger_locality ( &(eo_evecs_block[0]), evecs_num, gsx, outfile_tag, &(eo_spinor_field[0]));
  if(exitstatus != 0) {
    if(g_cart_id == 0) fprintf(stderr, "[] Error from check_vvdagger_locality, status was %d\n", exitstatus);
    EXIT(3);
  }

  retime = _GET_TIME;
  if(g_cart_id == 0) {
    fprintf(stdout, "# [test_lm_propagators] time total = %e seconds\n", retime-ratime);
  }

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

  if( eo_spinor_field != NULL ) {
    if( eo_spinor_field[0] != NULL ) free(eo_spinor_field[0]);
    free(eo_spinor_field);
  }

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(eo_evecs_block[0]);
#endif

  if(eo_evecs_field        != NULL ) free(eo_evecs_field);

  free_geometry();

#ifndef HAVE_TMLQCD_LIBWRAPPER
  if( evecs_eval          != NULL ) free( evecs_eval );
#endif




#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
/*  mpi_fini_xchange_contraction(); */
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_lm_propagators] %s# [test_lm_propagators] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_lm_propagators] %s# [test_lm_propagators] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
