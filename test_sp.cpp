/****************************************************
 * test_sp.cpp
 *
 * So 5. Jun 18:15:21 CEST 2016
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#define MAIN_PROGRAM

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "set_default.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "laplace_linalg.h"
#include "hyp_smear.h"
#include "Q_phi.h"
#include "invert_Qtm.h"
#include "ranlxd.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, status; 
  int i;
  int x0, x1, x2, x3;
  int filename_set = 0;
  int ix, iix, it;
  int threadid, nthreads;
  int no_eo_fields;
  double dnorm, dnorm2;
  int verbose = 0;
  char filename[200];

  FILE *ofs=NULL;
/*
  size_t items, bytes;
*/
  complex *sp_eo_e=NULL, *sp_eo_o=NULL, *sp_le_e=NULL, *sp_le_o=NULL, w;
  complex znorm, znorm2;
  double **eo_spinor_field=NULL;
  double ratime, retime;
  double *buffer;
  unsigned int Vhalf, VOL3;

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

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_cart_id==0) fprintf(stdout, "# [test_sp] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_xspace] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  status = tmLQCD_invert_init(argc, argv, 1);
  if(status != 0) {
    EXIT(14);
  }
  status = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(status != 0) {
    EXIT(15);
  }
  status = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(status != 0) {
    EXIT(16);
  }
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize T etc. */
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
		  "# [%2d] LXstart      = %3d\n"\
                  "# [%2d] LY_global    = %3d\n"\
                  "# [%2d] LY           = %3d\n"\
		  "# [%2d] LYstart      = %3d\n",\
		  g_cart_id, g_cart_id, T_global, g_cart_id, T, g_cart_id, Tstart,
		             g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart,
		             g_cart_id, LY_global, g_cart_id, LY, g_cart_id, LYstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_sp] ERROR from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "[apply_Coo] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(i,threadid) 
{
  nthreads = omp_get_num_threads();
  threadid = omp_get_thread_num();
  fprintf(stdout, "# [test_sp] thread%.4d number of threads = %d\n", threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_sp] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  Vhalf = VOLUME / 2;
  VOL3  = LX*LY*LZ;

  /* init and allocate spinor fields */
  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);


  no_eo_fields = 4;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));
  for(i=0; i<no_eo_fields; i++) alloc_spinor_field(&eo_spinor_field[i], (VOLUME+RAND)/2);

#if 0
  g_seed = 10000 + g_cart_id;
  rlxd_init(2, g_seed);

  /* set the spinor field */
  rangauss (g_spinor_field[0], VOLUME*24);
  rangauss (g_spinor_field[1], VOLUME*24);

  spinor_field_lexic2eo (g_spinor_field[0], eo_spinor_field[0], eo_spinor_field[1]);
  spinor_field_lexic2eo (g_spinor_field[1], eo_spinor_field[2], eo_spinor_field[3]);
#endif
  sprintf(filename, "%s%.5d", filename_prefix, 0);
  if(g_cart_id == 0) fprintf(stdout, "# [apply_Coo] reading C_oo_sym eigenvector from file %s\n", filename);
  status = read_lime_spinor(g_spinor_field[0], filename, 0);
  if( status != 0) {
    fprintf(stderr, "[apply_Coo] Error from read_lime_spinor, status was %d\n", status);
    EXIT(1);
  }

  sprintf(filename, "%s%.5d", filename_prefix, 2);
  if(g_cart_id == 0) fprintf(stdout, "# [apply_Coo] reading C_oo_sym eigenvector from file %s\n", filename);
  status = read_lime_spinor(g_spinor_field[1], filename, 0);
  if( status != 0) {
    fprintf(stderr, "[apply_Coo] Error from read_lime_spinor, status was %d\n", status);
    EXIT(1);
  }

  spinor_field_lexic2eo (g_spinor_field[0], eo_spinor_field[0], eo_spinor_field[1]);
  spinor_field_lexic2eo (g_spinor_field[1], eo_spinor_field[2], eo_spinor_field[3]);

  sp_eo_e = (complex*)malloc(T*sizeof(complex));
  sp_eo_o = (complex*)malloc(T*sizeof(complex));
  sp_le_e = (complex*)malloc(T*sizeof(complex));
  sp_le_o = (complex*)malloc(T*sizeof(complex));
  if (  sp_eo_e == NULL || sp_eo_o == NULL ||  sp_le_e == NULL || sp_le_o == NULL ) {
    fprintf(stderr, "[test_sp] Error from malloc\n");
    EXIT(1);
  }
#if 0
  /* scalar product complex and real part */
  spinor_scalar_product_co(&znorm, g_spinor_field[1], g_spinor_field[1], VOLUME);
  spinor_scalar_product_re(&dnorm, g_spinor_field[1], g_spinor_field[1], VOLUME);

  znorm2.re = 0.;
  znorm2.im = 0.;
  dnorm2    = 0.;
  for(ix=0; ix<VOLUME; ix++) {
    _co_eq_fv_dag_ti_fv(&w, g_spinor_field[1]+_GSI(ix), g_spinor_field[1]+_GSI(ix));
    znorm2.re += w.re;
    znorm2.im += w.im;
    dnorm2    += w.re;
  }
  fprintf(stdout, "# [test_sp] co\t%25.16e +I %25.16e\t%25.16e +I %25.16e\n", znorm.re, znorm.im, znorm2.re, znorm2.im);
  fprintf(stdout, "# [test_sp] re\t%25.16e\t%25.16e\n", dnorm, dnorm2);
#endif


  /* t-dependent scalar product in eo ordering */
  ratime = _GET_TIME;
  eo_spinor_spatial_scalar_product_co(sp_eo_e, eo_spinor_field[0], eo_spinor_field[2], 0);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [] eo_sp time = %e\n", retime-ratime);

  ratime = _GET_TIME;
  eo_spinor_spatial_scalar_product_co(sp_eo_o, eo_spinor_field[1], eo_spinor_field[3], 1);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [] eo_sp time = %e\n", retime-ratime);

  /* t-dependent scalar product in lexic ordering */
  memset(sp_le_e, 0, T*sizeof(complex));
  memset(sp_le_o, 0, T*sizeof(complex));

  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt [x0][x1][x2][x3];
      _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), g_spinor_field[1]+_GSI(ix));
      if(g_iseven[ix]) {
        sp_le_e[x0].re += w.re;
        sp_le_e[x0].im += w.im;
      } else {
        sp_le_o[x0].re += w.re;
        sp_le_o[x0].im += w.im;
      }
    }}}
  }

#ifdef HAVE_MPI
  buffer = (double*)malloc(2*T*sizeof(double));
  if ( buffer == NULL ) {
    fprintf(stderr, "[test_sp] Error from malloc\n");
    EXIT(2);
  }
  memcpy(buffer, sp_le_e, 2*T*sizeof(double));
  MPI_Allreduce(buffer, sp_le_e, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);

  memcpy(buffer, sp_le_o, 2*T*sizeof(double));
  MPI_Allreduce(buffer, sp_le_o, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);

  free(buffer);
#endif

  /* compare */
  for(x0 = 0; x0 < T; x0++) {
    x1 = x0 + g_proc_coords[0] * T;
    fprintf(stdout, "e %2d\t%3d\t%25.16e%25.16e\t%25.16e%25.16e\n", g_cart_id, x1,
       sp_le_e[x0].re, sp_le_e[x0].im, sp_eo_e[x0].re, sp_eo_e[x0].im);
  }

  for(x0 = 0; x0 < T; x0++) {
    x1 = x0 + g_proc_coords[0] * T;
    fprintf(stdout, "o %2d\t%3d\t%25.16e%25.16e\t%25.16e%25.16e\n", g_cart_id, x1,
       sp_le_o[x0].re, sp_le_o[x0].im, sp_eo_o[x0].re, sp_eo_o[x0].im);
  }


  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);

  for(i=0; i<no_eo_fields; i++) free(eo_spinor_field[i]);
  free(eo_spinor_field);

  free(sp_eo_e);
  free(sp_eo_o);
  free(sp_le_e);
  free(sp_le_o);

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_sp] %s# [test_sp] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_sp] %s# [test_sp] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

