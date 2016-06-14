/****************************************************
 * test.c
 *
 * Do 6. Aug 16:43:59 CEST 2015
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

using namespace cvc;

void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, no_fields=0;
  int filename_set = 0;
  int have_source_flag = 0;
  /* int x0, x1, x2, x3, ix; */
  int sx0, sx1, sx2, sx3;
  int exitstatus;
  int source_proc_coords[4], source_proc_id = -1;
  int verbose = 0;
  char filename[100];
  /* double ratime, retime; */
  double plaq;
  double spinor1[24], spinor2[24], spinor3[24];
  double dtmp[2];
  /* double *phi=NULL, *chi=NULL; */
  /* complex w, w1; */
  /* FILE *ofs; */

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
#endif

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[test] T and L's must be set\n");
    usage();
  }


  if(init_geometry() != 0) {
    fprintf(stderr, "[test] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

#if 0
  ranlxd(spinor1, 24);
  ranlxd(spinor2, 24);
  _fv_eq_fv(spinor3, spinor1);
  ranlxd(dtmp, 2);
  /* _fv_eq_a_pl_ib_g5_ti_fv(spinor2, spinor1, dtmp[0], dtmp[1]); */
  /* _fv_pl_eq_a_pl_ib_g5_ti_fv(spinor1, spinor2, dtmp[0], dtmp[1]); */
  /* _fv_pl_eq_a_g5_pl_ib_ti_fv(spinor1, spinor2, dtmp[0], dtmp[1]); */

  fprintf(stdout, "a <- %25.16e;  b <- %25.16e;\n", dtmp[0], dtmp[1]);
  for(i=0; i<12; i++) {
    fprintf(stdout, "s1[%d,%d] <- %25.16e+1.i*%25.16e; s3[%d,%d] <- %25.16e+1.i*%25.16e; s2[%d,%d] <- %25.16e+1.i*%25.16e\n", 
        i/3+1, i%3+1, spinor3[2*i], spinor3[2*i+1],
        i/3+1, i%3+1, spinor1[2*i], spinor1[2*i+1],
        i/3+1, i%3+1, spinor2[2*i], spinor2[2*i+1]);
  }

/*
  _fv_eq_fv(spinor2, spinor1);
  _fv_ti_eq_g5 (spinor2);
  _fv_eq_gamma_ti_fv(spinor3, 5, spinor1);
  for(i=0; i<12; i++) {
    fprintf(stdout, "\t%3d%25.16e%25.16e\t%25.16e%25.16e\n", i, spinor2[2*i], spinor2[2*i+1], spinor3[2*i], spinor3[2*i+1]);
  }
*/
#endif
   


#if 0

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [test] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [cvc_exact] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(560);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(561);
  }
  if(g_gauge_field == NULL) {
    fprintf(stderr, "[test] Error, g_gauge_field is NULL\n");
    EXIT(563);
  }
#endif

#ifdef HAVE_MPI
  xchange_gauge();
#endif


  /* measure the plaquette */

  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test] measured plaquette value: %25.16e\n", plaq);

  /* allocate memory for the spinor fields */
  no_fields = 120;
  if( (g_spinor_field = (double**)calloc(no_fields, sizeof(double*)) ) == NULL ) {
    fprintf(stderr, "[] Error from calloc, exit\n");
    EXIT(5);
  } else {
    for(i=0; i<no_fields; i++) {
      if( alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND) != 0 ) {
        fprintf(stderr, "[] Error from alloc_spinor_field, exit\n");
        EXIT(6);
      }
    }
  }
#if 0
#endif  /* of if 0 */
  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/

  sx0 = g_source_location / ( LX_global * LY_global * LZ_global);
  sx1 = (g_source_location % ( LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  sx2 = (g_source_location % ( LY_global * LZ_global)) / LZ_global;
  sx3 = (g_source_location % LZ_global);
  source_proc_coords[0] = sx0 / T;
  source_proc_coords[1] = sx1 / LX;
  source_proc_coords[2] = sx2 / LY;
  source_proc_coords[3] = sx3 / LZ;
#ifdef HAVE_MPI
  MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
#else
  source_proc_id = 0;
#endif
  have_source_flag = (int)(g_cart_id == source_proc_id);
  if(have_source_flag==1) {
    fprintf(stdout, "# [test] process %2d has source location\n", source_proc_id);
    fprintf(stdout, "# [test] global source coordinates: (%3d,%3d,%3d,%3d)\n",  sx0, sx1, sx2, sx3);
    fprintf(stdout, "# [test] source proc coordinates:   (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0],
        source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }
  sx0 = sx0 % T;
  sx1 = sx1 % LX;
  sx2 = sx2 % LY;
  sx3 = sx3 % LZ;
  if(have_source_flag==1)
    fprintf(stdout, "# [test] local source coordinates:  (%3d,%3d,%3d,%3d)\n",  sx0, sx1, sx2, sx3);


#if 0
#if defined HAVE_MPI
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact2_xspace_xspace] broadcasing contact term ...\n");
  MPI_Bcast(contact_term, 8, MPI_DOUBLE, have_source_flag, g_cart_grid);
  fprintf(stdout, "[%2d] contact term = "\
      "(%e + I %e, %e + I %e, %e + I %e, %e +I %e)\n",
      g_cart_id, contact_term[0], contact_term[1], contact_term[2], contact_term[3],
      contact_term[4], contact_term[5], contact_term[6], contact_term[7]);
#endif
#endif


  /****************************************
   * free the allocated memory, finalize
   ****************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  if(no_fields > 0 && g_spinor_field != NULL) { 
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field);
  }
#endif

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
