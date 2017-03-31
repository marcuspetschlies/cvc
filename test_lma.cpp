/****************************************************
 * test_lma.cpp
 *
 * Di 31. Mai 08:36:30 CEST 2016
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
#include "scalar_products.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, status;
  int i;
  int filename_set = 0;
  int ix;
  /* int threadid, nthreads; */
  int no_eo_fields;
  int gsx0, gsx1, gsx2, gsx3;
  int lsx0, lsx1, lsx2, lsx3;

  double norm, norm2;

  double plaq=0.;
  int verbose = 0;
  char filename[200];
/*
  FILE *ofs=NULL;
  size_t items, bytes;
*/
  double **eo_spinor_field=NULL;
  double ratime, retime;
  unsigned int Vhalf;
  int source_proc_coords[4], source_proc_id=0;
  int l_source_location;

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
  if(g_cart_id==0) fprintf(stdout, "# [apply_QSchur] Reading input from file %s\n", filename);
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
    fprintf(stderr, "[apply_QSchur] ERROR from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();


  Vhalf = VOLUME / 2;

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [apply_QSchur] reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix,"identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    // status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq);
    // status = read_ildg_nersc_gauge_field(g_gauge_field, filename);
    status = read_lime_gauge_field_doubleprec(filename);
    // status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
  }
  if(status != 0) {
    fprintf(stderr, "[apply_QSchur] Error, could not read gauge field\n");
    EXIT(11);
  }
  xchange_gauge();

  /* measure the plaquette */
  if(g_cart_id==0) fprintf(stdout, "# [apply_QSchur] read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [apply_QSchur] measured plaquette value 1st field: %25.16e\n", plaq);

  /* init and allocate spinor fields */
  no_fields = 3;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

  no_eo_fields = 5;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));
  for(i=0; i<no_eo_fields; i++) alloc_spinor_field(&eo_spinor_field[i], (VOLUME+RAND)/2);


  /***********************************************
   * read eo eigenvector
   ***********************************************/

  /* strcpy(filename, filename_prefix); */
  sprintf(filename, "%s", filename_prefix2);
  if(g_cart_id == 0) fprintf(stdout, "# [apply_QSchur] reading C_oo_sym eigenvector from file %s\n", filename);

  status = read_lime_spinor(g_spinor_field[0], filename, 0);
  if( status != 0) {
    fprintf(stderr, "[apply_QSchur] Error from read_lime_spinor, status was %d\n");
    EXIT(1);
  }

  /* random_spinor_field (g_spinor_field[0], VOLUME); */


  gsx0 = g_source_location / (LX_global * LY_global * LZ_global);
  gsx1 = (g_source_location % (LX_global * LY_global * LZ_global) ) / (LY_global * LZ_global);
  gsx2 = (g_source_location % (LY_global * LZ_global) ) / LZ_global;
  gsx3 = g_source_location % LZ_global;
  if(g_cart_id == 0) fprintf(stdout, "# [apply_QSchur] global source coordinates = (%d, %d, %d, %d)\n", gsx0, gsx1, gsx2, gsx3);
  source_proc_coords[0] = gsx0 / T;
  source_proc_coords[1] = gsx1 / LX;
  source_proc_coords[2] = gsx2 / LY;
  source_proc_coords[3] = gsx3 / LZ;

  lsx0 = gsx0 % T;
  lsx1 = gsx1 % LX;
  lsx2 = gsx2 % LY;
  lsx3 = gsx3 % LZ;

  l_source_location = g_ipt[lsx0][lsx1][lsx2][lsx3];

#ifdef HAVE_MPI
  MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
#endif
  if(source_proc_id == g_cart_id) {
    fprintf(stdout, "# [apply_QSchur] process %d has the source at %d with local coordinates = (%d, %d, %d, %d)\n", source_proc_id, l_source_location, lsx0, lsx1, lsx2, lsx3);
  }


  /* apply Q on even-odd decomposed field */

  /* xchange_field(g_spinor_field[0]); */

  spinor_field_lexic2eo (g_spinor_field[0], eo_spinor_field[0], eo_spinor_field[1]);

#ifdef HAVE_MPI
  xchange_eo_field(eo_spinor_field[0], 0);
  xchange_eo_field(eo_spinor_field[1], 1);
#endif

  /* apply Q in Schur-decompositon on even-odd decomosed field */

  Q_eo_SchurDecomp_B (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[0], eo_spinor_field[1], g_gauge_field, g_mu, eo_spinor_field[4]);

#ifdef HAVE_MPI
  xchange_eo_field(eo_spinor_field[2], 0);
  xchange_eo_field(eo_spinor_field[3], 1);
#endif

  Q_eo_SchurDecomp_A (eo_spinor_field[0], eo_spinor_field[1], eo_spinor_field[2], eo_spinor_field[3], g_gauge_field, g_mu, eo_spinor_field[4]);

  if(source_proc_id == g_cart_id) {
    ix = l_source_location;
    fprintf(stdout, "# [apply_QSchur] g_lexic2eosub = %d, g_lexic2eo = %d, VOUME = %d, RAND = %d\n", g_lexic2eosub[ix], g_lexic2eo[ix], VOLUME, RAND);
    if(g_iseven[ix]) {
      eo_spinor_field[0][+_GSI(g_lexic2eosub[ix]) ] -= 1.;
    } else {
      eo_spinor_field[1][+_GSI(g_lexic2eosub[ix]) ] -= 1.;
    }
  }

  spinor_scalar_product_re(&norm,  eo_spinor_field[0], eo_spinor_field[0], Vhalf);
  spinor_scalar_product_re(&norm2, eo_spinor_field[1], eo_spinor_field[1], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [apply_QSchur] eo norm = %e; norm2 = %e, sum = %e\n", sqrt(norm), sqrt(norm2), sqrt(norm + norm2));


  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);

  for(i=0; i<no_eo_fields; i++) free(eo_spinor_field[i]);
  free(eo_spinor_field);

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [apply_QSchur] %s# [apply_QSchur] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [apply_QSchur] %s# [apply_QSchur] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

