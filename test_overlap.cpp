/****************************************************
 * test_overlap.c
 *
 * Mi 16. Mär 15:24:46 CET 2016
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
#ifdef OPENMP
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
#include "laphs_io.h"
#include "laphs_utils.h"
#include "laphs.h"
#include "Q_phi.h"
#include "invert_Qtm.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  const double preset_eigenvalue = 7.864396614243382E-06;

  int c, mu, nu, status, sid;
  int it_src = 1;
  int is_src = 2;
  int iv_src = 3;
  int i, j, ncon=-1, is, idx;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int y0, y1, y2, y3;
  int threadid, nthreads;
  int no_eo_fields;
  int gsx0, gsx1, gsx2, gsx3;
  int lsx0, lsx1, lsx2, lsx3;

  double dtmp[4], norm, norm2, norm3;

  double plaq=0.;
  double *gauge_field_smeared = NULL;
  int verbose = 0;
  char filename[200];
  FILE *ofs=NULL;
  size_t items, bytes;
  complex w, w2;
  double **perambulator = NULL;
  double **eo_spinor_field=NULL;
  double ratime, retime;
  eigensystem_type es;
  randomvector_type rv, prv;
  perambulator_type peram;
  unsigned int Vhalf;
  int source_proc_coords[4], source_proc_id=0;
  int l_source_location;
  double spinor1[24];
  double **coo_evecs = NULL, *coo_buffer=NULL;
  int coo_nev = 0, coo_prec=32;


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:N:p:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      coo_nev = atoi(optarg);
      fprintf(stdout, "# [] number of C_oo eigenvectors set to %d\n", coo_nev);
      break;
    case 'p':
      coo_prec = atoi(optarg);
      fprintf(stdout, "# [] using C_oo eigenvector precision %d\n", coo_prec);
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
  if(g_cart_id==0) fprintf(stdout, "# [test_overlap] Reading input from file %s\n", filename);
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
    fprintf(stderr, "[test_overlap] ERROR from init_geometry\n");
    EXIT(101);
  }

  geometry();

  Vhalf = VOLUME / 2;

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [test_overlap] reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix,"identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    // status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq);
    // status = read_ildg_nersc_gauge_field(g_gauge_field, filename);
    status = read_lime_gauge_field_doubleprec(filename);
    // status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
  }
  if(status != 0) {
    fprintf(stderr, "[test_overlap] Error, could not read gauge field\n");
    EXIT(11);
  }
  xchange_gauge();

  /* measure the plaquette */
  if(g_cart_id==0) fprintf(stdout, "# [test_overlap] read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_overlap] measured plaquette value 1st field: %25.16e\n", plaq);

#if 0
  /* smear the gauge field */
  status = hyp_smear_3d (g_gauge_field, N_hyp, alpha_hyp, 0, 0);
  if(status != 0) {
    fprintf(stderr, "[test_overlap] Error from hyp_smear_3d, status was %d\n", status);
    EXIT(7);
  }

  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_overlap] measured plaquette value ofter hyp smearing = %25.16e\n", plaq);

  sprintf(filename, "%s_hyp.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [test_overlap] writing hyp-smeared gauge field to file %s\n", filename);

  status = write_lime_gauge_field(filename, plaq, Nconf, 64);
  if(status != 0) {
    fprintf(stderr, "[apply_lapace] Error friom write_lime_gauge_field, status was %d\n", status);
    EXIT(7);
  }
#endif

  /* init and allocate spinor fields */
  no_fields = 3;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

  no_eo_fields = 5;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));
  for(i=0; i<no_eo_fields; i++) alloc_spinor_field(&eo_spinor_field[i], (VOLUME+RAND)/2);

  /* init_eigensystem(&es); */
#if 0
  status = alloc_eigensystem (&es, T, laphs_eigenvector_number);
  if(status != 0) {
    fprintf(stderr, "[test_overlap] Error from alloc_eigensystem, status was %d\n", status);
    EXIT(7);
  }

  ratime = _GET_TIME;
  status = read_eigensystem(&es);
  if (status != 0) {
    fprintf(stderr, "# [test_overlap] Error from read_eigensystem, status was %d\n", status);
  }
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_overlap] time to read eigensystem %e\n", retime-ratime);

/*
  ratime = _GET_TIME;
  status = test_eigensystem(&es, g_gauge_field);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_overlap] time to test eigensystem %e\n", retime-ratime);
*/

#endif

  /***********************************************
   * read eo eigenvector
   ***********************************************/
/* arpack_evecs.0740.00100.pt00px00py00pz00 */
  sprintf(filename, "%s.pt%.2dpx%.2dpy%.2dpz%.2d", filename_prefix, 
      g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);


  /* sprintf(filename, "%s", filename_prefix2); */
  if(g_cart_id == 0) fprintf(stdout, "# [test_overlap] reading C_oo_sym eigenvector from file %s\n", filename);

/*
  status = read_lime_spinor(g_spinor_field[0], filename, 0);
  if( status != 0) {
    fprintf(stderr, "[test_overlap] Error from read_lime_spinor, status was %d\n");
    EXIT(1);
  }
*/

  if (coo_prec == 32)  {
    bytes = sizeof(float);
  } else if(coo_prec == 64) {
    bytes = sizeof(double);
  }
  coo_buffer = (double*)malloc(coo_nev * VOLUME*12*sizeof(double) );
  if(coo_buffer == NULL) {
    fprintf(stderr, "[] Error, could not allocate buffer\n");
    EXIT(11);
  }

  coo_evecs = (double**)malloc(coo_nev * sizeof(double*) );
  if(coo_evecs == NULL) {
    fprintf(stderr, "[] Error, could not allocate evecs\n");
    EXIT(12);
  }
  for(i=0; i<coo_nev; i++) {
    coo_evecs[i] = coo_buffer + (size_t)i * VOLUME*12;
  }


  ofs = fopen(filename, "r");
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for reading\n", filename);
    EXIT(1);
  }

  items = fread(coo_buffer, bytes, coo_nev*12*VOLUME, ofs);
  if( items != ((size_t)coo_nev)*12*VOLUME ) {
    fprintf(stderr, "[] Error, could not read proper amount of data from file %s\n", filename);
    EXIT(2);
  }

  fclose(ofs);

  if(coo_prec == 32) {
    for(ix = 12*VOLUME*coo_nev-1; ix >= 0; ix--) {
      ((double*)coo_buffer)[ix] = (double) (((float*)coo_buffer)[ix]);
    }
  }

  /* random_spinor_field (g_spinor_field[0], VOLUME); */

#if 0
  gsx0 = g_source_location / (LX_global * LY_global * LZ_global);
  gsx1 = (g_source_location % (LX_global * LY_global * LZ_global) ) / (LY_global * LZ_global);
  gsx2 = (g_source_location % (LY_global * LZ_global) ) / LZ_global;
  gsx3 = g_source_location % LZ_global;
  if(g_cart_id == 0) fprintf(stdout, "# [test_overlap] global source coordinates = (%d, %d, %d, %d)\n", gsx0, gsx1, gsx2, gsx3);
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
    fprintf(stdout, "# [test_overlap] process %d has the source at %d with local coordinates = (%d, %d, %d, %d)\n", source_proc_id, l_source_location, lsx0, lsx1, lsx2, lsx3);
  }
#endif

#if 0
  xchange_field(g_spinor_field[0]);
  Q_phi(g_spinor_field[1], g_spinor_field[0], g_mu);
#endif

#if 0
  if(source_proc_id == g_cart_id) {
    g_spinor_field[1][_GSI( g_ipt[lsx0][lsx1][lsx2][lsx3] )] -= 1.;
  }

  spinor_scalar_product_re(&norm, g_spinor_field[1], g_spinor_field[1],VOLUME);
  spinor_scalar_product_re(&norm2, g_spinor_field[0], g_spinor_field[0],VOLUME);
  if(g_cart_id == 0) {
    fprintf(stdout, "# [test_overlap] lexic norm = %e; norm2 = %e\n", sqrt(norm), sqrt(norm2));
  }
#endif

  /* decompose into even and odd part */
  /* spinor_field_lexic2eo (g_spinor_field[0], eo_spinor_field[0], eo_spinor_field[1]); */

#if 0
  /* apply Hopping matrix in eo decompositon */
  Hopping_eo(eo_spinor_field[2], eo_spinor_field[1], g_gauge_field, 0);
  Hopping_eo(eo_spinor_field[3], eo_spinor_field[0], g_gauge_field, 1);

  /* combine eo to lexic */
  spinor_field_eo2lexic(g_spinor_field[2], eo_spinor_field[2], eo_spinor_field[3]);
#endif

#if 0
#ifdef HAVE_MPI
  xchange_field(g_spinor_field[0]);
#endif
  Hopping(g_spinor_field[1], g_spinor_field[0]);

  sprintf( filename, "Hopping_comp_proc%.2d", g_cart_id);
  ofs = fopen(filename, "w");

  for(ix=0; ix<VOLUME; ix++) {

    _fv_eq_fv_mi_fv(spinor1, g_spinor_field[1]+_GSI(ix), g_spinor_field[2]+_GSI(ix));
    _co_eq_fv_dag_ti_fv(&w, spinor1, spinor1);
    fprintf(ofs, "proc%.2d %8d %16.7e %16.7e\n", g_cart_id, ix, w.re, w.im);

/*
    fprintf(ofs, "# proc%.2d ix = %8d\n", g_cart_id, ix);
    for(i=0; i<12; i++) {
      fprintf(ofs,"proc%.2d %3d%16.7e%17.7e\t%16.7e%16.7e\n", g_cart_id, i,
          g_spinor_field[1][_GSI(ix)+2*i], g_spinor_field[1][_GSI(ix)+2*i+1],
          g_spinor_field[2][_GSI(ix)+2*i], g_spinor_field[2][_GSI(ix)+2*i+1]);
    }
*/
  }
  fclose(ofs);
#endif

#if 0
  /* apply Q on even-odd decomposed field */
  Q_phi_eo( eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[0], eo_spinor_field[1], g_gauge_field, g_mu, eo_spinor_field[4]);

  if(source_proc_id == g_cart_id) {
    ix = l_source_location;
    fprintf(stdout, "# [test_overlap] g_lexic2eosub = %d, g_lexic2eo = %d, VOUME = %d, RAND = %d\n", g_lexic2eosub[ix], g_lexic2eo[ix], VOLUME, RAND);
    if(g_iseven[ix]) {
      eo_spinor_field[2][+_GSI(g_lexic2eosub[ix]) ] -= 1.;
    } else {
      eo_spinor_field[3][+_GSI(g_lexic2eosub[ix]) ] -= 1.;
    }
  }

  spinor_scalar_product_re(&norm,  eo_spinor_field[2], eo_spinor_field[2], Vhalf);
  spinor_scalar_product_re(&norm2, eo_spinor_field[3], eo_spinor_field[3], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_overlap] eo norm = %e; norm2 = %e, sum = %e\n", sqrt(norm), sqrt(norm2), sqrt(norm + norm2));

  spinor_field_eo2lexic(g_spinor_field[2], eo_spinor_field[2], eo_spinor_field[3]);
  for(ix=0; ix<VOLUME; ix++) {
    _fv_mi_eq_fv(g_spinor_field[1]+_GSI(ix), g_spinor_field[2]+_GSI(ix));
  }
  spinor_scalar_product_re(&norm,  g_spinor_field[1], g_spinor_field[1], VOLUME);
  if(g_cart_id == 0) fprintf(stdout, "# [test_overlap] eo2lexic norm = %e\n", sqrt(norm));

#endif

#if 0
  /* apply Q in Schur-decompositon on even-odd decomosed field */

#ifdef HAVE_MPI
  xchange_eo_field( eo_spinor_field[0], 0);
  xchange_eo_field( eo_spinor_field[1], 1);
#endif

  Q_eo_SchurDecomp_B (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[0], eo_spinor_field[1], g_gauge_field, g_mu, eo_spinor_field[4]);

#ifdef HAVE_MPI
  xchange_eo_field(eo_spinor_field[2], 0);
  xchange_eo_field(eo_spinor_field[3], 1);
#endif

  Q_eo_SchurDecomp_A (eo_spinor_field[0], eo_spinor_field[1], eo_spinor_field[2], eo_spinor_field[3], g_gauge_field, g_mu, eo_spinor_field[4]);
#endif
#if 0
  if(source_proc_id == g_cart_id) {
    ix = l_source_location;
    fprintf(stdout, "# [test_overlap] g_lexic2eosub = %d, g_lexic2eo = %d, VOUME = %d, RAND = %d\n", g_lexic2eosub[ix], g_lexic2eo[ix], VOLUME, RAND);
    if(g_iseven[ix]) {
      eo_spinor_field[0][+_GSI(g_lexic2eosub[ix]) ] -= 1.;
    } else {
      eo_spinor_field[1][+_GSI(g_lexic2eosub[ix]) ] -= 1.;
    }
  }

  spinor_scalar_product_re(&norm,  eo_spinor_field[0], eo_spinor_field[0], Vhalf);
  spinor_scalar_product_re(&norm2, eo_spinor_field[1], eo_spinor_field[1], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_overlap] eo norm = %e; norm2 = %e, sum = %e\n", sqrt(norm), sqrt(norm2), sqrt(norm + norm2));
#endif
#if 0
  spinor_field_eo2lexic(g_spinor_field[2], eo_spinor_field[0], eo_spinor_field[1]);

  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_gamma_ti_fv(spinor1, 5, g_spinor_field[2]+_GSI(ix));
    _fv_mi_eq_fv(g_spinor_field[1]+_GSI(ix), spinor1);
  }
  spinor_scalar_product_re(&norm,  g_spinor_field[1], g_spinor_field[1], VOLUME);
  if(g_cart_id == 0) fprintf(stdout, "# [test_overlap] eo2lexic norm = %e\n", sqrt(norm));
#endif


  /* copy first eigenvector to eo_spinor_field */
  memcpy(eo_spinor_field[0],  coo_evecs[0], VOLUME*12*sizeof(double));

  /* apply C^+ C^- */
  xchange_eo_field( eo_spinor_field[0], 1);
#if 0
  C_oo(eo_spinor_field[1], eo_spinor_field[0], g_gauge_field, -g_mu, eo_spinor_field[4]);
  xchange_eo_field( eo_spinor_field[1], 1);
  C_oo(eo_spinor_field[2], eo_spinor_field[1], g_gauge_field,  g_mu, eo_spinor_field[4]);

  norm = 4 * g_kappa * g_kappa;
  for(ix=0; ix<Vhalf; ix++ ) {
    _fv_ti_eq_re(eo_spinor_field[2]+_GSI(ix), norm);
  }

  spinor_scalar_product_re(&norm,  eo_spinor_field[0], eo_spinor_field[0], Vhalf);
  spinor_scalar_product_co(&w,  eo_spinor_field[0], eo_spinor_field[2], Vhalf);
  
  if(g_cart_id == 0) {
    fprintf(stdout, "# [] eigenvalue = %16.7e %16.7e; norm = %16.7e\n", w.re / norm, w.im / norm, sqrt(norm));
  }
#endif
#if 0
  for(ix=0; ix<Vhalf; ix++ )
/*  for(ix=0; ix<VOLUME; ix++ ) */
  {
    fprintf(stdout, "# [test_overlap] ix = %u\n", ix);
/*
    _co_eq_fv_dag_ti_fv(&w, eo_spinor_field[1], eo_spinor_field[3]);
    _co_eq_fv_dag_ti_fv(&w2, eo_spinor_field[1], eo_spinor_field[1]);

    fprintf(stdout, "%8d \t %16.7e%16.7e \t %16.7e%16.7e \t %16.7e\n", ix, w.re, w.im, w2.re, w2.im, w.re / w2.re );
*/

    for(i=0; i<12; i++) {
/*
      fprintf(stdout, "\t%3d %16.7e %16.7e \t %16.7e %16.7e\n", i, 
          g_spinor_field[0][_GSI(ix)+2*i], g_spinor_field[0][_GSI(ix)+2*i+1],
          g_spinor_field[1][_GSI(ix)+2*i], g_spinor_field[1][_GSI(ix)+2*i+1] );

      fprintf(stdout, "\t%3d %16.7e %16.7e \t %16.7e %16.7e\n", i, 
          eo_spinor_field[1][_GSI(ix)+2*i], eo_spinor_field[1][_GSI(ix)+2*i+1],
          eo_spinor_field[3][_GSI(ix)+2*i], eo_spinor_field[3][_GSI(ix)+2*i+1] );
*/

      fprintf(stdout, "\t%3d %16.7e %16.7e \t %16.7e %16.7e\n", i, 
          eo_spinor_field[1][_GSI(ix)+2*i], eo_spinor_field[1][_GSI(ix)+2*i+1],
          eo_spinor_field[3][_GSI(ix)+2*i], eo_spinor_field[3][_GSI(ix)+2*i+1]);

    }
  }
#endif


#if 0
  sprintf( filename, "eo_indizes_proc%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  for (ix=0; ix<VOLUME/2; ix++) {
    i = g_eo2lexic[ix];
    x0 = i / (LX*LY*LZ);
    x1 = ( i % (LX*LY*LZ) )  / (LY*LZ);
    x2 = ( i % (LY*LZ) )  / LZ;
    x3 = i % LZ;
    j = g_eo2lexic[ix+(VOLUME+RAND)/2];
    y0 = j / (LX*LY*LZ);
    y1 = ( j % (LX*LY*LZ) )  / (LY*LZ);
    y2 = ( j % (LY*LZ) )  / LZ;
    y3 = j % LZ;
    fprintf(ofs, "proc%.2d %8d \t x=%8d  x0=%2d x1=%2d x2=%2d x3=%2d \t y=%8d y0=%2d y1=%2d y2=%2d y3=%2d\n", g_cart_id, ix, i, x0, x1, x2, x3, j, y0, y1, y2, y3);
  }
  fclose(ofs);
#endif

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  /* fini_eigensystem (&es); */


  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);

  for(i=0; i<no_eo_fields; i++) free(eo_spinor_field[i]);
  free(eo_spinor_field);

  free_geometry();

  if(coo_evecs != NULL) free(coo_evecs);
  if(coo_buffer != NULL) free(coo_buffer);


  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_overlap] %s# [test_overlap] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_overlap] %s# [test_overlap] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);
}

