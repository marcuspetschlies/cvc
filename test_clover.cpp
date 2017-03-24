/****************************************************
 * test_clover.cpp
 *
 * Do 16. Jun 11:22:24 CEST 2016
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
#include "Q_clover_phi.h"
#include "scalar_products.h"
#include "ranlxd.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, status; 
  int i, k;
  int x0, x1, x2, x3;
  int y0, y1, y2, y3;
  int z0, z1, z2, z3;
  int isboundary, start_value_t, start_value_x, start_value_y, start_value_z;
  int xsrc[4], psrc[4], have_source = 0, xsrc_iseven;
  unsigned int ixsrc;
  int filename_set = 0;
  int ix, iix;
  int threadid, nthreads;
  int no_eo_fields;
  double plaq;
  double **clover=NULL, **mzz_up=NULL, **mzz_dn=NULL, **mzzinv_up=NULL, **mzzinv_dn=NULL;
/*  double U1[18], U2[18]; */
  int verbose = 0;
  char filename[200];

  FILE *ofs=NULL;
  complex w, w2;
  double **eo_spinor_field=NULL, *eo_spinor_work=NULL;
  double ratime, retime;
  double *spinor_ptr = NULL, *spinor_ptr2 = NULL;
  double *gauge_trafo=NULL, *gauge_ptr, U1[18], U2[18];
  double spinor1[24], norm, norm2;
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
  if(g_cart_id==0) fprintf(stdout, "# [test_clover] Reading input from file %s\n", filename);
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
    fprintf(stderr, "[test_clover] Error from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "[test_clover] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(i,threadid) 
{
  nthreads = omp_get_num_threads();
  threadid = omp_get_thread_num();
  fprintf(stdout, "# [test_clover] thread%.4d number of threads = %d\n", threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_clover] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  Vhalf = VOLUME / 2;
  VOL3  = LX*LY*LZ;

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [test_clover] reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix,"identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    /* status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq); */
    /* status = read_ildg_nersc_gauge_field(g_gauge_field, filename); */
    status = read_lime_gauge_field_doubleprec(filename);
    /* status = read_nersc_gauge_field(g_gauge_field, filename, &plaq); */
  }
  if(status != 0) {
    fprintf(stderr, "[test_clover] Error, could not read gauge field\n");
    EXIT(11);
  }
  xchange_gauge();
                        
  /* measure the plaquette */
  if(g_cart_id==0) fprintf(stdout, "# [test_clover] read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_clover] measured plaquette value 1st field: %25.16e\n", plaq);


  /* init and allocate spinor fields */

  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

  no_eo_fields = 7;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));
  for(i=0; i<no_eo_fields; i++) alloc_spinor_field(&eo_spinor_field[i], (VOLUME+RAND)/2);
  eo_spinor_work = eo_spinor_field[no_eo_fields-1];


#if 0
  /* clover_term (&clover, g_gauge_field, VOLUME); */
  _cm_eq_cm(U1, g_gauge_field);
  _cm_eq_antiherm_cm(U2, U1);
  for(i=0; i<9; i++) fprintf(stdout, "\tU1[%d,%d] = %25.16e + %25.16e*1.i\n", i/3+1, i%3+1, U1[2*i], U1[2*i+1]);
  for(i=0; i<9; i++) fprintf(stdout, "\tU2[%d,%d] = %25.16e + %25.16e*1.i\n", i/3+1, i%3+1, U2[2*i], U2[2*i+1]);
#endif


  /* calculate clover term */
  clover_term_init(&clover, 6);

  clover_term_init(&mzz_up, 6);
  clover_term_init(&mzz_dn, 6);
  clover_term_init(&mzzinv_up, 8);
  clover_term_init(&mzzinv_dn, 8);

  ratime = _GET_TIME;
  clover_term_eo (clover, g_gauge_field);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_clover] time for clover_term_eo = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (mzz_up, clover, g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (mzz_dn, clover, -g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (mzzinv_up, mzz_up);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (mzzinv_dn, mzz_dn);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

#if 0
  /* apply M_zz^-1 M_zz */
  g_seed = 10000 + g_cart_id;
  rlxd_init(2, g_seed);
  rangauss (eo_spinor_field[0], VOLUME*12);

  /* M_clover_zz_matrix ( eo_spinor_work, eo_spinor_field[0], mzz_up[1]); */
  M_clover_zz (eo_spinor_work, eo_spinor_field[0], g_mu, clover[1]);
  M_clover_zz_inv_matrix ( eo_spinor_field[1], eo_spinor_work, mzzinv_up[1]);
  spinor_field_norm_diff (&norm, eo_spinor_field[0], eo_spinor_field[1], Vhalf );
  if(g_cart_id == 0) fprintf(stdout, "# [test_clover] diff = %16.7e\n", norm);
#endif

#if 0
#if (defined PARALLELT) || (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  start_value_t = 1;
#else
  start_value_t = 0;
#endif
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  start_value_x = 1;
#else
  start_value_x = 0;
#endif
#if (defined PARALLELTXY) || (defined PARALLELTXYZ)
  start_value_y = 1;
#else
  start_value_y = 0;
#endif
#if (defined PARALLELTXYZ)
  start_value_z = 1;
#else
  start_value_z = 0;
#endif

  /* printf gauge field */
  sprintf(filename, "gauge_field.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  fprintf(ofs, "# [test_clover] gauge field\n");
  for(x0 = -start_value_t; x0 < T +start_value_t; x0++) {
  for(x1 = -start_value_x; x1 < LX+start_value_x; x1++) {
  for(x2 = -start_value_y; x2 < LY+start_value_y; x2++) {
  for(x3 = -start_value_z; x3 < LZ+start_value_z; x3++) {


    isboundary = 0;
    if(x0==-1 || x0== T) isboundary++;
    if(x1==-1 || x1==LX) isboundary++;
    if(x2==-1 || x2==LY) isboundary++;
    i#endiff(x3==-1 || x3==LZ) isboundary++;

    if(isboundary > 2) continue;

    y0=x0; y1=x1; y2=x2; y3=x3;
    if(x0==-1) y0=T +1;
    if(x1==-1) y1=LX+1;
    if(x2==-1) y2=LY+1;
    if(x3==-1) y3=LZ+1;

    z0 = ( x0 + g_proc_coords[0] * T  + T_global  ) % T_global;
    z1 = ( x1 + g_proc_coords[1] * LX + LX_global ) % LX_global;
    z2 = ( x2 + g_proc_coords[2] * LY + LY_global ) % LY_global;
    z3 = ( x3 + g_proc_coords[3] * LZ + LZ_global ) % LZ_global;

    ix = g_ipt[y0][y1][y2][y3];

      for(i=0; i<4; i++) {
        gauge_ptr = g_gauge_field+_GGI(ix,i);

        /* _co_eq_tr_cm(&w, gauge_ptr); */
        fprintf(ofs, "# [test_clover] p = %3d %3d %3d %3d   x = %3d %3d %3d %3d %3d %3d\n", g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3], 
            z0, z1, z2, z3, isboundary, i );

        for(k=0; k<9; k++) {
          fprintf(ofs, "\t%3d%3d%25.16e%25.16e\n", k/3, k%3, gauge_ptr[2*k], gauge_ptr[2*k+1]);
        }

      }
    }}}
  }
  fclose(ofs);
#endif
#if 0
  sprintf(filename, "clover_term.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  fprintf(ofs, "# [test_clover] clover term before gauge transform\n");
  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      iix = g_lexic2eosub[ix];
      for(i=0; i<6; i++) {
        if(g_iseven[ix]) {
          gauge_ptr = clover[0]+_GSWI(iix,i);
        } else {
          gauge_ptr = clover[1]+_GSWI(iix,i);
        }
        /* _co_eq_tr_cm(&w, gauge_ptr); */
        fprintf(ofs, "# [test_clover] x = %3d %3d %3d %3d\t%d\n", 
            x0+g_proc_coords[0]*T, x1+g_proc_coords[1]*LX, x2+g_proc_coords[2]*LY, x3+g_proc_coords[3]*LY, i);

        for(k=0; k<9; k++) {
          fprintf(ofs, "\t%3d%3d%25.16e%25.16e\n", k/3, k%3, gauge_ptr[2*k], gauge_ptr[2*k+1]);
        }

      }
    }}}
  }
  fclose(ofs);
#endif
#if 0
  rlxd_init(2, g_seed);
  init_gauge_trafo(&gauge_trafo, 1.0);

  apply_gauge_transform(g_gauge_field, gauge_trafo, g_gauge_field);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_clover] measured plaquette value after gt: %25.16e\n", plaq);

  ratime = _GET_TIME;
  clover_term_eo (clover[0], g_gauge_field);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_clover] time for clover_term_eo = %e seconds\n", retime-ratime);

  fprintf(stdout, "# [test_clover] clover term after gauge transform\n");
  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      iix = g_lexic2eosub[ix];
      for(i=0; i<6; i++) {
        if(g_iseven[ix]) {
          gauge_ptr = clover[0]+_GSWI(iix,i);
        } else {
          gauge_ptr = clover[1]+_GSWI(iix,i);
        }
        _co_eq_tr_cm(&w, gauge_ptr);
        fprintf(stdout, "# [test_clover] x = %3d %3d %3d %3d\t%d\t%25.16e%25.16e\n", x0, x1, x2, x3, i, w.re, w.im);

        _cm_eq_cm_dag_ti_cm(U1, gauge_trafo+18*ix, gauge_ptr);
        _cm_eq_cm_ti_cm(U2, U1, gauge_trafo+18*ix);

        for(k=0; k<9; k++) {
          /* fprintf(stdout, "\t%3d%3d%25.16e%25.16e\n", k/3, k%9, gauge_ptr[2*k], gauge_ptr[2*k+1]); */
          fprintf(stdout, "\t%3d%3d%25.16e%25.16e\n", k/3, k%3, U2[2*k], U2[2*k+1]);
        }

      }
    }}}
  }
#endif  /* of if 0 */


  /* read propagator */
  sprintf(filename, "%s.%.4d.00.%.2d.inverted", filename_prefix, Nconf, 0);
  if( read_lime_spinor(g_spinor_field[0], filename, 0) != 0 ) {
    fprintf(stderr, "[test_clover] Error, could not read file %s\n", filename);
    EXIT(9);
  }

  /* lexic to even-odd */
  ratime = _GET_TIME;
  spinor_field_lexic2eo (g_spinor_field[0], eo_spinor_field[0], eo_spinor_field[1]);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_clover] time for spinor_field_lexic2eo = %e seconds\n", retime-ratime);

  /* apply tmclover Dirac operator */
  ratime = _GET_TIME;
  Q_clover_phi_eo (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[0], eo_spinor_field[1], g_gauge_field, g_mu, eo_spinor_work, clover);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_clover] time for Q_clover_phi_eo = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  Q_clover_phi_matrix_eo (eo_spinor_field[4], eo_spinor_field[5], eo_spinor_field[0], eo_spinor_field[1], g_gauge_field, eo_spinor_work, mzz_up);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_clover] time for Q_clover_phi_matrix_eo = %e seconds\n", retime-ratime);


  /* Q_phi_eo (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[0], eo_spinor_field[1], g_gauge_field, g_mu, eo_spinor_field[4]); */

  /* xchange_field(g_spinor_field[0]); */
  /* Q_phi (g_spinor_field[1], g_spinor_field[0], g_mu); */
  /* Q_phi_tbc (g_spinor_field[1], g_spinor_field[0]); */

  /* source location */
  xsrc[0] = g_source_location / ( LX_global * LY_global * LZ_global);
  xsrc[1] = (g_source_location % ( LX_global * LY_global * LZ_global) ) / ( LY_global * LZ_global);
  xsrc[2] = (g_source_location % ( LY_global * LZ_global) ) / ( LZ_global);
  xsrc[3] = (g_source_location % ( LZ_global) );
  psrc[0] = xsrc[0] / T;
  psrc[1] = xsrc[1] / LX;
  psrc[2] = xsrc[2] / LY;
  psrc[3] = xsrc[3] / LZ;
  if( psrc[0] == g_proc_coords[0] && psrc[1] == g_proc_coords[1] && psrc[2] == g_proc_coords[2] && psrc[3] == g_proc_coords[3] ) {
    xsrc[0] -= g_proc_coords[0] * T;
    xsrc[1] -= g_proc_coords[1] * LX;
    xsrc[2] -= g_proc_coords[2] * LY;
    xsrc[3] -= g_proc_coords[3] * LZ;
    ixsrc = g_ipt[xsrc[0]][xsrc[1]][xsrc[2]][xsrc[3]];
    xsrc_iseven = g_iseven[ixsrc];
    ixsrc = g_lexic2eosub[ ixsrc ];
    have_source = 1;
    fprintf(stdout, "# [] proc %d has the source (%d, %d, %d, %d) = %u %d\n", g_cart_id, xsrc[0], xsrc[1], xsrc[2], xsrc[3], ixsrc, xsrc_iseven);
  } else {
    have_source = 0;
  }

#if 0
  /* printf */
  sprintf(filename, "spinor_diff.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      iix = g_lexic2eosub[ix];
      if(g_iseven[ix]) {
         spinor_ptr  = eo_spinor_field[2]+_GSI(iix);
         spinor_ptr2 = eo_spinor_field[4]+_GSI(iix);
      } else {
         spinor_ptr  = eo_spinor_field[3]+_GSI(iix);
         spinor_ptr2 = eo_spinor_field[5]+_GSI(iix);
      }
      _co_eq_fv_dag_ti_fv(&w, spinor_ptr, spinor_ptr);
      _co_eq_fv_dag_ti_fv(&w2, spinor_ptr2, spinor_ptr2);
      fprintf(ofs, "# [test_clover] x = %3d %3d %3d %3d %8d %8d\t%16.7e\t%16.7e\n", 
          x0+g_proc_coords[0]*T, x1+g_proc_coords[1]*LX, x2+g_proc_coords[2]*LY, x3+g_proc_coords[3]*LZ, ix, iix, sqrt(w.re), sqrt(w2.re));
      /* spinor_ptr = g_spinor_field[1]+_GSI(ix); */
      for(i=0; i<12; i++) {
        fprintf(ofs, "\t%3d%3d%25.16e%25.16e\t%25.16e%25.16e\n", i/3, i%3, spinor_ptr[2*i], spinor_ptr[2*i+1], spinor_ptr2[2*i], spinor_ptr2[2*i+1]);
      }
    }}}
  }
  fclose(ofs);
#endif


  if(have_source) {
    if(xsrc_iseven) {
      eo_spinor_field[4][_GSI(ixsrc)] -=1.;
    } else {
      eo_spinor_field[5][_GSI(ixsrc)] -=1.;
    }
  }
  spinor_scalar_product_re(&norm, eo_spinor_field[4], eo_spinor_field[4], Vhalf);
  spinor_scalar_product_re(&norm2, eo_spinor_field[5], eo_spinor_field[5], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_clover] norm diff = %e\n", sqrt(norm+norm2));


  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  for(i=0; i<no_eo_fields; i++) free(eo_spinor_field[i]);
  free(eo_spinor_field);

#if 0
  clover_term_fini(&clover);
  clover_term_fini(&mzzinv_up);
  clover_term_fini(&mzzinv_dn);
  clover_term_fini(&mzz_up);
  clover_term_fini(&mzz_dn);
#endif

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif


  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_clover] %s# [test_clover] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_clover] %s# [test_clover] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

