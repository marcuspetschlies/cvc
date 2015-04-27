/****************************************************
 * apply_Dtm.c
 *
 * Friday, 06th of January 2012
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
#ifdef MPI
#  include <mpi.h>
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
#include "Q_phi.h"
#include "read_input_parser.h"


void usage() {
  fprintf(stdout, "Code to apply D to propagator, reconstruct source, check residuum\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, mu, nu, status;
  int i, j, ncon=-1;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int y0, y1, y2, y3;
  int start_valuet=0, start_valuex=0, start_valuey=0;
  double diff1, diff2, buffer;
/*  double *chi=NULL, *psi=NULL; */
  double plaq, pl_ts, pl_xs, pl_global;
  double *smeared_gauge_field = NULL;
  double s[18], t[18], u[18], pl_loc;
  double spinor1[24], spinor2[24];
  double *pl_gather=NULL;
  complex prod, w;
  int verbose = 0;
  char filename[200];
  char file1[200];
  char file2[200];
  FILE *ofs=NULL;

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:N:c:C:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      ncon = atoi(optarg);
      break;
    case 'c':
      strcpy(file1, optarg);
      break;
    case 'C':
      strcpy(file2, optarg);
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
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);


  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stdout, "kappa should be > 0.n");
    usage();
  }

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
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(101);
  }

  geometry();

  /* read the gauge field */
  alloc_gauge_field(&cvc_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# reading gauge field from file %s\n", filename);

  read_lime_gauge_field_doubleprec(filename);
  xchange_gauge();

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value: %25.16e\n", plaq);

#ifdef MPI
  start_valuet = 1;
#endif

#if defined PARALLELTXY
  start_valuex = 1;
  start_valuey = 1;
#endif

#if defined PARALLELTX
  start_valuex = 1;
#endif

/*
  if(g_cart_id==0) fprintf(stdout, "# start_values = (%d, %d, %d)\n", start_valuet, start_valuex, start_valuey);
  sprintf(filename, "gauge_ascii.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  for(x0=-start_valuet; x0< T+start_valuet; x0++) {
  for(x1=-start_valuex; x1<LX+start_valuex; x1++) {
  for(x2=-start_valuey; x2<LY+start_valuey; x2++) {
  for(x3=0; x3<LZ; x3++) {
    y0 = x0; y1 = x1; y2 = x2; y3 = x3;
    if(y0==-1) y0 =  T+1;
    if(y1==-1) y1 = LX+1;
    if(y2==-1) y2 = LY+1;
    iix = g_ipt[y0][y1][y2][y3];
    if(iix==-1) continue;
    fprintf(ofs, "# x0=%3d, x1=%3d, x2=%3d, x3=%3d, iix=%5d\n", x0, x1, x2, x3, iix);
    for(i=0; i<4; i++) {
      ix = _GGI(iix, i);
      fprintf(ofs, "# \t direction i=%3d\n", i);
      for(j=0; j<9; j++) {
        fprintf(ofs, "%3d%25.16e%25.16e\n", j, cvc_gauge_field[ix+2*j], cvc_gauge_field[ix+2*j+1]);
      }
    }
  }}}}
  fclose(ofs);
*/
/*
  for(i=0; i<N_ape; i++) {
    APE_Smearing_Step(cvc_gauge_field, alpha_ape);
    xchange_gauge_field_timeslice(cvc_gauge_field);
  }
  xchange_gauge();
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value for smeared gauge field: %25.16e\n", plaq);
*/
/*
  for(i=0; i<N_ape; i++) {
    APE_Smearing_Step(cvc_gauge_field, alpha_ape);
    xchange_gauge_field_timeslice(cvc_gauge_field);
  }
*/


/*
  alloc_gauge_field(&smeared_gauge_field, VOLUMEPLUSRAND);
  memcpy((void*)smeared_gauge_field, (void*)cvc_gauge_field, VOLUMEPLUSRAND*72*sizeof(double));

  fuzzed_links2(cvc_gauge_field, smeared_gauge_field, Nlong); 
  xchange_gauge();
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value after fuzzing: %25.16e\n", plaq);
*/

  no_fields=2;
  cvc_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&cvc_spinor_field[i], VOLUME+RAND);
/*
  sprintf(file1, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix2, Nconf, g_source_timeslice, 0);
  fprintf(stdout, "# Reading prop. from file %s\n", file1);
  read_lime_spinor(cvc_spinor_field[0], file1, 0);
  xchange_field(cvc_spinor_field[0]);

  memcpy((void*)cvc_spinor_field[2], (void*)cvc_spinor_field[0], VOLUME*24*sizeof(double));


  for(i=0; i<N_Jacobi; i++) {
    Jacobi_Smearing_Step_one(cvc_gauge_field, cvc_spinor_field[0], cvc_spinor_field[1], kappa_Jacobi);
    xchange_field_timeslice(cvc_spinor_field[0]);
  }
*/

/*
  status = Fuzz_prop3(cvc_gauge_field, cvc_spinor_field[0], cvc_spinor_field[1], Nlong);
  if(status != 0) {
    fprintf(stderr, "[%2d] Error from Fuzz_prop3\n", g_cart_id);
  }
*/
/*
  prod.re = 0.;
  prod.im = 0.;
  for(ix=0; ix<VOLUME; ix++) {
    _co_eq_fv_dag_ti_fv(&w, cvc_spinor_field[2]+_GSI(ix), cvc_spinor_field[0]+_GSI(ix));
    prod.re += w.re;
    prod.im += w.im;
  }

#ifdef MPI
  MPI_Allreduce(&prod, &w, 2, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  prod.re = w.re / (double)(VOLUME*g_nproc_t*g_nproc_x*g_nproc_y);
  prod.im = w.im / (double)(VOLUME*g_nproc_t*g_nproc_x*g_nproc_y);
#else
  prod.re /= (double)VOLUME;
  prod.im /= (double)VOLUME;
#endif

  if(g_cart_id==0) fprintf(stdout, "# prod = %25.16e + I %25.16e\n", prod.re, prod.im);
*/

/*
  pl_loc=0;
            
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<3; mu++) {
    for(nu=mu+1; nu<4; nu++) {
      _cm_eq_cm_ti_cm(s, &cvc_gauge_field[72*ix+mu*18], &cvc_gauge_field[72*g_iup[ix][mu]+18*nu]);
      _cm_eq_cm_ti_cm(t, &cvc_gauge_field[72*ix+nu*18], &cvc_gauge_field[72*g_iup[ix][nu]+18*mu]);
      _cm_eq_cm_ti_cm_dag(u, s, t);
      _co_eq_tr_cm(&w, u);
      pl_loc += w.re;
    }
    }
  }

#ifdef MPI
  MPI_Reduce(&pl_loc, &pl_global, 1, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid);
#else
  pl_global = pl_loc;
#endif
  pl_global = pl_global / ((double)T_global * (double)(LX*g_nproc_x) * (double)(LY*g_nproc_y) * (double)LZ * 18.);

  if(g_cart_id==0) fprintf(stdout, "# plaquette value is %25.16e\n", pl_global);

#if (defined PARALLELTX) || (defined PARALLELTXY)

  MPI_Allreduce(&pl_loc, &pl_ts, 1, MPI_DOUBLE, MPI_SUM, g_ts_comm);
  if(g_ts_id==0) fprintf(stdout, "# [%2d] plaquette on timeslice value is %25.16e\n", g_cart_id, pl_ts);
  
  MPI_Allreduce(&pl_ts, &pl_xs, 1, MPI_DOUBLE, MPI_SUM, g_xs_comm);
  if(g_xs_id==0) fprintf(stdout, "# [%2d] plaquette after summation along time ray: %25.16e\n", g_cart_id, pl_xs);


  pl_gather = (double*)calloc(g_xs_nproc, sizeof(double));
  if(pl_gather==NULL) {
    fprintf(stderr, "ERROR, could not allocate pl_gather\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(102);
  }

  pl_ts = 0.; pl_xs = 0.;
  for(ix=0; ix<g_xs_nproc; ix++) pl_gather[ix]=0.;

  MPI_Allreduce(&pl_loc, &pl_ts, 1, MPI_DOUBLE, MPI_SUM, g_ts_comm);
  if(g_ts_id==0) fprintf(stdout, "# [%2d] plaquette on timeslice value is %25.16e\n", g_cart_id, pl_ts);
  
  MPI_Allgather(&pl_ts, 1, MPI_DOUBLE, pl_gather, 1, MPI_DOUBLE, g_xs_comm);
  if(g_xs_id==0) {
    fprintf(stdout, "# [%2d] plaquette after time ray (%d,%d,%d):\n", g_cart_id, g_proc_coords[1], g_proc_coords[2],g_proc_coords[3]);
    for(ix=0; ix<g_xs_nproc; ix++) fprintf(stdout, "[%2d] %3d %25.16e\n", g_cart_id, ix, pl_gather[ix]);
  }
  free(pl_gather);
#endif
*/
  /****************************************
   * read read the spinor fields
   ****************************************/
  sprintf(file1, "%s", filename_prefix);
  if(g_cart_id==0) fprintf(stdout, "# Reading prop. from file %s\n", file1);
  read_lime_spinor(cvc_spinor_field[0], file1, 0);
//  read_lime_spinor(cvc_spinor_field[1], file1, 1);
  xchange_field(cvc_spinor_field[0]);
  Q_phi_tbc(cvc_spinor_field[1], cvc_spinor_field[0]);
 

  sprintf(filename, "src.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<12; mu++) {
      fprintf(ofs, "%6d%3d%25.16e%25.16e\n", ix, mu, cvc_spinor_field[1][_GSI(ix)+2*mu],
        cvc_spinor_field[1][_GSI(ix)+2*mu+1]);
    }
  }
  fclose(ofs);
 
  diff1 = 0.;
  diff2 = 0.;
  cvc_spinor_field[1][_GSI(g_source_location)+2*g_sourceid] -= 1.;
  for(ix=0; ix<VOLUME; ix++) {
    _co_eq_fv_dag_ti_fv(&w, cvc_spinor_field[1]+_GSI(ix), cvc_spinor_field[1]+_GSI(ix));
    diff1 += w.re;
  }
/*  for(ix=0; ix<24*VOLUME; ix++) {
    diff2 += fabs( cvc_spinor_field[1][ix] - cvc_spinor_field[0][ix] );
  }*/
 
  fprintf(stdout, "# [%.2d] res. squ. %25.16e\n", g_cart_id, diff1);
#ifdef MPI
  MPI_Reduce(&diff1, &buffer, 1, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid);
#else
  buffer = diff1;
#endif
  if(g_cart_id==0) fprintf(stdout, "# total res. squ. %25.16e\n", buffer);

#ifdef MPI
  MPI_Reduce(&diff2, &buffer, 1, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid);
#else
  buffer = diff2;
#endif
  if(g_cart_id==0) fprintf(stdout, "# total abs diff %25.16e\n", buffer);

/*
  fprintf(stdout, "[%2d] ===============================\n", g_cart_id);

  for(i=0; i<1; i++) {

    sprintf(file1, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix2, Nconf, g_source_timeslice, i);
    fprintf(stdout, "# Reading prop. from file %s\n", file1);
    read_lime_spinor(cvc_spinor_field[0], file1, 0);
    xchange_field(cvc_spinor_field[0]);

    sprintf(file2, "%s.%.4d.%.2d.%.2d", filename_prefix, Nconf, g_source_timeslice, i);
    fprintf(stdout, "# Reading source from file %s\n", file2);
    read_lime_spinor(cvc_spinor_field[2], file2, 0);
    xchange_field(cvc_spinor_field[2]);

    Q_phi_tbc(cvc_spinor_field[1], cvc_spinor_field[0]);

    chi    = cvc_spinor_field[1];
    psi    = cvc_spinor_field[2];

    sprintf(filename, "comp_%.2d_proc%.2d", i, g_cart_id);
    ofs = fopen(filename, "w");
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "Error, could not open %s for writing\n", filename);
#ifdef MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
    }
    if(g_cart_id==0) fprintf(stdout, "# writing ascii data to file %s\n", filename);
 
    for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      fprintf(ofs, "# x0=%3d, x1=%3d, x2=%3d, x3=%3d\n", (x0+Tstart), (x1+LXstart), (x2+LYstart), x3);
      ix = _GSI( g_ipt[x0][x1][x2][x3] );
      for(mu=0; mu<12; mu++) {
        fprintf(ofs, "%3d%25.16e%25.16e%25.16e%25.16e\n", mu, 
          chi[ix+2*mu], chi[ix+2*mu+1], psi[ix+2*mu], psi[ix+2*mu+1]);
      }
    }}}}
    fclose(ofs);
*/
    /****************************************
     * calculate difference
     ****************************************/
/*
    chi = cvc_spinor_field[1];
    psi = cvc_spinor_field[2];
   
    ncon = 12;
    mdiffre = fabs(chi[0] - psi[0]);
    mdiffim = fabs(chi[1] - psi[1]);
    Mdiffre = 0.;
    Mdiffim = 0.;
    adiffre = 0.;
    adiffim = 0.;
    for(ix=0; ix<ncon*VOLUME; ix++) {
      adiffre += chi[2*ix  ] - psi[2*ix  ];
      adiffim += chi[2*ix+1] - psi[2*ix+1];
      hre = fabs(chi[2*ix  ] - psi[2*ix  ]);
      him = fabs(chi[2*ix+1] - psi[2*ix+1]);
      if(hre<mdiffre) mdiffre = hre;
      if(hre>Mdiffre) Mdiffre = hre;
      if(him<mdiffim) mdiffim = him;
      if(him>Mdiffim) Mdiffim = him;
    }
    adiffre /= (double)VOLUME * (double)ncon;
    adiffim /= (double)VOLUME * (double)ncon;

    fprintf(stdout, "[%2d] Results for files Dtm %s and %s:\n"\
                    "[%2d] --------------------------------\n"\
                    "[%2d] average difference\t%25.16e\t%25.16e\n"\
                    "[%2d] minimal abs. difference\t%25.16e\t%25.16e\n"\
                  "[%2d] maximal abs. difference\t%25.16e\t%25.16e\n",
      g_cart_id, file1, file2, g_cart_id, g_cart_id, adiffre, adiffim, 
      g_cart_id, mdiffre, mdiffim, g_cart_id, Mdiffre, Mdiffim);

  }
*/
  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free(cvc_gauge_field);
  free_geometry();
/*
  if(smeared_gauge_field != NULL) free(smeared_gauge_field);
*/
  for(i=0; i<no_fields; i++) free(cvc_spinor_field[i]);
  free(cvc_spinor_field);

#ifdef MPI
  MPI_Finalize();
#endif
  return(0);

}

