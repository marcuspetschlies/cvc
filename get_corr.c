/****************************************************
  
 * get_corr.c
 *
 * Wed Sep 23 17:58:30 CEST 2009
 *
 * PURPOSE
 * - recover time and momentum dep. correlators from
 *   vacuum pol. tensor files
 *   file pattern:
 *     
 * - correlators: \Pi_00 and 1/3 (\Pi_11+\Pi_22+\Pi_33)
 * DONE:
 * TODO:
 * CHANGES:
 * - introduced modes m:
 *   0 - keep different 3-momenta, use minimal and maximal
 *       values of 3-momentum squared (lattice momenta) Q_mu = 2/a sin(aq_mu)
 *   1 - as 0, but with continuum momenta (q_mu = 2 pi n_mu / a / L_mu)
 *   2 - average over 3-momenta belonging to the same O_h class,
 *       use minimal and maximal values of 3-momentum squared 
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ifftw.h"
#include <getopt.h>

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
#include "get_index.h"
//#include "make_H3orbits.h"
#include "contractions_io.h"

void usage() {
  fprintf(stdout, "Code to recover rho-rho correl.\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, mu, status;
  int filename_set = 0;
  int mode = 0;
  int l_LX_at, l_LXstart_at;
  int x0, x1, x2, x3, ix, iix, iiy, gid;
  int source_location;
  int Thp1, nclass;
  int *oh_count=(int*)NULL, *oh_id=(int*)NULL, oh_nc;
  int *picount;
  double *conn = (double*)NULL;
  double *conn2 = (double*)NULL;
  double **oh_val=(double**)NULL;
  double q[4], qsqr;
  int verbose = 0;
  char filename[800];
  double ratime, retime;
  FILE *ofs;
  fftw_complex *corrt=NULL;

  fftw_complex *pi00=(fftw_complex*)NULL, *pijj=(fftw_complex*)NULL, *piavg=(fftw_complex*)NULL;

  fftw_plan plan_m;

  while ((c = getopt(argc, argv, "h?vf:m:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'm':
      mode = atoi(optarg);
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
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "T and L's must be set\n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize fftw, create plan with FFTW_FORWARD ---  in contrast to
   * FFTW_BACKWARD in e.g. avc_exact */
  plan_m = fftw_create_plan(T_global, FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  if(plan_m==NULL) {
    fprintf(stderr, "Error, could not create fftw plan\n");
    return(1);
  }

  T            = T_global;
  Thp1         = T/2 + 1;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(1);
  }

  geometry();

  /****************************************
   * allocate memory for the contractions *
   ****************************************/
  conn = (double*)calloc(32*VOLUME, sizeof(double));
  if( (conn==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
    exit(3);
  }

/*
  conn2 = (double*)calloc(32*VOLUME, sizeof(double));
  if( (conn2==(double*)NULL) ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
    exit(4);
  }

  pi00 = (fftw_complex*)malloc(VOLUME*sizeof(fftw_complex));
  if( (pi00==(fftw_complex*)NULL) ) {
    fprintf(stderr, "could not allocate memory for pi00\n");
    exit(2);
  }

  pijj = (fftw_complex*)fftw_malloc(VOLUME*sizeof(fftw_complex));
  if( (pijj==(fftw_complex*)NULL) ) {
    fprintf(stderr, "could not allocate memory for pijj\n");
    exit(2);
  }
*/
  corrt = fftw_malloc(T*sizeof(fftw_complex));

  for(gid=g_gaugeid; gid<=g_gaugeid2; gid+=g_gauge_step) {

//    for(ix=0; ix<VOLUME; ix++) {pi00[ix].re = 0.; pi00[ix].im = 0.;}
//    for(ix=0; ix<VOLUME; ix++) {pijj[ix].re = 0.; pijj[ix].im = 0.;}
    /***********************
     * read contractions   *
     ***********************/
    ratime = (double)clock() / CLOCKS_PER_SEC;

//    strcpy(filename_prefix,"avc_vloc_p");
    strcpy(filename_prefix,"cvc_v_p");
    
//    for(source_location=0;source_location<1728; source_location++){
//      sprintf(filename, "%s.%.4d", filename_prefix, source_location);
      
      sprintf(filename, "%s.%.4d", filename_prefix,gid);
      fprintf(stdout, "# Reading data from file %s\n", filename);
      status = read_lime_contraction(conn, filename, 16, 0);
	if(status == 106) {
	fprintf(stderr, "Error: could not read from file %s; status was %d\n", filename, status);
	continue;
	}
/*
    sprintf(filename, "%s.%.4d.%.4d", filename_prefix2, gid);
    fprintf(stdout, "# Reading data from file %s\n", filename);
    status = read_lime_contraction(conn2, filename, 16, 0);
    if(status == 106) {
      fprintf(stderr, "Error: could not read from file %s; status was %d\n", filename, status);
      continue;
    }
*/
      retime = (double)clock() / CLOCKS_PER_SEC;
      fprintf(stdout, "# time to read contractions %e seconds\n", retime-ratime);

    /***********************
     * fill the correlator *
     ***********************/
      ratime = (double)clock() / CLOCKS_PER_SEC;
/*
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      for(x0=0; x0<T; x0++) {
        iix = g_ipt[0][x1][x2][x3]*T+x0;
        for(mu=1; mu<4; mu++) {
          ix = _GWI(5*mu,g_ipt[x0][x1][x2][x3],VOLUME);
          pijj[iix].re += ( conn[ix  ] - conn2[ix  ] ) * (double)Nsave / (double)(Nsave-1);
          pijj[iix].im += ( conn[ix+1] - conn2[ix+1] ) * (double)Nsave / (double)(Nsave-1);
        }
        ix = 2*g_ipt[x0][x1][x2][x3];
        pi00[iix].re += ( conn[ix  ] - conn2[ix  ] ) * (double)Nsave / (double)(Nsave-1);
        pi00[iix].im += ( conn[ix+1] - conn2[ix+1] ) * (double)Nsave / (double)(Nsave-1);
      }
    }}}
*/
      for(x0=0; x0<T; x0++) {
	ix = g_ipt[x0][0][0][0];
	corrt[x0].re = conn[_GWI(5,ix,VOLUME)  ] + conn[_GWI(10,ix,VOLUME)  ] + conn[_GWI(15,ix,VOLUME)  ];
	corrt[x0].im = conn[_GWI(5,ix,VOLUME)+1] + conn[_GWI(10,ix,VOLUME)+1] + conn[_GWI(15,ix,VOLUME)+1];
	corrt[x0].re /= (double)T;
	corrt[x0].im /= (double)T;
      }
/*    fftw(plan_m, 1, corrt, 1, T, (fftw_complex*)NULL, 0, 0); */
      fftw_one(plan_m, corrt, NULL);
      sprintf(filename, "rho.%.4d", gid);
      if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
	fprintf(stderr, "Error: could not open file %s for writing\n", filename);
	exit(5);
      }
      fprintf(stdout, "# writing VKVK data to file %s\n", filename);
      fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu);
    
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 0, 0, 0, corrt[0].re, 0., gid);
      for(x0=1; x0<(T/2); x0++) {
	fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 0, 0, x0, 
	  corrt[x0].re, corrt[T-x0].re, gid);
      }
      fprintf(ofs, "%3d%3d%3d%25.16e%25.16e%6d\n", 0, 0, (T/2), corrt[T/2].re, 0., gid);
      fclose(ofs);
      retime = (double)clock() / CLOCKS_PER_SEC;
      fprintf(stdout, "# time to fill correlator %e seconds\n", retime-ratime);
//   } of source_location
      
#ifdef _UNDEF
 
    free(conn);
/*    free(conn2); */

    /********************************
     * test: print correl to stdout *
     ********************************/
/*
  fprintf(stdout, "\n\n# *****************   pijj   *****************\n");
  for(ix=0; ix<LX*LY*LZ; ix++) {
    iix = ix*T;
    for(x0=0; x0<T; x0++) {
      fprintf(stdout, "%6d%3d%25.16e%25.16e\n", ix, x0, pijj[iix+x0].re, pijj[iix+x0].im);
    }
  }
  fprintf(stdout, "\n\n# *****************   pi00   *****************\n");
  for(ix=0; ix<LX*LY*LZ; ix++) {
    iix = ix*T;
    for(x0=0; x0<T; x0++) {
      fprintf(stdout, "%6d%3d%25.16e%25.16e\n", ix, x0, pi00[iix+x0].re, pi00[iix+x0].im);
    }
  }
*/

    /*****************************************
     * do the reverse Fourier transformation *
     *****************************************/
    ratime = (double)clock() / CLOCKS_PER_SEC;
    fftw(plan_m, LX*LY*LZ,  pi00, 1, T, (fftw_complex*)NULL, 0, 0);
    fftw(plan_m, LX*LY*LZ,  pijj, 1, T, (fftw_complex*)NULL, 0, 0);

    for(ix=0; ix<VOLUME; ix++) {
      pi00[ix].re /= (double)T; pi00[ix].im /= (double)T;
      pijj[ix].re /= 3.*(double)T; pijj[ix].im /= 3.*(double)T;
    }
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time for Fourier transform %e seconds\n", retime-ratime);

  /*****************************************
   * write to file
   *****************************************/
  ratime = (double)clock() / CLOCKS_PER_SEC;
  sprintf(filename, "pi00.%.4d", gid);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "Error: could not open file %s for writing\n", filename);
    exit(5);
  }
  fprintf(stdout, "# writing pi00-data to file %s\n", filename);
  fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu);
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[0][x1][x2][x3]*T;
/*    fprintf(ofs, "# px=%3d, py=%3d, pz=%3d\n", x1, x2, x3); */
    for(x0=0; x0<T; x0++) {
/*      fprintf(ofs, "%3d%25.16e%25.16e\n", x0, pi00[ix+x0].re, pi00[ix+x0].im); */
      fprintf(ofs, "%3d%3d%3d%3d%25.16e%25.16e\n", x1, x2, x3, x0, pi00[ix+x0].re, pi00[ix+x0].im);
    }
  }}}
  fclose(ofs);

  sprintf(filename, "pijj.%.4d", gid);
  if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
    fprintf(stderr, "Error: could not open file %s for writing\n", filename);
    exit(5);
  }
  fprintf(stdout, "# writing pijj-data to file %s\n", filename);
  fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu);
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[0][x1][x2][x3]*T;
/*    fprintf(ofs, "# px=%3d, py=%3d, pz=%3d\n", x1, x2, x3); */
    for(x0=0; x0<T; x0++) {
/*      fprintf(ofs, "%3d%25.16e%25.16e\n", x0, pijj[ix+x0].re, pijj[ix+x0].im); */
      fprintf(ofs, "%3d%3d%3d%3d%25.16e%25.16e\n", x1, x2, x3, x0, pijj[ix+x0].re, pijj[ix+x0].im);
    }
  }}}
  fclose(ofs);

  retime = (double)clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "# time to write correlator %e seconds\n", retime-ratime);

/*
  if(mode==0) {
    ratime = (double)clock() / CLOCKS_PER_SEC;
    if( (picount = (int*)malloc(VOLUME*sizeof(int))) == (int*)NULL) exit(110);
    sprintf(filename, "corr.00.mom");
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error: could not open file %s for writing\n", filename);
      exit(5);
    }
    for(ix=0; ix<VOLUME; ix++) picount[ix] = 0;
    for(x1=0; x1<LX; x1++) {
      q[1] = 2. * sin(M_PI * (double)x1 / (double)LX);
    for(x2=0; x2<LY; x2++) {
      q[2] = 2. * sin(M_PI * (double)x2 / (double)LY);
    for(x3=0; x3<LZ; x3++) {
      q[3] = 2. * sin(M_PI * (double)x3 / (double)LZ);
      qsqr = q[1]*q[1] + q[2]*q[2] + q[3]*q[3]; 
      if( qsqr>=g_qhatsqr_min-_Q2EPS && qsqr<= g_qhatsqr_max+_Q2EPS ) {
        ix = g_ipt[0][x1][x2][x3];
        picount[ix] = 1;
        fprintf(ofs, "%3d%3d%3d%6d%25.16e\n", x1, x2, x3, ix, qsqr);
      }
    }}}
    fclose(ofs);
    sprintf(filename, "corr_00.00.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error: could not open file %s for writing\n", filename);
      exit(5);
    }
    fprintf(stdout, "# writing corr_00-data to file %s\n", filename);
    fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu);
    for(ix=0; ix<VOLUME; ix++) {
      if(picount[ix]>0) {
        for(x0=0; x0<T; x0++) {
          fprintf(ofs, "%3d%3d%25.16e%25.16e\n", ix, x0, pi00[ix*T+x0].re, pi00[ix*T+x0].im);
        }
      }
    }
    fclose(ofs);
    sprintf(filename, "corr_jj.00.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error: could not open file %s for writing\n", filename);
      exit(5);
    }
    fprintf(stdout, "# writing corr_jj-data to file %s\n", filename);
    fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu);
    for(ix=0; ix<VOLUME; ix++) {
      if(picount[ix]>0) {
        for(x0=0; x0<T; x0++) {
          fprintf(ofs, "%3d%3d%25.16e%25.16e\n", ix, x0, pijj[ix*T+x0].re, pijj[ix*T+x0].im);
        }
      }
    }
    fclose(ofs);
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time for O_h averaging %e seconds\n", retime-ratime);
    free(picount);
  } else if(mode==1) {
    ratime = (double)clock() / CLOCKS_PER_SEC;
    if( (picount = (int*)malloc(VOLUME*sizeof(int))) == (int*)NULL) exit(110);
    sprintf(filename, "corr.01.mom");
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error: could not open file %s for writing\n", filename);
      exit(5);
    }
    if( (picount = (int*)malloc(VOLUME*sizeof(int))) == (int*)NULL) exit(110);
    for(ix=0; ix<VOLUME; ix++) picount[ix] = 0;
    for(x1=0; x1<LX; x1++) {
      q[1] = 2. * M_PI * (double)x1 / (double)LX;
    for(x2=0; x2<LY; x2++) {
      q[2] = 2. * M_PI * (double)x2 / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      q[3] = 2. * M_PI * (double)x3 / (double)LZ;
      qsqr = q[1]*q[1] + q[2]*q[2] + q[3]*q[3]; 
      if( qsqr>=g_qhatsqr_min-_Q2EPS && qsqr<= g_qhatsqr_max+_Q2EPS ) {
        ix = g_ipt[0][x1][x2][x3];
        picount[ix] = 1;
        fprintf(ofs, "%3d%3d%3d%6d%25.16e\n", x1, x2, x3, ix, qsqr);
      }
    }}}
    fclose(ofs);
    sprintf(filename, "corr_00.01.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error: could not open file %s for writing\n", filename);
      exit(5);
    }
    fprintf(stdout, "# writing corr_01-data to file %s\n", filename);
    fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu);
    for(ix=0; ix<VOLUME; ix++) {
      if(picount[ix]>0) {
        for(x0=0; x0<T; x0++) {
          fprintf(ofs, "%3d%3d%25.16e%25.16e\n", ix, x0, pi00[ix*T+x0].re, pi00[ix*T+x0].im);
        }
      }
    }
    fclose(ofs);
    sprintf(filename, "corr_jj.01.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error: could not open file %s for writing\n", filename);
      exit(5);
    }
    fprintf(stdout, "# writing corr_jj-data to file %s\n", filename);
    fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu);
    for(ix=0; ix<VOLUME; ix++) {
      if(picount[ix]>0) {
        for(x0=0; x0<T; x0++) {
          fprintf(ofs, "%3d%3d%25.16e%25.16e\n", ix, x0, pijj[ix*T+x0].re, pijj[ix*T+x0].im);
        }
      }
    }
    fclose(ofs);
    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time for writing: %e seconds\n", retime-ratime);
    free(picount);
  } else if(mode==2) {
    if(make_H3orbits(&oh_id, &oh_count, &oh_val, &oh_nc) != 0) return(123);
    ratime = (double)clock() / CLOCKS_PER_SEC;
    nclass = oh_nc / Thp1;
    if( (piavg = (fftw_complex*)malloc(oh_nc*sizeof(fftw_complex))) == (fftw_complex*)NULL) exit(110);
    if( (picount = (int*)malloc(oh_nc*sizeof(int))) == (int*)NULL) exit(110);

    for(ix=0; ix<oh_nc; ix++) {
      piavg[ix].re = 0.; 
      piavg[ix].im = 0.;
      picount[ix]  = 0;
    }

    for(ix=0; ix<LX*LY*LZ; ix++) {
      for(x0=0; x0<Thp1; x0++) {
        iix = ix*T+x0;
        iiy = oh_id[ix]*Thp1+x0;
        piavg[iiy].re += pi00[iix].re;
        piavg[iiy].im += pi00[iix].im;
        if(x0>0 && x0<T/2) {
          iix = ix*T+(T-x0);
          piavg[iiy].re += pi00[iix].re;
          piavg[iiy].im += pi00[iix].im;
        }
      }
      picount[oh_id[ix]]++;
    }
    for(ix=0; ix<nclass; ix++) {
      for(x0=0; x0<Thp1; x0++) {
        iix = ix*Thp1+x0;
        if(picount[ix]>0) {
          piavg[iix].re /= (double)picount[ix];
          piavg[iix].im /= (double)picount[ix];
          if(x0>0 && x0<T/2) {
            piavg[iix].re /= 2.;
            piavg[iix].im /= 2.;
          }
        }
      }
    }
    sprintf(filename, "corr02_00.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error: could not open file %s for writing\n", filename);
      exit(5);
    }
    fprintf(stdout, "# writing corr-00-data to file %s\n", filename);
    fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu);
    for(x1=0; x1<nclass; x1++) {
      if(oh_val[0][x1]>=g_qhatsqr_min-_Q2EPS && oh_val[0][x1]<=g_qhatsqr_max+_Q2EPS) {
        ix = x1*Thp1;
        for(x0=0; x0<Thp1; x0++) {
          fprintf(ofs, "%25.16e%3d%25.16e%25.16e%5d\n", oh_val[0][x1], x0, piavg[ix+x0].re, piavg[ix+x0].im, 
            picount[x1]);
        }
      }
    }
    fclose(ofs);

    for(ix=0; ix<oh_nc; ix++) {
      piavg[ix].re = 0.; 
      piavg[ix].im = 0.;
      picount[ix]  = 0;
    }

    for(ix=0; ix<LX*LY*LZ; ix++) {
      for(x0=0; x0<Thp1; x0++) {
        iix = ix*T+x0;
        iiy = oh_id[ix]*Thp1+x0;
        piavg[iiy].re += pijj[iix].re;
        piavg[iiy].im += pijj[iix].im;
        if(x0>0 && x0<T/2) {
          iix = ix*T+(T-x0);
          piavg[iiy].re += pijj[iix].re;
          piavg[iiy].im += pijj[iix].im;
        }
      }
      picount[oh_id[ix]]++;
    }
    for(ix=0; ix<nclass; ix++) {
      for(x0=0; x0<Thp1; x0++) {
        iix = ix*Thp1+x0;
        if(picount[ix]>0) {
          piavg[iix].re /= (double)picount[ix];
          piavg[iix].im /= (double)picount[ix];
          if(x0>0 && x0<T/2) {
            piavg[iix].re /= 2.;
            piavg[iix].im /= 2.;
          }
        }
    }}
  
    sprintf(filename, "corr02_jj.%.4d", gid);
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error: could not open file %s for writing\n", filename);
      exit(5);
    }
    fprintf(stdout, "# writing corr-jj-data to file %s\n", filename);
    fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n", gid, T_global, LX, LY, LZ, g_kappa, g_mu);
    for(x1=0; x1<nclass; x1++) {
      ix = x1*Thp1;
      for(x0=0; x0<Thp1; x0++) {
        fprintf(ofs, "%25.16e%3d%25.16e%25.16e%5d\n", oh_val[0][x1], x0, piavg[ix+x0].re, piavg[ix+x0].im, 
          picount[x1]);
      }
    }
    fclose(ofs);
    sprintf(filename, "corr.02.mom");
    if( (ofs=fopen(filename, "w")) == (FILE*)NULL ) {
      fprintf(stderr, "Error: could not open file %s for writing\n", filename);
      exit(5);
    }
    for(ix=0; ix<VOLUME; ix++) fprintf(ofs, "%5d%25.16e%5d", ix, oh_val[0][ix], picount[ix]);
    fclose(ofs);


    retime = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stdout, "# time for O_h averaging %e seconds\n", retime-ratime);

    free(piavg); free(picount);
  }
*/

#endif
  }

  /***************************************
   * free the allocated memory, finalize *
   ***************************************/
  free(corrt);
  free_geometry();
/*
  free(pi00);
  free(pijj);
*/
  fftw_destroy_plan(plan_m);

  return(0);

}
