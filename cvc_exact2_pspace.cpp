/****************************************************
  
 * cvc_exact2_pspace.cpp
 *
 * Thu Dec 22 10:34:56 CET 2016
 *
 * PURPOSE:
 * - originally copied from cvc_parallel_5d/avc_exact2_lowmem_pspace.c
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
#include "contractions_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform AV current correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:\n");
  fprintf(stdout, "         -g apply a random gauge transformation\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu, nu;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int have_source_flag = 0;
  int x0, x1, x2, x3, ix;
  int sx0, sx1, sx2, sx3;
  int check_position_space_WI=0, check_momentum_space_WI=0;
  int write_ascii=0;
  int outfile_prefix_set = 0;
  double *conn = (double*)NULL;
  double contact_term[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double phase[4];
  char filename[100], contype[400], outfile_prefix[400];
  double ratime, retime;
  complex w, w1;
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef HAVE_MPI
  fftwnd_mpi_plan plan_p;
  int *status;
#else
  fftwnd_plan plan_p;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "wWah?f:o:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_position_space_WI = 1;
      fprintf(stdout, "\n# [cvc_exact2_pspace] will check Ward identity in position space\n");
      break;
    case 'W':
      check_momentum_space_WI = 1;
      fprintf(stdout, "\n# [cvc_exact2_pspace] will check Ward identity in momentum space\n");
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "\n# [cvc_exact2_pspace] will write data in ASCII format too\n");
      break;
    case 'o':
      strcpy(outfile_prefix, optarg);
      fprintf(stdout, "\n# [cvc_exact2_pspace] will use prefix %s for output filenames\n", outfile_prefix);
      outfile_prefix_set = 1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [cvc_exact2_pspace] using global time stamp %s", ctime(&g_the_time));
  }

#if (defined PARALLELTX) || (defined PARALLELTXY)
  fprintf(stderr, "\nError, no implementation for this domain decomposition pattern\n");
  exit(123);
#endif

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "\n[cvc_exact2_pspace] T and L's must be set\n");
    usage();
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stderr, "\n[cvc_exact2_pspace] kappa should be > 0.n");
    usage();
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);
#ifdef HAVE_MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    EXIT(7);
  }
#endif

  /*******************************
   * initialize fftw
   *******************************/
#ifdef HAVE_OPENMP
  int exitstatus = fftw_threads_init();
  if(exitstatus != 0) {
    fprintf(stderr, "\n[cvc_exact2_pspace] Error from fftw_init_threads; status was %d\n", exitstatus);
    EXIT(120);
  }
#endif


  dims[0]=T_global; dims[1]=LX; dims[2]=LY; dims[3]=LZ;
#ifdef HAVE_MPI
  plan_p = fftwnd_mpi_create_plan(g_cart_grid, 4, dims, FFTW_BACKWARD, FFTW_MEASURE);
  fftwnd_mpi_local_sizes(plan_p, &T, &Tstart, &l_LX_at, &l_LXstart_at, &FFTW_LOC_VOLUME);
#else
  plan_p = fftwnd_create_plan(4, dims, FFTW_BACKWARD, FFTW_MEASURE | FFTW_IN_PLACE);
  T            = T_global;
  Tstart       = 0;
  l_LX_at      = LX;
  l_LXstart_at = 0;
  FFTW_LOC_VOLUME = T*LX*LY*LZ;
#endif
  fprintf(stdout, "# [%2d] fftw parameters:\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
		  "# [%2d] l_LX_at      = %3d\n"\
		  "# [%2d] l_LXstart_at = %3d\n"\
		  "# [%2d] FFTW_LOC_VOLUME = %3d\n", 
		  g_cart_id, g_cart_id, T, g_cart_id, Tstart, g_cart_id, l_LX_at,
		  g_cart_id, l_LXstart_at, g_cart_id, FFTW_LOC_VOLUME);

#ifdef HAVE_MPI
  if(T==0) {
    EXIT(2);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /* allocate memory for the contractions */
  conn = (double*)calloc(2 * 16 * VOLUME, sizeof(double));
  if( conn==(double*)NULL ) {
    fprintf(stderr, "could not allocate memory for contr. fields\n");
    EXIT(3);
  }

  /***********************************************************
   * prepare Fourier transformation arrays
   ***********************************************************/
  in  = (fftw_complex*)malloc(FFTW_LOC_VOLUME*sizeof(fftw_complex));
  if(in==(fftw_complex*)NULL) {    
    EXIT(4);
  }


  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "process %2d has source location\n", g_cart_id);
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);

  /* read the position space contractions */
  ratime = _GET_TIME;
  if(outfile_prefix_set) {
    sprintf(filename, "%s/cvc2_v_x.%.4d", outfile_prefix, Nconf);
  } else {
    sprintf(filename, "cvc2_v_x.%.4d", Nconf);
  }

  for(mu=0; mu<16; mu++) {
    int exitstatus = read_lime_contraction( &(conn[_GWI(mu,0,VOLUME)]), filename, 1, mu);

    if(exitstatus != 0 ) {
      fprintf(stderr, "[cvc_exact2_pspace] Error from read_lime_contractions, status was %d\n", exitstatus);
      EXIT(12);
    }
  }
  /* read_lime_contraction(conn, filename, 16, 0); */

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "\n# [cvc_exact2_pspace] time to read contraction data: %e seconds\n", retime-ratime);

  /* TEST */

  for(x0=0; x0<T;  x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix=g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      fprintf(stdout, "x %2d %2d %2d %2d \t %2d %2d \t %25.16e%25.16e\n", x0, x1, x2, x3, mu, nu, 
          conn[_GWI(4*mu+nu,ix,VOLUME)],
          conn[_GWI(4*mu+nu,ix,VOLUME)+1]);
    }}
  }}}}




  // read the contact terms
  sprintf(filename, "cvc2_v_ct.%.4d", Nconf);
  if( (ofs = fopen(filename, "r")) == NULL ) {
    fprintf(stderr, "\n[cvc_exact2_pspace] Error, could not open file %s for reading\n", filename);
    EXIT(117);
  }
  for(mu=0;mu<4;mu++) {
    fscanf(ofs, "%lf%lf", contact_term+2*mu, contact_term+2*mu+1);
  }
  fclose(ofs);
  // print contact term
  if(have_source_flag) {
    fprintf(stdout, "\n# [cvc_exact2_pspace] contact term\n");
    for(i=0;i<4;i++) {
      fprintf(stdout, "\t%d%25.16e%25.16e\n", i, contact_term[2*i], contact_term[2*i+1]);
    }
  }


#ifndef HAVE_MPI
  /* check the Ward identity in position space */
  if(check_position_space_WI) {
    sprintf(filename, "WI_X.%.4d", Nconf);
    ofs = fopen(filename,"w");
    fprintf(stdout, "\n# [cvc_exact2_pspace] checking Ward identity in position space ...\n");
    for(x0=0; x0<T;  x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      fprintf(ofs, "# t=%2d x=%2d y=%2d z=%2d\n", x0, x1, x2, x3);
      ix=g_ipt[x0][x1][x2][x3];
      for(nu=0; nu<4; nu++) {
        w.re = conn[_GWI(4*0+nu,ix,VOLUME)] + conn[_GWI(4*1+nu,ix,VOLUME)]
             + conn[_GWI(4*2+nu,ix,VOLUME)] + conn[_GWI(4*3+nu,ix,VOLUME)]
	     - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)]
	     - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)];

        w.im = conn[_GWI(4*0+nu,ix,VOLUME)+1] + conn[_GWI(4*1+nu,ix,VOLUME)+1]
            + conn[_GWI(4*2+nu,ix,VOLUME)+1] + conn[_GWI(4*3+nu,ix,VOLUME)+1]
	    - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)+1] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)+1]
	    - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)+1] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)+1];
      
        fprintf(ofs, "\t%3d%25.16e%25.16e\n", nu, w.re, w.im);
      }
    }}}}
    fclose(ofs);
  }
#endif

  /*********************************************
   * Fourier transformation 
   *********************************************/
  ratime = _GET_TIME;
  for(mu=0; mu<16; mu++) {
    memcpy((void*)in, (void*)&conn[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double));
#ifdef HAVE_MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
#  ifdef HAVE_OPENMP
    fftwnd_threads_one(num_threads, plan_p, in, NULL);
#  else
    fftwnd_one(plan_p, in, NULL);
#  endif
#endif
    memcpy((void*)&conn[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double));
  }

  for(x0=0; x0<T;  x0++) {
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix=g_ipt[x0][x1][x2][x3];
    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      fprintf(stdout, "p %2d %2d %2d %2d \t %2d %2d \t %25.16e%25.16e\n", x0, x1, x2, x3, mu, nu,
          conn[_GWI(4*mu+nu,ix,VOLUME)], conn[_GWI(4*mu+nu,ix,VOLUME)+1]);
    }}
  }}}}



#if 0
  /*****************************************
   * add phase factors
   *****************************************/
  for(mu=0; mu<4; mu++) {
    double *phi = conn + _GWI(5*mu,0,VOLUME);

    for(x0=0; x0<T; x0++) {
      phase[0] = 2. * (double)(Tstart+x0) * M_PI / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      phase[1] = 2. * (double)(x1) * M_PI / (double)LX;
    for(x2=0; x2<LY; x2++) {
      phase[2] = 2. * (double)(x2) * M_PI / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      phase[3] = 2. * (double)(x3) * M_PI / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
/*
      w.re =  cos( phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3 );
      w.im = -sin( phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3 );
*/
      w.re = 1.;
      w.im = 0.;

      _co_eq_co_ti_co(&w1,(complex*)( phi+2*ix ), &w);
      phi[2*ix  ] = w1.re - contact_term[2*mu  ];
      phi[2*ix+1] = w1.im - contact_term[2*mu+1];
    }}}}
  }  /* of mu */
#endif  /* of if 0 */

#if 0
  for(mu=0; mu<3; mu++) {
  for(nu=mu+1; nu<4; nu++) {
    double *phi = conn + _GWI(4*mu+nu,0,VOLUME);
    double *chi = conn + _GWI(4*nu+mu,0,VOLUME);

    for(x0=0; x0<T; x0++) {
      phase[0] =  (double)(Tstart+x0) * M_PI / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      phase[1] =  (double)(x1) * M_PI / (double)LX;
    for(x2=0; x2<LY; x2++) {
      phase[2] =  (double)(x2) * M_PI / (double)LY;
    for(x3=0; x3<LZ; x3++) {
      phase[3] =  (double)(x3) * M_PI / (double)LZ;
      ix = g_ipt[x0][x1][x2][x3];
      w.re =  cos( phase[mu] - phase[nu] - 2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      w.im =  sin( phase[mu] - phase[nu] - 2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      _co_eq_co_ti_co(&w1,(complex*)( phi+2*ix ), &w);
      phi[2*ix  ] = w1.re;
      phi[2*ix+1] = w1.im;

      w.re =  cos( phase[nu] - phase[mu] - 2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      w.im =  sin( phase[nu] - phase[mu] - 2.*(phase[0]*(sx0+Tstart)+phase[1]*sx1+phase[2]*sx2+phase[3]*sx3) );
      _co_eq_co_ti_co(&w1,(complex*)( chi+2*ix ), &w);
      chi[2*ix  ] = w1.re;
      chi[2*ix+1] = w1.im;
    }}}}
  }}  /* of mu and nu */

#endif
  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "Fourier transform in %e seconds\n", retime-ratime);

  /********************************
   * save momentum space results
   ********************************/
  ratime = _GET_TIME;
  if(outfile_prefix_set) {
    sprintf(filename, "%s/cvc2_v_p.%.4d", outfile_prefix, Nconf);
  } else {
    sprintf(filename, "cvc2_v_p.%.4d", Nconf);
  }
  /* sprintf(contype, "cvc - cvc in momentum space, all 16 components");
  write_lime_contraction(conn, filename, 64, 16, contype, Nconf, 0); */
  for(mu=0; mu<16; mu++) {
    sprintf(contype, "<comment>\n  cvc - cvc in momentum space\n</comment>\n<component>\n  %2d-%2d\n</component>\n", mu/4, mu%4);
    write_lime_contraction(&(conn[_GWI(mu,0,VOLUME)]), filename, 64, 1, contype, Nconf, mu>0);
  }

  if(write_ascii) {
    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      int imunu = 4*mu + nu;
#ifndef HAVE_MPI
      if(outfile_prefix_set) {
        sprintf(filename, "%s/cvc2_v_p.%.4d.mu%.2dnu%.2d.ascii", outfile_prefix, Nconf, mu, nu);
      } else {
        sprintf(filename, "cvc2_v_p.%.4d.mu%.2dnu%.2d.ascii", Nconf, mu, nu);
      }
      /* write_contraction(conn, (int*)NULL, filename, 16, 2, 0); */
#else
      sprintf(filename, "cvc2_v_p.%.4d.mu%.2dnu%.2d.ascii.%.2d", Nconf, mu, nu, g_cart_id);
#endif
      ofs = fopen(filename, "w");
      if( ofs == NULL ) {
        fprintf(stderr, "[cvc_exact2_pspace] Error from fopen\n");
        EXIT(116);
      }
      if( g_cart_id == 0 ) fprintf(ofs, "cvc2_v_p <- array(dim=c(%d, %d, %d, %d))\n", T_global,LX_global,LY_global,LZ_global);
      for(x0=0; x0<T;  x0++) {
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        ix=g_ipt[x0][x1][x2][x3];
          fprintf(ofs, "cvc2_v_p[%d, %d, %d, %d] <- %25.16e + %25.16e*1.i\n",
              x0+g_proc_coords[0]*T+1, x1+g_proc_coords[1]*LX+1,
              x2+g_proc_coords[2]*LY+1, x3+g_proc_coords[3]*LZ+1,
              conn[_GWI(imunu,ix,VOLUME)], conn[_GWI(imunu,ix,VOLUME)+1]);
      }}}}
      fclose(ofs);
    }}
  }  /* end of if write ascii */

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact2_pspace] saved momentum space results in %e seconds\n", retime-ratime);

  if(check_momentum_space_WI) {
#ifdef HAVE_MPI
    sprintf(filename, "WI_P.%.4d.%.2d", Nconf, g_cart_id);
#else
    sprintf(filename, "WI_P.%.4d", Nconf);
#endif  
    ofs = fopen(filename,"w");
    if(g_cart_id == 0) fprintf(stdout, "\n# [cvc_exact2_pspace] checking Ward identity in momentum space ...\n");
    for(x0=0; x0<T; x0++) {
      phase[0] = 2. * sin( (double)(Tstart+x0) * M_PI / (double)T_global );
    for(x1=0; x1<LX; x1++) {
      phase[1] = 2. * sin( (double)(x1) * M_PI / (double)LX );
    for(x2=0; x2<LY; x2++) {
      phase[2] = 2. * sin( (double)(x2) * M_PI / (double)LY );
    for(x3=0; x3<LZ; x3++) {
      phase[3] = 2. * sin( (double)(x3) * M_PI / (double)LZ );
      ix = g_ipt[x0][x1][x2][x3];
      fprintf(ofs, "# t=%2d x=%2d y=%2d z=%2d\n", x0, x1, x2, x3);
      for(nu=0;nu<4;nu++) {
        w.re = phase[0] * conn[_GWI(4*0+nu,ix,VOLUME)] + phase[1] * conn[_GWI(4*1+nu,ix,VOLUME)] 
             + phase[2] * conn[_GWI(4*2+nu,ix,VOLUME)] + phase[3] * conn[_GWI(4*3+nu,ix,VOLUME)];

        w.im = phase[0] * conn[_GWI(4*0+nu,ix,VOLUME)+1] + phase[1] * conn[_GWI(4*1+nu,ix,VOLUME)+1] 
             + phase[2] * conn[_GWI(4*2+nu,ix,VOLUME)+1] + phase[3] * conn[_GWI(4*3+nu,ix,VOLUME)+1];

        w1.re = phase[0] * conn[_GWI(4*nu+0,ix,VOLUME)] + phase[1] * conn[_GWI(4*nu+1,ix,VOLUME)] 
              + phase[2] * conn[_GWI(4*nu+2,ix,VOLUME)] + phase[3] * conn[_GWI(4*nu+3,ix,VOLUME)];

        w1.im = phase[0] * conn[_GWI(4*nu+0,ix,VOLUME)+1] + phase[1] * conn[_GWI(4*nu+1,ix,VOLUME)+1] 
              + phase[2] * conn[_GWI(4*nu+2,ix,VOLUME)+1] + phase[3] * conn[_GWI(4*nu+3,ix,VOLUME)+1];
        fprintf(ofs, "\t%d%25.16e%25.16e%25.16e%25.16e\n", nu, w.re, w.im, w1.re, w1.im);
      }
    }}}}
    fclose(ofs);
  }

  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  free_geometry();
  fftw_free(in);
  free(conn);
#ifdef HAVE_MPI
  fftwnd_mpi_destroy_plan(plan_p);
  free(status);
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "\n# [cvc_exact2_pspace] %s# [cvc_exact2_pspace] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "\n# [cvc_exact2_pspace] %s# [cvc_exact2_pspace] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
