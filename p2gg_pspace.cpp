/****************************************************
 * p2gg_pspace.cpp
 *
 * Fri Jan 15 17:53:37 CET 2016
 *
 * PURPOSE:
 * - originally copied from avc_exact2_lowmem_pspace.c
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
#ifdef OPENMP
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
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "contractions_io.h"

using namespace cvc;


int main(int argc, char **argv) {

  char outfile_name[] = "p2gg";

  int c, i, j, mu, nu, imunu;
  int filename_set = 0;
  int dims[4]      = {0,0,0,0};
  int l_LX_at, l_LXstart_at;
  int have_source_flag = 0;
  int x0, x1, x2, x3, ix;
  int y0, y1, y2, y3;
  int sx0, sx1, sx2, sx3;
  int gsx0, gsx1, gsx2, gsx3;
  int check_position_space_WI=0, check_momentum_space_WI=0;
  /* int nthreads=-1, threadid=-1; */
  int exitstatus;
  int write_ascii=0;
  double *conn = (double*)NULL;
  double contact_term[8];
  double phase[4], seq_phase[4];
  int verbose = 0;
  char filename[100], contype[1200];
  char outfile_tag[400];
  double ratime, retime;
  double *phi=NULL, *chi=NULL;
  complex w, w1;
  FILE *ofs;

  fftw_complex *in=(fftw_complex*)NULL;

#ifdef HAVE_MPI
  fftwnd_mpi_plan plan_p;
#else
  fftwnd_plan plan_p;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "wWah?vf:")) != -1) {
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
      fprintf(stdout, "# [p2gg_pspace] will check Ward identity in position space\n");
      break;
    case 'W':
      check_momentum_space_WI = 1;
      fprintf(stdout, "# [p2gg_pspace] will check Ward identity in momentum space\n");
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "# [p2gg_pspace] will write data in ASCII format too\n");
      break;
    case 'h':
    case '?':
    default:
      fprintf(stdout, "# [p2gg_pspace] unrecognized option\n");
      exit(0);
      break;
    }
  }

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_pspace] using global time stamp %s", ctime(&g_the_time));
  }

#if (defined PARALLELTX) || (defined PARALLELTXY) || ( defined PARALLELTXYZ)
  fprintf(stderr, "[p2gg_pspace] Error, no implementation for this domain decomposition pattern\n");
  exit(123);
#endif


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#endif

  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  fprintf(stdout, "# [p2gg_pspace] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[p2gg_pspace] T and L's must be set\n");
    EXIT(2);
  }
  if(g_kappa == 0.) {
    if(g_proc_id==0) fprintf(stderr, "[p2gg_pspace] kappa should be > 0.n");
    EXIT(3);
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(32);


  /*******************************
   * initialize fftw
   *******************************/
#ifdef OPENMP
  exitstatus = fftw_threads_init();
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_pspace] Error from fftw_init_threads; status was %d\n", exitstatus);
    exit(120);
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
    fprintf(stderr, "[p2gg_pspace] [%2d] local T is zero; exit\n", g_cart_id);
    EXIT(2);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg_pspace] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  /* allocate memory for the contractions */
  conn = (double*)calloc(32 * (VOLUME+RAND), sizeof(double));
  if( conn==(double*)NULL ) {
    fprintf(stderr, "[p2gg_pspace] could not allocate memory for contraction fields\n");
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

  /* global source coordinates */
  gsx0 = g_source_location / (LX_global * LY_global * LZ_global);
  gsx1 = (g_source_location % (LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  gsx2 = (g_source_location % (LY_global * LZ_global)) / LZ_global;
  gsx3 = (g_source_location %  LZ_global);
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] global source coordinates = (%d, %d, %d, %d)\n", gsx0, gsx1, gsx2, gsx3);

  /* check for source location */
  have_source_flag = (int)(gsx0 >=Tstart && gsx0 < (Tstart+T) );
  if(have_source_flag==1) fprintf(stdout, "# [p2gg_pspace] process %2d has source location\n", g_cart_id);

  /* local source coordinates */
  sx0 = gsx0 - Tstart;
  sx1 = gsx1;
  sx2 = gsx2;
  sx3 = gsx3;
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] local source coordinates = (%d, %d, %d, %d)\n", sx0, sx1, sx2, sx3);

  /* read the position space contractions */
  ratime = _GET_TIME;



  /* set the outfile tag */
  sprintf( outfile_tag, "t%.2dx%.2dy%.2dz%.2d.tseq%.2d.g%.2d.px%.2dpy%.2dpz%.2d",
      gsx0, gsx1, gsx2, gsx3, g_sequential_source_timeslice,
      g_sequential_source_gamma_id, g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);
  if(g_cart_id == 0) {
    fprintf(stdout, "# [p2gg_pspace] file flag set to %s\n", outfile_tag);
  }

  if(strcmp(g_outfile_prefix, "NA") == 0) {
    sprintf(filename, "%s_x.%s.%.4d", outfile_name, outfile_tag, Nconf);
  } else {
    sprintf(filename, "%s/%s_x.%s.%.4d", g_outfile_prefix, outfile_name, outfile_tag, Nconf);
  }
  exitstatus = read_lime_contraction(conn, filename, 16, 0);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_pspace] Error from read_lime_contraction for file %s, status was %d\n", filename, exitstatus);
    EXIT(102);
  }
  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_pspace] time to read contraction data: %e seconds\n", retime-ratime);


  /* read the contact terms */

  if(strcmp(g_outfile_prefix, "NA") == 0) {
    sprintf(filename, "%s_ct.%s.%.4d", outfile_name, outfile_tag, Nconf);
  } else {
    sprintf(filename, "%s/%s_ct.%s.%.4d", g_outfile_prefix, outfile_name, outfile_tag, Nconf);
  }

  if( (ofs = fopen(filename, "r")) == NULL ) {
    fprintf(stderr, "[p2gg_pspace] Error, could not open file %s for reading\n", filename);
    EXIT(117);
  }

  for(mu=0;mu<4;mu++) {
    fscanf(ofs, "%lf%lf", contact_term+2*mu, contact_term+2*mu+1);
  }
  fclose(ofs);
  // print contact term
  if(have_source_flag) {
    fprintf(stdout, "# [p2gg_pspace] contact term\n");
    for(i=0;i<4;i++) {
      fprintf(stdout, "\t%d%25.16e%25.16e\n", i, contact_term[2*i], contact_term[2*i+1]);
    }
  }


  /* check the Ward identity in position space */

  if(check_position_space_WI) {
#ifdef HAVE_MPI
    xchange_contraction(conn, 32);
#endif
    sprintf(filename, "p2gg_pspace_WI_X.%s.%.4d.%.4d", outfile_tag, Nconf, g_cart_id);
    ofs = fopen(filename,"w");
    if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] checking Ward identity in position space\n");
    for(x0=0; x0<T;  x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      fprintf(ofs, "# t=%2d x=%2d y=%2d z=%2d\n", x0+g_proc_coords[0] * T, x1+g_proc_coords[1]*LX, x2+g_proc_coords[2]*LY, x3+g_proc_coords[3]*LZ);
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

        fprintf(ofs, "\t%3d%25.16e%25.16e\n", nu, w.re, w.im);
      }
    }}}}
    fclose(ofs);
  }


  /*********************************************
   * Fourier transformation 
   *********************************************/
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] Fourier transform\n");
  ratime = _GET_TIME;
  for(mu=0; mu<16; mu++) {
    /* memcpy((void*)in, (void*)&conn[_GWI(mu,0,VOLUME)], 2*VOLUME*sizeof(double)); */

    for(ix=0; ix<VOLUME; ix++) {
      in[ix].re = conn[_GWI(mu, ix, VOLUME)  ];
      in[ix].im = conn[_GWI(mu, ix, VOLUME)+1];
    }

#ifdef HAVE_MPI
    fftwnd_mpi(plan_p, 1, in, NULL, FFTW_NORMAL_ORDER);
#else
#  ifdef OPENMP
    fftwnd_threads_one(num_threads, plan_p, in, NULL);
#  else
    fftwnd_one(plan_p, in, NULL);
#  endif
#endif
    /* memcpy((void*)&conn[_GWI(mu,0,VOLUME)], (void*)in, 2*VOLUME*sizeof(double)); */
    for(ix=0; ix<VOLUME; ix++) {
        conn[_GWI(mu, ix, VOLUME)  ] = in[ix].re;
        conn[_GWI(mu, ix, VOLUME)+1] = in[ix].im;
    }
  }


  /*****************************************
   * set the sequential phase exp(-i q_nu/2)
   *   for nu = 1,2,3
   *****************************************/
  seq_phase[0] = 0.;
  seq_phase[1] = M_PI * (double)g_seq_source_momentum[0] / (double)LX_global;
  seq_phase[2] = M_PI * (double)g_seq_source_momentum[1] / (double)LY_global;
  seq_phase[3] = M_PI * (double)g_seq_source_momentum[2] / (double)LZ_global;


  /*****************************************
   * add phase factor and contact term to
   *   diagonal tensor elements nu = mu
   *****************************************/
  for(mu=0; mu<4; mu++) {

    for(x0=0; x0<T; x0++) {
      phase[0] = 2. * (double)(x0 + g_proc_coords[0]*T)  * M_PI / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      phase[1] = 2. * (double)(x1 + g_proc_coords[1]*LX) * M_PI / (double)LX_global;
    for(x2=0; x2<LY; x2++) {
      phase[2] = 2. * (double)(x2 + g_proc_coords[2]*LY) * M_PI / (double)LY_global;
    for(x3=0; x3<LZ; x3++) {
      phase[3] = 2. * (double)(x3 + g_proc_coords[3]*LZ) * M_PI / (double)LZ_global;
      ix = g_ipt[x0][x1][x2][x3];

      phi = conn + _GWI(5*mu, ix, VOLUME);

      w.re =  cos( phase[0] * gsx0 + phase[1] * gsx1 + phase[2] * gsx2 + phase[3] * gsx3 );
      w.im = -sin( phase[0] * gsx0 + phase[1] * gsx1 + phase[2] * gsx2 + phase[3] * gsx3 );

      _co_eq_co_ti_co(&w1,(complex*)phi, &w);

      w1.re -= contact_term[2*mu  ];
      w1.im -= contact_term[2*mu+1];

      w.re =  cos( seq_phase[mu] );
      w.im = -sin( seq_phase[mu] );

      _co_eq_co_ti_co((complex*)phi , &w1, &w);

    }}}}
  }  /* of mu */

  /*****************************************
   * add phase factors to off-diagonal
   *   tensor elements nu != mu
   *****************************************/
  for(mu=0; mu<3; mu++) {
  for(nu=mu+1; nu<4; nu++) {

    for(x0=0; x0<T; x0++) {
      phase[0] =  (double)(x0 + g_proc_coords[0]*T ) * M_PI / (double)T_global;
    for(x1=0; x1<LX; x1++) {
      phase[1] =  (double)(x1 + g_proc_coords[1]*LX) * M_PI / (double)LX_global;
    for(x2=0; x2<LY; x2++) {
      phase[2] =  (double)(x2 + g_proc_coords[2]*LY) * M_PI / (double)LY_global;
    for(x3=0; x3<LZ; x3++) {
      phase[3] =  (double)(x3 + g_proc_coords[3]*LZ) * M_PI / (double)LZ_global;
      ix = g_ipt[x0][x1][x2][x3];

      /* Tensor_mu_nu */
      phi = conn + _GWI(4*mu+nu, ix, VOLUME);
      /* Tensor_nu_mu */
      chi = conn + _GWI(4*nu+mu, ix, VOLUME);

      w.re =  cos( -seq_phase[nu] + phase[mu] - phase[nu] - 2.*(phase[0] * gsx0 + phase[1] * gsx1 + phase[2] * gsx2+phase[3] * gsx3) );
      w.im =  sin( -seq_phase[nu] + phase[mu] - phase[nu] - 2.*(phase[0] * gsx0 + phase[1] * gsx1 + phase[2] * gsx2+phase[3] * gsx3) );
      _co_eq_co_ti_co(&w1,(complex*)phi, &w);
      phi[0] = w1.re;
      phi[1] = w1.im;

      w.re =  cos( -seq_phase[mu] + phase[nu] - phase[mu] - 2.*(phase[0] * gsx0 + phase[1] * gsx1 + phase[2] * gsx2 + phase[3] * gsx3) );
      w.im =  sin( -seq_phase[mu] + phase[nu] - phase[mu] - 2.*(phase[0] * gsx0 + phase[1] * gsx1 + phase[2] * gsx2 + phase[3] * gsx3) );
      _co_eq_co_ti_co(&w1,(complex*)chi, &w);
      chi[0] = w1.re;
      chi[1] = w1.im;
    }}}}
  }}  /* of mu and nu */

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_pspace] Fourier transform in %e seconds\n", retime-ratime);

  /********************************
   * save momentum space results
   ********************************/
  ratime = _GET_TIME;
  
  if(strcmp(g_outfile_prefix, "NA") == 0) {
    sprintf(filename, "%s_p.%s.%.4d", outfile_name, outfile_tag, Nconf);
  } else {
    sprintf(filename, "%s/%s_p.%s.%.4d", g_outfile_prefix, outfile_name, outfile_tag, Nconf);
  }

  sprintf(contype, "\n<description>P - cvc - cvc in momentum space, 4x4 components</description>\n"\
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
      gsx0, gsx1, gsx2, gsx3,
      g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
      g_sequential_source_gamma_id, g_sequential_source_timeslice,
      contact_term[0], contact_term[1],
      contact_term[2], contact_term[3],
      contact_term[4], contact_term[5],
      contact_term[6], contact_term[7]);



  /* strcpy(contype, "\nNA\n"); */
  exitstatus = write_lime_contraction(conn, filename, 64, 16, contype, Nconf, 0);
  if(exitstatus != 0) {
    fprintf(stderr, "[] Error from write_lime_contraction for file %s, status was %d\n", filename, exitstatus);
    EXIT(118);
  }


  if(write_ascii) {
#ifndef HAVE_MPI
    if(strcmp(g_outfile_prefix, "NA") == 0) {
      sprintf(filename, "%s_p.%s.%.4d.ascii", outfile_name, outfile_tag, Nconf);
    } else {
      sprintf(filename, "%s/%s_p.%s.%.4d.ascii", g_outfile_prefix, outfile_name, outfile_tag, Nconf);
    }
    write_contraction(conn, NULL, filename, 16, 2, 0);
#else
    sprintf(filename, "%s_x.%s.%.4d.ascii.%.2d", outfile_name, outfile_tag, Nconf, g_cart_id);
    ofs = fopen(filename, "w");
    for(x0=0; x0<T;  x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      fprintf(ofs, "# t=%2d x=%2d y=%2d z=%2d\n", x0+g_proc_coords[0]*T, x1+g_proc_coords[1]*LX, x2+g_proc_coords[2]*LY, x3+g_proc_coords[3]*LZ);
      ix=g_ipt[x0][x1][x2][x3];
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        imunu = 4*mu + nu;
        fprintf(ofs, "%3d%25.16e%25.16e\n", imunu, conn[_GWI(imunu,ix,VOLUME)], conn[_GWI(imunu,ix,VOLUME)+1]);
      }}
    }}}}
    fclose(ofs);
#endif
  }  /* end of if write_ascii */

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_pspace] saved momentum space results in %e seconds\n", retime-ratime);

  if(check_momentum_space_WI) {
    sprintf(filename, "p2gg_pspace_WI_P.%s.%.4d.%.4d", outfile_tag, Nconf, g_cart_id);

    ofs = fopen(filename,"w");
    if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] checking Ward identity in momentum space\n");
    for(x0=0; x0<T; x0++) {
      y0 = x0 + g_proc_coords[0]*T;
      phase[0] = 2. * sin( (double)y0 * M_PI / (double)T_global );

    for(x1=0; x1<LX; x1++) {
      y1 = x1 + g_proc_coords[1] * LX;
      phase[1] = 2. * sin( (double)y1 * M_PI / (double)LX_global );

    for(x2=0; x2<LY; x2++) {
      y2 = x2 + g_proc_coords[2] * LY;
      phase[2] = 2. * sin( (double)y2 * M_PI / (double)LY_global );

    for(x3=0; x3<LZ; x3++) {
      y3 = x3 + g_proc_coords[3] * LZ;
      phase[3] = 2. * sin( (double)y3 * M_PI / (double)LZ_global );

      ix = g_ipt[x0][x1][x2][x3];
      fprintf(ofs, "# pt=%2d px=%2d py=%2d pz=%2d\n", x0, x1, x2, x3);
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
  if(conn != NULL) free(conn);
#ifdef HAVE_MPI
  fftwnd_mpi_destroy_plan(plan_p);
  mpi_fini_xchange_contraction();
  MPI_Finalize();
#else
  fftwnd_destroy_plan(plan_p);
#endif


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_pspace] %s# [p2gg_pspace] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_pspace] %s# [p2gg_pspace] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
