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
#include "set_default.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "contractions_io.h"

using namespace cvc;


int main(int argc, char **argv) {

  /*
   * sign for g5 Gamma^\dagger g5
   *                                                0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   */
  const int sequential_source_gamma_id_sign[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };
  const double TWO_MPI = 2. * M_PI;


  int c, i, k,  mu, nu, imunu;
  int io_proc = 2;
  int i_sink_momentum;
  int filename_set = 0;
  int combine_up_dn = 0;
  int read_contact_term = 0;
  int have_source_flag = 0;
  int do_byte_swap = 0;
  int x0, x1, x2, x3, ix;
  int sx0, sx1, sx2, sx3;
  int gsx0, gsx1, gsx2, gsx3;
  int source_proc_coords[4], source_proc_id = -1;
  int check_position_space_WI=0;
  unsigned int VOL3;
  /* int nthreads=-1, threadid=-1; */
  int exitstatus;
  double *conn = NULL, ***conn_ft=NULL;
  double *conn_buffer = NULL, *phase_field=NULL;
  double contact_term[8];
  double phase, phase_shift, psnk[4], psrc[4];
  char filename[100];
  char outfile_tag[400];
  char outfile_name[100];
  double ratime, retime;
  complex w, w1;
  complex *conn_ptr_p=NULL, *conn_ptr_x=NULL;
  FILE *ofs = NULL;

  // in order to be able to initialise QMP if QPhiX is used, we need
  // to allow tmLQCD to intialise MPI via QMP
  // the input file has to be read 
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_init_parallel_and_read_input(argc, argv, 1, "invert.input");
#else
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
#endif

#ifdef HAVE_MPI
  double *mbuffer=NULL;
#endif

  while ((c = getopt(argc, argv, "bcwh?f:t:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_position_space_WI = 1;
      fprintf(stdout, "# [p2gg_pspace] will check Ward identity in position space\n");
      break;
    case 'c':
      combine_up_dn = 1;
      fprintf(stdout, "# [p2gg_pspace] will combine up and dn part\n");
      break;
    case 't':
      if (strcmp(optarg, "read") == 0)  {
        read_contact_term = 1;
        fprintf(stdout, "# [p2gg_pspace] will read contact term from file\n");
      } else if(strcmp(optarg, "wi") == 0)  {
        read_contact_term = 2;
        fprintf(stdout, "# [p2gg_pspace] will take contact term from Ward identity\n");
      } else {
        read_contact_term = 0;
        fprintf(stdout, "# [p2gg_pspace] will ignore contact term\n");
      }
      break;
    case 'b':
      do_byte_swap = 1;
      fprintf(stdout, "# [p2gg_pspace] will do byte swap on input data\n");
      break;
    case 'n':
      strcpy(outfile_name, optarg);
      fprintf(stdout, "# [p2gg_pspace] set outfile name to %s\n", outfile_name);
      break;
    case 'h':
    case '?':
    default:
      fprintf(stdout, "# [p2gg_pspace] unrecognized option\n");
      exit(0);
      break;
    }
  }  /* end of which */

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_pspace] using global time stamp %s", ctime(&g_the_time));
  }

#if (defined PARALLELTX) || (defined PARALLELTXY) || ( defined PARALLELTXYZ)
  fprintf(stderr, "[p2gg_pspace] Error, no implementation for this domain decomposition pattern\n");
  EXIT(123);
#endif


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
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


  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg_pspace] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  VOL3 = LX*LY*LZ;

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [test_gsp_2pt] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [test_gsp_2pt] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
#endif

  /***********************************************************
   * allocate memory for the contractions
   ***********************************************************/
  conn = (double*)calloc(32 * (VOLUME+RAND), sizeof(double));
  if( conn==(double*)NULL ) {
    fprintf(stderr, "[p2gg_pspace] could not allocate memory for contraction fields\n");
    EXIT(3);
  }

  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/

  /* global source coordinates */
  gsx0 = g_source_location / ( LX_global * LY_global * LZ_global);
  gsx1 = (g_source_location % ( LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  gsx2 = (g_source_location % ( LY_global * LZ_global)) / LZ_global;
  gsx3 = (g_source_location % LZ_global);
  /* g_cart_grid coordinates of process, which has the source location */
  source_proc_coords[0] = gsx0 / T;
  source_proc_coords[1] = gsx1 / LX;
  source_proc_coords[2] = gsx2 / LY;
  source_proc_coords[3] = gsx3 / LZ;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [p2gg_pspace] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx0, gsx1, gsx2, gsx3);
    fprintf(stdout, "# [p2gg_pspace] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  /* determine source locations process's rank in g_cart_grid */
#ifdef HAVE_MPI
  MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  have_source_flag = (int)(g_cart_id == source_proc_id);
#else
  source_proc_id = 0;
  have_source_flag = 1;
#endif
  if(have_source_flag) {
    fprintf(stdout, "# [p2gg_pspace] process %2d has source location\n", source_proc_id);
  }
  sx0 = gsx0 % T;
  sx1 = gsx1 % LX;
  sx2 = gsx2 % LY;
  sx3 = gsx3 % LZ;
  if(have_source_flag) {
    fprintf(stdout, "# [p2gg_pspace] local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
  }

  /**********************************************************
   * read the position space contractions
   **********************************************************/
  ratime = _GET_TIME;

  /* set the outfile tag */
  sprintf( outfile_tag, "t%.2dx%.2dy%.2dz%.2d.tseq%.2d.gi%.2d.pix%.2dpiy%.2dpiz%.2d",
      gsx0, gsx1, gsx2, gsx3, g_sequential_source_timeslice,
      g_sequential_source_gamma_id, g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);
  if(g_cart_id == 0) {
    fprintf(stdout, "# [p2gg_pspace] file flag set to %s\n", outfile_tag);
  }

  /* sprintf(filename, "%s/%s_x.%s.%.4d", filename_prefix, outfile_name, outfile_tag, Nconf); */
  /* sprintf(filename, "%s/%s_x.%.4d", filename_prefix, outfile_name, Nconf); */

  sprintf( filename, "%s/%s_x.t%.2dx%.2dy%.2dz%.2d.tseq%.2d.g%.2d.px%.2dpy%.2dpz%.2d.%.4d",
      filename_prefix, outfile_name,
      gsx0, gsx1, gsx2, gsx3, g_sequential_source_timeslice,
      g_sequential_source_gamma_id, g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2], Nconf);

  if(g_cart_id == 0 ) fprintf(stdout, "# [p2gg_pspace] reading contraction from file %s\n", filename);
  exitstatus = read_lime_contraction(conn, filename, 16, 0);
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_pspace] Error from read_lime_contraction for file %s, status was %d\n", filename, exitstatus);
    EXIT(102);
  }
  if(do_byte_swap) {
    if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] performing byte swap\n");
    byte_swap64_v2(conn, 32*VOLUME);
  }

  if(combine_up_dn) {

    conn_buffer = (double*)calloc(32 * (VOLUME), sizeof(double));
    if( conn_buffer==(double*)NULL ) {
      fprintf(stderr, "[p2gg_pspace] could not allocate memory for contraction field\n");
      EXIT(3);
    }

    sprintf(filename, "%s/%s_x.%.4d", filename_prefix2, outfile_name, Nconf);
    if(g_cart_id == 0 ) fprintf(stdout, "# [p2gg_pspace] reading dn part of contraction from file %s\n", filename);
    exitstatus = read_lime_contraction(conn_buffer, filename, 16, 0);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg_pspace] Error from read_lime_contraction for file %s, status was %d\n", filename, exitstatus);
      EXIT(102);
    }
    if(do_byte_swap) {
      if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] performing byte swap\n");
      byte_swap64_v2(conn, 32*VOLUME);
    }

    /* combine as in p2gg_xspace */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix) shared(conn,conn_buffer,sequential_source_gamma_id_sign,g_sequential_source_gamma_id)
#endif
    for(ix=0; ix<16*VOLUME; ix++) {
      /* real part */
      conn[2*ix  ] += sequential_source_gamma_id_sign[ g_sequential_source_gamma_id ] * conn_buffer[2*ix  ];

      /* imaginary part */
      conn[2*ix+1] -= sequential_source_gamma_id_sign[ g_sequential_source_gamma_id ] * conn_buffer[2*ix+1];
    }  /* of loop on ix */

    free(conn_buffer);
  }  /* end of if combine_up_dn */
  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_pspace] time to read contraction data: %e seconds\n", retime-ratime);

  /**********************************************************
   * read the contact terms 
   **********************************************************/
  if (read_contact_term == 1) {
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
      if( fscanf(ofs, "%lf%lf", contact_term+2*mu, contact_term+2*mu+1) != 2 ) {
        fprintf(stderr, "[p2gg_pspace] Error, could not read 2 items from file %s\n", filename);
        EXIT(118);
      }
    }
    fclose(ofs);
  } else if (read_contact_term == 2) {
    /* calculate the contact term from the Ward identity 
     * onle at source location
     */
    memset(contact_term, 0, 8*sizeof(double));
    if(source_proc_id == g_cart_id) {
      ix = g_ipt[sx0][sx1][sx2][sx3];
      for(nu=0; nu<4; nu++) {    
        w.re = conn[_GWI(4*0+nu,ix          ,VOLUME)  ] + conn[_GWI(4*1+nu,ix          ,VOLUME)  ]
             + conn[_GWI(4*2+nu,ix          ,VOLUME)  ] + conn[_GWI(4*3+nu,ix          ,VOLUME)  ]
             - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)  ] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)  ]
             - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)  ] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)  ];

        w.im = conn[_GWI(4*0+nu,ix          ,VOLUME)+1] + conn[_GWI(4*1+nu,ix          ,VOLUME)+1]
             + conn[_GWI(4*2+nu,ix          ,VOLUME)+1] + conn[_GWI(4*3+nu,ix          ,VOLUME)+1]
             - conn[_GWI(4*0+nu,g_idn[ix][0],VOLUME)+1] - conn[_GWI(4*1+nu,g_idn[ix][1],VOLUME)+1]
             - conn[_GWI(4*2+nu,g_idn[ix][2],VOLUME)+1] - conn[_GWI(4*3+nu,g_idn[ix][3],VOLUME)+1];
        contact_term[2*nu  ] = w.re;
        contact_term[2*nu+1] = w.im;
      }
    }
#ifdef HAVE_MPI
    MPI_Bcast(contact_term, 8, MPI_DOUBLE, source_proc_id, g_cart_grid);
#endif
  } else {
    /* set contact term to zero */
    for(mu=0; mu<8; mu++) { contact_term[mu] = 0.; }
  }
  /* print contact term */
  if(have_source_flag) {
    fprintf(stdout, "# [p2gg_pspace] contact term\n");
    for(i=0;i<4;i++) {
      fprintf(stdout, "\t%d%25.16e%25.16e\n", i, contact_term[2*i], contact_term[2*i+1]);
    }
  }

  /**********************************************************
   * subtract contact term
   **********************************************************/
  if(source_proc_id == g_cart_id) {
    ix = g_ipt[sx0][sx1][sx2][sx3];
    conn[_GWI( 0,ix, VOLUME)  ] -= contact_term[ 0];
    conn[_GWI( 0,ix, VOLUME)+1] -= contact_term[ 1];
    conn[_GWI( 5,ix, VOLUME)  ] -= contact_term[ 2];
    conn[_GWI( 5,ix, VOLUME)+1] -= contact_term[ 3];
    conn[_GWI(10,ix, VOLUME)  ] -= contact_term[ 4];
    conn[_GWI(10,ix, VOLUME)+1] -= contact_term[ 5];
    conn[_GWI(15,ix, VOLUME)  ] -= contact_term[ 6];
    conn[_GWI(15,ix, VOLUME)+1] -= contact_term[ 7];
  }

  /**********************************************************
   * check the Ward identity in position space
   **********************************************************/

  if(check_position_space_WI) {
    ratime = _GET_TIME;
#ifdef HAVE_MPI
    xchange_contraction(conn, 32);
#endif
    double dtmp = 0.;
    sprintf(filename, "p2gg_pspace_WI_X.%s.%.4d.%.4d", outfile_tag, Nconf, g_cart_id);
    /* ofs = fopen(filename,"w"); */
    if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] checking Ward identity in position space\n");
    for(x0=0; x0<T;  x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      /* fprintf(ofs, "# t=%2d x=%2d y=%2d z=%2d\n", x0+g_proc_coords[0] * T, x1+g_proc_coords[1]*LX, x2+g_proc_coords[2]*LY, x3+g_proc_coords[3]*LZ); */
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

        /* fprintf(ofs, "\t%3d%25.16e%25.16e\n", nu, w.re, w.im); */
        dtmp += w.re*w.re + w.im*w.im;
      }
    }}}}
    /* fclose(ofs); */
#ifdef HAVE_MPI
    double dtmp2;
    MPI_Allreduce(&dtmp, &dtmp2, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid);
    dtmp = dtmp2;
#endif
    dtmp = sqrt(dtmp);
    if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] WI norm = %e\n", dtmp);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] time to check Ward identity in position space = %e seconds\n", retime-ratime);

  }  /* of if check_position_space_WI */


  /*********************************************
   * Fourier transformation
   *   for list of sink momenta
   *********************************************/
  conn_ft = (double***)malloc(g_sink_momentum_number*sizeof(double**));
  if(conn_ft == NULL) {
    fprintf(stderr, "[p2gg_pspace] Error from malloc\n");
    EXIT(12);
  }
  conn_ft[0] = (double**)malloc(16*g_sink_momentum_number*sizeof(double*));
  if(conn_ft[0] == NULL) {
    fprintf(stderr, "[p2gg_pspace] Error from malloc\n");
    EXIT(13);
  }
  for(i=1; i<g_sink_momentum_number; i++) {
    conn_ft[i] = conn_ft[i-1] + 16;
  }
  conn_ft[0][0] = (double*)malloc(2*T_global*16*g_sink_momentum_number*sizeof(double));
  if(conn_ft[0][0] == NULL) {
    fprintf(stderr, "[p2gg_pspace] Error from malloc\n");
    EXIT(14);
  }
  k = 0;
  for(i=0; i<g_sink_momentum_number; i++) {
  for(mu=0; mu<16; mu++) {
    if(k==0) {
      k++;
      continue;
    }
    conn_ft[i][mu] = conn_ft[0][0] + k * 2*T_global;
    k++;
  }}

  phase_field = (double*)malloc(2*VOL3*sizeof(double));
  if(phase_field == NULL) {
    fprintf(stderr, "[p2gg_pspace] Error from malloc\n");
    EXIT(15);
  }

  /*********************************************
   * loop on sink momentum vectors
   *********************************************/
  for(i_sink_momentum=0; i_sink_momentum<g_sink_momentum_number; i_sink_momentum++) {

    if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] FT for sink momentum (%d, %d, %d)\n", 
        g_sink_momentum_list[i_sink_momentum][0], g_sink_momentum_list[i_sink_momentum][1], g_sink_momentum_list[i_sink_momentum][2]);

    ratime = _GET_TIME;

    /* sink momentum */
    psnk[0] = 0.;
    psnk[1] = TWO_MPI * g_sink_momentum_list[i_sink_momentum][0] / LX_global;  /* = 2pi/L1 p1 */
    psnk[2] = TWO_MPI * g_sink_momentum_list[i_sink_momentum][1] / LY_global;  /* = 2pi/L2 p2 */
    psnk[3] = TWO_MPI * g_sink_momentum_list[i_sink_momentum][2] / LZ_global;  /* = 2pi/L3 p3 */
    phase_shift = g_proc_coords[1]*LX*psnk[1] + g_proc_coords[2]*LY*psnk[2] + g_proc_coords[3]*LZ*psnk[3];

    /* source momentum */
    psrc[0] = 0.;
    psrc[1] = -( psnk[1] +  g_seq_source_momentum[0] / LX_global * TWO_MPI );
    psrc[2] = -( psnk[2] +  g_seq_source_momentum[1] / LY_global * TWO_MPI );
    psrc[3] = -( psnk[3] +  g_seq_source_momentum[2] / LZ_global * TWO_MPI );

    if(g_cart_id == 0) {
      fprintf(stdout, "# [p2gg_pspace] pi = (%f, %f, %f), pf = (%f, %f, %f)\n",
          psrc[1], psrc[2], psrc[3],
          psnk[1], psnk[2], psnk[3]);
    }

#ifdef HAVE_OPENMP
#pragma omp parallel for private(x1,x2,x3,ix,phase) shared(phase_shift)
#endif
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[0][x1][x2][x3];
      phase = x1*psnk[1] + x2*psnk[2] + x3*psnk[3] + phase_shift;
      phase_field[2*ix  ] = cos(phase);
      phase_field[2*ix+1] = sin(phase);
      /* TEST */
      /* fprintf(stdout, "%3d %3d %3d %25.16e %25.16e\n", x1, x2, x3, phase_field[2*ix], phase_field[2*ix+1]); */
    }}}


    for(mu=0; mu<4; mu++) {
    for(nu=0; nu<4; nu++) {
      imunu = 4*mu+nu;
      memset(conn_ft[i_sink_momentum][imunu], 0, 2*T_global*sizeof(double));

      for(x0=0; x0<T; x0++) {

        conn_ptr_p = (complex*)(conn_ft[i_sink_momentum][imunu] + 2*(x0 + g_proc_coords[0]*T) );
        
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix)
#endif
        for(ix=0; ix<VOL3; ix++) {
          conn_ptr_x = (complex*)(conn + _GWI(imunu,(x0*VOL3+ix),VOLUME) );
          _co_pl_eq_co_ti_co( conn_ptr_p, conn_ptr_x, (complex*)(phase_field+2*ix) );
        }

        /**********************************************************
         * multiply with phase at source location
         **********************************************************/
        phase = ( psrc[1] * gsx1 + psrc[2] * gsx2 + psrc[3] * gsx3 ) + 0.5 * (psnk[mu] + psrc[nu]);
        w.re  = cos(phase);
        w.im  = sin(phase);

        /* if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] phase at souce = %e, %e\n", w.re, w.im); */

        _co_eq_co_ti_co(&w1, conn_ptr_p, &w);
        _co_eq_co(conn_ptr_p, &w1);

      }  /* end of loop on x0 */
    }}  /* end of loop on mu and nu */

#if 0
#endif  /* of if 0 */

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [p2gg_pspace] time for FT = %e seconds\n", retime-ratime);
  }  /* end of loop on sink momentum list */

  /***************************************
   * free memory for x-space contractions
   ***************************************/
  if(conn        != NULL) free(conn);
  if(phase_field != NULL) free(phase_field);


#ifdef HAVE_MPI
  /********************************
   * exchange in t-direction
   ********************************/
  k = 32*T_global*g_sink_momentum_number;
  mbuffer = (double*)malloc(k * sizeof(double));
  if(mbuffer == NULL) {
    fprintf(stderr, "[p2gg_pspace] Error from malloc\n");
    EXIT(17);
  }
  memcpy(mbuffer, conn_ft[0][0], k * sizeof(double));
  exitstatus = MPI_Allgather(mbuffer, k, MPI_DOUBLE, conn_ft[0][0], k, MPI_DOUBLE, g_cart_grid);
  if(exitstatus != MPI_SUCCESS) {
    fprintf(stderr, "[] Error from MPI_Allgather, status was %d\n", exitstatus);
    EXIT(18);
  }
  free(mbuffer); mbuffer = NULL;
#endif


  /********************************
   * save momentum space results
   ********************************/
  ratime = _GET_TIME;
  
  if(io_proc == 2) {
    for(i_sink_momentum = 0; i_sink_momentum<g_sink_momentum_number; i_sink_momentum++) {
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        imunu = 4 * mu + nu;

        sprintf(filename, "%s_p.ti%.2d.gi%.2d.pix%.2dpiy%.2dpiz%.2d.pfx%.2dpfy%.2dpfz%.2d.mu%dnu%d.t%.2dx%.2dy%.2dz%.2d.%.4d",
            outfile_name, g_sequential_source_timeslice, g_sequential_source_gamma_id,
            g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
            g_sink_momentum_list[i_sink_momentum][0], g_sink_momentum_list[i_sink_momentum][1], g_sink_momentum_list[i_sink_momentum][2], mu, nu,
            gsx0, gsx1, gsx2, gsx3, Nconf);

        ofs = fopen(filename, "w");
        if (ofs == NULL) {
          fprintf(stderr, "[p2gg_pspace] Error, could not open file %s for writing\n", filename);
          EXIT(20);
        }

        for(x0=0; x0<T_global; x0++) {
          x1 = ( gsx0 + x0 ) % T_global;  /* distance from source timeslice */
          fprintf(ofs, "\t%3d%25.16e%25.16e\n", x0, conn_ft[i_sink_momentum][imunu][2*x1], conn_ft[i_sink_momentum][imunu][2*x1+1]);
        }

        fclose( ofs );

      }}  /* end of loop on mu and nu */
    }  /* end of loop on sink momentum list */

  }  /* end of if io_proc == 2 */

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_pspace] time for writing momentum space results = %e seconds\n", retime-ratime);

  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  free_geometry();
  if(conn_ft != NULL) {
    if(conn_ft[0] != NULL) {
      if(conn_ft[0][0] != NULL) free(conn_ft[0][0]);
      free(conn_ft[0]);
    }
    free(conn_ft);
  }
  
#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_pspace] %s# [p2gg_pspace] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_pspace] %s# [p2gg_pspace] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
