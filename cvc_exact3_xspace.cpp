/****************************************************
 * cvc_exact3_xspace.c
 *
 * Sun Nov 20 12:42:18 CET 2016
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
#include "contract_cvc_tensor.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, mu, nu, ir, ia, ib, imunu;
  int op_id = 0;
  int filename_set = 0;
  int source_location=0, have_source_flag = 0, have_shifted_source_flag = 0;
  int x0, x1, x2, x3, ix;
  int gsx[4];
  int sx0, sx1, sx2, sx3;
  int check_position_space_WI=0;
  int threadid=0;
  int exitstatus;
  int write_ascii=0;
  int source_proc_coords[4], source_proc_id = -1;
  int shifted_source_coords[4], shifted_source_proc_coords[4];
  double *conn = (double*)NULL;
  double contact_term[8];
  char filename[100], contype[400];
  double ratime, retime;
  double plaq;
  double *source=NULL, *propagator=NULL;
  complex w, w1;
/*
  int gperm[5][4], gperm2[4][4];
  int isimag[4];
  double gperm_sign[5][4], gperm2_sign[4][4];
  double Usourcebuff[72], *Usource[4];
  double spinor1[24], spinor2[24], U_[18];
  double *phi=NULL, *chi=NULL;
*/
  double *fwd_list[5][12], *bwd_list[5][12];
  int LLBase[4];
  FILE *ofs;

#ifdef HAVE_MPI
  int *status;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "wah?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_position_space_WI = 1;
      fprintf(stdout, "\n# [cvc_exact3_xspace] will check Ward identity in position space\n");
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "\n# [cvc_exact3_xspace] will write data in ASCII format too\n");
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
  fprintf(stdout, "# [cvc_exact3_xspace] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [] calling tmLQCD wrapper init functions\n");

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
/*  mpi_init_xchange_contraction(32); */
  mpi_init_xchange_contraction(2);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  g_num_threads = 1;
#endif

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[cvc_exact3_xspace] T and L's must be set\n");
    usage();
  }


#ifdef HAVE_MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    EXIT(7);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[cvc_exact3_xspace] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  LLBase[0] = T_global;
  LLBase[1] = LX_global;
  LLBase[2] = LY_global;
  LLBase[3] = LZ_global;

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] reading gauge field from file %s\n", filename);
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
  if(g_cart_id== 0) fprintf(stdout, "[] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(560);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(561);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[] Error, &g_gauge_field is NULL\n");
    EXIT(563);
  }
#endif

#ifdef HAVE_MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] measured plaquette value: %25.16e\n", plaq);

  /* allocate memory for the spinor fields */
  no_fields = 120;
#ifdef HAVE_TMLQCD_LIBWRAPPER
  no_fields++;
#endif

  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  source = g_spinor_field[no_fields-1];
#endif

  /* allocate memory for the contractions */
  conn = (double*)malloc(32 * VOLUME * sizeof(double));
  if( conn == NULL ) {
    fprintf(stderr, "[cvc_exact3_xspace] could not allocate memory for contr. fields\n");
    EXIT(3);
  }
  memset(conn, 0, 32*VOLUME*sizeof(double));

  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  gsx[0] = g_source_location / ( LX_global * LY_global * LZ_global);
  gsx[1] = (g_source_location % ( LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  gsx[2] = (g_source_location % ( LY_global * LZ_global)) / LZ_global;
  gsx[3] = (g_source_location % LZ_global);
  source_proc_coords[0] = gsx[0] / T;
  source_proc_coords[1] = gsx[1] / LX;
  source_proc_coords[2] = gsx[2] / LY;
  source_proc_coords[3] = gsx[3] / LZ;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [cvc_exact3_xspace] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx[0], gsx[1], gsx[2], gsx[3]);
    fprintf(stdout, "# [cvc_exact3_xspace] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  have_source_flag = (int)(g_cart_id == source_proc_id);
  if(have_source_flag==1) {
    fprintf(stdout, "# [cvc_exact3_xspace] process %2d has source location\n", source_proc_id);
  }
  sx0 = gsx[0] % T;
  sx1 = gsx[1] % LX;
  sx2 = gsx[2] % LY;
  sx3 = gsx[3] % LZ;
  have_source_flag = source_proc_id;
# else
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "[cvc_exact3_xspace] process %2d has source location\n", g_cart_id);
  sx0 = g_source_location/(LX*LY*LZ)-Tstart;
  sx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  sx2 = (g_source_location%(LY*LZ)) / LZ;
  sx3 = (g_source_location%LZ);
#ifdef HAVE_MPI
  MPI_Gather(&have_source_flag, 1, MPI_INT, status, 1, MPI_INT, 0, g_cart_grid);
  if(g_cart_id==0) {
    for(mu=0; mu<g_nproc; mu++) fprintf(stdout, "# [cvc_exact3_xspace] status[%1d]=%d\n", mu,status[mu]);
  }
  if(g_cart_id==0) {
    for(have_source_flag=0; status[have_source_flag]!=1; have_source_flag++);
    fprintf(stdout, "# [cvc_exact3_xspace] have_source_flag= %d\n", have_source_flag);
  }
  MPI_Bcast(&have_source_flag, 1, MPI_INT, 0, g_cart_grid);
#else
  have_source_flag = 0;
#endif

#endif

  /* TEST */
/*
  for(mu=0; mu<4; mu++) {
    for(i=0; i<9; i++) {
      fprintf(stdout, "Usource proc%.4d %3d %3d %25.16e%25.16e\n", g_cart_id, mu, i, Usource[mu][2*i], Usource[mu][2*i+1]);
    }
  }
*/

  // initialize the contact term
  memset(contact_term, 0, 8*sizeof(double));

#ifdef HAVE_MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

#ifdef HAVE_TMLQCD_LIBWRAPPER
  if(g_read_propagator) {
#endif

    /**********************************************************
     * read up spinor fields
     **********************************************************/
    for(mu=0; mu<5; mu++) {
      for(ia=0; ia<12; ia++) {
        get_filename(filename, mu, ia, 1);
        exitstatus = read_lime_spinor(g_spinor_field[mu*12+ia], filename, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[cvc_exact3_xspace] Error from read_lime_spinor, status was %d\n", exitstatus);
          EXIT(111);
        }
        xchange_field(g_spinor_field[mu*12+ia]);
      }  /* of loop on ia */
    }    /* of loop on mu */

    /**********************************************************
     * read dn spinor fields
     **********************************************************/
    for(mu=0; mu<5; mu++) {
      for(ia=0; ia<12; ia++) {

        get_filename(filename, mu, ia, -1);
        exitstatus = read_lime_spinor(g_spinor_field[60+mu*12+ia], filename, 0);
/*
        get_filename(filename, mu, ia, 1);
        exitstatus = read_lime_spinor(g_spinor_field[60+mu*12+ia], filename, 1);
*/
        if(exitstatus != 0) {
          fprintf(stderr, "[cvc_exact3_xspace] Error from read_lime_spinor, status was %d\n", exitstatus);
          EXIT(112);
        }
        xchange_field(g_spinor_field[60+mu*12+ia]);
      }  /* of loop on ia */
    }    /* of loop on mu */

#ifdef HAVE_TMLQCD_LIBWRAPPER
  }  /* end of if read propagator */
#endif

#ifdef HAVE_TMLQCD_LIBWRAPPER
  if(!g_read_propagator) {

    /***********************************************************
     * invert using tmLQCD invert
     ***********************************************************/
    if(g_tmLQCD_lat.no_operators != 2) {
      fprintf(stderr, "[] Error, confused about number of operators, expected 2 operators (up-type, dn-tpye)\n");
      EXIT(6);
    }
 
    /* op_id runs over 0 / up-type quarks and 1 / dn-type quarks */

    /*************************************************************
     *************************************************************
     **
     ** make sure eigenvectors are recalculated when chaning from
     **   op_id 0 to op_id 1
     **
     *************************************************************
     *************************************************************/

    for ( op_id = 0; op_id < 2; op_id++ ) {

      if(g_cart_id == 0) fprintf(stdout, "# [] using op_id = %d\n", op_id);

      for(mu=0; mu<5; mu++)
      {
  
        shifted_source_coords[0] = gsx[0];
        shifted_source_coords[1] = gsx[1];
        shifted_source_coords[2] = gsx[2];
        shifted_source_coords[3] = gsx[3];
  
        /* fprintf(stdout, "# [] proc%.4d global source coords = ( %d, %d, %d, %d )\n", g_cart_id,
            shifted_source_coords[0], shifted_source_coords[1],
            shifted_source_coords[2], shifted_source_coords[3]); */



        if(mu < 4) shifted_source_coords[mu] = ( shifted_source_coords[mu] + 1 ) % LLBase[mu];

        /* fprintf(stdout, "# [] proc%.4d shifted global source coords (%d) = ( %d, %d, %d, %d )\n", g_cart_id, mu,
            shifted_source_coords[0], shifted_source_coords[1],
            shifted_source_coords[2], shifted_source_coords[3]); */
        
        shifted_source_proc_coords[0] = shifted_source_coords[0] / T;
        shifted_source_proc_coords[1] = shifted_source_coords[1] / LX;
        shifted_source_proc_coords[2] = shifted_source_coords[2] / LY;
        shifted_source_proc_coords[3] = shifted_source_coords[3] / LZ;
 
        /* fprintf(stdout, "# [] proc%.4d shifted source proc coords (%d) = ( %d, %d, %d, %d )\n", g_cart_id, mu,
          shifted_source_proc_coords[0], shifted_source_proc_coords[1],
          shifted_source_proc_coords[2], shifted_source_proc_coords[3]); */

        MPI_Cart_rank(g_cart_grid, shifted_source_proc_coords, &source_proc_id);
        have_shifted_source_flag = (int)(g_cart_id == source_proc_id);
  
        shifted_source_coords[0] = shifted_source_coords[0] % T;
        shifted_source_coords[1] = shifted_source_coords[1] % LX;
        shifted_source_coords[2] = shifted_source_coords[2] % LY;
        shifted_source_coords[3] = shifted_source_coords[3] % LZ;

        /* fprintf(stdout, "# [] proc%.4d shifted local source coords (%d) = ( %d, %d, %d, %d )\n", g_cart_id, mu,
            shifted_source_coords[0], shifted_source_coords[1],
            shifted_source_coords[2], shifted_source_coords[3]); */

        if(have_shifted_source_flag) {
          fprintf(stdout, "# [] proc%.4d have shifted source flag\n", g_cart_id);
          fprintf(stdout, "# [] shifted source proc coords  (%d) = ( %d, %d, %d, %d )\n", mu,
              shifted_source_proc_coords[0], shifted_source_proc_coords[1],
              shifted_source_proc_coords[2], shifted_source_proc_coords[3]);
          fprintf(stdout, "# [] global shifted source coords (%d) = ( %d, %d, %d, %d )\n", mu,
              shifted_source_coords[0] + g_proc_coords[0] * T,  shifted_source_coords[1] + g_proc_coords[1] * LX,
              shifted_source_coords[2] + g_proc_coords[2] * LY, shifted_source_coords[3] + g_proc_coords[3] * LZ );
          fprintf(stdout, "# [] local shifted source coords  (%d) = ( %d, %d, %d, %d )\n", mu,
              shifted_source_coords[0], shifted_source_coords[1],
              shifted_source_coords[2], shifted_source_coords[3] );
        }

        for(ia=0; ia<12; ia++) {

          if(g_cart_id == 0) fprintf(stdout, "# [] inverting for operator %d for spin-color component (%d, %d)\n", op_id, ia/3, ia%3);

          memset(source, 0, 24*VOLUME*sizeof(double));
          if(have_shifted_source_flag) {
            source[ _GSI( g_ipt[shifted_source_coords[0]][shifted_source_coords[1]][shifted_source_coords[2]][shifted_source_coords[3]] ) + 2*ia ] = 1.;
          }

          xchange_field(source);

          propagator = g_spinor_field[12 * ( op_id * 5 + mu ) + ia];
  
          exitstatus = tmLQCD_invert(propagator, source, op_id, g_write_propagator);
          if(exitstatus != 0) {
            fprintf(stderr, "[] Error from tmLQCD_invert, status was %d\n", exitstatus);
            EXIT(7);
          }

          xchange_field(propagator);

        }  /* end of loop on spin-color component */

      }  /* end of loop on mu shift direction */

    }  /* end of loop on op_id */

  }  /* end of if not read propagator */


#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */

#ifdef HAVE_MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] reading / invert in %e seconds\n", retime-ratime);


  /**********************************************************
   **********************************************************
   **
   ** contractions
   **
   **********************************************************
   **********************************************************/  

  init_contract_cvc_tensor_gperm();
  init_contract_cvc_tensor_usource(g_gauge_field, gsx);


#ifdef HAVE_MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  for(mu=0; mu<5; mu++) {
    for(i=0; i<12; i++) {
      fwd_list[mu][i] = g_spinor_field[mu*12+i];
      bwd_list[mu][i] = g_spinor_field[60 + mu*12+i];
    }
  }

  contract_cvc_tensor(conn, contact_term, fwd_list, bwd_list);

  /* normalisation of contractions */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for(ix=0; ix<32*VOLUME; ix++) conn[ix] *= -0.25;

#ifdef HAVE_MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] contractions in %e seconds\n", retime-ratime);

  /* save results */
#ifdef HAVE_MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(strcmp(g_outfile_prefix, "NA") == 0) {
    sprintf(filename, "cvc2_v_x.%.4d", Nconf);
  } else {
    sprintf(filename, "%s/cvc2_v_x.%.4d", g_outfile_prefix, Nconf);
  }
  sprintf(contype, "cvc - cvc in position space, all 16 components");
  write_lime_contraction(conn, filename, 64, 16, contype, Nconf, 0);

  /* TEST */
/*
  for(ix=0;ix<VOLUME;ix++) {
    for(mu=0;mu<16;mu++) {
      fprintf(stdout, "%2d%6d%3d%25.16e%25.16e\n", g_cart_id, ix, mu, conn[_GWI(mu,ix,VOLUME)], conn[_GWI(mu,ix,VOLUME)+1]);
    }
  }
*/

  if(write_ascii) {
#ifndef HAVE_MPI
    if(strcmp(g_outfile_prefix, "NA") == 0) {
      sprintf(filename, "cvc2_v_x.%.4d.ascii", Nconf);
    } else {
      sprintf(filename, "%s/cvc2_v_x.%.4d.ascii", g_outfile_prefix, Nconf);
    }
    write_contraction(conn, NULL, filename, 16, 2, 0);
#else
    sprintf(filename, "cvc2_v_x.%.4d.ascii.%.2d", Nconf, g_cart_id);
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
  }

#ifdef HAVE_MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] saved position space results in %e seconds\n", retime-ratime);


#ifdef HAVE_MPI
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] broadcasing contact term ...\n");
  MPI_Bcast(contact_term, 8, MPI_DOUBLE, have_source_flag, g_cart_grid);
  fprintf(stdout, "[%2d] contact term = "\
      "(%e + I %e, %e + I %e, %e + I %e, %e +I %e)\n",
      g_cart_id, contact_term[0], contact_term[1], contact_term[2], contact_term[3],
      contact_term[4], contact_term[5], contact_term[6], contact_term[7]);
#endif

  /* subtract contact term */
  if( have_source_flag == g_cart_id ) {
    fprintf(stdout, "# [] process %d subtracting contact term\n", g_cart_id);
    ix = g_ipt[sx0][sx1][sx2][sx3];
    for(mu=0; mu<4; mu++) {
      conn[_GWI(5*mu,ix,VOLUME)    ] -= contact_term[2*mu  ];
      conn[_GWI(5*mu,ix,VOLUME) + 1] -= contact_term[2*mu+1];
    }
  }
  


  /* check the Ward identity in position space */
  if(check_position_space_WI) {

#if 0
#ifdef HAVE_MPI
    xchange_contraction(conn, 32);
    sprintf(filename, "post_xchange.%.4d", g_cart_id);
    ofs = fopen(filename,"w");

    int start_valuet = 1;
#  if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ
    int start_valuex = 1;
#  else
    int start_valuex = 0;
#  endif
#  if defined PARALLELTXY || defined PARALLELTXYZ
    int start_valuey = 1;
#  else
    int start_valuey = 0;
#  endif
#  if defined PARALLELTXYZ
    int start_valuez = 1;
#  else
    int start_valuez = 0;
#  endif

    int y0, y1, y2, y3, isboundary;
    for(x0=-start_valuet; x0<T+start_valuet;  x0++) {
    for(x1=-start_valuex; x1<LX+start_valuex; x1++) {
    for(x2=-start_valuey; x2<LY+start_valuey; x2++) {
    for(x3=-start_valuez; x3<LZ+start_valuez; x3++) {
      y0=x0; y1=x1; y2=x2; y3=x3;
      if(x0==-1) y0=T +1;
      if(x1==-1) y1=LX+1;
      if(x2==-1) y2=LY+1;
      if(x3==-1) y3=LZ+1;

      int z0 = ( (x0 + T  * g_proc_coords[0] ) + T_global  ) % T_global;
      int z1 = ( (x1 + LX * g_proc_coords[1] ) + LX_global ) % LX_global;
      int z2 = ( (x2 + LY * g_proc_coords[2] ) + LY_global ) % LY_global;
      int z3 = ( (x3 + LZ * g_proc_coords[3] ) + LZ_global ) % LZ_global;
     

      isboundary = 0;
      if(x0==-1 || x0== T) isboundary++;
      if(x1==-1 || x1==LX) isboundary++;
      if(x2==-1 || x2==LY) isboundary++;
      if(x3==-1 || x3==LZ) isboundary++;
      if(isboundary > 1) { continue; }
      ix = g_ipt[y0][y1][y2][y3];

      fprintf(ofs, "# t=%2d x=%2d y=%2d z=%2d\n", z0, z1, z2, z3);
      for(nu=0; nu < 16; nu++) {
        fprintf(ofs, "\t%3d%25.16e%25.16e\n", nu, conn[_GWI(nu,ix,VOLUME)], conn[_GWI(nu,ix,VOLUME)+1]);
      }
    }}}}
    fclose(ofs);
#endif
#endif
#ifdef HAVE_MPI
    unsigned int VOLUMEplusRAND = VOLUME + RAND;
    unsigned int stride = VOLUMEplusRAND;
    double *conn_buffer = (double*)malloc(32*VOLUMEplusRAND*sizeof(double));
    if(conn_buffer == NULL) {
      EXIT(14);
    }
    /* size_t bytes = 32 * VOLUME * sizeof(double);
     memcpy(conn_buffer, conn, bytes);
    xchange_contraction(conn_buffer, 32); */
    for(mu=0; mu<16; mu++) {
      memcpy(conn_buffer+2*mu*VOLUMEplusRAND, conn+2*mu*VOLUME, 2*VOLUME*sizeof(double));
      xchange_contraction(conn_buffer+2*mu*VOLUMEplusRAND, 2);
      /* if(g_cart_id == 0) { fprintf(stdout, "# [] xchanged for mu = %d\n", mu); fflush(stdout);} */
    }
#else
    double *conn_buffer = conn;
#endif
    /* sprintf(filename, "WI_X.%.4d.%.4d", Nconf, g_cart_id);
    ofs = fopen(filename,"w");
    */

    fprintf(stdout, "\n# [cvc_exact3_xspace] checking Ward identity in position space ...\n");
    for(nu=0; nu<4; nu++) {
      double norm=0;

      for(x0=0; x0<T;  x0++) {
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        /* fprintf(ofs, "# t=%2d x=%2d y=%2d z=%2d\n", x0, x1, x2, x3); */
        ix=g_ipt[x0][x1][x2][x3];

        w.re = conn_buffer[_GWI(4*0+nu,ix          ,stride)  ] + conn_buffer[_GWI(4*1+nu,ix          ,stride)  ]
             + conn_buffer[_GWI(4*2+nu,ix          ,stride)  ] + conn_buffer[_GWI(4*3+nu,ix          ,stride)  ]
             - conn_buffer[_GWI(4*0+nu,g_idn[ix][0],stride)  ] - conn_buffer[_GWI(4*1+nu,g_idn[ix][1],stride)  ]
             - conn_buffer[_GWI(4*2+nu,g_idn[ix][2],stride)  ] - conn_buffer[_GWI(4*3+nu,g_idn[ix][3],stride)  ];

        w.im = conn_buffer[_GWI(4*0+nu,ix          ,stride)+1] + conn_buffer[_GWI(4*1+nu,ix          ,stride)+1]
             + conn_buffer[_GWI(4*2+nu,ix          ,stride)+1] + conn_buffer[_GWI(4*3+nu,ix          ,stride)+1]
             - conn_buffer[_GWI(4*0+nu,g_idn[ix][0],stride)+1] - conn_buffer[_GWI(4*1+nu,g_idn[ix][1],stride)+1]
             - conn_buffer[_GWI(4*2+nu,g_idn[ix][2],stride)+1] - conn_buffer[_GWI(4*3+nu,g_idn[ix][3],stride)+1];

        /* fprintf(ofs, "\t%3d%25.16e%25.16e\n", nu, w.re, w.im); */
          norm += w.re*w.re + w.im*w.im;
      }}}}

#ifdef HAVE_MPI
      double norm2 = norm;
      if( MPI_Allreduce(&norm2, &norm, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid) != MPI_SUCCESS ) {
        fprintf(stderr, "[] Error from MPI_Allreduce\n");
        EXIT(12);
      }
#endif
      if(g_cart_id == 0) fprintf(stdout, "# [] WI nu = %d norm = %25.16e\n", nu, sqrt(norm));
    }


#ifdef HAVE_MPI
    free(conn_buffer);
#endif
    /* fclose(ofs); */
  }  /* end of if check_position_space_WI */



  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  /* free(g_gauge_field); */
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  free(conn);



#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  free(status);
  mpi_fini_xchange_contraction();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [cvc_exact3_xspace] %s# [cvc_exact3_xspace] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [cvc_exact3_xspace] %s# [cvc_exact3_xspace] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
