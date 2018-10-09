/****************************************************
 * cvc_exact3_xspace.cpp
 *
 * Fri Mar 10 09:24:21 CET 2017
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
#include "matrix_init.h"
#include "project.h"
#include "scalar_products.h"
#include "dummy_solver.h"

using namespace cvc;

int get_point_source_info (int gcoords[4], int lcoords[4], int*proc_id) {

  int source_proc_id = 0;
  int exitstatus;
#ifdef HAVE_MPI
  int source_proc_coords[4];
  source_proc_coords[0] = gcoords[0] / T;
  source_proc_coords[1] = gcoords[1] / LX;
  source_proc_coords[2] = gcoords[2] / LY;
  source_proc_coords[3] = gcoords[3] / LZ;
  exitstatus = MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  if(exitstatus !=  MPI_SUCCESS ) {
    fprintf(stderr, "[get_point_source_info] Error from MPI_Cart_rank, status was %d\n", exitstatus);
    EXIT(9);
  }
  if(source_proc_id == g_cart_id) {
    fprintf(stdout, "# [get_point_source_info] process %2d = (%3d,%3d,%3d,%3d) has source location\n", source_proc_id,
        source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
   }
#endif

  if(proc_id != NULL) *proc_id = source_proc_id;
  int x[4] = {-1,-1,-1,-1};
  /* local coordinates */
  if(g_cart_id == source_proc_id) {
    x[0] = gcoords[0] % T;
    x[1] = gcoords[1] % LX;
    x[2] = gcoords[2] % LY;
    x[3] = gcoords[3] % LZ;
    if(lcoords != NULL) {
      memcpy(lcoords,x,4*sizeof(int));
    }
  }
  return(0);
}  /* end of get_point_source_info */



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
  
  int c, i, mu, nu, ia,imunu;
  int op_id = 0;
  int filename_set = 0;
  int x0, x1, x2, x3;
  unsigned int ix, ixeo;
  int gsx[4];
  int sx[4];
  int check_position_space_WI=0;
  int exitstatus;
  int write_ascii=0, write_binary=0;
  int source_proc_id = 0;
  int g_shifted_source_coords[4], shifted_source_coords[4], shifted_source_proc_id = 0;
  int no_eo_fields = 0;
  unsigned int Vhalf;
  double *conn_e=NULL, *conn_o=NULL; 
  double contact_term[8];
  char filename[100];
  char contype[400];
  double ratime, retime;
  double plaq;
  double *source=NULL, *propagator=NULL, **eo_spinor_field = NULL;
  complex w;
  double *sprop_list_e[60], *sprop_list_o[60], *tprop_list_e[60], *tprop_list_o[60];
  double *gauge_field_with_phase = NULL;
  int LLBase[4];
  FILE *ofs;

#ifdef HAVE_MPI
  int *status;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "bwah?f:")) != -1) {
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
    case 'b':
      write_binary = 1;
      fprintf(stdout, "\n# [cvc_exact3_xspace] will write data in binary format\n");
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

  fprintf(stdout, "# [cvc_exact3_xspace] calling tmLQCD wrapper init functions\n");

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

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();



  Vhalf = VOLUME / 2;

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
    if(g_cart_id==0) fprintf(stdout, "\n# [cvc_exact3_xspace] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[cvc_exact3_xspace] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(560);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(561);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[cvc_exact3_xspace] Error, &g_gauge_field is NULL\n");
    EXIT(563);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  alloc_gauge_field(&gauge_field_with_phase, VOLUMEPLUSRAND);
  for( ix=0; ix<VOLUME; ix++ ) {
    for ( mu=0; mu<4; mu++ ) {
      _cm_eq_cm_ti_co ( gauge_field_with_phase+_GGI(ix,mu), g_gauge_field+_GGI(ix,mu), &co_phase_up[mu] );
    }
  }

#ifdef HAVE_MPI
  xchange_gauge();
  xchange_gauge_field(gauge_field_with_phase);
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] measured plaquette value: %25.16e\n", plaq);

  plaquette2(&plaq, gauge_field_with_phase );
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] gauge field with phase measured plaquette value: %25.16e\n", plaq);



  /* allocate memory for the spinor fields */
  no_eo_fields = 240;
  eo_spinor_field = (double**)malloc(no_eo_fields * sizeof(double*));
  eo_spinor_field[0] = (double*)malloc(no_eo_fields * _GSI(Vhalf) * sizeof(double));
  for(i=1; i<no_eo_fields; i++) eo_spinor_field[i] = eo_spinor_field[i-1] + _GSI(Vhalf);

  no_fields = 2;
  g_spinor_field = (double**)malloc(no_fields * sizeof(double*));
  g_spinor_field[0] = (double*)malloc(no_fields * _GSI(VOLUME+RAND) * sizeof(double));
  for(i=1; i<no_fields; i++) g_spinor_field[i] = g_spinor_field[i-1] + _GSI(VOLUME+RAND);
  source     = g_spinor_field[0];
  propagator = g_spinor_field[1];

  /* allocate memory for the contractions */
  conn_e = (double*)malloc(32 * VOLUME * sizeof(double));
  if( conn_e == NULL ) {
    fprintf(stderr, "[cvc_exact3_xspace] could not allocate memory for contr. fields\n");
    EXIT(3);
  }
  conn_o = conn_e + 32*Vhalf;

  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/
  gsx[0] = g_source_location / ( LX_global * LY_global * LZ_global);
  gsx[1] = (g_source_location % ( LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  gsx[2] = (g_source_location % ( LY_global * LZ_global)) / LZ_global;
  gsx[3] = (g_source_location % LZ_global);

  exitstatus = get_point_source_info (gsx, sx, &source_proc_id );

  ratime = _GET_TIME;

#ifdef HAVE_TMLQCD_LIBWRAPPER
  if(g_read_propagator) {
#endif

    /**********************************************************
     * read up spinor fields
     **********************************************************/
    for(mu=0; mu<5; mu++) {
      for(ia=0; ia<12; ia++) {
        get_filename(filename, mu, ia, 1);
        exitstatus = read_lime_spinor(g_spinor_field[0], filename, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[cvc_exact3_xspace] Error from read_lime_spinor, status was %d\n", exitstatus);
          EXIT(111);
        }
        // xchange_field(g_spinor_field[mu*12+ia]);
        //
        spinor_field_lexic2eo (g_spinor_field[0], eo_spinor_field[12*mu+ia], eo_spinor_field[60 + 12*mu+ia]);

      }  /* of loop on ia */
    }    /* of loop on mu */

    /**********************************************************
     * read dn spinor fields
     **********************************************************/
    for(mu=0; mu<5; mu++) {
      for(ia=0; ia<12; ia++) {

        get_filename(filename, mu, ia, -1);
        exitstatus = read_lime_spinor(g_spinor_field[0], filename, 0);
/*
        get_filename(filename, mu, ia, 1);
        exitstatus = read_lime_spinor(g_spinor_field[60+mu*12+ia], filename, 1);
*/
        if(exitstatus != 0) {
          fprintf(stderr, "[cvc_exact3_xspace] Error from read_lime_spinor, status was %d\n", exitstatus);
          EXIT(112);
        }
        // xchange_field(g_spinor_field[60+mu*12+ia]);
        
        spinor_field_lexic2eo (g_spinor_field[0], eo_spinor_field[120 + 12*mu+ia], eo_spinor_field[180 + 12*mu+ia]);

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
      fprintf(stderr, "[cvc_exact3_xspace] Error, confused about number of operators, expected 2 operators (up-type, dn-tpye)\n");
      EXIT(6);
    }
 
    for(mu=0; mu<5; mu++)
    {
  
      /*shifted, global source coordinates */
      g_shifted_source_coords[0] = gsx[0];
      g_shifted_source_coords[1] = gsx[1];
      g_shifted_source_coords[2] = gsx[2];
      g_shifted_source_coords[3] = gsx[3];
  
      if(mu < 4) g_shifted_source_coords[mu] = ( g_shifted_source_coords[mu] + 1 ) % LLBase[mu];

      exitstatus = get_point_source_info (g_shifted_source_coords, shifted_source_coords, &shifted_source_proc_id );

      for(ia=0; ia<12; ia++) {

        memset(source, 0, _GSI(VOLUME)*sizeof(double));
        if(shifted_source_proc_id == g_cart_id ) {
          source[ _GSI( g_ipt[shifted_source_coords[0]][shifted_source_coords[1]][shifted_source_coords[2]][shifted_source_coords[3]] ) + 2*ia ] = 1.;
        }
        xchange_field(source);

        for ( op_id = 0; op_id < 2; op_id++ ) {
          if(g_cart_id == 0) fprintf(stdout, "# [cvc_exact3_xspace] inverting for operator %d for spin-color component (%d, %d)\n", op_id, ia/3, ia%3);
          exitstatus = _TMLQCD_INVERT(propagator, source, op_id, g_write_propagator);
          if(exitstatus != 0) {
            fprintf(stderr, "[cvc_exact3_xspace] Error from solver, status was %d\n", exitstatus);
            EXIT(7);
          }
          // xchange_field(propagator);
  
          spinor_field_lexic2eo (propagator, eo_spinor_field[120*op_id + 12*mu+ia], eo_spinor_field[120*op_id + 60 + 12*mu+ia]);

        }  /* end of loop on op_id */

      }  /* end of loop on spin-color component */

    }  /* end of loop on mu shift direction */

  }  /* end of if not read propagator */


#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] reading / invert in %e seconds\n", retime-ratime);


  /**********************************************************
   **********************************************************
   **
   ** contractions
   **
   **********************************************************
   **********************************************************/  

  /**********************************************************
   * initialize and distribute Usource
   **********************************************************/
  init_contract_cvc_tensor_usource(g_gauge_field, gsx, co_phase_up);

  memset(conn_e, 0, 32*Vhalf*sizeof(double));
  memset(conn_o, 0, 32*Vhalf*sizeof(double));
  memset(contact_term, 0, 8*sizeof(double));

  ratime = _GET_TIME;

  /**********************************************************
   * propagator lists
   **********************************************************/
  for( mu=0; mu<5; mu++ ) {
    for( i=0; i<12; i++ ) {
      /* up-type - even */
      tprop_list_e [mu*12 + i] = eo_spinor_field[      mu*12 + i];
      /* up-type - odd */
      tprop_list_o [mu*12 + i] = eo_spinor_field[ 60 + mu*12 + i];
      /* dn-type - even */
      sprop_list_e [mu*12 + i] = eo_spinor_field[120 + mu*12 + i];
      /* dn-type - odd */
      sprop_list_o [mu*12 + i] = eo_spinor_field[180 + mu*12 + i];
    }
  }

  /**********************************************************
   * contraction of cvc tensor
   **********************************************************/
  contract_cvc_tensor_eo(conn_e, conn_o, contact_term, sprop_list_e, sprop_list_o, tprop_list_e, tprop_list_o, gauge_field_with_phase );

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] time for contract_cvc_tensor_eo ions in %e seconds\n", retime-ratime);


  /**********************************************************
   * subtract contact term
   **********************************************************/
  if( source_proc_id == g_cart_id ) {
    fprintf(stdout, "# [cvc_exact3_xspace] process %d subtracting contact term\n", g_cart_id);
    ix = g_ipt[sx[0]][sx[1]][sx[2]][sx[3]];
    ixeo = g_lexic2eosub[ix];
    if ( g_iseven[ix] ) {
      for(mu=0; mu<4; mu++) {
        conn_e[_GWI(5*mu,ixeo,Vhalf)    ] -= contact_term[2*mu  ];
        conn_e[_GWI(5*mu,ixeo,Vhalf) + 1] -= contact_term[2*mu+1];
      }
    } else {
      for(mu=0; mu<4; mu++) {
        conn_o[_GWI(5*mu,ixeo,Vhalf)    ] -= contact_term[2*mu  ];
        conn_o[_GWI(5*mu,ixeo,Vhalf) + 1] -= contact_term[2*mu+1];
      }
    }
  }

  /* save results as lime / lemon file */
  if( write_binary ) {
    ratime = _GET_TIME;
    if(strcmp(g_outfile_prefix, "NA") == 0) {
      sprintf(filename, "cvc3_v_x.%.4d", Nconf);
    } else {
      sprintf(filename, "%s/cvc3_v_x.%.4d", g_outfile_prefix, Nconf);
    }
    double *conn_buffer = (double*)malloc(2*VOLUME*sizeof(double));
    if(conn_buffer == NULL) {
      EXIT(14);
    }
    for(mu=0; mu<16; mu++) {
      complex_field_eo2lexic (conn_buffer, conn_e+2*mu*Vhalf, conn_o+2*mu*Vhalf );
      sprintf(contype, "<comment>\n  cvc - cvc in position space\n</comment>\n<component>\n  %2d-%2d\n</component>\n", mu/4, mu%4);
      write_lime_contraction(conn_buffer, filename, 64, 1, contype, Nconf, mu>0);
    }
    retime = _GET_TIME;
    if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] saved position space results in %e seconds\n", retime-ratime);
  }
#if 0
#endif  /* of if 0 */

  /* save results in plain text */
  if(write_ascii) {
    sprintf(filename, "cvc3_v_x.%.4d.ascii.%.2d", Nconf, g_cart_id);
    ofs = fopen(filename, "w");
    if( ofs == NULL ) {
      fprintf(stderr, "# [cvc_exact3_xspace] Error from fopen\n");
      EXIT(116);
    }
    if( g_cart_id == 0 ) fprintf(ofs, "w <- array(dim=c(%d, %d, %d, %d, %d, %d))\n", 4,4,T_global,LX_global,LY_global,LZ_global);
    for(x0=0; x0<T;  x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix=g_ipt[x0][x1][x2][x3];
      unsigned int ixeo = g_lexic2eosub[ix];
      double *conn = g_iseven[ix] ? conn_e : conn_o;
      for(mu=0; mu<4; mu++) {
      for(nu=0; nu<4; nu++) {
        imunu = 4*mu + nu;
        fprintf(ofs, "w[%d, %d, %d, %d, %d, %d] <- %25.16e + %25.16e*1.i\n", mu+1, nu+1, 
            x0+g_proc_coords[0]*T+1, x1+g_proc_coords[1]*LX+1, 
            x2+g_proc_coords[2]*LY+1, x3+g_proc_coords[3]*LZ+1,
            conn[_GWI(imunu,ixeo,Vhalf)], conn[_GWI(imunu,ixeo,Vhalf)+1]);
      }}
    }}}}
    fclose(ofs);
  }  /* end of if write ascii */


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
#endif  /* of if 0 */
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
      // memcpy(conn_buffer+2*mu*VOLUMEplusRAND, conn+2*mu*VOLUME, 2*VOLUME*sizeof(double));
      complex_field_eo2lexic (conn_buffer+2*mu*VOLUMEplusRAND, conn_e+2*mu*Vhalf, conn_o+2*mu*Vhalf );
      xchange_contraction(conn_buffer+2*mu*VOLUMEplusRAND, 2);
      /* if(g_cart_id == 0) { fprintf(stdout, "# [cvc_exact3_xspace] xchanged for mu = %d\n", mu); fflush(stdout);} */
    }
/*
#ifdef HAVE_MPI
#else
    double *conn_buffer = conn;
#endif
*/
    if( g_cart_id == 0 ) fprintf(stdout, "\n# [cvc_exact3_xspace] checking Ward identity in position space\n");
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
        fprintf(stderr, "[cvc_exact3_xspace] Error from MPI_Allreduce\n");
        EXIT(12);
      }
#endif
      if(g_cart_id == 0) fprintf(stdout, "# [cvc_exact3_xspace] WI nu = %d norm = %25.16e\n", nu, sqrt(norm));
    }
#ifdef HAVE_MPI
    free(conn_buffer);
#endif
  }  /* end of if check_position_space_WI */




  /***************************************************************************
   * momentum projections
   ***************************************************************************/

  double *conn_buffer = (double*)malloc(32 * VOLUME * sizeof(double));
  if(conn_buffer == NULL) {
    EXIT(14);
  }

  for(mu=0; mu<16; mu++) {
    complex_field_eo2lexic (conn_buffer+2 * mu * VOLUME, conn_e+2*mu*Vhalf, conn_o+2*mu*Vhalf );
  }

  double ***cvc_tp = NULL;
  exitstatus = init_3level_buffer(&cvc_tp, g_sink_momentum_number, 16, 2*T);
  if(exitstatus != 0) {
    fprintf(stderr, "[cvc_exact3_xspace] Error from init_3level_buffer, status was %d\n", exitstatus);
    EXIT(26);
  }

  ratime = _GET_TIME;
  exitstatus = momentum_projection (conn_buffer, cvc_tp[0][0], T*16, g_sink_momentum_number, g_sink_momentum_list);
  if(exitstatus != 0) {
    fprintf(stderr, "[cvc_exact3_xspace] Error from momentum_projection, status was %d\n", exitstatus);
    EXIT(26);
  }
  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [cvc_exact3_xspace] time for momentum projection = %e seconds\n", retime-ratime);

  for(mu=0; mu < 16; mu++) {
    sprintf(filename, "cvc3_v_p.%.4d.mu%.2dnu%.2d.ascii.%.2d", Nconf, mu/4, mu%4, g_cart_id);
    ofs = fopen(filename, "w");
    if( ofs == NULL ) {
      fprintf(stderr, "[cvc_exact3_xspace] Error from fopen\n");
      EXIT(12);
    }
    if(g_cart_id == 0) fprintf(ofs, "v <- array(dim=c(%d,%d))\n", g_sink_momentum_number, T_global);
    for(i = 0; i < g_sink_momentum_number; i++ ) {
      fprintf(ofs, "# %3d %3d %3d\n", g_sink_momentum_list[i][0], g_sink_momentum_list[i][1], g_sink_momentum_list[i][2]);
  
      for (x0 = 0; x0 < T; x0++) {
        fprintf(ofs, "v[%d, %d] <- %25.16e + %25.16e*1.i\n", i+1,
            x0+g_proc_coords[0]*T+1, cvc_tp[i][mu][2*x0], cvc_tp[i][mu][2*x0+1]);
      }
    }
    fclose(ofs);
  }
  free( conn_buffer );
  fini_3level_buffer(&cvc_tp);
#if 0
#endif /* of if 0 */


  /****************************************
   * free the allocated memory, finalize
   ****************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( g_spinor_field[0] );
  free( g_spinor_field );
  free( eo_spinor_field[0] );
  free( eo_spinor_field );
  free( conn_e );
  free( gauge_field_with_phase );
  free_geometry();
#if 0
#endif  /* end of if 0 */

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  free(status);
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_propagator();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [cvc_exact3_xspace] %s# [cvc_exact3_xspace] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [cvc_exact3_xspace] %s# [cvc_exact3_xspace] end of run\n", ctime(&g_the_time));
  }
  return(0);
}
