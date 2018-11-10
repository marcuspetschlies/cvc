/****************************************************
 * p2gg_xspace.c
 *
 * Fr 23. Okt 12:56:08 CEST 2015
 *
 * - originally copied from cvc_exact2_xspace.c
 *
 * PURPOSE:
 * - contractions for P -> gamma gamma in position space
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
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  EXIT(0);
}


int main(int argc, char **argv) {
  
  /*
   * sign for g5 Gamma^\dagger g5
   *                                                0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   * */
  const int sequential_source_gamma_id_sign[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };

  char outfile_name[] = "p2gg";

  const double PI2 =  2. * M_PI;

  int c, i, j, mu, nu, ir, is, ia, ib, imunu;
  int op_id = 0, iflavor;
  int filename_set = 0;
  int source_location, have_source_flag = 0, have_shifted_source_flag = 0;
  int x0, x1, x2, x3, ix;
  int gsx0, gsx1, gsx2, gsx3;
  int sx0, sx1, sx2, sx3;
  int isimag[4];
  int gperm[5][4], gperm2[4][4];
  int check_position_space_WI=0;
  int nthreads=-1, threadid=-1;
  int exitstatus;
  int write_ascii=0;
  int source_proc_coords[4], source_proc_id = -1;
  int shifted_source_coords[4], shifted_source_proc_coords[4];
  int seq_source_momentum[3], iseq_source_momentum;
  int ud_one_file = 0;
  int sequential_source_gamma_id, isequential_source_gamma_id, isequential_source_timeslice;
  double gperm_sign[5][4], gperm2_sign[4][4];
  double *conn  = NULL,  **conn_buffer = NULL;
  double contact_term[8];
  int verbose = 0;
  char filename[100], contype[1200], sequential_filename_prefix[200];
  char outfile_tag[200];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
/*  double *gauge_trafo = NULL; */
  double *phi=NULL, *chi=NULL, *source=NULL, *propagator=NULL;
  complex w, w1;
  double Usourcebuff[72], *Usource[4];
  double phase=0.;
/*  int LLBase[4]; */
  FILE *ofs;

#ifdef HAVE_MPI
  int *status;
#endif

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

  while ((c = getopt(argc, argv, "uwah?vf:")) != -1) {
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
      fprintf(stdout, "\n# [p2gg_xspace] will check Ward identity in position space\n");
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "\n# [p2gg_xspace] will write data in ASCII format too\n");
      break;
    case 'u':
      ud_one_file = 1;
      fprintf(stdout, "\n# [p2gg_xspace] will read u/d propagator from same file\n");
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
  if(filename_set==0) strcpy(filename, "p2gg.input");
  fprintf(stdout, "# [p2gg_xspace] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_xspace] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
  if(exitstatus != 0) {
    EXIT(14);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(15);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    EXIT(16);
  }
#endif

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(32);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#endif

  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stderr, "[p2gg_xspace] T and L's must be set\n");
    usage();
  }


#ifdef HAVE_MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    EXIT(1);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg_xspace] Error from init_geometry\n");
    EXIT(2);
  }

  geometry();
/*
  LLBase[0] = T_global;
  LLBase[1] = LX_global;
  LLBase[2] = LY_global;
  LLBase[3] = LZ_global;
*/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg_xspace] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg_xspace] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg_xspace] Error, &g_gauge_field is NULL\n");
    EXIT(5);
  }
#endif

#ifdef HAVE_MPI
  xchange_gauge();
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace] measured plaquette value: %25.16e\n", plaq);

  /* allocate memory for the spinor fields */
  no_fields = 180;
#ifdef HAVE_TMLQCD_LIBWRAPPER
  no_fields++;
#endif

  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  source = g_spinor_field[no_fields-1];
#endif

  /* allocate memory for the contractions */
  conn_buffer = (double**)malloc(2 * sizeof(double*));
  if(conn_buffer == NULL) {
    fprintf(stderr, "[p2gg_xspace] could not allocate memory for conn_buffer. fields\n");
    EXIT(62);
  }

  conn_buffer[0] = (double*)malloc(2*32*(VOLUME+RAND)*sizeof(double));
  if( conn_buffer[0] == NULL ) {
    fprintf(stderr, "[p2gg_xspace] could not allocate memory for contr. fields\n");
    EXIT(6);
  }
  memset(conn_buffer[0], 0, 2*32*VOLUME*sizeof(double));
  conn_buffer[1] = conn_buffer[0] + 32*(VOLUME+RAND);


  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  gsx0 = g_source_location / ( LX_global * LY_global * LZ_global);
  gsx1 = (g_source_location % ( LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  gsx2 = (g_source_location % ( LY_global * LZ_global)) / LZ_global;
  gsx3 = (g_source_location % LZ_global);
  source_proc_coords[0] = gsx0 / T;
  source_proc_coords[1] = gsx1 / LX;
  source_proc_coords[2] = gsx2 / LY;
  source_proc_coords[3] = gsx3 / LZ;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [p2gg_xspace] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx0, gsx1, gsx2, gsx3);
    fprintf(stdout, "# [p2gg_xspace] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  have_source_flag = (int)(g_cart_id == source_proc_id);
  if(have_source_flag==1) {
    fprintf(stdout, "# [p2gg_xspace] process %2d has source location\n", source_proc_id);
  }
  sx0 = gsx0 % T;
  sx1 = gsx1 % LX;
  sx2 = gsx2 % LY;
  sx3 = gsx3 % LZ;
# else
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "[p2gg_xspace] process %2d has source location\n", g_cart_id);
  gsx0 = g_source_location/(LX*LY*LZ)-Tstart;
  gsx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  gsx2 = (g_source_location%(LY*LZ)) / LZ;
  gsx3 = (g_source_location%LZ);

  sx0 = gsx0 - Tstart;
  sx1 = gsx1;
  sx2 = gsx2;
  sx3 = gsx3;
#endif
  Usource[0] = Usourcebuff;
  Usource[1] = Usourcebuff+18;
  Usource[2] = Usourcebuff+36;
  Usource[3] = Usourcebuff+54;
  if(have_source_flag==1) { 
    fprintf(stdout, "# [p2gg_xspace] local source coordinates: (%3d,%3d,%3d,%3d)\n", sx0, sx1, sx2, sx3);
    source_location = g_ipt[sx0][sx1][sx2][sx3];
    _cm_eq_cm_ti_co(Usource[0], &g_gauge_field[_GGI(source_location,0)], &co_phase_up[0]);
    _cm_eq_cm_ti_co(Usource[1], &g_gauge_field[_GGI(source_location,1)], &co_phase_up[1]);
    _cm_eq_cm_ti_co(Usource[2], &g_gauge_field[_GGI(source_location,2)], &co_phase_up[2]);
    _cm_eq_cm_ti_co(Usource[3], &g_gauge_field[_GGI(source_location,3)], &co_phase_up[3]);
  }
#ifdef HAVE_MPI
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  have_source_flag = source_proc_id;
  MPI_Bcast(Usourcebuff, 72, MPI_DOUBLE, have_source_flag, g_cart_grid);
#  else
  MPI_Gather(&have_source_flag, 1, MPI_INT, status, 1, MPI_INT, 0, g_cart_grid);
  if(g_cart_id==0) {
    for(mu=0; mu<g_nproc; mu++) fprintf(stdout, "# [p2gg_xspace] status[%1d]=%d\n", mu,status[mu]);
  }
  if(g_cart_id==0) {
    for(have_source_flag=0; status[have_source_flag]!=1; have_source_flag++);
    fprintf(stdout, "# [p2gg_xspace] have_source_flag= %d\n", have_source_flag);
  }
  MPI_Bcast(&have_source_flag, 1, MPI_INT, 0, g_cart_grid);
  MPI_Bcast(Usourcebuff, 72, MPI_DOUBLE, have_source_flag, g_cart_grid);
#  endif
  /* fprintf(stdout, "# [p2gg_xspace] proc%.4d have_source_flag = %d\n", g_cart_id, have_source_flag); */
#else
  /* HAVE_MPI not defined */
  have_source_flag = 0;
#endif

#ifdef HAVE_MPI
      ratime = MPI_Wtime();
#else
      ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  /***********************************************************
   *  initialize the Gamma matrices
   ***********************************************************/
  // gamma_5:
  gperm[4][0] = gamma_permutation[5][ 0] / 6;
  gperm[4][1] = gamma_permutation[5][ 6] / 6;
  gperm[4][2] = gamma_permutation[5][12] / 6;
  gperm[4][3] = gamma_permutation[5][18] / 6;
  gperm_sign[4][0] = gamma_sign[5][ 0];
  gperm_sign[4][1] = gamma_sign[5][ 6];
  gperm_sign[4][2] = gamma_sign[5][12];
  gperm_sign[4][3] = gamma_sign[5][18];
  // gamma_nu gamma_5
  for(nu=0;nu<4;nu++) {
    // permutation
    gperm[nu][0] = gamma_permutation[6+nu][ 0] / 6;
    gperm[nu][1] = gamma_permutation[6+nu][ 6] / 6;
    gperm[nu][2] = gamma_permutation[6+nu][12] / 6;
    gperm[nu][3] = gamma_permutation[6+nu][18] / 6;
    // is imaginary ?
    isimag[nu] = gamma_permutation[6+nu][0] % 2;
    // (overall) sign
    gperm_sign[nu][0] = gamma_sign[6+nu][ 0];
    gperm_sign[nu][1] = gamma_sign[6+nu][ 6];
    gperm_sign[nu][2] = gamma_sign[6+nu][12];
    gperm_sign[nu][3] = gamma_sign[6+nu][18];
    // write to stdout
    if(g_cart_id == 0) {
      fprintf(stdout, "# gamma_%d5 = (%f %d, %f %d, %f %d, %f %d)\n", nu,
          gperm_sign[nu][0], gperm[nu][0], gperm_sign[nu][1], gperm[nu][1], 
          gperm_sign[nu][2], gperm[nu][2], gperm_sign[nu][3], gperm[nu][3]);
    }
  }
  // gamma_nu
  for(nu=0;nu<4;nu++) {
    // permutation
    gperm2[nu][0] = gamma_permutation[nu][ 0] / 6;
    gperm2[nu][1] = gamma_permutation[nu][ 6] / 6;
    gperm2[nu][2] = gamma_permutation[nu][12] / 6;
    gperm2[nu][3] = gamma_permutation[nu][18] / 6;
    // (overall) sign
    gperm2_sign[nu][0] = gamma_sign[nu][ 0];
    gperm2_sign[nu][1] = gamma_sign[nu][ 6];
    gperm2_sign[nu][2] = gamma_sign[nu][12];
    gperm2_sign[nu][3] = gamma_sign[nu][18];
    // write to stdout
    if(g_cart_id == 0) {
    	fprintf(stdout, "# gamma_%d = (%f %d, %f %d, %f %d, %f %d)\n", nu,
        	gperm2_sign[nu][0], gperm2[nu][0], gperm2_sign[nu][1], gperm2[nu][1], 
        	gperm2_sign[nu][2], gperm2[nu][2], gperm2_sign[nu][3], gperm2[nu][3]);
    }
  }

  if ( g_read_propagator ) {
    /**********************************************************
     * read up spinor fields
     **********************************************************/
    for(mu=0; mu<5; mu++) {
      for(ia=0; ia<12; ia++) {

        get_filename(filename, mu, ia,  1 );
        exitstatus = read_lime_spinor(g_spinor_field[mu*12+ia], filename, 0);
      
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg_xspace] Error from read_lime_spinor, status was %d\n", exitstatus);
          EXIT(7);
        }
       xchange_field(g_spinor_field[mu*12+ia]);
       }  /* of loop on ia */
    }    /* of loop on mu */

    /**********************************************************
     * read dn spinor fields
     **********************************************************/
    for(mu=0; mu<5; mu++) {
      for(ia=0; ia<12; ia++) {

        if(ud_one_file) {
          get_filename(filename, mu, ia,  1); 
          exitstatus = read_lime_spinor(g_spinor_field[60+mu*12+ia], filename, 1);
        } else {
          get_filename(filename, mu, ia, -1);
          exitstatus = read_lime_spinor(g_spinor_field[60+mu*12+ia], filename, 0);
        }

        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg_xspace] Error from read_lime_spinor, status was %d\n", exitstatus);
          EXIT(7);
        }
        xchange_field(g_spinor_field[60+mu*12+ia]);
      }  /* of loop on ia */
    }    /* of loop on mu */
  
  } else {
    /**********************************************************
     * invert for up and dn propagator
     **********************************************************/
    for(mu=0; mu<5; mu++) {
      int shifted_source_location[4] = { gsx0, gsx1, gsx2, gsx3 };
      switch (mu) {
        case 0: shifted_source_location[0] = ( shifted_source_location[0] + 1 ) % T_global; break;;
        case 1: shifted_source_location[1] = ( shifted_source_location[1] + 1 ) % LX_global; break;;
        case 2: shifted_source_location[2] = ( shifted_source_location[2] + 1 ) % LY_global; break;;
        case 3: shifted_source_location[3] = ( shifted_source_location[3] + 1 ) % LZ_global; break;;
      }

      if ( g_cart_id == 0 ) {
        fprintf(stdout, "# [] shifted source coords = %3d %3d %3d %3d\n",
            shifted_source_location[0], shifted_source_location[1], shifted_source_location[2], shifted_source_location[3]);
      }

      /* loop on spin color */
      for(ia=0; ia<12; ia++) {
        memset( g_spinor_field[   mu*12+ia], 0, _GSI(VOLUME) * sizeof(double) );
        memset( g_spinor_field[60+mu*12+ia], 0, _GSI(VOLUME) * sizeof(double) );

        int have_source = ( shifted_source_location[0] / T  == g_proc_coords[0] &&
                            shifted_source_location[1] / LX == g_proc_coords[1] &&
                            shifted_source_location[1] / LY == g_proc_coords[1] &&
                            shifted_source_location[1] / LZ == g_proc_coords[1] );

        memset ( source, 0, _GSI(VOLUME) * sizeof(double) );
        propagator = g_spinor_field[mu*12+ia];
        if ( have_source ) {
          fprintf(stdout, "# [p2gg_xspace] process %d has the shifted source\n", g_cart_id);
          source[_GSI(g_ipt[shifted_source_location[0] % T][shifted_source_location[1] % LX][shifted_source_location[2] % LY][shifted_source_location[3] % LZ]) + 2*ia] = 1.;
        } 

        /* up propagator */
        exitstatus = tmLQCD_invert(propagator, source, 0, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg_xspace] Error from tmLQCD_invert, status was %d\n", exitstatus);
          EXIT(12);
        }

        propagator = g_spinor_field[60+mu*12+ia];
        /* dn propagator */
        exitstatus = tmLQCD_invert(propagator, source, 1, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[p2gg_xspace] Error from tmLQCD_invert, status was %d\n", exitstatus);
          EXIT(12);
        }
      }  /* end of loop on spini-color */
    }  /* end of loop on shifts mu */

  }  /* end of if read propagator else */

  /***************************************************************************
   * loop on sequential source time slices
   ***************************************************************************/
  for(isequential_source_timeslice=0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++) {

    g_sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [] using sequential source timeslice %d\n", g_sequential_source_timeslice);
    }

  /***************************************************************************
   * loop on sequential source gamma matrices
   ***************************************************************************/

  for(iseq_source_momentum=0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) {

    g_seq_source_momentum[0] = g_seq_source_momentum_list[iseq_source_momentum][0];
    g_seq_source_momentum[1] = g_seq_source_momentum_list[iseq_source_momentum][1];
    g_seq_source_momentum[2] = g_seq_source_momentum_list[iseq_source_momentum][2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [p2gg_xspace] using sequential source momentum = (%d, %d, %d)\n", g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);
    }

  /***************************************************************************
   * loop on sequential source gamma matrices
   ***************************************************************************/
  for(isequential_source_gamma_id=0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++) {

    sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [p2gg_xspace] using sequential source gamma id = %d\n", sequential_source_gamma_id);
    }


    // initialize the contact term
    memset(contact_term, 0, 8*sizeof(double));

  for(iflavor=0; iflavor<2; iflavor++) {

#ifdef HAVE_MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

    op_id                 = iflavor;
    g_propagator_position = iflavor;
    conn = conn_buffer[iflavor];

    /* reset conn to zero */
    memset(conn, 0, 32*(VOLUME+RAND)*sizeof(double));

    /* flavor-dependent sequential source momentum */
    seq_source_momentum[0] = (1 - 2*iflavor) * g_seq_source_momentum[0];
    seq_source_momentum[1] = (1 - 2*iflavor) * g_seq_source_momentum[1];
    seq_source_momentum[2] = (1 - 2*iflavor) * g_seq_source_momentum[2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [] using flavor-dependent sequential source momentum (%d, %d, %d)\n", 
          seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2]);
    }

    if(iflavor == 0) {
      strcpy(sequential_filename_prefix, g_sequential_filename_prefix);
    } else if(iflavor == 1) {
      if(ud_one_file) {
        strcpy(sequential_filename_prefix, g_sequential_filename_prefix);
      } else {
        strcpy(sequential_filename_prefix, g_sequential_filename_prefix2);
      }
    }
  
    sprintf( outfile_tag, "t%.2dx%.2dy%.2dz%.2d.tseq%.2d.g%.2d.px%.2dpy%.2dpz%.2d", 
      gsx0, gsx1, gsx2, gsx3, g_sequential_source_timeslice,
      sequential_source_gamma_id, g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);
    if(g_cart_id == 0) {
      fprintf(stdout, "# [p2gg_xspace] output file flag set to %s\n", outfile_tag);
    }

#ifndef HAVE_TMLQCD_LIBWRAPPER
    if(g_cart_id == 0) {
      fprintf(stderr, "[p2gg_xspace] Error, need tmLQCD libwrapper for inversion\n");
      EXIT(8);
    }
#endif
  
    /***********************************************************
     * invert using tmLQCD invert
     ***********************************************************/
#ifdef HAVE_TMLQCD_LIBWRAPPER
    if(g_tmLQCD_lat.no_operators > 2) {
      fprintf(stderr, "[p2gg_xspace] Error, confused about number of operators, expected 1 operator (up-type)\n");
      EXIT(9);
    }
#endif
  
    if(g_cart_id == 0) fprintf(stdout, "# [p2gg_xspace] using op_id = %d\n", op_id);
  
      /* the global sequential source timeslice */
      shifted_source_coords[0] = ( gsx0 + g_sequential_source_timeslice + T_global ) % T_global;
  
      /* TEST */
  /*
      shifted_source_coords[1] = ( gsx1 + g_sequential_source_location_x + LX_global ) % LX_global;
      shifted_source_coords[2] = ( gsx2 + g_sequential_source_location_y + LY_global ) % LY_global;
      shifted_source_coords[3] = ( gsx3 + g_sequential_source_location_z + LZ_global ) % LZ_global;
  */  
  
      shifted_source_proc_coords[0] = shifted_source_coords[0] / T;
  
      /* TEST */
  /*
      shifted_source_proc_coords[1] = shifted_source_coords[1] / LX;
      shifted_source_proc_coords[2] = shifted_source_coords[2] / LY;
      shifted_source_proc_coords[3] = shifted_source_coords[3] / LZ;
  */  
      have_shifted_source_flag = ( g_proc_coords[0] == shifted_source_proc_coords[0] );
  /*
      &&
          g_proc_coords[1] == shifted_source_proc_coords[1]  &&
          g_proc_coords[2] == shifted_source_proc_coords[2]  &&
          g_proc_coords[3] == shifted_source_proc_coords[3] );
  */
      if(have_shifted_source_flag) {
        fprintf(stdout, "# [p2gg_xspace] process %4d has global sequential source timeslice %d -> %d\n", g_cart_id, gsx0, shifted_source_coords[0] );
  /*
        fprintf(stdout, "# [p2gg_xspace] process %4d has global sequential source timeslice (%d,%d, %d, %d) -> (%d, %d, %d, %d)\n", g_cart_id, 
            gsx0, gsx1, gsx2, gsx3, shifted_source_coords[0], shifted_source_coords[1], shifted_source_coords[2], shifted_source_coords[3]);
  */
      }
      
      shifted_source_coords[0] = shifted_source_coords[0] % T;  /* local sequential source timeslice */
  /*
      shifted_source_coords[1] = shifted_source_coords[1] % LX;
      shifted_source_coords[2] = shifted_source_coords[2] % LY;
      shifted_source_coords[3] = shifted_source_coords[3] % LZ;
  */
      if(have_shifted_source_flag) {
        fprintf(stdout, "# [p2gg_xspace] process %4d has local sequential source timeslice %d\n", g_cart_id, shifted_source_coords[0]);
      }
  
      for(mu=0; mu<5; mu++) {
        for(ia=0; ia<12; ia++) {
  
          if(g_read_sequential_propagator) {
            /**********************************************************
             * read up-type sequential propagator
             **********************************************************/
            sprintf(filename, "%s.%.4d.%.2d.px%.2dpy%.2dpz%.2d.%.2d.inverted", sequential_filename_prefix, Nconf, mu,
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], ia);
  
            if(ud_one_file) {
              exitstatus = read_lime_spinor(g_spinor_field[120 + mu*12+ia], filename, g_propagator_position);
            } else {
              exitstatus = read_lime_spinor(g_spinor_field[120 + mu*12+ia], filename, 0);
            }
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg_xspace] Error from read_lime_spinor, status was %d\n", exitstatus);
              EXIT(10);
            }
            xchange_field(g_spinor_field[120 + mu*12+ia]);
  
          } else {
            /**********************************************************
             * read up spinor fields
             **********************************************************/
/*            
            if(ud_one_file) {
              get_filename(filename, mu, ia, 1 );
              exitstatus = read_lime_spinor(g_spinor_field[mu*12+ia], filename, g_propagator_position);
            } else {
              get_filename(filename, mu, ia, (1-2*g_propagator_position) );
              exitstatus = read_lime_spinor(g_spinor_field[mu*12+ia], filename, 0);
            }
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg_xspace] Error from read_lime_spinor, status was %d\n", exitstatus);
              EXIT(11);
            }
*/

            propagator = g_spinor_field[120 + 12 * mu + ia];
            memcpy(propagator, g_spinor_field[g_propagator_position*60 + mu*12+ia], 24*VOLUME*sizeof(double) );
            xchange_field(propagator);
  
            /**********************************************************
             * prepare sequential source
             **********************************************************/
            if(g_cart_id == 0) fprintf(stdout, "# [p2gg_xspace] preparing sequential source for spin-color component (%d, %d)\n", ia/3, ia%3);
  
            memset(source, 0, 24*(VOLUME+RAND)*sizeof(double));
  
            if(have_shifted_source_flag) {
  
              x0 = shifted_source_coords[0];
              for(x1=0;x1<LX;x1++) {
              for(x2=0;x2<LY;x2++) {
              for(x3=0;x3<LZ;x3++) {
                ix = g_ipt[x0][x1][x2][x3];
                phase = PI2 * ( 
                    ( x1 + g_proc_coords[1]*LX - gsx1 ) * seq_source_momentum[0] / (double)LX_global
                  + ( x2 + g_proc_coords[2]*LY - gsx2 ) * seq_source_momentum[1] / (double)LY_global
                  + ( x3 + g_proc_coords[3]*LZ - gsx3 ) * seq_source_momentum[2] / (double)LZ_global );
                w.re = cos(phase);
                w.im = sin(phase);
                _fv_eq_gamma_ti_fv(spinor1, sequential_source_gamma_id, propagator + _GSI(ix));
                _fv_eq_fv_ti_co(source + _GSI(ix), spinor1, &w);
              }}}
  
  
              /* TEST */
  /*
              ix = g_ipt[shifted_source_coords[0]][shifted_source_coords[1]][shifted_source_coords[2]][shifted_source_coords[3]];
              _fv_eq_fv(source + _GSI(ix), propagator + _GSI(ix));
  */
  
            }  /* end of if have_shifted_source_flag */
  
            xchange_field(source);
  
  
            /* TEST */
  /*
            sprintf(filename, "sequential_source.%.2d.%.2d.%.4d", mu, ia, g_cart_id);
            ofs = fopen(filename, "w");
            for(x0=0;x0<T;x0++) {
            for(x1=0;x1<LX;x1++) {
            for(x2=0;x2<LY;x2++) {
            for(x3=0;x3<LZ;x3++) {
              ix = g_ipt[x0][x1][x2][x3];
              fprintf(ofs, "# t=%2d x=%2d y=%2d z=%2d\n", x0+g_proc_coords[0]*T, x1+g_proc_coords[1]*LX, x2+g_proc_coords[2]*LY, x3+g_proc_coords[3]*LZ);
              for(i=0; i<12; i++) {
                fprintf(ofs, "%3d%25.16e%25.16e\n", i, source[_GSI(ix)+2*i], source[_GSI(ix)+2*i+1]);
              }
            }}}}
            fclose(ofs);
  */
  
            if(g_write_sequential_source) {
              sprintf(filename, "%s.%.4d.%.2d.px%.2dpy%.2dpz%.2d.%.2d", sequential_filename_prefix, Nconf, mu,
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], ia);
  
              if(g_cart_id == 0) {
                fprintf(stdout, "# [p2gg_xspace] writing sequential source to file %s\n", filename);
              }
  
              if(ud_one_file) {
                exitstatus = write_propagator(source, filename, g_propagator_position, g_propagator_precision);
              } else {
                exitstatus = write_propagator(source, filename, g_propagator_position, 0);
              }
              if( exitstatus != 0) {
                fprintf(stderr, "[p2gg_xspace] Error from write_propagator, status was %d\n", exitstatus);
                EXIT(24);
              }
            }
  
  
            /**********************************************************
             * invert
             **********************************************************/
#ifdef HAVE_TMLQCD_LIBWRAPPER
            if(g_cart_id == 0) fprintf(stdout, "# [p2gg_xspace] inverting for spin-color component (%d, %d)\n", ia/3, ia%3);
            exitstatus = tmLQCD_invert(propagator, source, op_id, 0);
  
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg_xspace] Error from tmLQCD_invert, status was %d\n", exitstatus);
              EXIT(12);
            }
  
            if(g_write_sequential_propagator) {
              sprintf(filename, "%s.%.4d.%.2d.px%.2dpy%.2dpz%.2d.%.2d.inverted", sequential_filename_prefix, Nconf, mu,
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], ia);
  
              if(ud_one_file) {
                exitstatus = write_propagator(propagator, filename, g_propagator_position, g_propagator_precision);
              } else {
                exitstatus = write_propagator(propagator, filename, g_propagator_position, 0);
              }
              if( exitstatus != 0 ) {
                fprintf(stderr, "[p2gg_xspace] Error from write_propagator, could not write file %s\n", filename);
                EXIT(14);
              }
  
  
            }
#else
            if(g_cart_id == 0) fprintf(stderr, "[p2gg_xspace] Error, need inverter if seq. prop. is not read\n");
            EXIT(63);

#endif
            xchange_field(propagator);
  
          }  /* end of else of if read sequential propagator */
  
        }  /* end of loop on spin-color component */
      }  /* end of loop on mu shift direction */
  
  
#ifdef HAVE_MPI
    retime = MPI_Wtime();
#else
    retime = (double)clock() / CLOCKS_PER_SEC;
#endif
    if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace] reading / invert in %e seconds\n", retime-ratime);
  
    /**********************************************************
     **********************************************************
     **
     ** contractions
     **
     **********************************************************
     **********************************************************/  
  
#ifdef HAVE_MPI
    ratime = MPI_Wtime();
#else
    ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  
    /**********************************************************
     * first contribution
     **********************************************************/  
  
    
    /* loop on the Lorentz index nu at source */
    for(nu=0; nu<4; nu++) 
    {
  
      for(ir=0; ir<4; ir++) {
  
        for(ia=0; ia<3; ia++) {
          /* phi = g_spinor_field[      4*12 + 3*ir            + ia]; */
          phi = g_spinor_field[120 +     4*12 + 3*ir            + ia];
  
        for(ib=0; ib<3; ib++) {
          /* chi = g_spinor_field[60 + nu*12 + 3*gperm[nu][ir] + ib]; */
          chi = g_spinor_field[(1 - g_propagator_position) * 60 + nu*12 + 3*gperm[nu][ir] + ib];
  
          /* 1) gamma_nu gamma_5 x U */
          for(mu=0; mu<4; mu++) 
          {
  
            imunu = 4*mu+nu;
/* #ifdef HAVE_OPENMP */
/* #pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu) */
/* #endif */
            for(ix=0; ix<VOLUME; ix++) {
/*
#  ifdef HAVE_OPENMP
              threadid = omp_get_thread_num();
              nthreads = omp_get_num_threads();
              fprintf(stdout, "[thread%d] number of threads = %d\n", threadid, nthreads);
#  endif
*/
  
              _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
  
              _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
              _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
  	    _fv_mi_eq_fv(spinor2, spinor1);
  	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
  	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
              _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
              if(!isimag[nu]) {
                conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
                conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
              } else {
                conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
                conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
              }
          
            }  /* of ix */
  
/* #ifdef HAVE_OPENMP */
/* #pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu) */
/* #endif */
            for(ix=0; ix<VOLUME; ix++) {
              _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
  
              _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
              _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
  	    _fv_pl_eq_fv(spinor2, spinor1);
  	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
  	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
              _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
              if(!isimag[nu]) {
                conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
                conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
              } else {
                conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
                conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
              }
  
            }  /* of ix */
  	}    /* of mu */
        }      /* of ib */
        }      /* of ia */
  
        for(ia=0; ia<3; ia++) {
          /* phi = g_spinor_field[      4*12 + 3*ir            + ia]; */
          phi = g_spinor_field[120 +      4*12 + 3*ir            + ia];
  
        for(ib=0; ib<3; ib++) {
          /* chi = g_spinor_field[60 + nu*12 + 3*gperm[ 4][ir] + ib]; */
          chi = g_spinor_field[(1 - g_propagator_position) * 60 + nu*12 + 3*gperm[ 4][ir] + ib];
          
  
          /* -gamma_5 x U */
          for(mu=0; mu<4; mu++) 
          {
  
            imunu = 4*mu+nu;
  
/* #ifdef HAVE_OPENMP */
/* #pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu) */
/* #endif */
            for(ix=0; ix<VOLUME; ix++) {
              _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
  
              _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
              _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
  	    _fv_mi_eq_fv(spinor2, spinor1);
  	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
  	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
              _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
              conn[_GWI(imunu,ix,VOLUME)  ] -= gperm_sign[4][ir] * w1.re;
              conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[4][ir] * w1.im;
  
            }  /* of ix */
  
/* #ifdef HAVE_OPENMP */
/* #pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu) */
/* #endif */
            for(ix=0; ix<VOLUME; ix++) {
              _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
  
              _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
              _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
  	    _fv_pl_eq_fv(spinor2, spinor1);
  	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
  	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
              _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
              conn[_GWI(imunu,ix,VOLUME)  ] -= gperm_sign[4][ir] * w1.re;
              conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[4][ir] * w1.im;
  
            }  /* of ix */
  	}    /* of mu */
        }      /* of ib */
  
        if(iflavor == 0) {
          /* contribution to contact term */
          if(have_source_flag == g_cart_id) {
            _fv_eq_cm_ti_fv(spinor1, Usource[nu], phi+_GSI(g_iup[source_location][nu]));
            _fv_eq_gamma_ti_fv(spinor2, nu, spinor1);
            _fv_mi_eq_fv(spinor2, spinor1);
            contact_term[2*nu  ] += -0.5 * spinor2[2*(3*ir+ia)  ];
            contact_term[2*nu+1] += -0.5 * spinor2[2*(3*ir+ia)+1];
          }
        }

        }  /* of ia */
      }    /* of ir */
  
    }  // of nu
  
/*
    if(have_source_flag == g_cart_id) {
      fprintf(stdout, "# [p2gg_xspace] contact term after 1st part:\n");
      fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 0, contact_term[0], contact_term[1]);
      fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 1, contact_term[2], contact_term[3]);
      fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 2, contact_term[4], contact_term[5]);
      fprintf(stdout, "\t%d\t%25.16e%25.16e\n", 3, contact_term[6], contact_term[7]);
    }
*/

    /**********************************************************
     * second contribution
     **********************************************************/  
  
    /* loop on the Lorentz index nu at source */
    for(nu=0; nu<4; nu++) 
    {
  
      for(ir=0; ir<4; ir++) {
  
        for(ia=0; ia<3; ia++) {
          /* phi = g_spinor_field[     nu*12 + 3*ir            + ia]; */
          phi = g_spinor_field[120 +     nu*12 + 3*ir            + ia];
  
        for(ib=0; ib<3; ib++) {
          /* chi = g_spinor_field[60 +  4*12 + 3*gperm[nu][ir] + ib]; */
          chi = g_spinor_field[(1 - g_propagator_position) * 60 +  4*12 + 3*gperm[nu][ir] + ib];
  
      
          /* 1) gamma_nu gamma_5 x U^dagger */
          for(mu=0; mu<4; mu++)
          {
  
            imunu = 4*mu+nu;
  
/* #ifdef HAVE_OPENMP */
/* #pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu) */
/* #endif */
            for(ix=0; ix<VOLUME; ix++) {
              _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
  
              _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
              _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
  	    _fv_mi_eq_fv(spinor2, spinor1);
  	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
  	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
              _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
              if(!isimag[nu]) {
                conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
                conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
              } else {
                conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
                conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
              }
          
            }  /* of ix */
  
/* #ifdef HAVE_OPENMP */
/* #pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu) */
/* #endif */
            for(ix=0; ix<VOLUME; ix++) {
              _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
  
              _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
              _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
  	    _fv_pl_eq_fv(spinor2, spinor1);
  	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
  	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
              _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
              if(!isimag[nu]) {
                conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
                conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
              } else {
                conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
                conn[_GWI(imunu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
              }
  
            }  /* of ix */
  	}    /* of mu */
  
        } /* of ib */
        } /* of ia */
  
        for(ia=0; ia<3; ia++) {
          /* phi = g_spinor_field[     nu*12 + 3*ir            + ia]; */
          phi = g_spinor_field[120 +     nu*12 + 3*ir            + ia];
  
        for(ib=0; ib<3; ib++) {
          /* chi = g_spinor_field[60 +  4*12 + 3*gperm[ 4][ir] + ib]; */
          chi = g_spinor_field[(1 - g_propagator_position) * 60 +  4*12 + 3*gperm[ 4][ir] + ib];
  
          /* -gamma_5 x U */
          for(mu=0; mu<4; mu++)
          {
  
            imunu = 4*mu+nu;
  
/* #ifdef HAVE_OPENMP */
/* #pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu) */
/* #endif */
            for(ix=0; ix<VOLUME; ix++) {
              _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
  
              _fv_eq_cm_ti_fv(spinor1, U_, phi+_GSI(g_iup[ix][mu]));
              _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
  	    _fv_mi_eq_fv(spinor2, spinor1);
  	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
  	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
              _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
              conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[4][ir] * w1.re;
              conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[4][ir] * w1.im;
          
            }  /* of ix */
  
/* #ifdef HAVE_OPENMP */
/* #pragma omp parallel for private(ix, spinor1, spinor2, U_, w, w1)  shared(imunu, ia, ib, nu, mu) */
/* #endif */
            for(ix=0; ix<VOLUME; ix++) {
              _cm_eq_cm_ti_co(U_, &g_gauge_field[_GGI(ix,mu)], &co_phase_up[mu]);
  
              _fv_eq_cm_dag_ti_fv(spinor1, U_, phi+_GSI(ix));
              _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
  	    _fv_pl_eq_fv(spinor2, spinor1);
  	    _fv_eq_gamma_ti_fv(spinor1, 5, spinor2);
  	    _co_eq_fv_dag_ti_fv(&w, chi+_GSI(g_iup[ix][mu]), spinor1);
              _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
              conn[_GWI(imunu,ix,VOLUME)  ] += gperm_sign[4][ir] * w1.re;
              conn[_GWI(imunu,ix,VOLUME)+1] += gperm_sign[4][ir] * w1.im;
  
            }  /* of ix */
  	}    /* of mu */
        }      /* of ib */
  
        if(iflavor == 0) {
          /* contribution to contact term */
          if(have_source_flag == g_cart_id)  {
            _fv_eq_cm_dag_ti_fv(spinor1, Usource[nu], phi+_GSI(source_location));
            _fv_eq_gamma_ti_fv(spinor2, nu, spinor1);
            _fv_pl_eq_fv(spinor2, spinor1);
            contact_term[2*nu  ] += 0.5 * spinor2[2*(3*ir+ia)  ];
            contact_term[2*nu+1] += 0.5 * spinor2[2*(3*ir+ia)+1];
          }
        }


        }  /* of ia */
      }    /* of ir */
    }      /* of nu */


#if 0
    if(check_position_space_WI) {
      xchange_contraction(conn, 32);
      sprintf(filename, "WI_X.%s.%.4d.%d.%.4d", outfile_tag, Nconf, iflavor, g_cart_id);
      ofs = fopen(filename,"w");
      if(g_cart_id == 0) fprintf(stdout, "# [p2gg_xspace] checking Ward identity in position space\n");
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
#endif  /* of if 0 */


  }  /* end of loop on flavors up and down */
  
  /* print contact term */
  if(g_cart_id == have_source_flag) {
    fprintf(stdout, "# [p2gg_xspace] contact term\n");
    for(i=0;i<4;i++) {
      fprintf(stdout, "\t%d%25.16e%25.16e\n", i, contact_term[2*i], contact_term[2*i+1]);
    }
  }

  /* combine up-type and dn-type part, normalisation of contractions */
/*
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
*/

  for(ix=0; ix<16*VOLUME; ix++) {
    /* real part */
    conn_buffer[0][2*ix  ] += sequential_source_gamma_id_sign[ sequential_source_gamma_id ] * conn_buffer[1][2*ix  ];
    conn_buffer[0][2*ix  ] *= -0.25;
    /* conn_buffer[0][2*ix  ] *= 0.25; */

    /* imaginary part */
    conn_buffer[0][2*ix+1] -= sequential_source_gamma_id_sign[ sequential_source_gamma_id ] * conn_buffer[1][2*ix+1];
    conn_buffer[0][2*ix+1] *= -0.25;
    /* conn_buffer[0][2*ix+1] *= 0.25; */
  }

#ifdef HAVE_MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace] contractions in %e seconds\n", retime-ratime);

#ifdef HAVE_MPI
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace] broadcasing contact term ...\n");
  MPI_Bcast(contact_term, 8, MPI_DOUBLE, have_source_flag, g_cart_grid);
  fprintf(stdout, "[%2d] contact term = "\
      "(%e + I %e, %e + I %e, %e + I %e, %e +I %e)\n",
      g_cart_id, contact_term[0], contact_term[1], contact_term[2], contact_term[3],
      contact_term[4], contact_term[5], contact_term[6], contact_term[7]);
#endif

  /* subtact the contact term */
  if(g_cart_id == have_source_flag) {
    ix = g_ipt[sx0][sx1][sx2][sx3];
    conn_buffer[0][_GWI( 0,ix,VOLUME)  ] -= contact_term[0];
    conn_buffer[0][_GWI( 0,ix,VOLUME)+1] -= contact_term[1];
    conn_buffer[0][_GWI( 5,ix,VOLUME)  ] -= contact_term[2];
    conn_buffer[0][_GWI( 5,ix,VOLUME)+1] -= contact_term[3];
    conn_buffer[0][_GWI(10,ix,VOLUME)  ] -= contact_term[4];
    conn_buffer[0][_GWI(10,ix,VOLUME)+1] -= contact_term[5];
    conn_buffer[0][_GWI(15,ix,VOLUME)  ] -= contact_term[6];
    conn_buffer[0][_GWI(15,ix,VOLUME)+1] -= contact_term[7];
  }



  /* save results */
#ifdef HAVE_MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(strcmp(g_outfile_prefix, "NA") == 0) {
    sprintf(filename, "%s_x.%s.%.4d", outfile_name, outfile_tag, Nconf);
  } else {
    sprintf(filename, "%s/%s_x.%s.%.4d", g_outfile_prefix, outfile_name, outfile_tag, Nconf);
  }

  sprintf(contype, "\n<description>P - cvc - cvc in position space, 4x4 components</description>\n"\
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
      sequential_source_gamma_id, g_sequential_source_timeslice,
      contact_term[0], contact_term[1],
      contact_term[2], contact_term[3],
      contact_term[4], contact_term[5],
      contact_term[6], contact_term[7]);
    
  write_lime_contraction(conn_buffer[0], filename, 64, 16, contype, Nconf, 0);

/*
  for(ix=0;ix<VOLUME;ix++) {
    for(mu=0;mu<16;mu++) {
      fprintf(stdout, "%2d%6d%3d%25.16e%25.16e\n", g_cart_id, ix, mu, conn[_GWI(mu,ix,VOLUME)], conn[_GWI(mu,ix,VOLUME)+1]);
    }
  }
*/

  if(write_ascii) {
    conn = conn_buffer[0];
#ifndef HAVE_MPI
    if(strcmp(g_outfile_prefix, "NA") == 0) {
      sprintf(filename, "%s_x.%s.%.4d.ascii", outfile_name, outfile_tag, Nconf);
    } else {
      sprintf(filename, "%s/%s_x.%s.%.4d.ascii", g_outfile_prefix, outfile_name, outfile_tag, Nconf);
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
  }

#ifdef HAVE_MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace] saved position space results in %e seconds\n", retime-ratime);


  /********************************************
   * check the Ward identity in position space 
   ********************************************/
  if(check_position_space_WI) {
    conn = conn_buffer[0];
    xchange_contraction(conn, 32);
    sprintf(filename, "WI_X.%s.%.4d.%.4d", outfile_tag, Nconf, g_cart_id);
    ofs = fopen(filename,"w");
    if(g_cart_id == 0) fprintf(stdout, "# [p2gg_xspace] checking Ward identity in position space\n");
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


  }  /* end of loop on sequential gamma id */

  }  /* end of loop on sequential source momentum */

  }  /* end of loop on sequential source timeslices */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  /* free(g_gauge_field); */
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);
  free_geometry();
  if(conn_buffer != NULL) {
    if(conn_buffer[0] != NULL) {
      free(conn_buffer[0]);
    }
    free(conn_buffer);
  }

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  free(status);
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_xspace] %s# [p2gg_xspace] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_xspace] %s# [p2gg_xspace] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
