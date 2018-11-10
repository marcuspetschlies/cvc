/****************************************************
 * p2gg_xspace_local.c
 *
 * Tue Jan  5 09:12:49 CET 2016
 *
 * - originally copied from p2gg_xspace.c
 *
 * PURPOSE:
 * - contractions for P -> gamma gamma in position space
 * - local vector current
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

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform lvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  EXIT(0);
}


int main(int argc, char **argv) {
  
  char outfile_name[] = "p2gg_local";

  int c, i, j, mu, nu, ir, is, ia, ib, imunu;
  int op_id = 0;
  int filename_set = 0;
  int source_location, have_source_flag = 0, have_shifted_source_flag = 0;
  int x0, x1, x2, x3, ix;
  int gsx0, gsx1, gsx2, gsx3;
  int sx0, sx1, sx2, sx3;
  int source_coords[4];
  int isimag[4];
  int gperm[5][4], gperm2[4][4];
  int check_position_space_WI=0;
  int nthreads=-1, threadid=-1;
  int exitstatus;
  int write_ascii=0;
  int source_proc_coords[4], source_proc_id = -1;
  int shifted_source_coords[4], shifted_source_proc_coords[4];
  double gperm_sign[5][4], gperm2_sign[4][4];
  double *conn = (double*)NULL;
  int verbose = 0;
  char filename[100], contype[400];
  double ratime, retime;
  double plaq;
  double spinor1[24], spinor2[24], U_[18];
  double *gauge_trafo=(double*)NULL;
  double *phi=NULL, *chi=NULL, *source=NULL, *propagator=NULL;
  complex w, w1;
  double phase=0.;
  int LLBase[4];
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

  while ((c = getopt(argc, argv, "ah?vf:")) != -1) {
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
      fprintf(stdout, "\n# [p2gg_xspace_local] will check Ward identity in position space\n");
      break;
    case 'a':
      write_ascii = 1;
      fprintf(stdout, "\n# [p2gg_xspace_local] will write data in ASCII format too\n");
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
  fprintf(stdout, "# [p2gg_xspace_local] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_xspace_local] calling tmLQCD wrapper init functions\n");

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
    if(g_proc_id==0) fprintf(stderr, "[p2gg_xspace_local] T and L's must be set\n");
    usage();
  }


#ifdef HAVE_MPI
  if((status = (int*)calloc(g_nproc, sizeof(int))) == (int*)NULL) {
    EXIT(1);
  }
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg_xspace_local] Error from init_geometry\n");
    EXIT(2);
  }

  geometry();

  LLBase[0] = T_global;
  LLBase[1] = LX_global;
  LLBase[2] = LY_global;
  LLBase[3] = LZ_global;

#ifndef HAVE_TMLQCD_LIBWRAPPER
  if( !g_read_sequential_propagator ) {
    if(g_cart_id == 0) {
      fprintf(stderr, "[p2gg_xspace_local] Error, need tmLQCD libwrapper for inversion\n");
      EXIT(8);
    }
  }
#endif

#ifdef HAVE_TMLQCD_LIBWRAPPER
  if(g_tmLQCD_lat.no_operators > 2) {
    fprintf(stderr, "[p2gg_xspace_local] Error, confused about number of operators, expected 1 operator (up-type)\n");
    EXIT(9);
  }

  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg_xspace_local] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }
#endif

  /* allocate memory for the spinor fields */
  no_fields = 24;
#ifdef HAVE_TMLQCD_LIBWRAPPER
  no_fields++;
#endif

  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  source = g_spinor_field[no_fields-1];
#endif

  /* allocate memory for the contractions */
  conn = (double*)malloc(32*(VOLUME+RAND)*sizeof(double));
  if( conn == NULL ) {
    fprintf(stderr, "[p2gg_xspace_local] could not allocate memory for contr. fields\n");
    EXIT(6);
  }

  memset(conn, 0, 32*VOLUME*sizeof(double));

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
    fprintf(stdout, "# [p2gg_xspace_local] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx0, gsx1, gsx2, gsx3);
    fprintf(stdout, "# [p2gg_xspace_local] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  have_source_flag = (int)(g_cart_id == source_proc_id);
  if(have_source_flag==1) {
    fprintf(stdout, "# [p2gg_xspace_local] process %2d has source location\n", source_proc_id);
  }
  sx0 = gsx0 % T;
  sx1 = gsx1 % LX;
  sx2 = gsx2 % LY;
  sx3 = gsx3 % LZ;
# else
  have_source_flag = (int)(g_source_location/(LX*LY*LZ)>=Tstart && g_source_location/(LX*LY*LZ)<(Tstart+T));
  if(have_source_flag==1) fprintf(stdout, "[p2gg_xspace_local] process %2d has source location\n", g_cart_id);
  gsx0 = g_source_location/(LX*LY*LZ)-Tstart;
  gsx1 = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
  gsx2 = (g_source_location%(LY*LZ)) / LZ;
  gsx3 = (g_source_location%LZ);

  sx0 = gsx0 - Tstart;
  sx1 = gsx1;
  sx2 = gsx2;
  sx3 = gsx3;
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

  // initialize the contact term
  memset(contact_term, 0, 8*sizeof(double));

#ifdef HAVE_MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif


  /**********************************************************
   * loop on shifted source locations
   **********************************************************/
  for(mu=1; mu<5; mu++) {

    /**********************************************************
     * determine current source coords and location
     **********************************************************/
    source_coords[0] = gsx0;
    source_coords[1] = gsx1;
    source_coords[2] = gsx2;
    source_coords[3] = gsx3;
    if(mu < 4 ) source_coords[mu]++;
    fprintf(stdout, "# [p2gg_xspace_local] current global source coords (%d, %d, %d, %d)\n",
        source_coords[0], source_coords[1], source_coords[2], source_coords[3]);

    
    /**********************************************************
     * read dn spinor fields
     **********************************************************/
    for(ia=0; ia<12; ia++) {
      get_filename(filename, mu, ia, 1); 
      exitstatus = read_lime_spinor(g_spinor_field[12 + ia], filename, 1-g_propagator_position);
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_xspace_local] Error from read_lime_spinor, status was %d\n", exitstatus);
        EXIT(7);
      }
      xchange_field(g_spinor_field[12 + ia]);
    }  /* of loop on ia */


  /***********************************************************
   * invert using tmLQCD invert
   ***********************************************************/
#ifdef HAVE_TMLQCD_LIBWRAPPER
#endif

  /***********************************************************
   * determine absolute sequential source timeslice
   ***********************************************************/
  op_id = g_propagator_position;  
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_xspace_local] using op_id = %d\n", op_id);

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
      fprintf(stdout, "# [p2gg_xspace_local] process %4d has global sequential source timeslice %d -> %d\n", g_cart_id, gsx0, shifted_source_coords[0] );
/*
      fprintf(stdout, "# [p2gg_xspace_local] process %4d has global sequential source timeslice (%d,%d, %d, %d) -> (%d, %d, %d, %d)\n", g_cart_id, 
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
      fprintf(stdout, "# [p2gg_xspace_local] process %4d has local sequential source timeslice %d\n", g_cart_id, shifted_source_coords[0]);
    }

      for(ia=0; ia<12; ia++) {

        if(g_read_sequential_propagator) {
          /**********************************************************
           * read up-type sequential propagator
           **********************************************************/
          sprintf(filename, "%s.%.4d.%.2d.px%.2d py%.2d pz%.2d.%.2d.inverted", filename_prefix3, Nconf, mu,
              g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2], ia);
 
          exitstatus = read_lime_spinor(g_spinor_field[ia], filename, g_propagator_position);
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg_xspace_local] Error from read_lime_spinor, status was %d\n", exitstatus);
            EXIT(10);
          }
          xchange_field(g_spinor_field[ia]);

        } else {
          /**********************************************************
           * read up spinor field
           **********************************************************/
          get_filename(filename, mu, ia, 1);
          exitstatus = read_lime_spinor(g_spinor_field[ia], filename, g_propagator_position);
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg_xspace_local] Error from read_lime_spinor, status was %d\n", exitstatus);
            EXIT(11);
          }
          xchange_field(g_spinor_field[ia]);

          /**********************************************************
           * prepare sequential source
           **********************************************************/
          if(g_cart_id == 0) fprintf(stdout, "# [p2gg_xspace_local] preparing sequential source for spin-color component (%d, %d)\n", ia/3, ia%3);

          propagator = g_spinor_field[ia];

          memset(source, 0, 24*VOLUME*sizeof(double));

          if(have_shifted_source_flag) {

            x0 = shifted_source_coords[0];
            for(x1=0;x1<LX;x1++) {
            for(x2=0;x2<LY;x2++) {
            for(x3=0;x3<LZ;x3++) {
              ix = g_ipt[x0][x1][x2][x3];
              phase = (1. - 2. * g_propagator_position) * 2. * M_PI * ( ( x1 + g_proc_coords[1]*LX ) * g_seq_source_momentum[0] / (double)LX_global
                                  + ( x2 + g_proc_coords[2]*LY ) * g_seq_source_momentum[1] / (double)LY_global
                                  + ( x3 + g_proc_coords[3]*LZ ) * g_seq_source_momentum[2] / (double)LZ_global );
              w.re = cos(phase);
              w.im = sin(phase);
              _fv_eq_gamma_ti_fv(spinor1, g_sequential_source_gamma_id, propagator + _GSI(ix));
              _fv_eq_fv_ti_co(source + _GSI(ix), spinor1, &w);
            }}}


          }  /* end of if have_shifted_source_flag */

          xchange_field(source);


          /**********************************************************
           * invert
           **********************************************************/
#ifdef HAVE_TMLQCD_LIBWRAPPER
          if(g_cart_id == 0) fprintf(stdout, "# [p2gg_xspace_local] inverting for spin-color component (%d, %d)\n", ia/3, ia%3);
          exitstatus = tmLQCD_invert(propagator, source, op_id, 0);

          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg_xspace_local] Error from tmLQCD_invert, status was %d\n", exitstatus);
            EXIT(12);
          }

          if(g_write_propagator) {
            fprintf(stdout, "# [p2gg_xspace_local] Warning, will not write propagator to file\n");
          }
#endif
          xchange_field(propagator);

        }  /* end of else of if read sequential propagator */

      }  /* end of loop on spin-color component */


#ifdef HAVE_MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace_local] reading / invert in %e seconds\n", retime-ratime);

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
   * contraction
   **********************************************************/  

    /* loop on the Lorentz index nu at source */
    for(nu=0; nu<4; nu++) 
    {

      for(ir=0; ir<4; ir++) {

        for(ia=0; ia<3; ia++) {
          phi = g_spinor_field[3*ir            + ia];

          chi = g_spinor_field[3*gperm[nu][ir] + ia];

          /* 1) gamma_nu gamma_5 x U */

          imunu = 4*mu+nu;

          for(ix=0; ix<VOLUME; ix++) {

            _fv_eq_gamma_ti_fv(spinor2, mu, phi+_GSI(ix) );
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
        
        }    /* of ia */
      }      /* of ir */
    }        /* of nu */

  }  /* end of loop on mu */

  
  /* normalisation of contractions */
  for(ix=0; ix<32*VOLUME; ix++) conn[ix] *= -1;

#ifdef HAVE_MPI
  retime = MPI_Wtime();
#else
  retime = (double)clock() / CLOCKS_PER_SEC;
#endif
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace_local] contractions in %e seconds\n", retime-ratime);

  /* save results */
#ifdef HAVE_MPI
  ratime = MPI_Wtime();
#else
  ratime = (double)clock() / CLOCKS_PER_SEC;
#endif

  if(strcmp(g_outfile_prefix, "NA") == 0) {
    sprintf(filename, "%s_x.%.4d", outfile_name, Nconf);
  } else {
    sprintf(filename, "%s/%s_x.%.4d.t%.2dx%.2dy%.2dz%.2d", g_outfile_prefix, outfile_name, Nconf,
        source_coords[0], source_coords[1], source_coords[2], source_coords[3]);
  }
  sprintf(contype, "P - lvc - lvc in position space, all 16 components");
  write_lime_contraction(conn, filename, 64, 16, contype, Nconf, 0);


  if(write_ascii) {
#ifndef HAVE_MPI
    if(strcmp(g_outfile_prefix, "NA") == 0) {
      sprintf(filename, "%s_x.%.4d.ascii", outfile_name, Nconf);
    } else {
      sprintf(filename, "%s/%s_x.%.4d.ascii", g_outfile_prefix, outfile_name, Nconf);
    }
    write_contraction(conn, NULL, filename, 16, 2, 0);
#else
    sprintf(filename, "%s_x.%.4d.ascii.%.2d", outfile_name, Nconf, g_cart_id);
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
  if(g_cart_id==0) fprintf(stdout, "# [p2gg_xspace_local] saved position space results in %e seconds\n", retime-ratime);


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
  mpi_fini_xchange_contraction();
  free(status);
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_xspace_local] %s# [p2gg_xspace_local] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_xspace_local] %s# [p2gg_xspace_local] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
