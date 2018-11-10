/****************************************************
 * p2gg_xm.c
 *
 * Thu Jun  1 12:04:49 CEST 2017
 *
 * - originally copied from hvp_xm.cpp
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "ifftw.h"

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
#include "read_input_parser.h"
#include "contractions_io.h"
#include "contract_cvc_tensor.h"
#include "form_factor_projection.h"
#include "matrix_init.h"


#define _OP_ID_UP 0
#define _OP_ID_DN 1


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -a                  : append [default false]\n");
  fprintf(stdout, "          -n <nev>            : number of eigenvectors [default 0]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  /*
   * sign for g5 Gamma^\dagger g5
   *                                                0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   * */
  // const int sequential_source_gamma_id_sign[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };

  const char infile_prefix[]  = "p2gg";

  int c;
  int filename_set = 0;
  int gsx[4];
  int exitstatus;
  int io_proc = 2;
  int evecs_num = 0;
  int append = 0;
  int check_WI_momentum_space = 0;

  char filename[100];
  char outfile_prefix[100];
  char mode_tag[6];

  // double ratime, retime;
  FILE *ofs = NULL;

  fftw_plan plan_p;

  struct AffReader_s *affr = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex ***aff_buffer = NULL; 
  double _Complex ***p2gg_buffer = NULL; 
  char aff_buffer_path[400];
  char aff_tag[400];
  /*  uint32_t aff_buffer_size; */

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

  while (( c = getopt(argc, argv, "wpah?f:n:m:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'a':
      append = 1;
      fprintf(stdout, "# [p2gg_xm] will append output to data files\n");
      break;
    case 'w':
      check_WI_momentum_space = 1;
      fprintf(stdout, "# [p2gg_xm] will check WI in momentum space\n");
      break;
    case 'n':
      evecs_num = atoi( optarg );
      fprintf(stdout, "# [p2gg_xm] number of eigenvalues set to %d\n", evecs_num);
      break;
    case 'm':
      strcpy(mode_tag, optarg);
      fprintf(stdout, "# [p2gg_xm] using mode tag %s\n", mode_tag);
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
  if(filename_set==0) strcpy(filename, "p2gg_xm.input");
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_xm] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_xm] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_xm] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_xm] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[p2gg_xm] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/
#if 0
  if ( evecs_num == 0 ) {
    fprintf(stderr, "[p2gg_xm] Error , number of eigenvalues is zero\n");
    EXIT(12);
  }

  evecs_eval = (double*)malloc(evecs_num*sizeof(double));
  evecs_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if(evecs_eval == NULL || evecs_lambdainv == NULL ) {
    fprintf(stderr, "[p2gg_xm] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  sprintf(filename, "eval.%.4d.nev%.4d", Nconf, evecs_num);
  ofs = fopen(filename, "r");
  for(i=0; i<evecs_num; i++) {
    fscanf(ofs, "%lf", evecs_eval+i);
  }
  fclose ( ofs );

  for(i=0; i<evecs_num; i++) {
    evecs_eval[i] = ((double*)(g_tmLQCD_defl.evals))[2*i];
    evecs_lambdainv[i] = 2.* g_kappa / evecs_eval[i];
    if( g_cart_id == 0 ) fprintf(stdout, "# [p2gg_xm] eval %4d %16.7e\n", i, evecs_eval[i] );
  }
#endif  /* of if 0 */
  if(io_proc == 2) {
    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [p2gg_xm] using aff version %s\n", aff_status_str);
  }

  plan_p = fftw_create_plan(T_global, FFTW_BACKWARD, FFTW_MEASURE);


  sprintf(outfile_prefix, "p2gg_%s", mode_tag);
  fprintf(stdout, "# [] using outfile_prefix %s\n", outfile_prefix);

  /***********************************************************
   ***********************************************************
   **
   ** loop on source locations
   **
   ***********************************************************
   ***********************************************************/
  for( int isource_location=0; isource_location < g_source_location_number; isource_location++) {
    /***********************************************************
     * determine source coordinates, find out, if source_location is in this process
     ***********************************************************/
    gsx[0] = g_source_coords_list[isource_location][0];
    gsx[1] = g_source_coords_list[isource_location][1];
    gsx[2] = g_source_coords_list[isource_location][2];
    gsx[3] = g_source_coords_list[isource_location][3];

    /***********************************************
     ***********************************************
     **
     ** writer for aff output file
     **
     ***********************************************
     ***********************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", infile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [p2gg_xm] reading data from file %s\n", filename);
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_xm] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
      if( (affn = aff_reader_root(affr)) == NULL ) {
        fprintf(stderr, "[p2gg_xm] Error, aff writer is not initialized\n");
        exit(103);
      }
    }  /* end of if io_proc == 2 */
    sprintf(aff_tag, "/%s/PJJ/t%.2dx%.2dy%.2dz%.2d", mode_tag, gsx[0], gsx[1], gsx[2], gsx[3] );

    exitstatus = init_3level_zbuffer ( &aff_buffer, 4, 4, T_global );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_xm] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }

    exitstatus = init_3level_zbuffer ( &p2gg_buffer, 4, 4, T_global );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_xm] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }

    for ( int iq = 0; iq < g_seq_source_momentum_number; iq++ ) {

      double q[3] = {
        2. * M_PI * g_seq_source_momentum_list[iq][0],
        2. * M_PI * g_seq_source_momentum_list[iq][1],
        2. * M_PI * g_seq_source_momentum_list[iq][2]
      };
      double q4[4] = { 0., q[0], q[1], q[2] };

      for ( int ig = 0; ig < g_sequential_source_gamma_id_number; ig++ ) {

        for ( int itseq = 0; itseq < g_sequential_source_timeslice_number; itseq++ ) {

          int tseq = g_sequential_source_timeslice_list[itseq];
          int tseq_shifted = ( gsx[0] + g_sequential_source_timeslice_list[itseq] + T_global ) % T_global;

          /* loop on sink momenta  */
          for ( int ip = 0; ip < g_sink_momentum_number; ip++ ) {
            double p[3] = {
              2.*M_PI * g_sink_momentum_list[ip][0] / (double)LX_global,
              2.*M_PI * g_sink_momentum_list[ip][1] / (double)LY_global,
              2.*M_PI * g_sink_momentum_list[ip][2] / (double)LZ_global
            }; 
            double p4[4] = { 0., p[0], p[1], p[2] };

            /* phase factor from shift by source location 3-vector */
            double _Complex phase_factor_source = cexp ( -I* ( ( p[0] + q[0] ) * gsx[1] + ( p[1] + q[1] ) * gsx[2] + ( p[2] + q[2] ) * gsx[3] ) );

            sprintf(aff_buffer_path, "%s/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/px%.2dpy%.2dpz%.2d", aff_tag, 
                g_seq_source_momentum_list[iq][0], g_seq_source_momentum_list[iq][1], g_seq_source_momentum_list[iq][2],
                g_sequential_source_gamma_id_list[ig], tseq,
                g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2] );
            affdir = aff_reader_chpath(affr, affn, aff_buffer_path);
            fprintf(stdout, "# [p2gg_xm] reading key %s\n", aff_buffer_path);
            exitstatus = aff_node_get_complex (affr, affdir, aff_buffer[0][0], (uint32_t)T_global*16);
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg_xm] Error from aff_node_get_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(5);
            }

            memcpy ( p2gg_buffer[0][0], aff_buffer[0][0], 16*T_global*sizeof(double _Complex) );

            /* write time dependence for fixed 3-momentum */
            for( int mu = 0; mu < 4; mu++ ) {
            for( int nu = 0; nu < 4; nu++ ) {

              /* phase factor due to half-site shift */
              double _Complex phase_factor_half = ( mu == nu ) ? 1. : cexp( 0.5 * I * (p4[mu] - ( p4[nu]+q4[nu] ) ));
              double _Complex phase_factor = phase_factor_source * phase_factor_half;

              for ( int it = 0; it < T_global; it++ ) {
                p2gg_buffer[mu][nu][it] *= phase_factor;
              }

              sprintf(filename, "%s_t.qx%.2dqy%.2dqz%.2d.gseq%.2d.tseq%.2d.px%.2dpy%.2dpz%.2d.mu%dnu%d", outfile_prefix, 
                  g_seq_source_momentum_list[iq][0], g_seq_source_momentum_list[iq][1], g_seq_source_momentum_list[iq][2],
                  g_sequential_source_gamma_id_list[ig], tseq,
                  g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2], mu, nu);
              ofs = ( append == 0 && isource_location == 0 ) ? fopen(filename, "w") : fopen(filename, "a");
              fprintf(ofs, "# %s\n", aff_buffer_path);
              for ( int it = 0; it < T_global; it++ ) {
                int k = ( it + gsx[0] ) % T_global;
                fprintf(ofs, "%3d %3d %3d %25.16e %25.16e %8d\n", mu, nu, it, 
                creal( p2gg_buffer[mu][nu][k] ), cimag( p2gg_buffer[mu][nu][k] ),
                Nconf);
              }
              fclose(ofs);
            }}
 
            /* FT in t <-> p_0; write energy dependence for fixed 3-momentum */
            memset( p2gg_buffer[0][0], 0, T_global*16*sizeof(double _Complex) );
    
            double _Complex ***p2gg_buffer2 = NULL;
            exitstatus = init_3level_zbuffer ( &p2gg_buffer2, T_global, 4, 4);
            if(exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_xm] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(45);
            }

            for( int mu = 0; mu < 4; mu++ ) {
            for( int nu = 0; nu < 4; nu++ ) {
   
              /* FT */
              fftw_one(plan_p, (fftw_complex*)(aff_buffer[mu][nu]), (fftw_complex*)(p2gg_buffer[mu][nu]) );

              for ( int it = 0; it < T_global; it++ ) {
                p4[0] = 2. * M_PI * it / (double)T_global;
                /* phase factor due to shift by source location */
                double _Complex phase_factor_source = cexp ( -I* ( 
                      ( p4[0] + q4[0] ) * gsx[0] + 
                      ( p4[1] + q4[1] ) * gsx[1] + 
                      ( p4[2] + q4[2] ) * gsx[2] + 
                      ( p4[3] + q4[3] ) * gsx[3] ) );

                /* phase factor due to half-site shift */
                double _Complex phase_factor_half = ( mu == nu ) ? 1. : cexp( 0.5 * I * ( p4[mu] - ( p4[nu] + q4[nu] ) ));
                /* full phase factor */
                double _Complex phase_factor = phase_factor_source * phase_factor_half;
          
                p2gg_buffer[mu][nu][it] *= phase_factor;
                p2gg_buffer2[it][mu][nu] = p2gg_buffer[mu][nu][it];
              }
              sprintf(filename, "%s_p.qx%.2dqy%.2dqz%.2d.gseq%.2d.tseq%.2d.px%.2dpy%.2dpz%.2d.mu%dnu%d", outfile_prefix, 
                  g_seq_source_momentum_list[iq][0], g_seq_source_momentum_list[iq][1], g_seq_source_momentum_list[iq][2],
                  g_sequential_source_gamma_id_list[ig], tseq,
                  g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2], mu, nu);
              ofs = ( append == 0 && isource_location == 0 ) ? fopen(filename, "w") : fopen(filename, "a");
              fprintf(ofs, "# %s\n", aff_buffer_path);
              for ( int it = 0; it < T_global; it++ ) {
                fprintf(ofs, "%3d %3d %3d %25.16e %25.16e %8d\n", mu, nu, it, 
                   creal( p2gg_buffer[mu][nu][it] ), cimag( p2gg_buffer[mu][nu][it] ),
                   Nconf);
              }
              fclose(ofs);
            }}

#if 0
            /* project to scalar factors */
            double _Complex p2gg_t, p2gg_l;
            sprintf(filename, "%s_s.qx%.2dqy%.2dqz%.2d.gseq%.2d.tseq%.2d.px%.2dpy%.2dpz%.2d.mu%dnu%d", outfile_prefix, 
                g_seq_source_momentum_list[iq][0], g_seq_source_momentum_list[iq][1], g_seq_source_momentum_list[iq][2],
                g_sequential_source_gamma_id_list[ig], tseq,
                g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2], mu, nu);
            ofs = ( append == 0 && isource_location == 0 )  ? fopen(filename, "w") : fopen(filename, "a");
            fprintf(ofs, "# %s\n", aff_buffer_path);
            for ( int it = 0; it < T_global; it++ ) {
              _project_pseudotensor_to_scalar (p2gg_buffer2[it], &p2gg_t, &p2gg_l, p);
              fprintf(ofs, "%3d %25.16e %25.16e %25.16e %25.16e %8d\n", it, creal(p2gg_t), cimag(p2gg_t), creal(p2gg_l), cimag(p2gg_l), Nconf);
           }
           fclose(ofs);
#endif  /* of if 0 */
            /* check momentum space WI */
            if ( check_WI_momentum_space == 1 ) {
              for ( int nu = 0; nu < 4; nu++ ) {
                for ( int it = 0; it < T_global; it++ ) {
                   double plat[4] = {
                     2.*sin( M_PI*it / (double)T_global),
                     2.*sin( M_PI * g_sink_momentum_list[ip][0] / (double)LX_global ),
                     2.*sin( M_PI * g_sink_momentum_list[ip][1] / (double)LY_global ),
                     2.*sin( M_PI * g_sink_momentum_list[ip][2] / (double)LZ_global ) };
      
                   //double _Complex wi_res1 = plat[0] * p2gg_buffer[ 0][nu][it] + plat[1] * p2gg_buffer[ 1][nu][it] + plat[2] * p2gg_buffer[ 2][nu][it] + plat[3] * p2gg_buffer[ 3][nu][it];
                   //double _Complex wi_res2 = plat[0] * p2gg_buffer[nu][ 0][it] + plat[1] * p2gg_buffer[nu][ 1][it] + plat[2] * p2gg_buffer[nu][ 2][it] + plat[3] * p2gg_buffer[nu][ 3][it];
                   double _Complex wi_res1 = plat[0] * p2gg_buffer2[it][ 0][nu] + plat[1] * p2gg_buffer2[it][ 1][nu] + plat[2] * p2gg_buffer2[it][ 2][nu] + plat[3] * p2gg_buffer2[it][ 3][nu];
                   double _Complex wi_res2 = plat[0] * p2gg_buffer2[it][nu][ 0] + plat[1] * p2gg_buffer2[it][nu][ 1] + plat[2] * p2gg_buffer2[it][nu][ 2] + plat[3] * p2gg_buffer2[it][nu][ 3];
      
                   fprintf(stdout, "# [p2gg_xm] WI q %2d %2d %2d g %2d t %2d p %2d %2d %2d %2d nu %d res1 %25.16e %25.16e res2 %25.16e %25.16e \n",
                       g_seq_source_momentum_list[iq][0], g_seq_source_momentum_list[iq][1], g_seq_source_momentum_list[iq][2],
                       g_sequential_source_gamma_id_list[ig], tseq_shifted,
                       it, g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2], nu,
                       creal(wi_res1), cimag(wi_res1),
                       creal(wi_res2), cimag(wi_res2) );
                }
              }
            }  /* end of if check_WI_momentum_space */
            fini_3level_zbuffer ( &p2gg_buffer2 );

          }  /* end of loop on sink momenta */

        }  /* end of loop on sequential source timeslices */
      }  /* end of loop on sequential source gamma ids */
    }  /* end of loop on sequential source momenta  */

    if(io_proc == 2) {
      aff_reader_close (affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_xm] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */

    fini_3level_zbuffer ( &aff_buffer );
    fini_3level_zbuffer ( &p2gg_buffer );

  }  /* end of loop on source locations */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

#if 0
  free ( evecs_eval );
  free ( evecs_lambdainv );
#endif


  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_xm] %s# [p2gg_xm] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_xm] %s# [p2gg_xm] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
