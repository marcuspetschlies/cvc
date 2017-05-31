/****************************************************
 * hvp_full.c
 *
 * Mon May 29 08:30:19 CEST 2017
 *
 * - originally copied from p2gg.c
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
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "project.h"
#include "matrix_init.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -w                  : check position space WI [default false]\n");
  EXIT(0);
}

int dummy_eo_solver (double * const propagator, double * const source, const int op_id) {
  memcpy(propagator, source, _GSI(VOLUME)/2*sizeof(double) );
  return(0);
}


int main(int argc, char **argv) {
  
  /*
   * sign for g5 Gamma^\dagger g5
   *                                                0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   * */
  // const int sequential_source_gamma_id_sign[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };

  const char infile_prefix[]  = "p2gg";
  const char outfile_prefix[] = "hvp_full";

  int c;
  int filename_set = 0;
  int isource_location;
  int gsx[4];
  int exitstatus;
  int io_proc = 2;
  int evecs_num = 0;
  int append = 0;
  char filename[100];
  // double ratime, retime;
  FILE *ofs = NULL;

  fftw_complex *fftw_buffer_in = NULL, *fftw_buffer_out = NULL;
  fftw_plan plan_p;

  struct AffReader_s *affr = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex ***aff_buffer = NULL; 
  double _Complex ***hvp_buffer = NULL; 
  char aff_buffer_path[400];
  char aff_tag[400];
  /*  uint32_t aff_buffer_size; */

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while (( c = getopt(argc, argv, "ah?f:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'a':
      append = 1;
      fprintf(stdout, "# [hvp_full] will append output to data files\n");
      break;
    case 'n':
      evecs_num = atoi( optarg );
      fprintf(stdout, "# [hvp_full] number of eigenvalues set to %d\n", evecs_num);
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
  if(filename_set==0) strcpy(filename, "hvp_full.input");
  if ( g_verbose > 2 ) fprintf(stdout, "# [hvp_full] Reading input from file %s\n", filename);
  read_input_parser(filename);


  if ( evecs_num == 0 ) {
    fprintf(stderr, "[hvp_full] Error , number of eigenvalues is zero\n");
    EXIT(12);
  }

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hvp_full] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hvp_full] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hvp_full] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[hvp_full] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/
#if 0
  evecs_eval = (double*)malloc(evecs_num*sizeof(double));
  evecs_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if(evecs_eval == NULL || evecs_lambdainv == NULL ) {
    fprintf(stderr, "[hvp_full] Error from malloc %s %d\n", __FILE__, __LINE__);
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
    if( g_cart_id == 0 ) fprintf(stdout, "# [hvp_full] eval %4d %16.7e\n", i, evecs_eval[i] );
  }
#endif  /* of if 0 */
  if(io_proc == 2) {
    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [hvp_full] using aff version %s\n", aff_status_str);
  }

  fftw_buffer_in = (fftw_complex*)malloc ( T_global * sizeof(fftw_complex ) );
  fftw_buffer_out = (fftw_complex*)malloc ( T_global * sizeof(fftw_complex ) );
  plan_p = fftw_create_plan(T_global, FFTW_BACKWARD, FFTW_MEASURE);


  /***********************************************************
   ***********************************************************
   **
   ** loop on source locations
   **
   ***********************************************************
   ***********************************************************/
  for(isource_location=0; isource_location < g_source_location_number; isource_location++) {
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
      fprintf(stdout, "# [hvp_full] reading data from file %s\n", filename);
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[hvp_full] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
      if( (affn = aff_reader_root(affr)) == NULL ) {
        fprintf(stderr, "[hvp_full] Error, aff writer is not initialized\n");
        exit(103);
      }
    }  /* end of if io_proc == 2 */
    sprintf(aff_tag, "/full/hvp/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );

    exitstatus = init_3level_zbuffer ( &aff_buffer, 4, 4, T_global );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[hvp_full] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }

    exitstatus = init_3level_zbuffer ( &hvp_buffer, 4, 4, T_global );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[hvp_full] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }

    /* loop on sink momenta  */
    for ( int ip = 0; ip < g_sink_momentum_number; ip++ ) {
      double p[4] = { 0.,
       2.*M_PI * g_sink_momentum_list[ip][0] / (double)LX_global,
       2.*M_PI * g_sink_momentum_list[ip][1] / (double)LY_global,
       2.*M_PI * g_sink_momentum_list[ip][2] / (double)LZ_global }; 

      /* phase factor from shift by source location 3-vector */
      double _Complex phase_factor_source = cexp ( -I* ( p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ));

      sprintf(aff_buffer_path, "%s/px%.2dpy%.2dpz%.2d", aff_tag, g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2] );
     
      affdir = aff_reader_chpath(affr, affn, aff_buffer_path);
      fprintf(stdout, "# [hvp_full] reading key %s\n", aff_buffer_path);
      exitstatus = aff_node_get_complex (affr, affdir, aff_buffer[0][0], (uint32_t)T_global*16);
      if(exitstatus != 0) {
        fprintf(stderr, "[hvp_full] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(5);
      }

      memcpy ( hvp_buffer[0][0], aff_buffer[0][0], 16*T_global*sizeof(double _Complex) );

      /* p_0 = 0, since we only consider the 3-dim. FT */
      p[0] = 0.;
      /* write time dependence for fixed 3-momentum */
      for( int mu = 0; mu < 4; mu++ ) {
      for( int nu = 0; nu < 4; nu++ ) {

        /* phase factor due to half-site shift */
        double _Complex phase_factor_half = ( mu == nu ) ? 1. : cexp( 0.5 * I * (p[mu] - p[nu] ));
        double _Complex phase_factor = phase_factor_source * phase_factor_half;

        for ( int it = 0; it < T_global; it++ ) {
          hvp_buffer[mu][nu][it] *= phase_factor;
        }

        sprintf(filename, "%s_t.px%.2dpy%.2dpz%.2d.mu%dnu%d", outfile_prefix, g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2], mu, nu);
        ofs = append == 1 ? fopen(filename, "a") : fopen(filename, "w");
        for ( int it = 0; it < T_global; it++ ) {
          int k = ( it + gsx[0] ) % T_global;
          fprintf(ofs, "%3d %3d %3d %25.16e %25.16e %8d\n", mu, nu, it, 
             creal( hvp_buffer[mu][nu][k] ), cimag( hvp_buffer[mu][nu][k] ),
             Nconf);
        }
        fclose(ofs);
      }}
 
      /* FT in t <-> p_0; write energy dependence for fixed 3-momentum */
      memset( hvp_buffer[0][0], 0, T_global*16*sizeof(double _Complex) );
      for( int mu = 0; mu < 4; mu++ ) {
      for( int nu = 0; nu < 4; nu++ ) {
   
        /* FT */
#if 0
        memcpy(fftw_buffer_in,  hvp_buffer[mu][nu], T_global * sizeof(double _Complex) );
        fftw_one(plan_p, fftw_buffer_in, fftw_buffer_out);
        memcpy( hvp_buffer[mu][nu], fftw_buffer_out, T_global * sizeof(double _Complex) );
#endif  /* of if 0 */
        fftw_one(plan_p, (fftw_complex*)(aff_buffer[mu][nu]), (fftw_complex*)(hvp_buffer[mu][nu]) );

        for ( int it = 0; it < T_global; it++ ) {
          p[0] = 2. * M_PI * it / (double)T_global;
          /* phase factor due to shift by source location */
          double _Complex phase_factor_source = cexp ( -I* (p[0] * gsx[0] +  p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ));
          /* phase factor due to half-site shift */
          double _Complex phase_factor_half = ( mu == nu ) ? 1. : cexp( 0.5 * I * (p[mu] - p[nu] ));
          /* full phase factor */
          double _Complex phase_factor = phase_factor_source * phase_factor_half;
          
          hvp_buffer[mu][nu][it] *= phase_factor;
        }

        sprintf(filename, "%s_p.px%.2dpy%.2dpz%.2d.mu%dnu%d", outfile_prefix, g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2], mu, nu);
        ofs = append == 1 ? fopen(filename, "a") : fopen(filename, "w");
        for ( int it = 0; it < T_global; it++ ) {
          fprintf(ofs, "%3d %3d %3d %25.16e %25.16e %8d\n", mu, nu, it, 
             creal( hvp_buffer[mu][nu][it] ), cimag( hvp_buffer[mu][nu][it] ),
             Nconf);
        }
        fclose(ofs);
      }}

      /* check momentum space WI */

      for ( int nu = 0; nu < 4; nu++ ) {
        for ( int it = 0; it < T_global; it++ ) {
           double plat[4] = {
             2.*sin( M_PI*it / (double)T_global),
             2.*sin( M_PI * g_sink_momentum_list[ip][0] / (double)LX_global ),
             2.*sin( M_PI * g_sink_momentum_list[ip][1] / (double)LY_global ),
             2.*sin( M_PI * g_sink_momentum_list[ip][2] / (double)LZ_global ) };

           double _Complex wi_res1 = plat[0] * hvp_buffer[ 0][nu][it] + plat[1] * hvp_buffer[ 1][nu][it] + plat[2] * hvp_buffer[ 2][nu][it] + plat[3] * hvp_buffer[ 3][nu][it];
           double _Complex wi_res2 = plat[0] * hvp_buffer[nu][ 0][it] + plat[1] * hvp_buffer[nu][ 1][it] + plat[2] * hvp_buffer[nu][ 2][it] + plat[3] * hvp_buffer[nu][ 3][it];

           fprintf(stdout, "# [hvp_full] WI p %2d %2d %2d %2d nu %d res1 %25.16e %25.16e res2 %25.16e %25.16e \n",
               it, g_sink_momentum_list[ip][0], g_sink_momentum_list[ip][1], g_sink_momentum_list[ip][2], nu,
               creal(wi_res1), cimag(wi_res1),
               creal(wi_res2), cimag(wi_res2) );
        }
      }
    }  /* end of loop on sink momenta */

    if(io_proc == 2) {
      aff_reader_close (affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[hvp_full] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */

    fini_3level_zbuffer ( &aff_buffer );
    fini_3level_zbuffer ( &hvp_buffer );

  }  /* end of loop on source locations */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

#if 0
  free ( evecs_eval );
  free ( evecs_lambdainv );
#endif

  free( fftw_buffer_in );
  free( fftw_buffer_out );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [hvp_full] %s# [hvp_full] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [hvp_full] %s# [hvp_full] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
