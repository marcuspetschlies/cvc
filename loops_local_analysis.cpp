/****************************************************
 * loops_local_analysis.cpp
 *
 * - originally copied from loops_hvp_analysis.cpp
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
#include <complex.h>
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
#include "prepare_propagator.h"
#include "project.h"
#include "matrix_init.h"
#include "clover.h"
#include "scalar_products.h"
#include "fft.h"
#include "Q_phi.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#define _SQR(_a) ((_a)*(_a))

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to build 2-point function from local loops\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc    [default cvc.input]\n");
  fprintf(stdout, "          -n                  : number of eigenvectors [default 0]\n");
  EXIT(0);
}

/******************************************************/
/******************************************************/

int main(int argc, char **argv) {
  
  const char prefix[] = "loops_hvp";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int evecs_num = 0;
  int transverse_projection = 0;
  char filename[100];
  /* double ratime, retime; */
  double _Complex ***local_loop_lma = NULL, ****local_loop_stoch_aux = NULL, ****local_loop_stoch_cum = NULL;

#ifdef HAVE_MPI
  MPI_Status mstatus;
#endif


#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char * aff_status_str;
  char aff_tag[400];
  struct AffNode_s *affn = NULL, *affdir=NULL;
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?f:p:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'p':
      transverse_projection = atoi( optarg );
      break;
    case 'n':
      evecs_num = atoi( optarg );
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
  /* fprintf(stdout, "# [loops_local_analysis] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);


  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [loops_local_analysis] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [loops_local_analysis] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [loops_local_analysis] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[loops_local_analysis] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[loops_local_analysis] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  /*********************************
   * set up geometry fields
   *********************************/
  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();


  /******************************************************/
  /******************************************************/

  /******************************************************
   * allocate memory for the eigenvector fields
   ******************************************************/

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [loops_local_analysis] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [loops_local_analysis] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  if(io_proc == 2) {
    if(g_tr_id != 0) {
      fprintf(stderr, "[loops_local_analysis] Error, io proc must be id 0 in g_tr_comm %s %d\n", __FILE__, __LINE__);
      EXIT(14);
    }
  }
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * open AFF file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    sprintf(filename, "%s.%.4d.aff", prefix, Nconf );
    fprintf(stdout, "# [loops_local_analysis] reading data from file %s\n", filename);
    affr = aff_reader(filename);
    aff_status_str = (char*)aff_reader_errstr( affr );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[loops_local_analysis] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
    if( (affn = aff_reader_root(affr)) == NULL ) {
      fprintf(stderr, "[loops_local_analysis] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
      EXIT(1);
    }
  }  /* end of if io_proc == 2 */
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * read cvc lma loops and cvc_stoch loops
   ***********************************************************/

  if( ( exitstatus = init_3level_zbuffer ( &local_loop_lma, g_sink_momentum_number, 16, T_global  ) ) != 0 ) {
    fprintf(stderr, "[loops_local_analysis] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  /***********************************************************/
  /***********************************************************/
 
  if( ( exitstatus = init_4level_zbuffer ( &local_loop_stoch_aux, g_sink_momentum_number, 16, g_nsample, T_global  ) ) != 0 ) {
    fprintf(stderr, "[loops_local_analysis] Error from init_4level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }

  /***********************************************************
   * read from AFF file
   ***********************************************************/

  /***********************************************************
   * loop on momenta
   ***********************************************************/
  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    /* loop on mu */
    for ( int mu = 0; mu < 16; mu++ ) {
      sprintf(aff_tag, "/loop/local/nev%.4d/px%.2dpy%.2dpz%.2d/mu%d", evecs_num, 
          g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], mu );

      affdir = aff_reader_chpath(affr, affn, aff_tag);
      exitstatus = aff_node_get_complex (affr, affdir, local_loop_lma[imom][mu], (uint32_t)T_global );
      if(exitstatus != 0 ) {
        fprintf(stderr, "[loops_local_analysis] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(2);
      }
    }

    /***********************************************************/
    /***********************************************************/
 
    /***********************************************************
     * loop on samples
     ***********************************************************/
    for ( int isample = 0; isample < g_nsample; isample++ ) {

      double _Complex *zbuffer = (double _Complex*) malloc ( T_global * sizeof (double _Complex) );
      if ( zbuffer == NULL ) {
        fprintf(stderr, "[loops_local_analysis] Error from malloc %s %d\n", __FILE__, __LINE__);
        EXIT(1);
      }


      /***********************************************************
       * loop on mu
       ***********************************************************/
      for ( int mu = 0; mu < 16; mu++ ) {
 
        /***********************************************************
         * loop on source timeslices
         ***********************************************************/
        for ( int t_src = 0; t_src < T_global; t_src++ ) {

          sprintf(aff_tag, "/loop/local/sample%.4d/tsrc%.2d/px%.2dpy%.2dpz%.2d/mu%d", isample, t_src, 
              g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], mu );

          affdir = aff_reader_chpath(affr, affn, aff_tag);
          exitstatus = aff_node_get_complex (affr, affdir, zbuffer, (uint32_t)T_global );
          if(exitstatus != 0 ) {
            fprintf(stderr, "[loops_local_analysis] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
          complex_field_pl_eq_complex_field ( (double*)local_loop_stoch_aux[imom][mu][isample], (double*)zbuffer, T_global );
        }

      }  /* end of loop on mu */

      free ( zbuffer );
      zbuffer = NULL;
    }  /* end of loop on samples */
#if 0
#endif
  }  /* end of loop on momenta */

  /***********************************************************/
  /***********************************************************/
  

  if( ( exitstatus = init_4level_zbuffer ( &local_loop_stoch_cum, g_sink_momentum_number, 16, g_nsample, T_global  ) ) != 0 ) {
    fprintf(stderr, "[loops_local_analysis] Error from init_4level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }
  /***********************************************************
   * sum stochastic loops over stochastic samples
   ***********************************************************/
  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
 
    for ( int mu = 0; mu < 16; mu++ ) {
        
      complex_field_pl_eq_complex_field ( (double*)local_loop_stoch_cum[imom][mu][0], (double*)local_loop_stoch_aux[imom][mu][0], T_global );
      
      for ( int isample = 1; isample < g_nsample; isample++ ) {
        memcpy ( local_loop_stoch_cum[imom][mu][isample],  local_loop_stoch_cum[imom][mu][isample-1], T_global * sizeof(double _Complex) ); 
        complex_field_pl_eq_complex_field ( (double*)local_loop_stoch_cum[imom][mu][isample], (double*)local_loop_stoch_aux[imom][mu][isample], T_global );
      }
    }
  }
#if 0
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * combine loops at source and sink
   ***********************************************************/
  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    int p[3] = { g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] };

    int imom2 = 0;
    while ( ( g_sink_momentum_list[imom2][0] != -p[0] ) || ( g_sink_momentum_list[imom2][1] != -p[1] ) ||  ( g_sink_momentum_list[imom2][2] != -p[2] ) ) {
      imom2++;
    }
    if ( imom2 == g_sink_momentum_number ) {
      fprintf( stdout, "[loops_local_analysis] Error, could not find negative p-vector in momentum list\n");
      EXIT(2);
    }

    fprintf (stdout, "# [loops_local_analysis] using imom = %d and imom2 = %d\n", imom, imom2 );

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * set output filename prefix
     ***********************************************************/
    strcpy ( g_outfile_prefix, "ll");

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * lma x lma
     ***********************************************************/
    sprintf ( filename, "%s.%s.%.4d.px%.2dpy%.2dpz%.2d.%s", g_outfile_prefix, "disc", Nconf, p[0], p[1], p[2], "lma-lma" );
    FILE *ofs = fopen( filename, "w" );
    if ( ofs == NULL ) {
      fprintf ( stderr, "[loops_local_analysis] Error, could not open file %s for writing\n", filename );
      EXIT(1);
    }

    for ( int mu = 0; mu < 16; mu++ ) {
      int nu = mu;
      for ( int dt = 0; dt < T_global; dt++ ) {
        double _Complex  zs = 0., zv = 0.;
  
        for ( int t = 0; t < T_global; t++ ) {
          zs += 
            ( local_loop_lma[imom ][mu][(t+dt)%T_global] - conj( local_loop_lma[imom2][mu][(t+dt)%T_global]) )
          * ( local_loop_lma[imom2][nu][t]               - conj( local_loop_lma[imom ][nu][t]              ) );
       
          zv += 
            ( local_loop_lma[imom ][mu][(t+dt)%T_global] + conj( local_loop_lma[imom2][mu][(t+dt)%T_global]) )
          * ( local_loop_lma[imom2][nu][t]               + conj( local_loop_lma[imom ][nu][t]              ) );
        } 
        zs /= (double)T_global;
        zv /= (double)T_global;
        fprintf ( ofs, "%3d%3d%4d%25.16e%25.16e%25.16e%25.16e\n", mu, nu, dt, creal(zs), cimag(zs),  creal(zv), cimag(zv));
      }
    }
    fclose ( ofs );

    /***********************************************************/
    /***********************************************************/
   

    /***********************************************************
     * lma x stoch + stoch x lma
     ***********************************************************/

    /***********************************************************
     * cumulative number of samples
     ***********************************************************/
    for ( int isample = 0; isample < g_nsample; isample ++ ) {

      sprintf ( filename, "%s.%s.%.4d.px%.2dpy%.2dpz%.2d.nsample%.4d.%s", g_outfile_prefix, "disc", Nconf, p[0], p[1], p[2], isample+1, "lma-stoch" );
      FILE *ofs = fopen( filename, "w" );
      if ( ofs == NULL ) {
        fprintf ( stderr, "[loops_local_analysis] Error, could not open file %s for writing\n", filename );
        EXIT(1);
      }

      for ( int mu = 0; mu < 16; mu++ ) {
        int nu = mu;
        for ( int dt = 0; dt < T_global; dt++ ) {
          double _Complex  zs = 0., zv = 0.;
  
          for ( int t = 0; t < T_global; t++ ) {

            zs += 
              ( local_loop_lma[imom ][mu][(t+dt)%T_global]                - conj( local_loop_lma[imom2][mu][(t+dt)%T_global]                ) )
            * ( local_loop_stoch_cum[imom2][nu][isample][t]               - conj( local_loop_stoch_cum[imom ][nu][isample][t]               ) );

            zs +=
              ( local_loop_stoch_cum[imom ][mu][isample][(t+dt)%T_global] - conj( local_loop_stoch_cum[imom2][mu][isample][(t+dt)%T_global] ) )
            * ( local_loop_lma[imom2][nu][t]                              - conj( local_loop_lma[imom ][nu][t]                              ) );


            zv += 
              ( local_loop_lma[imom ][mu][(t+dt)%T_global]                + conj( local_loop_lma[imom2][mu][(t+dt)%T_global]                ) )
            * ( local_loop_stoch_cum[imom2][nu][isample][t]               + conj( local_loop_stoch_cum[imom ][nu][isample][t]               ) );

            zv +=
              ( local_loop_stoch_cum[imom ][mu][isample][(t+dt)%T_global] + conj( local_loop_stoch_cum[imom2][mu][isample][(t+dt)%T_global] ) )
            * ( local_loop_lma[imom2][nu][t]                              + conj( local_loop_lma[imom ][nu][t]                              ) );

          }
          zs /= (double)( T_global * (isample+1) );
          zv /= (double)( T_global * (isample+1) );
          fprintf ( ofs, "%3d%3d%4d%25.16e%25.16e%25.16e%25.16e\n", mu, nu, dt, creal(zs), cimag(zs),  creal(zv), cimag(zv));
        }
      }
      fclose ( ofs );

    }  /* end of loop on samples */

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * stoch x stoch
     ***********************************************************/
   
    /***********************************************************
     * bias
     ***********************************************************/
    double _Complex **local_isoscalar_stoch_bias = NULL, **local_isovector_stoch_bias = NULL;
    if( ( exitstatus = init_2level_zbuffer ( &local_isoscalar_stoch_bias, 16, T_global  ) ) != 0 ) {
      fprintf(stderr, "[loops_local_analysis] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }

    if( ( exitstatus = init_2level_zbuffer ( &local_isovector_stoch_bias, 16, T_global  ) ) != 0 ) {
      fprintf(stderr, "[loops_local_analysis] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }

    /***********************************************************
     * cumulative number of samples
     ***********************************************************/
    for ( int isample = 0; isample < g_nsample; isample ++ ) {
   

      sprintf ( filename, "%s.%s.%.4d.px%.2dpy%.2dpz%.2d.nsample%.4d.%s", g_outfile_prefix, "disc", Nconf, p[0], p[1], p[2], isample+1, "stoch-stoch" );
      FILE *ofs = fopen( filename, "w" );
      if ( ofs == NULL ) {
        fprintf ( stderr, "[loops_local_analysis] Error, could not open file %s for writing\n", filename );
        EXIT(1);
      }

      for ( int mu = 0; mu < 4; mu++ ) {
        int nu = mu;
        for ( int dt = 0; dt < T_global; dt++ ) {
          double _Complex  zs = 0., zv = 0.;
  
          for ( int t = 0; t < T_global; t++ ) {
            zs += 
              ( local_loop_stoch_cum[imom ][mu][isample][(t+dt)%T_global] - conj( local_loop_stoch_cum[imom2][mu][isample][(t+dt)%T_global]) )
            * ( local_loop_stoch_cum[imom2][nu][isample][t]               - conj( local_loop_stoch_cum[imom ][nu][isample][t]              ) );
         
            zv += 
              ( local_loop_stoch_cum[imom ][mu][isample][(t+dt)%T_global] + conj( local_loop_stoch_cum[imom2][mu][isample][(t+dt)%T_global]) )
            * ( local_loop_stoch_cum[imom2][nu][isample][t]               + conj( local_loop_stoch_cum[imom ][nu][isample][t]              ) );
         
            local_isoscalar_stoch_bias[mu][dt] += 
              ( local_loop_stoch_aux[imom ][mu][isample][(t+dt)%T_global] - conj( local_loop_stoch_aux[imom2][mu][isample][(t+dt)%T_global]) )
            * ( local_loop_stoch_aux[imom2][nu][isample][t]               - conj( local_loop_stoch_aux[imom ][nu][isample][t]              ) );
         
            local_isovector_stoch_bias[mu][dt] += 
              ( local_loop_stoch_aux[imom ][mu][isample][(t+dt)%T_global] + conj( local_loop_stoch_aux[imom2][mu][isample][(t+dt)%T_global]) )
            * ( local_loop_stoch_aux[imom2][nu][isample][t]               + conj( local_loop_stoch_aux[imom ][nu][isample][t]              ) );
         
          }  /* end of loop on timeslice */

          /***********************************************************/
          /***********************************************************/

          /***********************************************************
           * subtract the current bias including samples up to isample
           ***********************************************************/
          zs -= local_isoscalar_stoch_bias[mu][dt];
          zv -= local_isovector_stoch_bias[mu][dt];

          /***********************************************************/
          /***********************************************************/

          /***********************************************************
           * normalize subtracted hvp
           ***********************************************************/
          if ( isample > 0 ) {
            zs /= (double)( T_global * ( isample * (isample+1) ) );
            zv /= (double)( T_global * ( isample * (isample+1) ) );
          }

          /***********************************************************/
          /***********************************************************/

          /***********************************************************
           * print to file
           ***********************************************************/
          fprintf ( ofs, "%3d%3d%4d%25.16e%25.16e%25.16e%25.16e\n", mu, nu, dt, creal(zs), cimag(zs),  creal(zv), cimag(zv));
        }
      }
      fclose ( ofs );

    }  /* end of loop on samples */


    fini_2level_zbuffer ( &local_isoscalar_stoch_bias );
    fini_2level_zbuffer ( &local_isovector_stoch_bias );
#if 0
#endif

  } /* end of loop on momenta */


  /***********************************************************/
  /***********************************************************/
 
  fini_3level_zbuffer ( &local_loop_lma );
  fini_4level_zbuffer ( &local_loop_stoch_aux );
  fini_4level_zbuffer ( &local_loop_stoch_cum );

  /***********************************************************/
  /***********************************************************/

  /***********************************************
   * close AFF file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    aff_reader_close (affr);
  }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */


  /***********************************************/
  /***********************************************/

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [loops_local_analysis] %s# [loops_local_analysis] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [loops_local_analysis] %s# [loops_local_analysis] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
