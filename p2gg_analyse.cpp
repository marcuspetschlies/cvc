/****************************************************
 * p2gg_analyse.c
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

#define MAIN_PROGRAM

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
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_i.h"
#include "uwerr.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  double const TWO_MPI = 2. * M_PI;

  int c;
  int filename_set = 0;
  int check_momentum_space_WI = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char key[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "Wh?f:N:S:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'W':
      check_momentum_space_WI = 1;
      fprintf ( stdout, "# [p2gg_analyse] check_momentum_space_WI set to %d\n", check_momentum_space_WI );
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
  /* fprintf(stdout, "# [p2gg_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [p2gg_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [p2gg_analyse] comment %s\n", line );
      continue;
    }

    sscanf( line, "%d %d %d %d %d", 
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf],
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+1,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+2,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+3,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+4 );

    count++;
  }

  fclose ( ofs );


  if ( g_verbose > 5 ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        fprintf ( stdout, "conf_src_list %6d %3d %3d %3d %3d\n", 
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4] );

      }
    }
  }

  /***********************************************************
   ***********************************************************
   **
   ** HVP
   **
   ***********************************************************
   ***********************************************************/
  double ****** hvp = init_6level_dtable ( g_sink_momentum_number, 4, 4, num_conf, num_src_per_conf, 2 * T );
  if ( hvp == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  /***********************************************************
   * loop on configs and source locations per config
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      Nconf = conf_src_list[iconf][isrc][0];

      /***********************************************************
       * copy source coordinates
       ***********************************************************/
      int const gsx[4] = {
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3],
          conf_src_list[iconf][isrc][4] };

      int source_proc_id = -1, sx[4];
      exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
      if( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_analyse] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(123);
      }

#ifdef HAVE_LHPC_AFF
      /***********************************************
       * writer for aff output file
       ***********************************************/
      struct AffNode_s *affn = NULL, *affdir = NULL;

      if(io_proc == 2) {
        sprintf ( filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
        fprintf(stdout, "# [p2gg_analyse] reading data to file %s\n", filename);
        affr = aff_reader ( filename );
        const char * aff_status_str = aff_reader_errstr ( affr );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[p2gg_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
          EXIT(15);
        }

        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[p2gg_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          return(103);
        }

      }  /* end of if io_proc == 2 */
#endif


      /**********************************************************
       * loop on momenta
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

        int const sink_momentum[3] = {
          g_sink_momentum_list[isink_momentum][0],
          g_sink_momentum_list[isink_momentum][1],
          g_sink_momentum_list[isink_momentum][2] };

        sprintf ( key , "/hvp/full/t%.2dx%.2dy%.2dz%.2d/px%.2dpy%.2dpz%.2d",
            gsx[0], gsx[1], gsx[2], gsx[3],
            sink_momentum[0], sink_momentum[1], sink_momentum[2] );

        if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse] key = %s\n", key );

        double *** buffer = init_3level_dtable( 4, 4, 2 * T );
        if( buffer == NULL ) {
          fprintf(stderr, "[p2gg_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(15);
        }


        affdir = aff_reader_chpath (affr, affn, key );
        uint32_t uitems = 16 * T;
        exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[0][0]), uitems );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[p2gg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(105);
        }

        /**********************************************************
         * loop on shifts in directions mu, nu
         **********************************************************/
        for( int mu = 0; mu < 4; mu++) {
        for( int nu = 0; nu < 4; nu++) {

          double const p[4] = {
              0., 
              TWO_MPI * sink_momentum[0] / LX_global,
              TWO_MPI * sink_momentum[1] / LY_global,
              TWO_MPI * sink_momentum[2] / LZ_global };

          double const phase = - ( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ) + 0.5 * ( p[mu] - p[nu] );

          double _Complex ephase = cexp ( phase * I );

          /**********************************************************
           * sort data from buffer into hvp,
           * add source phase
           **********************************************************/
          for ( int it = 0; it < T; it++ ) {
            int const tt = ( it - gsx[0] + T_global ) % T_global; 

            double _Complex ztmp = ( buffer[mu][nu][2*it] +  buffer[mu][nu][2*it+1] * I ) * ephase;

            hvp[isink_momentum][mu][nu][iconf][isrc][2*tt  ] = creal( ztmp );
            hvp[isink_momentum][mu][nu][iconf][isrc][2*tt+1] = cimag( ztmp );
          }

        }  /* end of loop on direction nu */
        }  /* end of loop on direction mu */

        fini_3level_dtable( &buffer );

      }  /* end of loop on sink momenta */
#if 0
#endif  /* if if 0 */

#ifdef HAVE_LHPC_AFF
      if(io_proc == 2) {
        aff_reader_close ( affr );
      }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    }  /* end of loop on source locations */

  }   /* end of loop on configurations */

  /****************************************
   * show all data
   ****************************************/
  if ( g_verbose > 5 ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ )
    {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
      {
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
          for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            for ( int it = 0; it < T; it++ ) {
              fprintf ( stdout, "c %6d s %3d p %3d %3d %3d m %d %d hvp %3d  %25.16e %25.16e\n", iconf, isrc, 
                  g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], mu, nu, it, 
                  hvp[imom][mu][nu][iconf][isrc][2*it], hvp[imom][mu][nu][iconf][isrc][2*it] );
            }
          }}
        }
      }
    }
  }

  /****************************************
   * check WI in momentum space
   ****************************************/
  if ( check_momentum_space_WI ) {

    for ( int iconf = 0; iconf < num_conf; iconf++ ) {

      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        double const wi_eps = 1.e-07;
  
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
  
          for ( int nu = 0; nu < 4; nu++ ) {
  
            double ** hvp_p = init_2level_dtable ( T_global, 8 );
  
            for ( int mu = 0; mu < 4; mu++ ) {
  
              /* FT t -> k */
              for ( int ik = 0; ik < T_global; ik++ ) {
  
                double const p0 = TWO_MPI * ik / T_global;

                for ( int it = 0; it  < T; it++ ) {
                  double const phase = p0 * ( it + ( (mu==0) - (nu==0) ) * 0.5 );
  
                  double const cphase = cos ( phase );
                  double const sphase = sin ( phase );
  
                  double a[2] = { hvp[imom][mu][nu][iconf][isrc][2*it  ], hvp[imom][mu][nu][iconf][isrc][2*it+1] };
  
                  double const b[2] = { cphase, sphase };
  
                  hvp_p[ik][2*mu  ] += a[0] * b[0] - a[1] * b[1];
                  hvp_p[ik][2*mu+1] += a[0] * b[1] + a[1] * b[0];
                }  /* end of loop on timeslices */
  
              }  /* end of loop on momentum component 0 */
  
            }  /* end of loop on mu */
  
            for ( int ik = 0; ik < T_global; ik++ ) {
              double const sinph[4] = {
                2. * sin ( ik * M_PI / T_global ),
                2. * sin ( g_sink_momentum_list[imom][0] * M_PI / LX_global ),
                2. * sin ( g_sink_momentum_list[imom][1] * M_PI / LY_global ),
                2. * sin ( g_sink_momentum_list[imom][2] * M_PI / LZ_global ) };
  
              double const avabsre =  0.25 * ( fabs( hvp_p[ik][0] ) + fabs( hvp_p[ik][2] ) + fabs( hvp_p[ik][4] ) + fabs( hvp_p[ik][6] ) );
              double const avabsim =  0.25 * ( fabs( hvp_p[ik][1] ) + fabs( hvp_p[ik][3] ) + fabs( hvp_p[ik][5] ) + fabs( hvp_p[ik][7] ) );

              double const absnormre =  
                sinph[0] * hvp_p[ik][0] +
                sinph[1] * hvp_p[ik][2] +
                sinph[2] * hvp_p[ik][4] +
                sinph[3] * hvp_p[ik][6];
  
              double const absnormim =
                sinph[0] * hvp_p[ik][1] +
                sinph[1] * hvp_p[ik][3] +
                sinph[2] * hvp_p[ik][5] +
                sinph[3] * hvp_p[ik][7];
  
              double const relnormre = absnormre / avabsre;
              double const relnormim = absnormim / avabsim;

              int okay = ( fabs(relnormre) < wi_eps && fabs(relnormim) < wi_eps ) ? 1 : 0;

              fprintf ( stdout, "# [p2gg_analyse] %d p %3d %3d %3d %3d %12.4e %12.4e  %12.4e %12.4e  ok %d\n", nu, ik,
                  g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], absnormre, absnormim, 
                  relnormre, relnormim, okay );
  
              if ( !okay ) {
                fprintf ( stderr, "[p2gg_analyse] Error in momentum space WI %s %d\n", __FILE__, __LINE__ );
                EXIT(2);
              }
            }  /* end of loop on momentum component 0 */
  
            fini_2level_dtable ( &hvp_p );
  
          }  /* end of loop on nu */

        }  /* end of loop on momenta */
  
      }  /* end of loop on sources per config */
    }  /* end of loop on configs */

  }  /* end of if check_momentum_space_WI */ 

  /****************************************
   * now some statistical analysis
   ****************************************/

  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    int const momentum[3] = {
        g_sink_momentum_list[imom][0],
        g_sink_momentum_list[imom][1],
        g_sink_momentum_list[imom][1] };

    /* for( int mu = 0; mu < 4; mu++) */
    for( int mu = 1; mu < 2; mu++)
    {
    /* for( int nu = 0; nu < 4; nu++) */
    for( int nu = 1; nu < 2; nu++)
    {

      int const Thp1 = T_global / 2 + 1;
      double *** data = init_3level_dtable ( num_conf, num_src_per_conf, 2 * Thp1 );
      double *** res = init_3level_dtable ( 2, T_global, 5 );

      /****************************************
       * fold
       ****************************************/
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
          data[iconf][isrc][0] = hvp[imom][mu][nu][iconf][isrc][0];
          data[iconf][isrc][1] = hvp[imom][mu][nu][iconf][isrc][1];

          for ( int it = 1; it < Thp1-1; it++ ) {
            data[iconf][isrc][2*it  ] = ( hvp[imom][mu][nu][iconf][isrc][2*it  ] + hvp[imom][mu][nu][iconf][isrc][2*(T_global-it)  ] ) * 0.5;
            data[iconf][isrc][2*it+1] = ( hvp[imom][mu][nu][iconf][isrc][2*it+1] + hvp[imom][mu][nu][iconf][isrc][2*(T_global-it)+1] ) * 0.5;

          }

          data[iconf][isrc][2*Thp1-2] = hvp[imom][mu][nu][iconf][isrc][2*Thp1-2];
          data[iconf][isrc][2*Thp1-1] = hvp[imom][mu][nu][iconf][isrc][2*Thp1-1];
        }
      }

      uwerr ustat;
      /****************************************
       * real and imag part
       ****************************************/
      for ( int it = 0; it < 2*T_global; it++ ) {

        uwerr_init ( &ustat );

        ustat.nalpha   = 2 * Thp1;  /* real and imaginary part */
        ustat.nreplica = 1;
        for (  int i = 0; i < ustat.nreplica; i++) ustat.n_r[i] = num_conf * num_src_per_conf / ustat.nreplica;
        ustat.s_tau = 1.5;
        sprintf ( ustat.obsname, "v%dv%d_PX%d_PY%d_PZ%d", mu, nu, momentum[0], momentum[1], momentum[2] );

        ustat.ipo = it + 1;  /* real / imag part : 2*it, shifted by 1 */

        /* uwerr_analysis ( hvp[imom][mu][nu][0][0], &ustat ); */
        uwerr_analysis ( data[0][0], &ustat );

        res[it%2][it/2][0] = ustat.value;
        res[it%2][it/2][1] = ustat.dvalue;
        res[it%2][it/2][2] = ustat.ddvalue;
        res[it%2][it/2][3] = ustat.tauint;
        res[it%2][it/2][4] = ustat.dtauint;

        uwerr_free ( &ustat );
      }  /* end of loop on ipos */

      sprintf ( filename, "v%dv%d_PX%d_PY%d_PZ%d.uwerr", mu, nu, momentum[0], momentum[1], momentum[2] );
      FILE * ofs = fopen ( filename, "w" );

      fprintf ( ofs, "# nalpha   = %llu\n", ustat.nalpha );
      fprintf ( ofs, "# nreplica = %llu\n", ustat.nreplica );
      for (  int i = 0; i < ustat.nreplica; i++) fprintf( ofs, "# nr[%d] = %llu\n", i, ustat.n_r[i] );
      fprintf ( ofs, "#\n" );

      for ( int it = 0; it < Thp1; it++ ) {

        fprintf ( ofs, "%3d %16.7e %16.7e %16.7e %16.7e %16.7e    %16.7e %16.7e %16.7e %16.7e %16.7e\n", it,
            res[0][it][0], res[0][it][1], res[0][it][2], res[0][it][3], res[0][it][4],
            res[1][it][0], res[1][it][1], res[1][it][2], res[1][it][3], res[1][it][4] );
      }


      fclose( ofs );

      fini_3level_dtable ( &res );
      fini_3level_dtable ( &data );
    }}

  }  /* end of loop on momenta */

  /**********************************************************
   * free hvp field
   **********************************************************/
  fini_6level_dtable ( &hvp );

  /**********************************************************
   **********************************************************
   **
   ** P -> gg 3-point function
   **
   **********************************************************
   **********************************************************/
  double ****** pgg = init_6level_dtable ( g_sink_momentum_number, 4, 4, num_conf, num_src_per_conf, 2 * T );
  if ( hvp == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  for ( int iseq_source_momentum = 0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) {

    int const seq_source_momentum[3] = {
      g_seq_source_momentum_list[iseq_source_momentum][0],
      g_seq_source_momentum_list[iseq_source_momentum][1],
      g_seq_source_momentum_list[iseq_source_momentum][2] };

    for( int isequential_source_gamma_id = 0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++) {

      int const sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];

      for ( int isequential_source_timeslice = 0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++) {

        int const sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];

        /***********************************************************
         * loop on configs and source locations per config
         ***********************************************************/
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
      
            Nconf = conf_src_list[iconf][isrc][0];
      
            /***********************************************************
             * copy source coordinates
             ***********************************************************/
            int const gsx[4] = {
                conf_src_list[iconf][isrc][1],
                conf_src_list[iconf][isrc][2],
                conf_src_list[iconf][isrc][3],
                conf_src_list[iconf][isrc][4] };
      
            int source_proc_id = -1, sx[4];
            exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_analyse] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(123);
            }
      
      #ifdef HAVE_LHPC_AFF
            /***********************************************
             * writer for aff output file
             ***********************************************/
            struct AffNode_s *affn = NULL, *affdir = NULL;
      
            if(io_proc == 2) {
              sprintf ( filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
              fprintf(stdout, "# [p2gg_analyse] reading data to file %s\n", filename);
              affr = aff_reader ( filename );
              const char * aff_status_str = aff_reader_errstr ( affr );
              if( aff_status_str != NULL ) {
                fprintf(stderr, "[p2gg_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
                EXIT(15);
              }
      
              if( (affn = aff_reader_root( affr )) == NULL ) {
                fprintf(stderr, "[p2gg_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
                return(103);
              }
      
            }  /* end of if io_proc == 2 */
      #endif
      
            /**********************************************************
             * loop on momenta
             **********************************************************/
            for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {
      
              int const sink_momentum[3] = {
                g_sink_momentum_list[isink_momentum][0],
                g_sink_momentum_list[isink_momentum][1],
                g_sink_momentum_list[isink_momentum][2] };
      
              sprintf ( key , "/p-cvc-cvc/full/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/px%.2dpy%.2dpz%.2d",
                  gsx[0], gsx[1], gsx[2], gsx[3],
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                  sequential_source_gamma_id, sequential_source_timeslice,
                  sink_momentum[0], sink_momentum[1], sink_momentum[2] );
      
              if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse] key = %s\n", key );
      
              double *** buffer = init_3level_dtable( 4, 4, 2 * T );
              if( buffer == NULL ) {
                fprintf(stderr, "[p2gg_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
                EXIT(15);
              }
      
              affdir = aff_reader_chpath (affr, affn, key );
              uint32_t uitems = 16 * T;
              exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[0][0]), uitems );
              if( exitstatus != 0 ) {
                fprintf(stderr, "[p2gg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
      STOPPED HERE
              /**********************************************************
               * loop on shifts in directions mu, nu
               **********************************************************/
              for( int mu = 0; mu < 4; mu++) {
              for( int nu = 0; nu < 4; nu++) {
      
                double const p[4] = {
                    0., 
                    TWO_MPI * sink_momentum[0] / LX_global,
                    TWO_MPI * sink_momentum[1] / LY_global,
                    TWO_MPI * sink_momentum[2] / LZ_global };
      
                double const phase = - ( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ) + 0.5 * ( p[mu] - p[nu] );
      
                double _Complex ephase = cexp ( phase * I );
      
                /**********************************************************
                 * sort data from buffer into hvp,
                 * add source phase
                 **********************************************************/
                for ( int it = 0; it < T; it++ ) {
                  int const tt = ( it - gsx[0] + T_global ) % T_global; 
      
                  double _Complex ztmp = ( buffer[mu][nu][2*it] +  buffer[mu][nu][2*it+1] * I ) * ephase;
      
                  hvp[isink_momentum][mu][nu][iconf][isrc][2*tt  ] = creal( ztmp );
                  hvp[isink_momentum][mu][nu][iconf][isrc][2*tt+1] = cimag( ztmp );
                }
      
              }  /* end of loop on direction nu */
              }  /* end of loop on direction mu */
      
              fini_3level_dtable( &buffer );
      
            }  /* end of loop on sink momenta */
      #if 0
      #endif  /* if if 0 */
      
      #ifdef HAVE_LHPC_AFF
            if(io_proc == 2) {
              aff_reader_close ( affr );
            }  /* end of if io_proc == 2 */
      #endif  /* of ifdef HAVE_LHPC_AFF */
      
          }  /* end of loop on source locations */
      
        }   /* end of loop on configurations */
      
        /****************************************
         * show all data
         ****************************************/
        if ( g_verbose > 5 ) {
          for ( int iconf = 0; iconf < num_conf; iconf++ )
          {
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
            {
              for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
                for ( int mu = 0; mu < 4; mu++ ) {
                for ( int nu = 0; nu < 4; nu++ ) {
                  for ( int it = 0; it < T; it++ ) {
                    fprintf ( stdout, "c %6d s %3d p %3d %3d %3d m %d %d pgg %3d  %25.16e %25.16e\n", iconf, isrc, 
                        g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], mu, nu, it, 
                        pgg[imom][mu][nu][iconf][isrc][2*it], pgg[imom][mu][nu][iconf][isrc][2*it] );
                  }
                }}
              }
            }
          }
        }

        /**********************************************************
         * free p2gg table
         **********************************************************/
        fini_6level_dtable ( &pgg );

      }  /* end of loop on sequential source timeslices */

    }  /* end of loop on sequential source gamma id */

  }  /* end of loop on seq source momentum */

#if 0
#endif  /* if if 0 */

  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/


  fini_3level_itable ( &conf_src_list );

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_analyse] %s# [p2gg_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_analyse] %s# [p2gg_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
