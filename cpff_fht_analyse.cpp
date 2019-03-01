/****************************************************
 * cpff_fht_analyse 
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
  fprintf(stdout, "Code to analyse cpff fht correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";

  char key[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [cpff_fht_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [cpff_fht_analyse] number of sources per config = %d\n", num_src_per_conf );
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
  /* fprintf(stdout, "# [cpff_fht_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [cpff_fht_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [cpff_fht_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [cpff_fht_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[cpff_fht_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[cpff_fht_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[cpff_fht_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [cpff_fht_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[cpff_fht_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[cpff_fht_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [cpff_fht_analyse] comment %s\n", line );
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


  if ( g_verbose > 4 ) {
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


  /**********************************************************
   * loop on gamma at sink
   **********************************************************/
  for ( int igf = 0; igf < g_source_gamma_id_number; igf++ ) {

    /**********************************************************
     * loop on gamma at sink
     **********************************************************/
    for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) {

      /**********************************************************
       * loop on momenta
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

        int const sink_momentum[3] = {
          g_sink_momentum_list[isink_momentum][0],
          g_sink_momentum_list[isink_momentum][1],
          g_sink_momentum_list[isink_momentum][2] };
 
        int const source_momentum[3] = {
          -sink_momentum[0],
          -sink_momentum[1],
          -sink_momentum[2] };

        /***********************************************************
         * allocate corr_std
         ***********************************************************/
        double *** corr_std = init_3level_dtable ( num_conf, num_src_per_conf, 2 * T );
        if ( corr_std == NULL ) {
          fprintf(stderr, "[cpff_fht_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
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

            int const gts = gsx[0];
            int source_timeslice = -1, source_proc_id = -1;

            exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
            if( exitstatus != 0 ) {
              fprintf(stderr, "[cpff_fht_analyse] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(123);
            }

#ifdef HAVE_HDF5
            /***********************************************************
             * filename for data file
             ***********************************************************/
            if(io_proc == 2) {
              sprintf ( filename, "%s.%.4d.t%d.h5", g_outfile_prefix, Nconf, gts );
              fprintf(stdout, "# [cpff_fht_analyse] reading data from file %s\n", filename);
            }  /* end of if io_proc == 2 */
#endif

            sprintf ( key , "/d+-g-d-g/std/t%d/s0/gf%d/gi%d/pix%dpiy%dpiz%d/px%dpy%dpz%d",
                gts, g_source_gamma_id_list[igf], g_source_gamma_id_list[igi],
                source_momentum[0], source_momentum[1], source_momentum[2],
                sink_momentum[0], sink_momentum[1], sink_momentum[2] );
            if ( g_verbose > 2 ) fprintf ( stdout, "# [cpff_fht_analyse] key = %s\n", key );

            double * buffer = init_1level_dtable ( 2*T );

            exitstatus = read_from_h5_file ( (void *)buffer, filename, key, io_proc );
            if( exitstatus != 0 ) {
              fprintf(stderr, "[cpff_fht_analyse] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }

            for ( int it = 0; it < T; it++ ) {
              corr_std[iconf][isrc][2*it  ] = buffer[2*( ( it + gts ) % T)  ];
              corr_std[iconf][isrc][2*it+1] = buffer[2*( ( it + gts ) % T)+1];
            }

            fini_1level_dtable ( &buffer );

          }  /* end of loop on sources per configuration */
        }  /* end of loop on configurations */


        /****************************************
         * show all data
         ****************************************/
        if ( g_verbose > 4 ) {
          sprintf ( filename, "d+-g-d-g_s0_gf%d_gi%d_pfx%.2dpfy%.2dpfz%.2d_pix%.2dpiy%.2dpiz%.2d.std",
              g_source_gamma_id_list[igf],
              g_source_gamma_id_list[igi],
              sink_momentum[0], sink_momentum[1], sink_momentum[2],
              source_momentum[0], source_momentum[1], source_momentum[2] );

          FILE * ofs = fopen ( filename, "w" );

          for ( int iconf = 0; iconf < num_conf; iconf++ )
          {
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
            {
              for ( int it = 0; it < T; it++ ) {
                fprintf ( ofs, "%6d %3d %2d %25.16e %25.16e\n", conf_src_list[iconf][isrc][0],
                    conf_src_list[iconf][isrc][1],
                    it, corr_std[iconf][isrc][2*it], corr_std[iconf][isrc][2*it+1] );
              }
            }
          }
          fclose ( ofs );
        }  /* end of if verbosity */

        /****************************************
         * statistical analysis
         ****************************************/
    
        int const Thp1 = T / 2 + 1;
        double *** data = init_3level_dtable ( num_conf, num_src_per_conf, 2 * Thp1 );
        double *** res = init_3level_dtable ( 2, Thp1, 5 );
    
        /****************************************
         * fold correlator
         ****************************************/
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            data[iconf][isrc][0] = corr_std[iconf][isrc][0];
            data[iconf][isrc][1] = corr_std[iconf][isrc][1];
    
            for ( int it = 1; it < Thp1-1; it++ ) {
              data[iconf][isrc][2*it  ] = ( corr_std[iconf][isrc][2*it  ] + corr_std[iconf][isrc][2*(T-it)  ] ) * 0.5;
              data[iconf][isrc][2*it+1] = ( corr_std[iconf][isrc][2*it+1] + corr_std[iconf][isrc][2*(T-it)+1] ) * 0.5;
    
            }
    
            data[iconf][isrc][2*Thp1-2] = corr_std[iconf][isrc][2*Thp1-2];
            data[iconf][isrc][2*Thp1-1] = corr_std[iconf][isrc][2*Thp1-1];
          }
        }
    
        uwerr ustat;
        /***********************************************************
         * real and imag part
         ***********************************************************/
        /* for ( int it = 0; it < 2*Thp1; it++ ) */
        for ( int it = 0; it < 2*Thp1; it+=2 )
        {
    
          uwerr_init ( &ustat );
    
          ustat.nalpha   = 2 * Thp1;  /* real and imaginary part */
          ustat.nreplica = 1;
          for (  int i = 0; i < ustat.nreplica; i++) ustat.n_r[i] = num_conf * num_src_per_conf / ustat.nreplica;
          ustat.s_tau = 1.5;
          sprintf ( ustat.obsname, "corr_std" );
    
          ustat.ipo = it + 1;  /* real / imag part : 2*it, shifted by 1 */
    
          uwerr_analysis ( data[0][0], &ustat );
    
          res[it%2][it/2][0] = ustat.value;
          res[it%2][it/2][1] = ustat.dvalue;
          res[it%2][it/2][2] = ustat.ddvalue;
          res[it%2][it/2][3] = ustat.tauint;
          res[it%2][it/2][4] = ustat.dtauint;
   
          uwerr_free ( &ustat );
        }  /* end of loop on ipos */
    
        sprintf ( filename, "%s_std_gf%d_gi%d_PX%d_PY%d_PZ%d.uwerr", g_outfile_prefix,
            g_source_gamma_id_list[igf], g_source_gamma_id_list[igi],
            sink_momentum[0], sink_momentum[1], sink_momentum[2] );

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

        /***********************************************************/
        /***********************************************************/

        /***********************************************************
         * loop on current gamma
         ***********************************************************/
        for ( int igc = 0; igc < g_sequential_source_gamma_id_number; igc++ ) {

          /***********************************************************
           * loop on sequential source momenta
           ***********************************************************/
          for ( int iseq_momentum = 0; iseq_momentum < g_seq_source_momentum_number; iseq_momentum++ ) {

            int const seq_source_momentum[3] = {
              g_source_momentum_list[iseq_momentum][0],
              g_source_momentum_list[iseq_momentum][1],
              g_source_momentum_list[iseq_momentum][2] };

            int const source_momentum[3] = {
              - ( sink_momentum[0] + seq_source_momentum[0] ),
              - ( sink_momentum[1] + seq_source_momentum[1] ),
              - ( sink_momentum[2] + seq_source_momentum[2] ) };

            double *** corr_fht = init_3level_dtable ( num_conf, num_src_per_conf, 2 * T );
            if ( corr_fht == NULL ) {
              fprintf(stderr, "[cpff_fht_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
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

                int const gts = gsx[0];
                int source_timeslice = -1, source_proc_id = -1;

                exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[cpff_fht_analyse] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(123);
                }

#ifdef HAVE_HDF5
                /***********************************************************
                 * filename for data file
                 ***********************************************************/
                if(io_proc == 2) {
                  sprintf ( filename, "%s.%.4d.t%d.h5", g_outfile_prefix, Nconf, gts );
                  fprintf(stdout, "# [cpff_fht_analyse] reading data from file %s\n", filename);
                }  /* end of if io_proc == 2 */
#endif

                sprintf ( key , "/d+-g-d-g/fht/t%d/s0/gf%d/gc%d/pcx%dpcy%dpcz%d/gi%d/pix%dpiy%dpiz%d/px%dpy%dpz%d",
                    gts, g_source_gamma_id_list[igf],
                    g_sequential_source_gamma_id_list[igc], seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                    g_source_gamma_id_list[igi],            source_momentum[0],     source_momentum[1],     source_momentum[2],
                    sink_momentum[0], sink_momentum[1], sink_momentum[2] );
                if ( g_verbose > 2 ) fprintf ( stdout, "# [cpff_fht_analyse] key = %s\n", key );

                double * buffer = init_1level_dtable ( 2*T );

                exitstatus = read_from_h5_file ( (void*)buffer, filename, key, io_proc );
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[cpff_fht_analyse] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

                for ( int it = 0; it < T; it++ ) {
                  corr_fht[iconf][isrc][2*it  ] = buffer[2*( ( it + gts ) % T)  ];
                  corr_fht[iconf][isrc][2*it+1] = buffer[2*( ( it + gts ) % T)+1];
                }

                fini_1level_dtable ( &buffer );

              }  /* end of loop on sources per configuration */
            }  /* end of loop on configurations */

            /****************************************
             * show all data
             ****************************************/
            if ( g_verbose > 4 ) {
              sprintf ( filename, "d+-g-d-g_s0_gf%d_gc%d_gi%d_pfx%.2dpfy%.2dpfz%.2d_pcx%.2dpcy%.2dpcz%.2d_pix%.2dpiy%.2dpiz%.2d.fht",
                  g_source_gamma_id_list[igf],
                  g_sequential_source_gamma_id_list[igc],
                  g_source_gamma_id_list[igi],
                  sink_momentum[0], sink_momentum[1], sink_momentum[2],
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                  source_momentum[0], source_momentum[1], source_momentum[2] );

              FILE * ofs = fopen ( filename, "w" );

              for ( int iconf = 0; iconf < num_conf; iconf++ )
              {
                for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
                {
                  for ( int it = 0; it < T; it++ ) {
                    fprintf ( ofs, "%6d %3d %2d %25.16e %25.16e\n", conf_src_list[iconf][isrc][0],
                        conf_src_list[iconf][isrc][1],
                        it, corr_fht[iconf][isrc][2*it], corr_fht[iconf][isrc][2*it+1] );
                  }
                }
              }
              fclose ( ofs );
            }  /* end of if verbosity */

            /****************************************
             * statistical analysis
             ****************************************/
        
            int const Thp1 = T / 2 + 1;
            double *** data = init_3level_dtable ( num_conf, num_src_per_conf, 2 * Thp1 );
            double *** res = init_3level_dtable ( 2, Thp1, 5 );
        
            /****************************************
             * fold correlator
             ****************************************/
            for ( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                data[iconf][isrc][0] = corr_fht[iconf][isrc][0];
                data[iconf][isrc][1] = corr_fht[iconf][isrc][1];
        
                for ( int it = 1; it < Thp1-1; it++ ) {
                  data[iconf][isrc][2*it  ] = ( corr_fht[iconf][isrc][2*it  ] + corr_fht[iconf][isrc][2*(T-it)  ] ) * 0.5;
                  data[iconf][isrc][2*it+1] = ( corr_fht[iconf][isrc][2*it+1] + corr_fht[iconf][isrc][2*(T-it)+1] ) * 0.5;
        
                }
        
                data[iconf][isrc][2*Thp1-2] = corr_fht[iconf][isrc][2*Thp1-2];
                data[iconf][isrc][2*Thp1-1] = corr_fht[iconf][isrc][2*Thp1-1];
              }
            }
        
            uwerr ustat;
            /***********************************************************
             * real and imag part
             ***********************************************************/
            for ( int it = 0; it < 2*Thp1; it++ )
            {
        
              uwerr_init ( &ustat );
        
              ustat.nalpha   = 2 * Thp1;  /* real and imaginary part */
              ustat.nreplica = 1;
              for (  int i = 0; i < ustat.nreplica; i++) ustat.n_r[i] = num_conf * num_src_per_conf / ustat.nreplica;
              ustat.s_tau = 1.5;
              sprintf ( ustat.obsname, "corr_fht" );
        
              ustat.ipo = it + 1;  /* real / imag part : 2*it, shifted by 1 */
        
              uwerr_analysis ( data[0][0], &ustat );
        
              res[it%2][it/2][0] = ustat.value;
              res[it%2][it/2][1] = ustat.dvalue;
              res[it%2][it/2][2] = ustat.ddvalue;
              res[it%2][it/2][3] = ustat.tauint;
              res[it%2][it/2][4] = ustat.dtauint;
       
              uwerr_free ( &ustat );
            }  /* end of loop on ipos */
        
            sprintf ( filename, "%s_fht_gf%d_gi%d_PX%d_PY%d_PZ%d.uwerr", g_outfile_prefix,
                g_source_gamma_id_list[igf], g_source_gamma_id_list[igi],
                sink_momentum[0], sink_momentum[1], sink_momentum[2] );
    
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
    
            /***********************************************************/
            /***********************************************************/

            fini_3level_dtable ( &corr_fht );

          }  /* end of loop on sequential source momenta */

        }  /* end of loop on sequential source gamma */


        fini_3level_dtable ( &corr_std );

      }  /* end of loop on sink momenta */

    }  /* end of loop on gamma at source */

  }  /* end of loop on gamma at sink */

#if 0
#endif

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
    fprintf(stdout, "# [cpff_fht_analyse] %s# [cpff_fht_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [cpff_fht_analyse] %s# [cpff_fht_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
