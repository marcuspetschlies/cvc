/****************************************************
 * twopt_uwerr_analyse 
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
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_i.h"
#include "uwerr.h"
#include "derived_quantities.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse cpff fht correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  

  const char flavor_prefix_std[] = "u+-g-u-g";
  const char flavor_prefix_std2[] = "d+-g-d-g";
  const char flavor_prefix_fht[] = "u+-g-suu-g";

  int const gamma_id_to_bin[16] = { 8, 1, 2, 4, 0, 15, 7, 14, 13, 11, 9, 10, 12, 3, 5, 6 };


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cB2.072.64_Nf211";
  int fold_propagator = 0;
  int use_disc = 0;
  int use_conn = 1;
  int use_reim = 0;

  char key[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "dch?f:N:S:F:R:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [twopt_uwerr_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [twopt_uwerr_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [twopt_uwerr_analyse] fold_propagator set to %d\n", fold_propagator );
      break;
    case 'R':
      use_reim = atoi ( optarg );
      fprintf ( stdout, "# [twopt_uwerr_analyse] use_reim set to %d\n", use_reim );
      break;
    case 'd':
      use_disc = 1;
      fprintf ( stdout, "# [twopt_uwerr_analyse] use_disc set to %d\n", use_disc );
      break;
    case 'c':
      use_conn = 1;
      fprintf ( stdout, "# [twopt_uwerr_analyse] use_conn set to %d\n", use_conn );
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
  /* fprintf(stdout, "# [twopt_uwerr_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [twopt_uwerr_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [twopt_uwerr_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [twopt_uwerr_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[twopt_uwerr_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[twopt_uwerr_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[twopt_uwerr_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [twopt_uwerr_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
 

  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[twopt_uwerr_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[twopt_uwerr_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [twopt_uwerr_analyse] comment %s\n", line );
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
        double **** corr_std = init_4level_dtable ( num_conf, num_src_per_conf, 3, 2 * T );
        if ( corr_std == NULL ) {
          fprintf(stderr, "[twopt_uwerr_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(16);
        }

        /***********************************************************
         * loop on configs and source locations per config
         ***********************************************************/
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          Nconf = conf_src_list[iconf][0][0];

          /***********************************************************
           * filename for data file 
           *
           * Cgx Cgx
           ***********************************************************/
          sprintf ( filename, "%s_px%dpy%dpz%d_Cgx_Cgx_fwd_parity1_n%.4d", g_outfile_prefix, 
              sink_momentum[0], sink_momentum[1], sink_momentum[2], Nconf );
          fprintf(stdout, "# [twopt_uwerr_analyse] reading data from file %s\n", filename);
          fflush( stdout );
          FILE *ofs = fopen ( filename, "r" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "[twopt_uwerr_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT(1);
          }

          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            char source_coords_tag[100];
            int itmp;
            fscanf( ofs, "# %s", source_coords_tag );
            /* if ( g_verbose > 4 ) fprintf ( stdout, "# [twopt_uwerr_analyse] source_coords_tag = %s %s %d\n", source_coords_tag, __FILE__, __LINE__ ); */
            for ( int it = 0; it < T; it++ ) {
              fscanf ( ofs, "%d %lf %lf\n", &itmp, corr_std[iconf][isrc][0]+2*it, corr_std[iconf][isrc][0]+2*it+1 );
            }
          }  /* end of loop on sources per configuration */
          fclose ( ofs );

          /***********************************************************
           * filename for data file 
           *
           * Cgy Cgy
           ***********************************************************/
          sprintf ( filename, "%s_px%dpy%dpz%d_Cgy_Cgy_fwd_parity1_n%.4d", g_outfile_prefix, 
              sink_momentum[0], sink_momentum[1], sink_momentum[2], Nconf );
          fprintf(stdout, "# [twopt_uwerr_analyse] reading data from file %s\n", filename);
          ofs = fopen ( filename, "r" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "[twopt_uwerr_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT(1);
          }

          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            char source_coords_tag[100];
            fscanf( ofs, "# %s", source_coords_tag );
            for ( int it = 0; it < T; it++ ) {
              int itmp;
              fscanf ( ofs, "%d %lf %lf\n", &itmp, corr_std[iconf][isrc][1]+2*it, corr_std[iconf][isrc][1]+2*it+1 );
            }
          }  /* end of loop on sources per configuration */
          fclose ( ofs );

          /***********************************************************
           * filename for data file 
           *
           * Cgz Cgz
           ***********************************************************/
          sprintf ( filename, "%s_px%dpy%dpz%d_Cgz_Cgz_fwd_parity1_n%.4d", g_outfile_prefix, 
              sink_momentum[0], sink_momentum[1], sink_momentum[2], Nconf );
          fprintf(stdout, "# [twopt_uwerr_analyse] reading data from file %s\n", filename);
          ofs = fopen ( filename, "r" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "[twopt_uwerr_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT(1);
          }

          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            char source_coords_tag[100];
            fscanf( ofs, "# %s", source_coords_tag );
            for ( int it = 0; it < T; it++ ) {
              int itmp;
              fscanf ( ofs, "%d %lf %lf\n", &itmp, corr_std[iconf][isrc][2]+2*it, corr_std[iconf][isrc][2]+2*it+1 );
            }
          }  /* end of loop on sources per configuration */
          fclose ( ofs );

        }  /* end of loop on configurations */


        /***********************************************************
         * show all data
         ***********************************************************/
        if ( g_verbose > 4 ) {
          sprintf ( filename, "%s.px%dpy%dpz%d.parity1.std", g_outfile_prefix, sink_momentum[0], sink_momentum[1], sink_momentum[2] );

          FILE * ofs = fopen ( filename, "w" );

          for ( int iconf = 0; iconf < num_conf; iconf++ )
          {
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
            {
              for ( int icomp = 0; icomp < 3; icomp++ ) {
                for ( int it = 0; it < T; it++ ) {
                  fprintf ( ofs, "%6d %3d %2d %4d %25.16e %25.16e\n", conf_src_list[iconf][isrc][0],
                      conf_src_list[iconf][isrc][1], icomp,
                      it, corr_std[iconf][isrc][icomp][2*it], corr_std[iconf][isrc][icomp][2*it+1] );
                }
              }  /* end of loop on components */
            }
          }
          fclose ( ofs );
        }  /* end of if verbosity */


        /***********************************************************
         ***********************************************************
         **
         ** statistical analysis for STD correlator
         **
         ***********************************************************
         ***********************************************************/
 
        int Nt = 0;
        if ( fold_propagator == 0 ) {
          Nt = T;   
        } else if ( fold_propagator == 1 ) {
          Nt = T / 2 + 1;
        }
        double **** data = init_4level_dtable ( num_conf, num_src_per_conf, 3, 2 * Nt );
        double *** res = init_3level_dtable ( 2, Nt, 5 );
    
        /***********************************************************
         * fold correlator
         ***********************************************************/
        if ( fold_propagator == 0 ) {
          memcpy ( data[0][0][0], corr_std[0][0][0], 3*num_conf*num_src_per_conf*2*Nt*sizeof(double) );
        } else if ( fold_propagator == 1 ) {
#if 0
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][isrc][0] = corr_std[iconf][isrc][0];
              data[iconf][isrc][1] = corr_std[iconf][isrc][1];
    
              for ( int it = 1; it < Nt-1; it++ ) {
                data[iconf][isrc][2*it  ] = ( corr_std[iconf][isrc][2*it  ] + corr_std[iconf][isrc][2*(T-it)  ] ) * 0.5;
                data[iconf][isrc][2*it+1] = ( corr_std[iconf][isrc][2*it+1] + corr_std[iconf][isrc][2*(T-it)+1] ) * 0.5;
    
              }
    
              data[iconf][isrc][2*Nt-2] = corr_std[iconf][isrc][2*Nt-2];
              data[iconf][isrc][2*Nt-1] = corr_std[iconf][isrc][2*Nt-1];
            }
          }
#endif
        }  /* end of if fold_propagator */
    
        uwerr ustat;
        /***********************************************************
         * real and imag part
         ***********************************************************/
        for ( int it = 0; it < 2*Nt; it++ )
        {
          uwerr_init ( &ustat );
    
          ustat.nalpha   = 2 * Nt;  /* real and imaginary part */
          ustat.nreplica = 1;
          for (  int i = 0; i < ustat.nreplica; i++) ustat.n_r[i] = 3 * num_conf * num_src_per_conf / ustat.nreplica;
          ustat.s_tau = 1.5;
          sprintf ( ustat.obsname, "corr_std" );
    
          ustat.ipo = it + 1;  /* real / imag part : 2*it, shifted by 1 */
    
          exitstatus = uwerr_analysis ( data[0][0][0], &ustat );
          if ( exitstatus == 0 ) {
            res[it%2][it/2][0] = ustat.value;
            res[it%2][it/2][1] = ustat.dvalue;
            res[it%2][it/2][2] = ustat.ddvalue;
            res[it%2][it/2][3] = ustat.tauint;
            res[it%2][it/2][4] = ustat.dtauint;
          } else {
            fprintf ( stderr, "[twopt_uwerr_analyse] Warning return status from uwerr_analysis was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          }
   
          uwerr_free ( &ustat );
        }  /* end of loop on ipos */
    
        sprintf ( filename, "%s_std_PX%d_PY%d_PZ%d.uwerr", g_outfile_prefix,
            sink_momentum[0], sink_momentum[1], sink_momentum[2] );

        FILE * ofs = fopen ( filename, "w" );
    
        fprintf ( ofs, "# nalpha   = %llu\n", ustat.nalpha );
        fprintf ( ofs, "# nreplica = %llu\n", ustat.nreplica );
        for (  int i = 0; i < ustat.nreplica; i++) fprintf( ofs, "# nr[%d] = %llu\n", i, ustat.n_r[i] );
        fprintf ( ofs, "#\n" );
    
        for ( int it = 0; it < Nt; it++ ) {
          fprintf ( ofs, "%3d %16.7e %16.7e %16.7e %16.7e %16.7e    %16.7e %16.7e %16.7e %16.7e %16.7e\n", it,
              res[0][it][0], res[0][it][1], res[0][it][2], res[0][it][3], res[0][it][4],
              res[1][it][0], res[1][it][1], res[1][it][2], res[1][it][3], res[1][it][4] );
        }
    
        fclose( ofs );
        fini_3level_dtable ( &res );
        fini_4level_dtable ( &data );

        /***********************************************************
         ***********************************************************
         **
         ** statistical analysis for log ratio STD
         **
         ***********************************************************
         ***********************************************************/
 
        Nt = 0;
        if ( fold_propagator == 0 ) {
          Nt = T;   
        } else if ( fold_propagator == 1 ) {
          Nt = T / 2 + 1;
        }
        data = init_4level_dtable ( num_conf, num_src_per_conf, 3, 2 * Nt );
        res = init_3level_dtable ( Nt, Nt, 5 );
    
        /***********************************************************
         * fold correlator
         ***********************************************************/
        if ( fold_propagator == 0 ) {
          memcpy ( data[0][0][0], corr_std[0][0][0], 3*num_conf*num_src_per_conf*2*Nt*sizeof(double) );
        } else if ( fold_propagator == 1 ) {
#if 0
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][isrc][0] = corr_std[iconf][isrc][0];
              data[iconf][isrc][1] = corr_std[iconf][isrc][1];
    
              for ( int it = 1; it < Nt-1; it++ ) {
                data[iconf][isrc][2*it  ] = ( corr_std[iconf][isrc][2*it  ] + corr_std[iconf][isrc][2*(T-it)  ] ) * 0.5;
                data[iconf][isrc][2*it+1] = ( corr_std[iconf][isrc][2*it+1] + corr_std[iconf][isrc][2*(T-it)+1] ) * 0.5;
    
              }
    
              data[iconf][isrc][2*Nt-2] = corr_std[iconf][isrc][2*Nt-2];
              data[iconf][isrc][2*Nt-1] = corr_std[iconf][isrc][2*Nt-1];
            }
          }
#endif
        }  /* end of if fold_propagator */
    
        /***********************************************************
         * symmetric acosh ratio
         ***********************************************************/
        for ( int itau = 0; itau < Nt/2; itau++ )
        {
          for ( int it = itau; it < Nt-itau; it++ )
          {
            uwerr_init ( &ustat );
    
            ustat.nalpha   = 2 * Nt;  /* real and imaginary part */
            ustat.nreplica = 1;
            for (  int i = 0; i < ustat.nreplica; i++) ustat.n_r[i] = 3 * num_conf * num_src_per_conf / ustat.nreplica;
            ustat.s_tau = 1.5;
            sprintf ( ustat.obsname, "corr_std_log_ratio_t%d_tau%d", it, itau );
    
            ustat.func  = acosh_ratio;
            ustat.dfunc = dacosh_ratio;
            ustat.para  = init_1level_itable ( 3 );
            ((int*)ustat.para)[0] = 2 * ( it - itau );
            ((int*)ustat.para)[1] = 2 * ( it + itau );
            ((int*)ustat.para)[2] = 2 *   it;

            exitstatus = uwerr_analysis ( data[0][0][0], &ustat );
            if ( exitstatus == 0 ) {
              res[it][itau][0] = ustat.value;
              res[it][itau][1] = ustat.dvalue;
              res[it][itau][2] = ustat.ddvalue;
              res[it][itau][3] = ustat.tauint;
              res[it][itau][4] = ustat.dtauint;
            } else {
              fprintf ( stderr, "[twopt_uwerr_analyse] Warning return status from uwerr_analysis was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            }
   
            uwerr_free ( &ustat );
          }  /* end of loop on tau */
        }  /* end of loop on t */
    
        /* sprintf ( filename, "%s_std_log_ratio_gf%d_gi%d_PX%d_PY%d_PZ%d.uwerr", g_outfile_prefix,
            g_source_gamma_id_list[igf], g_source_gamma_id_list[igi],
            sink_momentum[0], sink_momentum[1], sink_momentum[2] ); */

        sprintf ( filename, "%s_std_acosh_ratio_PX%d_PY%d_PZ%d.uwerr", g_outfile_prefix,
            sink_momentum[0], sink_momentum[1], sink_momentum[2] );

        ofs = fopen ( filename, "w" );
    
        fprintf ( ofs, "# nalpha   = %llu\n", ustat.nalpha );
        fprintf ( ofs, "# nreplica = %llu\n", ustat.nreplica );
        for (  int i = 0; i < ustat.nreplica; i++) fprintf( ofs, "# nr[%d] = %llu\n", i, ustat.n_r[i] );
        fprintf ( ofs, "#\n" );
    
        for ( int itau = 0; itau < Nt/2; itau++ )
        {
          for ( int it = itau; it < Nt-itau; it++ )
          {
            fprintf ( ofs, "%3d %3d %16.7e %16.7e %16.7e %16.7e %16.7e\n", it, itau,
                res[it][itau][0], res[it][itau][1], res[it][itau][2], res[it][itau][3], res[it][itau][4] );
          }
        }
    
        fclose( ofs );
        fini_3level_dtable ( &res );
        fini_4level_dtable ( &data );
#if 0
#endif  /* of if 0 */

        /**********************************************************
         * free corr_std field
         **********************************************************/
        fini_4level_dtable ( &corr_std );

      }  /* end of loop on sink momenta */


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
    fprintf(stdout, "# [twopt_uwerr_analyse] %s# [twopt_uwerr_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [twopt_uwerr_analyse] %s# [twopt_uwerr_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);
}
