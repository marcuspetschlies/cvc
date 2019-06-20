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

  char const reim_str[2][3] = { "re", "im" };

  char const gamma_id_to_Cg_ascii[16][10] = {
    "Cgy",
    "Cgzg5",
    "Cgt",
    "Cgxg5",
    "Cgygt",
    "Cgyg5gt",
    "Cgyg5",
    "Cgz",
    "Cg5gt",
    "Cgx",
    "Cgzg5gt",
    "C",
    "Cgxg5gt",
    "Cgxgt",
    "Cg5",
    "Cgzgt"
  };

  char const fbwd_str[2][4] = { "fwd", "bwd" };

  const int gamma_f1_number            = 3;
  /*                                       C gx C gy C gz */
  int gamma_f1_list[gamma_f1_number]   = {    9,   0,   7 };



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
        double ***** corr_std = init_5level_dtable ( num_conf, num_src_per_conf, 2, 3, 2 * T_global );
        if ( corr_std == NULL ) {
          fprintf(stderr, "[twopt_uwerr_analyse] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(16);
        }

        /***********************************************************
         * loop on configs and source locations per config
         ***********************************************************/
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          Nconf = conf_src_list[iconf][0][0];

          for ( int ifbwd = 0; ifbwd < 2; ifbwd++ ) {

            for ( int if1 = 0; if1 < 3; if1++ ) {

              /***********************************************************
               * filename for data file 
               ***********************************************************/
              sprintf ( filename, "%s_px%dpy%dpz%d_%s_%s_%s_parity%d_n%.4d", g_outfile_prefix, 
                  sink_momentum[0], sink_momentum[1], sink_momentum[2],
                  gamma_id_to_Cg_ascii[gamma_f1_list[if1]], gamma_id_to_Cg_ascii[gamma_f1_list[if1]],
                  fbwd_str[ifbwd], 1 - 2 * ifbwd, Nconf );

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
                for ( int it = 0; it < T_global; it++ ) {
                  fscanf ( ofs, "%d %lf %lf\n", &itmp, corr_std[iconf][isrc][ifbwd][if1]+2*it, corr_std[iconf][isrc][ifbwd][if1]+2*it+1 );
                }
              }  /* end of loop on sources per configuration */
 
              fclose ( ofs );

            }  /* end of loop on f1 gamma */

          }  /* end of loop on fwd, bwd */

        }  /* end of loop on configurations */

        /***********************************************************
         * show all data
         ***********************************************************/
        if ( g_verbose > 4 ) {
          sprintf ( filename, "%s.px%dpy%dpz%d.std", g_outfile_prefix, sink_momentum[0], sink_momentum[1], sink_momentum[2] );

          FILE * ofs = fopen ( filename, "w" );

          for ( int iconf = 0; iconf < num_conf; iconf++ )
          {
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
            {
              for ( int ifbwd = 0; ifbwd < 2; ifbwd++ )
              {
                for ( int icomp = 0; icomp < 3; icomp++ )
                {
                  for ( int it = 0; it < T_global; it++ ) {
                    fprintf ( ofs, "%6d %3d %s %2d %2d    %4d %25.16e %25.16e\n", 
                        conf_src_list[iconf][isrc][0],
                        conf_src_list[iconf][isrc][1], 
                        fbwd_str[ifbwd], 
                        1 - 2*ifbwd, 
                        icomp,
                        it, 
                        corr_std[iconf][isrc][ifbwd][icomp][2*it], corr_std[iconf][isrc][ifbwd][icomp][2*it+1] );
                  }
                }  /* end of loop on components */
              }
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
 
        for ( int ifbwd = 0; ifbwd < 2; ifbwd++ )
        {
          for ( int icomp = 0; icomp < 3; icomp++ )
          {
            for ( int ireim = 0; ireim < 2; ireim++ ) {

              double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T_global );
 
#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                  for ( int it = 0; it < T_global; it++ ) { 
                    data[iconf][isrc][it] = corr_std[iconf][isrc][ifbwd][icomp][2*it + ireim];
               }}}

              char obsname[200];
              sprintf ( obsname, "%s.%s_%s.%s.p%d.PX%d_PY%d_PZ%d.%s", g_outfile_prefix,
                  gamma_id_to_Cg_ascii[gamma_f1_list[icomp]], gamma_id_to_Cg_ascii[gamma_f1_list[icomp]], fbwd_str[ifbwd], 1 - 2 * ifbwd,
                  sink_momentum[0], sink_momentum[1], sink_momentum[2], reim_str[ireim] );

              exitstatus = apply_uwerr_real ( data[0][0], num_conf*num_src_per_conf, T_global, 0, 1, obsname );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[twopt_uwerr_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }

              fini_3level_dtable ( &data );

            }  /* end of loop on reim */
          }  /* end of loop on components */
        } /* end of loop on fbwd */

        /***********************************************************
         ***********************************************************
         **
         ** statistical analysis for log ratio STD
         **
         ***********************************************************
         ***********************************************************/
        for ( int ireim = 0; ireim < 2; ireim++ ) {
 
          double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T_global );

          /***********************************************************
           * fold correlator
           ***********************************************************/
#pragma omp paralle for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              for ( int it = 0; it < T_global; it++ ) {
                data[iconf][isrc][it] = 0.;
                for ( int icomp = 0; icomp < 3; icomp++ ) {
                  data[iconf][isrc][it] += (
                    corr_std[iconf][isrc][0][icomp][2 * it + ireim] + corr_std[iconf][isrc][1][icomp][2 * ( (T_global - it) % T_global ) + ireim]
                  );
                }
                data[iconf][isrc][it] /= 6.;
              }
            }
          }

          sprintf ( filename, "%s.avg.%s.p%d.PX%d_PY%d_PZ%d.%s.dat", g_outfile_prefix, fbwd_str[0], +1, sink_momentum[0], sink_momentum[1], sink_momentum[2], reim_str[ireim]);
          FILE * fdat = fopen ( filename, "w" );
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              for ( int it = 0; it < T_global; it++ ) {
                fprintf ( fdat, "%3d %25.16e\n", it, data[iconf][isrc][it] );
              }
            }
          }
          fclose ( fdat );

          char obsname[200];
          sprintf ( obsname, "%s.avg.%s.p%d.PX%d_PY%d_PZ%d.%s", g_outfile_prefix, fbwd_str[0], +1, sink_momentum[0], sink_momentum[1], sink_momentum[2], reim_str[ireim] );

          exitstatus = apply_uwerr_real ( data[0][0], num_conf*num_src_per_conf, T_global, 0, 1, obsname );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[twopt_uwerr_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

          if ( ireim == 0) {
            /***********************************************************
             * symmetric acosh ratio
             ***********************************************************/

            for ( int itau = 1; itau < T_global / 4; itau++ )
            {

              char obsname[200];
              /* sprintf ( obsname, "%s.avg.acoshratio.tau%d.PX%d_PY%d_PZ%d.%s", g_outfile_prefix, itau, sink_momentum[0], sink_momentum[1], sink_momentum[2], reim_str[ireim] ); */
              sprintf ( obsname, "%s.avg.logratio.tau%d.PX%d_PY%d_PZ%d.%s", g_outfile_prefix, itau, sink_momentum[0], sink_momentum[1], sink_momentum[2], reim_str[ireim] );

              /* int arg_first[3]  = { 0, 2*itau, itau };
              int arg_stride[3] = {1, 1, 1};
              */
              int arg_first[2]  = { 0, itau };
              int arg_stride[2] = {1, 1 };

              /* exitstatus = apply_uwerr_func ( data[0][0], num_conf*num_src_per_conf, T_global, T_global/2-itau, 3, arg_first, arg_stride, obsname, acosh_ratio, dacosh_ratio ); */
              exitstatus = apply_uwerr_func ( data[0][0], num_conf*num_src_per_conf, T_global, T_global/2-itau, 2, arg_first, arg_stride, obsname, log_ratio_1_1, dlog_ratio_1_1 );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[twopt_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }
            }

          }  /* end of if reim == 0 */

          fini_3level_dtable ( &data );

        }  /* end of loop on reim */

#if 0
#endif  /* of if 0 */

        /**********************************************************
         * free corr_std field
         **********************************************************/
        fini_5level_dtable ( &corr_std );

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
