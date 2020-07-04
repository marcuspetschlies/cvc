/****************************************************
 * htpp_analyse
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
#include "contract_diagrams.h"
#include "twopoint_function_utils.h"
#include "gamma.h"
#include "uwerr.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code form 2-pt function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  char const gamma_bin_to_name[16][8] = { "id", "gx", "gy", "gxgy", "gz", "gxgz", "gygz", "gtg5", "gt", "gxgt", "gygt", "gzg5", "gzgt", "gyg5", "gxg5", "g5" };

  int const gamma_parity_sign[16] = {       1,   -1,   -1,      1,   -1,      1,      1,     -1,    1,     -1,     -1,      1,     -1,      1,      1,   -1 };

  int const gamma_chargeconjugation_sign[16] = {
                                            1,   -1,   -1,     -1,   -1,     -1,     -1,      1,   -1,     -1,     -1,      1,     -1,      1,      1,    1 };

  int const gamma_g5herm_sign[16] = {       1,   -1,   -1,     -1,   -1,     -1,     -1,      1,   -1,     -1,     -1,      1,     -1,      1,      1,    1 };

  char const reim_str[2][3]  = { "re", "im" };
  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char ensemble_name[100] = "NA";
  char filename[100];
  int num_conf = 0, num_src_per_conf = 0;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:E:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [htpp_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [htpp_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [htpp_analyse] ensemble name set to %s\n", ensemble_name );
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
  if(filename_set==0) strcpy(filename, "twopt.input");
  /* fprintf(stdout, "# [htpp_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [htpp_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [htpp_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [htpp_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[htpp_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[htpp_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[htpp_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [htpp_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[htpp_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[htpp_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [htpp_analyse] comment %s\n", line );
      continue;
    }


    /***********************************************************
     * QLUA source coords files have ordering 
     * stream conf x y z t
     ***********************************************************/
    char streamc;
    sscanf( line, "%c %d %d %d %d %d",
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf],
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+1,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+2,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+3,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+4,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+5 );

    count++;
  }

  fclose ( ofs );

  /***********************************************************
   * show all configs and source locations
   ***********************************************************/
  if ( g_verbose > 4 ) {
    fprintf ( stdout, "# [htpp_analyse] conf_src_list conf t x y z\n" );
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        fprintf ( stdout, "%c  %6d %3d %3d %3d %3d\n", 
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] );
      }
    }
  }

  /***********************************************************
   * field to store all data
   ***********************************************************/
  int const n_tc = g_src_snk_time_separation + 1;
  int const num_meas_per_conf = num_src_per_conf * g_coherent_source_number;
  int const num_meas = num_conf * num_meas_per_conf;

  double ****** corr = init_6level_dtable (g_twopoint_function_number, g_sink_momentum_number, g_source_momentum_number, num_conf, num_meas_per_conf, 2 * n_tc );
  if ( corr == NULL ) {
    fprintf( stderr, "[htpp_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(2);
  }
     
  /***********************************************************
   * loop on configs 
   ***********************************************************/
  for( int iconf = 0; iconf < num_conf; iconf++ ) {
          
    /***********************************************************
     * loop on sources per config
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

      char const stream = conf_src_list[iconf][isrc][0];
      int const Nconf   = conf_src_list[iconf][isrc][1];
      int const t_base  = conf_src_list[iconf][isrc][2];

      /***********************************************************
       * store the source coordinates
       ***********************************************************/
      int const gsx[4] = {
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] };

#ifdef HAVE_LHPC_AFF
      /***********************************************************
       * reader for aff output file
       ***********************************************************/
      sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.aff", filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
   
      struct AffReader_s * affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[htpp_analyse] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
        EXIT(5);
      } else {
        if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_analyse] Reading data from file %s\n", filename );
      }
#endif

      /***************************************************************************
       * loop on twopoint functions
       ***************************************************************************/
      for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {

        twopoint_function_type * tp = &(g_twopoint_function_list[i_2pt]);

        twopoint_function_allocate ( tp );

        /***********************************************************
         * loop on sink momenta
         ***********************************************************/
        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

          /***********************************************************
           * loop on source momenta
           ***********************************************************/
          for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

            /***********************************************************
             * add some filter if wanted
             ***********************************************************/
            
            /***********************************************************
             * loop on parity
             ***********************************************************/
            for ( int iparity = 0; iparity < 2; iparity++ ) {
              int const sparity = 1 - 2 * iparity;

              int const parity_sign = gamma_parity_sign[tp->gi1[0]] * gamma_parity_sign[tp->gf1[0]] * gamma_parity_sign[tp->gf2];

              int const charge_conjugation_sign = gamma_chargeconjugation_sign[tp->gi1[0]] * gamma_chargeconjugation_sign[tp->gf1[0]] * gamma_chargeconjugation_sign[tp->gf2];

              int const g5herm_sign = gamma_g5herm_sign[tp->gi1[0]] * gamma_g5herm_sign[tp->gf1[0]] * gamma_g5herm_sign[tp->gf2]; 

              if ( g_verbose > 2 ) fprintf( stdout, "# [htpp_analyse] parity_sign = %d; charge_conjugation_sign = %d; g5herm_sign = %d\n", parity_sign, charge_conjugation_sign, g5herm_sign ); 

              int pf[3] = {
                sparity * g_sink_momentum_list[ipf][0],
                sparity * g_sink_momentum_list[ipf][1],
                sparity * g_sink_momentum_list[ipf][2] };

              int pi[3] = {
                sparity * g_source_momentum_list[ipi][0],
                sparity * g_source_momentum_list[ipi][1],
                sparity * g_source_momentum_list[ipi][2] };

              int pc[3] = {
                  -( pi[0] + pf[0] ),
                  -( pi[1] + pf[1] ),
                  -( pi[2] + pf[2] ) };

              double const amp_re_factor = 0.25 * ( iparity == 0 ? ( 1 + parity_sign * charge_conjugation_sign * g5herm_sign ) : ( parity_sign + charge_conjugation_sign * g5herm_sign) );
              double const amp_im_factor = 0.25 * ( iparity == 0 ? ( 1 - parity_sign * charge_conjugation_sign * g5herm_sign ) : ( parity_sign - charge_conjugation_sign * g5herm_sign) );

              /***********************************************************
               * loop on diagrams
               ***********************************************************/
              for ( int i_diag = 0; i_diag < tp->n; i_diag++ ) {

                char diagram_name[500];
                char key[500];

                twopoint_function_get_diagram_name ( diagram_name,  tp, i_diag );
        
                if ( strcmp ( tp->type , "m-j-m" ) == 0 ) {
                  sprintf ( key, "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/g1_%s/g2_%s/PX%d_PY%d_PZ%d",
                      diagram_name,
                      /* tp->pf1[0], tp->pf1[1], tp->pf1[2], */
                      pf[0], pf[1], pf[2],
                      gamma_bin_to_name[tp->gf1[0]], g_src_snk_time_separation,
                      gamma_bin_to_name[tp->gf2], gamma_bin_to_name[tp->gi1[0]],
                      /* tp->pf2[0], tp->pf2[1], tp->pf2[2]  */
                      pc[0], pc[1], pc[2] );
                } else {
                  continue;
                }

                if ( g_verbose > 2 ) {
                  fprintf ( stdout, "# [htpp_analyse] key = %s %s %d\n", key , __FILE__, __LINE__ );
                }

                exitstatus = read_aff_contraction ( (void*)(tp->c[i_diag][0][0]), affr, NULL, key, T_global * tp->d * tp->d );
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[htpp_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(12);
                }

                for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {
 
                  /* coherent source timeslice */
                  int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * icoh ) % T_global;
  
                  int const csx[4] = { t_coherent ,
                               ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh) % LX_global,
                               ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh) % LY_global,
                               ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh) % LZ_global };

                  /***********************************************************
                   * source phase factor
                   ***********************************************************/
                  double _Complex const ephase = cexp ( 2. * M_PI * ( 
                        pi[0] * csx[1] / (double)LX_global 
                      + pi[1] * csx[2] / (double)LY_global 
                      + pi[2] * csx[3] / (double)LZ_global ) * I );
            
                  if ( g_verbose > 4 ) fprintf ( stdout, "# [htpp_analyse] pi1 = %3d %3d %3d csx = %3d %3d %3d  ephase = %16.7e %16.7e\n",
                      pi[0], pi[1], pi[2],
                      csx[1], csx[2], csx[3],
                      creal( ephase ), cimag( ephase ) );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                  for ( int it = 0; it < n_tc; it++ ) {
                    /* order from source */
                    int const tt = ( csx[0] + it ) % tp->T; 
                    double _Complex const zbuffer = tp->c[i_diag][tt][0][0] * ephase;
              
                    corr[i_2pt][ipf][ipi][iconf][isrc * g_coherent_source_number + icoh][2*it  ] += amp_re_factor * creal ( zbuffer );
                    corr[i_2pt][ipf][ipi][iconf][isrc * g_coherent_source_number + icoh][2*it+1] += amp_im_factor * cimag ( zbuffer );
                  }

                }  /* end of loop on coherent sources */
              }  /* end of loop on diagrams */
            }  /* end of loop on parity */
          }  /* end of loop on source momenta */
        }  /* end of loop on sink momenta */

        twopoint_function_fini ( tp );
      }  /* end of loop on 2pt functions */

#ifdef HAVE_LHPC_AFF
      aff_reader_close ( affr );
#endif
    }  /* end of loop on base sources */

  }  /* end of loop on configs */
   
  /***********************************************************
   * write data to ascii file
   ***********************************************************/
  for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {

    twopoint_function_type * tp = &(g_twopoint_function_list[i_2pt]);

    for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
      for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

        int pc[3] = {
          -( g_sink_momentum_list[ipf][0] + g_source_momentum_list[ipf][0] ), 
          -( g_sink_momentum_list[ipf][1] + g_source_momentum_list[ipf][1] ), 
          -( g_sink_momentum_list[ipf][2] + g_source_momentum_list[ipf][2] ) };

        char output_filename[400];

        if ( strcmp ( tp->type , "m-j-m" ) == 0 ) {

          sprintf ( output_filename, "%s_pfx%dpfy%dpfz%d_gf_%s_dt%d_g1_%s_g2_%s_PX%d_PY%d_PZ%d",
              tp->name,
              g_sink_momentum_list[ipf][0], g_sink_momentum_list[ipf][1], g_sink_momentum_list[ipf][2],
              gamma_bin_to_name[tp->gf1[0]], g_src_snk_time_separation,
              gamma_bin_to_name[tp->gf2], gamma_bin_to_name[tp->gi1[0]],
              pc[0], pc[1], pc[2] );
        } else { 
          continue;
        }
        fprintf ( stdout, "# [htpp_analyse] output_filename = %s\n", output_filename );

        FILE * ofs = fopen ( output_filename, "w" );

        for( int iconf = 0; iconf < num_conf; iconf++ ) {
          for( int isrc = 0; isrc < num_src_per_conf; isrc++) {
            for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {

               int const csx[4] = { 
                   ( conf_src_list[iconf][isrc][2] + ( T_global / g_coherent_source_number ) * icoh ) % T_global,
                   ( conf_src_list[iconf][isrc][3] + (LX_global / g_coherent_source_number ) * icoh) % LX_global,
                   ( conf_src_list[iconf][isrc][4] + (LY_global / g_coherent_source_number ) * icoh) % LY_global,
                   ( conf_src_list[iconf][isrc][5] + (LZ_global / g_coherent_source_number ) * icoh) % LZ_global };

              /***********************************************************
               * output key
               ***********************************************************/
              fprintf ( ofs, "# /%c/conf%d/t%d_x%d_y%d_z%d\n", conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], csx[0], csx[1], csx[2], csx[3]  );
              for ( int it = 0; it < n_tc; it++ ) {
                fprintf ( ofs, "%4d %25.16e %25.16e\n", it,
                corr[i_2pt][ipf][ipi][iconf][isrc*g_coherent_source_number+icoh][2*it  ],
                corr[i_2pt][ipf][ipi][iconf][isrc*g_coherent_source_number+icoh][2*it+1] );
              } 

            }
          }
        }
        fclose ( ofs );
      }
    }
  }  /* end of loop on 2pt functions */

  /***************************************************************************
   ***************************************************************************
   **
   ** UWerr statistical analysis for corr
   **
   ***************************************************************************
   ***************************************************************************/
  for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {

    twopoint_function_type * tp = &(g_twopoint_function_list[i_2pt]);


    for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
      for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

        int pc[3] = {
            -( g_sink_momentum_list[ipf][0] + g_source_momentum_list[ipf][0] ),
            -( g_sink_momentum_list[ipf][1] + g_source_momentum_list[ipf][1] ),
            -( g_sink_momentum_list[ipf][2] + g_source_momentum_list[ipf][2] ) };

        for ( int ireim =0; ireim < 2; ireim++ ) {

          int const nmeas = num_conf * num_src_per_conf * g_coherent_source_number;
          double ** data = init_2level_dtable ( nmeas, n_tc );
          if ( data == NULL ) {
            fprintf ( stderr, "[htpp_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }

#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              for ( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {
                for ( int it = 0; it < n_tc; it++ ) {
                  data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it] = corr[i_2pt][ipf][ipi][iconf][isrc*g_coherent_source_number+icoh][2*it+ireim];
                }
              }
            }
          }

          char obs_name[400];

          sprintf ( obs_name, "%s_pfx%dpfy%dpfz%d_gf_%s_dt%d_g1_%s_g2_%s_PX%d_PY%d_PZ%d.%s",
                           tp->name,
                           g_sink_momentum_list[ipf][0], g_sink_momentum_list[ipf][1], g_sink_momentum_list[ipf][2],
                           gamma_bin_to_name[tp->gf1[0]], g_src_snk_time_separation,
                           gamma_bin_to_name[tp->gf2], gamma_bin_to_name[tp->gi1[0]],
                           pc[0], pc[1], pc[2], reim_str[ireim] );
        

          exitstatus = apply_uwerr_real (  data[0], nmeas, n_tc, 0, 1, obs_name );
          if ( exitstatus != NULL ) {
            fprintf ( stderr, "[htpp_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(14);
          }

          fini_2level_dtable ( &data );
        }  /* end of loop on ireim */
      }
    }
  }

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the correlator field
   ***************************************************************************/
  fini_6level_dtable ( &corr );

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
  fini_3level_itable ( &conf_src_list );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [htpp_analyse] %s# [htpp_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [htpp_analyse] %s# [htpp_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
