/****************************************************
 * htpp_compact_analyse
 *
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
#include "derived_quantities.h"
#include "cvc_timer.h"
#include "contractions_io.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code form 2-pt function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  EXIT(0);
}

char const gamma_id_to_ascii[16][10] = { "id", "gx", "gy", "gzgtg5", "gz", "gygtg5", "gxgtg5", "gtg5", "gt", "gxgt", "gygt", "gzg5", "gzgt", "gyg5", "gxg5", "g5" };

inline int mom_vec_p_eq_null ( int const p[3] ) {
  return ( p[0] == 0 && p[1] == 0 && p[2] == 0 );
}

inline int mom_vec_p_eq_q ( int const p[3], int const q[3] ) {
  return ( p[0] == q[0] && p[1] == q[1] && p[2] == q[2] );
}

inline int mom_vec_p_eq_mq ( int const p[3], int const q[3] ) {
  return ( p[0] == -q[0] && p[1] == -q[1] && p[2] == -q[2] );
}

inline int get_momentum_id ( int const p[3], int (* const momentum_list)[3] , int const momentum_number ) {

  for (int i = 0; i < momentum_number; i++ ) {
    if ( mom_vec_p_eq_q( p, momentum_list[i] ) ) return ( i );
  }
  return ( -1 );
}

/***************************************************************************
 * momentum filter
 ***************************************************************************/
inline int momentum_filter ( twopoint_function_type * const tp ) {

  if ( strcmp ( tp->type , "m-m" ) == 0 ) {

    return ( ( tp->pi1[0] + tp->pf1[0] == 0 ) &&
             ( tp->pi1[1] + tp->pf1[1] == 0 ) &&
             ( tp->pi1[2] + tp->pf1[2] == 0 ) );

  } else if ( strcmp ( tp->type , "m-j-m" ) == 0 ) {
  
    return ( ( tp->pi1[0] + tp->pf1[0] + tp->pf2[0] == 0 ) &&
             ( tp->pi1[1] + tp->pf1[1] + tp->pf2[1] == 0 ) &&
             ( tp->pi1[2] + tp->pf1[2] + tp->pf2[2] == 0 ) );

  } else if ( strcmp ( tp->type , "mxm-j-m" ) == 0 ) {

    return ( ( tp->pi1[0] + tp->pi2[0] + tp->pf1[0] + tp->pf2[0] == 0 ) &&
             ( tp->pi1[1] + tp->pi2[1] + tp->pf1[1] + tp->pf2[1] == 0 ) &&
             ( tp->pi1[2] + tp->pi2[2] + tp->pf1[2] + tp->pf2[2] == 0 ) );
  } else {
    return ( 1 == 0 );
  }

} /* end of mometnum_filter */

/***********************************************************
 *
 ***********************************************************/
int main(int argc, char **argv) {
  
  int const gamma_parity_sign[16] = {       1,   -1,   -1,      1,   -1,      1,      1,     -1,    1,     -1,     -1,      1,     -1,      1,      1,   -1 };

  int const gamma_chargeconjugation_sign[16] = {
                                            1,   -1,   -1,     -1,   -1,     -1,     -1,      1,   -1,     -1,     -1,      1,     -1,      1,      1,    1 };

  int const gamma_g5herm_sign[16] = {       1,   -1,   -1,     -1,   -1,     -1,     -1,      1,   -1,     -1,     -1,      1,     -1,      1,      1,    1 };

  int const gamma_timereversal_sign[16] = { 1,    1,    1,      1,    1,      1,      1,      1,   -1,     -1,     -1,     -1,     -1,     -1,     -1,   -1 };
  
  char const reim_str[2][3]  = { "re", "im" };


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char ensemble_name[100] = "NA";
  char filename[100];
  int num_conf = 0, num_src_per_conf = 0;
  int fold_correlator = 0;

  char corr_type [200] = "NA";

  int source_coords_layout[4] = { 0, 1, 2, 3 };

  struct timeval ta, tb;
  struct timeval start_time, end_time;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:E:F:T:s:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [htpp_compact_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [htpp_compact_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [htpp_compact_analyse] ensemble name set to %s\n", ensemble_name );
      break;
    case 'F':
      fold_correlator = atoi(  optarg );
      fprintf ( stdout, "# [htpp_compact_analyse] fold_correlator set to %d\n", fold_correlator );
      break;
    case 'T':
      strcpy (corr_type, optarg );
      fprintf ( stdout, "# [htpp_compact_analyse] fold_correlator set to %s\n", corr_type );
      break;
    case 's':
      sscanf ( optarg, "%d,%d,%d,%d", source_coords_layout,
          source_coords_layout+1, source_coords_layout+2, source_coords_layout+3 );
      fprintf(  stdout, "# [] source_coords_layout = %d, %d, %d, %d", source_coords_layout[0],
          source_coords_layout[1], source_coords_layout[2], source_coords_layout[3] );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /***********************************************************
   * timer for total time
   ***********************************************************/
  gettimeofday ( &start_time, (struct timezone *)NULL );


  /* set the default values */
  if(filename_set==0) strcpy(filename, "twopt.input");
  /* fprintf(stdout, "# [htpp_compact_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [htpp_compact_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [htpp_compact_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [htpp_compact_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[htpp_compact_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[htpp_compact_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[htpp_compact_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [htpp_compact_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[htpp_compact_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[htpp_compact_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [htpp_compact_analyse] comment %s\n", line );
      continue;
    }


    /***********************************************************
     * QLUA source coords files have ordering 
     * stream conf x y z t
     ***********************************************************/
    sscanf( line, "%c %d %d %d %d %d",
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf],
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+1,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf] + source_coords_layout[0] + 2,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf] + source_coords_layout[1] + 2,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf] + source_coords_layout[2] + 2,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf] + source_coords_layout[3] + 2 );

    count++;
  }

  fclose ( ofs );

  /***********************************************************
   * show all configs and source locations
   ***********************************************************/
  if ( g_verbose > 4 ) {
    fprintf ( stdout, "# [htpp_compact_analyse] conf_src_list conf t x y z\n" );
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
  if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_analyse] n_tc = %d %s %d\n", n_tc, __FILE__, __LINE__ );


  /***********************************************************
   * loop on total momenta
   ***********************************************************/
  for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

    int ptot[3] = {
        g_total_momentum_list[iptot][0],
        g_total_momentum_list[iptot][1],
        g_total_momentum_list[iptot][2] };

#ifndef HAVE_HDF5
    EXIT(123);
#endif

    /***********************************************************
     * reader for h5 file
     ***********************************************************/
    if ( ( strcmp ( corr_type , "m-j-m"   ) == 0 ) || 
         ( strcmp ( corr_type , "mxm-j-m" ) == 0 ) ) {
        sprintf ( filename, "%s/%s.dt%d.px%d_py%d_pz%d.h5", filename_prefix, filename_prefix2, g_src_snk_time_separation,
            ptot[0], ptot[1], ptot[2] );
    } else if ( strcmp ( corr_type , "m-m"     ) == 0 ) {
        sprintf ( filename, "%s/%s.px%d_py%d_pz%d.h5", filename_prefix, filename_prefix2, ptot[0], ptot[1], ptot[2] );
    } else {
      continue;
    }

    if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_analyse] filename = %s %s %d\n", filename, __FILE__, __LINE__ );
 

    int gamma_rho_number = 0;
    int gamma_v_number = 0;
    int * gamma_rho_list = NULL, * gamma_v_list = NULL;
    size_t ndim = 0, * dim = NULL;

    if ( strcmp ( corr_type , "m-j-m"   ) == 0 ) {
      exitstatus = read_from_h5_file_varsize ( (void**)&gamma_rho_list, filename, "gamma_rho",  "int", &ndim, &dim,  io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[htpp_compact_analyse] Error from read_from_h5_file_varsize , status %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(2);
      }  
    
      gamma_rho_number = dim[0];
      free ( dim ); dim = NULL;
    }

    if ( strcmp ( corr_type , "m-j-m"   ) == 0 || strcmp ( corr_type , "mxm-j-m"   ) == 0 ) { 
      exitstatus = read_from_h5_file_varsize ( (void**)&gamma_v_list, filename, "gamma_v",  "int", &ndim, &dim,  io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[htpp_compact_analyse] Error from read_from_h5_file_varsize , status %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(2);
      }  

      gamma_v_number = dim[0];
      free ( dim ); dim = NULL;
    }

    int * buffer = NULL;
    int momentum_number = 0;
    int *** momentum_list = NULL;
    if ( strcmp ( corr_type , "m-j-m"   ) == 0 || strcmp ( corr_type , "mxm-j-m"   ) == 0 ) { 
      exitstatus = read_from_h5_file_varsize ( (void**)&buffer, filename, "mom",  "int", &ndim, &dim,  io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[htpp_compact_analyse] Error from read_from_h5_file_varsize , status %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(2); 
      }  

      if ( ndim != 3 ) {
        EXIT(12);
      }
      momentum_number = dim[0];

      momentum_list = init_3level_itable ( dim[0], dim[1], dim[2] );
      if ( momentum_list == NULL ) {
        fprintf( stderr, "[htpp_compact_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__ );
        EXIT(14);
      }

      memcpy ( momentum_list[0][0], buffer, dim[0] * dim[1] * dim[2] * sizeof ( int ) );
 
      if ( buffer != NULL ) free ( buffer );
      free ( dim ); dim = NULL;
    }

    int nitem = 2 * n_tc;
    if ( strcmp ( corr_type , "mxm-j-m"   ) == 0 ) {
      nitem *= gamma_v_number * momentum_number;
    } else if ( strcmp ( corr_type , "m-j-m"   ) == 0 ) {
      nitem *= gamma_rho_number * gamma_v_number * momentum_number;
    }

    double *** corr = init_3level_dtable ( num_conf, num_src_per_conf, nitem );
 
    if ( corr == NULL ) {
      fprintf( stderr, "[htpp_compact_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
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

        char tag[400];
        if ( strcmp ( corr_type , "m-j-m"   ) == 0 || strcmp ( corr_type , "mxm-j-m"   ) == 0 ) {
          sprintf ( tag, "/s%c/c%d/t%dx%dy%dz%d", stream, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
        } else if ( strcmp ( corr_type , "m-m"   ) == 0 ) {
          sprintf ( tag, "/%s/gf_%s/gi_%s/s%c/c%d/t%dx%dy%dz%d", filename_prefix3,
             gamma_id_to_ascii[g_sink_gamma_id_list[0]], gamma_id_to_ascii[g_source_gamma_id_list[0]],
              stream, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
        }

        if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_analyse] tag = %s %s %d\n", tag, __FILE__, __LINE__ );

        exitstatus = read_from_h5_file ( corr[iconf][isrc], filename, tag, io_proc );
        if ( exitstatus != 0 ) {
          fprintf( stderr, "[htpp_compact_analyse] Error from read_from_h5_file for file %s tag %s, status %d %s %d\n", filename, tag, exitstatus, __FILE__, __LINE__ );
          EXIT(2);
        } 

      }  /* end of loop on sources */
    }  /* end of loop on configs */

    /***************************************************************************
     ***************************************************************************
     **
     ** UWerr statistical analysis for corr
     **
     ***************************************************************************
     ***************************************************************************/

    /***********************************************************
     * analyse per momentum and gamma config
     ***********************************************************/

    if ( strcmp ( corr_type , "m-j-m" ) == 0 ) {

      for ( int igi = 0; igi < gamma_rho_number; igi++ ) {
        for ( int igc = 0; igc < gamma_v_number; igc++ ) {
  
          for ( int imom = 0; imom < momentum_number; imom++ ) {
  
            int pf[3] = {
              momentum_list[imom][0][0],
              momentum_list[imom][0][1],
              momentum_list[imom][0][2] };
  
            int pi1[3] = {
              momentum_list[imom][1][0],
              momentum_list[imom][1][1],
              momentum_list[imom][1][2] };
  
            for ( int ireim = 0; ireim < 2; ireim++ ) {
  
              double ** data = init_2level_dtable ( num_conf, n_tc );
              if ( data == NULL ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
                EXIT(12);
              }
  
#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                
                  for ( int it = 0; it < n_tc; it++ ) {
  
                    int const idx =  2 * ( ( ( igi * gamma_v_number + igc ) * momentum_number + imom ) * n_tc + it ) + ireim ;
  
                    data[iconf][it] += corr[iconf][isrc][idx]; 
                  }
                }
                for ( int it = 0; it < n_tc; it++ ) data[iconf][it] /= (double)num_src_per_conf;
              }
  
              char obs_name[2000];
  
              sprintf ( obs_name, "%s.dt%d.px%d_py%d_pz%d.pfx%d_pfy%d_pfz%d.gi_%s.gc_%s.%s", filename_prefix2, g_src_snk_time_separation,
                  ptot[0], ptot[1], ptot[2],
                  pf[0], pf[1], pf[2],
                  gamma_id_to_ascii[gamma_rho_list[igi]],
                  gamma_id_to_ascii[gamma_v_list[igc]], reim_str[ireim] );
  
              exitstatus = apply_uwerr_real (  data[0], num_conf, n_tc, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(14);
              }
  
             /***************************************************************************
              * effective mass analysis
              ***************************************************************************/
              if ( strcmp( "m-m", corr_type ) == 0 ) {
              
                int const Thp1 = n_tc / 2 + 1;
       
                for ( int itau = 1; itau < Thp1/2; itau++ ) {
                  int narg = 3;
                  int arg_first[3] = { 0, 2 * itau, itau };
                  int arg_stride[3] = {1,1,1};
                  int nT = Thp1 - 2 * itau;
  
                  char obs_name2[2000];
                  sprintf ( obs_name2, "%s.acosh_ratio.tau%d", obs_name, itau );
  
                  exitstatus = apply_uwerr_func ( data[0], num_conf, n_tc, T_global, narg, arg_first, arg_stride, obs_name, acosh_ratio, dacosh_ratio );
                  if ( exitstatus != 0 ) {
                    fprintf ( stderr, "[htpp_compact_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                    EXIT(115);
                  }
  
                }
  
              }
  
              fini_2level_dtable ( &data );
  
            }  /* end of loop on re im */
  
          }  /* end of loop on momentum */
  
        }  /* end of loop on gamma_c */
  
      }  /* end of loop on gamma_i1 */

    }  /* end of if m-j-m */

    if ( strcmp ( corr_type , "mxm-j-m" ) == 0 ) {

      for ( int igc = 0; igc < gamma_v_number; igc++ ) {
  
          for ( int imom = 0; imom < momentum_number; imom++ ) {
  
            int pf[3] = {
              momentum_list[imom][0][0],
              momentum_list[imom][0][1],
              momentum_list[imom][0][2] };
  
            int pi1[3] = {
              momentum_list[imom][1][0],
              momentum_list[imom][1][1],
              momentum_list[imom][1][2] };
  
            for ( int ireim = 0; ireim < 2; ireim++ ) {
  
              double ** data = init_2level_dtable ( num_conf, n_tc );
              if ( data == NULL ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
                EXIT(12);
              }
  
#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                
                  for ( int it = 0; it < n_tc; it++ ) {
  
                    int const idx =  2 * ( ( igc * momentum_number + imom ) * n_tc + it ) + ireim ;
  
                    data[iconf][it] += corr[iconf][isrc][idx]; 
                  }
                }
                for ( int it = 0; it < n_tc; it++ ) data[iconf][it] /= (double)num_src_per_conf;
              }
  
              char obs_name[2000];
  
              sprintf ( obs_name, "%s.dt%d.px%d_py%d_pz%d.pfx%d_pfy%d_pfz%d.pi1x%d_pi1y%d_pi1z%d.gc_%s.%s", filename_prefix2, g_src_snk_time_separation,
                  ptot[0], ptot[1], ptot[2],
                  pf[0], pf[1], pf[2],
                  pi1[0], pi1[1], pi1[2],
                  gamma_id_to_ascii[gamma_v_list[igc]], reim_str[ireim] );
  
              exitstatus = apply_uwerr_real (  data[0], num_conf, n_tc, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(14);
              }
  
              fini_2level_dtable ( &data );
  
            }  /* end of loop on re im */
  
          }  /* end of loop on momentum */
  
      }  /* end of loop on gamma_c */
  
    }  /* end of if mxm-j-m */


    if ( strcmp ( corr_type , "m-m" ) == 0 ) {

            for ( int ireim = 0; ireim < 2; ireim++ ) {
  
              double ** data = init_2level_dtable ( num_conf, n_tc );
              if ( data == NULL ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
                EXIT(12);
              }
  
#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                
                  for ( int it = 0; it < n_tc; it++ ) {
                    data[iconf][it] += corr[iconf][isrc][2*it+ireim]; 
                  }
                }
                for ( int it = 0; it < n_tc; it++ ) data[iconf][it] /= (double)num_src_per_conf;
              }
  
              char obs_name[2000];
  
              sprintf ( obs_name, "%s.px%d_py%d_pz%d.g_%s.g_%s.%s", filename_prefix3,
                  ptot[0], ptot[1], ptot[2],
                  gamma_id_to_ascii[g_sink_gamma_id_list[0]],
                  gamma_id_to_ascii[g_source_gamma_id_list[0]], reim_str[ireim] );
  
              exitstatus = apply_uwerr_real (  data[0], num_conf, n_tc, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(14);
              }
  
             /***************************************************************************
              * effective mass analysis
              ***************************************************************************/
              
                int const Thp1 = n_tc / 2 + 1;
       
                for ( int itau = 1; itau < Thp1/2; itau++ ) {
                  int narg = 3;
                  int arg_first[3] = { 0, 2 * itau, itau };
                  int arg_stride[3] = {1,1,1};
                  int nT = Thp1 - 2 * itau;
  
                  char obs_name2[2000];
                  sprintf ( obs_name2, "%s.acosh_ratio.tau%d", obs_name, itau );
  
                  exitstatus = apply_uwerr_func ( data[0], num_conf, n_tc, nT, narg, arg_first, arg_stride, obs_name2, acosh_ratio, dacosh_ratio );
                  if ( exitstatus != 0 ) {
                    fprintf ( stderr, "[htpp_compact_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                    EXIT(115);
                  }
  
                }
  
              fini_2level_dtable ( &data );
  
            }  /* end of loop on re im */

    }  /* end of if m-m */

    fini_3level_dtable ( &corr );
    if ( gamma_rho_list != NULL ) free ( gamma_rho_list );
    if ( gamma_v_list   != NULL ) free ( gamma_v_list );
    fini_3level_itable ( &momentum_list );


  }  /* end of loop on total momentum */

  /***************************************************************************/
  /***************************************************************************/

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

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "htpp_compact_analyse", "total-time", io_proc == 2 );

  return(0);

}
