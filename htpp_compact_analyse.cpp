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

  } else if ( strcmp ( tp->type , "mxm-m" ) == 0 ) {

    return ( ( tp->pi1[0] + tp->pi2[0] + tp->pf1[0] == 0 ) &&
             ( tp->pi1[1] + tp->pi2[1] + tp->pf1[1] == 0 ) &&
             ( tp->pi1[2] + tp->pi2[2] + tp->pf1[2] == 0 ) );

  } else if ( strcmp ( tp->type , "mxm-mxm" ) == 0 ) {

    return ( ( tp->pi1[0] + tp->pi2[0] + tp->pf1[0] + tp->pf2[0] == 0 ) &&
             ( tp->pi1[1] + tp->pi2[1] + tp->pf1[1] + tp->pf2[1] == 0 ) &&
             ( tp->pi1[2] + tp->pi2[2] + tp->pf1[2] + tp->pf2[2] == 0 ) );

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

  double const TWO_MPI = 2. * M_PI;


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char ensemble_name[100] = "NA";
  char filename[100];
  int num_conf = 0, num_src_per_conf = 0;
  int fold_correlator = 0;
  int write_data = 0;

  char corr_type [200] = "NA";
  int add_source_phase = 0;
  int add_source_shift = 0;

  int source_coords_layout[4] = { 0, 1, 2, 3 };

  struct timeval ta, tb;
  struct timeval start_time, end_time;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:E:F:T:s:w:a:A:")) != -1) {
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
      fprintf ( stdout, "# [htpp_compact_analyse] corr_type set to %s\n", corr_type );
      break;
    case 's':
      sscanf ( optarg, "%d,%d,%d,%d", source_coords_layout,
          source_coords_layout+1, source_coords_layout+2, source_coords_layout+3 );
      fprintf(  stdout, "# [htpp_compact_analyse] source_coords_layout = %d, %d, %d, %d\n", source_coords_layout[0],
          source_coords_layout[1], source_coords_layout[2], source_coords_layout[3] );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [htpp_compact_analyse] write_data set to %d\n", write_data );
      break;
    case 'a':
      add_source_phase = atoi ( optarg );
      fprintf ( stdout, "# [htpp_compact_analyse] add_source_phase set to %d\n", add_source_phase );
      break;
    case 'A':
      add_source_shift = atoi ( optarg );
      fprintf ( stdout, "# [htpp_compact_analyse] add_source_shift set to %d\n", add_source_shift );
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
  sprintf ( filename, "source_coords.%s.anl" , ensemble_name );
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

    int n_tc = 0;
    /***********************************************************
     * reader for h5 file
     ***********************************************************/
    if ( ( strcmp ( corr_type , "m-j-m"   ) == 0 ) || 
         ( strcmp ( corr_type , "mxm-j-m" ) == 0 ) ) {
        sprintf ( filename, "%s/%s.dt%d.px%d_py%d_pz%d.h5", filename_prefix, filename_prefix2, g_src_snk_time_separation,
            ptot[0], ptot[1], ptot[2] );
        n_tc = g_src_snk_time_separation + 1;
    } else if ( ( strcmp ( corr_type , "m-m"     ) == 0 ) ||
                ( strcmp ( corr_type , "mxm-m"   ) == 0 ) ) {
        sprintf ( filename, "%s/%s.px%d_py%d_pz%d.h5", filename_prefix, filename_prefix2, ptot[0], ptot[1], ptot[2] );
        n_tc = T_global;
    } else if ( strcmp ( corr_type , "mxm-mxm" ) == 0 ) {
        sprintf ( filename, "%s/%s.px%d_py%d_pz%d.h5", filename_prefix, filename_prefix2, ptot[0], ptot[1], ptot[2] );
        n_tc = g_src_snk_time_separation + 1;
    } else {
      continue;
    }

    if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_analyse] filename = %s %s %d\n", filename, __FILE__, __LINE__ );
    if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_analyse] n_tc = %d %s %d\n", n_tc, __FILE__, __LINE__ );
 

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

    int gamma_sink_number = 1, *gamma_sink_list = NULL;

    if ( strcmp ( corr_type , "m-m"   ) == 0 || strcmp ( corr_type , "mxm-m"   ) == 0 ) 
    { 
      exitstatus = read_from_h5_file_varsize ( (void**)&gamma_sink_list, filename, "gamma_sink",  "int", &ndim, &dim,  io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[htpp_compact_analyse] Error from read_from_h5_file_varsize , status %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(2);
      }  

      gamma_sink_number = dim[0];
      if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_analyse] gamma_sink_number = %d   %s %d\n", gamma_sink_number, __FILE__, __LINE__ );
      free ( dim ); dim = NULL;
    }


    int gamma_source_number = 1, *gamma_source_list = NULL;

    if ( strcmp ( corr_type , "m-m"   ) == 0 ) 
    { 
      exitstatus = read_from_h5_file_varsize ( (void**)&gamma_source_list, filename, "gamma_source",  "int", &ndim, &dim,  io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[htpp_compact_analyse] Error from read_from_h5_file_varsize , status %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(2);
      }  

      gamma_source_number = dim[0];
      if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_analyse] gamma_source_number = %d   %s %d\n", gamma_source_number, __FILE__, __LINE__ );
      free ( dim ); dim = NULL;
    }


    int * buffer = NULL;
    int momentum_number = 0;
    int *** momentum_list = NULL;
    if (    strcmp ( corr_type , "m-j-m"   ) == 0 
         || strcmp ( corr_type , "mxm-j-m" ) == 0 
         || strcmp ( corr_type , "mxm-m"   ) == 0 
         || strcmp ( corr_type , "mxm-mxm" ) == 0 ) { 
      exitstatus = read_from_h5_file_varsize ( (void**)&buffer, filename, "mom",  "int", &ndim, &dim,  io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[htpp_compact_analyse] Error from read_from_h5_file_varsize for file %s    %s , status %d %s %d\n", 
            filename, "mom", exitstatus, __FILE__, __LINE__ );
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

      if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [htpp_compact_analyse] momentum_number = %d   %s %d\n", momentum_number, __FILE__, __LINE__ );
        for ( int i = 0; i < momentum_number; i++ ) {
          fprintf ( stdout, "  P %3d %3d %3d  pf1 %3d %3d %3d pi1 %3d %3d %3d\n", 
              ptot[0], ptot[1], ptot[2],
              momentum_list[i][0][0], momentum_list[i][0][1], momentum_list[i][0][2],
              momentum_list[i][1][0], momentum_list[i][1][1], momentum_list[i][1][2] );
        }
      }
    }

    int nitem = 2 * n_tc;
    if ( strcmp ( corr_type , "mxm-j-m"   ) == 0 ) {
      nitem *= gamma_v_number * momentum_number;
    } else if ( strcmp ( corr_type , "m-j-m"   ) == 0 ) {
      nitem *= gamma_rho_number * gamma_v_number * momentum_number;
    } else if ( strcmp ( corr_type , "m-m"   ) == 0 ) {
      nitem *= gamma_source_number * gamma_sink_number;
    } else if ( strcmp ( corr_type , "mxm-m"   ) == 0 ) {
      nitem *= gamma_sink_number * momentum_number; 
    } else if ( strcmp ( corr_type , "mxm-mxm"   ) == 0 ) {
      nitem *= momentum_number; 
    }

    double *** corr = init_3level_dtable ( num_conf, num_src_per_conf, nitem );
 
    if ( corr == NULL ) {
      fprintf( stderr, "[htpp_compact_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(2);
    }
    if ( g_verbose > 2 ) {
      fprintf ( stdout, "# [htpp_compact_analyse] corr_type %s  nitem = %d    %s %d\n", corr_type, nitem, __FILE__, __LINE__ );
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
        if ( strcmp ( corr_type , "m-m"   ) == 0  ) 
        {
          /*sprintf ( tag, "/%s/gf_%s/gi_%s/s%c/c%d/t%dx%dy%dz%d", filename_prefix3, gamma_id_to_ascii[g_sink_gamma_id_list[0]], 
              gamma_id_to_ascii[g_source_gamma_id_list[0]], stream, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] ); */
          sprintf ( tag, "/s%c/c%d/t%dx%dy%dz%d", stream, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
        } else {
          sprintf ( tag, "/s%c/c%d/t%dx%dy%dz%d", stream, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
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
     * m - j - m
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
  
            for ( int ireim = 0; ireim < 2; ireim++ )
            /* for ( int ireim = 1; ireim <= 1; ireim++ ) */
            {
  
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
  
              if ( num_conf >= 6 )  {
                exitstatus = apply_uwerr_real (  data[0], num_conf, n_tc, 0, 1, obs_name );
                if ( exitstatus != 0 ) {
                  fprintf ( stderr, "[htpp_compact_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(14);
                }
              }
  
             /***************************************************************************
              * 
              ***************************************************************************/
             if ( write_data ) {
               sprintf ( filename, "%s.corr", obs_name );
               FILE * fs = fopen ( filename, "w" );
               for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                 for ( int it = 0; it < n_tc; it++ ) {
                   fprintf ( fs, "%3d %25.16e %6d %c\n", it, data[iconf][it], conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] ) ;
                 }
               }

               fclose ( fs );
             }

  
              fini_2level_dtable ( &data );
  
            }  /* end of loop on re im */
  
          }  /* end of loop on momentum */
  
        }  /* end of loop on gamma_c */
  
      }  /* end of loop on gamma_i1 */

    }  /* end of if m-j-m */

    /***********************************************************
     * m x m - j -m
     ***********************************************************/
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
  
            for ( int ireim = 0; ireim < 2; ireim++ )
            {
  
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
  
              if ( num_conf >= 6 ) {
                exitstatus = apply_uwerr_real (  data[0], num_conf, n_tc, 0, 1, obs_name );
                if ( exitstatus != 0 ) {
                  fprintf ( stderr, "[htpp_compact_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(14);
                }
              }
  
              /***************************************************************************
               *
               ***************************************************************************/
              if ( write_data ) {
                sprintf ( filename, "%s.corr", obs_name );
                FILE * fs = fopen ( filename, "w" );
                for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                  for ( int it = 0; it < n_tc; it++ ) {
                    fprintf ( fs, "%3d %25.16e %6d %c\n", it, data[iconf][it], conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] ) ;
                  }
                }

#if 0
                for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                  for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                    fprintf ( fs, "# %c %6d %3d %3d %3d %3d\n", conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
                           conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5]) ;

                    for ( int it = 0; it < n_tc; it++ ) {
                      int const idx =  2 * ( ( igc * momentum_number + imom ) * n_tc + it ) + ireim ;
                      fprintf ( fs, "%3d %25.16e\n", it, corr[iconf][isrc][idx] );
                    }
                  }
                }
#endif
                fclose ( fs );
              }

              fini_2level_dtable ( &data );
  
            }  /* end of loop on re im */
  
          }  /* end of loop on momentum */
  
      }  /* end of loop on gamma_c */
  
    }  /* end of if mxm-j-m */


    /***********************************************************
     * m - m
     ***********************************************************/
    if ( strcmp ( corr_type , "m-m" ) == 0 ) {

      /* int const gamma_source_number = g_source_gamma_id_number;
      int const gamma_sink_number = g_sink_gamma_id_number;
      int * const gamma_sink_list   = g_sink_gamma_id_list;
      int * const gamma_source_list = g_source_gamma_id_list; */

      for ( int igi = 0; igi < gamma_source_number; igi++ ) 
      {
        for ( int igf = 0; igf < gamma_sink_number; igf++ ) 
        {

          for ( int ireim = 0; ireim < 2; ireim++ ) 
          {
  
              double ** data = init_2level_dtable ( num_conf, n_tc );
              if ( data == NULL ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
                EXIT(12);
              }
  
#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) 
              {
                for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
                {

                  double ephase[2] = {1., 0.};
                  if ( add_source_phase ) 
                  {
                    double const phase_pi1 = -TWO_MPI * (
                          ptot[0] * conf_src_list[iconf][isrc][3] / (double)LX_global
                        + ptot[1] * conf_src_list[iconf][isrc][4] / (double)LY_global 
                        + ptot[2] * conf_src_list[iconf][isrc][5] / (double)LZ_global );

                    ephase[0] = cos( phase_pi1 );
                    ephase[1] = sin( phase_pi1 );
                  }

                  for ( int it = 0; it < n_tc; it++ ) {

                    // int const idx =  2 * ( ( igi * gamma_sink_number +igf ) * n_tc + it ) + ireim ;
                    int const idx =  2 * ( ( igi * gamma_sink_number +igf ) * n_tc + it );

                    double const a[2] = { corr[iconf][isrc][idx], corr[iconf][isrc][idx+1] };

                    double const b[2] = {
                      a[0] * ephase[0] - a[1] * ephase[1] ,
                      a[0] * ephase[1] + a[1] * ephase[0] 
                    };

                    // data[iconf][it] += corr[iconf][isrc][idx];
                    data[iconf][it] += b[ireim];
                  }
                }
                for ( int it = 0; it < n_tc; it++ ) data[iconf][it] /= (double)num_src_per_conf;
              }

              char obs_name[2000];
  
              sprintf ( obs_name, "%s.px%d_py%d_pz%d.gf1_%s.gi1_%s.%s", filename_prefix3,
                  ptot[0], ptot[1], ptot[2],
                  gamma_id_to_ascii[gamma_sink_list[igf]],
                  gamma_id_to_ascii[gamma_source_list[igi]], reim_str[ireim] );
  
              if ( num_conf >= 6 ) {
                exitstatus = apply_uwerr_real (  data[0], num_conf, n_tc, 0, 1, obs_name );
                if ( exitstatus != 0 ) {
                  fprintf ( stderr, "[htpp_compact_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(14);
                }

               /***************************************************************************
                * effective mass analysis
                ***************************************************************************/
              
                int const Thp1 = n_tc / 2 + 1;
       
                for ( int itau = 1; itau < Thp1/8; itau++ ) {
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

              }
 
              /***************************************************************************
               *
               ***************************************************************************/
              if ( write_data ) {
                sprintf ( filename, "%s.corr", obs_name );
                FILE * fs = fopen ( filename, "w" );
                for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                  for ( int it = 0; it < n_tc; it++ ) {
                    fprintf ( fs, "%3d %25.16e %6d %c\n", it, data[iconf][it], conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] ) ;
                  }
                }
 
                fclose ( fs );
              }

              fini_2level_dtable ( &data );
  
            }  /* end of loop on re im */

        }
      }
    }  /* end of if m-m */

    /***********************************************************
     * m x m - m
     ***********************************************************/
    if ( strcmp ( corr_type , "mxm-m" ) == 0 ) {

      for ( int igf = 0; igf < gamma_sink_number; igf++ ) {
        for ( int imom = 0; imom < momentum_number; imom++ ) 
        {

          for ( int ireim = 0; ireim < 2; ireim++ ) {
  
              double ** data = init_2level_dtable ( num_conf, n_tc );
              if ( data == NULL ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
                EXIT(12);
              }
  
#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

                  for ( int it = 0; it < n_tc; it++ ) 
                  {

                    int const its = add_source_shift ? ( it + add_source_shift * conf_src_list[iconf][isrc][2] + T_global ) % T_global : it;

                    double ephase[2] = {1., 0.};

                    if ( add_source_phase ) 
                    {
                      double const phase_pi1 = TWO_MPI * (
                            momentum_list[imom][1][0] * conf_src_list[iconf][isrc][3] / (double)LX_global
                          + momentum_list[imom][1][1] * conf_src_list[iconf][isrc][4] / (double)LY_global 
                          + momentum_list[imom][1][2] * conf_src_list[iconf][isrc][5] / (double)LZ_global );
 
                      ephase[0] = cos( phase_pi1 );
                      ephase[1] = sin( phase_pi1 );
                    }


                    /* int const idx =  2 * ( ( igf * momentum_number + imom  ) * n_tc + it ) + ireim ; */

                    int const idx =  2 * ( ( igf * momentum_number + imom ) * n_tc + its );

                    double const a[2] = { corr[iconf][isrc][idx], corr[iconf][isrc][idx+1] };

                    double const b[2] = {
                      a[0] * ephase[0] - a[1] * ephase[1] ,
                      a[0] * ephase[1] + a[1] * ephase[0]
                    };

                    /* data[iconf][it] += corr[iconf][isrc][idx]; */
                    data[iconf][it] += b[ireim];

                  }
                }
                for ( int it = 0; it < n_tc; it++ ) data[iconf][it] /= (double)num_src_per_conf;
              }

              char obs_name[2000];
  
              sprintf ( obs_name, "%s.px%d_py%d_pz%d.gf1_%s.pi1x%d_pi1y%d_pi1z%d.%s", filename_prefix3,
                  ptot[0], ptot[1], ptot[2],
                  gamma_id_to_ascii[gamma_sink_list[igf]],
                  momentum_list[imom][1][0], momentum_list[imom][1][1], momentum_list[imom][1][2], reim_str[ireim] );
  
              exitstatus = apply_uwerr_real (  data[0], num_conf, n_tc, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(14);
              }
#if 0 
             /***************************************************************************
              * effective mass analysis
              ***************************************************************************/
              
                int const Thp1 = n_tc / 2 + 1;
       
                for ( int itau = 1; itau < Thp1/8; itau++ ) {
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
#endif
                /***************************************************************************
                 *
                 ***************************************************************************/
                if ( write_data && ireim == 1 ) {
                  sprintf ( filename, "%s.corr", obs_name );
                  FILE * fs = fopen ( filename, "w" );
                  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                    for ( int it = 0; it < n_tc; it++ ) {
                      fprintf ( fs, "%3d %25.16e %6d %c\n", it, data[iconf][it], conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] ) ;
                    }
                  }

                  fclose ( fs );
                }

              fini_2level_dtable ( &data );
  
            }  /* end of loop on re im */
        }
      }
    }  /* end of if mxm-m */

    /***************************************************************************/
    /***************************************************************************/

    /***********************************************************
     * m x m - m x m
     ***********************************************************/
    if ( strcmp ( corr_type , "mxm-mxm" ) == 0 ) {

      for ( int imom = 0; imom < momentum_number; imom++ ) 
      {

        for ( int ireim = 0; ireim < 2; ireim++ ) 
        {
  
              double ** data = init_2level_dtable ( num_conf, n_tc );
              if ( data == NULL ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
                EXIT(12);
              }
  
#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

                  for ( int it = 0; it < n_tc; it++ ) {

                    int const its = add_source_shift ? ( it + add_source_shift * conf_src_list[iconf][isrc][2] + T_global ) % T_global : it;

                    double ephase[2] = {1., 0.};

                    if ( add_source_phase ) 
                    {
                      double const phase_pi1 = add_source_phase * TWO_MPI * (
                            momentum_list[imom][1][0] * conf_src_list[iconf][isrc][3] / (double)LX_global
                          + momentum_list[imom][1][1] * conf_src_list[iconf][isrc][4] / (double)LY_global 
                          + momentum_list[imom][1][2] * conf_src_list[iconf][isrc][5] / (double)LZ_global );
 
                      ephase[0] = cos( phase_pi1 );
                      ephase[1] = sin( phase_pi1 );
                    }

                    /* int const idx =  2 * ( imom * n_tc + it ) + ireim ; */

                    int const idx =  2 * ( imom * n_tc + its );

                    double const a[2] = { corr[iconf][isrc][idx], corr[iconf][isrc][idx+1] };

                    double const b[2] = {
                      a[0] * ephase[0] - a[1] * ephase[1] ,
                      a[0] * ephase[1] + a[1] * ephase[0]
                    };

                    /* data[iconf][it] += corr[iconf][isrc][idx]; */

                    data[iconf][it] += b[ireim];

                  }
                }
                for ( int it = 0; it < n_tc; it++ ) data[iconf][it] /= (double)num_src_per_conf;
              }

              char obs_name[2000];
  
              sprintf ( obs_name, "%s.px%d_py%d_pz%d.pf1x%d_pf1y%d_pf1z%d.pi1x%d_pi1y%d_pi1z%d.%s",
                  filename_prefix3, ptot[0], ptot[1], ptot[2],
                  momentum_list[imom][0][0], momentum_list[imom][0][1], momentum_list[imom][0][2], 
                  momentum_list[imom][1][0], momentum_list[imom][1][1], momentum_list[imom][1][2], 
                  reim_str[ireim] );
  
              exitstatus = apply_uwerr_real (  data[0], num_conf, n_tc, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[htpp_compact_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(14);
              }
#if 0
             /***************************************************************************
              * effective mass analysis
              ***************************************************************************/
              
                int const Thp1 = n_tc / 2 + 1;
       
                for ( int itau = 1; itau < Thp1/8; itau++ ) {
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
#endif

                /***************************************************************************
                 *
                 ***************************************************************************/
                if ( write_data && ireim == 0 ) {
                  sprintf ( filename, "%s.corr", obs_name );
                  FILE * fs = fopen ( filename, "w" );
                  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                    for ( int it = 0; it < n_tc; it++ ) {
                      fprintf ( fs, "%3d %25.16e %6d %c\n", it, data[iconf][it], conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] ) ;
                    }
                  }

                  fclose ( fs );
                }


              fini_2level_dtable ( &data );
  
        }  /* end of loop on re im */
      }
    }  /* end of if mxm-mxm */

    fini_3level_dtable ( &corr );
    if ( gamma_rho_list    != NULL ) free ( gamma_rho_list );
    if ( gamma_v_list      != NULL ) free ( gamma_v_list );
    if ( gamma_source_list != NULL ) free ( gamma_source_list );
    if ( gamma_sink_list   != NULL ) free ( gamma_sink_list );
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
