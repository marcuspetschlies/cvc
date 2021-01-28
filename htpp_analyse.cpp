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
#include "derived_quantities.h"
#include "cvc_timer.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code form 2-pt function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  EXIT(0);
}


char const gamma_bin_to_name[16][8] = { "id", "gx", "gy", "gxgy", "gz", "gxgz", "gygz", "gtg5", "gt", "gxgt", "gygt", "gzg5", "gzgt", "gyg5", "gxg5", "g5" };

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

/***********************************************************
 * make observable name
 ***********************************************************/

inline void get_obs_name (char * const obs_name, twopoint_function_type * const tp , int const dt , const char * const reim_str) {
  
  if ( strcmp ( tp->type , "m-j-m" ) == 0 ) {

    sprintf ( obs_name,
        "%s_gf_%s_pfx%dpfy%dpfz%d_dt%d_gc_%s_pcx%dpcy%dpcz%d_gi_%s_pix%dpiy%dpiz%d",
        tp->name,
        gamma_bin_to_name[tp->gf1[0]], 
        tp->pf1[0], tp->pf1[1], tp->pf1[2],
        dt,
        gamma_bin_to_name[tp->gf2], 
        tp->pf2[0], tp->pf2[1], tp->pf2[2],
        gamma_bin_to_name[tp->gi1[0]], 
        tp->pi1[0], tp->pi1[1], tp->pi1[2] );

  } else if ( strcmp( tp->type , "mxm-j-m" ) == 0 )  {
    sprintf ( obs_name,
        "%s_gf_%s_pfx%dpfy%dpfz%d_dt%d_gc_%s_pcx%dpcy%dpcz%d_gi2_%s_pi2x%dpi2y%dpi2z%d_gi1_%s_pi1x%dpi1y%dpi1z%d",
        tp->name,
        gamma_bin_to_name[tp->gf1[0]],
        tp->pf1[0], tp->pf1[1], tp->pf1[2],
        dt,
        gamma_bin_to_name[tp->gf2], 
        tp->pf2[0], tp->pf2[1], tp->pf2[2],
        gamma_bin_to_name[tp->gi2],
        tp->pi2[0], tp->pi2[1], tp->pi2[2],
        gamma_bin_to_name[tp->gi1[0]],
        tp->pi1[0], tp->pi1[1], tp->pi1[2] );

  } else if ( strcmp( tp->type , "m-m" ) == 0 )  {
    sprintf ( obs_name,
        "%s_gf_%s_pfx%dpfy%dpfz%d_gi1_%s_pi1x%dpi1y%dpi1z%d",
        tp->name,
        gamma_bin_to_name[tp->gf1[0]],
        tp->pf1[0], tp->pf1[1], tp->pf1[2],
        gamma_bin_to_name[tp->gi1[0]],
        tp->pi1[0], tp->pi1[1], tp->pi1[2] );

  } else {
    sprintf ( obs_name, "NA" );
    return;
  }
  sprintf( obs_name, "%s.%s", obs_name, reim_str );
}  /* end of get_obs_name */


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

  struct timeval ta, tb;
  struct timeval start_time, end_time;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:E:F:")) != -1) {
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
    case 'F':
      fold_correlator = atoi(  optarg );
      fprintf ( stdout, "# [htpp_analyse] fold_correlator set to %d\n", fold_correlator );
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
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
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

  double ******** corr       = init_8level_dtable (g_twopoint_function_number, g_total_momentum_number, g_sink_momentum_number, g_source_momentum_number, 2, num_conf, num_meas_per_conf, 2 * n_tc );
  double ******* corr_latsym = init_7level_dtable (g_twopoint_function_number, g_total_momentum_number, g_sink_momentum_number, g_source_momentum_number, num_conf, num_meas_per_conf, 2 * n_tc );
  if ( corr == NULL || corr_latsym == NULL ) {
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
      if ( ( strcmp ( g_twopoint_function_list[0].type , "m-j-m" ) == 0 ) || ( strcmp ( g_twopoint_function_list[0].type , "mxm-j-m" ) == 0 ) ) {
        /* sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.aff", filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] ); */
        sprintf ( filename, "%s/%s.%.4d.t%d_x%d_y%d_z%d.aff", filename_prefix, filename_prefix2, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
      } else if ( ( strcmp ( g_twopoint_function_list[0].type , "m-m" ) == 0 )) {
        sprintf ( filename, "%s/stream_%c/%s.%.4d.tbase%.2d.aff", filename_prefix, stream, filename_prefix2, Nconf, gsx[0] );
      } else {
        continue;
      }
   
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
          for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

            /***********************************************************
             * loop on source momenta
             ***********************************************************/
            for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

              /***********************************************************
               * loop on parity
               ***********************************************************/
              for ( int iparity = 0; iparity < 2; iparity++ ) {
                int const sparity = 1 - 2 * iparity;

                int const parity_sign =
                      gamma_parity_sign[tp->gi1[0]]
                    * gamma_parity_sign[tp->gi2]
                    * gamma_parity_sign[tp->gf1[0]]
                    * gamma_parity_sign[tp->gf2];

                int const charge_conjugation_sign = 
                      gamma_chargeconjugation_sign[tp->gi1[0]]
                    * gamma_chargeconjugation_sign[tp->gi2]
                    * gamma_chargeconjugation_sign[tp->gf1[0]] 
                    * gamma_chargeconjugation_sign[tp->gf2];

                int const g5herm_sign =
                        gamma_g5herm_sign[tp->gi1[0]]
                      * gamma_g5herm_sign[tp->gi2]
                      * gamma_g5herm_sign[tp->gf1[0]]
                      * gamma_g5herm_sign[tp->gf2];

                if ( g_verbose > 2 ) fprintf( stdout, "# [htpp_analyse] parity_sign = %d; charge_conjugation_sign = %d; g5herm_sign = %d\n", parity_sign, charge_conjugation_sign, g5herm_sign ); 

                 int pf[3] = {
                  sparity * g_sink_momentum_list[ipf][0],
                  sparity * g_sink_momentum_list[ipf][1],
                  sparity * g_sink_momentum_list[ipf][2] };

                int pi1[3] = {
                  sparity * g_source_momentum_list[ipi][0],
                  sparity * g_source_momentum_list[ipi][1],
                  sparity * g_source_momentum_list[ipi][2] };

                int ptot[3] = {
                  sparity * g_total_momentum_list[iptot][0],
                  sparity * g_total_momentum_list[iptot][1],
                  sparity * g_total_momentum_list[iptot][2] };

                int pi2[3] = {
                  -( ptot[0] + pi1[0] ),
                  -( ptot[1] + pi1[1] ),
                  -( ptot[2] + pi1[2] )};

                int pc[3] = {
                  ptot[0] - pf[0],
                  ptot[1] - pf[1],
                  ptot[2] - pf[2] };
 
                /***********************************************************
                 * set twop function momenta an filter
                 ***********************************************************/
                memcpy( tp->pi1, pi1, 3*sizeof(int) );
                memcpy( tp->pi2, pi2, 3*sizeof(int) );
                memcpy( tp->pf1, pf,  3*sizeof(int) );
                memcpy( tp->pf2, pc,  3*sizeof(int) );

                if ( ! momentum_filter ( tp ) ) continue;

                /***********************************************************
                 * latsym selection weights
                 ***********************************************************/
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
                    sprintf ( key,
                        /* "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/g1_%s/g2_%s/PX%d_PY%d_PZ%d", */
                        "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/g1_%s/g2_%s/x%d_y%d_z%d",
                        diagram_name,
                        /* tp->pf1[0], tp->pf1[1], tp->pf1[2], */
                        pf[0], pf[1], pf[2],
                        gamma_bin_to_name[tp->gf1[0]], g_src_snk_time_separation,
                        gamma_bin_to_name[tp->gf2], gamma_bin_to_name[tp->gi1[0]],
                        /* tp->pf2[0], tp->pf2[1], tp->pf2[2]  */
                        pc[0], pc[1], pc[2] );
                  } else if ( strcmp ( tp->type , "mxm-j-m" ) == 0 ) {
                    sprintf ( key,
                        /* "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/PX%d_PY%d_PZ%d", */
                        "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/x%d_y%d_z%d",
                        diagram_name,
                        pf[0], pf[1], pf[2],
                        gamma_bin_to_name[tp->gf1[0]], g_src_snk_time_separation,
                        pi2[0], pi2[1], pi2[2], gamma_bin_to_name[tp->gi2],
                        gamma_bin_to_name[tp->gf2], gamma_bin_to_name[tp->gi1[0]],
                        pc[0], pc[1], pc[2] );
                  } else if ( strcmp ( tp->type , "m-m" ) == 0 ) {
                    sprintf ( key,
                        /* /fs-fc/t70x02y11z20/gf02_gi02/PX0_PY0_PZ0 */
                        "/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d_gi%.2d/PX%d_PY%d_PZ%d",
                        diagram_name, gsx[0], gsx[1], gsx[2], gsx[3],
                        tp->gf1[0], tp->gi1[0],
                        pf[0], pf[1], pf[2] );
                  } else {
                    continue;
                  }

                  if ( g_verbose > 2 ) {
                    fprintf ( stdout, "# [htpp_analyse] key = %s %s %d\n", key , __FILE__, __LINE__ );
                  }

                  gettimeofday ( &ta, (struct timezone *)NULL );

                  exitstatus = read_aff_contraction ( (void*)(tp->c[i_diag][0][0]), affr, NULL, key, T_global * tp->d * tp->d );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[htpp_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                    EXIT(12);
                  }

                  gettimeofday ( &tb, (struct timezone *)NULL );
                  show_time ( &ta, &tb, "htpp_analyse", "read_aff_contraction", io_proc == 2 );

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
                          pi1[0] * csx[1] / (double)LX_global 
                        + pi1[1] * csx[2] / (double)LY_global 
                        + pi1[2] * csx[3] / (double)LZ_global ) * I );
            
                    if ( g_verbose > 4 ) fprintf ( stdout, "# [htpp_analyse] pi1 = %3d %3d %3d csx = %3d %3d %3d  ephase = %16.7e %16.7e\n",
                        pi1[0], pi1[1], pi1[2],
                        csx[1], csx[2], csx[3],
                        creal( ephase ), cimag( ephase ) );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                    for ( int it = 0; it < n_tc; it++ ) {
                      /* order from source */
                      int const tt = ( csx[0] + it ) % tp->T; 
                      double _Complex const zbuffer = tp->c[i_diag][tt][0][0] * ephase;
              
                      corr[i_2pt][iptot][ipf][ipi][iparity][iconf][isrc * g_coherent_source_number + icoh][2*it  ] = creal ( zbuffer );
                      corr[i_2pt][iptot][ipf][ipi][iparity][iconf][isrc * g_coherent_source_number + icoh][2*it+1] = cimag ( zbuffer );

                      /*
                       * add correlator using discrete quantum numbers */
                      corr_latsym[i_2pt][iptot][ipf][ipi][iconf][isrc * g_coherent_source_number + icoh][2*it  ] += amp_re_factor * creal ( zbuffer );
                      corr_latsym[i_2pt][iptot][ipf][ipi][iconf][isrc * g_coherent_source_number + icoh][2*it+1] += amp_im_factor * cimag ( zbuffer );
                    }

                  }  /* end of loop on coherent sources */
                }  /* end of loop on diagrams */
              }  /* end of loop on parity */
            }  /* end of loop on source momenta */
          }  /* end of loop on total momentum */
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
      
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

          int pc[3] = {
            g_total_momentum_list[iptot][0] - g_sink_momentum_list[ipf][0],
            g_total_momentum_list[iptot][1] - g_sink_momentum_list[ipf][1],
            g_total_momentum_list[iptot][2] - g_sink_momentum_list[ipf][2] };

          int pi2[3] = {
            -g_total_momentum_list[iptot][0] - g_source_momentum_list[ipi][0],
            -g_total_momentum_list[iptot][1] - g_source_momentum_list[ipi][1],
            -g_total_momentum_list[iptot][2] - g_source_momentum_list[ipi][2] };
 
          memcpy ( tp->pi1, g_source_momentum_list[ipi], 3*sizeof(int) );
          memcpy ( tp->pi2, pi2, 3*sizeof(int) );
          memcpy ( tp->pf1, g_sink_momentum_list[ipf], 3*sizeof(int) );
          memcpy ( tp->pf2, pc, 3*sizeof(int) );

          if ( ! momentum_filter ( tp ) ) continue;


          char output_filename[2000];
            
          get_obs_name ( output_filename, tp , g_src_snk_time_separation, "corr" );

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
                  corr_latsym[i_2pt][iptot][ipf][ipi][iconf][isrc*g_coherent_source_number+icoh][2*it  ],
                  corr_latsym[i_2pt][iptot][ipf][ipi][iconf][isrc*g_coherent_source_number+icoh][2*it+1] );
                } 

              }
            }
          }
          fclose ( ofs );
        }  /* pi */
      }  /* ptot */
    }  /* pf */
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

    /***************************************************************************
     * UWerr analysis for latsym-averaged observables
     ***************************************************************************/
    for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {
     
          int pc[3] = {
            g_total_momentum_list[iptot][0] - g_sink_momentum_list[ipf][0],
            g_total_momentum_list[iptot][1] - g_sink_momentum_list[ipf][1],
            g_total_momentum_list[iptot][2] - g_sink_momentum_list[ipf][2] };

          int pi2[3] = {
            -g_total_momentum_list[iptot][0] - g_source_momentum_list[ipi][0],
            -g_total_momentum_list[iptot][1] - g_source_momentum_list[ipi][1],
            -g_total_momentum_list[iptot][2] - g_source_momentum_list[ipi][2] };

          memcpy ( tp->pi1, g_source_momentum_list[ipi], 3*sizeof(int) );
          memcpy ( tp->pi2, pi2,                         3*sizeof(int) );
          memcpy ( tp->pf1, g_sink_momentum_list[ipf],   3*sizeof(int) );
          memcpy ( tp->pf2, pc,                          3*sizeof(int) );

          if ( ! momentum_filter ( tp ) ) continue;

          for ( int ireim =0; ireim < 2; ireim++ ) {
            gettimeofday ( &ta, (struct timezone *)NULL );

            int const nmeas = num_conf * num_src_per_conf * g_coherent_source_number;
            double ** data = init_2level_dtable ( nmeas, n_tc );
            if ( data == NULL ) {
              fprintf ( stderr, "[htpp_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
              EXIT(12);
            }

#pragma omp parallel for shared(fold_correlator)
            for ( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                for ( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {
                  for ( int it = 0; it < n_tc; it++ ) {
                    data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it] = corr_latsym[i_2pt][iptot][ipf][ipi][iconf][isrc*g_coherent_source_number+icoh][2*it+ireim];
                  }

                  /***************************************************************************
                   * fold correlator·, use time reversal property
                   ***************************************************************************/
                  if ( (strcmp ( tp->type , "m-m" ) == 0) && fold_correlator ) {
                    for ( int it = 1; it < n_tc/2; it++ ) {
                      int const itt = ( n_tc - it ) % n_tc;
                      double const dtmp = 0.5 * ( 
                            data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it] 
                          + gamma_timereversal_sign[tp->gi1[0]] * gamma_timereversal_sign[tp->gf1[0]] * data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][itt] );
                      data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it]  = dtmp;
                      data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][itt] = dtmp * gamma_timereversal_sign[tp->gi1[0]] * gamma_timereversal_sign[tp->gf1[0]];
                    }
                  }

                }
              }
            }  /* end of loop on num_conf */

            char obs_name[2000];

            get_obs_name ( obs_name, tp , g_src_snk_time_separation, reim_str[ireim] );
            strcat ( obs_name, ".latsymavg" );

            exitstatus = apply_uwerr_real (  data[0], nmeas, n_tc, 0, 1, obs_name );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[htpp_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(14);
            }

           /***************************************************************************
            * effective mass analysis
            ***************************************************************************/
            if ( strcmp( "m-m", tp->type ) == 0 ) {
            
              int const Thp1 = n_tc / 2 + 1;
     
              for ( int itau = 1; itau < Thp1/2; itau++ ) {
                int narg = 3;
                int arg_first[3] = { 0, 2 * itau, itau };
                int arg_stride[3] = {1,1,1};
                int nT = Thp1 - 2 * itau;

                char obs_name2[2000];
                sprintf ( obs_name2, "%s.acosh_ratio.tau%d", obs_name, itau );

                exitstatus = apply_uwerr_func ( data[0], num_conf, n_tc, nT, narg, arg_first, arg_stride, obs_name, acosh_ratio, dacosh_ratio );
                if ( exitstatus != 0 ) {
                  fprintf ( stderr, "[htpp_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(115);
                }

              }

            }

            fini_2level_dtable ( &data );
  
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "htpp_analyse", "stats-uwerr-analysis", io_proc == 2 );

          }  /* end of loop on ireim */
        }  /* ipi */
      }  /* iptot */
    }  /* ipf */

    /***************************************************************************
     * UWerr analysis for non-latsy-averaged data
     ***************************************************************************/
    for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

          for ( int iparity = 0; iparity < 2; iparity++ ) {
            int const sparity = 1 - 2 * iparity;

            int pf[3] = {
              sparity * g_sink_momentum_list[ipf][0],
              sparity * g_sink_momentum_list[ipf][1],
              sparity * g_sink_momentum_list[ipf][2] };

            int ptot[3] =  {
              sparity * g_total_momentum_list[iptot][0],
              sparity * g_total_momentum_list[iptot][1],
              sparity * g_total_momentum_list[iptot][2] };
     
            int pc[3] = {
              ptot[0] - pf[0],
              ptot[1] - pf[1],
              ptot[2] - pf[2] };

            int pi1[3] = {
             sparity * g_source_momentum_list[ipi][0],
             sparity * g_source_momentum_list[ipi][1],
             sparity * g_source_momentum_list[ipi][2] };

            int pi2[3] = {
              -ptot[0] - pi1[0],
              -ptot[1] - pi1[1],
              -ptot[2] - pi1[2] };

            memcpy ( tp->pi1, pi1, 3*sizeof(int) );
            memcpy ( tp->pi2, pi2, 3*sizeof(int) );
            memcpy ( tp->pf1, pf,  3*sizeof(int) );
            memcpy ( tp->pf2, pc,  3*sizeof(int) );

            if ( ! momentum_filter ( tp ) ) continue;


            for ( int ireim =0; ireim < 2; ireim++ ) {
              gettimeofday ( &ta, (struct timezone *)NULL );

              int const nmeas = num_conf * num_src_per_conf * g_coherent_source_number;
              double ** data = init_2level_dtable ( nmeas, n_tc );
              if ( data == NULL ) {
                fprintf ( stderr, "[htpp_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
                EXIT(12);
              }

#pragma omp parallel for shared(fold_correlator)
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                  for ( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {
                    for ( int it = 0; it < n_tc; it++ ) {
                      data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it] = corr[i_2pt][iptot][ipf][ipi][iparity][iconf][isrc*g_coherent_source_number+icoh][2*it+ireim];
                    }

                    /***************************************************************************
                     * fold correlator·, use time reversal property
                     ***************************************************************************/
                    if ( (strcmp ( tp->type , "m-m" ) == 0) && fold_correlator ) {
                      for ( int it = 1; it < n_tc/2; it++ ) {
                        int const itt = ( n_tc - it ) % n_tc;
                        double const dtmp = 0.5 * ( 
                              data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it] 
                            + gamma_timereversal_sign[tp->gi1[0]] * gamma_timereversal_sign[tp->gf1[0]] * data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][itt] );
                        data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it]  = dtmp;
                        data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][itt] = dtmp * gamma_timereversal_sign[tp->gi1[0]] * gamma_timereversal_sign[tp->gf1[0]];
                      }
                    }

                  }
                }
              }  /* end of loop on num_conf */

              char obs_name[2000];

              get_obs_name ( obs_name, tp , g_src_snk_time_separation, reim_str[ireim] );

              exitstatus = apply_uwerr_real (  data[0], nmeas, n_tc, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[htpp_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(14);
              }

              fini_2level_dtable ( &data );
  
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_analyse", "stats-uwerr-analysis", io_proc == 2 );

            }  /* end of loop on ireim */
          }  /* end of loop on iparity */
        }  /* ipi */
      }  /* iptot */
    }  /* ipf */

    /***************************************************************************
     * simple orbit average for m-m 2pt
     ***************************************************************************/
    if ( strcmp( "m-m", tp->type ) == 0 ) {

      for ( int ireim =0; ireim < 2; ireim++ ) {

        int const nmeas = num_conf * num_src_per_conf * g_coherent_source_number;
        double ** data = init_2level_dtable ( nmeas, n_tc );
        if ( data == NULL ) {
          fprintf ( stderr, "[htpp_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(12);
        }

#pragma omp parallel for shared(fold_correlator)
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            for ( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {
              for ( int it = 0; it < n_tc; it++ ) {

                for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
                  int pf[3] = {  g_sink_momentum_list[ipf][0],  g_sink_momentum_list[ipf][1],  g_sink_momentum_list[ipf][2] };
                  int pi[3] = { -g_sink_momentum_list[ipf][0], -g_sink_momentum_list[ipf][1], -g_sink_momentum_list[ipf][2] };

                  int const ipi   = get_momentum_id ( pi, g_source_momentum_list, g_source_momentum_number );
                  int const iptot = get_momentum_id ( pf, g_total_momentum_list, g_total_momentum_number );
                  if ( ipi == -1 || iptot == -1 ) {
                    fprintf ( stderr, "[htpp_analyse] Error from get_momentum_id %s %d\n", __FILE__, __LINE__ );
                    EXIT(123);
                  } else if ( g_verbose > 4 && iconf == 0 && isrc == 0 && icoh == 0 ) 
                    fprintf( stdout, "# [htpp_analyse] ipf %d   ipi %d   iptot %d %s %d\n", ipf, ipi, iptot, __FILE__, __LINE__ );

                  data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it] += corr_latsym[i_2pt][iptot][ipf][ipi][iconf][isrc*g_coherent_source_number+icoh][2*it+ireim];
                }
                data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it] /= (double)g_sink_momentum_number;
              }

              /***************************************************************************
               * fold correlator·, use time reversal property
               ***************************************************************************/
              for ( int it = 1; it < n_tc/2; it++ ) {
                int const itt = ( n_tc - it ) % n_tc;
                double const dtmp = 0.5 * (
                      data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it]
                    + gamma_timereversal_sign[tp->gi1[0]] * gamma_timereversal_sign[tp->gf1[0]] * data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][itt] );
                data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][it]  = dtmp;
                data[(iconf*num_src_per_conf+isrc)*g_coherent_source_number+icoh][itt] = dtmp * gamma_timereversal_sign[tp->gi1[0]] * gamma_timereversal_sign[tp->gf1[0]];
              }
            }

          }
        } /* end of loop on num_conf */

        char obs_name[2000];
      
        sprintf ( obs_name,
            "%s_gf_%s_pfx%dpfy%dpfz%d_gi1_%s_pi1x%dpi1y%dpi1z%d.%s.orbit",
            tp->name,
            gamma_bin_to_name[tp->gf1[0]],
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
            gamma_bin_to_name[tp->gi1[0]],
            -g_sink_momentum_list[0][0], -g_sink_momentum_list[0][1], -g_sink_momentum_list[0][2],
            reim_str[ireim] );


        exitstatus = apply_uwerr_real (  data[0], nmeas, n_tc, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[htpp_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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

          exitstatus = apply_uwerr_func ( data[0], num_conf, n_tc, nT, narg, arg_first, arg_stride, obs_name, acosh_ratio, dacosh_ratio );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[htpp_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }

        }


        fini_2level_dtable ( &data );
      }  /* of ireim */

    }  /* end of if type = m-m */



  }  /* i2pt */

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the correlator field
   ***************************************************************************/
  fini_8level_dtable ( &corr );
  fini_7level_dtable ( &corr_latsym );

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
  show_time ( &start_time, &end_time, "htpp_analyse", "total-time", io_proc == 2 );

  return(0);

}
