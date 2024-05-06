/****************************************************
 * avx23_conn_analyse
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
#include "gamma.h"
#include "uwerr.h"
#include "derived_quantities.h"

#ifndef _SQR
#define _SQR(_a) ((_a)*(_a))
#endif

#define _TWOP_STATS     1

#define _TWOP_AVGX_H5   1

#define _THREEP_AVGX_H5 1

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse x23 correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}

/**********************************************************
 *
 **********************************************************/
inline int get_momentum_id ( int const q[3], int ** const p, int const n )
{
  int id = -1;
  for ( int i = 0; i < n; i++ ) {
    if ( ( q[0] == p[i][0] ) && ( q[1] == p[i][1] ) && ( q[2] == p[i][2] )  )
    {
      id = i;
      break;
    }
  }
  
  if ( id == -1 ) {
    fprintf(stderr, "[get_momentum_id] Error, momentum %3d %3d %3d not found   %s %d\n", q[0], q[1], q[2], __FILE__, __LINE__);
  } else if (g_verbose > 4 ) {
    fprintf( stdout, "# [get_momentum_id] momentum %3d %3d %3d id %2d    %s %d\n", q[0], q[1], q[2], id, __FILE__, __LINE__);
  }

  return(id);
}

/**********************************************************
 *
 **********************************************************/
int main(int argc, char **argv) {
  
  char const reim_str[2][3] = { "re", "im" };

  char const threep_tag[1][12] = { "4ik" };

  char const fbwd_str[2][4] = { "fwd", "bwd" };

  char const flavor_type_2pt[3][12] = { "l-gf-l-gi", "l-gf-s-gi",  "s-gf-l-gi" };

  char const flavor_type_3pt[3][16] = { "DDl-gc-ll-gi" , "DDl-gc-ls-gi" ,  "DDs-gc-sl-gi" };


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[600];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  int twop_fold_propagator = 0;
  int write_data = 0;

  double twop_weight[2]   = {0., 0.};
  /* double fbwd_weight[2]   = {1., 0.}; */
  /* double mirror_weight[2] = {0., 0.}; */

  double g_mus = 0.;

  int flavor_id_3pt = -1;

  int flavor_id_2pt = -1;

  double threep_operator_norm = 1.;

  struct timeval ta, tb, starttime, endtime;

  while ((c = getopt(argc, argv, "h?f:N:S:E:w:F:T:m:i:I:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [avx23_conn_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [avx23_conn_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [avx23_conn_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
   case 'F':
      twop_fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [avx23_conn_analyse] twop fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [avx23_conn_analyse] write_date set to %d\n", write_data );
      break;
    case 'T':
      sscanf( optarg, "%lf,%lf", twop_weight, twop_weight+1 );
      fprintf ( stdout, "# [avx23_conn_analyse] twop_weight set to %25.16e / %25.16e\n", twop_weight[0], twop_weight[1] );
      break;
/*    case 'B':
      sscanf( optarg, "%lf,%lf", fbwd_weight, fbwd_weight+1 );
      fprintf ( stdout, "# [avx23_conn_analyse] fbwd_weight set to %25.16e / %25.16e\n", fbwd_weight[0], fbwd_weight[1] );
      break;
    case 'M':
      sscanf( optarg, "%lf,%lf", mirror_weight, mirror_weight+1 );
      fprintf ( stdout, "# [avx23_conn_analyse] mirror_weight set to %25.16e / %25.16e\n", mirror_weight[0], mirror_weight[1] );
      break;
*/
    case 'm':
      g_mus = atof ( optarg );
      break;
    case 'i':
      flavor_id_2pt = atoi ( optarg );
      fprintf ( stdout, "# [avx23_conn_analyse] flavor_id_2pt set to %d \n", flavor_id_2pt );
      break;
    case 'I':
      flavor_id_3pt = atoi ( optarg );
      fprintf ( stdout, "# [avx23_conn_analyse] flavor_id_3pt set to %d \n", flavor_id_3pt );
      break;
    case 'n':
      threep_operator_norm = atof ( optarg );
      fprintf ( stdout, "# [avx23_conn_analyse] threep_operator_norm set to %e \n", threep_operator_norm );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }
  
  gettimeofday ( &starttime, (struct timezone *)NULL );

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cpff.input");
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [avx23_conn_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [avx23_conn_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [avx23_conn_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[avx23_conn_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[avx23_conn_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[avx23_conn_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [avx23_conn_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[avx23_conn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[avx23_conn_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [avx23_conn_analyse] comment %s\n", line );
      continue;
    }
    int itmp[5];
    char ctmp;

    sscanf( line, "%c %d %d %d %d %d", &ctmp, itmp, itmp+1, itmp+2, itmp+3, itmp+4 );

    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][0] = (int)ctmp;
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][1] = itmp[0];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][2] = itmp[1];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][3] = itmp[2];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][4] = itmp[3];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][5] = itmp[4];

    count++;
  }

  fclose ( ofs );

  if ( g_verbose > 3 ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        fprintf ( stdout, "conf_src_list %c %6d %3d %3d %3d %3d\n", 
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] );
      }
    }
  }

  /**********************************************************
   * set quark masses
   **********************************************************/
  double muval_2pt_list[2] = { 0., 0. };
  double muval_3pt_list[3] = {0., 0., 0. };

  switch ( flavor_id_2pt )
  {
    case 0:
      muval_2pt_list[0] = g_mu;
      muval_2pt_list[1] = g_mu;
    case 1:
      muval_2pt_list[0] = g_mu;
      muval_2pt_list[1] = g_mus;
      break;
    case 2:
      muval_2pt_list[0] = g_mus;
      muval_2pt_list[1] = g_mu;
      break;
    default:
      if ( io_proc == 2 ) fprintf ( stderr, "[avx23_conn_analyse] flavor_id_2pt = %d not implemented   %s %d\n", flavor_id_2pt, __FILE__, __LINE__ );
      EXIT(1);
      break;
  }

  switch ( flavor_id_3pt )
  {
    case 2:
      muval_3pt_list[0] = g_mus;
      muval_3pt_list[1] = g_mus;
      muval_3pt_list[2] = g_mu;
      break;
    case 1:
      muval_3pt_list[0] = g_mu;
      muval_3pt_list[1] = g_mu;
      muval_3pt_list[2] = g_mus;
      break;
    case 0:
      muval_3pt_list[0] = g_mu;
      muval_3pt_list[1] = g_mu;
      muval_3pt_list[2] = g_mu;
      break;
    default:
      if ( io_proc == 2 ) fprintf ( stderr, "[avx23_conn_analyse] flavor_id_3pt = %d not implemented   %s %d\n", flavor_id_3pt, __FILE__, __LINE__ );
      EXIT(1);
      break;
  }


  /**********************************************************
   **********************************************************
   ** 
   ** READ DATA
   ** 
   ** for two-point function
   **
   **********************************************************
   **********************************************************/
  double ***** twop = NULL;

  twop = init_5level_dtable ( g_sink_momentum_number, num_conf, num_src_per_conf, T_global, 2 );
  if( twop == NULL ) {
    fprintf ( stderr, "[avx23_conn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

/**********************************************************/
#if _TWOP_AVGX_H5
/**********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***********************************************************
   * loop on configs
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {

    double *** buffer = init_3level_dtable ( 2, T_global, 2 );

    /***********************************************************
     * loop on sources
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
    {

      /***********************************************************
       * open AFF reader
       ***********************************************************/
      char data_filename[500];
    
      /* avgx23.1300.t111.s2.h5 */
      /* sprintf( data_filename, "stream_%c/%s/%d/%s.%.4d.t%d.s%d.h5",
          conf_src_list[iconf][isrc][0],
          filename_prefix,
          conf_src_list[iconf][isrc][1],
          filename_prefix2,
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3] );
      */
      sprintf( data_filename, "stream_%c/%d/%s.%.4d.t%d.s%d.h5",
          conf_src_list[iconf][isrc][0],
          conf_src_list[iconf][isrc][1],
          filename_prefix2,
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3] );

      /***********************************************************
       * loop on sink momenta
       ***********************************************************/
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        int pf[3] = {
          g_sink_momentum_list[ipf][0],
          g_sink_momentum_list[ipf][1],
          g_sink_momentum_list[ipf][2] 
        };

        /* int pi[3] = { -pf[0], -pf[1], -pf[2] }; */

        char key[400], key2[400];

        if ( twop_weight[0] != 0. ) 
        {
          /* /l-gf-s-gi/mu-0.0007/mu0.0186/px-1py1pz0 */
          
          sprintf( key,  "/%s/mu%6.4f/mu%6.4f/px%dpy%dpz%d",
              flavor_type_2pt[flavor_id_2pt],
              -muval_2pt_list[0], muval_2pt_list[1],
              pf[0], pf[1], pf[2] );
          
          if ( g_verbose > 3 ) fprintf ( stdout, "# [avx23_conn_analyse] key  = %s %s %d\n", key, __FILE__, __LINE__  );

          exitstatus = read_from_h5_file ( (void*)(buffer[0][0]), data_filename, key, "double", io_proc );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[avx23_conn_analyse] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", 
                data_filename, key, exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
        }  /* end of twop_weight 0 */

        if ( twop_weight[1] != 0. ) 
        {

          sprintf( key2, "/%s/mu%6.4f/mu%6.4f/px%dpy%dpz%d",
              flavor_type_2pt[flavor_id_2pt],
              muval_2pt_list[0], -muval_2pt_list[1],
              pf[0], pf[1], pf[2] );

          if ( g_verbose > 3 ) fprintf ( stdout, "# [avx23_conn_analyse] key2 = %s %s %d\n", key2, __FILE__, __LINE__  );

          exitstatus = read_from_h5_file ( (void*)(buffer[1][0]), data_filename, key2, "double", io_proc );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[avx23_conn_analyse] Error from read_from_h5_file for file %s key2 %s, status was %d %s %d\n",
                data_filename, key2, exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
        }

        /***********************************************************
         * NOTE: NO SOURCE PHASE NECESSARY
         * ONLY REORDERING from source
         ***********************************************************/
#pragma omp parallel for
        for ( int it = 0; it < T_global; it++ ) {
          int const itt = ( it + conf_src_list[iconf][isrc][2] + T_global ) % T_global;
          twop[ipf][iconf][isrc][it][0] = ( twop_weight[0] * buffer[0][itt][0] + twop_weight[1] * buffer[1][itt][0] ) / ( fabs( twop_weight[0] ) + fabs( twop_weight[1] ) ); 
          twop[ipf][iconf][isrc][it][1] = ( twop_weight[0] * buffer[0][itt][1] + twop_weight[1] * buffer[1][itt][1] ) / ( fabs( twop_weight[0] ) + fabs( twop_weight[1] ) ); 
        }

      }  /* end of loop on sink momenta */

    }  /* end of loop on sources */

    fini_3level_dtable ( &buffer );
  }  /* end of loop on configs */
          
  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "avx23_conn_analyse", "read-twop-h5", g_cart_id == 0 );

#endif  /* end of _TWOP_AVGX_H5 */

  /**********************************************************/
  /**********************************************************/

  /**********************************************************
   * average 2-pt over momentum orbit
   **********************************************************/

  double **** twop_orbit = init_4level_dtable ( num_conf, num_src_per_conf, T_global, 2 );
  if( twop_orbit == NULL ) {
    fprintf ( stderr, "[avx23_conn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

#pragma omp parallel for
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      /* averaging starts here */
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

        for ( int it = 0; it < T_global; it++ ) {

          twop_orbit[iconf][isrc][it][0] += twop[imom][iconf][isrc][it][0];
          twop_orbit[iconf][isrc][it][1] += twop[imom][iconf][isrc][it][1];

        }  /* end of loop on it */
      }  /* end of loop on imom */

      /* multiply norm from averages over momentum orbit and source locations */
      double const norm = 1. / (double)g_sink_momentum_number;
      for ( int it = 0; it < 2*T_global; it++ ) {
        twop_orbit[iconf][isrc][0][it] *= norm;
      }
    }  /* end of loop on isrc */
  }  /* end of loop on iconf */

  /**********************************************************
   * write orbit-averaged data to ascii file, per source
   **********************************************************/
  if ( write_data == 1 ) {
    for ( int ireim = 0; ireim <=0; ireim++ ) {
      sprintf ( filename, "twop.%s.pseudoscalar.orbit.PX%d_PY%d_PZ%d.%s.corr", 
          flavor_type_2pt[flavor_id_2pt],
          g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
          reim_str[ireim]);
      FILE * fs = fopen( filename, "w" );
      for( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

          for ( int it = 0; it < T_global; it++ ) {
            fprintf( fs, "%4d %25.16e\n", it, twop_orbit[iconf][isrc][it][ireim] );
          }
        }
      }
      fclose ( fs );
    }  /* end of loop on reim */
  }  /* end of if write data */

#ifdef _TWOP_STATS
  /**********************************************************
   * 
   * STATISTICAL ANALYSIS
   * 
   **********************************************************/
  for ( int ireim = 0; ireim < 1; ireim++ ) {  /* real part only */

    double ** data = init_2level_dtable ( num_conf, T_global );
    if ( data == NULL ) {
      fprintf ( stderr, "[avx23_conn_analyse] Error from init_Xlevel_dtable %s %d\n",  __FILE__, __LINE__ );
      EXIT(1);
    }

    /* fill data array */
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {

      for ( int it = 0; it < T_global; it++ ) {
        int const itt = ( T_global - it ) % T_global;

        data[iconf][it ] = 0.;

        for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
          data[iconf][it] += twop_orbit[iconf][isrc][it][ireim] + twop_fold_propagator * twop_orbit[iconf][isrc][itt][ireim];
        } 
        data[iconf][it ] /= (double)num_src_per_conf * ( 1 + abs( twop_fold_propagator ) );
      }
    }

    char obs_name[100];
    sprintf( obs_name, "twop.%s.pseudoscalar.orbit.PX%d_PY%d_PZ%d.%s",
        flavor_type_2pt[flavor_id_2pt],
          g_sink_momentum_list[0][0],
          g_sink_momentum_list[0][1],
          g_sink_momentum_list[0][2], reim_str[ireim] );

    /**********************************************************
     * write data to ascii file
     **********************************************************/
    if ( write_data )
    {
      sprintf ( filename, "%s.corr", obs_name );

      FILE * fs = fopen( filename, "w" );
      for( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int it = 0; it < T_global; it++ ) {
          fprintf( fs, "%4d %25.16e %c %6d\n", it, data[iconf][it], conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
        }
      }
      fclose ( fs );
    }  /* end of if write data */


    if ( num_conf < 6 ) 
    {
      fprintf ( stderr, "[avx23_conn_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
      continue;
    } else {

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avx23_conn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }
  
      /**********************************************************
       * acosh ratio for m_eff
       **********************************************************/
      int const Thp1 = T_global / 2 + 1;
      for ( int itau = 1; itau < Thp1/2; itau++ ) {
        int narg = 3;
        int arg_first[3] = { 0, 2 * itau, itau };
        int arg_stride[3] = {1,1,1};
        int nT = Thp1 - 2 * itau;
  
        char obs_name2[100];
        sprintf ( obs_name2, "%s.acosh_ratio.tau%d", obs_name, itau );
  
        exitstatus = apply_uwerr_func ( data[0], num_conf, T_global, nT, narg, arg_first, arg_stride, obs_name2, acosh_ratio, dacosh_ratio );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[avx23_conn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(115);
        }
      }
    }  /* end of if num_conf < 6 */

    fini_2level_dtable ( &data );
  }  /* end of loop on reim */

#endif  /* of ifdef _TWOP_STATS */

  /**********************************************************
   * loop on source - sink time separations
   **********************************************************/
  for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) 
  {

    /**********************************************************
     * double-derivative and gamma index list
     *
     * [D^fb2_nu D^fb1_mu q]^+ gamma_kappa q = q^+ D^fb1_mu D^fb2_nu gamma_kappa q
     *
     * idx_map[][0] = mu  ( first derivative applied )
     * idx_map[][1] = nu  ( second derivative applied )
     *
     * forward / backward index for shift from 
     *
     * buffer[k][t][idd][fb2][fb1][kappa]
     **********************************************************/
    int const idx_num = 24;
    int const idx_map[24][3] = {
      { 0, 1, 2 },  /*  0 */
      { 0, 1, 3 },  /*  1 */
      { 0, 2, 1 },  /*  2 */
      { 0, 2, 3 },  /*  3 */
      { 0, 3, 1 },  /*  4 */
      { 0, 3, 2 },  /*  5 */
      { 1, 0, 2 },  /*  6 */
      { 1, 0, 3 },  /*  7 */
      { 1, 2, 0 },  /*  8 */
      { 1, 2, 3 },  /*  9 */
      { 1, 3, 0 },  /* 10 */
      { 1, 3, 2 },  /* 11 */
      { 2, 0, 1 },  /* 12 */
      { 2, 0, 3 },  /* 13 */
      { 2, 1, 0 },  /* 14 */
      { 2, 1, 3 },  /* 15 */
      { 2, 3, 0 },  /* 16 */
      { 2, 3, 1 },  /* 17 */
      { 3, 0, 1 },  /* 18 */
      { 3, 0, 2 },  /* 19 */
      { 3, 1, 0 },  /* 20 */
      { 3, 1, 2 },  /* 21 */
      { 3, 2, 0 },  /* 22 */
      { 3, 2, 1 }   /* 23 */
    };

    int const idx_perm[4][6] = {
      /* 0, 1, 2 */
      {  0,  2,  6,  8, 12, 14 },
      /* 0, 1, 3 */
      {  1,  4,  7, 10, 18, 20 },
      /* 0, 2, 3 */
      {  3,  5, 13, 16, 19, 22 },
      /* 1, 2, 3 */
      {  9, 11, 15, 17, 21, 23 }
    };

    double ****** threep = init_6level_dtable ( g_sink_momentum_number, num_conf, num_src_per_conf, idx_num, T_global, 2 );
    if ( threep == NULL ) {
      fprintf( stderr, "[avx23_conn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

/**********************************************************/
#if _THREEP_AVGX_H5
/**********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );
  
    /***********************************************************
     * loop on configs
     ***********************************************************/
    for ( int iconf = 0; iconf < num_conf; iconf++ ) 
    {
      double ****** buffer = init_6level_dtable ( 2, T_global, 12, 2, 2, 4 );
  
      /***********************************************************
       * loop on sources
       ***********************************************************/
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
      {
        /***********************************************************
         * open h5 reader
         ***********************************************************/
        char data_filename[500];
  
        /* sprintf( data_filename, "stream_%c/%s/%d/%s.%.4d.t%d.s%d.h5",
            conf_src_list[iconf][isrc][0],
            filename_prefix,
            conf_src_list[iconf][isrc][1],
            filename_prefix2,
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3] );
          */
 
        sprintf( data_filename, "stream_%c/%d/%s.%.4d.t%d.s%d.h5",
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            filename_prefix2,
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3] );
 
        if ( g_verbose > 2 ) fprintf ( stdout, "# [avx23_conn_analyse] reading from data filename %s %s %d\n", data_filename, __FILE__, __LINE__ );

        /***********************************************************
         * loop on sink momenta
         ***********************************************************/
        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
  
          int pf[3] = {
              g_sink_momentum_list[ipf][0],
              g_sink_momentum_list[ipf][1],
              g_sink_momentum_list[ipf][2] 
            };

          int pc[3] = {
            g_insertion_momentum_list[0][0],
            g_insertion_momentum_list[0][1],
            g_insertion_momentum_list[0][2] };
  
          int pi[3] = {
            -( pf[0] + pc[0] ),
            -( pf[1] + pc[1] ),
            -( pf[2] + pc[2] )
          };

          char key[400];

          if ( twop_weight[0] != 0. ) 
          {
            /***********************************************************
             * read h5 data key ...
             ***********************************************************/
            /* /DDl-gc-ll-gi/mu-0.0007/mu-0.0007/mu0.0007/dt48/px-1py-1pz0 */

            sprintf( key, "/%s/mu%6.4f/mu%6.4f/mu%6.4f/dt%d/px%dpy%dpz%d",
                flavor_type_3pt[flavor_id_3pt],
                -muval_3pt_list[0], -muval_3pt_list[1], muval_3pt_list[2],
                g_sequential_source_timeslice_list[idt],
                pf[0], pf[1], pf[2] );

            if ( g_verbose > 3 ) fprintf ( stdout, "# [avx23_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__ );
  
            exitstatus = read_from_h5_file ( (void*)(buffer[0][0][0][0][0]), data_filename, key,  "double", io_proc );
            if ( exitstatus != 0 ) {
              fprintf( stderr, "[avx23_conn_analyse] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", data_filename, key, exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }
          }  /* end of twop_weight 0 */

          if ( twop_weight[1] != 0. ) 
          {
            /***********************************************************
             * ... and twisted parity partner
             ***********************************************************/
            sprintf( key, "/%s/mu%6.4f/mu%6.4f/mu%6.4f/dt%d/px%dpy%dpz%d",
                flavor_type_3pt[flavor_id_3pt],
                muval_3pt_list[0], muval_3pt_list[1], -muval_3pt_list[2],
                g_sequential_source_timeslice_list[idt],
                -pf[0], -pf[1], -pf[2] );

            if ( g_verbose > 3 ) fprintf ( stdout, "# [avx23_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__ );
   
            exitstatus = read_from_h5_file ( (void*)(buffer[1][0][0][0][0]), data_filename, key,  "double", io_proc );
            if ( exitstatus != 0 ) {
              fprintf( stderr, "[]avx23_conn_analyse Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", data_filename, key, exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }
          }  /* end of twop_weight 1 */
  
          for ( int idd = 0; idd < 12; idd++ ) 
          {
            int const mu = idx_map[2*idd][0];
            int const nu = idx_map[2*idd][1];

            for ( int igc = 0; igc < 2; igc++ )
            {
              int const idx = 2 * idd + igc;
              int const kappa = idx_map[idx][2];

              /***********************************************************
               * NOTE: NO SOURCE PHASE NECESSARY
               * ONLY REORDERING NECESSARY
               ***********************************************************/
              double const threep_norm = 1. / ( fabs( twop_weight[0] ) + fabs( twop_weight[1] ) ) * threep_operator_norm;

              int const parity_sign = ( 2 * ( idx_map[idx][0] == 0 ) - 1 ) * ( 2 * ( idx_map[idx][1] == 0 ) - 1 ) * ( 2 * ( idx_map[idx][2] == 0 ) - 1 );

#pragma omp parallel for
              for ( int it = 0; it < T_global; it++ ) 
              {
                int const itt       = ( it + conf_src_list[iconf][isrc][2]               + 2*T_global ) % T_global;
                int const itt_pl_mu = ( it + conf_src_list[iconf][isrc][2] + ( mu == 0 ) + 2*T_global ) % T_global;
                int const itt_mi_mu = ( it + conf_src_list[iconf][isrc][2] - ( mu == 0 ) + 2*T_global ) % T_global;
                int const itt_pl_nu = ( it + conf_src_list[iconf][isrc][2] + ( nu == 0 ) + 2*T_global ) % T_global;
                int const itt_mi_nu = ( it + conf_src_list[iconf][isrc][2] - ( nu == 0 ) + 2*T_global ) % T_global;
              
                int const itt_pl_mu_pl_nu = ( it + conf_src_list[iconf][isrc][2] + ( mu == 0 ) + ( nu == 0 ) + 2*T_global ) % T_global;
                int const itt_pl_mu_mi_nu = ( it + conf_src_list[iconf][isrc][2] + ( mu == 0 ) - ( nu == 0 ) + 2*T_global ) % T_global;
                int const itt_mi_mu_pl_nu = ( it + conf_src_list[iconf][isrc][2] - ( mu == 0 ) + ( nu == 0 ) + 2*T_global ) % T_global;
                int const itt_mi_mu_mi_nu = ( it + conf_src_list[iconf][isrc][2] - ( mu == 0 ) - ( nu == 0 ) + 2*T_global ) % T_global;

                /***********************************************************
                 * 
                 ***********************************************************/
                double dtmp[2][2] = { 0., 0., 0., 0. };
                for ( int k = 0; k <= 1; k++ )              /* parity components */
                {
                  for ( int ireim = 0; ireim <= 1; ireim++ )  /* real and imaginary part */
                  {
                      /*               from (fwd+bwd)/2   from (Dright-Dleft)/2 */
                      dtmp[k][ireim] = 0.25             * 0.25 * ( 
                        /*
                         *   --> --> 
                         * +  D   D
                         *
                         */
                       + buffer[k][itt_pl_mu_pl_nu][idd][1][1][2*igc+ireim]
                       + buffer[k][itt_mi_mu_mi_nu][idd][0][0][2*igc+ireim]
                       - buffer[k][itt_pl_mu_mi_nu][idd][0][1][2*igc+ireim]
                       - buffer[k][itt_mi_mu_pl_nu][idd][1][0][2*igc+ireim]
                        /*
                         *   <-- <--
                         * +  D   D
                         *
                         */
                       + buffer[k][itt][idd][0][0][2*igc+ireim]
                       + buffer[k][itt][idd][1][1][2*igc+ireim]
                       - buffer[k][itt][idd][0][1][2*igc+ireim]
                       - buffer[k][itt][idd][1][0][2*igc+ireim]
                        /*
                         *   <-- -->
                         * -  D   D
                         *
                         */
                    -( + buffer[k][itt_pl_nu][idd][1][0][2*igc+ireim]
                       + buffer[k][itt_mi_nu][idd][0][1][2*igc+ireim]
                       - buffer[k][itt_pl_nu][idd][1][1][2*igc+ireim]
                       - buffer[k][itt_mi_nu][idd][0][0][2*igc+ireim] )
                        /*
                         *   --> <--
                         * -  D   D
                         *
                         */
                    -( + buffer[k][itt_pl_mu][idd][0][1][2*igc+ireim]
                       + buffer[k][itt_mi_mu][idd][1][0][2*igc+ireim]
                       - buffer[k][itt_pl_mu][idd][1][1][2*igc+ireim]
                       - buffer[k][itt_mi_mu][idd][0][0][2*igc+ireim] )
                    );
                  }  /* end of loop on real and imag part  */
                }  /* end of loop on k = 2 parity partners */
  
                /* threep[ipf][iconf][isrc][idx][it][0] = ( twop_weight[0] * dtmp[0][0] + parity_sign * twop_weight[1] * dtmp[1][0] ) * threep_norm;
                threep[ipf][iconf][isrc][idx][it][1] = ( twop_weight[0] * dtmp[0][1] + parity_sign * twop_weight[1] * dtmp[1][1] ) * threep_norm;
                */
                threep[ipf][iconf][isrc][idx][it][0] = threep_norm * (
                      twop_weight[0] * 0.5 * ( 1 + parity_sign ) * dtmp[0][0] 
                    + twop_weight[1] * 0.5 * ( 1 + parity_sign ) * dtmp[1][0] ) ;

                threep[ipf][iconf][isrc][idx][it][1] = threep_norm * (
                      twop_weight[0] * 0.5 * ( 1 - parity_sign ) * dtmp[0][1] 
                    - twop_weight[1] * 0.5 * ( 1 - parity_sign ) * dtmp[1][1] );
              }  /* end of loop on timeslices */
            }  /* end of loop on igc, gamma */
  
          }  /* end of loop on idd, double derivative */

        }  /* end of loop on sink momenta */

      }  /* end of loop on sources */
  
      fini_6level_dtable ( &buffer );
    }  /* end of loop on configs */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "avx23_conn_analyse", "read-threep-h5", g_cart_id == 0 );

#endif  /* end of if _THREEP_AVGX_H5 */

    /**********************************************************
     * show basic 3-pt function
     **********************************************************/
    for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
    {
      for ( int k = 0; k < 24; k++ )
      {
        for ( int ireim = 0; ireim < 2; ireim++ )
        {
          char obs_name[100];

          sprintf ( obs_name, "threep.%s.conn.gdd%d%d%d.dtsnk%d.PX%d_PY%d_PZ%d.%s",
                    flavor_type_3pt[flavor_id_3pt],
                    idx_map[k][0], idx_map[k][1], idx_map[k][2],
                    g_sequential_source_timeslice_list[idt],
                    g_sink_momentum_list[imom][0],
                    g_sink_momentum_list[imom][1],
                    g_sink_momentum_list[imom][2], reim_str[ireim] );

          sprintf ( filename, "%s.corr", obs_name );

          FILE * fs = fopen ( filename, "w" );

          for ( int iconf = 0; iconf < num_conf; iconf++ )
          {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ )
            {
              for ( int it = 0; it < T_global; it++ )
              {
                fprintf ( fs, "%3d %25.16e %c %6d   %3d %3d %3d %3d\n", it, threep[imom][iconf][isrc][k][it][ireim], 
                    conf_src_list[iconf][isrc][0],
                    conf_src_list[iconf][isrc][1],
                    conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
              }
            }
          }
 
          double ** data = init_2level_dtable ( num_conf, T_global );
          if ( data == NULL ) {
            fprintf ( stderr, "[avx23_conn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }

#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) 
          {
            for ( int it = 0; it < T_global; it++ )
            {
              double dtmp = 0.;
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                dtmp += threep[imom][iconf][isrc][k][it][ireim];
              }
              data[iconf][it] = dtmp / (double)num_src_per_conf;
            }
          }

          if ( num_conf >=  6 ) {
            /* apply UWerr analysis */
            exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[avx23_conn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }
          } else {
            fprintf ( stderr, "[avx23_conn_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
          }

          fini_2level_dtable ( &data );
        }
      }
    }  /* end of loop on imom */

    /**********************************************************
     * symmetrization of threep
     **********************************************************/

    double ****** threep_sym = init_6level_dtable ( g_sink_momentum_number, num_conf, num_src_per_conf, 4, T_global, 2 );
    if ( threep_sym == NULL ) {
      fprintf( stderr, "[avx23_conn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) 
    {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for ( int iconf = 0; iconf < num_conf; iconf++ )
      {
        for ( int isrc = 0; isrc < num_src_per_conf; isrc++ )
        {
          for ( int i = 0; i < 4; i++ )
          {
            for ( int it = 0; it < T_global; it++ )
            {
              for ( int ireim = 0; ireim < 2; ireim++ )
              {
                double dtmp = 0.;
                for ( int k = 0; k < 6; k++ )
                {
                  dtmp += threep[imom][iconf][isrc][idx_perm[i][k]][it][ireim];
                }
                threep_sym[imom][iconf][isrc][i][it][ireim] = dtmp / 6.;
              }
            }
          }
        }
      }

      /**********************************************************
       * write 3pt function to ascii file, per source
       **********************************************************/
      if ( write_data == 1)
      {
        for ( int k = 0; k < 4; k++ )
        {
          for ( int ireim = 0; ireim < 2; ireim++ )
          {
            sprintf ( filename, "threep_sym.%s.conn.gdd%d%d%d.dtsnk%d.PX%d_PY%d_PZ%d.%s.corr",
                  flavor_type_3pt[flavor_id_3pt],
                  idx_map[idx_perm[k][0]][0], idx_map[idx_perm[k][0]][1], idx_map[idx_perm[k][0]][2],
                  g_sequential_source_timeslice_list[idt],
                  g_sink_momentum_list[imom][0],
                  g_sink_momentum_list[imom][1],
                  g_sink_momentum_list[imom][2], reim_str[ireim] );

            FILE * fs = fopen ( filename, "w" );
            for ( int iconf = 0; iconf < num_conf; iconf++ )
            {
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ )
              {
                for ( int it = 0; it < T_global; it++ )
                {
                  fprintf (fs, "%3d  %25.16e   %c %6d   %3d %3d %3d %3d\n", it, threep_sym[imom][iconf][isrc][k][it][ireim],
                     conf_src_list[iconf][isrc][0],
                     conf_src_list[iconf][isrc][1],
                     conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
                }
              }
            }
            fclose ( fs );
          }  /* end of loop on reim */
        }  /* end of loop on operators k */
      }  /* of  if write_data */
    }  /* end of loop on momentum */

    fini_6level_dtable ( &threep );

    /**********************************************************
     * momentum orbit average
     **********************************************************/
    double ***** threep_orbit = init_5level_dtable ( num_conf, num_src_per_conf, 4, T_global, 2 );
    if ( threep_orbit == NULL ) {
      fprintf( stderr, "[avx23_conn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( int iconf = 0; iconf < num_conf; iconf++ )
    {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ )
      {
        for ( int i = 0; i < 4; i++ )
        {
          for ( int it = 0; it < T_global; it++ )
          {
            for ( int ireim = 0; ireim < 2; ireim++ )
            {
              double dtmp = 0.;
              int counter = 0.;

              for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
              {
                
                double const ip[4] = {1,
                    g_sink_momentum_list[imom][0],
                    g_sink_momentum_list[imom][1],
                    g_sink_momentum_list[imom][2] };

                if ( ip[idx_map[idx_perm[i][0]][0]] * ip[idx_map[idx_perm[i][0]][1]] * ip[idx_map[idx_perm[i][0]][2]] == 0 )
                {
                  if ( g_verbose > 2 ) fprintf ( stdout, "# [] skip p = %3d %3d %3d for i = %d\n", 
                      ip[1], ip[2], ip[3], i );
                  continue;
                } else {
                  counter++;

                  double const p[4] = {ip[0],
                    ip[1] * 2. * M_PI / (double)LX_global,
                    ip[2] * 2. * M_PI / (double)LY_global,
                    ip[3] * 2. * M_PI / (double)LZ_global };

                  double const norm = p[idx_map[idx_perm[i][0]][0]] * p[idx_map[idx_perm[i][0]][1]] * p[idx_map[idx_perm[i][0]][2]];

                  dtmp += threep_sym[imom][iconf][isrc][i][it][ireim] / norm;
                }
            

              }

              /* double const q[3] = {
                    g_sink_momentum_list[0][0] * 2. * M_PI / (double)LX_global,
                    g_sink_momentum_list[0][1] * 2. * M_PI / (double)LY_global,
                    g_sink_momentum_list[0][2] * 2. * M_PI / (double)LZ_global };

              double const qq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];

              threep_orbit[iconf][isrc][i][it][ireim] = dtmp / ( (double)g_sink_momentum_number * qq * qq ); */

              threep_orbit[iconf][isrc][i][it][ireim] = dtmp / (double)counter;
            }
          }
        }
      }
    }

    for ( int k = 0; k < 4; k++ ) 
    {
      for ( int ireim = 0; ireim < 2; ireim++ ) 
      {
        char obs_name[100];
        sprintf ( obs_name, "threep_orbit.%s.conn.gdd%d%d%d.dtsnk%d.PX%d_PY%d_PZ%d.%s",
              flavor_type_3pt[flavor_id_3pt],
              idx_map[idx_perm[k][0]][0], idx_map[idx_perm[k][0]][1], idx_map[idx_perm[k][0]][2],
              g_sequential_source_timeslice_list[idt],
              g_sink_momentum_list[0][0],
              g_sink_momentum_list[0][1],
              g_sink_momentum_list[0][2], reim_str[ireim] );

        /**********************************************************
         * write 3pt function to ascii file, per source
         **********************************************************/
        if ( write_data == 1) 
        {
          sprintf ( filename, "%s.corr", obs_name );

          FILE * fs = fopen ( filename, "w" );
          for ( int iconf = 0; iconf < num_conf; iconf++ )
          {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ )
            {
              for ( int it = 0; it < T_global; it++ )
              {
                fprintf (fs, "%3d  %25.16e   %c %6d   %3d %3d %3d %3d\n", it, threep_orbit[iconf][isrc][k][it][ireim],
                   conf_src_list[iconf][isrc][0],
                   conf_src_list[iconf][isrc][1],
                   conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
              }
            }
          }
          fclose ( fs );
          
        }  /* end of if write_data */

        /**********************************************************
         * STATISTICAL ANALYSIS for threep
         * with fixed source - sink separation
         **********************************************************/

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf ( stderr, "[avx23_conn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          for ( int it = 0; it < T_global; it++ ) {
            double dtmp = 0.;
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              dtmp += threep_orbit[iconf][isrc][k][it][ireim];
            }
            data[iconf][it] = dtmp / (double)num_src_per_conf;
          }
        }

        if ( write_data == 1 ) 
        {
          sprintf ( filename, "%s.srcavg.corr", obs_name );
          
          FILE * fs = fopen ( filename, "w" );
          for ( int iconf = 0; iconf < num_conf; iconf++ )
          {
            for ( int it = 0; it < T_global; it++ )
            {
              fprintf (fs, "%3d  %25.16e   %c %6d\n", it, data[iconf][it],
                 conf_src_list[iconf][0][0],
                 conf_src_list[iconf][0][1] );
            }
          }
          fclose ( fs );
        }

        if ( num_conf >=  6 ) {
          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[avx23_conn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
        } else {
          fprintf ( stderr, "[avx23_conn_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
        }

        fini_2level_dtable ( &data );

      }  /* end of loop on ireim */
    }  /* end of loop on 3-index combinations */

    fini_5level_dtable ( &threep_orbit );

    fini_6level_dtable ( &threep_sym );

  }  /* end of loop on dt = source sink time separations */

  /**********************************************************/
  /**********************************************************/

  fini_5level_dtable ( &twop );
  fini_4level_dtable ( &twop_orbit );

  /**********************************************************
   * free and finalize
   **********************************************************/

  fini_3level_itable ( &conf_src_list );

  free_geometry();

  gettimeofday ( &endtime, (struct timezone *)NULL );
  show_time ( &starttime, &endtime, "avx23_conn_analyse", "avx23_conn_analyse", g_cart_id == 0 );

  return(0);
}
