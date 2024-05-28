/****************************************************
 * avxn_conn_analyse 
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

#define _RAT_METHOD 1
/* not used for conn */
/* #define _RAT_SUB_METHOD 0 */

#define _TWOP_STATS 1

#define _TWOP_AFF         0
#define _TWOP_H5          0
#define _TWOP_AVGX_H5     1
#define _TWOP_AVGX_PLEGMA 0

#define _THREEP_AFF          0
#define _THREEP_H5           0
#define _THREEP_AVGX_H5      1
#define _THREEP_AVGX_PLEGMA  0

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse cpff fht correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}

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
inline void write_data_real ( double ** const data, char * const filename, int *** const lst, unsigned int const n0, unsigned int const n1 ) {

  FILE * fs = fopen ( filename, "w" );
  if ( fs == NULL ) {
    fprintf ( stderr, "[write_data_real] Error from fopen %s %d\n",  __FILE__, __LINE__ );
    EXIT(1);
  }

  for ( unsigned int i0 = 0; i0 < n0; i0++ ) {
    fprintf ( fs, "# %c %6d\n", lst[i0][0][0], lst[i0][0][1] );
    for ( unsigned int i1 = 0; i1 < n1; i1++ ) {
      fprintf ( fs, "%25.16e\n", data[i0][i1] );
    }
  }

  fclose ( fs );
}  /* end of write_data_real */


/**********************************************************
 *
 **********************************************************/
inline void write_data_real2_reim ( double **** const d, char * const filename, int *** const conf_src_list, int const n1, int const n2, int const n3, int const ir ) {

  FILE * fs = fopen ( filename, "w" );

  for ( int iconf = 0; iconf < n1; iconf++ ) {
    for ( int isrc = 0; isrc < n2; isrc++ ) {
      for ( int it = 0; it < n3; it++ ) {
        fprintf ( fs, "%3d %25.16e %c %6d %3d\n", it, d[iconf][isrc][it][ir],
            conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2] );
      }
    }
  }

  fclose ( fs );

}  /* end of write_data_real2_reim */

/**********************************************************
 *
 **********************************************************/
inline void src_avg_real2_reim ( double ** data, double ****corr, unsigned int const n0, unsigned int const n1, unsigned int const n2, int const ri ) {

#pragma omp parallel for
  for ( unsigned int iconf = 0; iconf < n0; iconf++ ) {
    for ( unsigned int it = 0; it < n2; it++ ) {
      double dtmp = 0.;

      for ( unsigned int isrc = 0; isrc < n1; isrc++ ) {
        dtmp += corr[iconf][isrc][it][ri];
      }
      data[iconf][it] = dtmp / (double)n1;
    }
  }
}  /* end of src_avg_real2_reim */

/**********************************************************
 *
 **********************************************************/
int main(int argc, char **argv) {
  
  char const reim_str[2][3] = { "re", "im" };

  char const threep_tag[3][12] = { "g4_D4", "gi_Dk", "g4_Dk" };

  char const fbwd_str[2][4] = { "fwd", "bwd" };

#if _TWOP_AVGX_PLEGMA
  char const flavor_type_2pt[2][2] = { "P", "K" };

  char const flavor_type_3pt[6][20] = { "up_pion", "dn_pion", "up_kaon", "up_kaon_inverted", "st_kaon", "st_kaon_inverted" };

  /**********************************************************
   * up_pion           = /l-gd-ll-gi/mu0.0005/mu0.0005/mu-0.0005/
   * dn_pion           = /l-gd-ll-gi/mu-0.0005/mu-0.0005/mu0.0005/
   * up_kaon           = /l-gd-ls-gi/mu0.0005/mu0.0005/mu-0.0136/
   * up_kaon_inverted  = /l-gd-ls-gi/mu-0.0005/mu-0.0005/mu0.0136/
   * st_kaon           = /s-gd-sl-gi/mu0.0136/mu0.0136/mu-0.0005/
   * st_kaon_inverted  = /s-gd-sl-gi/mu-0.0136/mu-0.0136/mu0.0005/
   *
   **********************************************************/


#else
  char const flavor_type_2pt[4][12] = { "u-gf-d-gi", "d-gf-u-gi",  "l-gf-l-gi" , "s-gf-l-gi" };

  char const flavor_type_3pt[5][12] = { "u-gd-sud-gi" , "d-gd-sdu-gi" ,  "s-gd-sl-gi", "l-gd-ls-gi" , "l-gd-ll-gi" };
#endif


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

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:E:w:F:T:m:i:I:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [avxn_conn_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [avxn_conn_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [avxn_conn_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
   case 'F':
      twop_fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] twop fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [avxn_conn_analyse] write_date set to %d\n", write_data );
      break;
    case 'T':
      sscanf( optarg, "%lf,%lf", twop_weight, twop_weight+1 );
      fprintf ( stdout, "# [avxn_conn_analyse] twop_weight set to %25.16e / %25.16e\n", twop_weight[0], twop_weight[1] );
      break;
/*    case 'B':
      sscanf( optarg, "%lf,%lf", fbwd_weight, fbwd_weight+1 );
      fprintf ( stdout, "# [avxn_conn_analyse] fbwd_weight set to %25.16e / %25.16e\n", fbwd_weight[0], fbwd_weight[1] );
      break;
    case 'M':
      sscanf( optarg, "%lf,%lf", mirror_weight, mirror_weight+1 );
      fprintf ( stdout, "# [avxn_conn_analyse] mirror_weight set to %25.16e / %25.16e\n", mirror_weight[0], mirror_weight[1] );
      break;
*/
    case 'm':
      g_mus = atof ( optarg );
      break;
    case 'i':
      flavor_id_2pt = atoi ( optarg );
      fprintf ( stdout, "# [avxn_conn_analyse] flavor_id_2pt set to %d \n", flavor_id_2pt );
      break;
    case 'I':
      flavor_id_3pt = atoi ( optarg );
      fprintf ( stdout, "# [avxn_conn_analyse] flavor_id_3pt set to %d \n", flavor_id_3pt );
      break;
    case 'n':
      threep_operator_norm = atof ( optarg );
      fprintf ( stdout, "# [avxn_conn_analyse] threep_operator_norm set to %e \n", threep_operator_norm );
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
  /* fprintf(stdout, "# [avxn_conn_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [avxn_conn_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [avxn_conn_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [avxn_conn_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[avxn_conn_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[avxn_conn_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[avxn_conn_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [avxn_conn_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[avxn_conn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[avxn_conn_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [avxn_conn_analyse] comment %s\n", line );
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

  switch ( flavor_id_2pt ) {
    case 2:
      muval_2pt_list[0] = g_mu;
      muval_2pt_list[1] = g_mu;
      break;
    case 3:
      muval_2pt_list[0] = g_mus;
      muval_2pt_list[1] = g_mu;
      break;
    default:
      if ( io_proc == 2 ) fprintf ( stderr, "[avxn_conn_analyse] flavor_id_2pt = %d not implemented   %s %d\n", flavor_id_2pt, __FILE__, __LINE__ );
      /* EXIT(1); */
      muval_2pt_list[0] = g_mu;
      muval_2pt_list[1] = g_mu;
      break;
  }

  switch ( flavor_id_3pt ) {
    case 2:
      muval_3pt_list[0] = g_mus;
      muval_3pt_list[1] = g_mus;
      muval_3pt_list[2] = g_mu;
      break;
    case 3:
      muval_3pt_list[0] = g_mu;
      muval_3pt_list[1] = g_mu;
      muval_3pt_list[2] = g_mus;
      break;
    case 4:
      muval_3pt_list[0] = g_mu;
      muval_3pt_list[1] = g_mu;
      muval_3pt_list[2] = g_mu;
      break;
    default:
      if ( io_proc == 2 ) fprintf ( stderr, "[avxn_conn_analyse] flavor_id_3pt = %d not implemented   %s %d\n", flavor_id_3pt, __FILE__, __LINE__ );
      /* EXIT(1); */
      muval_3pt_list[0] = g_mu;
      muval_3pt_list[1] = g_mu;
      muval_3pt_list[2] = g_mu;
      break;
  }


  /**********************************************************
   **********************************************************
   ** 
   ** READ DATA
   ** 
   **********************************************************
   **********************************************************/

  /***********************************************************
   * read twop function data
   ***********************************************************/
  double ***** twop = NULL;

  twop = init_5level_dtable ( g_sink_momentum_number, num_conf, num_src_per_conf, T_global, 2 );
  if( twop == NULL ) {
    fprintf ( stderr, "[avxn_conn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

/**********************************************************/
#if _TWOP_H5
/**********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***********************************************************
   * loop on configs
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {

    double ** buffer = init_2level_dtable ( T_global, 2 );

    /***********************************************************
     * loop on sources
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      /***********************************************************
       * open AFF reader
       ***********************************************************/
      char data_filename[500];
      char key[400];
    
      sprintf( data_filename, "stream_%c/%s/corr.%.4d.t%d.h5",
          conf_src_list[iconf][isrc][0],
          filename_prefix,
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2] );

      if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [avxn_conn_analyse] reading from data filename %s %s %d\n", data_filename, __FILE__, __LINE__ );
        fflush(stdout);
      }

      /***********************************************************
       * loop on sink momenta
       ***********************************************************/
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        int pf[3] = {
          g_sink_momentum_list[ipf][0],
          g_sink_momentum_list[ipf][1],
          g_sink_momentum_list[ipf][2] 
        };

        int pi[3] = {
          -pf[0],
          -pf[1],
          -pf[2] 
        };

        sprintf( key, "/u+-g-u-g/t%d/gf5/pfx%dpfy%dpfz%d/gi5/pix%dpiy%dpiz%d",
                conf_src_list[iconf][isrc][2], 
                pf[0], pf[1], pf[2],
                pi[0], pi[1], pi[2] );

        
        if ( g_verbose > 2 ) {
          fprintf ( stdout, "# [avxn_conn_analyse] key = %s\n", key );
          fflush(stdout);
        }

        exitstatus = read_from_h5_file ( (void*)buffer[0], data_filename, key,  "double",  io_proc );
        if ( exitstatus != 0 ) {
          fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        /***********************************************************
         * NOTE: NO SOURCE PHASE NECESSARY
         * ONLY REORDERING from source
         ***********************************************************/
#pragma omp parallel for
        for ( int it = 0; it < T_global; it++ ) {
          int const itt = ( it + conf_src_list[iconf][isrc][2] + T_global ) % T_global;
          twop[ipf][iconf][isrc][it][0] = buffer[itt][0];
          twop[ipf][iconf][isrc][it][1] = buffer[itt][1];
        }

        /***********************************************************
         * NOTE: opposite parity transformed case is given by 
         *       complex conjugate
         *       
         ***********************************************************/

      }  /* end of loop on sink momenta */

    }  /* end of loop on sources */

    fini_2level_dtable ( &buffer );
  }  /* end of loop on configs */
          
  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "avxn_conn_analyse", "read-twop-h5", g_cart_id == 0 );

#endif  /* end of _TWOP_H5 */


/**********************************************************/
#if _TWOP_AFF
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
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      /***********************************************************
       * open AFF reader
       ***********************************************************/
      char data_filename[500];
      char key[400];
    
      sprintf( data_filename, "stream_%c/%s/%s.%.4d.t%d.aff",
          conf_src_list[iconf][isrc][0],
          filename_prefix,
          filename_prefix2,
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2] );

      /***********************************************************
       * loop on sink momenta
       ***********************************************************/
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        int pf[3] = {
          g_sink_momentum_list[ipf][0],
          g_sink_momentum_list[ipf][1],
          g_sink_momentum_list[ipf][2] 
        };

        int pi[3] = {
          -pf[0],
          -pf[1],
          -pf[2] 
        };

        if ( twop_weight[0] != 0. ) {
          
          sprintf( key, "/u-gf-d-gi/t%d/s0/gf5/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
                  conf_src_list[iconf][isrc][2], 
                  pi[0], pi[1], pi[2],
                  pf[0], pf[1], pf[2]);

          if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s\n", key );

          exitstatus = read_aff_contraction ( (void*)(buffer[0][0]), NULL, data_filename, key, T_global );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[avxn_conn_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
        }

        if ( twop_weight[1] != 0. ) {
          sprintf( key, "/d-gf-u-gi/t%d/s0/gf5/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
                  conf_src_list[iconf][isrc][2], 
                  -pi[0], -pi[1], -pi[2],
                  -pf[0], -pf[1], -pf[2]);

          if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s\n", key );

          exitstatus = read_aff_contraction ( (void*)(buffer[1][0]), NULL, data_filename, key, T_global );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[avxn_conn_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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

        /***********************************************************
         * NOTE: opposite parity transformed case is given by 
         *       complex conjugate
         *       
         ***********************************************************/

      }  /* end of loop on sink momenta */

    }  /* end of loop on sources */

    fini_3level_dtable ( &buffer );
  }  /* end of loop on configs */
          
  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "avxn_conn_analyse", "read-twop-aff", g_cart_id == 0 );

#endif  /* end of _TWOP_AFF */

  /**********************************************************/
  /**********************************************************/

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
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      /***********************************************************
       * open AFF reader
       ***********************************************************/
      char data_filename[500];
    
      sprintf( data_filename, "stream_%c/%s/%d/%s.%.4d.t%d.s%d.h5",
          conf_src_list[iconf][isrc][0],
          filename_prefix,
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

        int pi[3] = {
          -pf[0],
          -pf[1],
          -pf[2] 
        };

        char key[400], key2[400];
        if ( twop_weight[0] != 0. ) {
          /* s-gf-l-gi/mu-0.0186/mu0.0007/t116/s0/gf5/gi5/pix-1piy0piz0/px1py0pz0 */
          
          sprintf( key, "/%s/mu%6.4f/mu%6.4f/t%d/s%d/gf5/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
              flavor_type_2pt[flavor_id_2pt],
                  -muval_2pt_list[0], muval_2pt_list[1],
                  conf_src_list[iconf][isrc][2], 
                  conf_src_list[iconf][isrc][3], 
                  pi[0], pi[1], pi[2],
                  pf[0], pf[1], pf[2]);

          sprintf( key2, "/%s/mu%6.4f/mu%6.4f/t%d/s%d/gf5/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
              flavor_type_2pt[flavor_id_2pt],
                  -muval_2pt_list[0], muval_2pt_list[1],
                  conf_src_list[iconf][isrc][2], 
                  0, 
                  pi[0], pi[1], pi[2],
                  pf[0], pf[1], pf[2]);



          if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__  );

          exitstatus = read_from_h5_file ( (void*)(buffer[0][0]), data_filename, key, "double", io_proc );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", 
                data_filename, key, exitstatus, __FILE__, __LINE__ );
            /* EXIT(1); */
          
            if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key2 = %s %s %d\n", key2, __FILE__, __LINE__  );

            exitstatus = read_from_h5_file ( (void*)(buffer[0][0]), data_filename, key2, "double", io_proc );
            if ( exitstatus != 0 ) {
              fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", 
                  data_filename, key2, exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }
          }
        }  /* end of twop_weight 0 */

        if ( twop_weight[1] != 0. ) {

          sprintf( key, "/%s/mu%6.4f/mu%6.4f/t%d/s%d/gf5/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
              flavor_type_2pt[flavor_id_2pt],
                  muval_2pt_list[0], -muval_2pt_list[1],
                  conf_src_list[iconf][isrc][2], 
                  conf_src_list[iconf][isrc][3], 
                  -pi[0], -pi[1], -pi[2],
                  -pf[0], -pf[1], -pf[2]);

          sprintf( key2, "/%s/mu%6.4f/mu%6.4f/t%d/s%d/gf5/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
              flavor_type_2pt[flavor_id_2pt],
                  muval_2pt_list[0], -muval_2pt_list[1],
                  conf_src_list[iconf][isrc][2],
                  0,
                  -pi[0], -pi[1], -pi[2],
                  -pf[0], -pf[1], -pf[2]);


          if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__ );

          exitstatus = read_from_h5_file ( (void*)(buffer[1][0]), data_filename, key, "double", io_proc );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n",
                data_filename, key, exitstatus, __FILE__, __LINE__ );
            /* EXIT(1); */

            if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key2 = %s %s %d\n", key2, __FILE__, __LINE__ );

            exitstatus = read_from_h5_file ( (void*)(buffer[1][0]), data_filename, key2, "double", io_proc );
            if ( exitstatus != 0 ) {
              fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file file %s key %s, status was %d %s %d\n",
                  data_filename, key2, exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }

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

        /***********************************************************
         * NOTE: opposite parity transformed case is given by 
         *       ???
         *       
         ***********************************************************/

      }  /* end of loop on sink momenta */

    }  /* end of loop on sources */

    fini_3level_dtable ( &buffer );
  }  /* end of loop on configs */
          
  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "avxn_conn_analyse", "read-twop-h5", g_cart_id == 0 );

#endif  /* end of _TWOP_AVGX_H5 */

  /**********************************************************/
  /**********************************************************/

/**********************************************************/
#if _TWOP_AVGX_PLEGMA
/**********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***********************************************************
   * loop on configs
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {

    /***********************************************************
     * loop on sources
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      /***********************************************************
       * open AFF reader
       ***********************************************************/
      char data_filename[500];
    
      sprintf( data_filename, "stream_%c/%d/%s%.4d_sx%.2dsy%.2dsz%.2dst%.3d_%s.h5",
          conf_src_list[iconf][isrc][0],
          conf_src_list[iconf][isrc][1],
          filename_prefix,
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][3],
          conf_src_list[iconf][isrc][4],
          conf_src_list[iconf][isrc][5],
          conf_src_list[iconf][isrc][2],
          flavor_type_2pt[flavor_id_2pt]);

      char momentum_tag[100];
      int * momentum_buffer = NULL;
      size_t * momentum_cdim = NULL, momentum_ncdim = 0;

      sprintf( momentum_tag, "/sx%.2dsy%.2dsz%.2dst%.2d/mvec", 
          conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5], conf_src_list[iconf][isrc][2] );
        
      exitstatus = read_from_h5_file_varsize ( (void**)&momentum_buffer, data_filename, momentum_tag,  "int", &momentum_ncdim, &momentum_cdim,  io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[avxn_conn_analyse] Error from read_from_h5_file_varsize for file %s key %s   %s %d\n", data_filename, momentum_tag, __FILE__, __LINE__);
        EXIT(15);
      }

      if ( momentum_ncdim != 2 || momentum_cdim[1] != 3 ) {
        fprintf ( stderr, "[avxn_conn_analyse] Error, unreccognized data format      %s %d\n", __FILE__, __LINE__ );
        EXIT(129);
      }

      int const momentum_number = (int)(momentum_cdim[0]);
      if ( g_verbose > 4 ) fprintf ( stdout, "# [avxn_conn_analyse] read %d momenta %s %d\n", momentum_number, __FILE__, __LINE__ );
      int ** momentum_list = init_2level_itable ( momentum_number, 3 );
      memcpy ( momentum_list[0], momentum_buffer, momentum_number * 3 * sizeof ( int ) );
      free ( momentum_buffer );
      free ( momentum_cdim );

 
      char key[400];
      sprintf( key, "/sx%.2dsy%.2dsz%.2dst%.2d/PPUP",
          conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5], conf_src_list[iconf][isrc][2] );

      if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__  );

      double **** buffer = init_4level_dtable ( T_global, momentum_number, 1, 2 );
      if ( buffer  == NULL ) {
        fprintf ( stderr, "[avxn_conn_analyse] Error from init_Xlevel_dtable      %s %d\n", __FILE__, __LINE__ );
        EXIT(129);
      }

      exitstatus = read_from_h5_file ( (void*)(buffer[0][0][0]), data_filename, key, "double", io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", 
        data_filename, key, exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      } 

      /***********************************************************
       * loop on sink momenta
       ***********************************************************/
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        int pf[3] = {
          g_sink_momentum_list[ipf][0],
          g_sink_momentum_list[ipf][1],
          g_sink_momentum_list[ipf][2] 
        };

        int pi[3] = {
          -pf[0],
          -pf[1],
          -pf[2] 
        };

        char key[400];

        int const pf_id = get_momentum_id ( pf, momentum_list, momentum_number );

        if ( pf_id == -1 ) {
          fprintf( stderr, "[avxn_conn_analyse] Error from get_momentum_id   %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /***********************************************************
         * NOTE: NO SOURCE PHASE NECESSARY
         * NOTE: NO REORDERING from source necessary
         ***********************************************************/
#pragma omp parallel for
        for ( int it = 0; it < T_global; it++ ) {
          twop[ipf][iconf][isrc][it][0] = buffer[it][pf_id][0][0];
          twop[ipf][iconf][isrc][it][1] = buffer[it][pf_id][0][1];
        }

      }  /* end of loop on sink momenta */

      fini_4level_dtable ( &buffer );

    }  /* end of loop on sources */

  }  /* end of loop on configs */
          
  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "avxn_conn_analyse", "read-twop-h5", g_cart_id == 0 );

#endif  /* end of _TWOP_AVGX_PLEGMA */

  /**********************************************************/
  /**********************************************************/

  /**********************************************************
   * average 2-pt over momentum orbit
   **********************************************************/

  double **** twop_orbit = init_4level_dtable ( num_conf, num_src_per_conf, T_global, 2 );
  if( twop_orbit == NULL ) {
    fprintf ( stderr, "[avxn_conn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
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
      fprintf ( stderr, "[avxn_conn_analyse] Error from init_Xlevel_dtable %s %d\n",  __FILE__, __LINE__ );
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

    if ( num_conf >= 6 )
    {
      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avxn_conn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }
    } else {
      fprintf ( stderr, "[avxn_conn_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
    }

    /**********************************************************
     * write data to ascii file
     **********************************************************/
    if ( write_data == 1 )
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

      if ( num_conf >= 6 )
      {
        exitstatus = apply_uwerr_func ( data[0], num_conf, T_global, nT, narg, arg_first, arg_stride, obs_name2, acosh_ratio, dacosh_ratio );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[avxn_conn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(115);
        }
      }
    }

    /* fini_3level_dtable ( &data ); */
    fini_2level_dtable ( &data );
  }  /* end of loop on reim */

#endif  /* of ifdef _TWOP_STATS */

  /**********************************************************
   * loop on source - sink time separations
   **********************************************************/
  for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {

    double ******** threep = init_8level_dtable ( g_sink_momentum_number, num_conf, num_src_per_conf, 4, 4, 2, T_global, 2 );
    if ( threep == NULL ) {
      fprintf( stderr, "[avxn_conn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

/**********************************************************/
#if _THREEP_H5
/**********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );
  
    /***********************************************************
     * loop on configs
     ***********************************************************/
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {

      double ** buffer = init_2level_dtable ( T_global, 2 );
  
      /***********************************************************
       * loop on sources
       ***********************************************************/
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        /***********************************************************
         * open h5 reader
         ***********************************************************/
        char data_filename[500];
        char key[400];
  
        sprintf( data_filename, "stream_%c/%s/corr.%.4d.t%d.h5",
            conf_src_list[iconf][isrc][0],
            filename_prefix,
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2] );
  
        if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_conn_analyse] reading from data filename %s %s %d\n", data_filename, __FILE__, __LINE__ );

        /***********************************************************
         * loop on sink momenta
         ***********************************************************/
        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
  
          int pf[3] = {
              g_sink_momentum_list[ipf][0],
              g_sink_momentum_list[ipf][1],
              g_sink_momentum_list[ipf][2] 
            };
  
          int pi[3] = {
            -pf[0],
            -pf[1],
            -pf[2] 
          };

          for ( int igc =0; igc < 4; igc++ ) {

            for ( int idim=0; idim<4; idim++ ) {

            for ( int idir=0; idir<2; idir++ ) {

              /* /sud+-g-u-g/t54/dt16/gf5/pfx0pfy0pfz0/gc0/Ddim0_dir0/gi5/pix0piy0piz0 */
              sprintf( key, "/sud+-g-u-g/t%d/dt%d/gf5/pfx%dpfy%dpfz%d/gc%d/Ddim%d_dir%d/gi5/pix%dpiy%dpiz%d",
                  conf_src_list[iconf][isrc][2], g_sequential_source_timeslice_list[idt],
                  pf[0], pf[1], pf[2], igc, idim, idir,
                  pi[0], pi[1], pi[2] );
  
              if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s\n", key );
  
              exitstatus = read_from_h5_file ( (void*)buffer[0], data_filename, key, "double", io_proc );
              if ( exitstatus != 0 ) {
                fprintf( stderr, "[]avxn_conn_analyse Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }
  
              /***********************************************************
               * NOTE: NO SOURCE PHASE NECESSARY
               * ONLY REORDERING NECESSARY
               ***********************************************************/
#pragma omp parallel for
              for ( int it = 0; it < T_global; it++ ) {
                int const itt = ( it + conf_src_list[iconf][isrc][2] + T_global ) % T_global;
                threep[ipf][iconf][isrc][igc][idim][idir][it][0] = buffer[itt][0];
                threep[ipf][iconf][isrc][igc][idim][idir][it][1] = buffer[itt][1];
              }
  
              /***********************************************************
               * NOTE: opposite parity transformed case is given by 
               *       complex conjugate
               ***********************************************************/

            }}  /* end of loop on idir, idim */

          }  /* end of loop on igc */

        }  /* end of loop on sink momenta */

      }  /* end of loop on sources */
  
      fini_2level_dtable ( &buffer );
    }  /* end of loop on configs */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "avxn_conn_analyse", "read-threep-h5", g_cart_id == 0 );

#endif  /* end of if _THREEP_H5 */

/**********************************************************/
#if _THREEP_AFF
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
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        /***********************************************************
         * open h5 reader
         ***********************************************************/
        char data_filename[500];
        char key[400];
  
        sprintf( data_filename, "stream_%c/%s/%s.%.4d.t%d.aff",
            conf_src_list[iconf][isrc][0],
            filename_prefix,
            filename_prefix2,
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2] );
  
        if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_conn_analyse] reading from data filename %s %s %d\n", data_filename, __FILE__, __LINE__ );

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

          for ( int igc =0; igc < 4; igc++ ) {

            for ( int idim=0; idim<4; idim++ ) {

              /* if ( igc != idim ) continue; */

            for ( int idir=0; idir<2; idir++ ) {

              if ( twop_weight[0] != 0. ) {
                /***********************************************************
                 * read AFF data key ...
                 ***********************************************************/
                sprintf( key, "/u-gd-sud-gi/t%d/s0/dt%d/gf5/gc%d/d%d/%s/gi5/pfx%dpfy%dpfz%d/px%dpy%dpz%d",
                    conf_src_list[iconf][isrc][2], g_sequential_source_timeslice_list[idt],
                    igc, idim, fbwd_str[idir],
                    pf[0], pf[1], pf[2], 
                    pi[0], pi[1], pi[2] );
                  
                if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__ );
  
                exitstatus = read_aff_contraction ( (void*)(buffer[0][0]), NULL, data_filename, key, T_global );
                if ( exitstatus != 0 ) {
                  fprintf( stderr, "[avxn_conn_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(1);
                }
  
              }

              if ( twop_weight[1] != 0. ) {
  
                /***********************************************************
                 * ... and twisted parity partner
                 ***********************************************************/
                sprintf( key, "/d-gd-sdu-gi/t%d/s0/dt%d/gf5/gc%d/d%d/%s/gi5/pfx%dpfy%dpfz%d/px%dpy%dpz%d",
                    conf_src_list[iconf][isrc][2], g_sequential_source_timeslice_list[idt],
                    igc, idim, fbwd_str[idir],
                    -pf[0], -pf[1], -pf[2], 
                    -pi[0], -pi[1], -pi[2] );
                  
                if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__ );
   
                exitstatus = read_aff_contraction ( (void*)(buffer[1][0]), NULL, data_filename, key, T_global );
                if ( exitstatus != 0 ) {
                  fprintf( stderr, "[]avxn_conn_analyse Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(1);
                }
              }
  
              /***********************************************************
               * NOTE: NO SOURCE PHASE NECESSARY
               * ONLY REORDERING NECESSARY
               ***********************************************************/
              double const threep_norm = 1. / ( fabs( twop_weight[0] ) + fabs( twop_weight[1] ) );
              int const parity_sign = ( 2 * ( igc == 0 ) - 1 ) * ( 2 * ( idim == 0 ) - 1 );
#pragma omp parallel for
              for ( int it = 0; it < T_global; it++ ) {
                int const itt = ( it + conf_src_list[iconf][isrc][2] + T_global ) % T_global;
                /***********************************************************
                 * add up with parity sign
                 ***********************************************************/
                threep[ipf][iconf][isrc][igc][idim][idir][it][0] = ( twop_weight[0] * buffer[0][itt][0] + parity_sign * twop_weight[1] * buffer[1][itt][0] ) * threep_norm;
                threep[ipf][iconf][isrc][igc][idim][idir][it][1] = ( twop_weight[0] * buffer[0][itt][1] + parity_sign * twop_weight[1] * buffer[1][itt][1] ) * threep_norm;
              }
  
              /***********************************************************
               * NOTE: opposite parity transformed case is given by 
               *       complex conjugate
               ***********************************************************/

            }}  /* end of loop on idir, idim */

          }  /* end of loop on igc */

        }  /* end of loop on sink momenta */

      }  /* end of loop on sources */
  
      fini_3level_dtable ( &buffer );
    }  /* end of loop on configs */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "avxn_conn_analyse", "read-threep-h5", g_cart_id == 0 );

#endif  /* end of if _THREEP_H5 */

  /**********************************************************/
  /**********************************************************/

/**********************************************************/
#if _THREEP_AVGX_H5
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
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        /***********************************************************
         * open h5 reader
         ***********************************************************/
        char data_filename[500];
  
        sprintf( data_filename, "stream_%c/%s/%d/%s.%.4d.t%d.s%d.h5",
            conf_src_list[iconf][isrc][0],
            filename_prefix,
            conf_src_list[iconf][isrc][1],
            filename_prefix2,
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3] );
  
        if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_conn_analyse] reading from data filename %s %s %d\n", data_filename, __FILE__, __LINE__ );

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

          for ( int igc =0; igc < 4; igc++ ) {

            for ( int idim=0; idim<4; idim++ ) {

            for ( int idir=0; idir<2; idir++ ) {

              char key[400], key2[400];

              if ( twop_weight[0] != 0. ) {
                /***********************************************************
                 * read h5 data key ...
                 ***********************************************************/
                /* /s-gd-sl-gi/mu-0.0186/mu-0.0186/mu0.0007/t116/s0/dt24/gf5/gc1/d2/fwd/gi5/pix0piy0piz0/px0py0pz0 */

                sprintf( key, "/%s/mu%6.4f/mu%6.4f/mu%6.4f/t%d/s%d/dt%d/gf5/gc%d/d%d/%s/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
                    flavor_type_3pt[flavor_id_3pt],
                    -muval_3pt_list[0], -muval_3pt_list[1], muval_3pt_list[2],
                    conf_src_list[iconf][isrc][2],
                    conf_src_list[iconf][isrc][3],
                    g_sequential_source_timeslice_list[idt],
                    igc, idim, fbwd_str[idir],
                    pi[0], pi[1], pi[2],
                    pf[0], pf[1], pf[2] );

                sprintf( key2, "/%s/mu%6.4f/mu%6.4f/mu%6.4f/t%d/s%d/dt%d/gf5/gc%d/d%d/%s/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
                    flavor_type_3pt[flavor_id_3pt],
                    -muval_3pt_list[0], -muval_3pt_list[1], muval_3pt_list[2],
                    conf_src_list[iconf][isrc][2], 0,
                    g_sequential_source_timeslice_list[idt],
                    igc, idim, fbwd_str[idir],
                    pi[0], pi[1], pi[2],
                    pf[0], pf[1], pf[2] );


                  
                if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__ );
  
                exitstatus = read_from_h5_file ( (void*)(buffer[0][0]), data_filename, key,  "double", io_proc );
                if ( exitstatus != 0 ) {
                  fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", data_filename, key, exitstatus, __FILE__, __LINE__ );
                  /* EXIT(1); */

                  if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key2 = %s %s %d\n", key2, __FILE__, __LINE__ );

                  exitstatus = read_from_h5_file ( (void*)(buffer[0][0]), data_filename, key2,  "double", io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", data_filename, key2, exitstatus, __FILE__, __LINE__ );
                    EXIT(1);
                  }

                }

              }

              if ( twop_weight[1] != 0. ) {
  
                /***********************************************************
                 * ... and twisted parity partner
                 ***********************************************************/
                sprintf( key, "/%s/mu%6.4f/mu%6.4f/mu%6.4f/t%d/s%d/dt%d/gf5/gc%d/d%d/%s/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
                    flavor_type_3pt[flavor_id_3pt],
                    muval_3pt_list[0], muval_3pt_list[1], -muval_3pt_list[2],
                    conf_src_list[iconf][isrc][2], 
                    conf_src_list[iconf][isrc][3], 
                    g_sequential_source_timeslice_list[idt],
                    igc, idim, fbwd_str[idir],
                    -pi[0], -pi[1], -pi[2],
                    -pf[0], -pf[1], -pf[2] );

                sprintf( key2, "/%s/mu%6.4f/mu%6.4f/mu%6.4f/t%d/s%d/dt%d/gf5/gc%d/d%d/%s/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
                    flavor_type_3pt[flavor_id_3pt],
                    muval_3pt_list[0], muval_3pt_list[1], -muval_3pt_list[2],
                    conf_src_list[iconf][isrc][2], 0,
                    g_sequential_source_timeslice_list[idt],
                    igc, idim, fbwd_str[idir],
                    -pi[0], -pi[1], -pi[2],
                    -pf[0], -pf[1], -pf[2] );

                  
                if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__ );
   
                exitstatus = read_from_h5_file ( (void*)(buffer[1][0]), data_filename, key,  "double", io_proc );
                if ( exitstatus != 0 ) {
                  fprintf( stderr, "[]avxn_conn_analyse Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", data_filename, key, exitstatus, __FILE__, __LINE__ );
                  /* EXIT(1); */

                  if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key2 = %s %s %d\n", key2, __FILE__, __LINE__ );

                  exitstatus = read_from_h5_file ( (void*)(buffer[1][0]), data_filename, key2,  "double", io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf( stderr, "[]avxn_conn_analyse Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", data_filename, key2, exitstatus, __FILE__, __LINE__ );
                    EXIT(1);
                  }

                }
              }
  
              /***********************************************************
               * NOTE: NO SOURCE PHASE NECESSARY
               * ONLY REORDERING NECESSARY
               ***********************************************************/
              double const threep_norm = 1. / ( fabs( twop_weight[0] ) + fabs( twop_weight[1] ) ) * threep_operator_norm;
              int const parity_sign = ( 2 * ( igc == 0 ) - 1 ) * ( 2 * ( idim == 0 ) - 1 );
#pragma omp parallel for
              for ( int it = 0; it < T_global; it++ ) {
                int const itt = ( it + conf_src_list[iconf][isrc][2] + T_global ) % T_global;
                /***********************************************************
                 * add up with parity sign
                 ***********************************************************/
                threep[ipf][iconf][isrc][igc][idim][idir][it][0] = ( twop_weight[0] * buffer[0][itt][0] + parity_sign * twop_weight[1] * buffer[1][itt][0] ) * threep_norm;
                threep[ipf][iconf][isrc][igc][idim][idir][it][1] = ( twop_weight[0] * buffer[0][itt][1] + parity_sign * twop_weight[1] * buffer[1][itt][1] ) * threep_norm;
              }
  
              /***********************************************************
               * NOTE: opposite parity transformed case is given by 
               *       complex conjugate
               ***********************************************************/

            }}  /* end of loop on idir, idim */

          }  /* end of loop on igc */

        }  /* end of loop on sink momenta */

      }  /* end of loop on sources */
  
      fini_3level_dtable ( &buffer );
    }  /* end of loop on configs */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "avxn_conn_analyse", "read-threep-h5", g_cart_id == 0 );

#endif  /* end of if _THREEP_AVGX_H5 */

  /**********************************************************/
  /**********************************************************/


/**********************************************************/
#if _THREEP_AVGX_PLEGMA
/**********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );
  
    /* {1,g1,g2,g3,g4,g5,g5g1,g5g2,g5g3,g5g4} */
    int const gc_map[4] = { 4, 1, 2, 3 };

    int const dim_map[4] = { 3, 0, 1, 2};


    /***********************************************************
     * loop on configs
     ***********************************************************/
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {

      double ***** buffer = init_5level_dtable ( T_global, 1, 10, 4, 2 );
      if ( buffer == NULL ) {
        fprintf( stderr, "[avxn_conn_analyse] Error from init_Xlevel_dtable    %s %d\n",  __FILE__, __LINE__ );
        EXIT(1);
      }
  
      /***********************************************************
       * loop on sources
       ***********************************************************/
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        /***********************************************************
         * open h5 reader
         ***********************************************************/
        char data_filename[500];
 
        /* threep_sx91sy14sz63st014_dt96_up_kaon_oneD.h5 */ 
        sprintf( data_filename, "stream_%c/%d/%s_sx%.2dsy%.2dsz%.2dst%.3d_dt%d_%s_oneD.h5",
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            filename_prefix2,
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5],
            conf_src_list[iconf][isrc][2],
            g_sequential_source_timeslice_list[idt],
            flavor_type_3pt[flavor_id_3pt] );
  
        if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_conn_analyse] reading from data filename %s %s %d\n", data_filename, __FILE__, __LINE__ );

        /***********************************************************
         * loop on sink momenta
         ***********************************************************/
        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
  
          int pf[3] = {
              g_sink_momentum_list[ipf][0],
              g_sink_momentum_list[ipf][1],
              g_sink_momentum_list[ipf][2] 
            };

          int pi[3] = {
            -pf[0],
            -pf[1], 
            -pf[2] };

          for ( int igc =0; igc < 4; igc++ ) {

            for ( int idim=0; idim<4; idim++ ) {

              char key[400];

              /* /sx91sy14sz63st14/pi=0_0_0/PJP_STD*/

              sprintf( key, "/sx%.2dsy%.2dsz%.2dst%.2d/pi=%d_%d_%d/PJP_STD",
                    conf_src_list[iconf][isrc][3],
                    conf_src_list[iconf][isrc][4],
                    conf_src_list[iconf][isrc][5],
                    conf_src_list[iconf][isrc][2],
                    pi[0], pi[1], pi[2] );

              if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__ );
  
              exitstatus = read_from_h5_file ( (void*)(buffer[0][0][0][0]), data_filename, key,  "double", io_proc );
              if ( exitstatus != 0 ) {
                fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", data_filename, key, exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }

              /***********************************************************
               * NOTE: NO SOURCE PHASE NECESSARY
               * ONLY REORDERING NECESSARY
               ***********************************************************/
              double const threep_norm = threep_operator_norm;
#pragma omp parallel for
              for ( int it = 0; it < T_global; it++ ) {
                threep[ipf][iconf][isrc][igc][idim][0][it][0] = buffer[it][0][gc_map[igc]][dim_map[idim]][0] * threep_norm;
                threep[ipf][iconf][isrc][igc][idim][0][it][1] = buffer[it][0][gc_map[igc]][dim_map[idim]][1] * threep_norm;
              }
  
            }  /* end of loop on idim */

          }  /* end of loop on igc */

        }  /* end of loop on sink momenta */

        /* threep_sx91sy14sz63st014_dt96_up_kaon_oneD.h5 */ 
        sprintf( data_filename, "stream_%c/%d/%s_sx%.2dsy%.2dsz%.2dst%.3d_dt%d_%s_oneD.h5",
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            filename_prefix2,
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5],
            conf_src_list[iconf][isrc][2],
            g_sequential_source_timeslice_list[idt],
            flavor_type_3pt[flavor_id_3pt+1] );
  
        if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_conn_analyse] reading from data filename %s %s %d\n", data_filename, __FILE__, __LINE__ );

        /***********************************************************
         * loop on sink momenta
         ***********************************************************/
        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
  
          int pf[3] = {
              -g_sink_momentum_list[ipf][0],
              -g_sink_momentum_list[ipf][1],
              -g_sink_momentum_list[ipf][2] 
            };

          int pi[3] = {
            -pf[0],
            -pf[1],
            -pf[2] };

          for ( int igc =0; igc < 4; igc++ ) {

            for ( int idim=0; idim<4; idim++ ) {

              char key[400];

              /* /sx91sy14sz63st14/pi=0_0_0/PJP_STD*/

              sprintf( key, "/sx%.2dsy%.2dsz%.2dst%.2d/pi=%d_%d_%d/PJP_STD",
                    conf_src_list[iconf][isrc][3],
                    conf_src_list[iconf][isrc][4],
                    conf_src_list[iconf][isrc][5],
                    conf_src_list[iconf][isrc][2],
                    pi[0], pi[1], pi[2] );

              if ( g_verbose > 3 ) fprintf ( stdout, "# [avxn_conn_analyse] key = %s %s %d\n", key, __FILE__, __LINE__ );
  
              exitstatus = read_from_h5_file ( (void*)(buffer[0][0][0][0]), data_filename, key,  "double", io_proc );
              if ( exitstatus != 0 ) {
                fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", data_filename, key, exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }

              /***********************************************************
               * NOTE: NO SOURCE PHASE NECESSARY
               * ONLY REORDERING NECESSARY
               ***********************************************************/
              double const threep_norm = threep_operator_norm;
              int const parity_sign = ( 2 * ( igc == 0 ) - 1 ) * ( 2 * ( idim == 0 ) - 1 );
#pragma omp parallel for
              for ( int it = 0; it < T_global; it++ ) {
                threep[ipf][iconf][isrc][igc][idim][0][it][0] = 0.5 * (
                      threep[ipf][iconf][isrc][igc][idim][0][it][0] + parity_sign * buffer[it][0][gc_map[igc]][dim_map[idim]][0] * threep_norm
                    );
                threep[ipf][iconf][isrc][igc][idim][0][it][1] = 0.5 * (
                      threep[ipf][iconf][isrc][igc][idim][0][it][1] + parity_sign * buffer[it][0][gc_map[igc]][dim_map[idim]][1] * threep_norm
                    );
              }
  
            }  /* end of loop on idim */

          }  /* end of loop on igc */

        }  /* end of loop on sink momenta */

      }  /* end of loop on sources */
  
      fini_5level_dtable ( &buffer );
    }  /* end of loop on configs */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "avxn_conn_analyse", "read-threep-h5", g_cart_id == 0 );

#endif  /* end of if _THREEP_AVGX_PLEGMA */

  /**********************************************************/
  /**********************************************************/

#if _RAT_METHOD
      /**********************************************************
       *
       * STATISTICAL ANALYSIS for products and ratios
       *
       * fixed source - sink separation
       *
       **********************************************************/

    double ***** threep_op = init_5level_dtable ( 3, num_conf, num_src_per_conf, T_global, 2 ) ;
    if ( threep_op == NULL ) {
      fprintf ( stderr, "[avxn_conn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        /**********************************************************
         * already ordered from source
         **********************************************************/
        int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;

        if ( g_verbose > 4 ) fprintf ( stdout, "# [avxn_conn_analyse] t_src %3d   dt %3d   tsink %3d\n", conf_src_list[iconf][isrc][2],
            g_sequential_source_timeslice_list[idt], tsink );

        /**********************************************************
         * !!! LOOK OUT:
         *       This includes the momentum orbit average !!!
         **********************************************************/
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

          double const mom[3] = { 
              2 * M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
              2 * M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
              2 * M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global };

          /**********************************************************
           * NO EXTRA PHASES
           **********************************************************/

          /**********************************************************
           * loop on insertion times
           * counted from source
           **********************************************************/
          for ( int it = 0; it < T_global; it++ ) {

            double tensor_sym_sub[4][4][2];
            double tensor_trace[2];
            int const itp1 = ( it + 1 + T_global ) % T_global;
            int const itm1 = ( it - 1 + T_global ) % T_global;

            /**********************************************************
             **********************************************************
             ** trace of the tensor
             **
             ** NOTE: SUM of right- and left-application
             **
             **********************************************************
             **********************************************************/
            for ( int ireim = 0; ireim < 2; ireim++ ) {
#if _THREEP_AVGX_PLEGMA
              tensor_trace[ireim] = 0.25 * (
                  threep[imom][iconf][isrc][0][0][0][it][ireim]
                + threep[imom][iconf][isrc][1][1][0][it][ireim]
                + threep[imom][iconf][isrc][2][2][0][it][ireim]
                + threep[imom][iconf][isrc][3][3][0][it][ireim] );
#else
              tensor_trace[ireim] = 0.25 * 0.25 * ( 
                       /* fwd                                             - bwd                                         */
                /* 
                 * right-application in time direction                                                                  
                 */
                        ( threep[imom][iconf][isrc][0][0][0][it  ][ireim] - threep[imom][iconf][isrc][0][0][1][it  ][ireim] )
                /* 
                 * left-application in time direction
                 *
                 * converted to right application with opposite sign for fwd and bwd and with bwd and fwd shift         
                 */
                +       ( threep[imom][iconf][isrc][0][0][0][itm1][ireim] - threep[imom][iconf][isrc][0][0][1][itp1][ireim] )
                /*
                 * spatial components; just multiplied by 2, since left-, right-application summed with zero 3-momentum 
                 */
                +  2. * ( threep[imom][iconf][isrc][1][1][0][it  ][ireim] - threep[imom][iconf][isrc][1][1][1][it  ][ireim] )
                +  2. * ( threep[imom][iconf][isrc][2][2][0][it  ][ireim] - threep[imom][iconf][isrc][2][2][1][it  ][ireim] )
                +  2. * ( threep[imom][iconf][isrc][3][3][0][it  ][ireim] - threep[imom][iconf][isrc][3][3][1][it  ][ireim] ) );
#endif
            }

            if ( g_verbose > 4 ) fprintf( stdout, "# [avxn_conn_analyse] tensor_trace = %25.16e %25.16e %s %d\n", tensor_trace[0], tensor_trace[1], __FILE__, __LINE__ );

            /**********************************************************
             * temporal gamma, temporal displacement
             **********************************************************/
            for ( int ireim = 0; ireim < 2; ireim++ ) {
#if _THREEP_AVGX_PLEGMA
              tensor_sym_sub[0][0][ireim] = threep[imom][iconf][isrc][0][0][0][it][ireim] - tensor_trace[ireim];
#else
              tensor_sym_sub[0][0][ireim] = /* g_0 D_0 */
                0.25 * ( 
                /*
                 * right-application 
                 */
                /*          fwd                                             - bwd */
                          ( threep[imom][iconf][isrc][0][0][0][it  ][ireim] - threep[imom][iconf][isrc][0][0][1][it  ][ireim] )
                /*
                 *  left-application. becomes right-application with fwd - bwd -> - ( bwd - fwd ) 
                 */
                /*          fwd                                             - bwd */
                    +     ( threep[imom][iconf][isrc][0][0][0][itm1][ireim] - threep[imom][iconf][isrc][0][0][1][itp1][ireim] )
                )
                /*
                 * subtract trace term
                 */
                   - tensor_trace[ireim];
#endif
            }

            /**********************************************************
             * spatial gamma, temporal displacement
             **********************************************************/
            for ( int imu = 1; imu < 4; imu++ ) {
              for ( int ireim = 0; ireim < 2; ireim++ ) {

#if _THREEP_AVGX_PLEGMA
                tensor_sym_sub[imu][0][ireim] = 0.5 * ( 
                    threep[imom][iconf][isrc][imu][0][0][it][ireim] 
                  + threep[imom][iconf][isrc][0][imu][0][it][ireim] );
                
                tensor_sym_sub[0][imu][ireim] = tensor_sym_sub[imu][0][ireim];
#else

                tensor_sym_sub[imu][0][ireim] =   /* 1/2 ( g_i D_0 + g_0 D_i ) */
                  0.125 * ( 
                      /*
                       * right application g_mu Dright_0, fwd - bwd
                       */
                          ( threep[imom][iconf][isrc][imu][0][0][it  ][ireim] - threep[imom][iconf][isrc][imu][0][1][it  ][ireim] )
                      /*
                       * left  application g_mu Dleft_0,  fwd - bwd
                       */
                    +     ( threep[imom][iconf][isrc][imu][0][0][itm1][ireim] - threep[imom][iconf][isrc][imu][0][1][itp1][ireim] )
                    /*
                     * left = right application g_0 D_mu 
                     */
                    + 2 * ( threep[imom][iconf][isrc][0][imu][0][it  ][ireim] - threep[imom][iconf][isrc][0][imu][1][it  ][ireim] )
                );

                /* no trace subtraction for off-diagonal */

                /* symmetrize tensor */
                tensor_sym_sub[0][imu][ireim] = tensor_sym_sub[imu][0][ireim];
#endif
              }  /* end of loop on reim */

            }  /* end of loop on spatial components */

            /**********************************************************
             * spatial gamma, spatial displacement
             **********************************************************/
            for ( int imu = 1; imu < 4; imu++ ) {
              for ( int idim = 1; idim < 4; idim++ ) {
                for ( int ireim = 0; ireim < 2; ireim++ ) {

#if _THREEP_AVGX_PLEGMA
                  tensor_sym_sub[imu][idim][ireim] = 0.5 * (
                      threep[imom][iconf][isrc][imu][idim][0][it][ireim]
                    + threep[imom][iconf][isrc][idim][imu][0][it][ireim] ) - (imu == idim) * tensor_trace[ireim];
#else
                  /* factor 0.5 from symmetrization; factor 2 from sum over l/r application */
                  tensor_sym_sub[imu][idim][ireim] =  /* 1/2 ( g_i D_k + g_k D_i ) */
                    0.125 * 2. * (
                    /* fwd                                                - bwd */
                       threep[imom][iconf][isrc][imu][idim][0][it][ireim] - threep[imom][iconf][isrc][imu][idim][1][it][ireim] 
                    /* fwd                                                - bwd */
                     + threep[imom][iconf][isrc][idim][imu][0][it][ireim] - threep[imom][iconf][isrc][idim][imu][1][it][ireim] 
                  )
                  /*
                   * subtract trace term for diagonal elements
                   */
                    - (imu == idim) * tensor_trace[ireim]; 
#endif
                }
              }
            } 

            /**********************************************************
             * show symmetrized tensor
             **********************************************************/
            if ( g_verbose > 4 ) {
              fprintf( stdout, "# tensor_sym_sub %c %6d %3d %3d  p %3d %3d %3d \n",
                  conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2], it,
                  g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2]);
              for ( int imu = 0; imu < 4; imu++ ) {
                for ( int idim = 0; idim < 4; idim++ ) {
                  fprintf( stdout, " %3d %3d %25.16e %25.16e\n", imu, idim, tensor_sym_sub[imu][idim][0], tensor_sym_sub[imu][idim][1] );
                }
              }

            }

            /**********************************************************
             * O44
             * real part only
             *
             * include overall - sign for fermion loop on Wick contraction
             **********************************************************/

            threep_op[0][iconf][isrc][it][0] += tensor_sym_sub[0][0][0];
            threep_op[0][iconf][isrc][it][1] += tensor_sym_sub[0][0][1];

            /**********************************************************
             * Oik
             * again only real part
             **********************************************************/
            for ( int i = 0; i < 3; i++ ) {
              for ( int k = 0; k < 3; k++ ) {
                threep_op[1][iconf][isrc][it][0] += tensor_sym_sub[i+1][k+1][0] * mom[i] * mom[k];
                threep_op[1][iconf][isrc][it][1] += tensor_sym_sub[i+1][k+1][1] * mom[i] * mom[k];
              }
            }

            /**********************************************************
             * O4k
             * want real observable,
             * so set real part of observable to imaginary part of tensor
             *
             * NOTE: temporal component = 0
             *       spatial components = 1, 2, 3
             **********************************************************/
            for ( int k = 0; k < 3; k++ ) {
              threep_op[2][iconf][isrc][it][0] += tensor_sym_sub[0][k+1][1] * mom[k];
              threep_op[2][iconf][isrc][it][1] += tensor_sym_sub[0][k+1][0] * mom[k];
            }

          }  /* end of loop on it */

        }  /* end of loop on imom */

        /**********************************************************
         * normalize
         *
         * add factor of 2 to normalization to account for combination of
         * u,+p , d,-p , u,-p , d,+p
         **********************************************************/
        /* O44 simple orbit average */
        double const norm44 = 2. / g_sink_momentum_number;
        for ( int it = 0; it < 2 * T_global; it++ ) {
          threep_op[0][iconf][isrc][0][it] *= norm44;
        }

        /* Oik divide by (p^2)^2 */
        double const mom[3] = {
          2 * M_PI * g_sink_momentum_list[0][0] / (double)LX_global,
          2 * M_PI * g_sink_momentum_list[0][1] / (double)LY_global,
          2 * M_PI * g_sink_momentum_list[0][2] / (double)LZ_global };

        double const mom_squared = mom[0] * mom[0] + mom[1] * mom[1] + mom[2] * mom[2];

        int const mom_is_zero = g_sink_momentum_list[0][0] == 0 && g_sink_momentum_list[0][1] == 0 && g_sink_momentum_list[0][2] == 0;

        double const normik = mom_is_zero ? 0. : 2. / mom_squared / (double)g_sink_momentum_number;

        for ( int it = 0; it < 2 * T_global; it++ ) {
          threep_op[1][iconf][isrc][0][it] *= normik;
        }

        /* O4k divide by (p^2) */
        double const norm4k = mom_is_zero ? 0. : 2. / mom_squared / (double)g_sink_momentum_number;

        for ( int it = 0; it < 2 * T_global; it++ ) {
          threep_op[2][iconf][isrc][0][it] *= norm4k;
        }

      }  /* end of loop on isrc */
    }  /* end of loop on iconf */

    /**********************************************************
     * write 3pt function to ascii file, per source
     **********************************************************/
    if ( write_data == 1) {
      for ( int k = 0; k <= 2; k++ ) {

        /**********************************************************
         * write 3pt k
         **********************************************************/
        for ( int ireim = 0; ireim < 2; ireim++ ) {
          sprintf ( filename, "threep.%s.conn.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s.corr",
              flavor_type_3pt[flavor_id_3pt],
              threep_tag[k],
              g_sequential_source_timeslice_list[idt],
              g_sink_momentum_list[0][0],
              g_sink_momentum_list[0][1],
              g_sink_momentum_list[0][2], reim_str[ireim] );

          write_data_real2_reim ( threep_op[k], filename, conf_src_list, num_conf, num_src_per_conf, T_global, ireim );
        }  /* end of loop on ireim */
      }
    }  /* end of if write_data */

    /**********************************************************
     *
     * STATISTICAL ANALYSIS for threep
     *
     * with fixed source - sink separation
     *
     **********************************************************/
    for ( int k = 0; k <= 2; k++ ) {

      for ( int ireim = 0; ireim < 2; ireim++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf ( stderr, "[avxn_conn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          for ( int it = 0; it < T_global; it++ ) {
            double dtmp = 0.;
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              dtmp += threep_op[k][iconf][isrc][it][ireim];
            }
            data[iconf][it] = dtmp / (double)num_src_per_conf;
          }
        }

        char obs_name[100];
        sprintf ( obs_name, "threep.%s.conn.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
            flavor_type_3pt[flavor_id_3pt],
            threep_tag[k],
            g_sequential_source_timeslice_list[idt],
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2], reim_str[ireim] );

        if ( num_conf >= 6 )
        {
          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[avxn_conn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
        } else {
          fprintf ( stderr, "[avxn_conn_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
        }

        if ( write_data == 1 )
        {
          sprintf ( filename, "%s.corr", obs_name );
          write_data_real ( data, filename, conf_src_list, num_conf, T_global );
        }

        fini_2level_dtable ( &data );
      }  /* end of loop on reim */

      /**********************************************************
       *
       * STATISTICAL ANALYSIS for ratio 
       *   with source - sink fixed
       *
       **********************************************************/
      for ( int ireim = 0; ireim < 2; ireim++ ) {

        /* UWerr parameters */
        int nT = g_sequential_source_timeslice_list[idt] + 1;
        int narg          = 2;
        int arg_first[2]  = { 0, nT };
        int arg_stride[2] = { 1,  0 };
        char obs_name[100];

        double ** data = init_2level_dtable ( num_conf, nT + 1 );
        if ( data == NULL ) {
          fprintf ( stderr, "[avxn_conn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /**********************************************************
         *
         **********************************************************/
        src_avg_real2_reim ( data, threep_op[k], num_conf, num_src_per_conf, nT, ireim );

#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          double dtmp = 0.;
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

            int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
            int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;

            dtmp += twop_orbit[iconf][isrc][tsink][ireim] + twop_fold_propagator * twop_orbit[iconf][isrc][tsink2][ireim];

          }
          data[iconf][nT] = dtmp / (double)num_src_per_conf / ( 1 + abs( twop_fold_propagator ) );
        }

        sprintf ( obs_name, "ratio.%s.conn.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
            flavor_type_3pt[flavor_id_3pt],
            threep_tag[k],
            g_sequential_source_timeslice_list[idt],
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2], reim_str[ireim] );

        if ( num_conf >= 6 )
        {
          exitstatus = apply_uwerr_func ( data[0], num_conf, nT+1, nT, narg, arg_first, arg_stride, obs_name, ratio_1_1, dratio_1_1 );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[avxn_conn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }
        }

        fini_2level_dtable ( &data );

      }  /* end of loop on reim */

    }

    fini_5level_dtable ( &threep_op );

#endif  /* end of ifdef _RAT_METHOD */

  
    fini_8level_dtable ( &threep );

  }  /* end of loop on dt = source sink time separations */

  /**********************************************************/
  /**********************************************************/

#if _TWOP_AFF || _THREEP_AFF
  read_aff_contraction ( NULL, NULL, NULL, NULL, 0 );
#endif

  fini_5level_dtable ( &twop );
  fini_4level_dtable ( &twop_orbit );

  /**********************************************************
   * free and finalize
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

  gettimeofday ( &endtime, (struct timezone *)NULL );
  show_time ( &starttime, &endtime, "avxn_conn_analyse", "avxn_conn_analyse", g_cart_id == 0 );

  return(0);
}
