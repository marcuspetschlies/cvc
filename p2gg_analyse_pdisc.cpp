/****************************************************
 * p2gg_analyse_pdisc
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <sys/time.h>
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
#include "derived_quantities.h"
#include "gamma.h"
#include "contract_loop_inline.h"

#define _CVC_H5           1

#define _LOOP_READ_H5    0
#define _LOOP_READ_ASCII 1

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  EXIT(0);
}

/****************************************************/
/****************************************************/

/****************************************************
 * MAIN PROGRAM
 ****************************************************/

int main(int argc, char **argv) {
  
  double const TWO_MPI = 2. * M_PI;

  char const reim_str[2][3] = {"re" , "im"};

  char const gamma_id_to_ascii[16][10] = {
    "gt",
    "gx",
    "gy",
    "gz",
    "1",
    "g5",
    "gtg5",
    "gxg5",
    "gyg5",
    "gzg5",
    "gtgx",
    "gtgy",
    "gtgz",
    "gxgy",
    "gxgz",
    "gygz"
  };


  int c;
  int filename_set = 0;
  int check_momentum_space_WI = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";
  struct timeval ta, tb;
  int operator_type = -1;
  int loop_stats = 1;
  int hvp_stats = 1;
  int write_data = 0;
  double loop_norm = 1.;
  double twop_norm = 1.;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:w:E:l:t:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_pdisc] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_pdisc] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_pdisc] write_data set to %d\n", write_data );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [p2gg_analyse_pdisc] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'l':
      loop_norm = atof ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_pdisc] loop_norm set to %f\n", loop_norm );
      break;
    case 't':
      twop_norm = atof ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_pdisc] twop_norm set to %f\n", twop_norm );
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
  /* fprintf(stdout, "# [p2gg_analyse_pdisc] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [p2gg_analyse_pdisc] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_analyse_pdisc] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_analyse_pdisc] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_analyse_pdisc] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_analyse_pdisc] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_analyse_pdisc] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_analyse_pdisc] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[p2gg_analyse_pdisc] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[p2gg_analyse_pdisc] Error from init_Xlevel_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [p2gg_analyse_pdisc] comment %s\n", line );
      continue;
    }

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

  if ( g_verbose > 5 ) {
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


  /***********************************************************
   ***********************************************************
   **
   ** VV 2-pt
   **
   ***********************************************************
   ***********************************************************/

  double ****** hvp = init_6level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, 3, 3, 2 * T );
  if ( hvp == NULL ) {
    fprintf(stderr, "[p2gg_analyse_pdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

#if _CVC_H5

  /***********************************************
   * reader 
   ***********************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  for ( int iconf = 0; iconf < num_conf; iconf++ ) 
  {
    sprintf ( filename, "stream_%c/%s/%d/%s.%d.h5", 
          conf_src_list[iconf][0][0],
          filename_prefix, 
          conf_src_list[iconf][0][1],
          filename_prefix2,
          conf_src_list[iconf][0][1] );

    fprintf(stdout, "# [p2gg_analyse_pdisc] reading data from file %s\n", filename);

    double ***** buffer = init_5level_dtable( T, 2, 3, 3, 2 );
    if( buffer == NULL ) {
      fprintf(stderr, "[p2gg_analyse_pdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(15);
    }

    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
    {
      int const gts = conf_src_list[iconf][isrc][2];

      for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
      {
        int const pf[3] = {
          g_sink_momentum_list[imom][0],
          g_sink_momentum_list[imom][1],
          g_sink_momentum_list[imom][2] };

        /**********************************************************
         * neutral case
         **********************************************************/
        char key[100];
        sprintf ( key , "/t%d/pfx%dpfy%dpfz%d", gts, pf[0], pf[1], pf[2] );

        exitstatus = read_from_h5_file (  buffer[0][0][0][0], filename, key,  "double", io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[p2gg_analyse] Error from read_from_h5_file for file %s key %s    %s %d\n",
             filename, key, __FILE__, __LINE__);
          EXIT(15);
        }
              
#pragma omp parallel for
        for ( int it = 0; it < T_global; it++ ) 
        {
          int const tt = ( it + gts + T_global ) % T_global;
        
          for( int mu = 0; mu < 3; mu++) 
          {
            for( int nu = 0; nu < 3; nu++) 
            {
              // include flavor sum, up + dn
              hvp[iconf][isrc][imom][mu][nu][2*it  ] = ( buffer[tt][0][mu][nu][0] + buffer[tt][1][mu][nu][0] ) * twop_norm;
              hvp[iconf][isrc][imom][mu][nu][2*it+1] = ( buffer[tt][0][mu][nu][1] + buffer[tt][1][mu][nu][1] ) * twop_norm;
            }
          }
        }
      
      
      }  // end of loop on momenta
    }  /* end of loop on source */
        
    fini_5level_dtable( &buffer );

  }  // end of loop on configs

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "p2gg_analyse", "read-pgg-flavor-tensor-components-neutral-h5", g_cart_id == 0 );
      
#endif  /* end of if _CVC_H5 */


  /****************************************
   * show all data
   ****************************************/
  if ( g_verbose > 5 ) {
    gettimeofday ( &ta, (struct timezone *)NULL );

    for ( int iconf = 0; iconf < num_conf; iconf++ )
    {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
      {
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) 
        {
          for ( int mu = 0; mu < 3; mu++ ) 
          {
            for ( int nu = 0; nu < 3; nu++ ) 
            {
              for ( int it = 0; it < T; it++ ) 
              {
                fprintf ( stdout, "%c  %6d s %3d p %3d %3d %3d m %d %d hvp %3d  %25.16e %25.16e\n", 
                    conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2],
                    g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], mu, nu, it, 
                    hvp[iconf][isrc][imom][mu][nu][2*it], hvp[iconf][isrc][imom][mu][nu][2*it+1] );
              }
            }
          }
        }
      }
    }
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_analyse_pdisc", "show-all-data", g_cart_id == 0 );
  }

  if ( hvp_stats ) 
  {
    /****************************************
     * combine source locations 
     ****************************************/
  
    double ***** hvp_src_avg = init_5level_dtable ( num_conf, g_sink_momentum_number, 3, 3, 2 * T_global );
    if ( hvp_src_avg == NULL ) {
      fprintf(stderr, "[p2gg_analyse_pdisc] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(16);
    }
  
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) 
    {
      memset( hvp_src_avg[iconf][0][0][0], 0, 18 * T_global * g_sink_momentum_number*sizeof(double) );
      double const norm = 1. / (double)num_src_per_conf;
  
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) 
      {
        for ( int mu = 0; mu < 3; mu++ ) {
        for ( int nu = 0; nu < 3; nu++ ) {
  
          for ( int it = 0; it < T_global; it++ ) 
          {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
            {
              hvp_src_avg[iconf][imom][mu][nu][2*it  ] += hvp[iconf][isrc][imom][mu][nu][2*it  ];
  
              hvp_src_avg[iconf][imom][mu][nu][2*it+1] += hvp[iconf][isrc][imom][mu][nu][2*it+1];
            }
            hvp_src_avg[iconf][imom][mu][nu][2*it  ] *= norm;
            hvp_src_avg[iconf][imom][mu][nu][2*it+1] *= norm;
          }
        }}
      }
    }
  
    double ** data = init_2level_dtable ( num_conf, T_global );
    if ( data == NULL ) {
      fprintf(stderr, "[p2gg_analyse_pdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(16);
    }

    /****************************************
     * STATISTICAL ANALYSIS
     ****************************************/
    for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
    {
      for ( int imu = 0; imu < 3; imu++)
      {
        for ( int inu = 0; inu < 3; inu++ )
        {
          for ( int ireim = 0; ireim < 2; ireim++ ) 
          {
  
#pragma omp parallel for
            for ( int iconf = 0; iconf < num_conf; iconf++)
            {
              for ( int it = 0; it < T_global; it++ )
              {
                data[iconf][it] = hvp_src_avg[iconf][imom][imu][inu][2*it+ireim];
              }
            }
  
            char obs_name[100];
            sprintf ( obs_name, "%s.PX%d_PY%d_PZ%d.mu%d.nu%d.%s", filename_prefix2,
                g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
               imu, inu, reim_str[ireim] );
  
            if ( num_conf >= 6 )
            {
              /* apply UWerr analysis */
              exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[p2gg_analyse_pdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }
            }
  
          }  /* end of loop on re / im */
        }
      }
    }  // end of loop on momenta
  
    fini_2level_dtable ( &data );
    fini_5level_dtable ( &hvp_src_avg );
  
  }  /* end of if loop_stats  */

  /**********************************************************
   **********************************************************
   **
   ** P -> gg disconnected P-loop 3-point function
   **
   **********************************************************
   **********************************************************/

  int const seq_source_momentum[3] = { 0, 0, 0 };
  int const seq_source_gamma = 4;

  /**********************************************************
   * read loop matrix data
   **********************************************************/
  double ** loops = init_2level_dtable ( num_conf, 2*T );
  if ( loops == NULL ) {
    fprintf ( stderr, "[p2gg_analyse_pdisc] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(112);
  }

  /**********************************************************
   * read the loop data from ASCII file
   **********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );
#if _LOOP_READ_ASCII
  for ( int iconf = 0; iconf < num_conf; iconf++ ) 
  {
    sprintf ( filename, "stream_%c/%s/loop.%.4d.stoch.Scalar.nev0.PX%d_PY%d_PZ%d.g_%s",
        conf_src_list[iconf][0][0], filename_prefix3, conf_src_list[iconf][0][1],
        seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], gamma_id_to_ascii[seq_source_gamma]); 

    if ( g_verbose > 0 ) fprintf ( stdout, "# [p2gg_analyse_pdisc] reading loop data from file %s %s %d\n", filename, __FILE__, __LINE__ );
 
    FILE * ofs = fopen ( filename, "r" );
    if ( ofs == NULL ) {
      fprintf ( stderr, "[p2gg_analyse_pdisc] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
      EXIT(113);
    }

    for ( int isample = 0; isample < g_nsample; isample++ ) 
    {
      int itmp;
      double dtmp[2];

      for ( int t = 0; t < T_global; t++ ) 
      {
        if ( fscanf ( ofs, "%d %lf %lf\n", &itmp, dtmp, dtmp+1 ) != 3 ) 
        {
          fprintf ( stderr, "[p2gg_analyse_pdisc] Error from fscanf %s %d\n", __FILE__, __LINE__ );
          EXIT(126);
        }
        loops[iconf][2*t  ] += dtmp[0];
        loops[iconf][2*t+1] += dtmp[1];
      }
    }
    
    fclose ( ofs );

#pragma omp parallel for
    for ( int t = 0; t < T_global; t++ ) 
    {
      loops[iconf][2*t  ] /= (double)g_nsample;
      loops[iconf][2*t+1] /= (double)g_nsample;
    }

  }  /* end of loop on configurations */

  if ( g_verbose > 5 )
  {
    for ( int iconf = 0; iconf < num_conf; iconf++ )
    {
      for ( int t = 0; t < T_global; t++ )
      {
        fprintf ( stdout, "l %6d %4d %25.16e %25.16e\n", iconf, t, loops[iconf][2*t], loops[iconf][2*t+1] );
      }
    }
  }

#endif  /* end of if loop read mode */

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "p2gg_analyse_pdisc", "read-loop-data", g_cart_id == 0 );

  /**********************************************************
   * loop on sequential source timeslices
   **********************************************************/
  for ( int isequential_source_timeslice = 0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++)
  {
    int const sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];

    /***********************************************************
     * prepare the disconnected contribution loop x hvp
     ***********************************************************/
 
    double ****** pgg_disc = init_6level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, 3, 3, 2 * T_global );
    if ( pgg_disc == NULL ) {
      fprintf(stderr, "[p2gg_analyse_pdisc] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(16);
    }

#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) 
    {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
      {
        /* calculate the sequential source timeslice; tseq = source timeslice + source-sink-timeseparation */
        /* pseudoscalar timeslice forward from source time */
        int const tseq  = ( conf_src_list[iconf][isrc][2] + sequential_source_timeslice + T_global ) % T_global;
        /* pseudoscalar timeslice backward from source time */
        int const tseq2 = ( conf_src_list[iconf][isrc][2] - sequential_source_timeslice + T_global ) % T_global;

        if ( g_verbose > 3 ) fprintf ( stdout, "# [p2gg_analyse_pdisc] tsnk (v) %3d dt %3d tsrc (loop) + %3d / - %3d\n",
            conf_src_list[iconf][isrc][2], sequential_source_timeslice, tseq , tseq2 );

        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) 
        {
          for ( int s1 = 0; s1 < 3; s1++ ) 
          {
            for ( int s2 = 0; s2 < 3; s2++ ) 
            {
              /* loop on forward time */
              for ( int it = 0; it < T_global; it++ ) 
              {
                /* symmetric backward time */
                int const it2 = ( -it + T_global ) % T_global;

                /* time-forward data */
                double const a[2] = {
                  hvp[iconf][isrc][imom][s1][s2][2*it  ],
                  hvp[iconf][isrc][imom][s1][s2][2*it+1]  };

                double const b[2] = { loops[iconf][2*tseq  ], loops[iconf][2*tseq+1] };

                //pgg_disc[iconf][isrc][imom][s1][s2][2*it  ] = a[0] * b[0] - a[1] * b[1];
                //pgg_disc[iconf][isrc][imom][s1][s2][2*it+1] = a[0] * b[1] + a[1] * b[0];
                pgg_disc[iconf][isrc][imom][s1][s2][2*it  ] = -a[1] * b[1];
                pgg_disc[iconf][isrc][imom][s1][s2][2*it+1] =  a[0] * b[1];

                /* time-backward data */
                double const c[2] = {
                  hvp[iconf][isrc][imom][s1][s2][2*it2  ],
                  hvp[iconf][isrc][imom][s1][s2][2*it2+1]  };

                double const d[2] = { loops[iconf][2*tseq2  ], loops[iconf][2*tseq2+1] };

                pgg_disc[iconf][isrc][imom][s1][s2][2*it  ] -= -c[1] * d[1];
                pgg_disc[iconf][isrc][imom][s1][s2][2*it+1] -=  c[0] * d[1];

                pgg_disc[iconf][isrc][imom][s1][s2][2*it  ] *= 0.5;
                pgg_disc[iconf][isrc][imom][s1][s2][2*it+1] *= 0.5;

              }  /* end of loop on timeslices */
            }  /* end of loop on s2 */
          }  /* end of loop on s1 */
        }  /* end of loop on momenta */
      }  /* end of loop on sources per conf */
    }  /* end of loop on configurations */
    
    if ( g_verbose > 5 )
    {
      for ( int iconf = 0; iconf < num_conf; iconf++ )
      {
        for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
        {
          for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
          {
            for ( int s1 = 0; s1 < 3; s1++ )
            {
              for ( int s2 = 0; s2 < 3; s2++ )
              {
                for ( int it = 0; it < T_global; it++ )
                {
                  fprintf (stdout, "pgg_disc %6d %4d    %3d %3d %3d     %3d %3d    %4d %25.16e %25.16e\n", 
                      conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2],
                      g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][1], s1, s2, it, 
                      pgg_disc[iconf][isrc][imom][s1][s2][2*it  ], pgg_disc[iconf][isrc][imom][s1][s2][2*it+1] );
                }
              }
            }
          }
        }
      }
    }    

    /****************************************
     * pgg_disc source average
     ****************************************/
    double ***** pgg = init_5level_dtable ( num_conf, g_sink_momentum_number, 3, 3, 2 * T_global );
    if ( pgg == NULL ) {
      fprintf(stderr, "[p2gg_analyse_pdisc] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(16);
    }
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) 
    {
      for ( int i = 0; i < g_sink_momentum_number * 18 * T_global; i++ ) 
      {
        pgg[iconf][0][0][0][i] = 0.;

        for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
        {
          pgg[iconf][0][0][0][i] += pgg_disc[iconf][isrc][0][0][0][i];
        }
        pgg[iconf][0][0][0][i] /= (double)num_src_per_conf;
      }
    }

    /****************************************
     * statistical analysis for orbit average
     *
     * ASSUMES MOMENTUM LIST IS AN ORBIT AND
     * SEQUENTIAL MOMENTUM IS ZERO
     ****************************************/
    for ( int ireim = 0; ireim < 2; ireim++ ) 
    {
      double ** data = init_2level_dtable ( num_conf, T_global );
      if ( data == NULL ) {
        fprintf ( stderr, "[p2gg_analyse_pdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(79);
      }

#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++)
      {
        for ( int it = 0; it < T_global; it++ )
        {
          double q[3] = {
            g_sink_momentum_list[0][0] * TWO_MPI / LX_global,
            g_sink_momentum_list[0][1] * TWO_MPI / LY_global,
            g_sink_momentum_list[0][2] * TWO_MPI / LZ_global };

          double const qq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];

          double const norm = 1. / ( qq * g_sink_momentum_number );
            
          for ( int imom = 0; imom < g_sink_momentum_number; imom++)
          {
            q[0] = g_sink_momentum_list[imom][0] * TWO_MPI / LX_global;
            q[1] = g_sink_momentum_list[imom][1] * TWO_MPI / LY_global;
            q[2] = g_sink_momentum_list[imom][2] * TWO_MPI / LZ_global;

            data[iconf][it] +=
              q[0] * ( pgg[iconf][imom][1][2][2*it + ireim] - pgg[iconf][imom][2][1][2*it + ireim] )
            + q[1] * ( pgg[iconf][imom][2][0][2*it + ireim] - pgg[iconf][imom][0][2][2*it + ireim] )
            + q[2] * ( pgg[iconf][imom][0][1][2*it + ireim] - pgg[iconf][imom][1][0][2*it + ireim] );
          }
          data[iconf][it] *= norm;
        }
      }

      char obs_name[100];
      sprintf ( obs_name, "pgg_pdisc.t%d.PX%d_PY%d_PZ%d.%s", 
          sequential_source_timeslice,
          g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

      if ( num_conf >= 6 )
      {
        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[p2gg_analyse_pdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }
      }

      if ( write_data == 1 ) {
        sprintf ( filename, "%s.corr", obs_name );
        FILE * ofs = fopen ( filename, "w" );
        if ( ofs == NULL ) {
          fprintf ( stdout, "[p2gg_analyse_pdisc] Error from fopen for file %s %s %d\n", obs_name, __FILE__, __LINE__ );
          EXIT(12);
        }

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          /* for ( int tau = -T_global/2+1; tau <= T_global/2; tau++ )  */
          for ( int it = 0; it < T_global; it++ ) 
          {
            /* int const it = ( tau < 0 ) ? tau + T_global : tau; */

            fprintf ( ofs, "%5d%25.16e   %c %8d\n", it, data[iconf][it], conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
          }
        }
        fclose ( ofs );

      }  /* end of if write data */

      fini_2level_dtable ( &data );
    }  /* end of loop on real / imag */


    /**********************************************************
     * free p2gg table
     **********************************************************/
    fini_6level_dtable ( &pgg_disc );
    fini_5level_dtable ( &pgg );

  }  /* end of loop on sequential source timeslices */

  fini_2level_dtable ( &loops ); 

  fini_6level_dtable ( &hvp );

#if 0
#endif  // of if 0

  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/

  fini_3level_itable ( &conf_src_list );

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_analyse_pdisc] %s# [p2gg_analyse_pdisc] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_analyse_pdisc] %s# [p2gg_analyse_pdisc] end of run\n", ctime(&g_the_time));
  }

  return(0);

}

