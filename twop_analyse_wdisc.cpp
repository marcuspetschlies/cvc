/****************************************************
 * twop_analyse_wdisc.c
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

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  EXIT(0);
}

#define _LOOP_H5 1
#define _LOOP_ASCII 0

#define _PLEGMA_CONVENTION 0

int main(int argc, char **argv) {
  
  double const TWO_MPI = 2. * M_PI;

  char const reim_str[2][3] = {"re" , "im"};

  /* int const reim_sign[2][2] = { {1, 1}, {1, -1} }; */

  char const oet_type_tag[2][8] = { "std", "gen" };

#if _PLEGMA_CONVENTION
  char const loop_type_tag[3][20] = { "localLoops", "oneD", "oneDC" };
#else
  char const loop_type_tag[2][20] = { "Scalar", "dOp" };
#endif

  int const gamma_tmlqcd_to_binary[16] = { 8, 1, 2, 4, 0,  15, 7, 14, 13, 11, 9, 10, 12, 3, 5, 6 };

  /***********************************************************
   * sign for g5 Gamma^\dagger g5
   *                          0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   *
   ***********************************************************/
  /* int const sigma_g5d[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 }; */

  /***********************************************************
   * sign for gt Gamma gt
   *                          0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   *
   ***********************************************************/
  int const sigma_t[16] ={  1, -1, -1, -1, +1, -1,  -1,  +1,  +1,  +1,  -1,  -1,  -1,  +1,  +1,  +1 };


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "na";
  int write_data = 0;
  int loop_nev = -1;
  int loop_nsample = 0;
  int loop_step = 1;
  int loop_transpose = 0;
  int loop_type = -1;
  int oet_type = -1;
  int loop_type_reim[2] = { -1, -1};
  double loop_norm[2] = { 1, 1. };
  char flavor_tag[2][20];

  /* struct timeval ta, tb; */

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:E:w:D:r:n:s:v:m:Q:q:O:t:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'w':
      write_data = atoi( optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] write_data set to %d\n", write_data );
      break;
    case 'D':
      loop_type = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] loop_type set to %d\n", loop_type );
      break;
    case 'O':
      oet_type = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] oet_type set to %d\n", oet_type );
      break;
    case 'r':
      sscanf ( optarg, "%d,%d", loop_type_reim, loop_type_reim+1 );
      fprintf ( stdout, "# [twop_analyse_wdisc] loop_type_reim set to %d  %d\n", loop_type_reim[0], loop_type_reim[1]);
      break;
    case 'm':
      loop_nsample = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] loop_nsample set to %d\n", loop_nsample );
      break;
    case 'n':
      sscanf ( optarg, "%lf,%lf", loop_norm, loop_norm+1 );
      fprintf ( stdout, "# [twop_analyse_wdisc] loop_norm set to %e  %e\n", loop_norm[0], loop_norm[1] );
      break;
    case 't':
      loop_transpose = atoi( optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] loop_transpose set to %d\n", loop_transpose );
      break;
    case 's':
      loop_step = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] loop_step set to %d\n", loop_step );
      break;
    case 'v':
      loop_nev = atoi( optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] loop_nev set to %d\n", loop_nev );
      break;
    case 'q':
      strcpy ( flavor_tag[1], optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] source flavor tag set to %s\n", flavor_tag[1] );
      break;
    case 'Q':
      strcpy ( flavor_tag[0], optarg );
      fprintf ( stdout, "# [twop_analyse_wdisc] sink   flavor tag set to %s\n", flavor_tag[0] );
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
  /* fprintf(stdout, "# [twop_analyse_wdisc] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [twop_analyse_wdisc] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [twop_analyse_wdisc] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [twop_analyse_wdisc] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[twop_analyse_wdisc] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[twop_analyse_wdisc] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  unsigned int const VOL3 = LX_global * LY_global * LZ_global;

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[twop_analyse_wdisc] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [twop_analyse_wdisc] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[twop_analyse_wdisc] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  read_source_coords_list ( conf_src_list, num_conf, num_src_per_conf, ensemble_name );

  /***********************************************************
   * how to normalize loops
   ***********************************************************/
  if ( strcmp ( oet_type_tag[oet_type], "std"  ) == 0 ) {
    loop_norm[0] *= -1;  /* -1 from single g5^ukqcd entering in the std-oet  */
    loop_norm[1] *= -1;  /* -1 from single g5^ukqcd entering in the std-oet  */
  }
  if ( g_verbose > 0 ) fprintf ( stdout, "# [twop_analyse_wdisc] oet_type %s loop_norm = %e   %e\n", oet_type_tag[oet_type], loop_norm[0], loop_norm[1] );


#if 0
  /**********************************************************
   * sink momentum list modulo parity
   **********************************************************/
  int ** sink_momentum_list = init_2level_itable ( g_sink_momentum_number, 3 );
  memcpy ( sink_momentum_list[0], g_sink_momentum_list[0], 3 * sizeof(int) );
  int sink_momentum_number = 1;

  for ( int i = 1; i < g_sink_momentum_number; i++ ) {
    int have_pmom = 0;
    for ( int k = 0; k < sink_momentum_number; k++ ) {
      if ( ( sink_momentum_list[k][0] == -g_sink_momentum_list[i][0] ) &&
           ( sink_momentum_list[k][1] == -g_sink_momentum_list[i][1] ) &&
           ( sink_momentum_list[k][2] == -g_sink_momentum_list[i][2] ) ) {
        have_pmom = 1;
        break;
      }
    }
    if ( !have_pmom ) {
      memcpy ( sink_momentum_list[sink_momentum_number], g_sink_momentum_list[i], 3 * sizeof(int) );
      sink_momentum_number++;
    }
  }
  if ( g_verbose > 2 ) {
    for ( int i = 0; i < sink_momentum_number; i++ ) {
      fprintf( stdout, "# [twop_analyse_wdisc] sink momentum %3d    %3d %3d %3d\n", i,
          sink_momentum_list[i][0], sink_momentum_list[i][1], sink_momentum_list[i][2] );
    }
  }
#endif


  /***********************************************************
   ***********************************************************
   **
   ** TWOPT
   **
   ***********************************************************
   ***********************************************************/

  double ***** loops_proj_sink  = init_5level_dtable ( g_sink_momentum_number, g_sink_gamma_id_number,   num_conf, loop_nsample, 2*T_global );
  if ( loops_proj_sink == NULL ) {
    fprintf ( stderr, "[twop_analyse_wdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }
  
  double ***** loops_proj_source = init_5level_dtable ( g_sink_momentum_number, g_source_gamma_id_number, num_conf, loop_nsample, 2*T_global );
  if ( loops_proj_source == NULL ) {
    fprintf ( stderr, "[twop_analyse_wdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

#if _LOOP_ASCII
  
  double ***** loops_matrix = init_5level_dtable ( num_conf, loop_nsample, T_global, 4, 8 );
  if ( loops_matrix == NULL ) {
    fprintf ( stderr, "[twop_analyse_wdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /**********************************************************
   * loop on momenta
   **********************************************************/
  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) 
  {

    //for ( int ipsign = 0; ipsign < 2; ipsign++ ) {

    //  int const psign = 1 - 2 * ipsign;

      /* int const sink_momentum[3] = {
          psign * sink_momentum_list[isink_momentum][0],
          psign * sink_momentum_list[isink_momentum][1],
          psign * sink_momentum_list[isink_momentum][2] }; */

      int const sink_momentum[3] = {
          g_sink_momentum_list[isink_momentum][0],
          g_sink_momentum_list[isink_momentum][1],
          g_sink_momentum_list[isink_momentum][2] };

      /**********************************************************
       * loop on momenta
       **********************************************************/
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      
        int replica = -1;
        switch ( conf_src_list[iconf][0][0] ) 
        {
          case 'a': replica = 0; break;
          case 'b': replica = 1; break;
        }


        if ( loop_nev < 0 ) {
          sprintf ( filename, "stream_%c/%s/loop.%d.stoch.%s.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0], filename_prefix,
              conf_src_list[iconf][0][1], loop_type_tag[loop_type], sink_momentum[0], sink_momentum[1], sink_momentum[2] );
        } else {
          sprintf ( filename, "stream_%c/%s/loop.%.4d_r%d.stoch.%s.%s.nev%d.Nstoch%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0],
              filename_prefix, conf_src_list[iconf][0][1], replica, oet_type_tag[oet_type], loop_type_tag[loop_type], loop_nev,
              loop_nsample,
              sink_momentum[0], sink_momentum[1], sink_momentum[2] );
          /* sprintf ( filename, "%s/loop.%.4d_r%c.stoch.%s.%s.nev%d.Nstoch%d.PX%d_PY%d_PZ%d",
              filename_prefix, conf_src_list[iconf][0][1], conf_src_list[iconf][0][0],
              oet_type_tag[oet_type], loop_type_tag[loop_type], loop_nev, loop_nsample, sink_momentum[0], sink_momentum[1], sink_momentum[2] ); */
        }

        if ( g_verbose > 0 ) fprintf ( stdout, "# [twop_analyse_wdisc] reading loop data from file %s %s %d\n", filename, __FILE__, __LINE__ );
        FILE * ofs = fopen ( filename, "r" );
        if ( ofs == NULL ) {
          fprintf ( stderr, "[twop_analyse_wdisc] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT(113);
        }

        for ( int isample = 0; isample < loop_nsample; isample++ ) {
          int itmp[4];
          for ( int t = 0; t < T_global; t++ ) {
            for ( int mu = 0; mu < 4; mu++ ) {
            for ( int nu = 0; nu < 4; nu++ ) {
              if ( fscanf ( ofs, 
                    /* "%d %d %d %d %lf %lf\n", itmp, itmp+1, itmp+2, itmp+3, */
                    "%d %d %d %lf %lf\n", itmp, itmp+1, itmp+2,
                    loops_matrix[iconf][isample][t][mu]+2*nu, loops_matrix[iconf][isample][t][mu]+2*nu+1 ) != 5 ) {
                fprintf ( stderr, "[twop_analyse_wdisc] Error from fscanf %s %d\n", __FILE__, __LINE__ );
                EXIT(126);
              }
              /* show all data */
              if ( g_verbose > 5 ) {
                fprintf ( stdout, "loop_mat c %6d s %3d t %3d m %d nu %d l %25.16e %25.16e\n", Nconf, isample, t, mu, nu,
                    loops_matrix[iconf][isample][t][mu][2*nu], loops_matrix[iconf][isample][t][mu][2*nu+1] );
              }
            }}
          }
        }
        fclose ( ofs );

      }  /* end of loop on configurations */

      /**********************************************************
       * loop on sink gamma
       **********************************************************/
      for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ )
      {
     
        int const gamma_id = g_sink_gamma_id_list[ isink_gamma ];

        int loop_st_sign = sigma_t [ gamma_id ];
        if ( strcmp ( oet_type_tag[oet_type], "std" ) == 0 ) {
          loop_st_sign *= -1;
        }

        /**********************************************************
         * project loop matrices to spin structure
         **********************************************************/
        gamma_matrix_type gf;
        gamma_matrix_ukqcd_binary ( &gf, gamma_tmlqcd_to_binary[ gamma_id] );
        if ( g_verbose > 1 ) gamma_matrix_printf ( &gf, "gseq_ukqcd", stdout );

        if ( g_verbose > 0 ) fprintf ( stdout, "# [twop_analyse_wdisc] WARNING: using loop_transpose = %d %s %d\n", loop_transpose, __FILE__, __LINE__ );
        project_loop ( loops_proj_sink[isink_momentum][isink_gamma][0][0], gf.m, loops_matrix[0][0][0][0], num_conf * loop_nsample * T_global, loop_transpose );

        if ( write_data == 1 ) {
          sprintf ( filename, "loop.%s.%s.%s.gf%d.px%d_py%d_pz%d",
              flavor_tag[0], oet_type_tag[oet_type], loop_type_tag[loop_type], g_sink_gamma_id_list[isink_gamma], sink_momentum[0], sink_momentum[1], sink_momentum[2] );

          FILE * fs = fopen ( filename, "w" );
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isample = 0; isample < loop_nsample; isample++ ) {
              for ( int it = 0; it < T_global; it++ ) {
                fprintf ( fs, "%4d %3d %25.16e %25.16e %c %6d\n", isample, it, 
                    loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it  ] / double(loop_step),
                    loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it+1] / double(loop_step),
                    conf_src_list[iconf][0][0],
                    conf_src_list[iconf][0][1] );
              }
            }
          }

          fclose ( fs );
        }

      }  /* end of loop on sink gamma id */

    // }  /* end of loop on psign */
  }  /* end of loop on sink momenta */

  /**********************************************************
   * loop on momenta
   **********************************************************/
  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

    //for ( int ipsign = 0; ipsign < 2; ipsign++ ) {

    //  int const psign = 1 - 2 * ipsign;

    /*  int const sink_momentum[3] = {
          -psign * sink_momentum_list[isink_momentum][0],
          -psign * sink_momentum_list[isink_momentum][1],
          -psign * sink_momentum_list[isink_momentum][2] }; */

      int const sink_momentum[3] = {
          g_sink_momentum_list[isink_momentum][0],
          g_sink_momentum_list[isink_momentum][1],
          g_sink_momentum_list[isink_momentum][2] };

      /**********************************************************
       * loop on momenta
       **********************************************************/
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {

        int replica = -1;
        switch ( conf_src_list[iconf][0][0] )
        {
          case 'a': replica = 0; break;
          case 'b': replica = 1; break;
        }

        if ( loop_nev < 0 ) {
          sprintf ( filename, "stream_%c/%s/loop.%d.stoch.%s.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0], filename_prefix,
              conf_src_list[iconf][0][1], loop_type_tag[loop_type], sink_momentum[0], sink_momentum[1], sink_momentum[2] );
        } else {
          sprintf ( filename, "stream_%c/%s/loop.%.4d_r%d.stoch.%s.%s.nev%d.Nstoch%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0],
              filename_prefix, conf_src_list[iconf][0][1], replica, oet_type_tag[oet_type], loop_type_tag[loop_type], loop_nev,
              loop_nsample,
              sink_momentum[0], sink_momentum[1], sink_momentum[2] );

          /* sprintf ( filename, "%s/loop.%.4d_r%c.stoch.%s.%s.nev%d.Nstoch%d.PX%d_PY%d_PZ%d",
             filename_prefix, conf_src_list[iconf][0][1], conf_src_list[iconf][0][0],
             oet_type_tag[oet_type], loop_type_tag[loop_type], loop_nev, loop_nsample, sink_momentum[0], sink_momentum[1], sink_momentum[2] );*/
        }

        if ( g_verbose > 0 ) fprintf ( stdout, "# [twop_analyse_wdisc] reading loop data from file %s %s %d\n", filename, __FILE__, __LINE__ );
        FILE * ofs = fopen ( filename, "r" );
        if ( ofs == NULL ) {
          fprintf ( stderr, "[twop_analyse_wdisc] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT(113);
        }

        for ( int isample = 0; isample < loop_nsample; isample++ ) {
          int itmp[4];
          for ( int t = 0; t < T_global; t++ ) {
            for ( int mu = 0; mu < 4; mu++ ) {
            for ( int nu = 0; nu < 4; nu++ ) {
              if ( fscanf ( ofs, 
                    /* "%d %d %d %d %lf %lf\n", itmp, itmp+1, itmp+2, itmp+3, */
                    "%d %d %d %lf %lf\n", itmp, itmp+1, itmp+2,
                    loops_matrix[iconf][isample][t][mu]+2*nu, loops_matrix[iconf][isample][t][mu]+2*nu+1 ) != 5 ) {
                fprintf ( stderr, "[twop_analyse_wdisc] Error from fscanf %s %d\n", __FILE__, __LINE__ );
                EXIT(126);
              }
              /* show all data */
              if ( g_verbose > 5 ) {
                fprintf ( stdout, "loop_mat c %6d s %3d t %3d m %d nu %d l %25.16e %25.16e\n", Nconf, isample, t, mu, nu,
                    loops_matrix[iconf][isample][t][mu][2*nu], loops_matrix[iconf][isample][t][mu][2*nu+1] );
              }
            }}
          }
        }
        fclose ( ofs );

      }

      /**********************************************************
       * loop on source gamma
       **********************************************************/
      for ( int isource_gamma   = 0; isource_gamma   < g_source_gamma_id_number;   isource_gamma++ )
      {
     
        int const gamma_id = g_source_gamma_id_list[ isource_gamma ];

        int loop_st_sign = sigma_t [ gamma_id ];
        if ( strcmp ( oet_type_tag[oet_type], "std" ) == 0 ) {
          loop_st_sign *= -1;
        }

        /**********************************************************
         * project loop matrices to spin structure
         **********************************************************/
        gamma_matrix_type gf;
        gamma_matrix_ukqcd_binary ( &gf, gamma_tmlqcd_to_binary[ gamma_id] );
        if ( g_verbose > 1 ) gamma_matrix_printf ( &gf, "gseq_ukqcd", stdout );

        if ( g_verbose > 0 ) fprintf ( stdout, "# [twop_analyse_wdisc] WARNING: using loop_transpose = %d %s %d\n", loop_transpose, __FILE__, __LINE__ );
        project_loop ( loops_proj_source[isink_momentum][isource_gamma][0][0], gf.m, loops_matrix[0][0][0][0], num_conf * loop_nsample * T_global, loop_transpose );

        if ( write_data == 1 ) {
          sprintf ( filename, "loop.%s.%s.%s.gi%d.px%d_py%d_pz%d",
              flavor_tag[0], oet_type_tag[oet_type], loop_type_tag[loop_type], g_source_gamma_id_list[isource_gamma], sink_momentum[0], sink_momentum[1], sink_momentum[2] );

          FILE * fs = fopen ( filename, "w" );
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isample = 0; isample < loop_nsample; isample++ ) {
              for ( int it = 0; it < T_global; it++ ) {
                fprintf ( fs, "%4d %3d %25.16e %25.16e %c %6d\n", isample, it, 
                    loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it] / double(loop_step),
                    loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it+1] / double(loop_step),
                    conf_src_list[iconf][0][0],
                    conf_src_list[iconf][0][1] );
              }
            }
          }

          fclose ( fs );
        }
      }  /* end of loop on source gamma id */

    // }  /* end of loop on psign */

  }  /* end of loop on sink momentum */

  fini_5level_dtable ( &loops_matrix );

#endif  /* end of if _LOOP_ASCII */


  /****************************************************/
  /****************************************************/

#if _LOOP_H5
  double ****** loops_matrix = init_6level_dtable ( g_sink_momentum_number, num_conf, loop_nsample, T_global, 4, 8 );
  if ( loops_matrix == NULL ) {
    fprintf ( stderr, "[twop_analyse_wdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  double loop_oet_norm[2] = { 0., 0.};
  int loop_oet_perm[2] = {0,0};

  if ( strcmp ( loop_type_tag[loop_type], "dOp" ) == 0  ) 
  {
    loop_oet_norm[0] = -4 * g_kappa;
    loop_oet_norm[1] = -4 * g_kappa;
    loop_oet_perm[0] = 0;
    loop_oet_perm[1] = 1;

  } else if ( strcmp ( loop_type_tag[loop_type], "Scalar" ) == 0  ) 
  {
    loop_oet_norm[0] =  8 * g_mu * g_kappa * g_kappa;
    loop_oet_norm[1] = -8 * g_mu * g_kappa * g_kappa;
    loop_oet_perm[0] = 1;
    loop_oet_perm[1] = 0;
  }
  if ( g_verbose > 0 ) fprintf ( stdout, "# [twop_analyse_wdisc] oet_type %s loop_oet_norm = %25.6e  %26.16e loop_oet_perm = %d %d   %s %d\n",
        loop_type_tag[loop_type], loop_oet_norm[0], loop_oet_norm[1], 
        loop_oet_perm[0], loop_oet_perm[1], __FILE__, __LINE__ );


  for ( int iconf = 0; iconf < num_conf; iconf++ )
  {

    sprintf ( filename, "stream_%c/%s/%d/%s.%.4d.h5", conf_src_list[iconf][0][0], filename_prefix, conf_src_list[iconf][0][1], filename_prefix2, conf_src_list[iconf][0][1] );

    size_t ncdim = 0, *cdim = NULL;
    int *ibuffer = NULL;
    exitstatus = read_from_h5_file_varsize ( (void**)&ibuffer, filename, "Momenta_list_xyz",  "int", &ncdim, &cdim, io_proc );
    if ( exitstatus != 0 || ncdim != 2 ) {
      fprintf ( stderr, "[twop_analyse_wdisc] Error from read_from_h5_file_varsize for filename %s %s %d\n", filename, __FILE__, __LINE__ );
      EXIT(113);
    }

    int ** loop_momentum_list = init_2level_itable ( cdim[0], cdim[1] );
    memcpy ( loop_momentum_list[0], ibuffer, cdim[0]*cdim[1] * sizeof ( int ) );
    if ( ibuffer != NULL ) free ( ibuffer );
    ibuffer = NULL;
    int const loop_momentum_number = cdim[0];

    int * sink_momentum_id_list = init_1level_itable ( g_sink_momentum_number );


    for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
    {
      int loop_momentum_id = -1;
      for ( int i=0; i<loop_momentum_number; i++ )
      {
        if ( 
            ( g_sink_momentum_list[imom][0] ==  loop_momentum_list[i][0] ) &&
            ( g_sink_momentum_list[imom][1] ==  loop_momentum_list[i][1] ) &&
            ( g_sink_momentum_list[imom][2] ==  loop_momentum_list[i][2] ) ) {
          loop_momentum_id = i;
          break;
        }
      }
      if ( loop_momentum_id == -1 ) {
        fprintf ( stderr, "[twop_analyse_wdisc] Error, sink momentum %3d %3d %3d not in loop_momentum_list %s %d\n",
           g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], __FILE__, __LINE__ );
        EXIT(124);
      } else {
        fprintf ( stdout, "# [twop_analyse_wdisc] sink momentum %3d %3d %3d is loop_momentum_id %d   %s %d\n", 
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
            loop_momentum_id, __FILE__, __LINE__ );
      }

      sink_momentum_id_list[imom] = loop_momentum_id;

    }  /* end of loop on sink momenta */

    double **** buffer = init_4level_dtable ( T_global, loop_momentum_number, 4, 8 );
    if ( buffer == NULL ) {
      fprintf ( stderr, "[twop_analyse_wdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(113);
    }

    for ( int isample = 0; isample < loop_nsample; isample += loop_step ) 
    {

      char key[200];
      sprintf ( key, "conf_%.4d/nstoch_%.4d/%s/loop", conf_src_list[iconf][0][1], isample+1, loop_type_tag[loop_type] );
      if ( g_verbose > 2 ) fprintf ( stdout, "# [twop_analyse_wdisc] key = %s %s %d\n", key, __FILE__, __LINE__ );

      exitstatus = read_from_h5_file ( (void *)buffer[0][0][0], filename, key,  "double", io_proc );
      if ( exitstatus != 0 || ncdim != 2 ) {
        fprintf ( stderr, "[twop_analyse_wdisc] Error from read_from_h5_file for filename %s %s %s %d\n", filename, key, __FILE__, __LINE__ );
        EXIT(113);
      }

      for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
      {
#pragma omp parallel for
        for ( int it = 0; it < T_global; it++ ) {
          for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            /* transpose and normalize */
            loops_matrix[imom][iconf][isample][it][mu][2*nu  ] = buffer[it][sink_momentum_id_list[imom]][nu][2*mu+loop_oet_perm[0]] * loop_oet_norm[0];
            loops_matrix[imom][iconf][isample][it][mu][2*nu+1] = buffer[it][sink_momentum_id_list[imom]][nu][2*mu+loop_oet_perm[1]] * loop_oet_norm[1];
          }}
        }
      }
    }  /* end of loop on samples */

    fini_4level_dtable ( &buffer );
    fini_2level_itable ( &loop_momentum_list );

    fini_1level_itable ( &sink_momentum_id_list );

  }  /* end of loop on configs */

  /**********************************************************
   * loop on momenta
   **********************************************************/
  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

    /**********************************************************
     * loop on sink gamma
     **********************************************************/
    for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ )
    {
     
      int const gamma_id = g_sink_gamma_id_list[ isink_gamma ];

      int loop_st_sign = sigma_t [ gamma_id ];
      if ( strcmp ( oet_type_tag[oet_type], "std" ) == 0 ) {
        loop_st_sign *= -1;
      }

      /**********************************************************
       * project loop matrices to spin structure
       **********************************************************/
      gamma_matrix_type gf;
      /* gamma_matrix_ukqcd_binary ( &gf, gamma_tmlqcd_to_binary[ gamma_id] );
      if ( g_verbose > 1 ) gamma_matrix_printf ( &gf, "gseq_ukqcd", stdout ); */

      gamma_matrix_set ( &gf, gamma_id, 1.0 );
      if ( g_verbose > 1 ) gamma_matrix_printf ( &gf, "g_cvc", stdout );


      if ( g_verbose > 0 ) fprintf ( stdout, "# [twop_analyse_wdisc] WARNING: using loop_transpose = %d %s %d\n", loop_transpose, __FILE__, __LINE__ );
      project_loop ( loops_proj_sink[isink_momentum][isink_gamma][0][0], gf.m, loops_matrix[isink_momentum][0][0][0][0], num_conf * loop_nsample * T_global, loop_transpose );

      if ( write_data == 1 ) {
        sprintf ( filename, "loop.%s.%s.%s.gf%d.px%d_py%d_pz%d",
            flavor_tag[0], oet_type_tag[oet_type], loop_type_tag[loop_type], g_sink_gamma_id_list[isink_gamma],
            g_sink_momentum_list[isink_momentum][0], g_sink_momentum_list[isink_momentum][1], g_sink_momentum_list[isink_momentum][2] );

        FILE * fs = fopen ( filename, "w" );

        for ( int iconf = 0; iconf < num_conf; iconf++ ) 
        {
          for ( int isample = 0; isample < loop_nsample; isample++ ) 
          {
            for ( int it = 0; it < T_global; it++ ) 
            {
              fprintf ( fs, "%4d %3d %25.16e %25.16e %c %6d\n", isample, it, 
                  loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it  ] / double(loop_step),
                  loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it+1] / double(loop_step),
                  conf_src_list[iconf][0][0],
                  conf_src_list[iconf][0][1] );
            }
          }
        }

        fclose ( fs );
      }

    }  /* end of loop on sink gamma id */

    /**********************************************************
     * loop on source gamma
     **********************************************************/
    for ( int isource_gamma   = 0; isource_gamma   < g_source_gamma_id_number;   isource_gamma++ )
    {
     
      int const gamma_id = g_source_gamma_id_list[ isource_gamma ];

      int loop_st_sign = sigma_t [ gamma_id ];
      if ( strcmp ( oet_type_tag[oet_type], "std" ) == 0 ) {
        loop_st_sign *= -1;
      }

      /**********************************************************
       * project loop matrices to spin structure
       **********************************************************/
      gamma_matrix_type gf;
      /* gamma_matrix_ukqcd_binary ( &gf, gamma_tmlqcd_to_binary[ gamma_id] );
      if ( g_verbose > 1 ) gamma_matrix_printf ( &gf, "g_ukqcd", stdout ); */

      gamma_matrix_set ( &gf, gamma_id, 1.0 );
      if ( g_verbose > 1 ) gamma_matrix_printf ( &gf, "g_cvc", stdout );



      if ( g_verbose > 0 ) fprintf ( stdout, "# [twop_analyse_wdisc] WARNING: using loop_transpose = %d %s %d\n", loop_transpose, __FILE__, __LINE__ );
      project_loop ( loops_proj_source[isink_momentum][isource_gamma][0][0], gf.m, loops_matrix[isink_momentum][0][0][0][0], num_conf * loop_nsample * T_global, loop_transpose );

      if ( write_data == 1 ) {
          sprintf ( filename, "loop.%s.%s.%s.gi%d.px%d_py%d_pz%d",
              flavor_tag[0], oet_type_tag[oet_type], loop_type_tag[loop_type], g_source_gamma_id_list[isource_gamma], 
              g_sink_momentum_list[isink_momentum][0], g_sink_momentum_list[isink_momentum][1], g_sink_momentum_list[isink_momentum][2] );


        FILE * fs = fopen ( filename, "w" );
        for ( int iconf = 0; iconf < num_conf; iconf++ ) 
        {
          for ( int isample = 0; isample < loop_nsample; isample++ ) 
          {
            for ( int it = 0; it < T_global; it++ ) 
            {
              fprintf ( fs, "%4d %3d %25.16e %25.16e %c %6d\n", isample, it, 
                  loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it] / double(loop_step),
                  loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it+1] / double(loop_step),
                  conf_src_list[iconf][0][0],
                  conf_src_list[iconf][0][1] );
            }
          }
        }

        fclose ( fs );
      }
    }  /* end of loop on source gamma id */

  }  /* end of loop on sink momentum */

  fini_6level_dtable ( &loops_matrix );

#endif  /* end of _LOOP_H5 */



  /**********************************************************
   **********************************************************
   **
   ** possibly select real or imaginary part
   **
   **********************************************************
   **********************************************************/
  if ( loop_type_reim[0] >= 0 ) {
    for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {
        for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isample = 0; isample < loop_nsample; isample++ ) {
              for ( int it = 0; it < T_global; it++ ) {
                loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it+1-loop_type_reim[0]] = 0.;
              }
            }
          }
        }
    }
  }

  if ( loop_type_reim[1] >= 0 ) {
    for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

        for ( int isource_gamma   = 0; isource_gamma   < g_source_gamma_id_number;   isource_gamma++ ) {
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isample = 0; isample < loop_nsample; isample++ ) {
              for ( int it = 0; it < T_global; it++ ) {
                loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it+1-loop_type_reim[1]] = 0.;
              }
            }
          }
        }

    }
  }

  /**********************************************************
   **********************************************************
   **
   ** combine loops to 2pt function
   **
   **********************************************************
   **********************************************************/

  /**********************************************************
   * loop on gamma id at sink
   **********************************************************/
  for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) 
  {

    /**********************************************************
     * loop on gamma id at source
     **********************************************************/
    for ( int isource_gamma   = 0; isource_gamma   < g_source_gamma_id_number;   isource_gamma++ ) 
    {

      double *** corr = init_3level_dtable ( num_conf, g_sink_momentum_number, 2*T_global );
      if ( corr == NULL ) {
        fprintf(stderr, "[twop_analyse_wdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      double *** corr_sub = init_3level_dtable ( num_conf, g_sink_momentum_number, 2*T_global );
      if ( corr_sub == NULL ) {
        fprintf(stderr, "[twop_analyse_wdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      double *** corr_bias = init_3level_dtable ( num_conf, g_sink_momentum_number, 2*T_global );
      if ( corr_bias == NULL ) {
        fprintf(stderr, "[twop_analyse_wdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }


      double *** corr_vev = init_3level_dtable ( num_conf, 2, 2);
      if ( corr_vev == NULL ) {
        fprintf(stderr, "[twop_analyse_wdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }


      /**********************************************************
       * loop on sink momenta
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ )
      {
        /* the rest is all per config */

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
        for ( int iconf =0; iconf < num_conf; iconf++ ) {

          double * loops_src = init_1level_dtable ( 2*T_global );
          double * loops_snk = init_1level_dtable ( 2*T_global );

          /* sum over samples at source */
            for ( int it = 0; it < T_global; it++ ) {
              for ( int isample = 0; isample<loop_nsample; isample++ ) {
                loops_src[2*it  ] += loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it  ];
                loops_src[2*it+1] += loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it+1];
              }
            }

          /* sum over samples at sink */
            for ( int it = 0; it < T_global; it++ ) {
              for ( int isample = 0; isample<loop_nsample; isample++ ) {
                loops_snk[2*it  ] += loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it  ];
                loops_snk[2*it+1] += loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it+1];
              }
            }

          /* multiply-add all source-sink time combinations, average over parity */
          for ( int it0 = 0; it0 < T_global; it0++ ) {
          for ( int it1 = 0; it1 < T_global; it1++ ) {
            int const idt = ( it1 - it0 + T_global ) % T_global;
#if 0
            corr[iconf][isink_momentum][2*idt  ] += 0.5 * ( 
                    loops_snk[0][2*it1] * loops_src[0][2*it0] - loops_snk[0][2*it1+1] * loops_src[0][2*it0+1] 
                  + loops_snk[1][2*it1] * loops_src[1][2*it0] - loops_snk[1][2*it1+1] * loops_src[1][2*it0+1] 
                  );

            corr[iconf][isink_momentum][2*idt+1] += 0.5 * ( 
                    loops_snk[0][2*it1] * loops_src[0][2*it0+1] + loops_snk[0][2*it1+1] * loops_src[0][2*it0] 
                  + loops_snk[1][2*it1] * loops_src[1][2*it0+1] + loops_snk[1][2*it1+1] * loops_src[1][2*it0] 
                  );
#endif
            /* loop_snk x loop_src^dagger */
            corr[iconf][isink_momentum][2*idt  ] +=  loops_snk[2*it1] * loops_src[2*it0  ] + loops_snk[2*it1+1] * loops_src[2*it0+1];

            corr[iconf][isink_momentum][2*idt+1] += -loops_snk[2*it1] * loops_src[2*it0+1] + loops_snk[2*it1+1] * loops_src[2*it0  ] ;
          }}
      

          /* sum the "bias" over samples, diagonal in samples */
          for ( int isample =0; isample <  loop_nsample; isample++ ) {
            for ( int it0 = 0; it0 < T_global; it0++ ) {
              double const dsrc[2] = {
                loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it0],
                loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it0+1] };

              for ( int it1 = 0; it1 < T_global; it1++ ) {
                double const dsnk[2] = {
                  loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it1],
                  loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it1+1] };

                int const idt = ( it1 - it0 + T_global ) % T_global;

                corr_bias[iconf][isink_momentum][2*idt  ] +=  dsnk[0] * dsrc[0] + dsnk[1] * dsrc[1]; 

                corr_bias[iconf][isink_momentum][2*idt+1] += -dsnk[0] * dsrc[1] + dsnk[1] * dsrc[0];

              }
            }
          }

          double const norm_sub = 1. / ( (double)T_global * loop_nsample * ( loop_nsample - 1 ) * loop_step * loop_step * VOL3 ) * loop_norm[0] * loop_norm[1] ;
          for ( int idt = 0; idt < 2*T_global; idt++ ) {
            corr_sub[iconf][isink_momentum][idt] =  ( corr[iconf][isink_momentum][idt] - corr_bias[iconf][isink_momentum][idt] ) * norm_sub;
           }
          for ( int idt = 0; idt < 2*T_global; idt++ ) {
            corr_bias[iconf][isink_momentum][idt] *= norm_sub;
           }

          double const norm = 1. / ( (double)T_global * loop_nsample * loop_nsample * loop_step * loop_step * VOL3 ) * loop_norm[0] * loop_norm[1];
          for ( int idt = 0; idt < 2* T_global; idt++ ) {
            corr[iconf][isink_momentum][idt] *= norm;
          }
 
 
          /**********************************************************
           * vacuum expectation value at source and sink
           **********************************************************/

	  double vev[2][2] = { {0.,0.}, {0.,0.}};
          for ( int isample =0; isample <  loop_nsample; isample++ ) 
          {
	    for ( int it = 0; it < T_global; it++ ) {

              /* at sink */
              vev[0][0] += loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it  ];   
              vev[0][1] += loops_proj_sink[isink_momentum][isink_gamma][iconf][isample][2*it+1];

              /* at source */
              vev[1][0] += loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it  ];
              vev[1][1] += loops_proj_source[isink_momentum][isource_gamma][iconf][isample][2*it+1];
            }
          }
          double const norm_vev = 1. / (double)( T_global * loop_nsample * loop_step * g_sink_momentum_number );
          corr_vev[iconf][0][0] +=  vev[0][0] * norm_vev * loop_norm[0] / (double)VOL3;
          corr_vev[iconf][0][1] +=  vev[0][1] * norm_vev * loop_norm[0] / (double)VOL3;
          corr_vev[iconf][1][0] +=  vev[1][0] * norm_vev * loop_norm[1];
          corr_vev[iconf][1][1] +=  vev[1][1] * norm_vev * loop_norm[1];

          fini_1level_dtable ( &loops_src );
          fini_1level_dtable ( &loops_snk );

        }  /* end of loop over configurations */

      }  /* of loop on sink momenta */

      /**********************************************************
       * write to lfile
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ )
      {
        char filename[100];
        sprintf ( filename, "%s.%s-%s.%s.%s.gf%d.gi%d.px%d_py%d_pz%d.corr", g_outfile_prefix,
            flavor_tag[0], flavor_tag[1],
            oet_type_tag[oet_type], loop_type_tag[loop_type],
            g_sink_gamma_id_list[isink_gamma], g_source_gamma_id_list[isource_gamma], 
            g_sink_momentum_list[isink_momentum][0], g_sink_momentum_list[isink_momentum][1], g_sink_momentum_list[isink_momentum][2]);

        FILE * fs = fopen( filename, "w" );

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( fs, "%3d %25.16e %25.16e  %c %6d\n", it, corr_sub[iconf][isink_momentum][2*it+0], corr_sub[iconf][isink_momentum][2*it+1],
                conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
          }

          fprintf ( fs, "%3s %25.16e %c %6d\n", "vev", corr_vev[iconf][0][0], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
          fprintf ( fs, "%3s %25.16e %c %6d\n", "vev", corr_vev[iconf][0][1], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
          fprintf ( fs, "%3s %25.16e %c %6d\n", "vev", corr_vev[iconf][1][0], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
          fprintf ( fs, "%3s %25.16e %c %6d\n", "vev", corr_vev[iconf][1][1], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
        }

        fclose( fs );

      }  /* of loop on sink momenta */

      /**********************************************************/
      /**********************************************************/

      /**********************************************************
       * STATISTICAL ANALYSIS
       * for corr
       **********************************************************/

      for ( int ireim = 0; ireim <=1; ireim++ ) 
      {

        double ** data = init_2level_dtable ( num_conf, T_global );

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = 0.;
            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
              data[iconf][it] += corr[iconf][imom][2*it+ireim];
            }
            data[iconf][it] /= (double)g_sink_momentum_number;
          }
        }

        char obs_name[100];
        sprintf ( obs_name, "%s.%s-%s.%s.%s.gf%d.gi%d.px%d_py%d_pz%d.%s", g_outfile_prefix,
            flavor_tag[0], flavor_tag[1],
            oet_type_tag[oet_type], loop_type_tag[loop_type],
            g_sink_gamma_id_list[isink_gamma], g_source_gamma_id_list[isource_gamma], 
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[twop_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

#if 0
        if ( write_data == 1 )
        {
          sprintf ( filename, "%s.corr" , obs_name );
          FILE * fs = fopen( filename, "w" );

          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              fprintf ( fs, "%3d %25.16e %c %6d\n", it, data[iconf][it], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
              /* fprintf ( fs, "%3d %25.16e %6d\n", it, data[iconf][it], iconf * g_gauge_step ); */
            }
          }

          fclose( fs );
        }
#endif

#if 0
      /****************************************
       * STATISTICAL ANALYSIS of effective
       * mass from time-split acosh ratio
       ****************************************/
      for ( int itau = 1; itau < T_global/2; itau++ )
      {

        char obs_name2[200];
        sprintf( obs_name2, "%s.acoshratio.tau%d", obs_name, itau );

        int arg_first[3]  = { 0, 2*itau, itau };
        int arg_stride[3] = {1, 1, 1};

        exitstatus = apply_uwerr_func ( data[0], num_conf, T_global, T_global/2-itau, 3, arg_first, arg_stride, obs_name2, acosh_ratio, dacosh_ratio );

        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[twop_analyse_wdisc] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }
      }
#endif

       /****************************************
        * STATISTICAL ANALYSIS
        * for corr_sub
        ****************************************/

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = 0.;
            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
              data[iconf][it] += corr_sub[iconf][imom][2*it+ireim];
            }
            data[iconf][it] /= (double)g_sink_momentum_number;
          }
        }

        sprintf ( obs_name, "%s.sub.%s-%s.%s.%s.gf%d.gi%d.px%d_py%d_pz%d.%s", g_outfile_prefix,
            flavor_tag[0], flavor_tag[1],
            oet_type_tag[oet_type], loop_type_tag[loop_type],
            g_sink_gamma_id_list[isink_gamma], g_source_gamma_id_list[isource_gamma], 
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[twop_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

#if 0
        if ( write_data == 1 ) 
        {
          sprintf ( filename, "%s.corr" , obs_name );
          FILE * fs = fopen( filename, "w" );

          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              fprintf ( fs, "%3d %25.16e %c %6d\n", it, data[iconf][it], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
              /* fprintf ( fs, "%3d %25.16e %6d\n", it, data[iconf][it], iconf * g_gauge_step ); */
            }
          }

          fclose( fs );
        }
#endif

        /****************************************
         * STATISTICAL ANALYSIS
         * for corr_bias
         ****************************************/

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = 0.;
            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
              data[iconf][it] += corr_bias[iconf][imom][2*it+ireim];
            }
            data[iconf][it] /= (double)g_sink_momentum_number;
          }
        }

        sprintf ( obs_name, "%s.bias.%s-%s.%s.%s.gf%d.gi%d.px%d_py%d_pz%d.%s", g_outfile_prefix,
            flavor_tag[0], flavor_tag[1],
            oet_type_tag[oet_type], loop_type_tag[loop_type],
            g_sink_gamma_id_list[isink_gamma], g_source_gamma_id_list[isource_gamma], 
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[twop_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

#if 0
        if ( write_data == 1 )
        {
          sprintf ( filename, "%s.corr" , obs_name );
          FILE * fs = fopen( filename, "w" );

          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              fprintf ( fs, "%3d %25.16e %c %6d\n", it, data[iconf][it], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
              /* fprintf ( fs, "%3d %25.16e %6d\n", it, data[iconf][it], iconf * g_gauge_step ); */
            }
          }

          fclose( fs );
        }
#endif

        fini_2level_dtable ( &data );

      }  /* end of loop on reim */


      /**********************************************************
       * disc.sub with vev subtraction
       **********************************************************/
      for ( int ireim = 0; ireim <=1; ireim++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global + 4 );

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = 0.;
            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
              data[iconf][it] += corr[iconf][imom][2*it+ireim];
            }
            data[iconf][it] /= (double)g_sink_momentum_number;
          }
          data[iconf][T_global  ] = corr_vev[iconf][0][0];
          data[iconf][T_global+1] = corr_vev[iconf][0][1];
          data[iconf][T_global+2] = corr_vev[iconf][1][0];
          data[iconf][T_global+3] = corr_vev[iconf][1][1];
        }

        char obs_name[100];
        sprintf ( obs_name, "%s.sub.vev.%s-%s.%s.%s.gf%d.gi%d.px%d_py%d_pz%d.%s", g_outfile_prefix,
            flavor_tag[0], flavor_tag[1],
            oet_type_tag[oet_type], loop_type_tag[loop_type],
            g_sink_gamma_id_list[isink_gamma], g_source_gamma_id_list[isource_gamma],
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

	int narg = 5;
	int arg_first[5]  = { 0, T_global, T_global+1, T_global+2, T_global+3 };
	int arg_stride[5] = { 1, 0,        0,          0,          0          };

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_func ( data[0], num_conf, T_global+4, T_global, narg, arg_first, arg_stride, obs_name, a_mi_w_ti_z_dag_re , da_mi_w_ti_z_dag_re );

        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[twop_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

#if 0
        if ( write_data == 1 )
        {
          sprintf ( filename, "%s.corr" , obs_name );
          FILE * fs = fopen( filename, "w" );

          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              fprintf ( fs, "%3d %25.16e %c %6d\n", it, data[iconf][it], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
              /* fprintf ( fs, "%3d %25.16e %25.16e %6d\n", it, data[iconf][it], data[iconf][T_global], iconf * g_gauge_step ); */
            }
            fprintf ( fs, "%3s %25.16e %c %6d\n", "vev", data[iconf][T_global], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
            fprintf ( fs, "%3s %25.16e %c %6d\n", "vev", data[iconf][T_global+1], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
            fprintf ( fs, "%3s %25.16e %c %6d\n", "vev", data[iconf][T_global+2], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
            fprintf ( fs, "%3s %25.16e %c %6d\n", "vev", data[iconf][T_global+3], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );
          }

          fclose( fs );
        }
#endif

        fini_2level_dtable ( &data );
      }

      /**********************************************************/
      /**********************************************************/

      /**********************************************************
       * free
       **********************************************************/
      fini_3level_dtable ( &corr );
      fini_3level_dtable ( &corr_sub );
      fini_3level_dtable ( &corr_bias );
      fini_3level_dtable ( &corr_vev );
    }  /* end of loop on source gamma id */
  }  /* end of loop on sink gamma id */

  fini_5level_dtable ( &loops_proj_sink );
  fini_5level_dtable ( &loops_proj_source );

  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/

  fini_3level_itable ( &conf_src_list );
#if 0
  fini_2level_itable ( &sink_momentum_list );
#endif

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
    fprintf(stdout, "# [twop_analyse_wdisc] %s# [twop_analyse_wdisc] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [twop_analyse_wdisc] %s# [twop_analyse_wdisc] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
