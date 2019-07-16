/****************************************************
 * avxn_analyse 
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

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse cpff fht correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  /* int const gamma_id_to_bin[16] = { 8, 1, 2, 4, 0, 15, 7, 14, 13, 11, 9, 10, 12, 3, 5, 6 }; */

  char const reim_str[2][3] = { "re", "im" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  /* int use_disc = 0;
  int use_conn = 1; */
  int twop_fold_propagator = 0;
  int twop_use_reim = 0;
  int loop_use_reim = 0;

  char loop_type[10] = "LpsDw";

  char key[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:R:r:E:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      twop_fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] twop fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'R':
      twop_use_reim = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] twop use_reim set to %d\n", twop_use_reim );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [avxn_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'r':
      loop_use_reim = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] loop use_reim set to %d\n", loop_use_reim );
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
  /* fprintf(stdout, "# [avxn_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [avxn_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [avxn_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [avxn_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[avxn_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[avxn_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[avxn_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [avxn_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  /* sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf); */
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[avxn_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [avxn_analyse] comment %s\n", line );
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
   * gamma matrices
   **********************************************************/
  init_gamma_matrix ();
 
  gamma_matrix_type gamma_mu[4];

  gamma_matrix_ukqcd_binary ( &(gamma_mu[0]), 1 ); /* gamma_x */
  gamma_matrix_ukqcd_binary ( &(gamma_mu[1]), 2 ); /* gamma_y */
  gamma_matrix_ukqcd_binary ( &(gamma_mu[2]), 4 ); /* gamma_z */
  gamma_matrix_ukqcd_binary ( &(gamma_mu[3]), 8 ); /* gamma_t */

  if ( g_verbose > 2 ) {
    gamma_matrix_printf ( &(gamma_mu[0]), "gamma_x", stdout );
    gamma_matrix_printf ( &(gamma_mu[1]), "gamma_y", stdout );
    gamma_matrix_printf ( &(gamma_mu[2]), "gamma_z", stdout );
    gamma_matrix_printf ( &(gamma_mu[3]), "gamma_t", stdout );
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
  double ****** twop = NULL;

  twop = init_6level_dtable ( g_sink_momentum_number, num_conf, num_src_per_conf, 2, T_global, 2 );
  if( twop == NULL ) {
    fprintf ( stderr, "[avxn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

    int const sink_momentum[3] = {
        g_sink_momentum_list[isink_momentum][0],
        g_sink_momentum_list[isink_momentum][1],
        g_sink_momentum_list[isink_momentum][2] };
  
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      Nconf = conf_src_list[iconf][0][0];
     
      for( int imeson = 0; imeson < 2; imeson++ ) {

        sprintf( filename, "twop/twop.%.4d.pseudoscalar.%d.PX%d_PY%d_PZ%d",  Nconf, imeson+1,
            ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][0],
            ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][1],
            ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][2] );
     
        FILE * dfs = fopen ( filename, "r" );
        if( dfs == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT (24);
        } else {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_analyse] reading data from file %s filename \n", filename );
        }

        for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
          char line[400];

          for ( int it = -1; it < T_global; it++ ) {
            if ( fgets ( line, 100, ofs) == NULL ) {
              fprintf ( stderr, "[avxn_analyse] Error from fgets, expecting line input for conf %3d src %3d filename %s %s %d\n", 
                  iconf, isrc, filename, __FILE__, __LINE__ );
              EXIT (26);
            } 
          
            if ( line[0] == '#' &&  it == -1 ) {
              fprintf ( stdout, "# [avxn_analyse] reading key %s\n", line );
              continue;
            } /* else {
              fprintf ( stderr, "[avxn_analyse] Error in layout of file %s %s %d\n", filename, __FILE__, __LINE__ );
              EXIT(27);
            }
              */
            sscanf ( line, "%lf %lf\n", twop[isink_momentum][iconf][isrc][imeson][it], twop[isink_momentum][iconf][isrc][imeson][it]+1 );
         
            if ( g_verbose > 4 ) fprintf ( stdout, "%d %25.16e %25.16e\n" , it, twop[isink_momentum][iconf][isrc][imeson][it][0],
                twop[isink_momentum][iconf][isrc][imeson][it][1] );
          }

        }
        fclose ( dfs );
      }
    }
  }  /* end of loop on sink momenta */
#if 0
#endif  /* of if 0 */

  /**********************************************************
   * average 2-pt over momentum orbit
   **********************************************************/

  double **** twop_orbit = init_4level_dtable ( num_conf, num_src_per_conf, T_global, 2 );
  if( twop_orbit == NULL ) {
    fprintf ( stderr, "[avxn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

  double const twop_use_re = ( twop_use_reim == 1 || twop_use_reim == 3 ) ? 1. : 0.;
  double const twop_use_im = ( twop_use_reim == 2 || twop_use_reim == 3 ) ? 1. : 0.;

#pragma omp parallel for
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      /* averaging starts here */
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
        double const source_phase = -2. * M_PI * ( 
            g_sink_momentum_list[imom][0] * conf_src_list[iconf][isrc][2] + 
            g_sink_momentum_list[imom][1] * conf_src_list[iconf][isrc][3] + 
            g_sink_momentum_list[imom][2] * conf_src_list[iconf][isrc][4] );

        double const ephase[2] = { cos ( source_phase ), sin ( source_phase ) };

        for ( int it = 0; it < T_global; it++ ) {
          double const a[2] = { twop[imom][iconf][isrc][0][it][0] , twop[imom][iconf][isrc][0][it][1] };

          double const b[2] = { twop[imom][iconf][isrc][1][it][0] , twop[imom][iconf][isrc][1][it][1] };
             
          double const cre = ( a[0] + b[0] ) * ephase[0] - ( a[1] - b[1] ) * ephase[1];
          double const cim = ( a[1] + b[1] ) * ephase[0] + ( a[0] - b[0] ) * ephase[1];

          twop_orbit[iconf][isrc][it][0] += 0.5 * cre * twop_use_re;
          twop_orbit[iconf][isrc][it][1] += 0.5 * cim * twop_use_im;

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
   * 
   * STATISTICAL ANALYSIS
   * 
   **********************************************************/

  for ( int ireim = 0; ireim < 2; ireim++ ) {

    double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T_global );
    if ( data == NULL ) {
      fprintf ( stderr, "[avxn_analyse] Error from init_3level_dtable %s %d\n",  __FILE__, __LINE__ );
      EXIT(1);
    }

    /* fill data array */
    if ( twop_fold_propagator) {
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
          for ( int it = 0; it <= T_global/2; it++ ) {
            int const itt = ( T_global - it ) % T_global;
              data[iconf][isrc][it ] = 0.5 * ( twop_orbit[iconf][isrc][it][ireim] + twop_orbit[iconf][isrc][itt][ireim] );
              data[iconf][isrc][itt] = data[iconf][isrc][it];
         } 
       }
      }
    } else {
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][isrc][it] = twop_orbit[iconf][isrc][it][ireim];
          }
        }
      }
    }

    char obs_name[100];
    sprintf( obs_name, "twop.pseudoscalar.orbit.PX%d_PY%d_PZ%d.%s",
          g_sink_momentum_list[0][0],
          g_sink_momentum_list[0][1],
          g_sink_momentum_list[0][2], reim_str[ireim] );

    /* apply UWerr analysis */
    exitstatus = apply_uwerr_real ( data[0][0], num_conf*num_src_per_conf, T_global, 0, 1, obs_name );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }

    fini_3level_dtable ( &data );
  }  /* end of loop on reim */


  /**********************************************************
   *
   * read loop data
   *
   **********************************************************/
  double ****** loop = NULL;
  double ****** loop_sub = NULL;

  loop = init_6level_dtable ( g_insertion_momentum_number, num_conf, 4, 4, T_global, 2 );
  if ( loop == NULL ) {
    fprintf ( stdout, "[avxn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(25);
  }

  for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int idir = 0; idir < 4; idir++ ) {

        double _Complex *** zloop_buffer = init_3level_ztable ( T_global, 4, 4 );

        sprintf ( filename, "loops/loop.%.4d.stoch.%s.nev0.Nstoch1.mu%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0],
            loop_type,
            idir,
            g_insertion_momentum_list[imom][0],
            g_insertion_momentum_list[imom][1],
            g_insertion_momentum_list[imom][2] );

        FILE * dfs = fopen ( filename, "r" );
        if( dfs == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT (24);
        } else {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_analyse] reading data from file %s filename \n", filename );
        }

        for ( int it = 0; it < T_global; it++ ) {
          int itmp[3];
          double dtmp[2];
          for ( int ia = 0; ia < 4; ia++ ) {
          for ( int ib = 0; ib < 4; ib++ ) {
            fscanf ( dfs, "%d %d %d %lf %lf\n", itmp, itmp+1, itmp+2, dtmp, dtmp+1 );
            zloop_buffer[it][ia][ib] = dtmp[0] + dtmp[1] * I;
         
            if ( g_verbose > 4 ) fprintf (stdout,"loop %3d %3d %3d  %25.16e %25.16e\n", 
                itmp[0], itmp[1], itmp[2], creal( zloop_buffer[it][ia][ib]), cimag( zloop_buffer[it][ia][ib]) );
          }}
        }
        fclose ( dfs );

#pragma omp parallel for
        for ( int imu = 0; imu < 4; imu++ ) {
          for ( int it = 0; it < T_global; it++ ) {

            double _Complex ztmp = 0.;
            for ( int ia = 0; ia < 4; ia++ ) {
            for ( int ib = 0; ib < 4; ib++ ) {
              ztmp += zloop_buffer[it][ia][ib] * gamma_mu[imu].m[ib][ia];
            }}
            loop[imom][iconf][imu][idir][it][0] = 0.5 * creal ( ztmp );
            loop[imom][iconf][imu][idir][it][1] = 0.5 * cimag ( ztmp );
          }
        }  /* end of loop on mu */

        fini_3level_ztable ( &zloop_buffer );
      }  /* end of loop on directions */
    }  /* end of loop on configs */
  }  /* end of loop on insertion momenta */

#if 0
#endif  /* of if 0 */

  /**********************************************************
   *
   * build trace-subtracted tensor
   *
   **********************************************************/
  loop_sub = init_6level_dtable ( g_insertion_momentum_number, num_conf, 4, 4, T_global, 2 );
  if ( loop_sub == NULL ) {
    fprintf ( stdout, "[avxn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(25);
  }

  for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {

    for ( int imu = 0; imu < 4; imu++ ) {
      for ( int idir = 0; idir < 4; idir++ ) {

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            loop_sub[imom][iconf][imu][idir][it][0] = 0.5 * ( loop[imom][iconf][imu][idir][it][0] + loop[imom][iconf][idir][imu][it][0] );
            loop_sub[imom][iconf][imu][idir][it][1] = 0.5 * ( loop[imom][iconf][imu][idir][it][1] + loop[imom][iconf][idir][imu][it][1] );
            /* subtract trace for diagonal */
            if ( imu == idir ) {
              loop_sub[imom][iconf][imu][idir][it][0] -= 0.25 * ( 
                   loop[imom][iconf][0][0][it][0]
                 + loop[imom][iconf][1][1][it][0]
                 + loop[imom][iconf][2][2][it][0]
                 + loop[imom][iconf][3][3][it][0] );

              loop_sub[imom][iconf][imu][idir][it][1] -= 0.25 * ( 
                   loop[imom][iconf][0][0][it][1]
                 + loop[imom][iconf][1][1][it][1]
                 + loop[imom][iconf][2][2][it][1]
                 + loop[imom][iconf][3][3][it][1] );
            }
          }
        }
      }
    }
  }  /* end of loop on insertion momentum */

  fini_6level_dtable ( &loop );

#if 0
  /**********************************************************
   * loop data
   **********************************************************/
  for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {

    for ( int ireim = 0; ireim < 2; ireim++ ) {
      
      for ( int imu = 0; imu < 4; imu++ ) {
      for ( int idir = 0; idir < 4; idir++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = loop_sub[imom][iconf][imu][idir][it][ireim];
          }
        }

        char obs_name[100];
        sprintf ( obs_name, "loop_sub.stoch.%s.nev0.Nstoch1.g%d_D%d.PX%d_PY%d_PZ%d.%s",
            loop_type, imu, idir,
            g_insertion_momentum_list[imom][0],
            g_insertion_momentum_list[imom][1],
            g_insertion_momentum_list[imom][2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        fini_2level_dtable ( &data );
      }}
    }  /* end of loop on re / im */
  }  /* end of loop momenta */

#endif  /* of if 0 */

  /**********************************************************
   *
   * STATISTICAL ANALYSIS for products and ratios
   *
   **********************************************************/

  int const parity_sign_tensor[4] = { 1, -1, -1, -1 };

  double const loop_use_re = ( loop_use_reim == 1 || loop_use_reim == 3 ) ? 1. : 0.;
  double const loop_use_im = ( loop_use_reim == 2 || loop_use_reim == 3 ) ? 1. : 0.;

  for ( int imu = 0; imu < 4; imu++ ) {
    for ( int idir = imu; idir < 4; idir++ ) {
      for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {
 
        double **** threep_orbit = init_4level_dtable ( num_conf, num_src_per_conf, T_global, 2 ) ;
        if ( threep_orbit == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

            int const tins = ( g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][1] ) % T_global;
            if ( g_verbose > 4 ) fprintf ( stdout, "# [avxn_analyse] t_src %3d   dt %3d   t_ins %3d\n", conf_src_list[iconf][isrc][1],
                g_sequential_source_timeslice_list[idt], tins );

            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

              int const parity_sign_tensor[4] = { 1, 
                2 * ( g_sink_momentum_list[imom][0] >= 0 ) - 1,
                2 * ( g_sink_momentum_list[imom][1] >= 0 ) - 1,
                2 * ( g_sink_momentum_list[imom][2] >= 0 ) - 1 };
              double const parity_sign_value = (double) ( parity_sign_tensor[imu] * parity_sign_tensor[idir] );

              double const source_phase = -2. * M_PI * (
                  g_sink_momentum_list[imom][0] * conf_src_list[iconf][isrc][2] +
                  g_sink_momentum_list[imom][1] * conf_src_list[iconf][isrc][3] +
                  g_sink_momentum_list[imom][2] * conf_src_list[iconf][isrc][4] );

              double const ephase[2] = { cos ( source_phase ), sin ( source_phase ) };

              for ( int it = 0; it < T_global; it++ ) {

                double const a[2] = { twop[imom][iconf][isrc][0][it][0] , twop[imom][iconf][isrc][0][it][1] };
                double const b[2] = { twop[imom][iconf][isrc][1][it][0] , twop[imom][iconf][isrc][1][it][1] };
          
                double const c[2] = { 
                  loop_sub[0][iconf][imu][idir][tins][0] * loop_use_re,
                  loop_sub[0][iconf][imu][idir][tins][1] * loop_use_im };

                /* twop x source phase */
                double const d[2] = { ( ( a[0] + b[0] ) * ephase[0] - ( a[1] - b[1] ) * ephase[1] ) * 0.5 * twop_use_re ,
                                      ( ( a[1] + b[1] ) * ephase[0] + ( a[0] - b[0] ) * ephase[1] ) * 0.5 * twop_use_im };

                /* ( twop x source phase ) x loop */
                threep_orbit[iconf][isrc][it][0] += ( d[0] * c[0] - d[1] * c[1] ) * parity_sign_value;
                threep_orbit[iconf][isrc][it][1] += ( d[1] * c[0] + d[0] * c[1] ) * parity_sign_value;

              }  /* end of loop on it */

              /* normalize */
              double const norm = 1. / g_sink_momentum_number;
              for ( int it = 0; it < 2 * T_global; it++ ) {
                threep_orbit[iconf][isrc][0][it] *= norm;
              }

            }  /* end of loop on imom */
          }
        }

        /**********************************************************
         *
         * STATISTICAL ANALYSIS for products and ratios
         *
         **********************************************************/
        for ( int ireim = 0; ireim < 2; ireim++ ) {
          double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T_global );
          if ( data == NULL ) {
            fprintf ( stderr, "[avxn_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }

#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][isrc][it] = threep_orbit[iconf][isrc][it][ireim];
          }}}


          char obs_name[100];
          sprintf ( obs_name, "threep_orbit.g%d_D%d.dt%d.PX%d_PY%d_PZ%d.%s",
            imu, idir, g_sequential_source_timeslice_list[idt],
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2], reim_str[ireim] );

          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0][0], num_conf*num_src_per_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

          fini_3level_dtable ( &data );
        }

        for ( int ireim = 0; ireim < 1; ireim++ ) {
          double *** data = init_3level_dtable ( num_conf, num_src_per_conf, 2 * T_global );
          if ( data == NULL ) {
            fprintf ( stderr, "[avxn_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }

          /* use folding by default */
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
          for ( int it = 0; it <= T_global/2; it++ ) {
            int const iit = ( T_global - it ) % T_global;
            /* BETTER make this a real complex calculation ??? */
            data[iconf][isrc][it ] = 0.5 * ( threep_orbit[iconf][isrc][it][ireim] + threep_orbit[iconf][isrc][iit][ireim] );
            data[iconf][isrc][iit] = data[iconf][isrc][it];


            data[iconf][isrc][T_global + it ] = 0.5 * ( twop_orbit[iconf][isrc][it][ireim] + twop_orbit[iconf][isrc][iit][ireim] );
            data[iconf][isrc][T_global + iit] = data[iconf][isrc][T_global + it];

          }}}

          char obs_name[100];
          sprintf ( obs_name, "ratio_orbit.g%d_D%d.dt%d.PX%d_PY%d_PZ%d.%s",
            imu, idir, g_sequential_source_timeslice_list[idt],
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2], reim_str[ireim] );

          /* apply UWerr analysis */
          int narg = 2;
          int arg_first[2] = { 0, T_global };
          int arg_stride[2] = { 1, 1 };

          exitstatus = apply_uwerr_func ( data[0][0], num_conf*num_src_per_conf, 2*T_global, T_global, narg, arg_first, arg_stride, obs_name, ratio_1_1, dratio_1_1 );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }

          fini_3level_dtable ( &data );
        }



        fini_4level_dtable ( &threep_orbit );
      }
    }
  }
#if 0
#endif  /* of if 0 */


  /**********************************************************/
  /**********************************************************/

  fini_6level_dtable ( &loop_sub );
  fini_6level_dtable ( &twop );
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

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [avxn_analyse] %s# [avxn_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [avxn_analyse] %s# [avxn_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);
}
