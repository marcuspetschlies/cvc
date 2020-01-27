/****************************************************
 * nn_2pt_analyse 
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

#define _LOOP_ANALYSIS

#define _RAT_METHOD
#undef _FHT_METHOD_ALLT
#define _FHT_METHOD_ACCUM

#define _TWOP_STATS

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse cpff fht correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}


/**********************************************************
 *
 **********************************************************/
int main(int argc, char **argv) {
  
  /* int const gamma_id_to_bin[16] = { 8, 1, 2, 4, 0, 15, 7, 14, 13, 11, 9, 10, 12, 3, 5, 6 }; */

  char const reim_str[2][3] = { "re", "im" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[600];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  int twop_fold_propagator = 0;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:E:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [nn_2pt_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [nn_2pt_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      twop_fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [nn_2pt_analyse] twop fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [nn_2pt_analyse] ensemble_name set to %s\n", ensemble_name );
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
  if(filename_set==0) strcpy(filename, "cvc.input");
  /* fprintf(stdout, "# [nn_2pt_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [nn_2pt_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
  set_omp_number_threads ();

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[nn_2pt_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[nn_2pt_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [nn_2pt_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[nn_2pt_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[nn_2pt_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [nn_2pt_analyse] comment %s\n", line );
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
   * gamma matrices
   **********************************************************/
  init_gamma_matrix ();
 
  gamma_matrix_type gamma_id, gamma_t, gamma_proj[2];

  gamma_matrix_ukqcd_binary ( &(gamma_id), 0 ); /* unit matrix */
  gamma_matrix_ukqcd_binary ( &(gamma_t ), 8 ); /* gamma_t */

  gamma_matrix_init ( &(gamma_proj[0]) );
  gamma_matrix_init ( &(gamma_proj[1]) );

  gamma_matrix_eq_gamma_matrix_pl_gamma_matrix_ti_re ( &(gamma_proj[0]), &gamma_id, &gamma_t,  1. );
  gamma_matrix_eq_gamma_matrix_pl_gamma_matrix_ti_re ( &(gamma_proj[1]), &gamma_id, &gamma_t, -1. );


  if ( g_verbose > 2 ) {
    gamma_matrix_printf ( &(gamma_proj[0]), "gamma_proj+", stdout );
    gamma_matrix_printf ( &(gamma_proj[1]), "gamma_proj-", stdout );
  }

  /**********************************************************
   * output filename and file stream
   **********************************************************/
  sprintf( filename, "%s/twop.nucleon_zeromom.SS.dat", filename_prefix );
  ofs = fopen ( filename, "w" );

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
    double _Complex ***** twop = init_5level_ztable ( num_conf, num_src_per_conf, T_global, 4, 4 );
    if( twop == NULL ) {
      fprintf ( stderr, "[nn_2pt_analyse] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT (24);
    }


  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    Nconf = conf_src_list[iconf][0][1];

    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
     
      sprintf( filename, "%s/%.4d/twop.%.4d.nucleon_zeromom.SS.%.2d.%.2d.%.2d.%.2d.dat",
          filename_prefix, Nconf, Nconf,
          conf_src_list[iconf][isrc][3],
          conf_src_list[iconf][isrc][4],
          conf_src_list[iconf][isrc][5],
          conf_src_list[iconf][isrc][2] );
     
      FILE * dfs = fopen ( filename, "r" );
      if( dfs == NULL ) {
        fprintf ( stderr, "[nn_2pt_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
        EXIT (24);
      } else {
        if ( g_verbose > 2 ) fprintf ( stdout, "# [nn_2pt_analyse] reading data from file %s %s %d\n", filename, __FILE__, __LINE__ );
      }
      fflush ( stdout );
      fflush ( stderr );

      int itmp[7];
      double dtmp[4];

      for( int it =0; it  < T_global; it++ ) {

        for ( int ia = 0; ia < 4; ia++ ) {
        for ( int ib = 0; ib < 4; ib++ ) {
          
          fscanf( dfs, " %d %d %d %d %d %d %d %lf %lf %lf %lf\n",
               itmp, itmp+1, itmp+2, itmp+3, itmp+4, itmp+5, itmp+6, 
               dtmp, dtmp+1, dtmp+2, dtmp+3 );

          twop[iconf][isrc][it][ia][ib] = ( ( dtmp[0] + dtmp[2] ) + ( dtmp[1] + dtmp[3] ) * I ) * 0.5;
        }}
      }

      fclose ( dfs );
        
      double _Complex ** buffer = init_2level_ztable ( 2, T_global );

      for ( int ip = 0; ip < 2; ip++ ) {
#pragma omp parallel for
        for( int it =0; it  < T_global; it++ ) {
          buffer[ip][it] = 0.;

          for ( int ia = 0; ia < 4; ia++ ) {
          for ( int ib = 0; ib < 4; ib++ ) {
            buffer[ip][it] += gamma_proj[ip].m[ia][ib] * twop[iconf][isrc][it][ib][ia];
          }}
          buffer[ip][it] *= 0.25;
        }
      }

      for( int it =0; it  < T_global; it++ ) {
        const int itt = ( T_global - it ) % T_global;
        fprintf ( ofs, "%25.16e %25.16e    %25.16e %25.16e\n", 
             creal( buffer[0][it] ),   cimag( buffer[0][it] ),
            -creal( buffer[1][itt] ), -cimag( buffer[1][itt] ) );
      }

      fini_2level_ztable (&buffer );
    }  /* end of loop on sources per config */
  }  /* end of loop on configs */

  fini_5level_ztable ( &twop );

  fclose ( ofs );

  /**********************************************************
   * free and finalize
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
    fprintf(stdout, "# [nn_2pt_analyse] %s# [nn_2pt_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [nn_2pt_analyse] %s# [nn_2pt_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);
}
