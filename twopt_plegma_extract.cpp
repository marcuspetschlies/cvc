/****************************************************
 * twopt_plegma_extract
 *
 * list of mesons:
 * pseudoscalar, scalar, g5g1, g5g2, g5g3, g5g4, g1, g2, g3, g4
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

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_i.h"
#include "contract_loop.h"
#include "ranlxd.h"

using namespace cvc;


void usage() {
  fprintf(stdout, "Code to extract twop data\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "twop";
  const char flavor_tag[2][3] = { "uu", "dd" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[400];
  /* char twopt_type[20] = "NA"; */
  char ensemble_name[200];
  int num_src_per_conf = 0;
  int num_conf = 0;
  int nmeson = 10;


  struct timeval ta, tb;
  long unsigned int seconds, useconds;

  char output_filename[400];

  char data_tag[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:E:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [twopt_plegma_extract] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [twopt_plegma_extract] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [twopt_plegma_extract] ensemble_name set to %s\n", ensemble_name );
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
  if(filename_set==0) sprintf ( filename, "%s.input", outfile_prefix );
  /* fprintf(stdout, "# [twopt_plegma_extract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [twopt_plegma_extract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[twopt_plegma_extract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***************************************************************************
   * some additional xchange operations
   ***************************************************************************/
  mpi_init_xchange_contraction(2);

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[twopt_plegma_extract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [twopt_plegma_extract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


    /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  /* sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf); */
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[twopt_plegma_extract] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[twopt_plegma_extract] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [twopt_plegma_extract] comment %s\n", line );
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

  /***************************************************************************
   * allocate memory for contractions
   *
   * match g_sink_momentum == -sink_momentum
   ***************************************************************************/
  int * momentum_matchid = init_1level_itable ( g_source_momentum_number );
  for ( int i = 0; i < g_source_momentum_number; i++ ) {
    momentum_matchid[i] = -1;
    for ( int k = 0; k < g_sink_momentum_number; k++ ) {
      if ( ( g_source_momentum_list[i][0] == -g_sink_momentum_list[k][0] ) &&
           ( g_source_momentum_list[i][1] == -g_sink_momentum_list[k][1] ) &&
           ( g_source_momentum_list[i][2] == -g_sink_momentum_list[k][2] ) ) {
        momentum_matchid[i] = k;
        break;
      }
    }
    if ( momentum_matchid[i] == -1 ) {
      fprintf ( stderr, "[twopt_plegma_extract] Error, no match found for g_sink_momentum %d %s %d\n", i, __FILE__, __LINE__ );
      EXIT(1);
    } else {
      if ( g_verbose > 4 ) fprintf ( stdout, "# [twopt_plegma_extract] momentum %3d p %3d, %3d %3d  machid %3d\n", i,
          g_source_momentum_list[i][0], g_source_momentum_list[i][1], g_source_momentum_list[i][2], momentum_matchid[i] ); 
    }
  }

  /***************************************************************************
   *
   ***************************************************************************/
  double **** twopt = init_4level_dtable ( T_global, g_sink_momentum_number, nmeson, 2 );
  if ( twopt == NULL ) {
    fprintf(stderr, "[twopt_plegma_extract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }
  double * buffer = init_1level_dtable ( 2 * T_global );
  if ( buffer == NULL ) {
    fprintf(stderr, "[twopt_plegma_extract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  /***************************************************************************
   * loop on source coordinates
   ***************************************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ )
  {

    /***************************************************************************
     * data filename for first source coords
     ***************************************************************************/
    sprintf ( filename, "%s/%.4d_r%c_2.h5", filename_prefix, conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] );

    if ( io_proc == 2 && g_verbose > 2 ) fprintf ( stdout, "# [twopt_plegma_extract] twop filename = %s\n", filename );

    /***************************************************************************
     * loop on source coordinates
     ***************************************************************************/
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ )
    {

      for ( int twop_meson_id = 0; twop_meson_id <= 1; twop_meson_id++ )
      {

        /* h5ls 0016_r1.h5/corr/0016_r1/sx74sy07sz14st140/du */
 
        sprintf ( data_tag, "/%.4d_r%c/sx%.2dsy%.2dsz%.2dst%.2d/%s" , conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][0], 
            conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4],  conf_src_list[iconf][isrc][5], conf_src_list[iconf][isrc][2], 
            flavor_tag[ twop_meson_id ]);

        if ( g_verbose > 2 ) fprintf( stdout, "# [twopt_plegma_extract] data_tag = %s\n", data_tag);

        exitstatus = read_from_h5_file ( (void*)(twopt[0][0][0]), filename, data_tag, "double", io_proc );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[twopt_plegma_extract] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        for ( int imom = 0; imom < g_source_momentum_number; imom++ ) {

          sprintf ( output_filename, "twop.pseudoscalar.PX%d_PY%d_PZ%d.h5",
              g_source_momentum_list[imom][0], g_source_momentum_list[imom][1], g_source_momentum_list[imom][2] );

          if ( g_verbose > 2 ) fprintf ( stdout, "# [twopt_plegma_extract] h5 output filename = %s %s %d\n", output_filename, __FILE__, __LINE__ );

          for ( int it = 0; it < T_global; it++ ) {
            buffer[2*it  ] = twopt[it][momentum_matchid[imom]][0][0];
            buffer[2*it+1] = twopt[it][momentum_matchid[imom]][0][1];
          }

          char h5_tag[200];
          sprintf ( h5_tag, "/s%c/c%d/t%dx%dy%dz%d/%s", 
              conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], 
              conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5],
              flavor_tag[ twop_meson_id ] );

          if ( g_verbose > 2 ) fprintf ( stdout, "# [twopt_plegma_extract] h5 tag = %s %s %d\n", h5_tag, __FILE__, __LINE__  );


          int const ncdim = 1;
          int const cdim[1] = { 2 * T_global };

          exitstatus = write_h5_contraction ( (void*)buffer, NULL, output_filename, h5_tag, "double", ncdim, cdim  );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[twopt_plegma_extract] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

        }  /* end of loop on momenta */

      }  /* end of meson ids */
  
    }  /* end of loop on sources */

  }  /* end of loop on configs */

  fini_4level_dtable ( &twopt );
  fini_1level_dtable ( &buffer );

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  free_geometry();
  fini_1level_itable ( &momentum_matchid );
  fini_3level_itable ( &conf_src_list );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [twopt_plegma_extract] %s# [twopt_plegma_extract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [twopt_plegma_extract] %s# [twopt_plegma_extract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
