/****************************************************
 * compact 
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
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_i.h"
#include "gamma.h"
#include "uwerr.h"
#include "derived_quantities.h"

#define _TWOP_AFF

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse < Y-O_g Ybar > correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}

/**********************************************************
 *
 **********************************************************/
int main(int argc, char **argv) {
  
  char const correlator_prefix[2][20] = { "N-N" };

  /* char const flavor_tag[4][20]        = {  }; */

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[600];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  int operator_type = -1;
  int flavor_type[4];
  int flavor_num = 0;
  struct timeval ta, tb;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:E:O:F:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [compact] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [compact] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [compact] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [compact] operator_type set to %d\n", operator_type );
      break;
    case 'F':
      flavor_type[flavor_num] = atoi ( optarg );
      fprintf ( stdout, "# [compact] flavor_type %d set to %d\n", flavor_num, flavor_type[flavor_num] );
      flavor_num++;
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
  /* fprintf(stdout, "# [compact] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [compact] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
  set_omp_number_threads ();

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[compact] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[compact] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [compact] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[compact] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[compact] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [compact] comment %s\n", line );
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

  /***********************************************************
   * twop array
   ***********************************************************/
  double ******** twop = init_8level_dtable ( g_sink_gamma_id_number, g_source_gamma_id_number, g_sink_momentum_number, num_conf, num_src_per_conf, 2, T_global, 2 );
  if( twop == NULL ) {
    fprintf ( stderr, "[compact] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

  /***********************************************************
   *
   * READ IN
   *
   ***********************************************************/
#ifdef _TWOP_AFF
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
      struct AffReader_s *affr = NULL;
      struct AffNode_s *affn = NULL, *affpath = NULL, *affdir = NULL;
      char key[400];
      char data_filename[500];
    
      sprintf( data_filename, "%s/stream_%c/light/p2gg_twop_local/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff",
          filename_prefix,
          conf_src_list[iconf][0][0], 
          filename_prefix2, 
          conf_src_list[iconf][0][1], conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );

      fprintf(stdout, "# [compact] reading data from file %s\n", data_filename);
      affr = aff_reader ( data_filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[compact] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
  
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[compact] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }

      /***********************************************************
       * loop on flavor
       ***********************************************************/
      for ( int iflavor = 0; iflavor < flavor_num ; iflavor++ ) {
        const int flavor_id = flavor_type[iflavor];

        sprintf( key, "/%s/%s/t%.2dx%.2dy%.2dz%.2d",
              correlator_prefix[operator_type], flavor_tag[flavor_id],
              conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
  
        affpath = aff_reader_chpath (affr, affn, key );
        if( affpath == NULL ) {
          fprintf(stderr, "[compact] Error from aff_reader_chpath for path %s %s %d\n", key, __FILE__, __LINE__);
          EXIT(105);
        } else {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [compact] path = %s\n", key );
        }

        /***********************************************************
         * loop on gamma at sink
         ***********************************************************/
        for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ ) {

        /***********************************************************
         * loop on gamma at source
         ***********************************************************/
        for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) {

          /***********************************************************
           * loop on sink momenta
           ***********************************************************/
          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
  
            sprintf( key, "gf%.2d/gi%.2d/px%dpy%dpz%d",
                g_sink_gamma_id_list[igf], g_source_gamma_id_list[igi],
                g_sink_momentum_list[ipf][0], g_sink_momentum_list[ipf][1], g_sink_momentum_list[ipf][2] );
  
            affdir = aff_reader_chpath (affr, affpath, key );
            if( affdir == NULL ) {
              fprintf(stderr, "[compact] Error from aff_reader_chpath for key %s %s %d\n", key, __FILE__, __LINE__);
              EXIT(105);
            } else {
              if ( g_verbose > 2 ) fprintf ( stdout, "# [compact] read key = %s\n", key );
            }
  
            uint32_t uitems = T_global;
            int texitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)twop[igf][igi][ipf][iconf][isrc][iflavor][0], uitems );
            if( texitstatus != 0 ) {
              fprintf(stderr, "[compact] Error from aff_node_get_complex, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
  
          }  /* end of loop on sink momenta */
        }  /* end of loop on source gamma id */
        }  /* end of loop on sink gamma id */
      }  /* end of loop on flavor  */

      /**********************************************************
       * close the reader
       **********************************************************/
      aff_reader_close ( affr );
  
    }  /* end of loop on sources */
  }  /* end of loop on configs */

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "compact", "read-aff", g_cart_id == 0 );
#endif  /* of if _TWOP_AFF */

  /***********************************************************
   *
   * WRITE OUT
   *
   ***********************************************************/
#ifdef _TWOP_AFF
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***********************************************************
   * loop on flavor
   ***********************************************************/
  for ( int iflavor = 0; iflavor < flavor_num ; iflavor++ ) {
    const int flavor_id = flavor_type[iflavor];

    /***********************************************************
     * loop on gamma at sink
     ***********************************************************/
    for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ ) {

      /***********************************************************
       * loop on gamma at source
       ***********************************************************/
      for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) {

        /***********************************************************
         * loop on sink momenta
         ***********************************************************/
        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

          /***********************************************************
           * open AFF writer
           ***********************************************************/
          struct AffWriter_s *affw = NULL;
          struct AffNode_s *affn = NULL, *affpath = NULL, *affdir = NULL;
          char key[400];
          char data_filename[500];
    
          sprintf( data_filename, "%s/%s.%s.gf%.2d.gi%.2d.px%dpy%dpz%d.aff",
              filename_prefix3,
              correlator_prefix[operator_type], flavor_tag[flavor_id],
              g_sink_gamma_id_list[igf], g_source_gamma_id_list[igi],
              g_sink_momentum_list[ipf][0], g_sink_momentum_list[ipf][1], g_sink_momentum_list[ipf][2] );


          affw = aff_writer ( data_filename );
          char * aff_status_str = (char*)aff_writer_errstr ( affw );
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[compact] Error from aff_writer for filename %s, status was %s %s %d\n", data_filename, aff_status_str, __FILE__, __LINE__);
            EXIT(15);
          } else {
            if ( g_verbose > 2 ) fprintf ( stdout, "# [compact] writing to file %s %s %d\n", data_filename, __FILE__, __LINE__ );
          }
  
          if( (affn = aff_writer_root( affw )) == NULL ) {
            fprintf(stderr, "[compact] Error, aff wirter is not initialized %s %d\n", __FILE__, __LINE__);
            EXIT(103);
          }

          /***********************************************************
           * loop on configs
           ***********************************************************/
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {

            /***********************************************************
             * loop on sources
             ***********************************************************/
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
  
              sprintf( key, "/stream_%c/conf_%d/t%.2dx%.2dy%.2dz%.2d",
                  conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
                  conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
    
              affdir = aff_writer_mkpath(affw, affn, key );
              if( affdir == NULL ) {
                fprintf(stderr, "[compact] Error from aff_writer_mkpath for path %s %s %d\n", key, __FILE__, __LINE__);
                EXIT(105);
              } else {
                if ( g_verbose > 2 ) fprintf ( stdout, "# [compact] write key = %s\n", key );
              }
  

              int texitstatus = aff_node_put_complex (affw, affdir, (double _Complex*)(twop[igf][igi][ipf][iconf][isrc][iflavor][0]), (uint32_t)T_global );
              if( texitstatus != 0 ) {
                fprintf(stderr, "[compact] Error from aff_node_put_complex, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
  
            }  /* end of loop on sources */
          }  /* end of loop on configs */

          /**********************************************************
           * close the writer
           **********************************************************/
          aff_status_str = (char*)aff_writer_close (affw);
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[compact] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(32);
          }
        }  /* end of loop on sink momenta */
      }  /* end of loop on source gamma id */
    }  /* end of loop on sink gamma id */
  }  /* end of loop on flavor */
  

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "compact", "write-aff", g_cart_id == 0 );
#endif  /* of if _TWOP_AFF */

  fini_8level_dtable ( &twop );

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

}
