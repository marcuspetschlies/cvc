/****************************************************
 * compact-block-3pt 
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

#define _INPUT_AFF  1

#define _OUTPUT_H5  1

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
  
  char const correlator_prefix[2][20] = { "p-lvc-lvc" };

  char const flavor_tag[4][20]        = { "fl0" , "fl1" };

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
      fprintf ( stdout, "# [compact-block-3pt] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [compact-block-3pt] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [compact-block-3pt] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [compact-block-3pt] operator_type set to %d\n", operator_type );
      break;
    case 'F':
      flavor_type[flavor_num] = atoi ( optarg );
      fprintf ( stdout, "# [compact-block-3pt] flavor_type %d set to %d\n", flavor_num, flavor_type[flavor_num] );
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
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [compact-block-3pt] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [compact-block-3pt] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
  set_omp_number_threads ();

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[compact-block-3pt] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[compact-block-3pt] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [compact-block-3pt] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  /* sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf); */
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[compact-block-3pt] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[compact-block-3pt] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [compact-block-3pt] comment %s\n", line );
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

  for ( int ipseq = 0; ipseq < g_seq_source_momentum_number; ipseq++ ) 
  {
    int pseq[3] = {
        g_seq_source_momentum_list[ipseq][0],
        g_seq_source_momentum_list[ipseq][1],
        g_seq_source_momentum_list[ipseq][2] };

  for ( int igseq = 0; igseq < g_sequential_source_gamma_id_number; igseq++ ) 
  {
    int gseq = g_sequential_source_gamma_id_list[igseq];

  for ( int itseq =0; itseq < g_sequential_source_timeslice_number; itseq++ ) 
  {
    int tseq = g_sequential_source_timeslice_list[itseq];

    /***********************************************************
     * twop array
     ***********************************************************/
    double ****** threep_buffer = init_6level_dtable ( 2, g_sink_momentum_number, num_src_per_conf, g_sink_gamma_id_number, g_source_gamma_id_number, 2*T_global );
    if( threep_buffer == NULL ) {
      fprintf ( stderr, "[compact-block-3pt] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT (24);
    }

    /***********************************************************
     *
     * READ INPUT DATA
     *
     ***********************************************************/

    /***********************************************************
     * loop on configs
     ***********************************************************/
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {

      gettimeofday ( &ta, (struct timezone *)NULL );

      /***********************************************************
       * loop on sources
       ***********************************************************/
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
  
#if _INPUT_AFF

        /***********************************************************
         * open AFF reader
         ***********************************************************/
        struct AffReader_s *affr = NULL;
        struct AffNode_s *affn = NULL, *affpath = NULL, *affdir = NULL;
        char key[400];
        char data_filename[500];

        sprintf( data_filename, "%s/stream_%c/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff",
            filename_prefix,
            conf_src_list[iconf][isrc][0], 
            filename_prefix3,
            conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );

        fprintf(stdout, "# [compact-block-3pt] reading data from file %s\n", data_filename);
        affr = aff_reader ( data_filename );
        const char * aff_status_str = aff_reader_errstr ( affr );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[compact-block-3pt] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
          EXIT(15);
        }
  
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[compact-block-3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }

        /***********************************************************
         * loop on flavor
         ***********************************************************/
        for ( int iflavor = 0; iflavor < flavor_num ; iflavor++ ) {
          const int flavor_id = flavor_type[iflavor];

          sprintf( key, "/%s/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/%s",
                correlator_prefix[operator_type],
                conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5],
                pseq[0], pseq[1], pseq[2],
                gseq,
                tseq,
                flavor_tag[flavor_id] );
  
          affpath = aff_reader_chpath (affr, affn, key );
          if( affpath == NULL ) {
            fprintf(stderr, "[compact-block-3pt] Error from aff_reader_chpath for path %s %s %d\n", key, __FILE__, __LINE__);
            EXIT(105);
          } else {
            if ( g_verbose > 2 ) fprintf ( stdout, "# [compact-block-3pt] path = %s\n", key );
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
                fprintf(stderr, "[compact-block-3pt] Error from aff_reader_chpath for key %s %s %d\n", key, __FILE__, __LINE__);
                EXIT(105);
              } else {
                if ( g_verbose > 2 ) fprintf ( stdout, "# [compact-block-3pt] read key = %s\n", key );
              }
  
              uint32_t uitems = T_global;
              int texitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)threep_buffer[iflavor][ipf][isrc][igf][igi], uitems );
              if( texitstatus != 0 ) {
                fprintf(stderr, "[compact-block-3pt] Error from aff_node_get_complex, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
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

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "compact-block-3pt", "read-aff-per-conf", g_cart_id == 0 );

#endif  /* of if _INPUT_AFF */

#if _OUTPUT_H5
      /***********************************************************
       *
       * WRITE OUTPUT TO AFF
       *
       ***********************************************************/
      gettimeofday ( &ta, (struct timezone *)NULL );

      /***********************************************************
       * loop on flavor
       ***********************************************************/
      for ( int iflavor = 0; iflavor < flavor_num ; iflavor++ ) {
        const int flavor_id = flavor_type[iflavor];

        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
          int pf[3] = {
            g_sink_momentum_list[imom][0],
            g_sink_momentum_list[imom][1],
            g_sink_momentum_list[imom][2] };
    
          /***********************************************************
           * filename and key
           ***********************************************************/
          char key[400];
          char data_filename[500];
        
          sprintf( data_filename, "%s.%s.%s.qx%d_qy%d_qz%d.gseq_%d.tseq_%d.px%d_py%d_pz%d.h5", 
              g_outfile_prefix, correlator_prefix[operator_type], flavor_tag[flavor_id],
              pseq[0], pseq[1], pseq[2],
              gseq, tseq,
              pf[0], pf[1], pf[2] );

          if ( g_verbose > 1 ) fprintf ( stdout, "# [compact-block-3pt] output filename = %s\n", data_filename );
    
          if (  iconf == 0 ) 
          {
            sprintf( key, "/mom" );
        
            int const ncdim   = 2;              
            int const cdim[2] = { 1, 3 };

            int texitstatus = write_h5_contraction ( pf, NULL, data_filename, key, "int", ncdim, cdim );
            if( texitstatus != 0 && texitstatus != 19 ) {
              fprintf(stderr, "[compact-block-3pt] Error from write_h5_contraction, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
              EXIT(105);
            } else if (texitstatus == 19 ) {
              fprintf(stderr, "[compact-block-3pt] WARNING from write_h5_contraction; could not create data set %s / %s  %s %d\n", data_filename, key, __FILE__, __LINE__);
            }
          }
  
          if ( iconf == 0 ) 
          {
            sprintf( key, "/gf" );
        
            int const ncdim   = 1;              
            int const cdim[1] = { g_sink_gamma_id_number };

            int texitstatus = write_h5_contraction ( g_sink_gamma_id_list, NULL, data_filename, key, "int", ncdim, cdim );
            if( texitstatus != 0 && texitstatus != 19 ) {
              fprintf(stderr, "[compact-block-3pt] Error from write_h5_contraction, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
              EXIT(105);
            } else if (texitstatus == 19 ) {
              fprintf(stderr, "[compact-block-3pt] WARNING from write_h5_contraction; could not create data set %s / %s  %s %d\n", data_filename, key, __FILE__, __LINE__);
            }
          }

          if ( iconf == 0 )
          {
            sprintf( key, "/gi" );
        
            int const ncdim   = 1;              
            int const cdim[1] = { g_source_gamma_id_number };

            int texitstatus = write_h5_contraction ( g_source_gamma_id_list, NULL, data_filename, key, "int", ncdim, cdim );
            if( texitstatus != 0 && texitstatus != 19 ) {
              fprintf(stderr, "[compact-block-3pt] Error from write_h5_contraction, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
              EXIT(105);
            } else if (texitstatus == 19 ) {
              fprintf(stderr, "[compact-block-3pt] WARNING from write_h5_contraction; could not create data set %s / %s  %s %d\n", data_filename, key, __FILE__, __LINE__);
            }
          }

          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
          {

            sprintf( key, "/stream_%c/conf_%d/t%d_x%d_y%d_z%d",
                conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
                conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
        
            if ( g_verbose > 1 ) fprintf ( stdout, "# [compact-block-3pt] write key = %s\n", key );
      
            int const ncdim   = 3;
            int const cdim[3] = { g_sink_gamma_id_number, g_source_gamma_id_number, 2*T_global };

            int texitstatus = write_h5_contraction ( threep_buffer[iflavor][imom][isrc][0][0], NULL, data_filename, key, "double", ncdim, cdim );
            if( texitstatus != 0 && texitstatus != 19 ) {
              fprintf(stderr, "[compact-block-3pt] Error from write_h5_contraction, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
              EXIT(105);
            } else if (texitstatus == 19 ) {
              fprintf(stderr, "[compact-block-3pt] WARNING from write_h5_contraction; could not create data set %s / %s  %s %d\n", data_filename, key, __FILE__, __LINE__);
            }
          }  /* end of loop on sources */

        }  /* end of loop on sink momenta */
      }  /* end of loop on flavor */
      
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "compact-block-3pt", "write-h5-per-conf", g_cart_id == 0 );

#endif  /* of if _OUTPUT_H5 */

    }  /* end of loop on configs */

    fini_6level_dtable ( &threep_buffer );

  }  /* end of loop on tseq */
  }  /* end of loop on gseq */
  }  /* end of loop on pseq */

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
