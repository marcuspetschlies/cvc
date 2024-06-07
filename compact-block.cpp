/****************************************************
 * compact-block 
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
#define _INPUT_H5   0

#define _OUTPUT_H5  1

using namespace cvc;

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
  
  char const correlator_prefix[2][20] = { "local-local" , "charged"};
#if _INPUT_AFF
  char const flavor_tag[4][20]        = { "d-gf-u-gi" , "u-gf-d-gi" , "u-gf-u-gi", "d-gf-d-gi" };
#elif _INPUT_H5
  char const flavor_tag[2][20]        = { "u-v-u-v" , "u-s-u-s" };
#endif

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
      fprintf ( stdout, "# [compact-block] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [compact-block] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [compact-block] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [compact-block] operator_type set to %d\n", operator_type );
      break;
    case 'F':
      flavor_type[flavor_num] = atoi ( optarg );
      fprintf ( stdout, "# [compact-block] flavor_type %d set to %d\n", flavor_num, flavor_type[flavor_num] );
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
  /* fprintf(stdout, "# [compact-block] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [compact-block] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
  set_omp_number_threads ();

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[compact-block] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[compact-block] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [compact-block] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  /* sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf); */
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[compact-block] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[compact-block] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [compact-block] comment %s\n", line );
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
   *
   * READ INPUT DATA
   *
   ***********************************************************/

  /***********************************************************
   * loop on configs
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
 
    /***********************************************************
     * loop on flavor
     ***********************************************************/
    for ( int iflavor = 0; iflavor < flavor_num ; iflavor++ ) {
      const int flavor_id = flavor_type[iflavor];

#if _INPUT_AFF

      /***********************************************************
       * twop array
       ***********************************************************/
      double ****** twop_buffer = init_6level_dtable ( 2, g_sink_momentum_number, num_src_per_conf, g_sink_gamma_id_number, g_source_gamma_id_number, 2*T_global );
     if( twop_buffer == NULL ) {
         fprintf ( stderr, "[compact-block] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT (24);
      }

      gettimeofday ( &ta, (struct timezone *)NULL );

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


        /* sprintf( data_filename, "%s/stream_%c/%s/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff",
            filename_prefix,
            conf_src_list[iconf][isrc][0], 
            filename_prefix2,
            filename_prefix3,
            conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] ); */


        sprintf( data_filename, "%s/stream_%c/%d/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff",
              filename_prefix,
              conf_src_list[iconf][isrc][0], 
              conf_src_list[iconf][isrc][1], 
              filename_prefix3,
              conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );

        if ( g_verbose > 2 ) fprintf(stdout, "# [compact-block] reading data from file %s\n", data_filename);
        affr = aff_reader ( data_filename );
        const char * aff_status_str = aff_reader_errstr ( affr );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[compact-block] Error from aff_reader for file %s, status was %s %s %d\n", data_filename, aff_status_str, __FILE__, __LINE__);
          EXIT(15);
        }
  
        if( (affn = aff_reader_root( affr )) == NULL ) {
           fprintf(stderr, "[compact-block] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
           EXIT(103);
        }

        sprintf( key, "/%s/%s/t%.2dx%.2dy%.2dz%.2d",
                correlator_prefix[operator_type], flavor_tag[flavor_id],
                conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
  
        affpath = aff_reader_chpath (affr, affn, key );
        if( affpath == NULL ) {
          fprintf(stderr, "[compact-block] Error from aff_reader_chpath for path %s %s %d\n", key, __FILE__, __LINE__);
          EXIT(105);
        } else {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [compact-block] path = %s\n", key );
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
                fprintf(stderr, "[compact-block] Error from aff_reader_chpath for key %s %s %d\n", key, __FILE__, __LINE__);
                EXIT(105);
              } else {
                if ( g_verbose > 2 ) fprintf ( stdout, "# [compact-block] read key = %s\n", key );
              }
  
              uint32_t uitems = T_global;
              int texitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)twop_buffer[iflavor][ipf][isrc][igf][igi], uitems );
              if( texitstatus != 0 ) {
                fprintf(stderr, "[compact-block] Error from aff_node_get_complex, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
  
            }  /* end of loop on sink momenta */
          }  /* end of loop on source gamma id */
        }  /* end of loop on sink gamma id */

        /**********************************************************
         * close the reader
         **********************************************************/
        aff_reader_close ( affr );

      }  /* end of loop on src */

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "compact-block", "read-aff-per-conf", g_cart_id == 0 );

#endif  /* of if _INPUT_AFF */

  /**********************************************************/
  /**********************************************************/

#if _INPUT_H5

      double ****** twop_buffer = NULL;
      double **** buffer = NULL;
      int momentum_number = 0;
      int ** momentum_list = NULL;
      char flavor_list[2], op_list[2];

      gettimeofday ( &ta, (struct timezone *)NULL );

      /***********************************************
       * reader for aff input file
       ***********************************************/
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        char data_filename[500];

        sprintf( data_filename, "%s/stream_%c/%d/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.h5",
            filename_prefix,
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            filename_prefix3,
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );

        fprintf(stdout, "# [compact-block] reading data from file %s\n", data_filename);

        if ( isrc == 0 ) {

          char momentum_tag[12] = "/mom_snk";
          int * momentum_buffer = NULL;
          size_t * momentum_cdim = NULL, momentum_ncdim = 0;

          exitstatus = read_from_h5_file_varsize ( (void**)&momentum_buffer, data_filename, momentum_tag,  "int", &momentum_ncdim, &momentum_cdim,  io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[compact-block] Error from read_from_h5_file_varsize for file %s key %s   %s %d\n", data_filename, momentum_tag, __FILE__, __LINE__);
            EXIT(15);
          }

          if ( momentum_ncdim != 2 || momentum_cdim[1] != 3 ) {
            fprintf ( stderr, "[compact-block] Error from read_from_h5_file_varsize for file data %s %d\n", __FILE__, __LINE__ );
            EXIT(129);
          }

          momentum_number = (int)(momentum_cdim[0]);
          if ( g_verbose > 4 ) fprintf ( stdout, "# [p2gg_analyse] read %d momenta %s %d\n", momentum_number, __FILE__, __LINE__ );
          momentum_list = init_2level_itable ( momentum_number, 3 );
          memcpy ( momentum_list[0], momentum_buffer, momentum_number * 3 * sizeof ( int ) );
          free ( momentum_buffer );
          free ( momentum_cdim );

          /**********************************************************
           * gamma at sink list
           **********************************************************/
          sscanf ( flavor_tag[flavor_id], "%c-%c-%c-%c", flavor_list, op_list, flavor_list+1, op_list+1 );
          if( g_verbose > 4 ) fprintf ( stdout, "# [compact-block] flavor_list = %c  %c    op_list = %c  %c   %s %d\n",
              flavor_list[0], flavor_list[1], op_list[0], op_list[1], __FILE__, __LINE__ ); 

          char gamma_tag[12];
          int * gamma_buffer = NULL;
          size_t * gamma_cdim = NULL, gamma_ncdim = 0;

          sprintf( gamma_tag, "/gamma_%c", op_list[0] );
          if ( g_verbose > 4 ) fprintf ( stdout, "# [] gamma_tag = %s    %s %d\n", gamma_tag, __FILE__, __LINE__ );

          exitstatus = read_from_h5_file_varsize ( (void**)&gamma_buffer, data_filename, gamma_tag,  "int", &gamma_ncdim, &gamma_cdim,  io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[compact-block] Error from read_from_h5_file_varsize for file %s key %s   %s %d\n", data_filename, gamma_tag, __FILE__, __LINE__);
            EXIT(15);
          }

          if ( gamma_ncdim != 1 ) {
            fprintf ( stderr, "[compact-block] Error from read_from_h5_file_varsize for file data %s %d\n", __FILE__, __LINE__ );
            EXIT(129);
          }

          g_sink_gamma_id_number = (int)(gamma_cdim[0]);
          if ( g_verbose > 4 ) fprintf ( stdout, "# [p2gg_analyse] read %d gamma at sink %s %d\n", g_sink_gamma_id_number, __FILE__, __LINE__ );
          memcpy ( g_sink_gamma_id_list, gamma_buffer, g_sink_gamma_id_number * sizeof ( int ) );
          free ( gamma_buffer );
          free ( gamma_cdim );

          /**********************************************************
           * gamma at source list
           **********************************************************/
          sprintf( gamma_tag, "/gamma_%c", op_list[1] );
          exitstatus = read_from_h5_file_varsize ( (void**)&gamma_buffer, data_filename, gamma_tag,  "int", &gamma_ncdim, &gamma_cdim,  io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[compact-block] Error from read_from_h5_file_varsize for file %s key %s   %s %d\n", data_filename, gamma_tag, __FILE__, __LINE__);
            EXIT(15);
          }

          if ( gamma_ncdim != 1 ) {
            fprintf ( stderr, "[compact-block] Error from read_from_h5_file_varsize for file data %s %d\n", __FILE__, __LINE__ );
            EXIT(129);
          }

          g_source_gamma_id_number = (int)(gamma_cdim[0]);
          if ( g_verbose > 4 ) fprintf ( stdout, "# [p2gg_analyse] read %d gamma at source %s %d\n", g_source_gamma_id_number, __FILE__, __LINE__ );
          memcpy ( g_source_gamma_id_list, gamma_buffer, g_source_gamma_id_number * sizeof ( int ) );
          free ( gamma_buffer );
          free ( gamma_cdim );

          /***********************************************************
           * twop array
           ***********************************************************/
          twop_buffer = init_6level_dtable ( 2, g_sink_momentum_number, num_src_per_conf, g_sink_gamma_id_number, g_source_gamma_id_number, 2*T_global );
          if( twop_buffer == NULL ) {
            fprintf ( stderr, "[compact-block] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT (24);
          }

          /**********************************************************
           *
           **********************************************************/
          buffer = init_4level_dtable( T_global, g_sink_gamma_id_number, g_source_gamma_id_number, 2 * momentum_number );
          if( buffer == NULL ) {
            fprintf(stderr, "[compact-block] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(15);
          }

        }  /* end of if irsc == 0 */
 
        /**********************************************************
         *
         **********************************************************/
        char key[400];
        sprintf ( key , "/%s/%s", correlator_prefix[operator_type], flavor_tag[flavor_id] );

        exitstatus = read_from_h5_file (  buffer[0][0][0], data_filename, key,  "double", io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[p2gg_analyse] Error from read_from_h5_file for file %s key %s    %s %d\n",
             data_filename, key, __FILE__, __LINE__);
          EXIT(15);
        }
              
        /**********************************************************
         * loop on momenta
         **********************************************************/
        for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

          int const sink_momentum[3] = {
                    g_sink_momentum_list[isink_momentum][0],
                    g_sink_momentum_list[isink_momentum][1],
                    g_sink_momentum_list[isink_momentum][2] };

          int const  sink_momentum_id = get_momentum_id (  sink_momentum, momentum_list, momentum_number );

          if ( sink_momentum_id == -1 ) EXIT(127);

#pragma omp parallel for
          for ( int it = 0; it < T_global; it++ ) {
        
            for( int mu = 0; mu < g_sink_gamma_id_number; mu++) {
            for( int nu = 0; nu < g_source_gamma_id_number; nu++) {

              twop_buffer[iflavor][isink_momentum][isrc][mu][nu][2*it  ] = buffer[it][mu][nu][2 * sink_momentum_id     ];
              twop_buffer[iflavor][isink_momentum][isrc][mu][nu][2*it+1] = buffer[it][mu][nu][2 * sink_momentum_id + 1 ];

            }}
          }
      
        }  /* end of loop sink momentum */
 

      }  /* end of loop on source */
  
      fini_2level_itable ( &momentum_list );

      fini_4level_dtable( &buffer );
  
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "compact-block", "read-h5-per-conf", g_cart_id == 0 );
   
#endif  /* end of _INPUT_H5 */


#if _OUTPUT_H5
    /***********************************************************
     *
     * WRITE OUTPUT TO H5
     *
     ***********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

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
      
        sprintf( data_filename, "%s.%s.%s.px%d_py%d_pz%d.h5", g_outfile_prefix, correlator_prefix[operator_type], flavor_tag[flavor_id],
            pf[0], pf[1], pf[2] );

        if ( g_verbose > 2 ) fprintf ( stdout, "# [compact-block] output filename = %s\n", data_filename );
  
        if ( iconf == 0 ) 
        {
          sprintf( key, "/mom" );
      
          int const ncdim   = 2;              
          int const cdim[2] = { 1, 3 };

          int texitstatus = write_h5_contraction ( pf, NULL, data_filename, key, "int", ncdim, cdim );
          if( texitstatus != 0 && texitstatus != 19 ) {
            fprintf(stderr, "[compact-block] Error from write_h5_contraction, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
            EXIT(105);
          } else if (texitstatus == 19 ) {
            fprintf(stderr, "[compact-block] WARNING from write_h5_contraction; could not create data set %s / %s  %s %d\n", data_filename, key, __FILE__, __LINE__);
          }
        }

        if ( iconf == 0 ) 
        {
          sprintf( key, "/gf" );
      
          int const ncdim   = 1;              
          int const cdim[1] = { g_sink_gamma_id_number };

          int texitstatus = write_h5_contraction ( g_sink_gamma_id_list, NULL, data_filename, key, "int", ncdim, cdim );
          if( texitstatus != 0 && texitstatus != 19  ) {
            fprintf(stderr, "[compact-block] Error from write_h5_contraction, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
            EXIT(105);
          } else if (texitstatus == 19 ) {
            fprintf(stderr, "[compact-block] WARNING from write_h5_contraction; could not create data set %s / %s  %s %d\n", data_filename, key, __FILE__, __LINE__);
          }
        }

        if ( iconf == 0 )
        {
          sprintf( key, "/gi" );
      
          int const ncdim   = 1;              
          int const cdim[1] = { g_source_gamma_id_number };

          int texitstatus = write_h5_contraction ( g_source_gamma_id_list, NULL, data_filename, key, "int", ncdim, cdim );
          if( texitstatus != 0 && texitstatus != 19  ) {
            fprintf(stderr, "[compact-block] Error from write_h5_contraction, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
            EXIT(105);
          } else if (texitstatus == 19 ) {
            fprintf(stderr, "[compact-block] WARNING from write_h5_contraction; could not create data set %s / %s  %s %d\n", data_filename, key, __FILE__, __LINE__);
          }
        }

        for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
        {
          sprintf( key, "/stream_%c/conf_%d/t%d_x%d_y%d_z%d",
              conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
              conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
        
          if ( g_verbose > 2 ) fprintf ( stdout, "# [compact-block] write key = %s\n", key );
      
          int const ncdim   = 3;
          int const cdim[3] = { g_sink_gamma_id_number, g_source_gamma_id_number, 2*T_global };

          int texitstatus = write_h5_contraction ( twop_buffer[iflavor][imom][isrc][0][0], NULL, data_filename, key, "double", ncdim, cdim );
          if( texitstatus != 0 && texitstatus != 19 ) {
            fprintf(stderr, "[compact-block] Error from write_h5_contraction, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
            EXIT(105);
          } else if (texitstatus == 19 ) {
            fprintf(stderr, "[compact-block] WARNING from write_h5_contraction; could not create data set %s / %s  %s %d\n", data_filename, key, __FILE__, __LINE__);
          }
        }

      }  /* end of loop on sink momenta */
      
#endif  /* of if _OUTPUT_H5 */

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "compact-block", "write-h5-per-conf", g_cart_id == 0 );
   


      fini_6level_dtable ( &twop_buffer );

    }  /* end of loop on flavor */

  }  /* end of loop on configs */

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
