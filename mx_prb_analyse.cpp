/****************************************************
 * mx_prb_analyse
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
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_c.h"
#include "contract_diagrams.h"
#include "twopoint_function_utils.h"
#include "gamma.h"
#include "uwerr.h"
#include "derived_quantities.h"
#include "cvc_timer.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code form 2-pt function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  EXIT(0);
}

typedef struct {
  int nc;
  int ns;
  int * conf;
  int ***src;
  char *stream;
} conf_src_list_type;


  char const gamma_id_to_Cg_ascii[16][10] = {
    "Cgy",
    "Cgzg5",
    "Cgt",
    "Cgxg5",
    "Cgygt",
    "Cgyg5gt",
    "Cgyg5",
    "Cgz",
    "Cg5gt",
    "Cgx",
    "Cgzg5gt",
    "C",
    "Cgxg5gt",
    "Cgxgt",
    "Cg5",
    "Cgzgt"
  };


  char const gamma_id_to_ascii[16][10] = {
    "gt",
    "gx",
    "gy",
    "gz",
    "id",
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

/***************************************************************************
 *
 ***************************************************************************/
void make_key_string ( char * key, twopoint_function_type *tp, const char * type, const char * diagram_name , const int * sx ) {

  const int * gsx = ( sx == NULL ) ? tp->source_coords : sx;
  const char * tp_type = ( type ==  NULL ) ? tp->type : type;

  if ( strcmp ( tp_type, "b-b" ) == 0 ) {

    sprintf ( key, "/%s/T%d_X%d_Y%d_Z%d/Gi_%s/Gf_%s/%s/px%.2dpy%.2dpz%.2d", tp->name, gsx[0], gsx[1], gsx[2], gsx[3],
        gamma_id_to_Cg_ascii[tp->gi1[0]],
        gamma_id_to_Cg_ascii[tp->gf1[0]],
        diagram_name,
        tp->pf1[0], tp->pf1[1], tp->pf1[2] );

  } else if ( strcmp( tp_type , "b-qq-b" ) == 0 ) {

    sprintf ( key, "/%s/T%d_X%d_Y%d_Z%d/Gf_%s/Gc_%s/Gi_%s/QX%d_QY%d_QZ%d/%s/px%.2dpy%.2dpz%.2d", tp->name, gsx[0], gsx[1], gsx[2], gsx[3],
        gamma_id_to_Cg_ascii[tp->gf1[0]],
        gamma_id_to_ascii[tp->gf2],
        gamma_id_to_Cg_ascii[tp->gi1[0]],
        tp->pf2[0], tp->pf2[1], tp->pf2[2],
        diagram_name,
        tp->pf1[0], tp->pf1[1], tp->pf1[2] );

  }
}  /* end of make_key_string */


/***************************************************************************
 *
 ***************************************************************************/
void make_correlator_string ( char * name , twopoint_function_type * tp , const char * type ) {

  const char * tp_type = ( type ==  NULL ) ? tp->type : type;

 if ( strcmp ( tp_type, "b-b" ) == 0 ) {
   sprintf ( name,  "%s.Gi_%s_%s.Gf_%s_%s.px%dpy%dpz%d", tp->name,
       gamma_id_to_Cg_ascii[tp->gi1[0]],
       gamma_id_to_ascii[tp->gi1[1]],
       gamma_id_to_Cg_ascii[tp->gf1[0]],
       gamma_id_to_ascii[tp->gf1[1]],
       tp->pf1[0], tp->pf1[1], tp->pf1[2] );

  } else if ( strcmp( tp_type, "b-qq-b" ) == 0 ) {

    sprintf ( name,  "%s.Gi_%s_%s.Gc_%s.Gf_%s_%s.qx%dqy%dqz%d.px%dpy%dpz%d", tp->name,
        gamma_id_to_Cg_ascii[tp->gi1[0]],
        gamma_id_to_ascii[tp->gi1[1]],
        gamma_id_to_ascii[tp->gf2],
        gamma_id_to_Cg_ascii[tp->gf1[0]],
        gamma_id_to_ascii[tp->gf1[1]],
        tp->pf2[0], tp->pf2[1], tp->pf2[2],
        tp->pf1[0], tp->pf1[1], tp->pf1[2] );
  }
}  /* end of make_correlator_string */

/***************************************************************************
 *
 ***************************************************************************/
void make_diagram_list_string ( char * s, twopoint_function_type * tp ) {
  char comma = ',';
  char bar  = '_';
  char * s_ = s;
  strcpy ( s, tp->diagrams );
  while ( *s_ != '\0' ) {
    if ( *s_ ==  comma ) *s_ = bar;
    s_++;
  }
  if ( g_verbose > 2 ) fprintf ( stdout, "# [make_diagram_list_string] %s ---> %s\n", tp->diagrams, s );
  return;
}  /* end of make_diagram_list_string */


/***************************************************************************
 *
 ***************************************************************************/
int main(int argc, char **argv) {
 
  char const reim_str[2][3] = { "re", "im" };

  char const op_mx_list[4][12] = { "g5", "m", "D", "g5sigmaG"  };

  char const op_twop_list[3][12] = { "g5", "id", "g5" };

  char const diag_mx_list[2][4] = { "b", "d" };

  char const diag_twop_list[2][4] = { "m1", "m2" };

  char const op_c_list[4][2] = { "s", "p", "v", "a" };

  char const flavor_tag[4][2] = { "u", "d", "s", "c" };


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char ensemble_name[100] = "NA";
  char filename[600];
  int num_conf = 0, num_src_per_conf = 0;
  struct timeval ta, tb;

  double _Complex op_norm_list[4] = { I, 1., 1., I };

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:E:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [mx_prb_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [mx_prb_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [mx_prb_analyse] ensemble name set to = %s\n", ensemble_name );
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
  if(filename_set==0) strcpy(filename, "twopt.input");
  /* fprintf(stdout, "# [mx_prb_analyse] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [mx_prb_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [mx_prb_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [mx_prb_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[mx_prb_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[mx_prb_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * gamma matrices
   ***********************************************************/

  gamma_matrix_type gammaMat[16];
  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set ( gammaMat+i, i, 1. );
  }

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[mx_prb_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [mx_prb_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[mx_prb_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  conf_src_list_type conf_src_list;


  conf_src_list.conf = init_1level_itable ( num_conf );
  if ( conf_src_list.conf == NULL ) {
    fprintf(stderr, "[mx_prb_analyse] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.src = init_3level_itable ( num_conf, num_src_per_conf, 4 );
  if ( conf_src_list.src == NULL ) {
    fprintf(stderr, "[mx_prb_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.stream = init_1level_ctable ( num_conf );
  if ( conf_src_list.stream == NULL ) {
    fprintf(stderr, "[mx_prb_analyse] Error from init_1level_ctable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.nc = num_conf;
  conf_src_list.ns = num_src_per_conf;

  char line[100];
  int countc = -1, counts=0;
  int conf_prev = -1;

  while ( fgets ( line, 100, ofs) != NULL && countc < num_conf && counts <= num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [mx_prb_analyse] comment %s\n", line );
      continue;
    }

    if( g_verbose > 4 ) fprintf ( stdout, "# [mx_prb_analyse] line = \"%s\"\n", line );

    int conf_tmp, src_tmp[4];
    char stream_tmp;

    /***********************************************************
     * stream conf t x y z
     ***********************************************************/
    sscanf( line, "%c %d %d %d %d %d", &stream_tmp, &conf_tmp, src_tmp, src_tmp+1, src_tmp+2, src_tmp+3 );

    if ( g_verbose > 5 ) fprintf ( stdout, "# [mx_prb_analyse] before: conf_tmp = %4d   conf_prev = %4d   countc = %d   counts = %d\n", conf_tmp, conf_prev, countc, counts );

    if ( conf_tmp != conf_prev ) {
      /* new config */
      countc++;
      counts=0;
      conf_prev = conf_tmp;

      conf_src_list.stream[countc] = stream_tmp;
      conf_src_list.conf[countc]   = conf_tmp;
    }

    if ( g_verbose > 5 ) fprintf ( stdout, "# [mx_prb_analyse] after : conf_tmp = %4d   conf_prev = %4d   countc = %d   counts = %d\n", conf_tmp, conf_prev, countc, counts );

    memcpy ( conf_src_list.src[countc][counts] , src_tmp, 4*sizeof(int) );

    counts++;

#if 0
#endif
  }

  fclose ( ofs );

  /***********************************************************
   * show all configs and source locations
   ***********************************************************/
  if ( g_verbose > 4 ) {
    fprintf ( stdout, "# [mx_prb_analyse] conf_src_list conf t x y z\n" );
    for ( int iconf = 0; iconf < conf_src_list.nc; iconf++ ) {
      for( int isrc = 0; isrc < conf_src_list.ns; isrc++ ) {
        fprintf ( stdout, "  %2c %6d %3d %3d %3d %3d\n", 
            conf_src_list.stream[iconf],
            conf_src_list.conf[iconf],
            conf_src_list.src[iconf][isrc][0],
            conf_src_list.src[iconf][isrc][1],
            conf_src_list.src[iconf][isrc][2],
            conf_src_list.src[iconf][isrc][3] );
      }
    }
  }

  fflush ( stdout );

  double _Complex ***** corr_op = init_5level_ztable ( 4, 4, num_conf, num_src_per_conf, T_global  );
  if ( corr_op == NULL ) {
    fprintf ( stderr, "[mx_prb_analyse] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(15);
  }
  double _Complex ***** corr_op_b = init_5level_ztable ( 4, 4, num_conf, num_src_per_conf, T_global  );
  if ( corr_op_b == NULL ) {
    fprintf ( stderr, "[mx_prb_analyse] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(15);
  }
  double _Complex ***** corr_op_d = init_5level_ztable ( 4, 4, num_conf, num_src_per_conf, T_global  );
  if ( corr_op_d == NULL ) {
    fprintf ( stderr, "[mx_prb_analyse] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(15);
  }

  double _Complex **** corr_twop = init_4level_ztable ( 3, num_conf, num_src_per_conf, T_global  );
  if ( corr_twop == NULL ) {
    fprintf ( stderr, "[mx_prb_analyse] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(15);
  }

  /***********************************************************
   * loop on configs 
   ***********************************************************/
  for( int iconf = 0; iconf < num_conf; iconf++ ) {
 
    /***********************************************************
     * loop on sources per config
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

      /***********************************************************
       * store the source coordinates
       ***********************************************************/
      int const gsx[4] = {
        conf_src_list.src[iconf][isrc][0],
        conf_src_list.src[iconf][isrc][1],
        conf_src_list.src[iconf][isrc][2],
        conf_src_list.src[iconf][isrc][3] };

      char data_filename[500];
            
      sprintf ( data_filename, "%s/stream_%c/%d/%s.%.4d.t%dx%dy%dz%d.aff", filename_prefix2, 
          conf_src_list.stream[iconf],
          conf_src_list.conf[iconf],
          filename_prefix, conf_src_list.conf[iconf], gsx[0], gsx[1], gsx[2], gsx[3] );
      if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [mx_prb_analyse] data_filename   = %s %s %d\n", data_filename, __FILE__, __LINE__ );
      }

      struct AffReader_s *affr = NULL;
      struct AffNode_s *affn = NULL, *affdir = NULL, 
                       *affpath1 = NULL, *affpath2 = NULL;
      char key[400];

      affr = aff_reader ( data_filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[mx_prb_analyse] Error from aff_reader for filename %s, status was %s %s %d\n", data_filename, aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[mx_prb_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }

      /***************************************************************************
       ***************************************************************************
       **
       **  read mx prb operator correlators
       **
       ***************************************************************************
       ***************************************************************************/

      /* for ( int iop_mx = 0; iop_mx < 4; iop_mx++ ) */
      for ( int iop_mx = 0; iop_mx < 2; iop_mx++ )
      {

        double _Complex ***buffer = init_3level_ztable ( 2, 2, T_global );
        if ( buffer == NULL ) {
          fprintf ( stderr, "[mx_prb_analyse] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
          EXIT(15);
        }

        for ( int iop_c = 0; iop_c < 4; iop_c ++ ) {

          /* for ( int iflavor = 0; iflavor <= 1 ; iflavor++ ) */
          for ( int iflavor = 0; iflavor < 1 ; iflavor++ )
          {

            /***************************************************************************
             * Op  tag
             ***************************************************************************/
            /* /fl_u/op_g5/d_b/c_a */
 
            sprintf ( key, "/fl_%s/op_%s/d_d/c_%s", flavor_tag[iflavor], op_mx_list[iop_mx], op_c_list[iop_c] );
            if ( g_verbose > 2 ) fprintf( stdout, "# [mx_prb_analyse] key for path1 = %s %s %d\n", key, __FILE__, __LINE__ );

            affdir = aff_reader_chpath ( affr, affn, key );
            if ( affdir == NULL ) {
              fprintf ( stderr, "[mx_prb_analyse] Error from aff_reader_chpath for key %s %s %d\n", key, __FILE__, __LINE__ );
              EXIT(1);
            }

            /***********************************************************
             * read data block from AFF file
             ***********************************************************/
            uint32_t uitems = (uint32_t)T_global;
            exitstatus = aff_node_get_complex ( affr, affdir, buffer[iflavor][0], uitems );
            if( exitstatus != 0 ) {
              fprintf(stderr, "[mx_prb_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }

            sprintf ( key, "/fl_%s/op_%s/d_b/c_%s", flavor_tag[iflavor], op_mx_list[iop_mx], op_c_list[iop_c] );
            if ( g_verbose > 2 ) fprintf( stdout, "# [mx_prb_analyse] key for path2 = %s %s %d\n", key, __FILE__, __LINE__ );

            affdir = aff_reader_chpath ( affr, affn, key );
            if ( affdir == NULL ) {
              fprintf ( stderr, "[mx_prb_analyse] Error from aff_reader_chpath for key %s %s %d\n", key, __FILE__, __LINE__ );
              EXIT(1);
            }

            /***********************************************************
             * read data block from AFF file
             ***********************************************************/
            exitstatus = aff_node_get_complex ( affr, affdir, buffer[iflavor][1], uitems );
            if( exitstatus != 0 ) {
              fprintf(stderr, "[mx_prb_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }

          }  /* end of loop on iflavor */

          for ( int it = 0; it < T_global; it++ ) {
            int const itt = ( it + gsx[0] ) % T_global;
            /* corr_op[iop_mx][iop_c][iconf][isrc][it] = ( -buffer[0][0][itt] + buffer[0][1][itt] ) + conj( ( -buffer[1][0][itt] + buffer[1][1][itt] ) );
            corr_op[iop_mx][iop_c][iconf][isrc][it] *= 0.5 * op_norm_list[iop_mx]; */

            corr_op[iop_mx][iop_c][iconf][isrc][it] = ( -buffer[0][0][itt] + buffer[0][1][itt] );
            corr_op[iop_mx][iop_c][iconf][isrc][it] *= op_norm_list[iop_mx];

            corr_op_d[iop_mx][iop_c][iconf][isrc][it] =  -buffer[0][0][itt];
            corr_op_d[iop_mx][iop_c][iconf][isrc][it] *= op_norm_list[iop_mx];

            corr_op_b[iop_mx][iop_c][iconf][isrc][it] =   buffer[0][1][itt];
            corr_op_b[iop_mx][iop_c][iconf][isrc][it] *= op_norm_list[iop_mx];
          }
 
        }  /* end of loop on 4q operator componenst spva */

        fini_3level_ztable ( &buffer );
      
      }  /* end of loop on mixing operators */

      /***************************************************************************
       ***************************************************************************
       **
       **  read 2-point correlators
       **
       ** for now 2 out of 4
       ***************************************************************************
       ***************************************************************************/

      for ( int iop_twop = 0; iop_twop < 3; iop_twop++ ) {
      
        double _Complex **buffer = init_2level_ztable ( 2, T_global );
        if ( buffer == NULL ) {
          fprintf ( stderr, "[mx_prb_analyse] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
          EXIT(15);
        }

         /***************************************************************************
          * Op  tag
          ***************************************************************************/
         /* /fl_u/op_g5/d_m1/c_g5 */

        char diag[4];
        switch ( iop_twop )  {
          case 0:
          case 1:
            strcpy ( diag, "m1" );
            break;
          case 2:
            strcpy ( diag, "m2" );
            break;
        }
 
          sprintf ( key, "/fl_%s/op_%s/d_%s/c_%s", flavor_tag[0], 
              op_twop_list[iop_twop], diag, op_twop_list[iop_twop] );
          if ( g_verbose > 2 ) fprintf( stdout, "# [mx_prb_analyse] key for path1 = %s %s %d\n", key, __FILE__, __LINE__ );

          affdir = aff_reader_chpath ( affr, affn, key );
          if ( affdir == NULL ) {
            fprintf ( stderr, "[mx_prb_analyse] Error from aff_reader_chpath for key %s %s %d\n", key, __FILE__, __LINE__ );
            EXIT(1);
          }

          /***********************************************************
           * read data block from AFF file
           ***********************************************************/
          uint32_t uitems = (uint32_t)T_global;
          exitstatus = aff_node_get_complex ( affr, affdir, buffer[0], uitems );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[mx_prb_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }
#if 0
          sprintf ( key, "/fl_%s/op_%s/d_%s/c_%s", flavor_tag[1], op_twop_list[iop_twop], diag, op_twop_list[iop_twop] );
          if ( g_verbose > 2 ) fprintf( stdout, "# [mx_prb_analyse] key for path2 = %s %s %d\n", key, __FILE__, __LINE__ );

          affdir = aff_reader_chpath ( affr, affn, key );
          if ( affdir == NULL ) {
            fprintf ( stderr, "[mx_prb_analyse] Error from aff_reader_chpath for key %s %s %d\n", key, __FILE__, __LINE__ );
            EXIT(1);
          }

          /***********************************************************
           * read data block from AFF file
           ***********************************************************/
          exitstatus = aff_node_get_complex ( affr, affdir, buffer[1], uitems );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[mx_prb_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }
#endif
          for ( int it = 0; it < T_global; it++ ) {
            int const itt = ( it + gsx[0] ) % T_global;
            /* corr_twop[iop_twop][iconf][isrc][it] = buffer[0][itt] + buffer[1][itt]; */
            corr_twop[iop_twop][iconf][isrc][it] = buffer[0][itt];
          }
 
        fini_2level_ztable ( &buffer );

      }  /* end of loop on twop function correlators */

      aff_reader_close ( affr );

    }  /* end of loop on source pos */

  }  /* end of loop on configs */

  /***************************************************************************
   ***************************************************************************
   **
   ** output to file
   **
   ***************************************************************************
   ***************************************************************************/
  for ( int iop_twop = 0; iop_twop < 3; iop_twop++ ) 
  {

    char diag[4];
    switch ( iop_twop )  {
      case 0:
      case 1:
        strcpy ( diag, "m1" );
        break;
      case 2:
        strcpy ( diag, "m2" );
        break;
    }

    sprintf ( filename, "fl_%s.op_%s.d_%s.c_%s", flavor_tag[0],
        op_twop_list[iop_twop], diag, op_twop_list[iop_twop] );
    if ( g_verbose > 2 ) fprintf( stdout, "# [mx_prb_analyse] filename = %s %s %d\n", filename, __FILE__, __LINE__ );

    FILE * ofs = fopen ( filename, "w" );

    double *** buffer = init_3level_dtable ( 2, num_conf, T_global );

#pragma omp parallel for
    for( int iconf = 0; iconf < num_conf; iconf++ ) {

      for( int isrc = 0; isrc < num_src_per_conf; isrc++) {
        for ( int it = 0; it < T_global; it++ ) {
          buffer[0][iconf][it] += creal( corr_twop[iop_twop][iconf][isrc][it] );
          buffer[1][iconf][it] += cimag( corr_twop[iop_twop][iconf][isrc][it] );
        }
      }
      for ( int it = 0; it < T_global; it++ ) {
        buffer[0][iconf][it] /= (double)num_src_per_conf;
      }
      for ( int it = 0; it < T_global; it++ ) {
        buffer[1][iconf][it] /= (double)num_src_per_conf;
      }
      
      for ( int it = 0; it < T_global; it++ ) {
        fprintf( ofs, "%25.16e %25.16e %c %6d\n", 
            buffer[0][iconf][it], buffer[1][iconf][it], conf_src_list.stream[iconf], conf_src_list.conf[iconf] );
      }

    }

    fclose ( ofs );

    /***************************************************************************
     * UWerr analysis
     ***************************************************************************/
    for ( int ireim = 0; ireim <= 1; ireim++ ) {

      char obs_name[500];
      sprintf ( obs_name,  "%s.%s", filename, reim_str[ireim] );

      exitstatus = apply_uwerr_real ( buffer[ireim][0], num_conf, T_global, 0, 1, obs_name );
      if ( exitstatus != 0  ) {
        fprintf ( stderr, "[mx_prb_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(16);
      }

      /***********************************************************
       * effective mass analysis
       * only for real part
       ***********************************************************/
      if ( ireim == 0 ) {
        for ( int itau = 1; itau < T_global/2; itau++ ) {

          char obs_name2[500];
          sprintf ( obs_name2,  "%s.acosh_ratio.tau%d", obs_name, itau );

          int arg_first[3]  = { 0, 2*itau, itau };
          int arg_stride[3] = {1, 1, 1};

          exitstatus = apply_uwerr_func ( buffer[ireim][0], num_conf, T_global, T_global/2-itau, 3, arg_first, arg_stride,
              obs_name2, acosh_ratio, dacosh_ratio );
          if ( exitstatus != 0  ) {
            fprintf ( stderr, "[mx_prb_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(16);
          }
        }
      }

    }  /* end of loop on ireim */

    fini_3level_dtable ( &buffer );

  }

  /***************************************************************************
   *
   ***************************************************************************/

  for ( int iuse = 0; iuse < 3; iuse++ ) {
    char diag_str[4] = "";
    double _Complex ***** _corr_op = NULL;

    switch ( iuse ) {
      case 0:
        strcpy ( diag_str, "b+d" );
        _corr_op = corr_op;
        break;
      case 1:
        strcpy ( diag_str, "b" );
        _corr_op = corr_op_b;
        break;
      case 2:
        strcpy ( diag_str, "d" );
        _corr_op = corr_op_d;
        break;
    }


    /* for ( int iop_mx = 0; iop_mx < 4; iop_mx++ ) */
    for ( int iop_mx = 0; iop_mx < 2; iop_mx++ )
    {
  
      for ( int iop_c = 0; iop_c < 4; iop_c ++ ) {
  
        double *** buffer = init_3level_dtable ( 2, num_conf, T_global );
        if ( buffer == NULL ) {
          fprintf ( stderr, "[] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(12);
        }

#pragma omp parallel for
        for( int iconf = 0; iconf < num_conf; iconf++ ) {
  
          for( int isrc = 0; isrc < num_src_per_conf; isrc++) {
            for ( int it = 0; it < T_global; it++ ) {
              buffer[0][iconf][it] += creal ( _corr_op[iop_mx][iop_c][iconf][isrc][it] );
              buffer[1][iconf][it] += cimag ( _corr_op[iop_mx][iop_c][iconf][isrc][it] );
            }
          }
          for ( int it = 0; it < T_global; it++ ) {
            buffer[0][iconf][it] /= (double)num_src_per_conf;
          }
          for ( int it = 0; it < T_global; it++ ) {
            buffer[1][iconf][it] /= (double)num_src_per_conf;
          }
        }
   
        sprintf ( filename, "fl_%s.op_%s.d_%s.c_%s", flavor_tag[0], op_mx_list[iop_mx], diag_str, op_c_list[iop_c] );
        if ( g_verbose > 2 ) fprintf( stdout, "# [mx_prb_analyse] filename = %s %s %d\n", filename, __FILE__, __LINE__ );
  
        FILE * ofs = fopen ( filename, "w" );
          
        for( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            fprintf( ofs, "%25.16e %25.16e %c %6d\n", 
                buffer[0][iconf][it], buffer[1][iconf][it], conf_src_list.stream[iconf], conf_src_list.conf[iconf] );
          }
        }
        fclose ( ofs );
  
        /***************************************************************************
         * UWerr analysis
         ***************************************************************************/
        for ( int ireim = 0; ireim <= 1; ireim++ ) {
  
          char obs_name[500];
          sprintf ( obs_name,  "%s.%s", filename, reim_str[ireim] );
  
          exitstatus = apply_uwerr_real ( buffer[ireim][0], num_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0  ) {
            fprintf ( stderr, "[mx_prb_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(16);
          }
  
  
          /***********************************************************
           * effective mass analysis
           ***********************************************************/
          if ( ireim < 1 ) {
            for ( int itau = 1; itau < T_global/2; itau++ ) {
  
              char obs_name2[500];
              sprintf ( obs_name2,  "%s.acosh_ratio.tau%d", obs_name, itau );
  
              int arg_first[3]  = { 0, 2*itau, itau };
              int arg_stride[3] = {1, 1, 1};
  
              exitstatus = apply_uwerr_func ( buffer[ireim][0], num_conf, T_global, T_global/2-itau, 3, arg_first, arg_stride,
                  obs_name2, acosh_ratio, dacosh_ratio );
              if ( exitstatus != 0  ) {
                fprintf ( stderr, "[mx_prb_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(16);
              }
            }
          }
  
        }  /* end of loop on ireim */
  
  
  
        fini_3level_dtable ( &buffer );
  
  
      }  /* end of loop on 4q components */
    }  /* end of loop on mx operators */

  }  /* end of iuse for b+d, b and d */

  /***************************************************************************
   * ratios
   ***************************************************************************/

  for ( int iop = 0; iop <= 1; iop++ ) {

    for ( int iop_c = 0; iop_c < 4; iop_c ++ )
    {
    
      double ** buffer = init_2level_dtable ( num_conf, 2 * T_global );

#pragma omp parallel for
      for( int iconf = 0; iconf < num_conf; iconf++ ) {

        for( int isrc = 0; isrc < num_src_per_conf; isrc++) {
          for ( int it = 0; it < T_global; it++ ) {
            buffer[iconf][it] += creal ( corr_op[iop][iop_c][iconf][isrc][it] );
    
            buffer[iconf][T_global + it] += creal ( corr_twop[iop][iconf][isrc][it] );
          }
        }
        for ( int it = 0; it < 2 * T_global; it++ ) {
            buffer[iconf][it] /= (double)num_src_per_conf;
        }
      } 

      /***************************************************************************
       * UWerr analysis
       ***************************************************************************/

      char obs_name[500];
      sprintf ( obs_name, "fl_%s.op_%s.c_%s.ratio_1_1", flavor_tag[0], op_mx_list[iop], op_c_list[iop_c] );

      int arg_first[2]  = { 0, T_global};
      int arg_stride[2] = {1, 1 };

      exitstatus = apply_uwerr_func ( buffer[0], num_conf, 2*T_global, T_global, 2, arg_first, arg_stride,
          obs_name, ratio_1_1, dratio_1_1 );
      if ( exitstatus != 0  ) {
        fprintf ( stderr, "[mx_prb_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(16);
      }

      fini_2level_dtable ( &buffer );
    }

  }

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
  fini_5level_ztable ( &corr_op );
  fini_5level_ztable ( &corr_op_b );
  fini_5level_ztable ( &corr_op_d );
  fini_4level_ztable ( &corr_twop );
  fini_1level_itable ( &(conf_src_list.conf) );
  fini_3level_itable ( &(conf_src_list.src) );
  fini_1level_ctable ( &(conf_src_list.stream) );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [mx_prb_analyse] %s# [mx_prb_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [mx_prb_analyse] %s# [mx_prb_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
