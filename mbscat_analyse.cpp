/****************************************************
 * mbscat_analyse
 *
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
#include "table_init_c.h"
#include "contract_diagrams.h"
#include "twopoint_function_utils.h"
#include "gamma.h"
#include "uwerr.h"
#include "derived_quantities.h"
#include "rotations.h"

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
 * replace fwd slash by dot to make output filename
 ***************************************************************************/
void slash_to_dot ( char * name ) {

  for ( ; *name != '\0';  name++ ) {
    if ( *name == '/' ) *name = '.';
  }
}

/***************************************************************************
 *  h5 data filename
 ***************************************************************************/
int make_data_filename ( char * name , const twopoint_function_type * tp, const int pref[3], const int conf_id) {
  
  sprintf ( name, "%s.%s.PX%dPY%dPZ%d.%s.%.4d.t%dx%dy%dz%d.h5", tp->name, tp->group, pref[0], pref[1], pref[2], tp->irrep, conf_id,
      tp->source_coords[0], tp->source_coords[1], tp->source_coords[2], tp->source_coords[3] );

  return(0);
}  /* end of make_data_filename  */

/***************************************************************************
 * data key / group name for reading from h5 file
 ***************************************************************************/
int make_group_string ( char * key, twopoint_function_type *tp, const char * type , const char * data_set_name , const int data_set_id ) {

  const char * tp_type = ( type ==  NULL ) ? tp->type : type;
  char tp_data_set_name[500];
  if ( data_set_name == NULL ) {
    if ( data_set_id < 0 || data_set_id >= tp->n ) {
      fprintf ( stderr, "[make_group_string] Error, data_set_id out of bounds %s %d\n", __FILE__, __LINE__ );
      return ( 1 );
    }
    twopoint_function_get_diagram_name ( tp_data_set_name,  tp, data_set_id );
  }

  if ( strcmp ( tp_type, "b-b" ) == 0 ) {

    sprintf ( key, "/%s/%s/gf1%.2d_%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/gi1%.2d_%.2d/pi1x%.2dpi1y%.2dpi1z%.2d/t%.2dx%.2dy%.2dz%.2d/%s", tp->name, tp->fbwd,
        tp->gf1[0], tp->gf1[1], tp->pf1[0], tp->pf1[1], tp->pf1[2],
        tp->gi1[0], tp->gi1[1], tp->pi1[0], tp->pi1[1], tp->pi1[2],
        tp->source_coords[0], tp->source_coords[1], tp->source_coords[2], tp->source_coords[3],
        tp_data_set_name );

  } else if ( strcmp( tp_type , "mxb-b" ) == 0 ) {

    sprintf ( key, "/%s/%s/gf1%.2d_%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/gi2%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi1%.2d_%.2d/pi1x%.2dpi1y%.2dpi1z%.2d/t%.2dx%.2dy%.2dz%.2d/%s",
        tp->name, tp->fbwd,
        tp->gf1[0], tp->gf1[1], tp->pf1[0], tp->pf1[1], tp->pf1[2],
        tp->gi2,                tp->pi2[0], tp->pi2[1], tp->pi2[2],
        tp->gi1[0], tp->gi1[1], tp->pi1[0], tp->pi1[1], tp->pi1[2],
        tp->source_coords[0], tp->source_coords[1], tp->source_coords[2], tp->source_coords[3],
        tp_data_set_name );

  } else if ( strcmp( tp_type , "mxb-mxb" ) == 0 ) {

    sprintf ( key, "/%s/%s/gf2%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/gf1%.2d_%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/gi2%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi1%.2d_%.2d/pi1x%.2dpi1y%.2dpi1z%.2d/t%.2dx%.2dy%.2dz%.2d/%s",
        tp->name, tp->fbwd,
        tp->gf2,                tp->pf2[0], tp->pf2[1], tp->pf2[2],
        tp->gf1[0], tp->gf1[1], tp->pf1[0], tp->pf1[1], tp->pf1[2],
        tp->gi2,                tp->pi2[0], tp->pi2[1], tp->pi2[2],
        tp->gi1[0], tp->gi1[1], tp->pi1[0], tp->pi1[1], tp->pi1[2],
        tp->source_coords[0], tp->source_coords[1], tp->source_coords[2], tp->source_coords[3],
        tp_data_set_name );
  }
}  /* end of make_group_string */


/***************************************************************************
 * correlator name for output file
 ***************************************************************************/
void make_correlator_string ( char * name , twopoint_function_type * tp , const char * type , const int data_set_id ) {

  const char * tp_type = ( type ==  NULL ) ? tp->type : type;

  char tp_data_set_name[500];
  twopoint_function_get_diagram_name ( tp_data_set_name,  tp, data_set_id );

  slash_to_dot ( tp_data_set_name );

  sprintf ( name, "%s.%s.%s.%s", tp->name, tp->group, tp->irrep, tp->fbwd );

  if ( strcmp ( tp_type, "b-b" ) == 0 ) {
    sprintf ( name, "%s.gf1_%s_%s.pf1x%.2dpf1y%.2dpf1z%.2d.gi1_%s_%s.pi1x%.2dpi1y%.2dpi1z%.2d", name,
        gamma_id_to_Cg_ascii[tp->gf1[0]], gamma_id_to_ascii[tp->gf1[1]], tp->pf1[0], tp->pf1[1], tp->pf1[2],
        gamma_id_to_Cg_ascii[tp->gi1[0]], gamma_id_to_ascii[tp->gi1[1]], tp->pi1[0], tp->pi1[1], tp->pi1[2] );

  } else if ( strcmp( tp_type , "mxb-b" ) == 0 ) {

    sprintf ( name, "%s.gf1_%s_%s.pf1x%.2dpf1y%.2dpf1z%.2d.gi2_%s.pi2x%.2dpi2y%.2dpi2z%.2d.gi1_%s_%s.pi1x%.2dpi1y%.2dpi1z%.2d",
        name,
        gamma_id_to_Cg_ascii[tp->gf1[0]], gamma_id_to_ascii[tp->gf1[1]], tp->pf1[0], tp->pf1[1], tp->pf1[2],
        gamma_id_to_ascii[tp->gi2],                                      tp->pi2[0], tp->pi2[1], tp->pi2[2],
        gamma_id_to_Cg_ascii[tp->gi1[0]], gamma_id_to_ascii[tp->gi1[1]], tp->pi1[0], tp->pi1[1], tp->pi1[2] );

  } else if ( strcmp( tp_type , "mxb-mxb" ) == 0 ) {

    sprintf ( name, "%s.gf2_%s.pf2x%.2dpf2y%.2dpf2z%.2d.gf1_%s_%s.pf1x%.2dpf1y%.2dpf1z%.2d.gi2_%s.pi2x%.2dpi2y%.2dpi2z%.2d.gi1_%s_%s.pi1x%.2dpi1y%.2dpi1z%.2d",
        name, 
        gamma_id_to_ascii[tp->gf2],                                      tp->pf2[0], tp->pf2[1], tp->pf2[2],
        gamma_id_to_Cg_ascii[tp->gf1[0]], gamma_id_to_ascii[tp->gf1[1]], tp->pf1[0], tp->pf1[1], tp->pf1[2],
        gamma_id_to_ascii[tp->gi2],                                      tp->pi2[0], tp->pi2[1], tp->pi2[2],
        gamma_id_to_Cg_ascii[tp->gi1[0]], gamma_id_to_ascii[tp->gi1[1]], tp->pi1[0], tp->pi1[1], tp->pi1[2] );
  }

  strcat ( name, tp_data_set_name );

}  /* end of make_correlator_string */


/***************************************************************************
 *
 * main program
 *
 ***************************************************************************/
int main(int argc, char **argv) {
 
  char const reim_str[2][3] = { "re", "im" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char ensemble_name[100] = "NA";
  char filename[600];
  int num_conf = 0, num_src_per_conf = 0;
  int write_data = 0;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:E:W:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [mbscat_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [mbscat_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [mbscat_analyse] ensemble name set to = %s\n", ensemble_name );
      break;
    case 'W':
      write_data = atoi( optarg );
      fprintf ( stdout, "# [mbscat_analyse] write_data to = %d\n", write_data );
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
  /* fprintf(stdout, "# [mbscat_analyse] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [mbscat_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [mbscat_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [mbscat_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[mbscat_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[mbscat_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[mbscat_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [mbscat_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /****************************************************
   * set cubic group single/double cover
   * rotation tables
   ****************************************************/
  rot_init_rotation_table();

  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[mbscat_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  conf_src_list_type conf_src_list;


  conf_src_list.conf = init_1level_itable ( num_conf );
  if ( conf_src_list.conf == NULL ) {
    fprintf(stderr, "[mbscat_analyse] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.src = init_3level_itable ( num_conf, num_src_per_conf, 4 );
  if ( conf_src_list.src == NULL ) {
    fprintf(stderr, "[mbscat_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.stream = init_1level_ctable ( num_conf );
  if ( conf_src_list.stream == NULL ) {
    fprintf(stderr, "[mbscat_analyse] Error from init_1level_ctable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.nc = num_conf;
  conf_src_list.ns = num_src_per_conf;

  char line[100];
  int countc = -1, counts=0;
  int conf_prev = -1;

  while ( fgets ( line, 100, ofs) != NULL && countc < num_conf && counts <= num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [mbscat_analyse] comment %s\n", line );
      continue;
    }

    if( g_verbose > 4 ) fprintf ( stdout, "# [mbscat_analyse] line = \"%s\"\n", line );

    int conf_tmp, src_tmp[4];
    char stream_tmp;

    /***********************************************************
     * stream conf t x y z
     ***********************************************************/
    sscanf( line, "%c %d %d %d %d %d", &stream_tmp, &conf_tmp, src_tmp, src_tmp+1, src_tmp+2, src_tmp+3 );

    if ( g_verbose > 5 ) fprintf ( stdout, "# [mbscat_analyse] before: conf_tmp = %4d   conf_prev = %4d   countc = %d   counts = %d\n", conf_tmp, conf_prev, countc, counts );

    if ( conf_tmp != conf_prev ) {
      /* new config */
      countc++;
      counts=0;
      conf_prev = conf_tmp;

      conf_src_list.stream[countc] = stream_tmp;
      conf_src_list.conf[countc]   = conf_tmp;
    }

    if ( g_verbose > 5 ) fprintf ( stdout, "# [mbscat_analyse] after : conf_tmp = %4d   conf_prev = %4d   countc = %d   counts = %d\n", conf_tmp, conf_prev, countc, counts );

    memcpy ( conf_src_list.src[countc][counts] , src_tmp, 4*sizeof(int) );

    counts++;

  }

  fclose ( ofs );

  /***********************************************************
   * show all configs and source locations
   ***********************************************************/
  if ( g_verbose > 4 ) {
    fprintf ( stdout, "# [mbscat_analyse] conf_src_list conf t x y z\n" );
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


  /***************************************************************************
   * loop on twopoint functions
   ***************************************************************************/
  for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

    twopoint_function_type * tp = &(g_twopoint_function_list[i2pt]);

    twopoint_function_allocate ( tp );

    if ( g_verbose > 2 ) {
      twopoint_function_print ( tp, "tp", stdout );
    }

    double _Complex **** corr = init_4level_ztable ( tp->n, num_conf, num_src_per_conf, tp->T  );
    if ( corr == NULL ) {
      fprintf ( stderr, "[mbscat_analyse] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(15);
    }

    /***************************************************************************
     * total and reference frame momentum
     ***************************************************************************/
    int Ptot[3], Pref[3], refframerot;
    if ( strcmp( tp->type, "m-m" ) == 0 ) {
      Ptot[0] = tp->pf2[0];
      Ptot[1] = tp->pf2[1];
      Ptot[2] = tp->pf2[2];

    } else if ( strcmp( tp->type, "b-b" ) == 0 || strcmp( tp->type, "mxb-b" ) == 0 ) {
      Ptot[0] = tp->pf1[0];
      Ptot[1] = tp->pf1[1];
      Ptot[2] = tp->pf1[2];

    } else if ( strcmp( tp->type, "mxb-mxb" ) == 0 || strcmp( tp->type, "mxb-b" ) == 0 ) {
      Ptot[0] = tp->pf1[0] + tp->pf2[0];
      Ptot[1] = tp->pf1[1] + tp->pf2[1];
      Ptot[2] = tp->pf1[2] + tp->pf2[2];
    }

    exitstatus = get_reference_rotation ( Pref, &refframerot, Ptot );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[mbscat_analyse] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(4);
    } else if ( g_verbose > 1 ) {
      fprintf ( stdout, "# [mbscat_analyse] twopoint_function %3d Ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n", i2pt,
      Ptot[0], Ptot[1], Ptot[2], refframerot, Pref[0], Pref[1], Pref[2]);
    }

    /***********************************************************
     * loop on configs 
     ***********************************************************/
    for( int iconf = 0; iconf < num_conf; iconf++ ) {
          
      int const Nconf = conf_src_list.conf[iconf];

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


        memcpy ( tp->source_coords, gsx , 4 * sizeof( int ) );

        char data_filename[500];
       
        exitstatus = make_data_filename ( data_filename, tp, Pref , Nconf );
     
        if ( g_verbose > 2 ) {
          fprintf ( stdout, "# [mbscat_analyse] data_filename   = %s\n", data_filename );
        }

        /***********************************************************
         * data sets inside tp
         ***********************************************************/
        for ( int idiag = 0; idiag < tp->n; idiag++ ) {

          /***********************************************************
           * key for tp
           ***********************************************************/
          char key[500];
          exitstatus = make_group_string ( key, tp, NULL , NULL, idiag );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[mbscat_analyse] Error from make_group_string for file %s group %s, status was %d %s %d\n",
                data_filename, key, exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

          if ( g_verbose > 2 ) {
            fprintf ( stdout, "# [mbscat_analyse] key = %s\n", key );
          }

          /***********************************************************
           * read data block from HDF5 file
           ***********************************************************/
          read_from_h5_file ( tp->c[idiag][0][0], data_filename, key, io_proc );
          if ( exitstatus != 0 ) {
              fprintf(stderr, "[mbscat_analyse] Error from read_aff_contraction for file %s key %s, status was %d %s %d\n", 
                  data_filename, key, exitstatus, __FILE__, __LINE__ );
              EXIT(12);
            }

          if ( g_verbose > 2 ) {
            twopoint_function_show_data ( tp, stdout );
          }

          /***********************************************************
           * apply norm factors to diagrams
           ***********************************************************/
          if ( ( exitstatus = twopoint_function_apply_diagram_norm ( tp ) )  != 0 ) {
            fprintf( stderr, "[mbscat_analyse] Error from twopoint_function_apply_diagram_norm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(103);
          }

          /***********************************************************
           * add up normalized diagrams to entry 0
           *
           * NO accumulation necessary here anymore
           ***********************************************************/
#if 0
          if ( ( exitstatus = twopoint_function_accum_diagrams ( tp->c[idiag], tp ) ) != 0 ) {
            fprintf( stderr, "[mbscat_analyse] Error from twopoint_function_accum_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(104);
          }
#endif

          /***********************************************************
           * add boundary phase
           *
           * HAS already been added
           ***********************************************************/
#if 0
          if ( ( exitstatus = correlator_add_baryon_boundary_phase ( tp->c[idiag], gsx[0], +1, tp->T ) ) != 0 ) {
            fprintf( stderr, "[mbscat_analyse] Error from correlator_add_baryon_boundary_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(103);
          }
#endif

          /***********************************************************
           * add source phase
           *
           * HAS already been added
           ***********************************************************/
#if 0
          if ( ( exitstatus = correlator_add_source_phase ( tp->c[idiag], tp->pi1, &(gsx[1]), tp->T ) ) != 0 ) {
            fprintf( stderr, "[mbscat_analyse] Error from correlator_add_source_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(104);
          }
#endif

          /***********************************************************
            * reorder from source time forward
           *
           * IS already ordered from source
           ***********************************************************/
#if 0
          if ( ( exitstatus = reorder_to_relative_time ( tp->c[idiag], tp->c[idiag], gsx[0], +1, tp->T ) ) != 0 ) {
            fprintf( stderr, "[mbscat_analyse] Error from reorder_to_relative_time, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(104);
          }
#endif

          /***********************************************************
           * spin matrix multiplication left and right
           *
           * HAS already been multiplied
           ***********************************************************/
#if 0
          if ( ( exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( tp->c[0], tp->c[0], gammaMat[tp->gf1[1]], gammaMat[tp->gi1[1]], tp->T ) ) != 0 ) {
            fprintf( stderr, "[mbscat_analyse] Error from contract_diagram_zm4x4_field_mul_gamma_lr, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }
#endif

        /***********************************************************
         * trace of 4x4
         *
         * THIS NEEDS TO BE GENERALIZED
         ***********************************************************/
          if ( ( exitstatus = contract_diagram_co_eq_tr_zm4x4_field ( corr[idiag][iconf][isrc], tp->c[0], tp->T ) ) != 0 ) {
            fprintf( stderr, "[mbscat_analyse] Error from contract_diagram_co_eq_tr_zm4x4_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }

        }  /* end of loop on data sets */
        
      }  /* end of loop on source locations */
      
    }  /* end of loop on configurations */
  
    for ( int idiag = 0; idiag < tp->n; idiag++ ) {

      char correlator_name[500];
      make_correlator_string ( correlator_name, tp, NULL , idiag );

      /***********************************************************
       * write correlator to file
       ***********************************************************/
      if ( write_data == 1 ) {
        
        sprintf ( filename, "%s.corr", correlator_name );

        FILE * ofs = fopen ( filename, "w" );
        for( int iconf = 0; iconf < num_conf; iconf++ ) {

          for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

            fprintf ( ofs, "# %s %c %6d   %3d %3d %3d %3d\n", ensemble_name, conf_src_list.stream[iconf], conf_src_list.conf[iconf],
                conf_src_list.src[iconf][isrc][0],
                conf_src_list.src[iconf][isrc][1],
                conf_src_list.src[iconf][isrc][2],
                conf_src_list.src[iconf][isrc][3] );

            for ( int it = 0; it < tp->T; it++ ) {
              fprintf ( ofs, "%3d %25.16e %25.16e\n" , it, creal( corr[idiag][iconf][isrc][it] ), cimag( corr[idiag][iconf][isrc][it] ) );
            } 

          }  /* end of loop on source locations */
        }  /* end of loop on configs */
        fclose ( ofs );

      }  /* end of if write_data */
   
      /***************************************************************************
       * UWerr analysis for correlator 
       ***************************************************************************/
      for ( int ireim = 0; ireim < 2; ireim++ ) {

        if ( num_conf < 6 ) {
          fprintf ( stderr, "[mbscat_analyse] number of observations too small, continue %s %d\n", __FILE__, __LINE__ );
          continue;
        }

        double ** data = init_2level_dtable ( num_conf, tp->T );
        if ( data == NULL ) {
          fprintf ( stderr, "[mbscat_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(16);
        }

        /* block data over sources */
#pragma omp parallel for
        for( int iconf = 0; iconf < num_conf; iconf++ ) {

          for ( int it = 0; it < tp->T; it++ ) {
            data[iconf][it] = 0.;
            for( int isrc = 0; isrc < num_src_per_conf; isrc++) {
             data[iconf][it] += *(((double*)( corr[idiag][iconf][isrc]+it )) + ireim );
            } 
            data[iconf][it] /= (double)num_src_per_conf;
          }
        }  /* end of loop on configs */

        char obs_name[500];
        sprintf ( obs_name,  "%s.%s", correlator_name, reim_str[ireim] );

        exitstatus = apply_uwerr_real ( data[0], num_conf, tp->T, 0, 1, obs_name );
        if ( exitstatus != 0  ) {
          fprintf ( stderr, "[mbscat_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(16);
        }

#if 0
        for ( int itau = 1; itau < tp->T/2; itau++ ) {
          char obs_name2[500];
  
          sprintf ( obs_name2,  "%s.log_ratio.tau%d", obs_name, itau );
  
          int arg_first[2]  = {0, itau};
          int arg_stride[2] = {1,1};
        
          exitstatus = apply_uwerr_func ( data[0], num_conf, tp->T, tp->T/2-itau, 2, arg_first, arg_stride, obs_name2, log_ratio_1_1, dlog_ratio_1_1 );
          if ( exitstatus != 0  ) {
            fprintf ( stderr, "[mbscat_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(16);
          }
        }
#endif  /* of if 0 */

        fini_2level_dtable ( &data );

      }  /* end of loop on reim */

    }

    /***************************************************************************
     * free fields
     ***************************************************************************/

    fini_4level_ztable ( &corr );
    twopoint_function_fini ( tp );

  }  /* end of loop on 2-point functions */

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
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
    fprintf(stdout, "# [mbscat_analyse] %s# [mbscat_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [mbscat_analyse] %s# [mbscat_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
