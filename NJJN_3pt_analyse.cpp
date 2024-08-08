/****************************************************
 * NJJN_3pt_analyse
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

#undef _RAT
#undef _RAT_SUB

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

  char const gamma_id_to_group[16][3] = {
    "vv", "vv", "vv", "vv", 
    "ss", "pp", 
    "aa", "aa", "aa", "aa", 
    "tt", "tt", "tt", "tt", "tt", "tt" 
  };

/***************************************************************************
 *
 ***************************************************************************/
void make_key_string ( char * key, twopoint_function_type *tp, const char * type, const char * diagram_name , int const isample, int const tseq ) {

  const char * tp_type = ( type ==  NULL ) ? tp->type : type;

  if ( strcmp ( tp_type , "N-qbGqqbGq-N" ) == 0 ) {

    /* /n-dbGddbGd-n/nsample1/Gc_ss/tseq14/Gf_Cg5/Gi_Cg5/b  */
    if (    strcmp ( tp->name, "n-dbGddbGd-n") == 0  
         || strcmp ( tp->name, "p-ubGuubGu-p") == 0
         || strcmp ( tp->name, "p-dbGddbGd-p") == 0  
         || strcmp ( tp->name, "n-ubGuubGu-n") == 0 ) {
  
      sprintf ( key, "/%s/nsample%d/Gc_%s/tseq%d/Gf_%s/Gi_%s/%s", tp->name, isample,
          gamma_id_to_group[tp->gf2],
          tseq,
          gamma_id_to_Cg_ascii[tp->gf1[0]],
          gamma_id_to_Cg_ascii[tp->gi1[0]],
          diagram_name );

    }
  }

}  /* end of make_key_string */

/***************************************************************************/
/***************************************************************************/

/***************************************************************************
 *
 ***************************************************************************/
void make_correlator_string ( char * name , twopoint_function_type * tp , const char * type, int * const ptseq ) {

  const char * tp_type = ( type ==  NULL ) ? tp->type : type;

  char tseq_str[8] = "";
  if ( ptseq != NULL ) {
    sprintf ( tseq_str, "tseq%d.", *ptseq );
  }

  if ( strcmp ( tp_type, "N-qbGqqbGq-N" ) == 0 ) {

    if (    strcmp ( tp->name, "n-dbGddbGd-n") == 0
         || strcmp ( tp->name, "p-ubGuubGu-p") == 0
         || strcmp ( tp->name, "p-dbGddbGd-p") == 0
         || strcmp ( tp->name, "n-ubGuubGu-n") == 0 ) {

      sprintf ( name, "%s.Gc_%s.%sGf_%s.Gi_%s", tp->name,
          gamma_id_to_group[tp->gf2],
          tseq_str,
          gamma_id_to_Cg_ascii[tp->gf1[0]],
          gamma_id_to_Cg_ascii[tp->gi1[0]]);
    }
  }

}  /* end of make_correlator_string */

/***************************************************************************/
/***************************************************************************/

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

/***************************************************************************/
/***************************************************************************/

/***************************************************************************
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
      fprintf ( stdout, "# [NJJN_3pt_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [NJJN_3pt_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [NJJN_3pt_analyse] ensemble name set to = %s\n", ensemble_name );
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
  /* fprintf(stdout, "# [NJJN_3pt_analyse] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [NJJN_3pt_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [NJJN_3pt_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [NJJN_3pt_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[NJJN_3pt_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[NJJN_3pt_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[NJJN_3pt_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [NJJN_3pt_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[NJJN_3pt_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  conf_src_list_type conf_src_list;


  conf_src_list.conf = init_1level_itable ( num_conf );
  if ( conf_src_list.conf == NULL ) {
    fprintf(stderr, "[NJJN_3pt_analyse] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.src = init_3level_itable ( num_conf, num_src_per_conf, 4 );
  if ( conf_src_list.src == NULL ) {
    fprintf(stderr, "[NJJN_3pt_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.stream = init_1level_ctable ( num_conf );
  if ( conf_src_list.stream == NULL ) {
    fprintf(stderr, "[NJJN_3pt_analyse] Error from init_1level_ctable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.nc = num_conf;
  conf_src_list.ns = num_src_per_conf;

  char line[100];
  int countc = -1, counts=0;
  int conf_prev = -1;
  char stream_prev = '0';


  while ( fgets ( line, 100, ofs) != NULL && countc < num_conf && counts <= num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [NJJN_3pt_analyse] comment %s\n", line );
      continue;
    }

    if( g_verbose > 4 ) fprintf ( stdout, "# [NJJN_3pt_analyse] line = \"%s\"\n", line );

    int conf_tmp, src_tmp[4];
    char stream_tmp;

    /***********************************************************
     * stream conf t x y z
     ***********************************************************/
    sscanf( line, "%c %d %d %d %d %d", &stream_tmp, &conf_tmp, src_tmp, src_tmp+1, src_tmp+2, src_tmp+3 );

    if ( g_verbose > 5 ) fprintf ( stdout, "# [NJJN_3pt_analyse] before: conf_tmp = %4d   conf_prev = %4d   countc = %d   counts = %d\n", conf_tmp, conf_prev, countc, counts );

    if ( conf_tmp != conf_prev || stream_tmp != stream_prev ) {
      /* new config */
      countc++;
      counts=0;
      conf_prev = conf_tmp;
      stream_prev = stream_tmp;

      conf_src_list.stream[countc] = stream_tmp;
      conf_src_list.conf[countc]   = conf_tmp;
    }

    if ( g_verbose > 5 ) fprintf ( stdout, "# [NJJN_3pt_analyse] after : conf_tmp = %4d   conf_prev = %4d   countc = %d   counts = %d\n", conf_tmp, conf_prev, countc, counts );

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
    fprintf ( stdout, "# [NJJN_3pt_analyse] conf_src_list conf t x y z\n" );
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

  double _Complex ***** corr = init_5level_ztable ( g_twopoint_function_number, g_sequential_source_timeslice_number, num_conf, num_src_per_conf, T_global  );
  if ( corr == NULL ) {
    fprintf ( stderr, "[NJJN_3pt_analyse] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
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
            
      sprintf ( data_filename, "%s/stream_%c/%d/%s.%.4d.t%dx%dy%dz%d.aff", filename_prefix2, conf_src_list.stream[iconf], conf_src_list.conf[iconf], 
          filename_prefix, conf_src_list.conf[iconf], gsx[0], gsx[1], gsx[2], gsx[3] );
      /* sprintf ( data_filename, "%s/stream_%c/%s.%.4d.t%dx%dy%dz%d.aff", filename_prefix2, conf_src_list.stream[iconf], filename_prefix, conf_src_list.conf[iconf], gsx[0], gsx[1], gsx[2], gsx[3] ); */
      if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [NJJN_3pt_analyse] data_filename   = %s\n", data_filename );
      }

      struct AffReader_s *affr = NULL;
      struct AffNode_s *affn = NULL;
      char key[400];

      affr = aff_reader ( data_filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[NJJN_3pt_analyse] Error from aff_reader for filename %s, status was %s %s %d\n", 
            data_filename, aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[NJJN_3pt_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }

      /***************************************************************************
       * loop on twopoint functions
       ***************************************************************************/
      for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {

        /***************************************************************************
         * loop on sink momenta
         ***************************************************************************/
        for ( int itseq = 0; itseq < g_sequential_source_timeslice_number; itseq++ ) {

          int const tseq = g_sequential_source_timeslice_list[ itseq ];

          twopoint_function_type * tp = &(g_twopoint_function_list[i_2pt]);

          twopoint_function_allocate ( tp );

          if ( g_verbose > 2 ) {
            twopoint_function_print ( tp, "tp", stdout );
          }

          twopoint_function_type tp_aux;
          twopoint_function_copy ( &tp_aux, tp, 0 );

          memcpy ( tp->source_coords, gsx , 4 * sizeof( int ) );

          /***********************************************************
           * loop on diagrams
           ***********************************************************/
          for ( int i_diag = 0; i_diag < tp->n; i_diag++ ) {

            char diagram_name[500];

            twopoint_function_get_diagram_name ( diagram_name,  tp, i_diag );

            if ( g_verbose > 2 ) {
              fprintf ( stdout, "# [NJJN_3pt_analyse] diagram_name = %s\n", diagram_name );
            }

            double _Complex *** zbuffer = init_3level_ztable ( tp->T, tp->d, tp->d );


            for ( int isample = g_sourceid; isample <=g_sourceid2; isample += g_sourceid_step )
            {

              make_key_string ( key, &tp_aux, tp_aux.type, diagram_name, isample, tseq );

              if ( g_verbose > 2 ) {
               fprintf ( stdout, "# [NJJN_3pt_analyse] key = %s\n", key );
              }


              /***********************************************************
               * read data block from AFF file
               ***********************************************************/
              exitstatus = read_aff_contraction ( zbuffer[0][0] , NULL, data_filename, key, tp->d * tp->d * tp->T );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[NJJN_3pt_analyse] Error from form read_aff_contraction for file %s key %s, status was %d %s %d\n", 
                    data_filename, key, exitstatus, __FILE__, __LINE__ );
                EXIT(12);
              }

              for ( int i = 0; i < tp->d*tp->d*tp->T ; i++ ) {
                tp->c[i_diag][0][0][i] += zbuffer[0][0][i];
              }

            }  /* end of loop on samples */

            /* normalize sample average */
            for ( int i = 0; i < tp->d*tp->d*tp->T ; i++ ) {
              tp->c[i_diag][0][0][i] /= (double) ( ( g_sourceid2 - g_sourceid ) / g_sourceid_step + 1 );
            }

            fini_3level_ztable ( &zbuffer );

          }  /* end of loop on diagrams */

          if ( g_verbose > 2 ) {
            twopoint_function_show_data ( tp, stdout );
          }

          /***********************************************************
           * apply norm factors to diagrams
           ***********************************************************/
          if ( ( exitstatus = twopoint_function_apply_diagram_norm ( tp ) )  != 0 ) {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from twopoint_function_apply_diagram_norm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(103);
          }

          /***********************************************************
           * add up normalized diagrams to entry 0
           ***********************************************************/
          if ( ( exitstatus = twopoint_function_accum_diagrams ( tp->c[0], tp ) ) != 0 ) {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from twopoint_function_accum_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(104);
          }

          /***********************************************************
           * add boundary phase
           ***********************************************************/
          /* if ( ( exitstatus = correlator_add_baryon_boundary_phase ( tp->c[0], gsx[0], +1, tp->T ) ) != 0 ) {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from correlator_add_baryon_boundary_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(103);
          } */
          if( g_propagator_bc_type == 0 ) {
            /* multiply with phase factor*/
            if ( g_verbose > 3 ) fprintf(stdout, "# [correlator_add_baryon_boundary_phase] multiplying with boundary phase factor\n");
       
            /* assume lattice ordering */
            /* lattice timeslice = sink timeslice = tseq distance from source timeslice */
            /* int const ir = ( tseq + gsx[0] + T_global ) % T_global; */
            double _Complex const w = cexp ( I * 3. * M_PI*(double)tseq / (double)T_global  );
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
            for ( int it = 0; it < tp->T; it++ ) {
              int const itt = ( it + gsx[0] + T_global ) % T_global;
            
              /* for ( int ix = 0; ix < tp->d * tp->d; ix++ ) {
                corr[i_2pt][itseq][iconf][isrc][it] += tp->c[0][itt][0][ix] * w;
              } */
              corr[i_2pt][itseq][iconf][isrc][it] = tp->c[0][itt][0][0] * w;
            }
          }


          /***********************************************************
           * add source phase
           ***********************************************************/
          /* if ( ( exitstatus = correlator_add_source_phase ( tp->c[0], tp->pi1, &(gsx[1]), tp->T ) ) != 0 ) {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from correlator_add_source_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(104);
          } */

          /***********************************************************
           * reorder from source time forward
           ***********************************************************/
          /* if ( ( exitstatus = reorder_to_relative_time ( tp->c[0], tp->c[0], gsx[0], +1, tp->T ) ) != 0 ) {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from reorder_to_relative_time, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(104);
          } */

          /***********************************************************
           * spin matrix multiplication left and right
           ***********************************************************/
          /* if ( ( exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( tp->c[0], tp->c[0], gammaMat[tp->gf1[1]], gammaMat[tp->gi1[1]], tp->T ) ) != 0 ) {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from contract_diagram_zm4x4_field_mul_gamma_lr, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          } */

#if 0
          /***********************************************************
           * project to spin parity and trace
           ***********************************************************/

          double _Complex *** zbuffer = init_3level_ztable ( tp->T, tp->d, tp->d );
          if ( zbuffer == NULL ) {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
            EXIT(105);
          }
  
          if ( ( exitstatus = correlator_spin_parity_projection ( zbuffer, tp->c[0],  1., tp->T ) ) != 0 )
          {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from correlator_spin_parity_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }
  
        
          if ( ( exitstatus = contract_diagram_co_eq_tr_zm4x4_field ( corr[i_2pt][itseq][0][iconf][isrc], zbuffer, tp->T ) ) != 0 ) {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from contract_diagram_co_eq_tr_zm4x4_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }
  
  
          if ( ( exitstatus = correlator_spin_parity_projection ( zbuffer, tp->c[0], -1., tp->T ) ) != 0 )
          {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from correlator_spin_parity_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }
  
          if ( ( exitstatus = contract_diagram_co_eq_tr_zm4x4_field ( corr[i_2pt][itseq][1][iconf][isrc], zbuffer, tp->T ) ) != 0 ) {
            fprintf( stderr, "[NJJN_3pt_analyse] Error from contract_diagram_co_eq_tr_zm4x4_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }

          fini_3level_ztable ( &zbuffer );
#endif

          /***************************************************************************
           * free memory
           ***************************************************************************/
          twopoint_function_fini ( tp );


        }  /* end of loop on source - sink separation */

      }  /* end of loop on 2pt functions */

      aff_reader_close ( affr );

    }  /* end of loop on source locations */

  }  /* end of loop on configurations */
  
  /***************************************************************************
   * fwd, bwd average
   ***************************************************************************/
  for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {
    twopoint_function_type * tp = &( g_twopoint_function_list[i_2pt] );

    if ( tp->T == T_global && strcmp( tp->fbwd, "fbwd") == 0 )  {
      if ( g_verbose > 2 ) fprintf ( stdout, "# [NJJN_3pt_analyse] fwd / bwd average\n" );
      for ( int itseq = 0; itseq < g_sequential_source_timeslice_number; itseq++ ) {
#pragma omp parallel for
        for( int iconf = 0; iconf < num_conf; iconf++ ) {
          for( int isrc = 0; isrc < num_src_per_conf; isrc++) {
            for ( int it = 0; it <= T_global/2; it++ ) {
              int const itt = ( T_global - it ) % T_global;
              double _Complex const zp[2] = { corr[i_2pt][itseq][iconf][isrc][it] , corr[i_2pt][itseq][iconf][isrc][itt] };
              double _Complex const zm[2] = { corr[i_2pt][itseq][iconf][isrc][it] , corr[i_2pt][itseq][iconf][isrc][itt] };

              corr[i_2pt][itseq][iconf][isrc][it ] = 0.5 * ( zp[0] - zm[1] );
              corr[i_2pt][itseq][iconf][isrc][itt] = corr[i_2pt][itseq][iconf][isrc][it];

              corr[i_2pt][itseq][iconf][isrc][it ] = 0.5 * ( zm[0] - zp[1] );
              corr[i_2pt][itseq][iconf][isrc][itt] = corr[i_2pt][itseq][iconf][isrc][it];
            }
          }
        }
      }
    }
  }

  /***************************************************************************
   * UWerr analysis
   *
   *   for correlator corr
   ***************************************************************************/
  for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {
    
    twopoint_function_type * tp = &( g_twopoint_function_list[i_2pt] );

    for ( int itseq = 0; itseq < g_sequential_source_timeslice_number; itseq++ ) {

      char correlator_name[400];
      make_correlator_string ( correlator_name,  tp , NULL , &(g_sequential_source_timeslice_list[itseq]) );

      char diagram_name[100];
      make_diagram_list_string ( diagram_name, tp );


      for ( int ireim = 0; ireim <= 1; ireim++ ) 
      {

          double ** data = init_2level_dtable ( num_conf, tp->T );
          if ( data == NULL ) {
            fprintf ( stderr, "[NJJN_3pt_analyse] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(16);
          }

          /* block data over sources */
#pragma omp parallel for
          for( int iconf = 0; iconf < num_conf; iconf++ ) {
 
            for ( int it = 0; it < tp->T; it++ ) {

              data[iconf][it] = 0.;

              for( int isrc = 0; isrc < num_src_per_conf; isrc++) {
                data[iconf][it] += *(((double*)( corr[i_2pt][itseq][iconf][isrc]+it )) + ireim );
              }

              data[iconf][it] /= (double)num_src_per_conf;
            }
          }

          char obs_name[500];
        
          sprintf ( obs_name,  "%s.%s.parity%d.%s", correlator_name, diagram_name, +1, reim_str[ireim] );

          if ( num_conf < 6 ) {
            fprintf ( stderr, "[NJJN_3pt_analyse] number of observations too small, continue %s %d\n", __FILE__, __LINE__ );
          } else {
            exitstatus = apply_uwerr_real ( data[0], num_conf, tp->T, 0, 1, obs_name );
            if ( exitstatus != 0  ) {
              fprintf ( stderr, "[NJJN_3pt_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(16);
            }
          }

          /***************************************************************************
           * write data
           ***************************************************************************/
          char data_filename[600];
          sprintf ( data_filename, "%s.corr", obs_name);
          FILE * dfs = fopen( data_filename, "w" );

          for( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < tp->T; it++ ) {
              fprintf( dfs, "%4d %25.16e    %c %6d\n", it, data[iconf][it], conf_src_list.stream[iconf], conf_src_list.conf [iconf] );
            }
          }

          fclose ( dfs );
          
          fini_2level_dtable ( &data );
 
      }  /* end of loop on reim */

    }

  }  /* end of loop on 2-point functions */

  /***************************************************************************
   * UWerr analysis
   *
   *   for sum of correlator corr
   ***************************************************************************/
  for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {
    
    twopoint_function_type * tp = &( g_twopoint_function_list[i_2pt] );

      char correlator_name[400];
      make_correlator_string ( correlator_name,  tp , NULL , NULL );

      char diagram_name[100];
      make_diagram_list_string ( diagram_name, tp );


      for ( int ireim = 0; ireim <= 1; ireim++ ) 
      {

          double *** data = init_3level_dtable ( 2, num_conf, g_sequential_source_timeslice_number );
          if ( data == NULL ) {
            fprintf ( stderr, "[NJJN_3pt_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(16);
          }

          /* block data over sources */
#pragma omp parallel for
          for( int iconf = 0; iconf < num_conf; iconf++ ) {
            
            for ( int itseq = 0; itseq < g_sequential_source_timeslice_number; itseq++ ) {
              data[0][iconf][itseq] = 0.;

              for( int isrc = 0; isrc < num_src_per_conf; isrc++) {
                for ( int it = 0; it < tp->T; it++ ) {
                  data[0][iconf][itseq] += *(((double*)( corr[i_2pt][itseq][iconf][isrc]+it )) + ireim );
                }
              }
              data[0][iconf][itseq] /= (double)num_src_per_conf;

              data[1][iconf][itseq] = 0.;

              for( int isrc = 0; isrc < num_src_per_conf; isrc++) {
                for ( int it = 0; it <= g_sequential_source_timeslice_list[itseq]; it++ ) {
                  data[1][iconf][itseq] += *(((double*)( corr[i_2pt][itseq][iconf][isrc]+it )) + ireim );
                }
              }
              data[1][iconf][itseq] /= (double)num_src_per_conf;
            }
          }

          if ( num_conf < 6 ) {
            fprintf ( stderr, "[NJJN_3pt_analyse] number of observations too small, continue %s %d\n", __FILE__, __LINE__ );
          } else {
          
            char obs_name[500];
            sprintf ( obs_name,  "%s.%s.parity%d.tsum.%s", correlator_name, diagram_name, +1, reim_str[ireim] );

            exitstatus = apply_uwerr_real ( data[0][0], num_conf, g_sequential_source_timeslice_number, 0, 1, obs_name );
            if ( exitstatus != 0  ) {
              fprintf ( stderr, "[NJJN_3pt_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(16);
            }

            sprintf ( obs_name,  "%s.%s.parity%d.rsum.%s", correlator_name, diagram_name, +1, reim_str[ireim] );

            exitstatus = apply_uwerr_real ( data[1][0], num_conf, g_sequential_source_timeslice_number, 0, 1, obs_name );
            if ( exitstatus != 0  ) {
              fprintf ( stderr, "[NJJN_3pt_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(16);
            }

          }

          /***************************************************************************
           * write data
           ***************************************************************************/
          {
            char data_filename[600];
            sprintf ( data_filename, "%s.%s.parity%d.tsum.%s.corr", correlator_name, diagram_name, +1, reim_str[ireim] );
            FILE * dfs = fopen( data_filename, "w" );

            for( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int it = 0; it < g_sequential_source_timeslice_number; it++ ) {
                fprintf( dfs, "%4d %25.16e    %c %6d\n", it, data[0][iconf][it], conf_src_list.stream[iconf], conf_src_list.conf [iconf] );
              }
            }

            fclose ( dfs );

            sprintf ( data_filename, "%s.%s.parity%d.rsum.%s.corr", correlator_name, diagram_name, +1, reim_str[ireim] );
            dfs = fopen( data_filename, "w" );

            for( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int it = 0; it < g_sequential_source_timeslice_number; it++ ) {
                fprintf( dfs, "%4d %25.16e    %c %6d\n", it, data[1][iconf][it], conf_src_list.stream[iconf], conf_src_list.conf [iconf] );
              }
            }

            fclose ( dfs );
          
          }

          fini_3level_dtable ( &data );
 
      }  /* end of loop on reim */

  }  /* end of loop on 2-point functions */

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
  fini_5level_ztable ( &corr );
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
    fprintf(stdout, "# [NJJN_3pt_analyse] %s# [NJJN_3pt_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [NJJN_3pt_analyse] %s# [NJJN_3pt_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
