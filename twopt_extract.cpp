/****************************************************
 * twopt_extract
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

#define _H5_READ_MOMENTUM_BLOCK 0
#define _H5_READ_PER_MOMENTUM   1

void usage() {
  fprintf(stdout, "Code to extract loop data\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "loop";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int Qsq = -1;
  int stream = -1;
  int confid = -1;
  int exdef_nev = -1;
  int hier_prob_D = 0;
  char twopt_type[20] = "NA";
  int nsample = 0;
  int nstep = 0;
  int sink_momentum_number = 0;

  struct timeval ta, tb;
  long unsigned int seconds, useconds;

  char output_filename[400];
  int cumulative = -1;

  char data_tag[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:Q:C:S:V:H:O:R:T:P:A:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'Q':
      Qsq = atoi ( optarg );
      break;
    case 'C':
      confid = atoi ( optarg );
      break;
    case 'S':
      stream = atoi ( optarg );
      break;
    case 'V':
      exdef_nev = atoi ( optarg );
      break;
    case 'H':
      hier_prob_D = atoi ( optarg );
      break;
    case 'O':
      strcpy ( twopt_type, optarg );
      break;
    case 'R':
      nsample = atoi ( optarg );
      break;
    case 'T':
      nstep = atoi ( optarg );
      break;
    case 'P':
      sink_momentum_number = atoi ( optarg );
      break;
    case 'A':
      cumulative = atoi ( optarg );
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
  /* fprintf(stdout, "# [twopt_extract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [twopt_extract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[twopt_extract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[twopt_extract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [twopt_extract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

#if _H5_READ_MOMENTUM_BLOCK
  /***************************************************************************
   * data filename for first source coords
   ***************************************************************************/
  sprintf ( filename, "%s/twop.%.4d_r%d_mesons_Qsq%d_SS.%.2d.%.2d.%.2d.%.2d.h5", filename_prefix, confid, stream, Qsq, 
      g_source_coords_list[0][1], g_source_coords_list[0][2], g_source_coords_list[0][3], g_source_coords_list[0][0] );

  if ( io_proc == 2 && g_verbose > 2 ) fprintf ( stdout, "# [twopt_extract] twop filename = %s\n", filename );

  /***************************************************************************
   * count momenta and build momentum list
   ***************************************************************************/
  if (io_proc == 2 && g_verbose > 1 ) fprintf ( stdout, "# [twopt_extract] number of momenta <= %3d is %3d\n", Qsq, sink_momentum_number );

  int ** sink_momentum_list = init_2level_itable ( sink_momentum_number, 3 );
  if ( sink_momentum_list == NULL ) {
    fprintf(stderr, "[twopt_extract] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }

  exitstatus = loop_get_momentum_list_from_h5_file ( sink_momentum_list, filename, sink_momentum_number, io_proc );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopt_extract] Error from loop_get_momentum_list_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  if ( g_verbose > 2 && io_proc == 2 ) {
    for ( int imom = 0; imom < sink_momentum_number; imom++ ) {
      fprintf ( stdout, " %3d  ( %3d, %3d, %3d)\n", imom, sink_momentum_list[imom][0], sink_momentum_list[imom][1], sink_momentum_list[imom][2] );
    }
  }

  /***************************************************************************
   * allocate memory for contractions
   *
   * match g_sink_momentum == -sink_momentum
   ***************************************************************************/
  int * sink_momentum_matchid = init_1level_itable ( g_sink_momentum_number );
  for ( int i = 0; i < g_sink_momentum_number; i++ ) {
    sink_momentum_matchid[i] = -1;
    for ( int k = 0; k < sink_momentum_number; k++ ) {
      if ( ( g_sink_momentum_list[i][0] == -sink_momentum_list[k][0] ) &&
           ( g_sink_momentum_list[i][1] == -sink_momentum_list[k][1] ) &&
           ( g_sink_momentum_list[i][2] == -sink_momentum_list[k][2] ) ) {
        sink_momentum_matchid[i] = k;
        break;
      }
    }
    if ( sink_momentum_matchid[i] == -1 ) {
      fprintf ( stderr, "[twopt_extract] Error, no match found for g_sink_momentum %d %s %d\n", i, __FILE__, __LINE__ );
      EXIT(1);
    } else {
      if ( g_verbose > 4 ) fprintf ( stdout, "# [twopt_extract] momentum %3d p %3d, %3d %3d  machid %3d\n", i,
          g_sink_momentum_list[i][0], g_sink_momentum_list[i][1], g_sink_momentum_list[i][2], sink_momentum_matchid[i] ); 
    }
  }
#endif  /* of _H5_READ_MOMENTUM_BLOCK */

  /***************************************************************************
   *
   ***************************************************************************/
#if _H5_READ_MOMENTUM_BLOCK
  double *** twopt = init_3level_dtable ( T, sink_momentum_number, 2 );
#elif  _H5_READ_PER_MOMENTUM
  double ** twopt = init_2level_dtable ( T, 2 );
#endif
  if ( twopt == NULL ) {
    fprintf(stderr, "[twopt_extract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }


  /***************************************************************************
   * loop on source coordinates
   ***************************************************************************/
  for ( int isrc = 0; isrc < g_source_location_number; isrc++ )
  {

    char data_filename[400];
    /* sprintf ( data_filename, "%s/twop.%.4d_r%d_mesons_Qsq%d_SS.%.2d.%.2d.%.2d.%.2d.h5", filename_prefix, confid, stream, Qsq, 
        g_source_coords_list[isrc][1], g_source_coords_list[isrc][2], g_source_coords_list[isrc][3], g_source_coords_list[isrc][0] ); */
    sprintf ( data_filename, "twop.%.4d_mesons_Qsq%d_SS.%.2d.%.2d.%.2d.%.2d.h5", confid, Qsq, 
        g_source_coords_list[isrc][1], g_source_coords_list[isrc][2], g_source_coords_list[isrc][3], g_source_coords_list[isrc][0] );

    if ( g_verbose > 2 ) fprintf ( stdout, "# [twopt_extract] loop filename = %s\n", data_filename );

    for ( int twop_meson_id = 1; twop_meson_id <= 2; twop_meson_id++ ) {

/***************************************************************************
 ***************************************************************************
 **/
#if _H5_READ_MOMENTUM_BLOCK
/**
 ***************************************************************************
 ***************************************************************************/
      sprintf ( data_tag, "/conf_%.4d/sx%.2dsy%.2dsz%.2dst%.2d/%s/twop_meson_%d" , confid, 
          g_source_coords_list[isrc][1], g_source_coords_list[isrc][2], g_source_coords_list[isrc][3], g_source_coords_list[isrc][0],
          twopt_type, twop_meson_id );

      if ( g_verbose > 2 ) fprintf( stdout, "# [twopt_extract] data_tag = %s\n", data_tag);

      exitstatus = loop_read_from_h5_file ( twopt, data_filename, data_tag, sink_momentum_number, 1, io_proc );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[twopt_extract] Error from loop_read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

        sprintf ( output_filename, "twop.%.4d.%s.%d.PX%d_PY%d_PZ%d", confid, twopt_type, twop_meson_id,
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );

        if ( g_verbose > 2 ) fprintf ( stdout, "# [twopt_extract] loop filename = %s\n", output_filename );

        FILE * ofs = isrc == 0 ? fopen ( output_filename, "w" ) : fopen ( output_filename, "a" );
        if ( ofs == NULL ) {
          fprintf ( stderr, "[twopt_extract] Error from fopen for filename %s %s %d\n", output_filename, __FILE__, __LINE__ );
          EXIT(23);
        }

        fprintf ( ofs, "# %s\n", data_tag );

        for ( int it = 0; it < T; it++ ) {
          fprintf ( ofs, "%25.16e %25.16e\n", twopt[it][sink_momentum_matchid[imom]][0], twopt[it][sink_momentum_matchid[imom]][1] );
        }

        fclose ( ofs );

      }  /* end of loop on momenta */

/***************************************************************************
 ***************************************************************************
 **/
#elif _H5_READ_PER_MOMENTUM
/**
 ***************************************************************************
 ***************************************************************************/
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      
        sprintf ( data_tag, "/conf_%.4d/sx%.2dsy%.2dsz%.2dst%.2d/%s/mom_xyz_%+d_%+d_%+d/twop_meson_%d" , confid, 
            g_source_coords_list[isrc][1], g_source_coords_list[isrc][2], g_source_coords_list[isrc][3], g_source_coords_list[isrc][0],
            twopt_type, 
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
            twop_meson_id );

        if ( g_verbose > 2 ) fprintf( stdout, "# [twopt_extract] data_tag = %s\n", data_tag);

        exitstatus = read_from_h5_file ( twopt[0], data_filename, data_tag, io_proc );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[twopt_extract] Error from loop_read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        sprintf ( output_filename, "twop.%.4d.%s.%d.PX%d_PY%d_PZ%d", confid, twopt_type, twop_meson_id,
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );

        if ( g_verbose > 2 ) fprintf ( stdout, "# [twopt_extract] loop filename = %s\n", output_filename );

        FILE * ofs = isrc == 0 ? fopen ( output_filename, "w" ) : fopen ( output_filename, "a" );
        if ( ofs == NULL ) {
          fprintf ( stderr, "[twopt_extract] Error from fopen for filename %s %s %d\n", output_filename, __FILE__, __LINE__ );
          EXIT(23);
        }

        fprintf ( ofs, "# %s\n", data_tag );

        for ( int it = 0; it < T; it++ ) {
          fprintf ( ofs, "%25.16e %25.16e\n", twopt[it][0], twopt[it][1] );
        }

        fclose ( ofs );
      }  /* end of loop on momenta */

#endif  /* _H5_READ_PER_MOMENTUM */

    }  /* end of meson ids */
  }  /* end of loop on sources */

#if _H5_READ_MOMENTUM_BLOCK
  fini_3level_dtable ( &twopt );
#elif _H5_READ_PER_MOMENTUM
  fini_2level_dtable ( &twopt );
#endif

  /*****************************************************************/
  /*****************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  free_geometry();
#if _H5_READ_MOMENTUM_BLOCK
  fini_2level_itable ( &sink_momentum_list );
#endif

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor ();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [twopt_extract] %s# [twopt_extract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [twopt_extract] %s# [twopt_extract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
