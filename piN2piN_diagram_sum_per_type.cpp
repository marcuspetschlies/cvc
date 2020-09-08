/****************************************************
 * piN2piN_diagram_sum_per_type
 * 
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM
#include <hdf5.h>

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_timer.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "table_init_i.h"
#include "table_init_z.h"
#include "table_init_2pt.h"
#include "table_init_d.h"
#include "contract_diagrams.h"
#include "zm4x4.h"
#include "gamma.h"
#include "twopoint_function_utils.h"
#include "rotations.h"
#include "group_projection.h"

using namespace cvc;


/***********************************************************
 * give reader id depending on diagram type 
 ***********************************************************/
static inline int diagram_name_to_reader_id ( char * name ) {
  char c = name[0];
  switch (c) {
    case 'm':  /* M-type  */
    case 'n':  /* N-type */
    case 'd':  /* D-type */
    case 't':  /* T-type, triangle */
    case 'b':  /* B-type  */
      return(0);
      break;
    case 'w':  /* W-type  */
      return(1);
      break;
    case 'z':  /* Z-type  */
      return(2);
      break;
    case 's':  /* S-type  */
      return(3);
      break;
    default:
      return(-1);
      break;
  }
  return(-1);
}  /* end of diagram_name_to_reader_id */

#define _V3_EQ_V3(_P,_Q) ( ( (_P)[0] == (_Q)[0] ) && ( (_P)[1] == (_Q)[1] ) && ( (_P)[2] == (_Q)[2]) )

#define _V3_NORM_SQR(_P) ( (_P)[0] * (_P)[0] + (_P)[1] * (_P)[1] + (_P)[2] * (_P)[2]  )

/***********************************************************
 * return diagram tag and number of tags
 ***********************************************************/

static inline int twopt_name_to_diagram_tag ( int * num, char * tag, const char * name ) {

  if (        strcmp( name, "N-N"       ) == 0 )  {
    *num = 1;
    tag[0] = 'N';
  } else if ( strcmp( name, "D-D"       ) == 0 )  {
    *num = 1;
    tag[0] = 'D';
  } else if ( strcmp( name, "pixN-D"    ) == 0 )  {
    *num = 1;
    tag[0] = 'T';
  } else if ( strcmp( name, "pixN-pixN" ) == 0 )  {
    *num = 4;
    tag[0] = 'B';
    tag[1] = 'W';
    tag[2] = 'Z';
    tag[3] = 'M';
  } else if ( strcmp( name, "pi-pi"     ) == 0 )  {
    *num = 1;
    tag[0] = 'P';
  } else {
    fprintf( stderr, "[twopt_name_to_diagram_tag] Error, unrecognized twopt name %s %s %d\n", name, __FILE__, __LINE__ );
    return(1);
  }
  return( 0 );
}  /* end of twopt_name_to_diagram_tag */

/***********************************************************
 ***********************************************************/

/***********************************************************
 *
 ***********************************************************/
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

/***********************************************************/
/***********************************************************/


/***********************************************************
 * combine diagrams
 ***********************************************************/
int twopt_combine_diagrams ( twopoint_function_type * const tp_sum, twopoint_function_type * const tp, int const ntp, struct AffReader_s *** affr, struct AffWriter_s * affw ) {

  int const ndiag = tp_sum->n;
  int const io_true = ( g_verbose > 2 );

  struct timeval ta, tb;
  int exitstatus;

  /******************************************************
   * loop on diagram in twopt
   ******************************************************/
  for ( int idiag = 0; idiag < ndiag; idiag++ ) {

    char diagram_name[10];
    twopoint_function_get_diagram_name ( diagram_name, tp_sum, idiag );

    int const affr_diag_id = diagram_name_to_reader_id ( diagram_name );
    if ( affr_diag_id == -1 ) {
      fprintf ( stderr, "[twopt_combine_diagrams] Error from diagram_name_to_reader_id for diagram name %s %s %d\n", diagram_name, __FILE__, __LINE__ );
      return(127);
    }

    /******************************************************
     * loop on source locations
     ******************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    for( int isrc = 0; isrc < ntp; isrc++) {

      /******************************************************
       * read the twopoint function diagram items
       *
       * get which aff reader from diagram name
       ******************************************************/
      char key[500];
      char key_suffix[400];
      unsigned int const nc = tp_sum->d * tp_sum->d * tp_sum->T;

      exitstatus = contract_diagram_key_suffix_from_type ( key_suffix, &(tp[isrc]) );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[twopt_combine_diagrams] Error from contract_diagram_key_suffix_from_type, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        return(1);
      }

      sprintf( key, "/%s/%s/%s/%s", tp[isrc].name, diagram_name, tp[isrc].fbwd, key_suffix );
      if ( g_verbose > 3 ) fprintf ( stdout, "# [twopt_combine_diagrams] key = %s %s %d\n", key, __FILE__, __LINE__ );

      exitstatus = read_aff_contraction ( tp[isrc].c[idiag][0][0], affr[affr_diag_id][isrc], NULL, key, nc );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[twopt_combine_diagrams] Error from read_aff_contraction for key %s, status was %d %s %d\n", key, exitstatus, __FILE__, __LINE__ );
        return(129);
      }

    }  /* end of loop on source locations */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "twopt_combine_diagrams", "read-diagram-all-src", io_true  );

  }  /* end of loop on diagrams */

  /******************************************************
   * average over source locations
   ******************************************************/
  for( int isrc = 0; isrc < ntp; isrc++) {

    double const norm = 1. / (double)ntp;
#pragma omp parallel for
    for ( int i = 0; i < tp_sum->n * tp_sum->d * tp_sum->d * tp_sum->T; i++ ) {
      tp_sum->c[0][0][0][i] += tp[isrc].c[0][0][0][i] * norm;
    }
  }

  /******************************************************
   * apply diagram norm
   ******************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  if ( ( exitstatus = twopoint_function_apply_diagram_norm ( tp_sum ) ) != 0 ) {
    fprintf ( stderr, "[twopt_combine_diagrams] Error from twopoint_function_apply_diagram_norm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(213);
  }

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "twopt_combine_diagrams", "twopoint_function_apply_diagram_norm", io_true  );

  /******************************************************
   * add up diagrams
   ******************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  if ( ( exitstatus = twopoint_function_accum_diagrams ( tp_sum->c[0], tp_sum ) ) != 0 ) {
    fprintf ( stderr, "[twopt_combine_diagrams] Error from twopoint_function_accum_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(216);
  }

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "twopt_combine_diagrams", "twopoint_function_accum_diagrams", io_true  );

  /******************************************************
   * write to aff file
   ******************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  char key[500], key_suffix[400], diagram_list_string[60];

  /* key suffix */
  exitstatus = contract_diagram_key_suffix_from_type ( key_suffix, tp_sum );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopt_combine_diagrams] Error from contract_diagram_key_suffix_from_type, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(12);
  }
 
  make_diagram_list_string ( diagram_list_string, tp_sum );

  /* full key */
  sprintf( key, "/%s/%s/%s%s", tp_sum->name, diagram_list_string, tp_sum->fbwd, key_suffix );
  if ( g_verbose > 2 ) fprintf ( stdout, "# [twopt_combine_diagrams] key = %s %s %d\n", key, __FILE__, __LINE__ );

  unsigned int const nc = tp_sum->d * tp_sum->d * tp_sum->T;
  exitstatus = write_aff_contraction ( tp_sum->c[0][0][0], affw, NULL, key, nc);
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopt_combine_diagrams] Error from write_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(12);
  }

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "twopt_combine_diagrams", "write-key-to-file", io_true );

  return ( 0 );
}  /* end of twopt_combine_diagrams */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
 
  char const twopt_name_list[6][20] = { "N-N", "D-D","D-pixN","pixN-D", "pixN-pixN", "pi-pi" };
  int const twopt_name_number = 5;

  char const twopt_type_list[5][20] = { "b-b","b-mxb", "mxb-b" , "mxb-mxb", "m-m" };
  /* int const twopt_type_number = 3; */ /* m-m not included here */


  const int momentum_orbit_000[ 1][3] = { {0,0,0} };

  const int momentum_orbit_001[ 6][3] = { {0,0,1}, {0,0,-1}, {0,1,0}, {0,-1,0}, {1,0,0}, {-1,0,0} };

  const int momentum_orbit_110[12][3] = { {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0}, {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1}, {0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1} };

  const int momentum_orbit_111[ 8][3] = { {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1}, {-1,1,1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1} };

  const int (* momentum_orbit_list[4])[3] = { momentum_orbit_000, momentum_orbit_001, momentum_orbit_110, momentum_orbit_111 };
  int const momentum_orbit_nelem[4]   = {1, 6, 12, 8 };
  int const momentum_orbit_num        = 4;
  int const momentum_orbit_pref[4][3] = { {0,0,0}, {0,0,1}, {1,1,0}, {1,1,1} };

  int const momentum_pi2list[27][3] = { {0,0,0}, {0,0,1}, {0,0,-1}, {0,1,0}, {0,-1,0}, {1,0,0}, {-1,0,0}, {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0}, {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1}, {0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1}, {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1}, {-1,1,1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1} };

  int const frame_list[27] = { 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 };

  int const p2_cutoff = 3;


  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[200];
  char outputfilename[200];
  char tagname[200];
  FILE *ofs = NULL;

  struct timeval ta, tb;
  struct timeval start_time, end_time;

  /***********************************************************
   * initialize MPI if used
   ***********************************************************/
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  /***********************************************************
   * evaluate command line arguments
   ***********************************************************/
  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      exit(1);
      break;
    }
  }

  /***********************************************************
   * timer for total time
   ***********************************************************/
  gettimeofday ( &start_time, (struct timezone *)NULL );

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  read_input_parser(filename);

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[piN2piN_diagram_sum_per_type] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************
   * package-own initialization of MPI parameters
   ***********************************************************/
  mpi_init(argc, argv);

  /***********************************************************
   * report git version
   ***********************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [piN2piN_diagram_sum_per_type] git version = %s\n", g_gitversion);
  }

  /***********************************************************
   * set geometry
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
  geometry();

  /****************************************************
   * set cubic group single/double cover
   * rotation tables
   ****************************************************/
  rot_init_rotation_table();

  /***********************************************************
   * set io process
   ***********************************************************/
  int const io_proc = get_io_proc ();

  /******************************************************
   * check source coords list
   ******************************************************/
  for ( int i = 0; i < g_source_location_number; i++ ) {
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] +  T_global ) %  T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /******************************************************
   * total number of source locations, 
   * base x coherent
   *
   * fill a list of all source coordinates
   ******************************************************/
  int const source_location_number = g_source_location_number * g_coherent_source_number;
  int ** source_coords_list = init_2level_itable ( source_location_number, 4 );
  if( source_coords_list == NULL ) {
    fprintf ( stderr, "[] Error from init_2level_itable %s %d\n", __FILE__, __LINE__ );
    EXIT( 43 );
  }
  for ( int ib = 0; ib < g_source_location_number; ib++ ) {
    g_source_coords_list[ib][0] = ( g_source_coords_list[ib][0] +  T_global ) %  T_global;
    g_source_coords_list[ib][1] = ( g_source_coords_list[ib][1] + LX_global ) % LX_global;
    g_source_coords_list[ib][2] = ( g_source_coords_list[ib][2] + LY_global ) % LY_global;
    g_source_coords_list[ib][3] = ( g_source_coords_list[ib][3] + LZ_global ) % LZ_global;

    int const t_base = g_source_coords_list[ib][0];
    
    for ( int ic = 0; ic < g_coherent_source_number; ic++ ) {
      int const ibc = ib * g_coherent_source_number + ic;

      int const t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * ic ) % T_global;
      source_coords_list[ibc][0] = t_coherent;
      source_coords_list[ibc][1] = ( g_source_coords_list[ib][1] + (LX_global/2) * ic ) % LX_global;
      source_coords_list[ibc][2] = ( g_source_coords_list[ib][2] + (LY_global/2) * ic ) % LY_global;
      source_coords_list[ibc][3] = ( g_source_coords_list[ib][3] + (LZ_global/2) * ic ) % LZ_global;
    }
  }

  /******************************************************
   *
   * b-b type
   *
   ******************************************************/

  /******************************************************
   * loop diagram names
   ******************************************************/
  for ( int iname = 0; iname < 2; iname++ ) {

  /******************************************************
   * check if matching 2pts are in the list
   ******************************************************/
    int twopt_id_number = 0;
    int * twopt_id_list = init_1level_itable ( g_twopoint_function_number );
    for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {
      if ( strcmp ( g_twopoint_function_list[i2pt].name , twopt_name_list[iname] ) == 0 ) {
        twopt_id_list[twopt_id_number] = i2pt;
        twopt_id_number++;
      }
    }
    if ( twopt_id_number == 0 ) {
      if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] skip twopoint name %s %s %d\n", twopt_name_list[iname], __FILE__, __LINE__ );
      continue;
    } else if ( g_verbose > 2 ) {
      fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] number of twopoint ids name %s %d  %s %d\n", twopt_name_list[iname], twopt_id_number, __FILE__, __LINE__ );
    }

    twopoint_function_type * tp = &(g_twopoint_function_list[twopt_id_list[twopt_id_number-1]]);

    gettimeofday ( &ta, (struct timezone *)NULL );

    /******************************************************
     * HDF5 readers
     * 
     * for b-b affr_diag_tag_num is 1
     ******************************************************/
    int hdf5_diag_tag_num = 0;
    char hdf5_diag_tag_list[12];

    exitstatus = twopt_name_to_diagram_tag (&hdf5_diag_tag_num, hdf5_diag_tag_list, twopt_name_list[iname] );
    if ( exitstatus != 0 ) {
      fprintf( stderr, "[piN2piN_diagram_sum_per_type] Error from twopt_name_to_diagram_tag, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(123);
    }
    else {
      for (int i=0; i < hdf5_diag_tag_num; ++i) {
        printf("Name of the twopoint function %s String of hdf5 filename %c\n",twopt_name_list[iname],hdf5_diag_tag_list[i]);
      }
    }

#ifdef HAVE_HDF5
    /***********************************************************
     * read data block from h5 file
     ***********************************************************/
    snprintf ( filename, 200, "%s%04d_sx%02dsy%02dsz%02dst%03d_%c.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         hdf5_diag_tag_list[0] );

/*           double ** buffer = init_2level_dtable ( tp->T, sink_momentum_number, tp->gamma_size*tp->gamma_size, tp->d * tp->d, 2 );*/
    int ** buffer_mom = init_2level_itable ( 27, 3 );
    if ( buffer_mom == NULL ) {
      fprintf(stderr, "[piN2piN_diagram_sum_per_type]  Error from ,init_4level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    snprintf(tagname, 200, "/sx%02dsy%02dsz%02dst%02d/mvec",source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3]);

    exitstatus = read_from_h5_file ( (void*)(buffer_mom[0]), filename, tagname, io_proc, 1 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }
    int **indextable=(int **)malloc(sizeof(int*)*4);
    for (int i_total_momentum=0; i_total_momentum < 4; ++i_total_momentum){
      indextable[i_total_momentum]=(int *)malloc(sizeof(int)*momentum_orbit_nelem[i_total_momentum]);
      for (int i_pi2=0; i_pi2 < momentum_orbit_nelem[i_total_momentum]; ++i_pi2){
        for (int j=0; j<g_sink_momentum_number; ++j){
          if ( ( momentum_orbit_list[i_total_momentum][i_pi2][0] == buffer_mom[j][0] ) &&  
               ( momentum_orbit_list[i_total_momentum][i_pi2][1] == buffer_mom[j][1] ) &&
               ( momentum_orbit_list[i_total_momentum][i_pi2][2] == buffer_mom[j][2] ) ){
            indextable[i_total_momentum][i_pi2]=j;
            break;
          }
        }
      }
    }
#endif

    /* total number of readers */
    for ( int i = 0 ; i < hdf5_diag_tag_num; i++ ) {
      double *****buffer_sum = init_5level_dtable(tp->T, g_sink_momentum_number, tp->number_of_gammas, tp->d * tp->d, 2  );
      for (int time_extent = 0; time_extent < tp->T ; ++ time_extent ){
        for (int momentum_number=0; momentum_number < g_sink_momentum_number; ++momentum_number){
          for (int spin_structures=0; spin_structures < tp->number_of_gammas; ++spin_structures ){
            for (int spin_inner=0; spin_inner < tp->d*tp->d; ++spin_inner) {
              for (int realimag=0; realimag < 2; ++realimag){
                buffer_sum[time_extent][momentum_number][spin_structures][spin_inner][realimag]=0.;
              }
            }
          }
        }
      }
      for ( int k = 0; k < source_location_number; k++ ) {
        double *****buffer_source = init_5level_dtable(tp->T, g_sink_momentum_number, tp->number_of_gammas, tp->d * tp->d, 2  );

        //Diagramm0000_sx07sy08sz14st002_D.h5
        snprintf ( filename, 200, "%s%04d_sx%02dsy%02dsz%02dst%03d_%c.h5", 
                         filename_prefix, 
                         Nconf,
		         source_coords_list[k][0], 
                         source_coords_list[k][1], 
                         source_coords_list[k][2], 
                         source_coords_list[k][3],
                         hdf5_diag_tag_list[i] );
        snprintf ( tagname, 200, "/sx%02dsy%02dsz%02dst%02d/%c",source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         hdf5_diag_tag_list[i]);
        exitstatus = read_from_h5_file ( (void*)(buffer_source[0][0][0][0]), filename, tagname, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(12);
        }
        for (int time_extent = 0; time_extent < tp->T ; ++ time_extent ){
          for (int momentum_number=0; momentum_number < g_sink_momentum_number; ++momentum_number){
            for (int spin_structures=0; spin_structures < tp->number_of_gammas; ++spin_structures ){
              for (int spin_inner=0; spin_inner < tp->d*tp->d; ++spin_inner) {
                for (int realimag=0; realimag < 2; ++realimag){
                  buffer_sum[time_extent][momentum_number][spin_structures][spin_inner][realimag]+=buffer_source[time_extent][momentum_number][spin_structures][spin_inner][realimag];
                }
              }
            }
          }
        }
        fini_5level_dtable(&buffer_source);
      }

      /******************************************************
       * loop on total momentum / frames
       ******************************************************/
      for ( int i_total_momentum = 0; i_total_momentum < 4; i_total_momentum++) {

        snprintf ( filename, 200, "%s%04d_PX%02dPY%02dPZ%02d_%c.h5",
                         filename_prefix,
                         Nconf,
                         momentum_orbit_pref[i_total_momentum][0],
                         momentum_orbit_pref[i_total_momentum][1],
                         momentum_orbit_pref[i_total_momentum][2],
                         hdf5_diag_tag_list[i] );
        fprintf ( stdout, "# [test_hdf5] create new file\n" );

        hid_t file_id, group_id, dataset_id, dataspace_id;  /* identifiers */
        herr_t      status;


        file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        for (int i_pi2=0; i_pi2 < momentum_orbit_nelem[i_total_momentum]; ++i_pi2){

          snprintf ( tagname, 200, "/pf1x%02dpf1y%02dpf1z%02d",    momentum_orbit_list[i_total_momentum][i_pi2][0],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][1],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][2]);
          /* Create a group named "/MyGroup" in the file. */
          group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

          /* Close the group. */
          status = H5Gclose(group_id);

          hsize_t dims[4];
          dims[0]=tp->T;
          dims[1]=tp->number_of_gammas;
          dims[2]=tp->d*tp->d;
          dims[3]=2;
          dataspace_id = H5Screate_simple(4, dims, NULL);



          snprintf ( tagname, 200, "/pf1x%02dpf1y%02dpf1z%02d/%c", momentum_orbit_list[i_total_momentum][i_pi2][0],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][1],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][2],
                                                                   hdf5_diag_tag_list[i]);

          /* Create a dataset in group "MyGroup". */
          dataset_id = H5Dcreate2(file_id, tagname, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

          double ****buffer_write= init_4level_dtable(tp->T,tp->number_of_gammas,tp->d*tp->d,2);
          for (int time_extent = 0; time_extent < tp->T ; ++ time_extent ){ 
            for (int spin_structures=0; spin_structures < tp->number_of_gammas; ++spin_structures ){
              for (int spin_inner=0; spin_inner < tp->d*tp->d; ++spin_inner) {
                for (int realimag=0; realimag < 2; ++realimag){
                  buffer_write[time_extent][spin_structures][spin_inner][realimag]=buffer_sum[time_extent][indextable[i_total_momentum][i_pi2]][spin_structures][spin_inner][realimag]/(double)g_source_location_number;
                }
              }
            }
          }
        


          /* Write the first dataset. */
          status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(buffer_write[0][0][0][0]));

          /* Close the data space for the first dataset. */
          status = H5Sclose(dataspace_id);

          /* Close the first dataset. */
          status = H5Dclose(dataset_id);

          fini_4level_dtable(&buffer_write);

        }

      }
      fini_5level_dtable(&buffer_sum);
    }//hdf5 names
    for (int i_total_momentum=0; i_total_momentum<4; ++i_total_momentum)
      free(indextable[i_total_momentum]);
    free(indextable);

  }  /* end of loop on twopoint function names */
    

#if 0
      gettimeofday ( &ta, (struct timezone *)NULL );

      sprintf ( filename, "%s.%.4d.PX%d_PY%d_PZ%d.aff", twopt_name_list[iname], Nconf,
          g_total_momentum_list[ipref][0], g_total_momentum_list[ipref][1], g_total_momentum_list[ipref][2] );
      
      struct AffWriter_s * affw = aff_writer(filename);
      if ( const char * aff_status_str = aff_writer_errstr ( affw ) ) {
        fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(48);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "open-writer", io_proc == 2 );

      /******************************************************/
      /******************************************************/

      /******************************************************
       * loop on 2-point functions
       ******************************************************/
      for ( int i2pt = 0; i2pt < twopt_id_number; i2pt++ ) {

        gettimeofday ( &ta, (struct timezone *)NULL );

        /* if ( strcmp ( g_twopoint_function_list[i2pt].name , twopt_name_list[iname] ) != 0 ) {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] skip twopoint %6d %s %d\n", i2pt, __FILE__, __LINE__ );
          continue;
        } */

        int const twopt_id = twopt_id_list[i2pt];

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "check-valid-twopt", io_proc == 2 );

        if ( g_verbose > 4 ) {
          gettimeofday ( &ta, (struct timezone *)NULL );
          /******************************************************
           * print the 2-point function parameters
           ******************************************************/
          sprintf ( filename, "twopoint_function_%d.show", twopt_id  );
          if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
            fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from fopen %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
          twopoint_function_print ( &(g_twopoint_function_list[twopt_id]), "TWPT", ofs );
          fclose ( ofs );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "print-twopoint", io_proc == 2 );
        }  /* end of if g_verbose */

        
        /******************************************************
         * loop on pf1
         ******************************************************/
        for ( int ipf1 = 0; ipf1 < momentum_orbit_nelem[iorbit]; ipf1++ ) {

          int const pf1[3] = {
            momentum_orbit_list[iorbit][ipf1][0],
            momentum_orbit_list[iorbit][ipf1][1],
            momentum_orbit_list[iorbit][ipf1][2] };

          int const pi1[3] = { -pf1[0], -pf1[1], -pf1[2] };

          /******************************************************
           * allocate tp_sum
           ******************************************************/
          gettimeofday ( &ta, (struct timezone *)NULL );
  
          twopoint_function_type tp_sum;
          twopoint_function_type * tp = init_1level_2pttable ( source_location_number );

          twopoint_function_init ( &tp_sum );
          twopoint_function_copy ( &tp_sum, &( g_twopoint_function_list[twopt_id]), 0 );
          
          if ( twopoint_function_allocate ( &tp_sum) == NULL ) {
            fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
            EXIT(131);
          }
          memcpy ( tp_sum.pi1 , pi1, 3 * sizeof( int ) );
          memcpy ( tp_sum.pf1 , pf1, 3 * sizeof( int ) );
  
          for ( int i = 0; i < source_location_number; i++ ) {
            twopoint_function_init ( &(tp[i]) );
            twopoint_function_copy ( &(tp[i]), &( g_twopoint_function_list[twopt_id]), 0 );
            if ( twopoint_function_allocate ( &(tp[i]) ) == NULL ) {
              fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
              EXIT(125);
            }
            memcpy ( tp[i].source_coords , source_coords_list[i], 4 * sizeof ( int ) );
            memcpy ( tp[i].pi1 , pi1, 3 * sizeof( int ) );
            memcpy ( tp[i].pf1 , pf1, 3 * sizeof( int ) );
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "init-copy-allocate-twopt", io_proc == 2 );

          /******************************************************
           * loop on diagram in twopt
           ******************************************************/
          exitstatus = twopt_combine_diagrams ( &tp_sum, tp , source_location_number, affr, affw );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from twopt_combine_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(12);
          }

          /******************************************************
           * deallocate tp_sum and tp list
           ******************************************************/
          for ( int i = 0; i < source_location_number; i++ ) {
            twopoint_function_fini ( &( tp[i] ) );
          }
          fini_1level_2pttable ( &tp );
          twopoint_function_fini ( &tp_sum );

        }  /* end of loop on pf1 */

      }  /* end of loop on 2-point functions */

      fini_1level_itable ( &twopt_id_list );

      /******************************************************
       * close AFF readers
       ******************************************************/
      for ( int ir = 0; ir < affr_num; ir++ ) {
        aff_reader_close ( affr[0][ir] );
      }
      free ( affr[0] );
      free ( affr );

      if ( const char * aff_status_str = aff_writer_close ( affw ) ) {
        fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(46);
      }

  /******************************************************/
  /******************************************************/

  /******************************************************
   *
   * mxb-b type
   *
   ******************************************************/

  /******************************************************
   * loop on total momentum / frames
   ******************************************************/
  for ( int ipref = 0; ipref < g_total_momentum_number; ipref++ ) {

    int iorbit = 0;
    for ( ; iorbit < momentum_orbit_num; iorbit++ ) {
      if ( _V3_EQ_V3( g_total_momentum_list[ipref] , momentum_orbit_pref[iorbit] ) ) break;
    }
    if ( iorbit == momentum_orbit_num ) {
      fprintf( stderr, "[piN2piN_diagram_sum_per_type] Error, could not find orbit %s %d\n", __FILE__, __LINE__ );
      EXIT(123);
    }

    /******************************************************
     * loop diagram names
     ******************************************************/
    for ( int iname = 2; iname <= 2; iname++ ) {

      /******************************************************
       * check if matching 2pts are in the list
       ******************************************************/
      int twopt_id_number = 0;
      int * twopt_id_list = init_1level_itable ( g_twopoint_function_number );
      for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {
        if ( strcmp ( g_twopoint_function_list[i2pt].name , twopt_name_list[iname] ) == 0 ) {
          twopt_id_list[twopt_id_number] = i2pt;
          twopt_id_number++;
        }
      }
      if ( twopt_id_number == 0 ) {
        if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] skip twopoint name %s %s %d\n", twopt_name_list[iname], __FILE__, __LINE__ );
        continue;
      } else if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] number of twopoint ids name %s %d  %s %d\n", twopt_name_list[iname], twopt_id_number, __FILE__, __LINE__ );
      }

      if( g_verbose > 4 ) fprintf( stdout, "# [piN2piN_diagram_sum_per_type] starting diagram name %s %s %d\n", twopt_name_list[iname], __FILE__, __LINE__ );

      gettimeofday ( &ta, (struct timezone *)NULL );

      /******************************************************
       * AFF readers
       *
       * open AFF files for name and pref
       * 
       * for mxb-b affr_diag_tag_num is 1
       ******************************************************/
      struct AffReader_s *** affr = NULL;
      int affr_diag_tag_num = 0;
      char affr_diag_tag_list[12];

      exitstatus = twopt_name_to_diagram_tag (&affr_diag_tag_num, affr_diag_tag_list, twopt_name_list[iname] );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[piN2piN_diagram_sum_per_type] Error from twopt_name_to_diagram_tag, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(123);
      }

      /* total number of readers */
      int const affr_num = source_location_number * affr_diag_tag_num;
      affr = (struct AffReader_s *** )malloc ( affr_diag_tag_num * sizeof ( struct AffReader_s ** )) ;
      if ( affr == NULL ) {
        fprintf( stderr, "[piN2piN_diagram_sum_per_type] Error from malloc %s %d\n", __FILE__, __LINE__ );
        EXIT(124);
      }
      affr[0] = (struct AffReader_s ** )malloc ( affr_num * sizeof ( struct AffReader_s *)) ;
      if ( affr[0] == NULL ) {
        fprintf( stderr, "[piN2piN_diagram_sum_per_type] Error from malloc %s %d\n", __FILE__, __LINE__ );
        EXIT(124);
      }
      for( int i = 1; i< affr_diag_tag_num; i++ ) {
        affr[i] = affr[i-1] + source_location_number;
      }
      for ( int i = 0 ; i < affr_diag_tag_num; i++ ) {
        for ( int k = 0; k < source_location_number; k++ ) {
          sprintf ( filename, "%s.%c.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", filename_prefix, affr_diag_tag_list[i],
              g_total_momentum_list[ipref][0], g_total_momentum_list[ipref][1], g_total_momentum_list[ipref][2],
              Nconf,
              source_coords_list[k][0], source_coords_list[k][1], source_coords_list[k][2], source_coords_list[k][3] );

          affr[i][k] = aff_reader ( filename );
          if ( const char * aff_status_str =  aff_reader_errstr ( affr[i][k] ) ) {
            fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from aff_reader for filename %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
            EXIT(45);
          } else {
            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagram_sum_per_type] opened data file %s for reading %s %d\n", filename, __FILE__, __LINE__);
          }
        }
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "open-reader-list", io_proc == 2 );

      /******************************************************
       * AFF writer
       ******************************************************/

      gettimeofday ( &ta, (struct timezone *)NULL );

      sprintf ( filename, "%s.%.4d.PX%d_PY%d_PZ%d.aff", twopt_name_list[iname], Nconf,
          g_total_momentum_list[ipref][0], g_total_momentum_list[ipref][1], g_total_momentum_list[ipref][2] );
      
      struct AffWriter_s * affw = aff_writer(filename);
      if ( const char * aff_status_str = aff_writer_errstr ( affw ) ) {
        fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(48);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "open-writer", io_proc == 2 );

      /******************************************************/
      /******************************************************/

      /******************************************************
       * loop on 2-point functions
       ******************************************************/
      for ( int i2pt = 0; i2pt < twopt_id_number; i2pt++ ) {

        gettimeofday ( &ta, (struct timezone *)NULL );

        int const twopt_id = twopt_id_list[i2pt];

        /* if ( strcmp ( g_twopoint_function_list[i2pt].name , twopt_name_list[iname] ) != 0 ) {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] skip twopoint %6d %s %d\n", i2pt, __FILE__, __LINE__ );
          continue;
        } */

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "check-valid-twopt", io_proc == 2 );

        if ( g_verbose > 4 ) {
          gettimeofday ( &ta, (struct timezone *)NULL );
          /******************************************************
           * print the 2-point function parameters
           ******************************************************/
          sprintf ( filename, "twopoint_function_%d.show", twopt_id );
          if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
            fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from fopen %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
          twopoint_function_print ( &(g_twopoint_function_list[twopt_id]), "TWPT", ofs );
          fclose ( ofs );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "print-twopoint", io_proc == 2 );
        }  /* end of if g_verbose */

        
        /******************************************************
         * loop on pf1
         ******************************************************/
        for ( int ipf1 = 0; ipf1 < momentum_orbit_nelem[iorbit]; ipf1++ ) {

          int const pf1[3] = {
            momentum_orbit_list[iorbit][ipf1][0],
            momentum_orbit_list[iorbit][ipf1][1],
            momentum_orbit_list[iorbit][ipf1][2] };

        /******************************************************
         * loop on pi2
         ******************************************************/
        for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

          int const pi2[3] = {
            g_seq_source_momentum_list[ipi2][0],
            g_seq_source_momentum_list[ipi2][1],
            g_seq_source_momentum_list[ipi2][2] };

          int const pi1[3] = {
            -pf1[0] - pi2[0],
            -pf1[1] - pi2[1],
            -pf1[2] - pi2[2] };

          if ( _V3_NORM_SQR(pi1) > p2_cutoff ) { 
            if ( g_verbose > 4 ) fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] skip pf1 = %3d %3d %3d   pi2 = %3d %3d %3d   pi1 = %3d %3d %3d for cutoff %d\n",
                pf1[0], pf1[1], pf1[2],
                pi2[0], pi2[1], pi2[2],
                pi1[0], pi1[1], pi1[2], p2_cutoff );
            continue;
          }

          /******************************************************
           * allocate tp_sum
           ******************************************************/
          gettimeofday ( &ta, (struct timezone *)NULL );
  
          twopoint_function_type tp_sum;
          twopoint_function_type * tp = init_1level_2pttable ( source_location_number );

          twopoint_function_init ( &tp_sum );
          twopoint_function_copy ( &tp_sum, &( g_twopoint_function_list[twopt_id]), 0 );
          
          if ( twopoint_function_allocate ( &tp_sum) == NULL ) {
            fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
            EXIT(131);
          }

          memcpy ( tp_sum.pi1 , pi1, 3 * sizeof( int ) );
          memcpy ( tp_sum.pi2 , pi2, 3 * sizeof( int ) );
          memcpy ( tp_sum.pf1 , pf1, 3 * sizeof( int ) );
  
          for ( int i = 0; i < source_location_number; i++ ) {
            twopoint_function_init ( &(tp[i]) );
            twopoint_function_copy ( &(tp[i]), &( g_twopoint_function_list[twopt_id]), 0 );
            if ( twopoint_function_allocate ( &(tp[i]) ) == NULL ) {
              fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
              EXIT(125);
            }
            memcpy ( tp[i].source_coords , source_coords_list[i], 4 * sizeof ( int ) );
            memcpy ( tp[i].pi1 , pi1, 3 * sizeof( int ) );
            memcpy ( tp[i].pi2 , pi2, 3 * sizeof( int ) );
            memcpy ( tp[i].pf1 , pf1, 3 * sizeof( int ) );
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "init-copy-allocate-twopt", io_proc == 2 );

          /******************************************************
           * combine diagrams
           ******************************************************/
          exitstatus = twopt_combine_diagrams ( &tp_sum, tp , source_location_number, affr, affw );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from twopt_combine_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(12);
          }

          /******************************************************
           * deallocate tp_sum and tp list
           ******************************************************/
          for ( int i = 0; i < source_location_number; i++ ) {
            twopoint_function_fini ( &( tp[i] ) );
          }
          fini_1level_2pttable ( &tp );
          twopoint_function_fini ( &tp_sum );

        }  /* end of loop on pi2 */
        }  /* end of loop on pf1 */

      }  /* end of loop on 2-point functions */

      fini_1level_itable ( &twopt_id_list );

      /******************************************************
       * close AFF readers
       ******************************************************/
      for ( int ir = 0; ir < affr_num; ir++ ) {
        aff_reader_close ( affr[0][ir] );
      }
      free ( affr[0] );
      free ( affr );

      if ( const char * aff_status_str = aff_writer_close ( affw ) ) {
        fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(46);
      }

    }  /* end of loop on twopoint function names */

  }  /* end of loop on reference moving frames */

  /******************************************************/
  /******************************************************/

  /******************************************************
   *
   * mxb-mxb type
   *
   ******************************************************/

  /******************************************************
   * loop on total momentum / frames
   ******************************************************/
  for ( int ipref = 0; ipref < g_total_momentum_number; ipref++ ) {

    int iorbit = 0;
    for ( ; iorbit < momentum_orbit_num; iorbit++ ) {
      if ( _V3_EQ_V3( g_total_momentum_list[ipref] , momentum_orbit_pref[iorbit] ) ) break;
    }
    if ( iorbit == momentum_orbit_num ) {
      fprintf( stderr, "[piN2piN_diagram_sum_per_type] Error, could not find orbit %s %d\n", __FILE__, __LINE__ );
      EXIT(123);
    }

    /******************************************************
     * loop diagram names
     ******************************************************/
    for ( int iname = 3; iname <= 3; iname++ ) {

      /******************************************************
       * check if matching 2pts are in the list
       ******************************************************/
      int twopt_id_number = 0;
      int * twopt_id_list = init_1level_itable ( g_twopoint_function_number );
      for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {
        if ( strcmp ( g_twopoint_function_list[i2pt].name , twopt_name_list[iname] ) == 0 ) {
          twopt_id_list[twopt_id_number] = i2pt;
          twopt_id_number++;
        }
      }
      if ( twopt_id_number == 0 ) {
        if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] skip twopoint name %s %s %d\n", twopt_name_list[iname], __FILE__, __LINE__ );
        continue;
      } else if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] number of twopoint ids name %s %d  %s %d\n", twopt_name_list[iname], twopt_id_number, __FILE__, __LINE__ );
      }

      gettimeofday ( &ta, (struct timezone *)NULL );

      /******************************************************
       * AFF readers
       *
       * open AFF files for name and pref
       * 
       * for mxb-b affr_diag_tag_num is 1
       ******************************************************/
      struct AffReader_s *** affr = NULL;
      int affr_diag_tag_num = 0;
      char affr_diag_tag_list[12];

      exitstatus = twopt_name_to_diagram_tag (&affr_diag_tag_num, affr_diag_tag_list, twopt_name_list[iname] );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[piN2piN_diagram_sum_per_type] Error from twopt_name_to_diagram_tag, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(123);
      }

      /* total number of readers */
      int const affr_num = source_location_number * affr_diag_tag_num;
      affr = (struct AffReader_s *** )malloc ( affr_diag_tag_num * sizeof ( struct AffReader_s ** )) ;
      if ( affr == NULL ) {
        fprintf( stderr, "[piN2piN_diagram_sum_per_type] Error from malloc %s %d\n", __FILE__, __LINE__ );
        EXIT(124);
      }
      affr[0] = (struct AffReader_s ** )malloc ( affr_num * sizeof ( struct AffReader_s *)) ;
      if ( affr[0] == NULL ) {
        fprintf( stderr, "[piN2piN_diagram_sum_per_type] Error from malloc %s %d\n", __FILE__, __LINE__ );
        EXIT(124);
      }
      for( int i = 1; i< affr_diag_tag_num; i++ ) {
        affr[i] = affr[i-1] + source_location_number;
      }
      for ( int i = 0 ; i < affr_diag_tag_num; i++ ) {
        for ( int k = 0; k < source_location_number; k++ ) {
          sprintf ( filename, "%s.%c.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", filename_prefix, affr_diag_tag_list[i],
              g_total_momentum_list[ipref][0], g_total_momentum_list[ipref][1], g_total_momentum_list[ipref][2],
              Nconf,
              source_coords_list[k][0], source_coords_list[k][1], source_coords_list[k][2], source_coords_list[k][3] );

          affr[i][k] = aff_reader ( filename );
          if ( const char * aff_status_str =  aff_reader_errstr ( affr[i][k] ) ) {
            fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from aff_reader for filename %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
            EXIT(45);
          } else {
            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagram_sum_per_type] opened data file %s for reading %s %d\n", filename, __FILE__, __LINE__);
          }
        }
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "open-reader-list", io_proc == 2 );

      /******************************************************
       * AFF writer
       ******************************************************/

      gettimeofday ( &ta, (struct timezone *)NULL );

      sprintf ( filename, "%s.%.4d.PX%d_PY%d_PZ%d.aff", twopt_name_list[iname], Nconf,
          g_total_momentum_list[ipref][0], g_total_momentum_list[ipref][1], g_total_momentum_list[ipref][2] );
      
      struct AffWriter_s * affw = aff_writer(filename);
      if ( const char * aff_status_str = aff_writer_errstr ( affw ) ) {
        fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(48);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "open-writer", io_proc == 2 );

      /******************************************************/
      /******************************************************/

      /******************************************************
       * loop on 2-point functions
       ******************************************************/
      for ( int i2pt = 0; i2pt < twopt_id_number; i2pt++ ) {

        gettimeofday ( &ta, (struct timezone *)NULL );

        int const twopt_id = twopt_id_list[i2pt];

        /* if ( strcmp ( g_twopoint_function_list[i2pt].name , twopt_name_list[iname] ) != 0 ) {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] skip twopoint %6d %s %d\n", i2pt, __FILE__, __LINE__ );
          continue;
        } */

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "check-valid-twopt", io_proc == 2 );

        if ( g_verbose > 4 ) {
          gettimeofday ( &ta, (struct timezone *)NULL );
          /******************************************************
           * print the 2-point function parameters
           ******************************************************/
          sprintf ( filename, "twopoint_function_%d.show", twopt_id );
          if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
            fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from fopen %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
          twopoint_function_print ( &(g_twopoint_function_list[twopt_id]), "TWPT", ofs );
          fclose ( ofs );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "print-twopoint", io_proc == 2 );
        }  /* end of if g_verbose */

        
        /******************************************************
         * loop on pf1
         ******************************************************/
        for ( int iptot = 0; iptot < momentum_orbit_nelem[iorbit]; iptot++ ) {

          int const ptot[3] = {
            momentum_orbit_list[iorbit][iptot][0],
            momentum_orbit_list[iorbit][iptot][1],
            momentum_orbit_list[iorbit][iptot][2] };

        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

          int const pf1[3] = {
            g_sink_momentum_list[ipf1][0],
            g_sink_momentum_list[ipf1][1],
            g_sink_momentum_list[ipf1][2] };

          int const pf2[3] = {
            ptot[0] - pf1[0],
            ptot[1] - pf1[1],
            ptot[2] - pf1[2] };

          if ( _V3_NORM_SQR(pf2) > p2_cutoff ) continue;

        /******************************************************
         * loop on pi2
         ******************************************************/
        for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

          int const pi2[3] = {
            g_seq_source_momentum_list[ipi2][0],
            g_seq_source_momentum_list[ipi2][1],
            g_seq_source_momentum_list[ipi2][2] };

          int const pi1[3] = {
            -ptot[0] - pi2[0],
            -ptot[1] - pi2[1],
            -ptot[2] - pi2[2] };

          if ( _V3_NORM_SQR(pi1) > p2_cutoff ) continue;

          /******************************************************
           * allocate tp_sum
           ******************************************************/
          gettimeofday ( &ta, (struct timezone *)NULL );
  
          twopoint_function_type tp_sum;
          twopoint_function_type * tp = init_1level_2pttable ( source_location_number );

          twopoint_function_init ( &tp_sum );
          twopoint_function_copy ( &tp_sum, &( g_twopoint_function_list[twopt_id]), 0 );
          
          if ( twopoint_function_allocate ( &tp_sum) == NULL ) {
            fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
            EXIT(131);
          }

          memcpy ( tp_sum.pi1 , pi1, 3 * sizeof( int ) );
          memcpy ( tp_sum.pi2 , pi2, 3 * sizeof( int ) );
          memcpy ( tp_sum.pf1 , pf1, 3 * sizeof( int ) );
          memcpy ( tp_sum.pf2 , pf2, 3 * sizeof( int ) );
  
          for ( int i = 0; i < source_location_number; i++ ) {
            twopoint_function_init ( &(tp[i]) );
            twopoint_function_copy ( &(tp[i]), &( g_twopoint_function_list[twopt_id]), 0 );
            if ( twopoint_function_allocate ( &(tp[i]) ) == NULL ) {
              fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
              EXIT(125);
            }
            memcpy ( tp[i].source_coords , source_coords_list[i], 4 * sizeof ( int ) );
            memcpy ( tp[i].pi1 , pi1, 3 * sizeof( int ) );
            memcpy ( tp[i].pi2 , pi2, 3 * sizeof( int ) );
            memcpy ( tp[i].pf1 , pf1, 3 * sizeof( int ) );
            memcpy ( tp[i].pf2 , pf2, 3 * sizeof( int ) );
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_diagram_sum_per_type", "init-copy-allocate-twopt", io_proc == 2 );

          /******************************************************
           * combine diagrams
           ******************************************************/
          exitstatus = twopt_combine_diagrams ( &tp_sum, tp , source_location_number, affr, affw );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[piN2piN_diagram_sum_per_type] Error from twopt_combine_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(12);
          }

          /******************************************************
           * deallocate tp_sum and tp list
           ******************************************************/
          for ( int i = 0; i < source_location_number; i++ ) {
            twopoint_function_fini ( &( tp[i] ) );
          }
          fini_1level_2pttable ( &tp );
          twopoint_function_fini ( &tp_sum );

        }  /* end of loop on pi2 */
        }  /* end of loop on pf1 */
        }  /* end of loop on ptot */

      }  /* end of loop on 2-point functions */

      fini_1level_itable ( &twopt_id_list );

      /******************************************************
       * close AFF readers
       ******************************************************/
      for ( int ir = 0; ir < affr_num; ir++ ) {
        aff_reader_close ( affr[0][ir] );
      }
      free ( affr[0] );
      free ( affr );

      if ( const char * aff_status_str = aff_writer_close ( affw ) ) {
        fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(46);
      }

    }  /* end of loop on twopoint function names */

  }  /* end of loop on reference moving frames */
#endif

  /******************************************************/
  /******************************************************/

  /******************************************************
   * finalize
   *
   * free the allocated memory, finalize
   ******************************************************/
  free_geometry();
  fini_2level_itable ( &source_coords_list  );

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "piN2piN_diagram_sum_per_type", "total-time", io_proc == 2 );

  
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_diagram_sum_per_type] %s# [piN2piN_diagram_sum_per_type] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_diagram_sum_per_type] %s# [piN2piN_diagram_sum_per_type] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
