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
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>
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

/* ************************************************************************
 *
 * routine that multiplies with the external gamma5 structure at the source
 * used for gammas like C, Cg4, cg1g4g5 etc.
 * note the multiplication with the sign due to the adjoint of gamma5
 *
 * ************************************************************************/

static inline void mult_with_gamma5_matrix_source ( double ** buffer_write ) {

    double **buffer_temporary=init_2level_dtable(16,2);
                 
    buffer_temporary[0][0]=buffer_write[2][0]; buffer_temporary[0][1]=buffer_write[2][1];    
    buffer_temporary[1][0]=buffer_write[3][0]; buffer_temporary[1][1]=buffer_write[3][1];
    buffer_temporary[2][0]=buffer_write[0][0]; buffer_temporary[2][1]=buffer_write[0][1];
    buffer_temporary[3][0]=buffer_write[1][0]; buffer_temporary[3][1]=buffer_write[1][1];

    buffer_temporary[4][0]=buffer_write[6][0]; buffer_temporary[4][1]=buffer_write[6][1];
    buffer_temporary[5][0]=buffer_write[7][0]; buffer_temporary[5][1]=buffer_write[7][1];
    buffer_temporary[6][0]=buffer_write[4][0]; buffer_temporary[6][1]=buffer_write[4][1];
    buffer_temporary[7][0]=buffer_write[5][0]; buffer_temporary[7][1]=buffer_write[5][1];

    buffer_temporary[8][0]=buffer_write[10][0]; buffer_temporary[8][1]=buffer_write[10][1];
    buffer_temporary[9][0]=buffer_write[11][0]; buffer_temporary[9][1]=buffer_write[11][1];
    buffer_temporary[10][0]=buffer_write[8][0]; buffer_temporary[10][1]=buffer_write[8][1];
    buffer_temporary[11][0]=buffer_write[9][0]; buffer_temporary[11][1]=buffer_write[9][1];

    buffer_temporary[12][0]=buffer_write[14][0]; buffer_temporary[12][1]=buffer_write[14][1];
    buffer_temporary[13][0]=buffer_write[15][0]; buffer_temporary[13][1]=buffer_write[15][1];
    buffer_temporary[14][0]=buffer_write[12][0]; buffer_temporary[14][1]=buffer_write[12][1];
    buffer_temporary[15][0]=buffer_write[13][0]; buffer_temporary[15][1]=buffer_write[13][1];
   
    for (int i=0; i<16; ++i){
      buffer_write[i][0]=buffer_temporary[i][0]*-1.;//because of the adjoint gamma5
    }
    
    fini_2level_dtable(&buffer_temporary);
}


/* ************************************************************************
 *
 * routine that multiplies with the external gamma5 structure at the sink
 * used for gammas like C, Cg4, cg1g4g5 etc.
 *
 * ************************************************************************/


static inline void mult_with_gamma5_matrix_sink ( double ** buffer_write ) {

    double **buffer_temporary=init_2level_dtable(16,2);

    buffer_temporary[0][0]=buffer_write[8][0]; buffer_temporary[0][1]=buffer_write[8][1];
    buffer_temporary[1][0]=buffer_write[9][0]; buffer_temporary[1][1]=buffer_write[9][1];
    buffer_temporary[2][0]=buffer_write[10][0]; buffer_temporary[2][1]=buffer_write[10][1];
    buffer_temporary[3][0]=buffer_write[11][0]; buffer_temporary[3][1]=buffer_write[11][1];

    buffer_temporary[4][0]=buffer_write[12][0]; buffer_temporary[4][1]=buffer_write[12][1];
    buffer_temporary[5][0]=buffer_write[13][0]; buffer_temporary[5][1]=buffer_write[13][1];
    buffer_temporary[6][0]=buffer_write[14][0]; buffer_temporary[6][1]=buffer_write[14][1];
    buffer_temporary[7][0]=buffer_write[15][0]; buffer_temporary[7][1]=buffer_write[15][1];

    buffer_temporary[8][0]=buffer_write[ 0][0]; buffer_temporary[8][1]=buffer_write[0][1];
    buffer_temporary[9][0]=buffer_write[ 1][0]; buffer_temporary[9][1]=buffer_write[1][1];
    buffer_temporary[10][0]=buffer_write[2][0]; buffer_temporary[10][1]=buffer_write[2][1];
    buffer_temporary[11][0]=buffer_write[3][0]; buffer_temporary[11][1]=buffer_write[3][1];

    buffer_temporary[12][0]=buffer_write[4][0]; buffer_temporary[12][1]=buffer_write[4][1];
    buffer_temporary[13][0]=buffer_write[5][0]; buffer_temporary[13][1]=buffer_write[5][1];
    buffer_temporary[14][0]=buffer_write[6][0]; buffer_temporary[14][1]=buffer_write[6][1];
    buffer_temporary[15][0]=buffer_write[7][0]; buffer_temporary[15][1]=buffer_write[7][1];

    for (int i=0; i<16; ++i){
      buffer_write[i][0]=buffer_temporary[i][0];
    }

    fini_2level_dtable(&buffer_temporary);
}

static inline void sink_and_source_gamma_list( char *filename, char *tagname, int number_of_gammas_source, int number_of_gammas_sink, char ***gamma_string_source, char ***gamma_string_sink, int pion_source){

    hid_t file_id, dataset_id, attr_id; /* identifiers */
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2 ( file_id, tagname, H5P_DEFAULT );
    attr_id = H5Aopen_idx (dataset_id, 0);
    hid_t atype = H5Aget_type(attr_id);
    hsize_t sz = H5Aget_storage_size(attr_id);
    char* date_source = (char *)malloc(sizeof(char)*(sz+1));
    H5Aread( attr_id,atype, (void*)date_source);
    H5Aclose(attr_id );
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    printf("data secr %s %s\n",date_source, filename);

    *gamma_string_source =(char **)malloc(sizeof(char *)*number_of_gammas_source);
    for (int j=0; j<number_of_gammas_source; j++){
      (*gamma_string_source)[j]=(char *)malloc(sizeof(char)*100);
    }

    *gamma_string_sink =(char **)malloc(sizeof(char *)*number_of_gammas_sink);
    for (int j=0; j<number_of_gammas_sink; j++){
      (*gamma_string_sink)[j]=(char *)malloc(sizeof(char)*100);
    }
    char *token; 
    token=strtok(date_source,",{");
    /* walk through other tokens */
    /* we do ignore the external gammas here because they are always one */
    token = strtok(NULL, ",{");        
    token = strtok(NULL, ",{");
    for (int j=0; j<number_of_gammas_source;++j){
      token = strtok(NULL, ",{");
      if (token == NULL){
        fprintf(stderr,"# [sink_and_source_gamma_list] Error in obtaining the source gammas, check out your hdf5 file\n");
        exit(1);
      }
      if (j==(number_of_gammas_source-1)){
        snprintf((*gamma_string_source)[j],strlen(token),"%s",token);
      }
      else{
        snprintf((*gamma_string_source)[j],100,"%s",token);
      }
      fprintf(stdout,"# [sink_and_source_gamma_list] Gamma string list at the source %d %s\n", j, (*gamma_string_source)[j]); 
    }
    if (pion_source ==1) {
      /* if the source contains the pion we have to jump over its gamma structure */
      token=strtok(NULL,",{");
    }
    for (int j=0; j<number_of_gammas_sink;++j){
      token=strtok(NULL, ",{");
      if (token == NULL){
        fprintf(stderr,"# [sink_and_source_gamma_list] Error in obtaining the source gammas, check out your hdf5 file\n");
        exit(1);
      }
      if (j==(number_of_gammas_sink-1)){
        char *temp=strtok(token, "}");
        snprintf((*gamma_string_sink)[j],strlen(temp)+1,"%s",temp);
      }
      else{
        snprintf((*gamma_string_sink)[j],100,"%s",token);
      }
      fprintf(stdout,"# [sink_and_source_gamma_list] Gamma string list at the sink %d %s\n", j, (*gamma_string_sink)[j]);
    }

    free(date_source);
}





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

static inline int twopt_name_to_diagram_tag ( int * num, char **names, char **tag, const char * name ) {

  if (        strcmp( name, "N-N"       ) == 0 )  {
    *num = 1;
    sprintf(*tag , "N");
    sprintf(*names, "N");
  } else if ( strcmp( name, "D-D"       ) == 0 )  {
    *num = 1;
    sprintf(*tag , "D");
    sprintf(*names , "D");
  } else if ( strcmp( name, "pixN-D"    ) == 0 )  {
    *num = 1;
    sprintf(*tag, "T");
    sprintf(*names, "T");
  } else if ( strcmp( name, "D-pixN"    ) == 0 )  {
    *num = 1;
    sprintf(*tag, "T1");
    sprintf(*names, "TpiNsink");
  } else if ( strcmp( name, "pixN-pixN" ) == 0 )  {
    *num = 11;
    sprintf(tag[0],"B1");
    sprintf(tag[1],"B2");
    sprintf(tag[2],"W1");
    sprintf(tag[3],"W2");
    sprintf(tag[4],"W3");
    sprintf(tag[5],"W4");
    sprintf(tag[6],"Z1");
    sprintf(tag[7],"Z2");
    sprintf(tag[8],"Z3");
    sprintf(tag[9],"Z4");
    sprintf(tag[10],"M");
    sprintf(names[0],"B");
    sprintf(names[1],"B");
    sprintf(names[2],"W");
    sprintf(names[3],"W");
    sprintf(names[4],"W");
    sprintf(names[5],"W");
    sprintf(names[6],"Z");
    sprintf(names[7],"Z");
    sprintf(names[8],"Z");
    sprintf(names[9],"Z");
    sprintf(names[10],"M");
  } else if ( strcmp( name, "pi-pi"     ) == 0 )  {
    *num = 1;
    sprintf(*tag,"P");
    sprintf(*names,"P");
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
 
  char const twopt_name_list[6][20] = { "N-N", "D-D","D-pixN", "pixN-D", "pixN-pixN", "pi-pi" };

  /* int const twopt_type_number = 3; */ /* m-m not included here */


  const int momentum_orbit_000[ 1][3] = { {0,0,0} };

  const int momentum_orbit_001[ 6][3] = { {0,0,1}, {0,0,-1}, {0,1,0}, {0,-1,0}, {1,0,0}, {-1,0,0} };

  const int momentum_orbit_110[12][3] = { {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0}, {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1}, {0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1} };

  const int momentum_orbit_111[ 8][3] = { {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1}, {-1,1,1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1} };

  const int (* momentum_orbit_list[4])[3] = { momentum_orbit_000, momentum_orbit_001, momentum_orbit_110, momentum_orbit_111 };
  int const momentum_orbit_nelem[4]   = {1, 6, 12, 8 };
  int const momentum_orbit_pref[4][3] = { {0,0,0}, {0,0,1}, {1,1,0}, {1,1,1} };





  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[400];
  char tagname[400];
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
  fprintf(stdout, "# [piN2piN_diagram_sum_per_type] Warning, resetting global thread number to 1\n");
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
    fprintf(stderr, "# [piN2piN_diagram_sum_per_type] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
    fprintf ( stderr, "# [piN2piN_diagram_sum_per_type] Error from init_2level_itable %s %d\n", __FILE__, __LINE__ );
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
  {
  /******************************************************
   * Correlation function produced in this section names N,D
   ******************************************************/


#ifdef HAVE_HDF5
  /***********************************************************
   * read data block from h5 file
   ***********************************************************/
  snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_N.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3]);

  int ** buffer_mom = init_2level_itable ( 27, 3 );
  if ( buffer_mom == NULL ) {
      fprintf(stderr, "# [piN2piN_diagram_sum_per_type]  Error from ,init_4level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
  }
  snprintf(tagname, 400, "/sx%02dsy%02dsz%02dst%02d/mvec",source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3]);

  fprintf ( stdout, "# [piN2piN_diagram_sum_per_type] open existing file %s\n", filename );
  
  exitstatus = read_from_h5_file ( (void*)(buffer_mom[0]), filename, tagname, io_proc, 1 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "# [piN2piN_diagram_sum_per_type] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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

  /* loop over the source positions */
  for ( int k = 0; k < source_location_number; k++ ) {

    /* loop over the baryon two point functions D and N */

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
      printf("# [piN2piN_diagram_sum_per_type] Two point functions number %d\n",g_twopoint_function_number);
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
      char **hdf5_diag_tag_list_tag=(char **)malloc(sizeof(char*)*12);
      char **hdf5_diag_tag_list_name=(char **)malloc(sizeof(char*)*12);
      for(int i=0; i<12; ++i){
        hdf5_diag_tag_list_tag[i]=(char *)malloc(sizeof(char)*20);
        hdf5_diag_tag_list_name[i]=(char *)malloc(sizeof(char)*20);
      }

      exitstatus = twopt_name_to_diagram_tag (&hdf5_diag_tag_num, hdf5_diag_tag_list_name, hdf5_diag_tag_list_tag, twopt_name_list[iname] );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "# [piN2piN_diagram_sum_per_type] Error from twopt_name_to_diagram_tag, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(123);
      }
      else {
        for (int i=0; i < hdf5_diag_tag_num; ++i) {
          printf("# [piN2piN_diagram_sum_per_type] Name of the twopoint function %s String of hdf5 filename %s\n",twopt_name_list[iname],hdf5_diag_tag_list_name[i]);
          printf("# [piN2piN_diagram_sum_per_type] Name of the twopoint function %s String of hdf5 tag %s\n",twopt_name_list[iname],hdf5_diag_tag_list_tag[i]);
        }
      }
      if ( hdf5_diag_tag_num != 1) {
        fprintf(stderr,"# [piN2piN_diagram_sum_per_type] hdf5_diag_tag_num should be 1, the diagrams D1,D2,D3.. and N1,N2 should already be summed over in PLEGMA \n");
        exit(1);
      }

      /* total number of readers */
      for ( int i = 0 ; i < hdf5_diag_tag_num; i++ ) {

        /* Getting the list of gammas at the source and sink */
        snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_%s.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         hdf5_diag_tag_list_name[i] ); 
        snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%02d/%s",source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         hdf5_diag_tag_list_tag[i]);
        printf("# [piN2piN_diagram_sum_per_type] Filename: %s\n", filename);
        printf("# [piN2piN_diagram_sum_per_type] Tagname: %s\n", tagname);

        char **gamma_string_list_source;
        char **gamma_string_list_sink;

        sink_and_source_gamma_list( filename, tagname, tp->number_of_gammas_source, tp->number_of_gammas_sink, &gamma_string_list_source, &gamma_string_list_sink, 0);

        fprintf(stdout,"# [piN2piN_diagram_sum_per_type] Number of gammas at the source %d\n", tp->number_of_gammas_source);
        double *****buffer_source = init_5level_dtable(tp->T, g_sink_momentum_number, tp->number_of_gammas_source*tp->number_of_gammas_sink, tp->d * tp->d, 2  );

        snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_%s.h5", 
                         filename_prefix, 
                         Nconf,
		         source_coords_list[k][0], 
                         source_coords_list[k][1], 
                         source_coords_list[k][2], 
                         source_coords_list[k][3],
                         hdf5_diag_tag_list_name[i] );
        snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%02d/%s",source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         hdf5_diag_tag_list_tag[i]);
        exitstatus = read_from_h5_file ( (void*)(buffer_source[0][0][0][0]), filename, tagname, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "# [piN2piN_diagram_sum_per_type] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(12);
        }

        /******************************************************
         * loop on total momentum / frames
         ******************************************************/
        for ( int i_total_momentum = 0; i_total_momentum < 4; i_total_momentum++) {

          snprintf ( filename, 400, "%s%04d_PX%02dPY%02dPZ%02d_%s.h5",
                         filename_prefix,
                         Nconf,
                         momentum_orbit_pref[i_total_momentum][0],
                         momentum_orbit_pref[i_total_momentum][1],
                         momentum_orbit_pref[i_total_momentum][2],
                         hdf5_diag_tag_list_name[i] );

          hid_t file_id, group_id, dataset_id, dataspace_id;  /* identifiers */
          herr_t      status;

          struct stat fileStat;
          if(stat( filename, &fileStat) < 0 ) {
          /* Open an existing file. */
            fprintf ( stdout, "# [test_hdf5] create new file %s\n",filename );
            file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
          } else {
            fprintf ( stdout, "# [test_hdf5] open existing file %s\n", filename );
            file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
          }

          /* check if the group for the source position already exists, if not then 
           * lets create it
           */
          snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/", source_coords_list[k][0],
                                                                 source_coords_list[k][1],
                                                                 source_coords_list[k][2],
                                                                 source_coords_list[k][3]);
          status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

          status = H5Gget_objinfo (file_id, tagname, 0, NULL);
          if (status != 0){

             /* Create a group named "/MyGroup" in the file. */
             group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

             /* Close the group. */
             status = H5Gclose(group_id);

          }

          for (int i_pi2=0; i_pi2 < momentum_orbit_nelem[i_total_momentum]; ++i_pi2){


            snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/pf1x%02dpf1y%02dpf1z%02d/",
                                                                 source_coords_list[k][0],
                                                                 source_coords_list[k][1],
                                                                 source_coords_list[k][2],
                                                                 source_coords_list[k][3],
                                                                 momentum_orbit_list[i_total_momentum][i_pi2][0],
                                                                 momentum_orbit_list[i_total_momentum][i_pi2][1],
                                                                 momentum_orbit_list[i_total_momentum][i_pi2][2]);

            /* Create a group named "/MyGroup" in the file. */
            group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            /* Close the group. */
            status = H5Gclose(group_id);

            for (int source_gamma=0; source_gamma < tp->number_of_gammas_source; ++source_gamma ) {


              snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/pf1x%02dpf1y%02dpf1z%02d/gi1%s,%s/",    
                                                                 source_coords_list[k][0],
                                                                 source_coords_list[k][1],
                                                                 source_coords_list[k][2],
                                                                 source_coords_list[k][3],    
                                                                 momentum_orbit_list[i_total_momentum][i_pi2][0],
                                                                 momentum_orbit_list[i_total_momentum][i_pi2][1],
                                                                 momentum_orbit_list[i_total_momentum][i_pi2][2],
                                                                 gamma_string_list_source[source_gamma],
                                                                 ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1" );
              /* Create a group named "/MyGroup" in the file. */
              group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

              /* Close the group. */
              status = H5Gclose(group_id);
            
              for (int sink_gamma=0; sink_gamma < tp->number_of_gammas_sink; ++sink_gamma ) {

                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1");
               

                status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                if (status != 0){

                   /* Create a group named "/MyGroup" in the file. */
                   group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                   /* Close the group. */
                   status = H5Gclose(group_id);

                }

                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][0],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][1],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][2]);



                status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                if (status != 0){

                   /* Create a group named "/MyGroup" in the file. */
                   group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                   /* Close the group. */
                   status = H5Gclose(group_id);

                }
                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi1%s,%s",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][0],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][1],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][2],
                                                                   gamma_string_list_source[source_gamma],
                                                                   ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1");

                                                                   
                                                                   
                status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);     
                                                                   
                status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                if (status != 0){

                   /* Create a group named "/MyGroup" in the file. */
                   group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                                                                   
                   /* Close the group. */
                   status = H5Gclose(group_id); 

                }

 
                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi1%s,%s/pi1x%02dpi1y%02dpi1z%02d/",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][0],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][1],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][2],
                                                                   gamma_string_list_source[source_gamma],
                                                                   ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   -momentum_orbit_list[i_total_momentum][i_pi2][0],
                                                                   -momentum_orbit_list[i_total_momentum][i_pi2][1],
                                                                   -momentum_orbit_list[i_total_momentum][i_pi2][2]);

                status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                if (status != 0){

                   /* Create a group named "/MyGroup" in the file. */
                   group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                   /* Close the group. */
                   status = H5Gclose(group_id);

                }



                /* printf("# [piN2piN_diagram_sum_per_type] Tagname created %s\n",tagname); */
                /* Create a group named "/MyGroup" in the file. */
                group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                /* Close the group. */
                status = H5Gclose(group_id);

                hsize_t dims[3];
                dims[0]=tp->T;
                dims[1]=tp->d*tp->d;
                dims[2]=2;
                dataspace_id = H5Screate_simple(3, dims, NULL);

                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi1%s,%s/pi1x%02dpi1y%02dpi1z%02d/%s",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][0],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][1],
                                                                   momentum_orbit_list[i_total_momentum][i_pi2][2],
                                                                   gamma_string_list_source[source_gamma],
                                                                   ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   -momentum_orbit_list[i_total_momentum][i_pi2][0],
                                                                   -momentum_orbit_list[i_total_momentum][i_pi2][1],
                                                                   -momentum_orbit_list[i_total_momentum][i_pi2][2],
                                                                   hdf5_diag_tag_list_tag[i]);

                /* Create a dataset in group "MyGroup". */
                dataset_id = H5Dcreate2(file_id, tagname, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                double ***buffer_write= init_3level_dtable(tp->T,tp->d*tp->d,2);
                for (int time_extent = 0; time_extent < tp->T ; ++ time_extent ){ 
                  for (int spin_inner=0; spin_inner < tp->d*tp->d; ++spin_inner) {
                    for (int realimag=0; realimag < 2; ++realimag){
                      buffer_write[time_extent][spin_inner][realimag]=buffer_source[time_extent][indextable[i_total_momentum][i_pi2]][source_gamma*tp->number_of_gammas_sink+sink_gamma][spin_inner][realimag]/(double)g_source_location_number;
                    }
                  }
                  if ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ){
                    mult_with_gamma5_matrix_source(buffer_write[time_extent]);

                  }
                  if ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ){
                    mult_with_gamma5_matrix_sink(buffer_write[time_extent]);

                  }

                }
                /* Write the first dataset. */
                status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(buffer_write[0][0][0]));

                /* Close the data space for the first dataset. */
                status = H5Sclose(dataspace_id);

                /* Close the first dataset. */
                status = H5Dclose(dataset_id);

                fini_3level_dtable(&buffer_write);

              }/*sink gamma*/

            }/*source_gamma*/

          }/* momentum */
          /* Close the file. */
          status = H5Fclose(file_id);

        }//loop on total momentum
        fini_5level_dtable(&buffer_source);
        for (int j=0; j<tp->number_of_gammas_source; ++j)
          free(gamma_string_list_source[j]);
        free(gamma_string_list_source);
        for (int j=0; j<tp->number_of_gammas_sink; ++j)
          free(gamma_string_list_sink[j]);
        free(gamma_string_list_sink);
      }//hdf5 names
      fini_2level_itable(&buffer_mom);
      fini_1level_itable(&twopt_id_list);
      for (int j=0;j<12; ++j){
        free(hdf5_diag_tag_list_tag[j]);
        free(hdf5_diag_tag_list_name[j]);
      }     
      free(hdf5_diag_tag_list_tag);
      free(hdf5_diag_tag_list_name);

    }  /* end of loop on twopoint function names */

  } /* end of loop on source positions */


  for (int i_total_momentum=0; i_total_momentum<4; ++i_total_momentum)
    free(indextable[i_total_momentum]);
  free(indextable); 

  } /* end of loop on N,D diagrams */


  /******************************************************
   *
   * mxb-b type
   *
   ******************************************************/

  {


  #ifdef HAVE_HDF5
  /***********************************************************
   * read data block from h5 file
   ***********************************************************/

  int ** buffer_mom = init_2level_itable ( 343, 9 );
  if ( buffer_mom == NULL ) {
      fprintf(stderr, "[piN2piN_diagram_sum_per_type]  Error from ,init_4level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
  }
  snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_TpiNsink.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3]);
  snprintf(tagname, 400, "/sx%02dsy%02dsz%02dst%02d/mvec",source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3]);

  exitstatus = read_from_h5_file ( (void*)(buffer_mom[0]), filename, tagname, io_proc, 1 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(12);
  }
  int **indextable=(int **)malloc(sizeof(int*)*4);
  int *num_elements=(int *)malloc(sizeof(int)*4);
  for (int i=0; i<4; ++i)
    num_elements[i]=0;
  for (int i=0; i<343; ++i){
    num_elements[(buffer_mom[i][3]+buffer_mom[i][6])*(buffer_mom[i][3]+buffer_mom[i][6])+(buffer_mom[i][4]+buffer_mom[i][7])*(buffer_mom[i][4]+buffer_mom[i][7])+(buffer_mom[i][5]+buffer_mom[i][8])*(buffer_mom[i][5]+buffer_mom[i][8])]++;
  }
  for (int i=0; i<4; ++i){
    indextable[i]=(int *)malloc(sizeof(int)*num_elements[i]);
  }
  for (int i=0; i<4; ++i)
    num_elements[i]=0;
  for (int i=0; i<343; ++i){
    int tot_mom=(buffer_mom[i][3]+buffer_mom[i][6])*(buffer_mom[i][3]+buffer_mom[i][6])+(buffer_mom[i][4]+buffer_mom[i][7])*(buffer_mom[i][4]+buffer_mom[i][7])+(buffer_mom[i][5]+buffer_mom[i][8])*(buffer_mom[i][5]+buffer_mom[i][8]);
    indextable[tot_mom][num_elements[tot_mom]]=i;
    num_elements[tot_mom]++;
  }
#endif

  for ( int k = 0; k < source_location_number; k++ ) {

    for ( int iname = 2; iname < 3; iname++ ) {

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
       * for mxb-b affr_diag_tag_num is 1
       ******************************************************/
      int hdf5_diag_tag_num = 0;
      char **hdf5_diag_tag_list_tag=(char **)malloc(sizeof(char*)*12);
      char **hdf5_diag_tag_list_name=(char **)malloc(sizeof(char*)*12);
      for(int i=0; i<12; ++i){
        hdf5_diag_tag_list_tag[i]=(char *)malloc(sizeof(char)*20);
        hdf5_diag_tag_list_name[i]=(char *)malloc(sizeof(char)*20);
      }

      exitstatus = twopt_name_to_diagram_tag (&hdf5_diag_tag_num, hdf5_diag_tag_list_name,hdf5_diag_tag_list_tag, twopt_name_list[iname] );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "# [piN2piN_diagram_sum_per_type] Error from twopt_name_to_diagram_tag, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(123);
      }
      else {
        for (int i=0; i < hdf5_diag_tag_num; ++i) {
          printf("# [piN2piN_diagram_sum_per_type] Name of the twopoint function %s String of hdf5 filename %s\n",twopt_name_list[iname],hdf5_diag_tag_list_name[i]);
          printf("# [piN2piN_diagram_sum_per_type] Name of the twopoint function %s String of hdf5 tag %s\n",twopt_name_list[iname],hdf5_diag_tag_list_tag[i]);
        }
      }


      /* total number of readers */


      for ( int i = 0 ; i < hdf5_diag_tag_num; i++ ) {
      /* Getting the list of gammas */
        snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_%s.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         hdf5_diag_tag_list_name[i] );
        snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%02d/%s",source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         hdf5_diag_tag_list_tag[i]);

        char **gamma_string_list_source;
        char **gamma_string_list_sink;

        printf("#[piN2piN_diagram_sum_per_type] Number of gammas at the sink %d \n", tp->number_of_gammas_sink);
        sink_and_source_gamma_list( filename, tagname, tp->number_of_gammas_source, tp->number_of_gammas_sink, &gamma_string_list_source, &gamma_string_list_sink, 0);


       
        double *****buffer_source = init_5level_dtable(tp->T, 343, tp->number_of_gammas_sink*tp->number_of_gammas_source, tp->d * tp->d, 2  );

        snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_%s.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         hdf5_diag_tag_list_name[i] );
        snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%02d/%s",source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         hdf5_diag_tag_list_tag[i]);
        exitstatus = read_from_h5_file ( (void*)(buffer_source[0][0][0][0]), filename, tagname, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "#[piN2piN_diagram_sum_per_type] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(12);
        } 

        /******************************************************
         * loop on total momentum / frames
         ******************************************************/
        for ( int i_total_momentum = 0; i_total_momentum < 4; i_total_momentum++) {

          hid_t file_id, group_id, dataset_id, dataspace_id;  /* identifiers */
          herr_t      status;

          snprintf ( filename, 400, "%s%04d_PX%02dPY%02dPZ%02d_%s.h5",
                         filename_prefix,
                         Nconf,
                         momentum_orbit_pref[i_total_momentum][0],
                         momentum_orbit_pref[i_total_momentum][1],
                         momentum_orbit_pref[i_total_momentum][2],
                         hdf5_diag_tag_list_name[i] );
          fprintf ( stdout, "# [test_hdf5] create new file %s\n", filename );

          file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

          /* check if the group for the source position already exists, if not then 
           * lets create it
           */
          snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/", source_coords_list[k][0],
                                                                 source_coords_list[k][1],
                                                                 source_coords_list[k][2],
                                                                 source_coords_list[k][3]);
          status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

          status = H5Gget_objinfo (file_id, tagname, 0, NULL);
          if (status != 0){

             /* Create a group named "/MyGroup" in the file. */
             group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

             /* Close the group. */
             status = H5Gclose(group_id);

          }


          for (int i_pi2=0; i_pi2 < num_elements[i_total_momentum]; ++i_pi2){

            for (int source_gamma =0 ; source_gamma < tp->number_of_gammas_source ; ++source_gamma ){

              for (int sink_gamma=0; sink_gamma < tp->number_of_gammas_sink; ++sink_gamma ) {

                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/", source_coords_list[k][0],
                                                                       source_coords_list[k][1],
                                                                       source_coords_list[k][2],
                                                                       source_coords_list[k][3]);
                status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                if (status != 0){

                 /* Create a group named "/MyGroup" in the file. */
                 group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                 /* Close the group. */
                 status = H5Gclose(group_id);

                }

                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/", source_coords_list[k][0],
                                                                       source_coords_list[k][1],
                                                                       source_coords_list[k][2],
                                                                       source_coords_list[k][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][6],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][7],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][8]);

                status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);     
                                                        
                status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                if (status != 0){                                  
                           
                 /* Create a group named "/MyGroup" in the file. */
                 group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                                                                   
                 /* Close the group. */
                 status = H5Gclose(group_id);
                                         
                }

                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/", source_coords_list[k][0],
                                                                       source_coords_list[k][1],
                                                                       source_coords_list[k][2],
                                                                       source_coords_list[k][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][6],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][7],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][8],
                                                                       gamma_string_list_sink[sink_gamma],
                                                                       ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1");


                status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                if (status != 0){

                 /* Create a group named "/MyGroup" in the file. */
                 group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                 /* Close the group. */
                 status = H5Gclose(group_id);

                }
                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/", source_coords_list[k][0],
                                                                       source_coords_list[k][1],
                                                                       source_coords_list[k][2],
                                                                       source_coords_list[k][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][6],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][7],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][8],
                                                                       gamma_string_list_sink[sink_gamma],
                                                                       ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][4],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][5]);



                status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                if (status != 0){

                 /* Create a group named "/MyGroup" in the file. */
                 group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                 /* Close the group. */
                 status = H5Gclose(group_id);

                }

                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi1%s,%s/", source_coords_list[k][0],
                                                                       source_coords_list[k][1],
                                                                       source_coords_list[k][2],
                                                                       source_coords_list[k][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][6],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][7],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][8],
                                                                       gamma_string_list_sink[sink_gamma],
                                                                       ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][4],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][5],
                                                                       gamma_string_list_source[source_gamma],
                                                                       ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1");

                status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                if (status != 0){

                 /* Create a group named "/MyGroup" in the file. */
                 group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                 /* Close the group. */
                 status = H5Gclose(group_id);

                }

                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi1%s,%s/pi1x%02dpi1y%02dpi1z%02d/", source_coords_list[k][0],
                                                                       source_coords_list[k][1],
                                                                       source_coords_list[k][2],
                                                                       source_coords_list[k][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][6],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][7],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][8],
                                                                       gamma_string_list_sink[sink_gamma],
                                                                       ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][4],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][5],
                                                                       gamma_string_list_source[source_gamma],
                                                                       ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][6]+ buffer_mom[indextable[i_total_momentum][i_pi2]][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][7]+ buffer_mom[indextable[i_total_momentum][i_pi2]][4],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][8]+ buffer_mom[indextable[i_total_momentum][i_pi2]][5]);


                status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                if (status != 0){

                 /* Create a group named "/MyGroup" in the file. */
                 group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                 /* Close the group. */
                 status = H5Gclose(group_id);

                }

            
                hsize_t dims[3];
                dims[0]=tp->T;
                dims[1]=tp->d*tp->d;
                dims[2]=2;
                dataspace_id = H5Screate_simple(3, dims, NULL);

                snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi1%s,%s/pi1x%02dpi1y%02dpi1z%02d/", source_coords_list[k][0],
                                                                       source_coords_list[k][1],
                                                                       source_coords_list[k][2],
                                                                       source_coords_list[k][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][6],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][7],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][8],
                                                                       gamma_string_list_sink[sink_gamma],
                                                                       ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][4],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][5],
                                                                       gamma_string_list_source[source_gamma],
                                                                       ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][6]+ buffer_mom[indextable[i_total_momentum][i_pi2]][3],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][7]+ buffer_mom[indextable[i_total_momentum][i_pi2]][4],
                                                                       buffer_mom[indextable[i_total_momentum][i_pi2]][8]+ buffer_mom[indextable[i_total_momentum][i_pi2]][5],
                                                                       hdf5_diag_tag_list_tag[i]);


                /* Create a dataset in group "MyGroup". */
                dataset_id = H5Dcreate2(file_id, tagname, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                double ***buffer_write= init_3level_dtable(tp->T,tp->d*tp->d,2);
                for (int time_extent = 0; time_extent < tp->T ; ++ time_extent ){
                  for (int spin_inner=0; spin_inner < tp->d*tp->d; ++spin_inner) {
                    for (int realimag=0; realimag < 2; ++realimag){
                      buffer_write[time_extent][spin_inner][realimag]=buffer_source[time_extent][indextable[i_total_momentum][i_pi2]][source_gamma*tp->number_of_gammas_sink+sink_gamma][spin_inner][realimag]/(double)g_source_location_number;
                    }
                  }
                  if ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ){
                    mult_with_gamma5_matrix_source(buffer_write[time_extent]);

                  }
                  if ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ){
                    mult_with_gamma5_matrix_sink(buffer_write[time_extent]);

                  }
                }


                /* Write the first dataset. */
                status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(buffer_write[0][0][0]));

                /* Close the data space for the first dataset. */
                status = H5Sclose(dataspace_id);

                /* Close the first dataset. */
                status = H5Dclose(dataset_id);

                fini_3level_dtable(&buffer_write);

              } /*gamma sink*/

            } /*gamma source */

          } /* pi2 corresponding to a given p_tot */
          /* Close the file. */
          status = H5Fclose(file_id);

        } /*loop on total momentum */
        fini_5level_dtable(&buffer_source);

        for (int j=0; j<tp->number_of_gammas_sink; ++j)
          free(gamma_string_list_sink[j]);
        free(gamma_string_list_sink);

        for (int j=0; j<tp->number_of_gammas_source; ++j)
          free(gamma_string_list_source[j]);
        free(gamma_string_list_source);


      }/*hdf5 names*/
      fini_1level_itable(&twopt_id_list);
      for (int j=0;j<12; ++j){
        free(hdf5_diag_tag_list_tag[j]);
        free(hdf5_diag_tag_list_name[j]);
      }
      free(hdf5_diag_tag_list_tag);
      free(hdf5_diag_tag_list_name);

    }  /* end of loop on twopoint function names */

  } /* end of loop on source positions */


  for (int i_total_momentum=0; i_total_momentum<4; ++i_total_momentum)
    free(indextable[i_total_momentum]);
  free(indextable);
  free(num_elements);
  fini_2level_itable(&buffer_mom);

  } /* end of loop on producing TpiNsink */


  /******************************************************
   *
   * b-mxb type
   *
   ******************************************************/

  /* bracketing the production of T diagram */
  {


#ifdef HAVE_HDF5
    /***********************************************************
     * read data block from h5 file
     ***********************************************************/

    int const pi2[3] = {
    g_seq_source_momentum_list[0][0],
    g_seq_source_momentum_list[0][1],
    g_seq_source_momentum_list[0][2] }; 
    printf("# [piN2piN_diagram_sum_per_type] seq momentum %d %d %d %d\n", pi2[0], pi2[1], pi2[2], g_seq_source_momentum_number);

    snprintf ( filename, 200, "%s%04d_sx%02dsy%02dsz%02dst%03d_T.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3] );

    int ** buffer_mom = init_2level_itable ( 27, 6 );
    if ( buffer_mom == NULL ) {
       fprintf(stderr, "# [piN2piN_diagram_sum_per_type]  Error from ,init_4level_dtable %s %d\n", __FILE__, __LINE__ );
       EXIT(12);
    }
    snprintf(tagname, 200, "/sx%02dsy%02dsz%02dst%02d/pi2=%d_%d_%d/mvec",source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         pi2[0],pi2[1],pi2[2]);

    exitstatus = read_from_h5_file ( (void*)(buffer_mom[0]), filename, tagname, io_proc, 1 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "# [piN2piN_diagram_sum_per_type] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }
    int **indextable=(int **)malloc(sizeof(int*)*4);
    int *num_elements=(int *)malloc(sizeof(int)*4);
    for (int i=0; i<4; ++i)
      num_elements[i]=0;
    for (int i=0; i<27; ++i){
      num_elements[(buffer_mom[i][3])*(buffer_mom[i][3])+(buffer_mom[i][4])*(buffer_mom[i][4])+(buffer_mom[i][5])*(buffer_mom[i][5])]++;
    }
    for (int i=0; i<4; ++i){
      indextable[i]=(int *)malloc(sizeof(int)*num_elements[i]);
    } 
    for (int i=0; i<4; ++i)
      num_elements[i]=0;
    for (int i=0; i<27; ++i){
      int tot_mom=(buffer_mom[i][3])*(buffer_mom[i][3])+(buffer_mom[i][4])*(buffer_mom[i][4])+(buffer_mom[i][5])*(buffer_mom[i][5]);
      indextable[tot_mom][num_elements[tot_mom]]=i;
      num_elements[tot_mom]++;
    }
#endif

    for ( int k = 0; k < source_location_number; k++ ) {

      for ( int iname = 3; iname < 4; iname++ ) {

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
         * for b-mxb affr_diag_tag_num is 1
         ******************************************************/

        int hdf5_diag_tag_num = 0;
        char **hdf5_diag_tag_list_tag=(char **)malloc(sizeof(char*)*12);
        char **hdf5_diag_tag_list_name=(char **)malloc(sizeof(char*)*12);
        for(int i=0; i<12; ++i){
          hdf5_diag_tag_list_tag[i]=(char *)malloc(sizeof(char)*20);
          hdf5_diag_tag_list_name[i]=(char *)malloc(sizeof(char)*20);
        }

        exitstatus = twopt_name_to_diagram_tag (&hdf5_diag_tag_num, hdf5_diag_tag_list_name,hdf5_diag_tag_list_tag, twopt_name_list[iname] );
        if ( exitstatus != 0 ) {
          fprintf( stderr, "# [piN2piN_diagram_sum_per_type] Error from twopt_name_to_diagram_tag, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(123);
        } 
        else {
          if (hdf5_diag_tag_num != 1) {
            fprintf(stderr,  "# [piN2piN_diagram_sum_per_type] Error from twopt_name_to_diagram_tag, the T-piN diagrams are already summed for T1,2,3,4,5,6\n");
          }
        }

        int const pi2[3] = {
             g_seq_source_momentum_list[0][0],
             g_seq_source_momentum_list[0][1],
             g_seq_source_momentum_list[0][2] };


        char **gamma_string_list_source;
        char **gamma_string_list_sink;


        snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_%s.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         hdf5_diag_tag_list_name[0] );

        snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%02d/pi2=%d_%d_%d/%s",source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         pi2[0],pi2[1],pi2[2],
                         hdf5_diag_tag_list_tag[0]);


        sink_and_source_gamma_list( filename, tagname, tp->number_of_gammas_source, tp->number_of_gammas_sink, &gamma_string_list_source, &gamma_string_list_sink, 1);



        /******************************************************
         * loop on sequential momentum / frames
         ******************************************************/

        for (int i_pi2=0; i_pi2 < g_seq_source_momentum_number; ++i_pi2) {



          double *****buffer_source = init_5level_dtable(tp->T, 27, tp->number_of_gammas_sink*tp->number_of_gammas_source, tp->d * tp->d, 2  );

          int const pi2[3] = {
          g_seq_source_momentum_list[i_pi2][0],
          g_seq_source_momentum_list[i_pi2][1],
          g_seq_source_momentum_list[i_pi2][2] };

          snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_%s.h5",
                       filename_prefix,
                       Nconf,
                       source_coords_list[k][0],
                       source_coords_list[k][1],
                       source_coords_list[k][2],
                       source_coords_list[k][3],
                       hdf5_diag_tag_list_name[0] );

          snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%02d/pi2=%d_%d_%d/%s",source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         pi2[0],pi2[1],pi2[2],
                         hdf5_diag_tag_list_tag[0]);
          exitstatus = read_from_h5_file ( (void*)(buffer_source[0][0][0][0]), filename, tagname, io_proc );
          if ( exitstatus != 0 ) {
             fprintf(stderr, "# [piN2piN_diagram_sum_per_type] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
             EXIT(12);
          }

          for ( int i_total_momentum = 0; i_total_momentum < 4; i_total_momentum++) {



            snprintf ( filename, 400, "%s%04d_PX%02dPY%02dPZ%02d_%s.h5",
                         filename_prefix,
                         Nconf,
                         momentum_orbit_pref[i_total_momentum][0],
                         momentum_orbit_pref[i_total_momentum][1],
                         momentum_orbit_pref[i_total_momentum][2],
                         hdf5_diag_tag_list_name[0] );

            hid_t file_id, group_id, dataset_id, dataspace_id;  /* identifiers */
            herr_t      status;
 
            struct stat fileStat;
            if(stat( filename, &fileStat) < 0 ) {
            /* Open an existing file. */
              fprintf ( stdout, "# [test_hdf5] create new file %s\n", filename );
              file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            } else {
              fprintf ( stdout, "# [test_hdf5] open existing file\n" );
              file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
            }





            /* check if the group for the source position already exists, if not then 
             * lets create it
             */
            snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/", source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3]);
            status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

            status = H5Gget_objinfo (file_id, tagname, 0, NULL);
            if (status != 0){

              /* Create a group named "/MyGroup" in the file. */
              group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

              /* Close the group. */
              status = H5Gclose(group_id);

            }


            for (int i_pf1=0; i_pf1 < num_elements[i_total_momentum]; ++i_pf1){

              for (int source_gamma =0 ; source_gamma < tp->number_of_gammas_source ; ++source_gamma ){

                for (int sink_gamma=0; sink_gamma < tp->number_of_gammas_sink; ++sink_gamma ) {

                  snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1");

                  status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                  status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                  if (status != 0){

                     /* Create a group named "/MyGroup" in the file. */
                     group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                     /* Close the group. */
                     status = H5Gclose(group_id);

                  }

                  snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][5]);

                  status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                  status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                  if (status != 0){

                     /* Create a group named "/MyGroup" in the file. */
                     group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                     /* Close the group. */
                     status = H5Gclose(group_id);

                  }

                  snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi25",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][5]);

                  status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                  status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                  if (status != 0){

                     /* Create a group named "/MyGroup" in the file. */
                     group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                     /* Close the group. */
                     status = H5Gclose(group_id);

                  }

                  snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi25/pi2x%02dpi2y%02dpi2z%02d/",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][5],
                                                                   pi2[0],
                                                                   pi2[1],
                                                                   pi2[2]);


                  status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                  status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                  if (status != 0){

                     /* Create a group named "/MyGroup" in the file. */
                     group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                     /* Close the group. */
                     status = H5Gclose(group_id);

                  }

                  snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi25/pi2x%02dpi2y%02dpi2z%02d/gi1%s,%s",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][5],
                                                                   pi2[0],
                                                                   pi2[1],
                                                                   pi2[2],
                                                                   gamma_string_list_source[source_gamma],
                                                                   ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1");



                  status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                  status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                  if (status != 0){

                     /* Create a group named "/MyGroup" in the file. */
                     group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                     /* Close the group. */
                     status = H5Gclose(group_id);

                  }

                  snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi25/pi2x%02dpi2y%02dpi2z%02d/gi1%s,%s/pi1x%02dpi1y%02dpi1z%02d/",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][5],
                                                                   pi2[0],
                                                                   pi2[1],
                                                                   pi2[2],
                                                                   gamma_string_list_source[source_gamma],
                                                                   ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][3]-pi2[0],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][4]-pi2[1],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][5]-pi2[2]);



                  status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                  status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                  if (status != 0){

                     /* Create a group named "/MyGroup" in the file. */
                     group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                     /* Close the group. */
                     status = H5Gclose(group_id);

                  }




                  hsize_t dims[3];
                  dims[0]=tp->T;
                  dims[1]=tp->d*tp->d;
                  dims[2]=2;
                  dataspace_id = H5Screate_simple(3, dims, NULL);

                  snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi25/pi2x%02dpi2y%02dpi2z%02d/gi1%s,%s/pi1x%02dpi1y%02dpi1z%02d/%s",
                                                                   source_coords_list[k][0],
                                                                   source_coords_list[k][1],
                                                                   source_coords_list[k][2],
                                                                   source_coords_list[k][3],
                                                                   gamma_string_list_sink[sink_gamma],
                                                                   ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][5],
                                                                   pi2[0],
                                                                   pi2[1],
                                                                   pi2[2],
                                                                   gamma_string_list_source[source_gamma],
                                                                   ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][3]-pi2[0],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][4]-pi2[1],
                                                                   buffer_mom[indextable[i_total_momentum][i_pf1]][5]-pi2[2],
                                                                   hdf5_diag_tag_list_tag[0]);

                  /* Create a dataset in group "MyGroup". */
                  dataset_id = H5Dcreate2(file_id, tagname, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                  double ***buffer_write= init_3level_dtable(tp->T,tp->d*tp->d,2);
                  for (int time_extent = 0; time_extent < tp->T ; ++ time_extent ){
                    for (int spin_inner=0; spin_inner < tp->d*tp->d; ++spin_inner) {
                      for (int realimag=0; realimag < 2; ++realimag){
                        //printf("index %d \n", indextable[i_total_momentum][i_pf1] );
                        buffer_write[time_extent][spin_inner][realimag]=buffer_source[time_extent][indextable[i_total_momentum][i_pf1]][source_gamma*tp->number_of_gammas_sink+sink_gamma][spin_inner][realimag];
                      }
                    }
                    if ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ){
                     mult_with_gamma5_matrix_source(buffer_write[time_extent]);

                    }
                    if ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ){
                     mult_with_gamma5_matrix_sink(buffer_write[time_extent]);

                    }
                  }
  
                  /* Write the first dataset. */
                  status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(buffer_write[0][0][0]));

                  /* Close the data space for the first dataset. */
                  status = H5Sclose(dataspace_id);

                  /* Close the first dataset. */
                  status = H5Dclose(dataset_id);
                 
                  fini_3level_dtable(&buffer_write);

                } /*end of loop on sink gammas */

              } /*end of loop on source gammas */


              /* Close the file. */
              status = H5Fclose(file_id);

           } /*enf of loop on pf1*/

         } /* end of loop on total momentum */

         fini_5level_dtable(&buffer_source);

       } /* end of loop on sequential momentum */


       for (int j=0;j<12; ++j){
         free(hdf5_diag_tag_list_tag[j]);
         free(hdf5_diag_tag_list_name[j]);
       }
       free(hdf5_diag_tag_list_tag);
       free(hdf5_diag_tag_list_name);

       for (int j=0; j<tp->number_of_gammas_sink; ++j)
         free(gamma_string_list_sink[j]);
       free(gamma_string_list_sink);

       for (int j=0; j<tp->number_of_gammas_source; ++j)
         free(gamma_string_list_source[j]);
       free(gamma_string_list_source);

       fini_1level_itable(&twopt_id_list);

     } /*name of the twopoint functions */

   }/* loop on source positions */


   for (int i_total_momentum=0; i_total_momentum<4; ++i_total_momentum)
     free(indextable[i_total_momentum]);
   free(indextable);
   free(num_elements);
 
   fini_2level_itable(&buffer_mom);


   }/* end of loop on T diagram */

   printf(" [piN2piN_diagram_sum_per_type] finishing T diagrams \n");

  /******************************************************
   *
   * mxb-mxb type
   *
   ******************************************************/

   { /* Creating piN-piN diagrams by summing B1,B2,W1,W2,W3,W4,Z1,Z2,Z3,Z4 */


    int const pi2[3] = {
            g_seq_source_momentum_list[0][0],
            g_seq_source_momentum_list[0][1],
            g_seq_source_momentum_list[0][2] };
#ifdef HAVE_HDF5
    /***********************************************************
     * read data block from h5 file
     ***********************************************************/
    snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_B.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3] );
    int ** buffer_mom = init_2level_itable ( 343, 9 );
    if ( buffer_mom == NULL ) {
       fprintf(stderr, "# [piN2piN_diagram_sum_per_type]  Error from ,init_2level_itable %s %d\n", __FILE__, __LINE__ );
       EXIT(12);
    }
    snprintf(tagname, 400, "/sx%02dsy%02dsz%02dst%02d/pi2=%d_%d_%d/mvec",source_coords_list[0][0],
                         source_coords_list[0][1],
                         source_coords_list[0][2],
                         source_coords_list[0][3],
                         pi2[0],pi2[1],pi2[2]);

    exitstatus = read_from_h5_file ( (void*)(buffer_mom[0]), filename, tagname, io_proc, 1 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[piN2piN_diagram_sum_per_type] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }
    int **indextable=(int **)malloc(sizeof(int*)*4);
    int *num_elements=(int *)malloc(sizeof(int)*4);
    for (int i=0; i<4; ++i)
      num_elements[i]=0;
    for (int i=0; i<343; ++i){
      num_elements[(buffer_mom[i][3]+buffer_mom[i][6])*(buffer_mom[i][3]+buffer_mom[i][6])+(buffer_mom[i][4]+buffer_mom[i][7])*(buffer_mom[i][4]+buffer_mom[i][7])+(buffer_mom[i][5]+buffer_mom[i][8])*(buffer_mom[i][5]+buffer_mom[i][8])]++;
    }
    for (int i=0; i<4; ++i){
      indextable[i]=(int *)malloc(sizeof(int)*num_elements[i]);
    }
    for (int i=0; i<4; ++i)
      num_elements[i]=0;
    for (int i=0; i<343; ++i){
      int tot_mom=(buffer_mom[i][3]+buffer_mom[i][6])*(buffer_mom[i][3]+buffer_mom[i][6])+(buffer_mom[i][4]+buffer_mom[i][7])*(buffer_mom[i][4]+buffer_mom[i][7])+(buffer_mom[i][5]+buffer_mom[i][8])*(buffer_mom[i][5]+buffer_mom[i][8]);
      indextable[tot_mom][num_elements[tot_mom]]=i;
      num_elements[tot_mom]++;
    }
#endif
   /* total number of readers */

    for ( int k = 0; k < source_location_number; k++ ) {

      for ( int iname = 4; iname < 5; iname++ ) {

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
         * for mxb-mxb affr_diag_tag_num is 1
         ******************************************************/
        int hdf5_diag_tag_num = 0;
        char **hdf5_diag_tag_list_tag=(char **)malloc(sizeof(char*)*12);
        char **hdf5_diag_tag_list_name=(char **)malloc(sizeof(char*)*12);
        for(int i=0; i<12; ++i){
          hdf5_diag_tag_list_tag[i]=(char *)malloc(sizeof(char)*20);
          hdf5_diag_tag_list_name[i]=(char *)malloc(sizeof(char)*20);
        }

        exitstatus = twopt_name_to_diagram_tag (&hdf5_diag_tag_num, hdf5_diag_tag_list_name,hdf5_diag_tag_list_tag, twopt_name_list[iname] );
        if ( exitstatus != 0 ) {
          fprintf( stderr, "# [piN2piN_diagram_sum_per_type] Error from twopt_name_to_diagram_tag, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(123);
        }
        else {
          for (int i=0; i < hdf5_diag_tag_num; ++i) {
            printf("#[piN2piN_diagram_sum_per_type] Name of the twopoint function %s String of hdf5 filename %s\n",twopt_name_list[iname],hdf5_diag_tag_list_name[i]);
            printf("#[piN2piN_diagram_sum_per_type] Name of the twopoint function %s String of hdf5 tag %s\n",twopt_name_list[iname],hdf5_diag_tag_list_tag[i]);
          }
        }

        int const pi2[3] = {
             g_seq_source_momentum_list[0][0],
             g_seq_source_momentum_list[0][1],
             g_seq_source_momentum_list[0][2] };


        char **gamma_string_list_source;
        char **gamma_string_list_sink;


        snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_%s.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         hdf5_diag_tag_list_name[0] );

        snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%02d/pi2=%d_%d_%d/%s",source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         pi2[0],pi2[1],pi2[2],
                         hdf5_diag_tag_list_tag[0]);


        sink_and_source_gamma_list( filename, tagname, tp->number_of_gammas_source, tp->number_of_gammas_sink, &gamma_string_list_source, &gamma_string_list_sink, 1);



        for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
          int const pi2[3] = {
             g_seq_source_momentum_list[ipi2][0],
             g_seq_source_momentum_list[ipi2][1],
             g_seq_source_momentum_list[ipi2][2] };


          double *****buffer_sum = init_5level_dtable(tp->T, 343, tp->number_of_gammas_source*tp->number_of_gammas_sink, tp->d * tp->d, 2  );
          for (int time_extent = 0; time_extent < tp->T ; ++ time_extent ){
            for (int momentum_number=0; momentum_number < 343; ++momentum_number){
              for (int spin_structures=0; spin_structures < tp->number_of_gammas_sink*tp->number_of_gammas_source; ++spin_structures ){
                for (int spin_inner=0; spin_inner < tp->d*tp->d; ++spin_inner) {
                  for (int realimag=0; realimag < 2; ++realimag){
                    buffer_sum[time_extent][momentum_number][spin_structures][spin_inner][realimag]=0.;
                  }
                }
              }
            }
          }

          for (int i=0; i < hdf5_diag_tag_num; ++i) {
            double *****buffer_source = init_5level_dtable(tp->T, 343, tp->number_of_gammas_sink*tp->number_of_gammas_source, tp->d * tp->d, 2  );
            snprintf ( filename, 400, "%s%04d_sx%02dsy%02dsz%02dst%03d_%s.h5",
                         filename_prefix,
                         Nconf,
                         source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         hdf5_diag_tag_list_name[i] );
           snprintf ( tagname,400, "/sx%02dsy%02dsz%02dst%02d/pi2=%d_%d_%d/%s",source_coords_list[k][0],
                         source_coords_list[k][1],
                         source_coords_list[k][2],
                         source_coords_list[k][3],
                         pi2[0],pi2[1],pi2[2],
                         hdf5_diag_tag_list_tag[i]);
           exitstatus = read_from_h5_file ( (void*)(buffer_source[0][0][0][0]), filename, tagname, io_proc );
           if ( exitstatus != 0 ) {
             fprintf(stderr, "# [piN2piN_diagram_sum_per_type] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
             EXIT(12);
           }
           for (int time_extent = 0; time_extent < tp->T ; ++ time_extent ){
             for (int momentum_number=0; momentum_number < 343; ++momentum_number){
               for (int spin_structures=0; spin_structures < tp->number_of_gammas_source*tp->number_of_gammas_sink; ++spin_structures ){
                 for (int spin_inner=0; spin_inner < tp->d*tp->d; ++spin_inner) {
                   for (int realimag=0; realimag < 2; ++realimag){
                     buffer_sum[time_extent][momentum_number][spin_structures][spin_inner][realimag]+=buffer_source[time_extent][momentum_number][spin_structures][spin_inner][realimag];
                   }
                 }
               }
             }
           }
           fini_5level_dtable(&buffer_source);
         }//summation ends over B1,B2,W1,W2,W3,W4,Z1,Z2,Z3,Z4


         /******************************************************
          * loop on total momentum / frames
          ******************************************************/
         for ( int i_total_momentum = 0; i_total_momentum < 4; i_total_momentum++) {


           snprintf ( filename, 400, "%s%04d_PX%02dPY%02dPZ%02d_piN.h5",
                         filename_prefix,
                         Nconf,
                         momentum_orbit_pref[i_total_momentum][0],
                         momentum_orbit_pref[i_total_momentum][1],
                         momentum_orbit_pref[i_total_momentum][2]);
           fprintf ( stdout, "# [test_hdf5] create new file %s\n" );

           hid_t file_id, group_id, dataset_id, dataspace_id;  /* identifiers */
           herr_t      status;

           struct stat fileStat;
           if(stat( filename, &fileStat) < 0 ) {
             /* Open an existing file. */
             fprintf ( stdout, "# [test_hdf5] create new file\n" );
             file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
           } else {
             fprintf ( stdout, "# [test_hdf5] open existing file\n" );
             file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
           }


           /* check if the group for the source position already exists, if not then 
            * lets create it
            */
           snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/", source_coords_list[k][0],
                                                                  source_coords_list[k][1],
                                                                  source_coords_list[k][2],
                                                                  source_coords_list[k][3]);
           status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

           status = H5Gget_objinfo (file_id, tagname, 0, NULL);
           if (status != 0){

             /* Create a group named "/MyGroup" in the file. */
             group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

             /* Close the group. */
             status = H5Gclose(group_id);

           }

       
           for (int i_pf1=0; i_pf1 < num_elements[i_total_momentum]; ++i_pf1){

              for (int source_gamma =0 ; source_gamma < tp->number_of_gammas_source ; ++source_gamma ){

               for (int sink_gamma=0; sink_gamma < tp->number_of_gammas_sink; ++sink_gamma ) {

                 snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25", source_coords_list[k][0],
                                                                        source_coords_list[k][1],
                                                                        source_coords_list[k][2],
                                                                        source_coords_list[k][3]);
                 status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                 status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                 if (status != 0){

                  /* Create a group named "/MyGroup" in the file. */
                  group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                  /* Close the group. */
                  status = H5Gclose(group_id);

                 }

                 snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d", source_coords_list[k][0],
                                                                        source_coords_list[k][1],
                                                                        source_coords_list[k][2],
                                                                        source_coords_list[k][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][6],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][7],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][8]);
                 status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                 status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                 if (status != 0){

                  /* Create a group named "/MyGroup" in the file. */
                  group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                  /* Close the group. */
                  status = H5Gclose(group_id);

                 }

                 snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s", source_coords_list[k][0],
                                                                        source_coords_list[k][1],
                                                                        source_coords_list[k][2],
                                                                        source_coords_list[k][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][6],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][7],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][8],
                                                                        gamma_string_list_sink[sink_gamma],
                                                                        ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1");

                 status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                 status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                 if (status != 0){

                  /* Create a group named "/MyGroup" in the file. */
                  group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                  /* Close the group. */
                  status = H5Gclose(group_id);

                 }

                 snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/", source_coords_list[k][0],
                                                                        source_coords_list[k][1],
                                                                        source_coords_list[k][2],
                                                                        source_coords_list[k][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][6],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][7],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][8],
                                                                        gamma_string_list_sink[sink_gamma],
                                                                        ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][5]);

                 status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                 status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                 if (status != 0){

                  /* Create a group named "/MyGroup" in the file. */
                  group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                  /* Close the group. */
                  status = H5Gclose(group_id);

                 }

                 snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi25", source_coords_list[k][0],
                                                                        source_coords_list[k][1],
                                                                        source_coords_list[k][2],
                                                                        source_coords_list[k][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][6],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][7],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][8],
                                                                        gamma_string_list_sink[sink_gamma],
                                                                        ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][5]);

                 status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                 status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                 if (status != 0){

                  /* Create a group named "/MyGroup" in the file. */
                  group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                  /* Close the group. */
                  status = H5Gclose(group_id);

                 }

                 snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi25/pi2x%02dpi2y%02dpi2z%02d/", source_coords_list[k][0],
                                                                        source_coords_list[k][1],
                                                                        source_coords_list[k][2],
                                                                        source_coords_list[k][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][6],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][7],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][8],
                                                                        gamma_string_list_sink[sink_gamma],
                                                                        ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][5],
                                                                        pi2[0],
                                                                        pi2[1],
                                                                        pi2[2]);


                 status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                 status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                 if (status != 0){

                  /* Create a group named "/MyGroup" in the file. */
                  group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                  /* Close the group. */
                  status = H5Gclose(group_id);

                 }

                 snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi25/pi2x%02dpi2y%02dpi2z%02d/gi1%s,%s", source_coords_list[k][0],
                                                                        source_coords_list[k][1],
                                                                        source_coords_list[k][2],
                                                                        source_coords_list[k][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][6],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][7],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][8],
                                                                        gamma_string_list_sink[sink_gamma],
                                                                        ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][5],
                                                                        pi2[0],
                                                                        pi2[1],
                                                                        pi2[2],
                                                                        gamma_string_list_source[source_gamma],
                                                                        ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1");



                 status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                 status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                 if (status != 0){

                  /* Create a group named "/MyGroup" in the file. */
                  group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                  /* Close the group. */
                  status = H5Gclose(group_id);

                 }
                 snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi25/pi2x%02dpi2y%02dpi2z%02d/gi1%s,%s/pi1x%02dpi1y%02dpi1z%02d/", source_coords_list[k][0],
                                                                        source_coords_list[k][1],
                                                                        source_coords_list[k][2],
                                                                        source_coords_list[k][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][6],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][7],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][8],
                                                                        gamma_string_list_sink[sink_gamma],
                                                                        ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][5],
                                                                        pi2[0],
                                                                        pi2[1],
                                                                        pi2[2],
                                                                        gamma_string_list_source[source_gamma],
                                                                        ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][6]+buffer_mom[indextable[i_total_momentum][i_pf1]][3]-pi2[0],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][7]+buffer_mom[indextable[i_total_momentum][i_pf1]][4]-pi2[1],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][8]+buffer_mom[indextable[i_total_momentum][i_pf1]][5]-pi2[2]);


                 status = H5Eset_auto(NULL, H5P_DEFAULT, NULL);

                 status = H5Gget_objinfo (file_id, tagname, 0, NULL);
                 if (status != 0){

                  /* Create a group named "/MyGroup" in the file. */
                  group_id = H5Gcreate2(file_id, tagname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                  /* Close the group. */
                  status = H5Gclose(group_id);

                 }



                 hsize_t dims[3];
                 dims[0]=tp->T;
                 dims[1]=tp->d*tp->d;
                 dims[2]=2;
                 dataspace_id = H5Screate_simple(3, dims, NULL);

                 snprintf ( tagname, 400, "/sx%02dsy%02dsz%02dst%03d/gf25/pf2x%02dpf2y%02dpf2z%02d/gf1%s,%s/pf1x%02dpf1y%02dpf1z%02d/gi25/pi2x%02dpi2y%02dpi2z%02d/gi1%s,%s/pi1x%02dpi1y%02dpi1z%02d/piN", source_coords_list[k][0],
                                                                        source_coords_list[k][1],
                                                                        source_coords_list[k][2],
                                                                        source_coords_list[k][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][6],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][7],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][8],
                                                                        gamma_string_list_sink[sink_gamma],
                                                                        ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][3],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][4],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][5],
                                                                        pi2[0],
                                                                        pi2[1],
                                                                        pi2[2],
                                                                        gamma_string_list_source[source_gamma],
                                                                        ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ) ? "5" : "1",
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][6]+buffer_mom[indextable[i_total_momentum][i_pf1]][3]-pi2[0],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][7]+buffer_mom[indextable[i_total_momentum][i_pf1]][4]-pi2[1],
                                                                        buffer_mom[indextable[i_total_momentum][i_pf1]][8]+buffer_mom[indextable[i_total_momentum][i_pf1]][5]-pi2[2]);



                 /* Create a dataset in group "MyGroup". */


                 printf("tagname %s\n", tagname);
                 double ***buffer_write= init_3level_dtable(tp->T,tp->d*tp->d,2);
                 for (int time_extent = 0; time_extent < tp->T ; ++ time_extent ){
                   for (int spin_inner=0; spin_inner < tp->d*tp->d; ++spin_inner) {
                     for (int realimag=0; realimag < 2; ++realimag){
                       buffer_write[time_extent][spin_inner][realimag]=buffer_sum[time_extent][indextable[i_total_momentum][i_pf1]][source_gamma*tp->number_of_gammas_sink+sink_gamma][spin_inner][realimag];
                     }
                   }
                   if ((strcmp(gamma_string_list_source[source_gamma],"C")==0) || (strcmp(gamma_string_list_source[source_gamma],"Cg4")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_source[source_gamma],"cg3g4g5")==0) ){
                     mult_with_gamma5_matrix_source(buffer_write[time_extent]);

                   }
                   if ((strcmp(gamma_string_list_sink[sink_gamma],"C")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"Cg4")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg1g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg2g4g5")==0) || (strcmp(gamma_string_list_sink[sink_gamma],"cg3g4g5")==0) ){
                     mult_with_gamma5_matrix_sink(buffer_write[time_extent]);
                   }

                 }

                 /* Write the first dataset. */
                 status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(buffer_write[0][0][0]));

                 /* Close the data space for the first dataset. */
                 status = H5Sclose(dataspace_id);

                 /* Close the first dataset. */
                 status = H5Dclose(dataset_id);

                 fini_3level_dtable(&buffer_write);

               } /* sink gamma */

             } /*source gamma */


           }/*loop on pf1 momentum corresponding to a given ptot momentum */
           /* Close the file. */
           status = H5Fclose(file_id);

         }//ptot


         fini_5level_dtable(&buffer_sum);
     
       } /* loop on sequential momentum */

       fini_1level_itable(&twopt_id_list);
       for (int j=0;j<12; ++j){
         free(hdf5_diag_tag_list_tag[j]);
         free(hdf5_diag_tag_list_name[j]);
       }
       free(hdf5_diag_tag_list_tag);
       free(hdf5_diag_tag_list_name);

       for (int j=0; j<tp->number_of_gammas_sink; ++j)
         free(gamma_string_list_sink[j]);
       free(gamma_string_list_sink);

       for (int j=0; j<tp->number_of_gammas_source; ++j)
         free(gamma_string_list_source[j]);
       free(gamma_string_list_source);

     } /*name of the twopoint functions pixN-pixN*/

   } /*end of loop on the source positions */
   for (int i_total_momentum=0; i_total_momentum<4; ++i_total_momentum)
     free(indextable[i_total_momentum]);
   free(indextable);
   free(num_elements);

   fini_2level_itable(&buffer_mom);


  } /*end of creating B1,B2,W1,W2,W3,W4,Z1,Z2,Z3,Z4 */


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
