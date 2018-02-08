/***************************************************
 * gsp_utils.cpp
 *
 * Di 6. Feb 15:02:07 CET 2018
 *
 ***************************************************/
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
#include <omp.h>
#endif

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "cvc_complex.h"
#include "global.h"
#include "ilinalg.h"
#include "cvc_geometry.h"
#include "io_utils.h"
#include "read_input_parser.h"
#include "cvc_utils.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "scalar_products.h"
#include "iblas.h"
#include "project.h"
#include "matrix_init.h"
#include "gsp.h"


namespace cvc {

const int gamma_adjoint_sign[16] = {
  /* the sequence is:
       0, 1, 2, 3, id, 5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
       0  1  2  3   4  5    6    7    8    9   10   11   12   13   14   15 */
       1, 1, 1, 1,  1, 1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1
};


/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 ***********************************************************************************************
 **
 ** gsp_read_node
 ** - read actually 6 x T aff nodes from file ( or read binary file )
 **
 ***********************************************************************************************
 ***********************************************************************************************/
int gsp_read_node (double _Complex ***gsp, int numV, int momentum[3], int gamma_id, char *prefix, char*tag, int timeslice ) {

  char filename[200];
  int exitstatus;

  const int gsp_name_num = 6;
  const char *gsp_name_list[gsp_name_num] = { "v-v", "w-v", "w-w", "xv-xv", "xw-xv", "xw-xw" };

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex *aff_buffer = NULL;
  char aff_buffer_path[200];
  /*  uint32_t aff_buffer_size; */
#endif

#ifdef HAVE_LHPC_AFF
  aff_status_str = (char*)aff_version();
  fprintf(stdout, "# [gsp_read_node] using aff version %s\n", aff_status_str);

  sprintf(filename, "%s.aff", prefix );
  fprintf(stdout, "# [gsp_read_node] reading gsp data from file %s\n", filename);
  affr = aff_reader(filename);

  aff_status_str = (char*)aff_reader_errstr(affr);
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[gsp_read_node] Error from aff_reader, status was %s\n", aff_status_str);
    return(1);
  }

  if( (affn = aff_reader_root(affr)) == NULL ) {
    fprintf(stderr, "[gsp_read_node] Error, aff reader is not initialized\n");
    return(2);
  }

  exitstatus = init_1level_zbuffer ( &aff_buffer, numV * numV );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[gsp_read_node] Error from init_1level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }
#endif

#ifdef HAVE_LHPC_AFF
  uint32_t items = (uint32_t)numV * numV;

  for ( int igsp = 0; igsp < gsp_name_num; igsp++ ) {

    sprintf ( aff_buffer_path, "%s/%s/t%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, gsp_name_list[igsp], timeslice+g_proc_coords[0]*T, momentum[0], momentum[1], momentum[2], gamma_id );
    if(g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [gsp_read_node] current aff path = %s\n", aff_buffer_path);
    affdir = aff_reader_chpath(affr, affn, aff_buffer_path);
    if ( ( exitstatus = aff_node_get_complex (affr, affdir, gsp[igsp][0], (uint32_t)items) ) != 0 ) {
      fprintf(stderr, "[gsp_read_node] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(4);
    }
  }  /* end of loop on gsp names */
#else

  sprintf(filename, "%s.t%.2d.px%.2dpy%.2dpz%.2d.g%.2d.dat", prefix, timeslice+g_proc_coords[0]*T, momentum[0], momentum[1], momentum[2], gamma_id);
  FILE *ifs = fopen(filename, "r");
  if(ifs == NULL) {
    fprintf(stderr, "[gsp_read_node] Error, could not open file %s for writing\n", filename);
    return(5);
  }

  size_t items = 6 * numV * numV;


  if( fread( gsp[0][0], sizeof(double _Complex), items, ifs) != items ) {
    fprintf(stderr, "[gsp_read_node] Error, could not read proper amount of data to file %s\n", filename);
    return(7);
  }

  fclose(ifs);

  }  /* end of loop on timeslices */
/*
  byte_swap64_v2( (double*)(gsp[0][0][0]), 12*T*numV*numV);
*/

#endif

#ifdef HAVE_LHPC_AFF
  aff_reader_close (affr);
#endif

  return(0);

}  /* end of gsp_read_node */

/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 *
 ***********************************************************************************************/
STOPPED HERE
int gsp_prepare_from_file ( double _Complex ***gsp, int numV, int momentum[3], int gamma_id, int ns, char *prefix, char*tag ) {

  int pvec[3];
  int gid;
  int sign = 0;
  if ( ns == 0 ) {
    int momid = -1;
    for ( int i = 0, i < g_sink_momentum_number; i++ ) {
      if ( 
          ( g_sink_momentum_list[i][0] = -momentum[0] ) && 
          ( g_sink_momentum_list[i][1] = -momentum[1] ) && 
          ( g_sink_momentum_list[i][2] = -momentum[2] )  ) {
        momid = i;
        break;
      }
    }
    if ( momid == -1 ) {
      fprintf ( stderr, "[gsp_prepare_from_file] Error from gsp_read_node, status was %d\n", exitstatus );
      return(1);
    }
    memcpy ( pvec, g_sink_momentum_list[imom], 3*sizeof(int) );

  } else if ( ns == 1 ) {
    memcpy ( pvec, momentum, 3*sizeof(int) );
    gid  = gamma_id;
    sign = 1;
  }

  double _Complex ***gsp_buffer = NULL;
  exitstatus = init_3level_zbuffer ( &gsp_buffer, 6, numV, numV );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[gsp_prepare_from_file] Error from init_3level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

  for ( int x0 = 0; x0 < T; x0++ ) {

    exitstatus = gsp_read_node ( gsp_buffer, numV, pvec, gid, prefix, tag, x0 );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[gsp_prepare_from_file] Error from gsp_read_node, status was %d\n", exitstatus );
      return(1);
    }

  }  /* end of loop on timeslices */

  fini_3level_zbuffer ( &gsp_buffer );
  return(0);
}  /* gsp_prepare_from_file */

/***********************************************************************************************/
/***********************************************************************************************/

int gsp_write_eval(double *eval, int num, char*tag) {
  
  double ratime, retime;
  char filename[200];

#ifdef HAVE_LHPC_AFF
  int status;
  struct AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  char aff_buffer_path[200];
/*  uint32_t aff_buffer_size; */
#else
  FILE *ofs = NULL;
#endif
 
  ratime = _GET_TIME;

  if(g_cart_id == 0) {
  /***********************************************
   * output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
    sprintf(filename, "%s.aff", tag);
    fprintf(stdout, "# [gsp_write_eval] writing eigenvalue data from file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[gsp_write_eval] Error from aff_writer, status was %s\n", aff_status_str);
      return(1);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[gsp_write_eval] Error, aff writer is not initialized\n");
      return(2);
    }

    sprintf(aff_buffer_path, "/%s/eigenvalues", tag);
    fprintf(stdout, "# [gsp_write_eval] current aff path = %s\n", aff_buffer_path);

    affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
    status = aff_node_put_double (affw, affdir, eval, (uint32_t)num); 
    if(status != 0) {
      fprintf(stderr, "[gsp_write_eval] Error from aff_node_put_double, status was %d\n", status);
      return(3);
    }
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[gsp_write_eval] Error from aff_writer_close, status was %s\n", aff_status_str);
      return(4);
    }
#else
    sprintf(filename, "%s.eval", tag );
    ofs = fopen(filename, "w");
    if(ofs == NULL) {
      fprintf(stderr, "[gsp_write_eval] Error, could not open file %s for writing\n", filename);
      return(5);
    }
    for( int ievecs = 0; ievecs < num; ievecs++ ) {
      fprintf(ofs, "%25.16e\n", eval[ievecs] );
    }
    fclose(ofs);
#endif
  }  /* end of if g_cart_id == 0 */

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [gsp_write_eval] time for gsp_write_eval = %e seconds\n", retime-ratime);

  return(0);
}  /* end of gsp_write_eval */

/***********************************************************************************************/
/***********************************************************************************************/

int gsp_read_eval(double **eval, int num, char*tag) {
  
  double ratime, retime;
  char filename[200];

#ifdef HAVE_LHPC_AFF
  int status;
  struct AffReader_s *affr = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  char aff_buffer_path[200];
/*  uint32_t aff_buffer_size; */
#else
  FILE *ifs = NULL;
#endif


  ratime = _GET_TIME;

  /***********************************************
   * allocate
   ***********************************************/
  if(*eval == NULL) {
    *eval = (double*)malloc(num*sizeof(double));
    if(*eval == NULL) {
      fprintf(stderr, "[gsp_read_eval] Error from malloc\n");
      return(10);
    }
  }

  /***********************************************
   * input file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  sprintf(filename, "%s.aff", tag);
  fprintf(stdout, "# [gsp_read_eval] reading eigenvalue data from file %s\n", filename);
  affr = aff_reader(filename);
  aff_status_str = (char*)aff_reader_errstr(affr);
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[gsp_read_eval] Error from aff_reader, status was %s\n", aff_status_str);
    return(1);
  }

  if( (affn = aff_reader_root(affr)) == NULL ) {
    fprintf(stderr, "[gsp_read_eval] Error, aff reader is not initialized\n");
    return(2);
  }

  sprintf(aff_buffer_path, "/%s/eigenvalues", tag);
  fprintf(stdout, "# [gsp_read_eval] current aff path = %s\n", aff_buffer_path);

  uint32_t items = num;
  affdir = aff_reader_chpath(affr, affn, aff_buffer_path);
  status = aff_node_get_double (affr, affdir, *eval, items ); 
  if(status != 0) {
    fprintf(stderr, "[gsp_read_eval] Error from aff_node_put_double, status was %d\n", status);
    return(3);
  }
  aff_reader_close (affr);
#else
  sprintf(filename, "%s.eval", tag );
  ifs = fopen(filename, "r");
  if(ifs == NULL) {
    fprintf(stderr, "[gsp_read_eval] Error, could not open file %s for reading\n", filename);
    return(5);
  }
  for( int ievecs = 0; ievecs < num; ievecs++ ) {
    if( fscanf(ifs, "%lf", (*eval)+ievecs ) != 1 ) {
      return(6);
    }
  }
  fclose(ifs);
#endif
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [gsp_read_eval] time for gsp_read_eval = %e seconds\n", retime-ratime);

  return(0);
}  /* end of gsp_read_eval */

/***********************************************************************************************/
/***********************************************************************************************/

}  /* end of namespace cvc */
