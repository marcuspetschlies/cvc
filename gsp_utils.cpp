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
#include "table_init_z.h"
#include "rotations.h"
#include "gsp_utils.h"


namespace cvc {

#if 0
const int gamma_adjoint_sign[16] = {
  /* the sequence is:
       0, 1, 2, 3, id, 5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
       0  1  2  3   4  5    6    7    8    9   10   11   12   13   14   15 */
       1, 1, 1, 1,  1, 1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1
};
#endif  // of if 0

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

  double _Complex * aff_buffer = init_1level_ztable ( (size_t)numV * (size_t)numV );
  if( aff_buffer == NULL ) {
    fprintf(stderr, "[gsp_read_node] Error from init_1level_ztable %s %d\n",  __FILE__, __LINE__);
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

/*
  byte_swap64_v2( (double*)(gsp[0][0][0]), 12*T*numV*numV);
*/

#endif

#ifdef HAVE_LHPC_AFF
  fini_1level_ztable (&aff_buffer );
  aff_reader_close (affr);
#endif

  return(0);

}  /* end of gsp_read_node */

/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 *
 ***********************************************************************************************/
int gsp_prepare_from_file ( double _Complex ***gsp, int numV, int momentum[3], int gamma_id, int ns, char *prefix, char*tag ) {

  int pvec[3];
  int g5_gamma_id = -1;
  int sign = 0;
  double _Complex Z_1 = 1.;

  memset ( gsp[0][0], 0, numV*numV*T*sizeof(double _Complex) );

  if ( ns == 0 ) {
    int momid = -1;
    for ( int i = 0; i < g_sink_momentum_number; i++ ) {
      if ( 
          ( g_sink_momentum_list[i][0] = -momentum[0] ) && 
          ( g_sink_momentum_list[i][1] = -momentum[1] ) && 
          ( g_sink_momentum_list[i][2] = -momentum[2] )  ) {
        momid = i;
        break;
      }
    }
    if ( momid == -1 ) {
      fprintf ( stderr, "[gsp_prepare_from_file] Error, no inverse momentum found for (%3d, %3d, %3d) %s %d\n",
          momentum[0], momentum[1], momentum[2], __FILE__, __LINE__ );
      return(1);
    }
    memcpy ( pvec, g_sink_momentum_list[momid], 3*sizeof(int) );

    g5_gamma_id  = g_gamma_mult_table[5][gamma_id];
    sign         = g_gamma_mult_sign[5][gamma_id] * g_gamma_adjoint_sign[g5_gamma_id];

  } else if ( ns == 1 ) {
    memcpy ( pvec, momentum, 3*sizeof(int) );
    g5_gamma_id  = g_gamma_mult_table[5][gamma_id];
    sign         = g_gamma_mult_sign[5][gamma_id];
  }

  double _Complex ***gsp_buffer = init_3level_ztable ( 6, (size_t)numV, (size_t)numV );
  if ( gsp_buffer == NULL ) {
    fprintf ( stderr, "[gsp_prepare_from_file] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( g_cart_id == 0 && g_verbose > 1 ) {
    fprintf ( stdout, "# [] ns %d mom %3d %3d %3d pvec %3d %3d %3d gamma_id %2d g5 gamma_id %2d sign %d", ns, 
        momentum[0], momentum[1], momentum[2], pvec[0], pvec[1], pvec[2], gamma_id, g5_gamma_id, sign );
  }

  /***********************************************
   * loop on timeslices
   ***********************************************/
  for ( int x0 = 0; x0 < T; x0++ ) {

    int exitstatus = gsp_read_node ( gsp_buffer, numV, pvec, g5_gamma_id, prefix, tag, x0 );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[gsp_prepare_from_file] Error from gsp_read_node, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(1);
    }

    /***********************************************
     * add parts depending on ns
     *
     * ordering { "v-v", "w-v", "w-w", "xv-xv", "xw-xv", "xw-xw" };
     ***********************************************/
    if ( ns )  {
      /***********************************************
       * add v-v, w-w, xv-xv, xw-xw
       ***********************************************/
      
      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[0], Z_1, numV );
      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[2], Z_1, numV );
      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[3], Z_1, numV );
      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[5], Z_1, numV );

    } else {
      /***********************************************
       * add v-w, xv-xw
       ***********************************************/

      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[1], Z_1, numV );
      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[4], Z_1, numV );
    }

  }  /* end of loop on timeslices */

  fini_3level_ztable ( &gsp_buffer );
  return(0);
}  /* gsp_prepare_from_file */

/***********************************************************************************************/
/***********************************************************************************************/

int gsp_write_eval(double * const eval, unsigned int const num, char * const filename_prefix, char * const tag) {
  
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
    sprintf(filename, "%s.aff", filename_prefix );
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
    uint32_t unum = num;
    status = aff_node_put_double (affw, affdir, eval, unum); 
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
    sprintf(filename, "%s.eval", filename_prefix );
    ofs = fopen(filename, "w");
    if(ofs == NULL) {
      fprintf(stderr, "[gsp_write_eval] Error, could not open file %s for writing\n", filename);
      return(5);
    }
    for( unsigned int ievecs = 0; ievecs < num; ievecs++ ) {
      fprintf(ofs, "%25.16e\n", eval[ievecs] );
    }
    fclose(ofs);
#endif
  }  /* end of if g_cart_id == 0 */

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [gsp_write_eval] time for gsp_write_eval = %e seconds\n", retime-ratime);

  return(0);
}  // end of gsp_write_eval

/***********************************************************************************************/
/***********************************************************************************************/

int gsp_read_eval(double ** eval, unsigned int num, char * const filename_prefix, char * const tag) {
  
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
      fprintf(stderr, "[gsp_read_eval] Error from malloc %s %d\n", __FILE__, __LINE__ );
      return(10);
    }
  }

  /***********************************************
   * input file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  sprintf(filename, "%s.aff", filename_prefix );
  fprintf(stdout, "# [gsp_read_eval] reading eigenvalue data from file %s\n", filename);
  affr = aff_reader(filename);
  aff_status_str = (char*)aff_reader_errstr(affr);
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[gsp_read_eval] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
    return(1);
  }

  if( (affn = aff_reader_root(affr)) == NULL ) {
    fprintf(stderr, "[gsp_read_eval] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  sprintf(aff_buffer_path, "/%s/eigenvalues", tag);
  fprintf(stdout, "# [gsp_read_eval] current aff path = %s\n", aff_buffer_path);

  uint32_t items = (uint32_t)num;
  affdir = aff_reader_chpath(affr, affn, aff_buffer_path);
  status = aff_node_get_double (affr, affdir, *eval, items ); 
  if(status != 0) {
    fprintf(stderr, "[gsp_read_eval] Error from aff_node_put_double, status was %d %s %d\n", status, __FILE__, __LINE__ );
    return(3);
  }
  aff_reader_close (affr);
#else
  sprintf(filename, "%s.eval", filename_prefix );
  ifs = fopen(filename, "r");
  if(ifs == NULL) {
    fprintf(stderr, "[gsp_read_eval] Error, could not open file %s for reading %s %d\n", filename, __FILE__, __LINE__ );
    return(5);
  }
  for( unsigned int ievecs = 0; ievecs < num; ievecs++ ) {
    if( fscanf(ifs, "%lf", (*eval)+ievecs ) != 1 ) {
      return(6);
    }
  }
  fclose(ifs);
#endif

  if ( g_verbose > 3 ) {
    for ( int i = 0; i < num; i++ ) fprintf ( stdout, "# [gsp_read_eval] eval %4d  %25.16e\n", i,  (*eval)[i] );
  }


  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [gsp_read_eval] time for gsp_read_eval = %e seconds\n", retime-ratime);

  return(0);
}  // end of gsp_read_eval

/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 * read V^+ Gamma_cvc W from file
 ***********************************************************************************************/

/***********************************************************************************************
 * gsp_read_cvc_node
 ***********************************************************************************************/
int gsp_read_cvc_node (
    double _Complex ** const fac,
    unsigned int const numV,
    unsigned int const block_length,
    int const momentum[3],
    char * const type,
    int const mu,
    char * const prefix,
    char * const tag,
    int const timeslice,
    unsigned int const nt
) {

  char filename[200];
  char key_prefix[200];
  char key[500];
  int exitstatus;
  int const numB = numV / block_length;
  int const file_t = ( timeslice / (T_global/nt) )  * (T_global/nt);

  if ( g_verbose > 2 ) fprintf ( stdout, "# [gsp_read_cvc_node] timeslice for filename = %d\n", file_t );

#ifdef HAVE_LHPC_AFF

  sprintf(filename, "%s.t%.2d.aff", prefix, file_t );
  fprintf(stdout, "# [gsp_read_cvc_node] reading gsp data from file %s\n", filename);
  struct AffReader_s * affr = aff_reader(filename);

  const char * aff_status_str = aff_reader_errstr(affr);
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[gsp_read_cvc_node] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    return(1);
  }

  struct AffNode_s *affn = aff_reader_root(affr);
  if( affn == NULL ) {
    fprintf(stderr, "[gsp_read_cvc_node] Error, aff reader is not initialized\n");
    return(2);
  }
#endif
 
  double _Complex ** buffer = init_2level_ztable ( (size_t)numV, (size_t)block_length );
  if( buffer == NULL ) {
    fprintf(stderr, "[gsp_read_cvc_node] Error from init_2level_ztable %s %d\n",  __FILE__, __LINE__);
    return(3);
  }

  // set the key prefix
  if ( strcmp( type, "mu" ) == 0 ) {
    sprintf ( key_prefix, "%s/t%.2d/%s%d", tag, timeslice, type, mu );
  } else {
    sprintf ( key_prefix, "%s/%s/t%.2d", tag, type, timeslice );
  }

#ifdef HAVE_LHPC_AFF
  uint32_t uitems = (uint32_t)numV * block_length;

  // loop on blocks
  for ( int ib = 0; ib < numB; ib++ ) {

    sprintf ( key, "%s/b%.2d/px%.2dpy%.2dpz%.2d", key_prefix, ib, momentum[0], momentum[1], momentum[2] );
    if(g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [gsp_read_cvc_node] key = %s\n", key );
    struct AffNode_s * affdir = aff_reader_chpath ( affr, affn, key );
    if ( ( exitstatus = aff_node_get_complex ( affr, affdir, buffer[0], (uint32_t)uitems) ) != 0 ) {
      fprintf(stderr, "[gsp_read_cvc_node] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(4);
    }
#else

    size_t items = numV * block_length;
   
    if ( strcmp( type, "mu" ) == 0 ) {
      sprintf(filename, "%s.t%.2d.%s%d.b%.2d.px%.2dpy%.2dpz%.2d.dat", prefix, file_t, type, mu, ib, momentum[0], momentum[1], momentum[2] );
    } else {
      sprintf(filename, "%s.t%.2d.%s.b%.2d.px%.2dpy%.2dpz%.2d.dat", prefix, file_t, type, ib, momentum[0], momentum[1], momentum[2] );
    }

    FILE *ifs = fopen(filename, "r");
    if(ifs == NULL) {
      fprintf(stderr, "[gsp_read_cvc_node] Error, could not open file %s for writing\n", filename);
      return(5);
    }

    if( fread( buffer[0], sizeof(double _Complex), items, ifs) != items ) {
      fprintf(stderr, "[gsp_read_cvc_node] Error, could not read proper amount of data to file %s\n", filename);
      return(7);
    }
    fclose(ifs);
    // byte_swap64_v2( (double*)(gsp[0][0][0]), 12*T*numV*numV);
#endif

    size_t const bytes = block_length * sizeof( double _Complex );

    for ( unsigned int iv = 0; iv < numV; iv++ ) {
      memcpy ( &(fac[iv][ib*block_length]), buffer[iv], bytes );
    }

  }  // end of loop on blocks

#ifdef HAVE_LHPC_AFF
  aff_reader_close (affr);
#endif

  fini_2level_ztable ( &buffer );
  return(0);

}  // end of gsp_read_cvc_node

/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 * gsp_read_cvc_node
 ***********************************************************************************************/
int gsp_read_cvc_mee_node (
    double _Complex **** const fac,
    unsigned int const numV,
    int const momentum[3],
    char * const prefix,
    char * const tag,
    int const timeslice
) {

  char filename[200];
  char key[500];
  int exitstatus;
  int const file_t = 0;

  if ( g_verbose > 2 ) fprintf ( stdout, "# [gsp_read_cvc_mee_node] timeslice for filename = %d\n", file_t );

#ifdef HAVE_LHPC_AFF

  sprintf(filename, "%s.t%.2d.aff", prefix, file_t );
  fprintf(stdout, "# [gsp_read_cvc_mee_node] reading gsp data from file %s\n", filename);
  struct AffReader_s * affr = aff_reader(filename);

  const char * aff_status_str = aff_reader_errstr(affr);
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[gsp_read_cvc_mee_node] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    return(1);
  }

  struct AffNode_s *affn = aff_reader_root(affr);
  if( affn == NULL ) {
    fprintf(stderr, "[gsp_read_cvc_mee_node] Error, aff reader is not initialized\n");
    return(2);
  }
 

  for ( int imu = 0; imu < 4; imu++ ) {
    for ( int inu = 0; inu < 4; inu++ ) {

      for ( int idt = 0; idt < 3; idt++ ) {

        // set the key prefix
        sprintf ( key, "%s/mu%d/nu%d/px%.2dpy%.2dpz%.2d/t%.2d/dt%d", tag, imu, inu, momentum[0], momentum[1], momentum[2], timeslice, idt-1 );


        uint32_t uitems = (uint32_t)numV;

        if(g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [gsp_read_cvc_mee_node] key = %s\n", key );
        struct AffNode_s * affdir = aff_reader_chpath ( affr, affn, key );
        if ( ( exitstatus = aff_node_get_complex ( affr, affdir, fac[imu][inu][idt], (uint32_t)uitems) ) != 0 ) {
          fprintf(stderr, "[gsp_read_cvc_mee_node] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }
      }  // end of loop on dt
    }  // end of loop on nu
  }  // end of loop on mu

  aff_reader_close (affr);
#else

  fprintf(stderr, "[gsp_read_cvc_mee_node] Error, FILE stream read-in not yet implemented\n");
  return(5);

#endif

  return(0);

}  // end of gsp_read_cvc_mee_node


/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 * gsp_read_cvc_node
 ***********************************************************************************************/
int gsp_read_cvc_mee_ct_node (
    double _Complex *** const fac,
    unsigned int const numV,
    int const momentum[3],
    char * const prefix,
    char * const tag,
    int const timeslice
) {

  char filename[200];
  char key[500];
  int exitstatus;
  int const file_t = 0;

  if ( g_verbose > 2 ) fprintf ( stdout, "# [gsp_read_cvc_mee_ct_node] timeslice for filename = %d\n", file_t );

#ifdef HAVE_LHPC_AFF

  sprintf(filename, "%s.t%.2d.aff", prefix, file_t );
  fprintf(stdout, "# [gsp_read_cvc_mee_ct_node] reading gsp data from file %s\n", filename);
  struct AffReader_s * affr = aff_reader(filename);

  const char * aff_status_str = aff_reader_errstr(affr);
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[gsp_read_cvc_mee_ct_node] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    return(1);
  }

  struct AffNode_s *affn = aff_reader_root(affr);
  if( affn == NULL ) {
    fprintf(stderr, "[gsp_read_cvc_mee_ct_node] Error, aff reader is not initialized\n");
    return(2);
  }
 

  for ( int imu = 0; imu < 4; imu++ ) {

    for ( int idt = 0; idt < 5; idt++ ) {

      // set the key prefix
      sprintf ( key, "%s/mu%d/px%.2dpy%.2dpz%.2d/t%.2d/dt%.2d", tag, imu, momentum[0], momentum[1], momentum[2], timeslice, idt-2 );

      uint32_t uitems = (uint32_t)numV;

      if(g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [gsp_read_cvc_mee_ct_node] key = %s\n", key );
      struct AffNode_s * affdir = aff_reader_chpath ( affr, affn, key );
      if ( ( exitstatus = aff_node_get_complex ( affr, affdir, fac[imu][idt], (uint32_t)uitems) ) != 0 ) {
        fprintf(stderr, "[gsp_read_cvc_mee_ct_node] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(4);
      }
    }  // end of loop on dt
  }  // end of loop on mu

  aff_reader_close (affr);

  return ( 0 );
#else
  return ( 1 );
#endif

}  // end of gsp_read_cvc_mee_ct_node

/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 * gsp_read_cvc_node
 ***********************************************************************************************/
int gsp_read_cvc_ct_node (
    double _Complex ** const fac,
    unsigned int const numV,
    char * const prefix,
    char * const tag,
    int const timeslice
) {

  char filename[200];
  char key[500];
  int exitstatus;
  int const file_t = 0;

  if ( g_verbose > 2 ) fprintf ( stdout, "# [gsp_read_cvc_ct_node] timeslice for filename = %d\n", file_t );

#ifdef HAVE_LHPC_AFF

  sprintf(filename, "%s.t%.2d.aff", prefix, file_t );
  fprintf(stdout, "# [gsp_read_cvc_ct_node] reading gsp data from file %s\n", filename);
  struct AffReader_s * affr = aff_reader(filename);

  const char * aff_status_str = aff_reader_errstr(affr);
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[gsp_read_cvc_ct_node] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    return(1);
  }

  struct AffNode_s *affn = aff_reader_root(affr);
  if( affn == NULL ) {
    fprintf(stderr, "[gsp_read_cvc_ct_node] Error, aff reader is not initialized\n");
    return(2);
  }
 

  for ( int imu = 0; imu < 4; imu++ ) {

    // set the key
    sprintf ( key, "%s/mu%d/t%.2d", tag, imu, timeslice );

    uint32_t uitems = (uint32_t)numV;

    if(g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [gsp_read_cvc_ct_node] key = %s\n", key );
    struct AffNode_s * affdir = aff_reader_chpath ( affr, affn, key );
    if ( ( exitstatus = aff_node_get_complex ( affr, affdir, fac[imu], (uint32_t)uitems) ) != 0 ) {
      fprintf(stderr, "[gsp_read_cvc_ct_node] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(4);
    }
  }  // end of loop on mu

  aff_reader_close (affr);

  return ( 0 );
#else
  return ( 1 );
#endif

}  // end of gsp_read_cvc_ct_node


/***********************************************************************************************/
/***********************************************************************************************/

#if 0
/***********************************************************************************************
 *
 ***********************************************************************************************/
int gsp_prepare_cvc_from_file ( double _Complex ***gsp, int numV, int momentum[3], int gamma_id, int ns, char *prefix, char*tag ) {

  int pvec[3];
  int g5_gamma_id;
  int sign = 0;
  double _Complex Z_1 = 1.;

  memset ( gsp[0][0], 0, numV*numV*T*sizeof(double _Complex) );

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
      fprintf ( stderr, "[gsp_prepare_cvc_from_file] Error from gsp_read_node, status was %d\n", exitstatus );
      return(1);
    }
    memcpy ( pvec, g_sink_momentum_list[imom], 3*sizeof(int) );

    g5_gamma_id  = g_gamma_mult_table[5][gamma_id];
    sign         = g_gamma_mult_sign[5][gamma_id] * g_gamma_adjoint_sign[g5_gamma_id];

  } else if ( ns == 1 ) {
    memcpy ( pvec, momentum, 3*sizeof(int) );
    g5_gamma_id  = g_gamma_mult_table[5][gamma_id];
    sign         = g_gamma_mult_sign[5][gamma_id];
  }

  double _Complex ***gsp_buffer = init_3level_ztable ( 6, (size_t)numV, (size_t)numV );
  if ( gsp_buffer == NULL ) {
    fprintf ( stderr, "[gsp_prepare_cvc_from_file] Error from init_3level_ztable %s %d\n",  __FILE__, __LINE__ );
    return(1);
  }

  if ( g_cart_id == 0 && g_verbose > 1 ) {
    fprintf ( stdout, "# [] ns %d mom %3d %3d %3d pvec %3d %3d %3d gamma_id %2d g5 gamma_id %2d sign %d", ns, 
        momentum[0], momentum[1], momentum[2], pvec[0], pvec[1], pvec[2], gamma_id, g5_gamma_id, sign );
  }

  /***********************************************
   * loop on timeslices
   ***********************************************/
  for ( int x0 = 0; x0 < T; x0++ ) {

    exitstatus = gsp_read_node ( gsp_buffer, numV, pvec, g5_gamma_id, prefix, tag, x0 );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[gsp_prepare_cvc_from_file] Error from gsp_read_node, status was %d\n", exitstatus );
      return(1);
    }

    /***********************************************
     * add parts depending on ns
     *
     * ordering { "v-v", "w-v", "w-w", "xv-xv", "xw-xv", "xw-xw" };
     ***********************************************/
    if ( ns )  {
      /***********************************************
       * add v-v, w-w, xv-xv, xw-xw
       ***********************************************/
      
      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[0], Z_1, numV );
      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[2], Z_1, numV );
      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[3], Z_1, numV );
      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[5], Z_1, numV );

    } else {
      /***********************************************
       * add v-w, xv-xw
       ***********************************************/

      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[1], Z_1, numV );
      rot_mat_pl_eq_mat_ti_co ( gsp[x0], gsp_buffer[4], Z_1, numV );
    }

  }  /* end of loop on timeslices */

  fini_3level_ztable ( &gsp_buffer );
  return(0);
}  // gsp_prepare_cvc_from_file */
#endif  // if 0

/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 *
 ***********************************************************************************************/
int gsp_ft_p0_shift ( double _Complex * const s_out, double _Complex * const s_in, int const pvec[3], int const mu , int const nu, int const sign ) {

  double _Complex * r = init_1level_ztable ( (size_t)T );
  if ( r == NULL ) return( 1 );

  double const p3[3] = {
    sign * M_PI * pvec[0] / (double)LX_global,
    sign * M_PI * pvec[1] / (double)LY_global,
    sign * M_PI * pvec[2] / (double)LZ_global 
  };

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( int ip0 = 0; ip0 < T; ip0 ++ ) {

    double const p[4] = { sign * M_PI * ip0 / (double)T_global, p3[0], p3[1], p3[2] };

    // loop on x0
    for ( int it = 0; it < T; it++ ) {
      double const phase = p[0] * 2 * ( it + g_proc_coords[0] * T);
      double _Complex const ephase = cexp ( phase * I );
      r[ip0] += s_in[it] * ephase;
    }

    // shift if mu = 
    double const dp = ( ( mu >= 0 && mu < 4 ) ? p[mu] : 0 ) - ( ( nu >= 0 && nu < 4 ) ? p[nu] : 0 );
    r[ip0] *= cexp( dp * I );

  }  // end of loop on p0

#ifdef HAVE_MPI 
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY )|| ( defined PARALLELTXYZ )
  if ( MPI_Allreduce( r, s_out, 2*T, MPI_DOUBLE, MPI_SUM, g_tr_comm   ) != MPI_SUCCESS ) return(2);
#  else
  if ( MPI_Allreduce( r, s_out, 2*T, MPI_DOUBLE, MPI_SUM, g_cart_grid ) != MPI_SUCCESS ) return(2);
#  endif
#else
  memcpy ( s_out, r, T*sizeof( double _Complex ) );
#endif

  fini_1level_ztable ( &r );
  return( 0 );

}  // end of gsp_ft_p0_shift

/***********************************************************************************************/
/***********************************************************************************************/

}  /* end of namespace cvc */
