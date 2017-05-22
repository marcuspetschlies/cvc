/****************************************************
 * contract_factorized.c
 * 
 * Mon May 22 17:12:00 CEST 2017
 *
 * PURPOSE:
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
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
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "matrix_init.h"
#include "project.h"

namespace cvc {

/******************************************************
 *
 ******************************************************/
int contract_v3  (double **v3, double*phi, fermion_propagator_type*prop, unsigned int N ) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for(unsigned int ix=0; ix < N; ix++) {
    _v3_eq_fv_dot_fp( v3[ix], phi+_GSI(ix), prop[ix]);
  }
}  /* contract_v3 */

/******************************************************
 *
 ******************************************************/
int contract_v2 (double **v2, double *phi, fermion_propagator_type *prop1, fermion_propagator_type *prop2, unsigned int N ) {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  double v1[72];
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(unsigned int ix=0; ix < N; ix++) {
    _v1_eq_fv_eps_fp( v1, phi+_GSI(ix), prop1[ix] );

    _v2_eq_v1_eps_fp( v2[ix], v1, prop2[ix] );
  }

#ifdef HAVE_OPENMP
}
#endif
}  /* end of contract_v2 */

/******************************************************
 *
 ******************************************************/
int contract_v1 (double **v1, double *phi, fermion_propagator_type *prop1, unsigned int N ) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for(unsigned int ix=0; ix < N; ix++) {
    _v1_eq_fv_eps_fp( v1[ix], phi+_GSI(ix), prop1[ix] );
  }
}  /* end of contract_v1 */

}  /* end of namespace cvc */


#if 0
int contract_vn_write_to_aff_file (double _Complex **v, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc ) {

  int exitstatus, i;
  double ratime, retime;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_buffer_path[200];
  double *buffer = NULL;
  double _Complex *aff_buffer = NULL;
  double _Complex *zbuffer = NULL;

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }

    zbuffer = (double _Complex*)malloc(  momentum_number * T_global * sizeof(double _Complex) );
    if( zbuffer == NULL ) {
      fprintf(stderr, "[] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(6);
    }
  }

  ratime = _GET_TIME;

  /* reorder cvc_tp into buffer with order time - momentum */
  buffer = (double*)malloc(  momentum_number * T * sizeof(double _Complex) );
  if( buffer == NULL ) {
    fprintf(stderr, "[] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
  i = 0;
  for( int it = 0; it < T; it++ ) {
    for( int ip=0; ip<momentum_number; ip++) {
      buffer[i] = v[ip][it];
      buffer[i++] = cvc_tp[ip][mu][2*it+1];
    }
  }

#ifdef HAVE_MPI
  i = momentum_number * 32 * T;
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  if(io_proc>0) {
    exitstatus = MPI_Gather(buffer, i, MPI_DOUBLE, zbuffer, i, MPI_DOUBLE, 0, g_tr_comm);
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(3);
    }
  }
#  else
  exitstatus = MPI_Gather(buffer, i, MPI_DOUBLE, zbuffer, i, MPI_DOUBLE, 0, g_cart_grid);
  if(exitstatus != MPI_SUCCESS) {
    fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }
#  endif

#else
  memcpy(zbuffer, buffer, momentum_number * 32 * T * sizeof(double) );
#endif
  free( buffer );

  if(io_proc == 2) {

    /* reverse the ordering back to momentum - munu - time */
    aff_buffer = (double _Complex*)malloc( momentum_number * 16 * T_global * sizeof(double _Complex) );
    if(aff_buffer == NULL) {
      fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(2);
    }
    i = 0;
    for( int ip=0; ip<momentum_number; ip++) {
      for( int mu=0; mu<16; mu++ ) {
        for( int it = 0; it < T_global; it++ ) {
          int offset = (it * 16 + mu ) * momentum_number + ip;
          aff_buffer[i++] = zbuffer[offset];
        }
      }
    }
    free( zbuffer );

    for(i=0; i < momentum_number; i++) {
      sprintf(aff_buffer_path, "%s/px%.2dpy%.2dpz%.2d", tag, momentum_list[i][0], momentum_list[i][1], momentum_list[i][2] );
      /* fprintf(stdout, "# [cvc_tensor_tp_write_to_aff_file] current aff path = %s\n", aff_buffer_path); */
      affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
      exitstatus = aff_node_put_complex (affw, affdir, aff_buffer+16*T_global*i, (uint32_t)T_global*16);
      if(exitstatus != 0) {
        fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(5);
      }
    }
    free( aff_buffer );
  }  /* if io_proc == 2 */

#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif

  retime = _GET_TIME;
  if(io_proc == 2) fprintf(stdout, "# [cvc_tensor_tp_write_to_aff_file] time for saving momentum space results = %e seconds\n", retime-ratime);

  return(0);

}  /* end of contract_vn_write_to_aff_file */
#endif
