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
  return(0);
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
  return(0);
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
  return(0);
}  /* end of contract_v1 */


/******************************************************
 *
 ******************************************************/
int contract_v2_from_v1 (double **v2, double **v1, fermion_propagator_type *prop, unsigned int N ) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for(unsigned int ix=0; ix < N; ix++) {
    _v2_eq_v1_eps_fp( v2[ix], v1[ix], prop[ix] );
  }
  return(0);
}  /* end of contract_v2_from_v1 */


/******************************************************
 *
 ******************************************************/
int contract_v4 (double **v4, double *phi, fermion_propagator_type *prop1, fermion_propagator_type *prop2, unsigned int N ) {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  fermion_propagator_type fp;
  create_fp( &fp );

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(unsigned int ix=0; ix < N; ix++) {

    _fp_eq_fp_eps_contract13_fp( fp, prop1[ix], prop2[ix] );

    _v4_eq_fv_dot_fp( v4[ix], phi+_GSI(ix), fp);
  }
  free_fp( &fp );

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  return(0);
}  /* end of contract_v4 */


/******************************************************
 *
 ******************************************************/
int contract_v5 (double **v5, fermion_propagator_type *prop1, fermion_propagator_type *prop2, fermion_propagator_type *prop3, unsigned int N ) {
  const size_t bytes = 32 * sizeof(double);
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  fermion_propagator_type fp;
  create_fp ( &fp );
  spinor_propagator_type sp;
  create_sp ( &sp );

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(unsigned int ix=0; ix < N; ix++) {

    _fp_eq_fp_eps_contract13_fp( fp, prop2[ix], prop3[ix] );
    _sp_eq_fp_del_contract23_fp( sp, prop1[ix], fp);
    memcpy(v5[ix], sp[0], bytes );
  }
  free_fp( &fp );
  free_fp( &sp );
#ifdef HAVE_OPENMP
  }  /* end of parallel region */
#endif
    return(0);
}  /* end of contract_v5 */

/******************************************************
 *
 ******************************************************/
int contract_v6 (double **v6, fermion_propagator_type *prop1, fermion_propagator_type *prop2, fermion_propagator_type *prop3, unsigned int N ) {
  const size_t bytes = 32 * sizeof(double);
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  fermion_propagator_type fp;
  create_fp ( &fp );
  spinor_propagator_type sp;
  create_sp ( &sp );

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(unsigned int ix=0; ix < N; ix++) {

    _fp_eq_fp_eps_contract13_fp( fp, prop2[ix], prop3[ix] );
    _sp_eq_fp_del_contract34_fp( sp, prop1[ix], fp);
    memcpy(v6[ix], sp[0], bytes );
  }
  free_fp( &fp );
  free_fp( &sp );
#ifdef HAVE_OPENMP
  }  /* end of parallel region */
#endif
    return(0);
}  /* end of contract_v6 */

/******************************************************
 * vp[t][p][c]
 *
 * t timeslice
 * p momentum
 * c color
 ******************************************************/
int contract_vn_momentum_projection (double *** const vp, double ** const vx, int const n, const int (* const momentum_list)[3], int const momentum_number) {

  const unsigned int VOL3 = LX*LY*LZ;
  int exitstatus;
  double ratime, retime;

  ratime = _GET_TIME;

  for ( int it = 0; it < T; it++ ) {
    unsigned int offset = 2 * n * it * VOL3;
    exitstatus = momentum_projection2 ( vx[0]+offset, vp[it][0], n, momentum_number, momentum_list, NULL);
    if(exitstatus != 0) {
      fprintf(stderr, "[contract_vn_momentum_projection] Error from momentum_projection2, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(3);
    }
  }  /* end of loop on timeslices */

  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [contract_vn_momentum_projection] time for momentum projection = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);

  return(0);
}  /* end of contract_vn_momentum_projection */

/******************************************************
 * write a vn block to AFF file
 ******************************************************/
#ifdef HAVE_LHPC_AFF
int contract_vn_write_aff (double *** const vp, int const n, struct AffWriter_s*affw, char*tag, const int (* const momentum_list)[3], int const momentum_number, int const io_proc ) {

  int exitstatus;
  double ratime, retime;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_buffer_path[400];
  double _Complex ***zbuffer = NULL;

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_vn_write_aff] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }

    exitstatus = init_3level_zbuffer ( &zbuffer,  T_global, momentum_number, n );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[contract_vn_write_aff] Error from init_3level_zbuffer %s %d\n", __FILE__, __LINE__);
      return(6);
    }
  } else if (io_proc == 1 ) {
    exitstatus = init_3level_zbuffer ( &zbuffer,  1, 1, 1 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[contract_vn_write_aff] Error from init_3level_zbuffer %s %d\n", __FILE__, __LINE__);
      return(6);
    }
  }

  ratime = _GET_TIME;

#ifdef HAVE_MPI
  int i = 2 * momentum_number * n * T;
  if(io_proc>0) {
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
    exitstatus = MPI_Gather(vp[0][0], i, MPI_DOUBLE, zbuffer[0][0], i, MPI_DOUBLE, 0, g_tr_comm);
#  else
    exitstatus = MPI_Gather(vp[0][0], i, MPI_DOUBLE, zbuffer[0][0], i, MPI_DOUBLE, 0, g_cart_grid);
#  endif
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[contract_vn_write_aff] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(3);
    }
  }
#else
  memcpy(zbuffer[0][0], vp[0][0], momentum_number * n * T * sizeof(double _Complex) );
#endif

  if(io_proc == 2) {

    /* reverse the ordering back to momentum - munu - time */
    double _Complex **aff_buffer = NULL;
    exitstatus = init_2level_zbuffer ( &aff_buffer, T_global, n );
    if( exitstatus != 0) {
      fprintf(stderr, "[contract_vn_write_aff] Error from init_2level_zbuffer %s %d\n", __FILE__, __LINE__);
      return(2);
    }

    for( int ip=0; ip<momentum_number; ip++) {

      for( int it = 0; it < T_global; it++ ) {
        for( int mu=0; mu<n; mu++ ) {
          aff_buffer[it][mu] = zbuffer[it][ip][mu];
        }
      }

      sprintf(aff_buffer_path, "%s/px%.2dpy%.2dpz%.2d", tag, momentum_list[ip][0], momentum_list[ip][1], momentum_list[ip][2] );
      /* fprintf(stdout, "# [contract_vn_write_aff] current aff path = %s\n", aff_buffer_path); */

      affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);

      exitstatus = aff_node_put_complex (affw, affdir, aff_buffer[0], (uint32_t)T_global*n);
      if(exitstatus != 0) {
        fprintf(stderr, "[contract_vn_write_aff] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(5);
      }

    }
    fini_2level_zbuffer ( &aff_buffer );
  }  /* if io_proc == 2 */
  if ( io_proc > 0 ) {
    fini_3level_zbuffer ( &zbuffer );
  }

#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif

  retime = _GET_TIME;
  if(io_proc == 2) fprintf(stdout, "# [contract_vn_write_aff] time for saving momentum space results = %e seconds\n", retime-ratime);

  return(0);

}  /* end of contract_vn_write_aff */
#endif  /* end of if def HAVE_LHPC_AFF */

/******************************************************
 * write a vn block to AFF file
 ******************************************************/
#ifdef HAVE_LHPC_AFF
int contract_vn_write_aff_interval (double ***vp, int n, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int tmin, int tmax, int io_proc ) {
  
  int exitstatus;
  double ratime, retime;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_buffer_path[400];
  double _Complex ***zbuffer = NULL;
  int nT = tmax - tmin + 1;

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_vn_write_aff_interval] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }

    exitstatus = init_3level_zbuffer ( &zbuffer,  T_global, momentum_number, n );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[contract_vn_write_aff_interval] Error from init_3level_zbuffer %s %d\n", __FILE__, __LINE__);
      return(6);
    }
  } else if (io_proc == 1 ) {
    exitstatus = init_3level_zbuffer ( &zbuffer,  1, 1, 1 );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[contract_vn_write_aff_interval] Error from init_3level_zbuffer %s %d\n", __FILE__, __LINE__);
      return(6);
    }
  }

  ratime = _GET_TIME;

#ifdef HAVE_MPI
  int i = 2 * momentum_number * n * T;
  if(io_proc>0) {
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
    exitstatus = MPI_Gather(vp[0][0], i, MPI_DOUBLE, zbuffer[0][0], i, MPI_DOUBLE, 0, g_tr_comm);
#  else
    exitstatus = MPI_Gather(vp[0][0], i, MPI_DOUBLE, zbuffer[0][0], i, MPI_DOUBLE, 0, g_cart_grid);
#  endif
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[contract_vn_write_aff_interval] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(3);
    }
  }
#else
  memcpy(zbuffer[0][0], vp[0][0], momentum_number * n * T * sizeof(double _Complex) );
#endif

  if(io_proc == 2) {

    /* reverse the ordering back to momentum - munu - time */
    double _Complex **aff_buffer = NULL;
    exitstatus = init_2level_zbuffer ( &aff_buffer, nT, n );
    if( exitstatus != 0) {
      fprintf(stderr, "[contract_vn_write_aff_interval] Error from init_2level_zbuffer %s %d\n", __FILE__, __LINE__);
      return(2);
    }

    for( int ip=0; ip<momentum_number; ip++) {

      for( int it = 0; it < nT; it++ ) {
        for( int mu = 0; mu < n; mu++ ) {
          aff_buffer[it][mu] = zbuffer[it+tmin][ip][mu];
        }
      }

      sprintf(aff_buffer_path, "%s/px%.2dpy%.2dpz%.2d", tag, momentum_list[ip][0], momentum_list[ip][1], momentum_list[ip][2] );
      /* fprintf(stdout, "# [contract_vn_write_aff_interval] current aff path = %s\n", aff_buffer_path); */

      affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);

      exitstatus = aff_node_put_complex (affw, affdir, aff_buffer[0], (uint32_t)nT*n);
      if(exitstatus != 0) {
        fprintf(stderr, "[contract_vn_write_aff_interval] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(5);
      }

    }
    fini_2level_zbuffer ( &aff_buffer );
  }  /* if io_proc == 2 */
  if ( io_proc > 0 ) {
    fini_3level_zbuffer ( &zbuffer );
  }

#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif

  retime = _GET_TIME;
  if(io_proc == 2) fprintf(stdout, "# [contract_vn_write_aff_interval] time for saving momentum space results = %e seconds\n", retime-ratime);

  return(0);

} /* end of contract_vn_write_aff_interval */
#endif  /* of if def HAVE_LHPC_AFF */

}  /* end of namespace cvc */
