/************************************************
 * prepare_propagator.cpp
 *
 * Wed Feb  5 16:32:16 EET 2014
 *
 * PURPOSE:
 * - build propagator points
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif
#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif
#ifdef __cplusplus
}
#endif

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "iblas.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "read_input_parser.h"
#include "ranlxd.h"
/* #include "smearing_techniques.h" */
/* #include "fuzz.h" */
#include "matrix_init.h"
#include "project.h"
#include "prepare_source.h"
#include "dummy_solver.h"
#include "table_init_d.h"


#ifndef _NON_ZERO
#  define _NON_ZERO (5.e-14)
#endif

namespace cvc {
#if 0
/************************************************************
 * prepare_seqprop_point_from_stochastic_oneend
 *   prepare propagator point from stochastic propagators
 *   output --- fp_out
 *   input  --- phi (stochastic, up), chi (stochastic, dn), prop (point source propagator),
 *              idsource, idsink (gamma matrix id at source and sink),
 *              ncol number of colors for the stochastic propagator
 *              momontum (sink momentum $\kvec_f$)
 *              N - number of sites
 * - NOTE that n_s = 4, n_c must be 3
 ************************************************************/
int prepare_seqprop_point_from_stochastic_oneend (double**fp_out, double **phi, double **chi,
    double **prop, const int idsource, int idsink, int ncol, double*phase_field, unsigned int N) {

  int n_s = 4;
  int n_c = 3;  // number of colors for the point propagator
  int psource[4], isimag, mu, c, ia, iprop;
  unsigned int ix;
  double ssource[4], spinor1[24], spinor2[24];
  complex w, w1, w2, ctmp;

  psource[0] = gamma_permutation[idsource][ 0] / 6;
  psource[1] = gamma_permutation[idsource][ 6] / 6;
  psource[2] = gamma_permutation[idsource][12] / 6;
  psource[3] = gamma_permutation[idsource][18] / 6;
  isimag = gamma_permutation[idsource][ 0] % 2;
  /* sign from the source gamma matrix; the minus sign
   * in the lower two lines is the action of gamma_5 */
  ssource[0] =  gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]];
  ssource[1] =  gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]];
  ssource[2] =  gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]];
  ssource[3] =  gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]];

  
  for(iprop=0; iprop<n_s*n_c; iprop++) {  // loop on spinor index of point source propagator
     
    for(mu=0; mu<4; mu++) {           // loop on spinor index of stochastic propagators
      for(c=0; c<ncol; c++) {         // loop on color  index of stochastic propagators

        // chi[psource[mu]]^dagger x exp(i k_f z_f) g5 x g_sink x prop[iprop]
        ctmp.re = 0.;
        ctmp.im = 0.;

        for(ix=0; ix<N; ix++) {
          _fv_eq_gamma_ti_fv(spinor1, idsink, prop[iprop]+_GSI(ix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, chi[psource[mu]*ncol+c]+_GSI(ix), spinor2);
          w1.re = sin(phase_field[ix]);
          w1.im = cos(phase_field[ix]);
          _co_eq_co_ti_co(&w2, &w, &w1);
          ctmp.re += w2.re;
          ctmp.im += w2.im;
        }

        for(ix=0; ix<N; ix++) {

          for(ia=0; ia<n_c*n_s; ia++) {
            w.re = phi[mu*n_c+c][_GSI(ix) + 2*ia  ];
            w.im = phi[mu*n_c+c][_GSI(ix) + 2*ia+1];

            _co_eq_co_ti_co(&w1, &w, &ctmp);

            if( !isimag ) {
              fp_out[iprop][_GSI(ix) + 2*ia  ] +=  ssource[mu] * w1.re;
              fp_out[iprop][_GSI(ix) + 2*ia+1] +=  ssource[mu] * w1.im;
            } else {
              fp_out[iprop][_GSI(ix) + 2*ia  ] += -ssource[mu] * w1.im;
              fp_out[iprop][_GSI(ix) + 2*ia+1] +=  ssource[mu] * w1.re;
            }
          }
        }  // of loop on ix
      }    // of loop on colors for stochastic propagator
    }      // of loop on spin   for stochastic propagator
  }  // of loop on iprop
  return(0);
}  // end of prepare_seqprop_point_from_stochastic_oneend
#endif

#if 0
/************************************************************
 * prepare_prop_point_from_stochastic
 *   prepare propagator point from stochastic sources
 *   and propagators
 *   - fp_out = phi chi^dagger
 *   - phi - propagator
 *   - chi - source
 * - NOTE this is without spin or color dilution
 ************************************************************/
int prepare_prop_point_from_stochastic (double**fp_out, double*phi, double*chi,
    double**prop, int idsink, double*phase_field, unsigned int N) {

  int n_s = 4;
  int n_c = 3;
  int ia, iprop;
  unsigned int ix;
  double spinor1[24];
  complex w, w1, w2, ctmp;

  for(iprop=0; iprop<12; iprop++) {

    // chi[psource[mu]]^dagger x exp(i k_f z_f) g5 x g_sink x prop[iprop]
    ctmp.re = 0.;
    ctmp.im = 0.;

    for(ix=0; ix<N; ix++) {
      _fv_eq_gamma_ti_fv(spinor1, idsink, prop[iprop]+_GSI(ix));
      _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor1);
      w1.re = sin(phase_field[ix]);
      w1.im = cos(phase_field[ix]);
      _co_eq_co_ti_co(&w2, &w, &w1);
      ctmp.re += w2.re;
      ctmp.im += w2.im;
    }

    for(ix=0; ix<N; ix++) {

      for(ia=0; ia<n_s*n_c; ia++) {
        w.re = phi[_GSI(ix) + 2*ia  ];
        w.im = phi[_GSI(ix) + 2*ia+1];

        _co_eq_co_ti_co(&w1, &w, &ctmp);

        fp_out[iprop][2*ia  ] += w1.re;
        fp_out[iprop][2*ia+1] += w1.im;
      }
    }  // of loop on sites

  }    // of loop on iprop
  return(0);
}  // end of prepare_prop_point_from_stochastic
#endif

/*******************************************************************
 * seqn using stochastic
 *
 * shoulde be safe, if seq_prop = prop
 *
 * prop_aux  is nprop  x [VOL3x12] (C) = [12xVOL3] x nprop (F)
 *
 * stoch_aux is nstoch x [VOL3x12] (C) = [12xVOL3] x nstoch (F)
 *
 * we calculate p = stoch_aux^H x prop_aux, which is nstoch x nprop (F)
 * 
 * we calculate prop_aux = stoch_aux x p, which is
 *
 * [12xVOL3] x nstoch (F) x nstoch x nprop (F) = [12xVOL3] x nprop (F) = nprop x [VOL3x12] (C)
 *
 *******************************************************************/

int prepare_seqn_stochastic_vertex_propagator_sliced3d (double**seq_prop, double**stoch_prop, double**stoch_source, 
    double**prop, int nstoch, int nprop, int momentum[3], int gid) {

  const unsigned int VOL3 = LX*LY*LZ;
  const size_t sizeof_spinor_field_timeslice = _GSI(VOL3) * sizeof(double);
  const unsigned int spinor_field_timeslice_items = g_fv_dim * VOL3;
  const unsigned int items_p = nstoch * nprop;
  const size_t bytes_p       = items_p * sizeof(double _Complex);

  int i, it;
  int BLAS_M, BLAS_N, BLAS_K;
  int BLAS_LDA, BLAS_LDB ,BLAS_LDC;
  char BLAS_TRANSA, BLAS_TRANSB;
  double _Complex *p = NULL;
  double _Complex BLAS_ALPHA, BLAS_BETA;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex *prop_aux=NULL, *stoch_aux=NULL;
  double ratime, retime;

  ratime = _GET_TIME;

  double *phase = (double*)malloc(2*VOL3*sizeof(double)) ;
  if( phase == NULL ) {
    fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d] Error from malloc\n");
    return(1);
  }
  make_lexic_phase_field_3d (phase,momentum);
  /***************************
   * apply vertex
   ***************************/
  for(i=0; i<nprop; i++) {
    /* multiply with gamma[gid] */
    spinor_field_eq_gamma_ti_spinor_field ( seq_prop[i], gid, prop[i], VOLUME);
    /* multiply with complex phase */
    for(it=0; it<T; it++) {
      unsigned int ix = _GSI( g_ipt[it][0][0][0] );
      spinor_field_eq_spinor_field_ti_complex_field (seq_prop[i]+ix, seq_prop[i]+ix, phase, VOL3);
    }
  }
  free(phase);
 
  if( (prop_aux = (double _Complex*)malloc(nprop * sizeof_spinor_field_timeslice)) == NULL ) {
    fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d] Error from malloc\n");
    return(1);
  }
  if( (stoch_aux = (double _Complex*)malloc(nstoch * sizeof_spinor_field_timeslice)) == NULL ) {
    fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d] Error from malloc\n");
    return(2);
  }
  if( (p = (double _Complex*)malloc( bytes_p )) == NULL ) {
    fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d] Error from malloc\n");
    return(3);
  }

  /* loop on timeslices */
  for(it = 0; it<T; it++) {
    unsigned int offset = it * _GSI(VOL3);

    for(i=0; i<nprop; i++) {
      memcpy(prop_aux  + i*spinor_field_timeslice_items, seq_prop[i] + offset, sizeof_spinor_field_timeslice );
    }
    for(i=0; i<nstoch; i++) {
      memcpy(stoch_aux + i*spinor_field_timeslice_items, stoch_source[i] + offset, sizeof_spinor_field_timeslice );
    }

    /* projection on stochastic source */
    BLAS_ALPHA  = 1.;
    BLAS_BETA   = 0.;
    BLAS_TRANSA = 'C';
    BLAS_TRANSB = 'N';
    BLAS_M      = nstoch;
    BLAS_K      = g_fv_dim*VOL3;
    BLAS_N      = nprop;
    BLAS_A      = stoch_aux;
    BLAS_B      = prop_aux;
    BLAS_C      = p;
    BLAS_LDA    = BLAS_K;
    BLAS_LDB    = BLAS_K;
    BLAS_LDC    = BLAS_M;

    F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
    /* allreduce across all processes */
    double _Complex *p_buffer = (double _Complex*)malloc( bytes_p );
    if( p_buffer == NULL ) {
      fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d] Error from malloc\n");
      return(2);
    }

    memcpy(p_buffer, p, bytes_p);
    int status = MPI_Allreduce(p_buffer, p, 2*items_p, MPI_DOUBLE, MPI_SUM, g_ts_comm);
    if(status != MPI_SUCCESS) {
      fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d] Error from MPI_Allreduce, status was %d\n", status);
      return(1);
    }
    free(p_buffer); p_buffer = NULL;
#endif

#if 0
    /* TEST */
    {
      int i, k;
      for(i=0; i<nprop; i++) {
        for(k=0; k<nstoch; k++) {
          fprintf(stdout, "p2 proc%.2d t%.2d %2d %2d %25.16e %25.16e\n", g_cart_id, it, i, k, creal(p[i*nstoch+k]), cimag(p[i*nstoch+k]) );
        }
      }
    }
#endif

    for(i=0; i<nstoch; i++) {
      memcpy(stoch_aux + i*spinor_field_timeslice_items, stoch_prop[i] + offset, sizeof_spinor_field_timeslice );
    }

    /* expand in stochastic propagator */
    BLAS_ALPHA  =  1.;
    BLAS_BETA   =  0.;
    BLAS_TRANSA = 'N';
    BLAS_TRANSB = 'N';
    BLAS_M      = g_fv_dim * VOL3;
    BLAS_K      = nstoch;
    BLAS_N      = nprop;
    BLAS_A      = stoch_aux;
    BLAS_B      = p;
    BLAS_C      = prop_aux;
    BLAS_LDA    = BLAS_M;
    BLAS_LDB    = BLAS_K;
    BLAS_LDC    = BLAS_M;

    F_ZGEMM(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

    for(i=0; i<nprop; i++) {
      memcpy( seq_prop[i] + offset, prop_aux  + i*spinor_field_timeslice_items, sizeof_spinor_field_timeslice );
    }

  }  /* end of loop on timeslices */ 

  free(prop_aux);
  free(stoch_aux);
  free(p);

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "[prepare_seqn_stochastic_vertex_propagator_sliced3d] time for prepare_seqn_stochastic_vertex_propagator_sliced3d = %e seconds\n", retime-ratime);
  return(0);
}  /* prepare_seqn_stochastic_vertex_propagator_sliced3d */

/*******************************************************************
 * seqn using stochastic
 *
 * shoulde be safe, if seq_prop = prop
 *
 * prop_aux  is nprop  x [VOL3x12] (C) = [12xVOL3] x nprop (F)
 *
 * stoch_aux is nstoch x [VOL3x12] (C) = [12xVOL3] x nstoch (F)
 *
 * we calculate p = stoch_aux^H x prop_aux, which is nstoch x nprop (F) = nprop x nstoch (C)
 * 
 * we calculate prop_aux = stoch_aux x p, which is
 *
 * [12xVOL3] x nstoch (F) x nstoch x nprop (F) = [12xVOL3] x nprop (F) = nprop x [VOL3x12] (C)
 *
 *******************************************************************/

int prepare_seqn_stochastic_vertex_propagator_sliced3d_oet (double**seq_prop, double**stoch_prop_0, double**stoch_prop_p, 
    double**prop, int momentum[3], int gid, int idsource)
{
  const int nprop = g_fv_dim;
  const int nstoch = 4;

  const unsigned int VOL3 = LX*LY*LZ;
  const size_t sizeof_spinor_field_timeslice = _GSI(VOL3) * sizeof(double);
  const unsigned int spinor_field_timeslice_items = g_fv_dim * VOL3;
  const unsigned int items_p = nstoch * nprop;
  const size_t bytes_p       = items_p * sizeof(double _Complex);

  const int psource[4] = { gamma_permutation[idsource][ 0] / 6,
                           gamma_permutation[idsource][ 6] / 6,
                           gamma_permutation[idsource][12] / 6,
                           gamma_permutation[idsource][18] / 6 };
  const int isimag = gamma_permutation[idsource][ 0] % 2;
            /* sign from the source gamma matrix; the minus sign
             *    * in the lower two lines is the action of gamma_5 */
  const double ssource[4] =  { gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]],
                               gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]],
                               gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]],
                               gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]] };


  int i, it;
  int BLAS_M, BLAS_N, BLAS_K;
  int BLAS_LDA, BLAS_LDB ,BLAS_LDC;
  char BLAS_TRANSA, BLAS_TRANSB;
  double _Complex *p = NULL;
  double _Complex BLAS_ALPHA, BLAS_BETA;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex *prop_aux=NULL, *stoch_aux=NULL;
  double ratime, retime;

  ratime = _GET_TIME;

  double *phase = (double*)malloc(2*VOL3*sizeof(double)) ;
  if( phase == NULL ) {
    fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d_oet] Error from malloc\n");
    return(1);
  }
  make_lexic_phase_field_3d (phase,momentum);
  /*********************************************************************
   * apply vertex
   * 
   * note: we should apply the adjoint vertex to stoch_prop_0 instead,
   * only 4 fields
   *********************************************************************/
  for(i=0; i<nprop; i++) {
    /* multiply with g5  gamma[gid] */
    spinor_field_eq_gamma_ti_spinor_field ( seq_prop[i], gid, prop[i], VOLUME);
    g5_phi( seq_prop[i], VOLUME );
    /* multiply with complex phase */
    for(it=0; it<T; it++) {
      unsigned int ix = _GSI( g_ipt[it][0][0][0] );
      spinor_field_eq_spinor_field_ti_complex_field (seq_prop[i]+ix, seq_prop[i]+ix, phase, VOL3);
    }
  }
  free(phase);
 
  if( (prop_aux = (double _Complex*)malloc(nprop * sizeof_spinor_field_timeslice)) == NULL ) {
    fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d_oet] Error from malloc\n");
    return(1);
  }
  if( (stoch_aux = (double _Complex*)malloc(nstoch * sizeof_spinor_field_timeslice)) == NULL ) {
    fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d_oet] Error from malloc\n");
    return(2);
  }
  if( (p = (double _Complex*)malloc( bytes_p )) == NULL ) {
    fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d_oet] Error from malloc\n");
    return(3);
  }

  /* loop on timeslices */
  for(it = 0; it<T; it++) {
    unsigned int offset = it * _GSI(VOL3);

    for(i=0; i<nprop; i++) {
      memcpy(prop_aux  + i*spinor_field_timeslice_items, seq_prop[i] + offset, sizeof_spinor_field_timeslice );
    }
    for(i=0; i<nstoch; i++) {
      memcpy(stoch_aux + i*spinor_field_timeslice_items, stoch_prop_0[i] + offset, sizeof_spinor_field_timeslice );
    }

    /* projection on stochastic source */
    BLAS_ALPHA  = 1.;
    BLAS_BETA   = 0.;
    BLAS_TRANSA = 'C';
    BLAS_TRANSB = 'N';
    BLAS_M      = nstoch;
    BLAS_K      = g_fv_dim*VOL3;
    BLAS_N      = nprop;
    BLAS_A      = stoch_aux;
    BLAS_B      = prop_aux;
    BLAS_C      = p;
    BLAS_LDA    = BLAS_K;
    BLAS_LDB    = BLAS_K;
    BLAS_LDC    = BLAS_M;

    F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
    /* allreduce across all processes */
    double _Complex *p_buffer = (double _Complex*)malloc( bytes_p );
    if( p_buffer == NULL ) {
      fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d_oet] Error from malloc\n");
      return(2);
    }

    memcpy(p_buffer, p, bytes_p);
    int status = MPI_Allreduce(p_buffer, p, 2*items_p, MPI_DOUBLE, MPI_SUM, g_ts_comm);
    if(status != MPI_SUCCESS) {
      fprintf(stderr, "[prepare_seqn_stochastic_vertex_propagator_sliced3d_oet] Error from MPI_Allreduce, status was %d\n", status);
      return(1);
    }
    free(p_buffer); p_buffer = NULL;
#endif

    for(i=0; i<nstoch; i++) {
      memcpy(stoch_aux + i*spinor_field_timeslice_items, stoch_prop_p[i] + offset, sizeof_spinor_field_timeslice );
    }

    /******************************************
     ******************************************
     **
     ** remember: ssource CONTAINS signs from
     **   multiplication with i; so we only need
     **   to swap real and imaginary part if
     **   isimag is true; we do
     **   conj(a+ib)*i = (a-ib)*i = b + ia
     **
     ******************************************
     ******************************************/
    for(i=0; i<nprop; i++) {
      double _Complex ztmp[4];
      memcpy(ztmp, p+i*4, 4*sizeof(double _Complex));
      if(isimag) {
        p[4*i + 0] = ztmp[psource[0]] * (-I) * ssource[0];
        p[4*i + 1] = ztmp[psource[1]] * (-I) * ssource[1];
        p[4*i + 2] = ztmp[psource[2]] * (-I) * ssource[2];
        p[4*i + 3] = ztmp[psource[3]] * (-I) * ssource[3];
      } else {
        p[4*i + 0] = ztmp[psource[0]] * ssource[0];
        p[4*i + 1] = ztmp[psource[1]] * ssource[1];
        p[4*i + 2] = ztmp[psource[2]] * ssource[2];
        p[4*i + 3] = ztmp[psource[3]] * ssource[3];
      }
    }

    /* TEST */
/*
    {
      int i, k;
      for(i=0; i<nprop; i++) {
        for(k=0; k<nstoch; k++) {
          fprintf(stdout, "p2 proc%.2d t%.2d %2d %2d %25.16e %25.16e\n", g_cart_id, it, i, k, creal(p[i*nstoch+k]), cimag(p[i*nstoch+k]) );
        }
      }
    }
*/

    /* expand in stochastic propagator */
    BLAS_ALPHA  =  1.;
    BLAS_BETA   =  0.;
    BLAS_TRANSA = 'N';
    BLAS_TRANSB = 'N';
    BLAS_M      = g_fv_dim * VOL3;
    BLAS_K      = nstoch;
    BLAS_N      = nprop;
    BLAS_A      = stoch_aux;
    BLAS_B      = p;
    BLAS_C      = prop_aux;
    BLAS_LDA    = BLAS_M;
    BLAS_LDB    = BLAS_K;
    BLAS_LDC    = BLAS_M;

    F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

    for(i=0; i<nprop; i++) {
      memcpy( seq_prop[i] + offset, prop_aux  + i*spinor_field_timeslice_items, sizeof_spinor_field_timeslice );
    }

  }  /* end of loop on timeslices */ 

  free(prop_aux);
  free(stoch_aux);
  free(p);

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "[prepare_seqn_stochastic_vertex_propagator_sliced3d_oet] time for prepare_seqn_stochastic_vertex_propagator_sliced3d_oet = %e seconds\n", retime-ratime);

  return(0);
}  /* end of prepare_seqn_stochastic_vertex_propagator_sliced3d_oet */

/*******************************************************************
 * seqn using stochastic - vertex - stochastic oet
 *
 *******************************************************************/

int prepare_seq_stochastic_vertex_stochastic_oet (double**seq_prop, double**stoch_prop_0, double**stoch_prop_p, 
    int gid, int source_coords[4], int nsample)
{
  const int nprop = g_fv_dim;
  const int nstoch = 4;

  const size_t sizeof_spinor_field = _GSI(VOLUME) * sizeof(double);

  const int pvertex[4] = { gamma_permutation[gid][ 0] / 6,
                           gamma_permutation[gid][ 6] / 6,
                           gamma_permutation[gid][12] / 6,
                           gamma_permutation[gid][18] / 6 };
  const int isimag = gamma_permutation[gid][ 0] % 2;
            /* sign from the source gamma matrix; the minus sign
             *    * in the lower two lines is the action of gamma_5 */
  const double svertex[4] =  { gamma_sign[gid][ 0] * gamma_sign[5][gamma_permutation[gid][ 0]],
                               gamma_sign[gid][ 6] * gamma_sign[5][gamma_permutation[gid][ 6]],
                               gamma_sign[gid][12] * gamma_sign[5][gamma_permutation[gid][12]],
                               gamma_sign[gid][18] * gamma_sign[5][gamma_permutation[gid][18]] };


  int i, isample;
  double ratime, retime;
  double sp_source[96];
  int source_proc_id = 0;
  unsigned int offset = 0;

  ratime = _GET_TIME;

  /* initialize seq_prop to zero */
  for(i=0; i<12; i++) {
    memset( seq_prop[i], 0, sizeof_spinor_field );
  }

#ifdef HAVE_MPI
  int source_proc_coords[4] = { source_coords[0] / T, source_coords[1] / LX, source_coords[2] / LY, source_coords[3] / LZ };

  if(g_cart_id == 0) {
    fprintf(stdout, "# [prepare_seq_stochastic_vertex_stochastic_oet] global source coordinates: (%3d,%3d,%3d,%3d)\n",
        source_coords[0], source_coords[1], source_coords[2], source_coords[3]);
    fprintf(stdout, "# [prepare_seq_stochastic_vertex_stochastic_oet] source proc coordinates: (%3d,%3d,%3d,%3d)\n",
        source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  if(source_proc_id == g_cart_id) {
    fprintf(stdout, "# [prepare_seq_stochastic_vertex_stochastic_oet] process %4d has source location\n", source_proc_id);
  }
#endif
  if(source_proc_id == g_cart_id) {
    offset = _GSI( g_ipt[source_coords[0]%T][source_coords[1]%LX][source_coords[2]%LY][source_coords[3]%LZ] );
  }

  /* loop on samples */
  for(isample = 0; isample < nsample; isample++) {

    if(source_proc_id == g_cart_id) {
      if( !isimag ) {
        for(i=0; i<4; i++) {
          _fv_eq_fv( sp_source+i*24, stoch_prop_0[4*isample + pvertex[i]] + offset );
          _fv_ti_eq_re( sp_source+i*24, svertex[i] );
        }
      } else {
        double sp2[24];
        for(i=0; i<4; i++) {
          _fv_eq_fv( sp2, stoch_prop_0[4*isample + pvertex[i]] + offset );
          _fv_eq_fv_ti_im ( sp_source+i*24, sp2, -svertex[i] );
        }
      }
    } else {
      memset(sp_source, 0, 96 * sizeof(double) );
    }  /* end of if source_proc_id = g_cart_id */

#ifdef HAVE_MPI
    if( MPI_Bcast( sp_source, 96, MPI_DOUBLE, source_proc_id, g_cart_grid ) != MPI_SUCCESS ) {
      fprintf(stderr, "[prepare_seq_stochastic_vertex_stochastic_oet] Error from MPI_Bcast\n");
      return(1);
    }
#endif
 
    /* loop on source spin-color index */
    for(i=0; i<12; i++) {
      double *phi = seq_prop[i];
      complex w;

      _co_eq_co_conj(&w, (complex*)(sp_source+2*i) );
      spinor_field_pl_eq_spinor_field_ti_co (phi, stoch_prop_p[4*isample+0], w, VOLUME);

      _co_eq_co_conj(&w, (complex*)(sp_source+24+2*i) );
      spinor_field_pl_eq_spinor_field_ti_co (phi, stoch_prop_p[4*isample+1], w, VOLUME);

      _co_eq_co_conj(&w, (complex*)(sp_source+48+2*i) );
      spinor_field_pl_eq_spinor_field_ti_co (phi, stoch_prop_p[4*isample+2], w, VOLUME);

      _co_eq_co_conj(&w, (complex*)(sp_source+72+2*i) );
      spinor_field_pl_eq_spinor_field_ti_co (phi, stoch_prop_p[4*isample+3], w, VOLUME);

    }  /* end of loop on source spin-color */

  }  /* end of loop on samples */
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "[prepare_seq_stochastic_vertex_stochastic_oet] time for prepare_seq_stochastic_vertex_stochastic_oet = %e seconds\n", retime-ratime);

  return(0);
}  /* end of prepare_seq_stochastic_vertex_stochastic_oet */


/**********************************************************
 * make a poin-to-all propagator
 * 4 (spin) x 3 (color) right-hand sides
 **********************************************************/
int point_to_all_fermion_propagator_clover_eo ( double **eo_spinor_field_e, double **eo_spinor_field_o,  int op_id,
    int global_source_coords[4], double *gauge_field, double **mzz, double **mzzinv, int check_propagator_residual, double **eo_spinor_work ) {

  const size_t sizeof_eo_spinor_field = _GSI( VOLUME / 2 ) * sizeof(double);

  int exitstatus;
  int local_source_coords[4];
  int source_proc_id;

  /* source info for shifted source location */
  if( (exitstatus = get_point_source_info ( global_source_coords, local_source_coords, &source_proc_id) ) != 0 ) {
    fprintf(stderr, "[point_to_all_fermion_propagator_clover_eo] Error from get_point_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(15);
  }

  /* loop on spin and color indices */
  for(int i=0; i<12; i++) {

    /* A^-1 g5 source */
    exitstatus = init_clover_eo_spincolor_pointsource_propagator (eo_spinor_field_e[i], eo_spinor_field_o[i],
      global_source_coords, i, gauge_field, mzzinv[0], (int)(source_proc_id == g_cart_id), eo_spinor_work[0]);
    if(exitstatus != 0 ) {
      fprintf(stderr, "[point_to_all_fermion_propagator_clover_eo] Error from init_eo_spincolor_pointsource_propagator; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(18);
    }

    /* C^-1 */
    if(g_cart_id == 0 && g_verbose > 1) fprintf(stdout, "# [point_to_all_fermion_propagator_clover_eo] calling tmLQCD_invert_eo\n");
    memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
    memcpy(eo_spinor_work[0], eo_spinor_field_o[i], sizeof_eo_spinor_field);
#ifdef HAVE_TMLQCD_LIBWRAPPER
    exitstatus = tmLQCD_invert_eo ( eo_spinor_work[1], eo_spinor_work[0], op_id);
#else
    exitstatus = 1;
#endif
    if(exitstatus != 0) {
      fprintf(stderr, "[point_to_all_fermion_propagator_clover_eo] Error from _TMLQCD_INVERT_EO, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }
    memcpy(eo_spinor_field_o[i], eo_spinor_work[1], sizeof_eo_spinor_field);

    /* B^-1 excl. C^-1 */
    exitstatus = fini_clover_eo_propagator (eo_spinor_field_e[i], eo_spinor_field_o[i], eo_spinor_field_e[i], eo_spinor_field_o[i],
      gauge_field, mzzinv[0], eo_spinor_work[0]);
    if(exitstatus != 0) {
      fprintf(stderr, "[point_to_all_fermion_propagator_clover_eo] Error from fini_eo_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(20);
    }

  }  /* end of loop on spin-color */

  if( check_propagator_residual ) {
    check_point_source_propagator_clover_eo( eo_spinor_field_e, eo_spinor_field_o, eo_spinor_work, gauge_field, mzz, mzzinv, global_source_coords, 12 );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[point_to_all_fermion_propagator_clover_eo] Error from check_point_source_propagator_clover_eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(21);
    }
  }

  return(0);
}  /* end of point_to_all_fermion_propagator_clover_eo */

/**********************************************************
 * T_global - set of stochastic timeslice propagators
 * projected onto subspace orthogonal to span { eo_evecs_block }
 *
 * NOTE: P_V^orth and P_t do NOT COMMUTE,
 *       P_V^orth AFTER P_t
 **********************************************************/
int prepare_clover_eo_stochastic_timeslice_propagator (
    double**prop, double*source,
    double *eo_evecs_block, double*evecs_lambdainv, int evecs_num, 
    double*gauge_field_with_phase, double **mzz[2], double **mzzinv[2], 
    int op_id, int check_propagator_residual) {

  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = LX * LY * LZ / 2;
  const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double);
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);

  double **eo_spinor_work = NULL;
  int exitstatus;

  exitstatus = init_2level_buffer( &eo_spinor_work, 4, _GSI((VOLUME+RAND)/2) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[prepare_clover_eo_stochastic_timeslice_propagator] Error from init_2level_buffer, status was %d\n", exitstatus);
    return(1);
  }


  for ( int it = 0; it < T_global; it++ ) {

    memset(eo_spinor_work[0], 0, sizeof_eo_spinor_field);
    if ( it / T == g_proc_coords[0] ) {
      unsigned int offset = (it%T) * _GSI(VOL3half);
      memcpy( eo_spinor_work[0] + offset, source + offset, sizeof_eo_spinor_field_timeslice  );
    }

    /* orthogonal projection */

    if ( op_id == 0 ) {
      /* work0 <- orthogonal projection work0 */
      exitstatus = project_propagator_field ( eo_spinor_work[0], eo_spinor_work[0], 0, eo_evecs_block, 1, evecs_num, Vhalf);
      if(exitstatus != 0) {
        fprintf(stderr, "[prepare_clover_eo_stochastic_timeslice_propagator] Error from project_propagator_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(35);
      }
    } else if ( op_id == -1 ) {
      /* apply C 
       * work1 <- C work0; aux work2
       */
      C_clover_oo ( eo_spinor_work[1], eo_spinor_work[0], gauge_field_with_phase, eo_spinor_work[2], g_mzz_up[1], g_mzzinv_up[0] );

      /* weighted parallel projection 
       * work1 <- V 2kappa/Lambda V^+ work1
       */
      exitstatus = project_propagator_field_weighted ( eo_spinor_work[1], eo_spinor_work[1], 1, eo_evecs_block, evecs_lambdainv, 1, evecs_num, Vhalf);
      if (exitstatus != 0) {
        fprintf(stderr, "[prepare_clover_eo_stochastic_timeslice_propagator] Error from project_propagator_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(35);
      }

      /* work1 <- work1 * 2 kappa */
      spinor_field_ti_eq_re ( eo_spinor_work[1], 2*g_kappa, Vhalf);
      /* apply Cbar 
       * work2 <- Cbar work1, aux work3
       */
      C_clover_oo ( eo_spinor_work[2], eo_spinor_work[1], gauge_field_with_phase, eo_spinor_work[3], g_mzz_dn[1], g_mzzinv_dn[0] );

      /* work0 <- work0 - work2 */
      spinor_field_eq_spinor_field_mi_spinor_field( eo_spinor_work[0], eo_spinor_work[0], eo_spinor_work[2], Vhalf);
    }

    /* invert */
    memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);

#ifdef HAVE_TMLQCD_LIBWRAPPER
    exitstatus = tmLQCD_invert_eo(eo_spinor_work[1], eo_spinor_work[0], op_id);
#else
        exitstatus = 1;
#endif

    if(exitstatus != 0) {
      fprintf(stderr, "[prepare_clover_eo_stochastic_timeslice_propagator] Error from tmlqcd_invert_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }
    memcpy( prop[it], eo_spinor_work[1], sizeof_eo_spinor_field );

    if( check_propagator_residual ) {
      exitstatus = check_oo_propagator_clover_eo( &(prop[it]), &(eo_spinor_work[0]), &(eo_spinor_work[1]), gauge_field_with_phase, mzz[op_id], mzzinv[op_id], 1 );
      if(exitstatus != 0) {
        fprintf(stderr, "[prepare_clover_eo_stochastic_timeslice_propagator] Error from check_oo_propagator_clover_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(19);
      }
    }

  }  /* end of loop on timeslices */

  fini_2level_buffer( &eo_spinor_work);

  return(0);
}  /* end of prepare_clover_eo_stochastic_timeslice_propagator */

/************************************************************************************
 * build a sequential propagator from forward propagators and a set of stochastic
 * timeslice propagators
 *
 * Half the sequential inversion,
 *
 * sequential = A^-1 g5 Gamma_seq(pvec_seq) forward
 ************************************************************************************/

int prepare_clover_eo_sequential_propagator_timeslice (
    double *sequential_propagator_e, double *sequential_propagator_o,
    double *forward_propagator_e,    double *forward_propagator_o,
    int momentum[3], int gamma_id, int timeslice, 
    double*gauge_field, double**mzz, double**mzzinv) {

  typedef int (*mom3_ptr_type)[3];

  const unsigned int Vhalf = VOLUME / 2;
  const size_t sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof( double );
  const unsigned int VOL3half = LX * LY * LZ / 2;

  int exitstatus;
  double**aux = NULL;

  exitstatus = init_2level_buffer ( &aux, 2, _GSI( (VOLUME+RAND)/2 ) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[prepare_clover_eo_sequential_propagator_timeslice] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /* all processes set e/o sequential propagator to zero */
  memset( sequential_propagator_e, 0, sizeof_eo_spinor_field);
  memset( sequential_propagator_o, 0, sizeof_eo_spinor_field);

  /* processes with sequential source timeslice, set the sequential source */
  if ( timeslice / T == g_proc_coords[0] ) {
    const unsigned int offset = (timeslice % T) * _GSI(VOL3half);
    double _Complex ***momentum_phase = NULL;

    exitstatus = init_3level_zbuffer ( &momentum_phase, 2, 1, VOL3half );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[prepare_clover_eo_sequential_propagator_timeslice] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(1);
    }

    make_eo_phase_field_timeslice ( momentum_phase[0], 1, (mom3_ptr_type)(&momentum), timeslice, 0);
    make_eo_phase_field_timeslice ( momentum_phase[1], 1, (mom3_ptr_type)(&momentum), timeslice, 1);

    spinor_field_eq_spinor_field_ti_complex_field ( sequential_propagator_e+offset, forward_propagator_e+offset, (double*)(momentum_phase[0][0]), VOL3half);
    spinor_field_eq_spinor_field_ti_complex_field ( sequential_propagator_o+offset, forward_propagator_o+offset, (double*)(momentum_phase[1][0]), VOL3half);

    spinor_field_eq_gamma_ti_spinor_field( sequential_propagator_e+offset, gamma_id, sequential_propagator_e+offset, VOL3half );
    spinor_field_eq_gamma_ti_spinor_field( sequential_propagator_o+offset, gamma_id, sequential_propagator_o+offset, VOL3half );

    g5_phi ( sequential_propagator_e+offset, VOL3half );
    g5_phi ( sequential_propagator_o+offset, VOL3half );

    fini_3level_zbuffer ( &momentum_phase );
  }

  Q_clover_eo_SchurDecomp_Ainv ( sequential_propagator_e, sequential_propagator_o, sequential_propagator_e, sequential_propagator_o, gauge_field, mzzinv[0], aux[0]);

  fini_2level_buffer( &aux );
  return(0);

}  /* end of prepare_clover_eo_sequential_propagator_timeslice */

/************************************************************************************
 *
 ************************************************************************************/
int select_stochastic_timeslice_propagator ( double***eo_stochastic_source_allt, double***eo_stochastic_propagator_allt, 
    double ***eo_stochastic_source, double ***eo_stochastic_propagator, int sequential_source_timeslice, int ns, int adj ) {

  const unsigned int Vhalf                      = VOLUME / 2;        /* half the 4-volume */
  const unsigned int VOL3half                   = LX * LY * LZ / 2;  /* half the 3-volume */
  const size_t sizeof_eo_spinor_field           = _GSI(Vhalf) * sizeof( double );     /* bytes in a half-volume spinor field */
  const size_t sizeof_eo_spinor_field_timeslice = _GSI(VOL3half) * sizeof( double );  /* bytes in a half-volume spinor field timeslice */
  const unsigned int offset                     = _GSI(VOL3half);    /* offset (double elements) for a half-volume spinor field timeslice */

  if ( adj == 0 ) {

    /************************************************************************************
     * collect the propagator
     ************************************************************************************/
    for ( int it = 0; it < T; it++ ) {
      unsigned int timeslice_offset = it * offset;
      for ( int isample = 0; isample < ns; isample ++ ) {
        memcpy ( eo_stochastic_propagator_allt[it][isample], eo_stochastic_propagator[sequential_source_timeslice][isample]+timeslice_offset, sizeof_eo_spinor_field_timeslice );
      }
    }  /* end of loop on timeslices */

    /************************************************************************************
     * collect the source
     ************************************************************************************/
    memset( eo_stochastic_source_allt[0][0], 0, ns * sizeof_eo_spinor_field );
    if ( sequential_source_timeslice / T == g_proc_coords[0] ) { 
      int it = sequential_source_timeslice % T;
      for ( int isample = 0; isample < ns; isample ++ ) {
        memcpy ( eo_stochastic_source_allt[it][isample], eo_stochastic_source[it][isample], sizeof_eo_spinor_field_timeslice );
      }
    }  /* end of if have seqential source timeslice */

  } else if (adj == 1) {

    /************************************************************************************
     * collect the propagator
     ************************************************************************************/
    memset( eo_stochastic_propagator_allt[0][0], 0, ns * sizeof_eo_spinor_field );
    if ( sequential_source_timeslice / T == g_proc_coords[0] ) {
      unsigned int sequential_source_timeslice_offset = ( sequential_source_timeslice % T ) * offset;

      for ( int it = 0; it < T; it++ ) {
        int iit = ( it + g_proc_coords[0] * T ) % T_global;  /* global timeslice, i.e. which stochastic propagator */
        for ( int isample = 0; isample < ns; isample ++ ) {
          memcpy ( eo_stochastic_propagator_allt[it][isample], eo_stochastic_propagator_allt[iit][isample] + sequential_source_timeslice_offset, sizeof_eo_spinor_field_timeslice );
        }
      }  /* end of loop on timeslices */
    }  /* end of if have source */

    /************************************************************************************
     * collect the source
     ************************************************************************************/
    memcpy ( eo_stochastic_source_allt[0][0], eo_stochastic_source[0][0], ns*sizeof_eo_spinor_field );
  }  /* end of if adj */

  return(0);
}  /* end of select_stochastic_timeslice_propagator */

/**********************************************************/
/**********************************************************/

/**********************************************************
 * make a poin-to-all propagator
 * 4 (spin) x 3 (color) right-hand sides
 **********************************************************/
int point_to_all_fermion_propagator_clover_full2eo ( double **eo_spinor_field_e, double **eo_spinor_field_o,  int op_id,
    int global_source_coords[4], double *gauge_field, double **mzz, double **mzzinv, int check_propagator_residual ) {

  const size_t sizeof_spinor_field    = _GSI( VOLUME )     * sizeof(double);
  const size_t sizeof_eo_spinor_field = _GSI( VOLUME / 2 ) * sizeof(double);

  int exitstatus;
  int local_source_coords[4];
  int source_proc_id;
  double **eo_spinor_work = NULL;
  double *spinor_work[2];

  /* source info for shifted source location */
  if( (exitstatus = get_point_source_info ( global_source_coords, local_source_coords, &source_proc_id) ) != 0 ) {
    fprintf(stderr, "[point_to_all_fermion_propagator_clover_full2eo] Error from get_point_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(15);
  }

  exitstatus = init_2level_buffer ( &eo_spinor_work, 5, _GSI( (VOLUME+RAND)/2 ) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[point_to_all_fermion_propagator_clover_full2eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
  spinor_work[0] = eo_spinor_work[0];
  spinor_work[1] = eo_spinor_work[2];

  /* loop on spin and color indices */
  for(int i=0; i<12; i++) {

    memset ( spinor_work[0], 0, sizeof_spinor_field );
    memset ( spinor_work[1], 0, sizeof_spinor_field );
    if ( source_proc_id == g_cart_id ) {
      spinor_work[0][ _GSI( g_ipt[local_source_coords[0]][local_source_coords[1]][local_source_coords[2]][local_source_coords[3]])+2*i ] = 1.;
    }

    exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], op_id );
    if(exitstatus < 0) {
      fprintf(stderr, "[point_to_all_fermion_propagator_clover_full2eo] Error from tmLQCD_invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }
    spinor_field_lexic2eo ( spinor_work[1], eo_spinor_field_e[i], eo_spinor_field_o[i] );

  }  /* end of loop on spin-color */

  if( check_propagator_residual ) {
    exitstatus = check_point_source_propagator_clover_eo( eo_spinor_field_e, eo_spinor_field_o, eo_spinor_work, gauge_field, mzz, mzzinv, global_source_coords, 12 );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[point_to_all_fermion_propagator_clover_full2eo] Error from check_point_source_propagator_clover_eo; status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(21);
    }
  }

  fini_2level_buffer ( &eo_spinor_work );
  return(0);
}  /* end of point_to_all_fermion_propagator_clover_full2eo */


/**********************************************************/
/**********************************************************/

/**********************************************************
 * check residual for a source, propagator pair
 *
 * source, prop: full spinor fields
 **********************************************************/
int check_residuum_full ( double **source, double **prop, double *gauge_field, double const mutm, double **mzz, int const nfields ) {

  const unsigned int Vhalf = VOLUME / 2;
  const size_t sizeof_spinor_field    = _GSI( VOLUME )     * sizeof(double);
  const size_t sizeof_eo_spinor_field = _GSI( VOLUME / 2 ) * sizeof(double);

  int exitstatus;

  double **eo_spinor_field = init_2level_dtable ( 6, _GSI( (size_t)Vhalf ) );
  if ( eo_spinor_field == NULL ) {
    fprintf(stderr, "[check_residuum_full] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  double **eo_spinor_work = init_2level_dtable ( 2, _GSI( (size_t)( VOLUME+RAND)/2 ) );
  if ( eo_spinor_work == NULL ) {
    fprintf(stderr, "[check_residuum_full] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  for ( int i = 0; i < nfields; i++ ) {

    // prop no. i to eo
    spinor_field_lexic2eo ( source[i], eo_spinor_field[0], eo_spinor_field[1] );
    spinor_field_lexic2eo ( prop[i],   eo_spinor_field[2], eo_spinor_field[3] );

    // apply Dirac operator
    Q_clover_phi_eo ( eo_spinor_field[4], eo_spinor_field[5], eo_spinor_field[2], eo_spinor_field[3], gauge_field, mutm, eo_spinor_work[0], mzz);


    // norm diff to source
    double norm_e = 0., norm_o = 0.;
    spinor_field_norm_diff ( &norm_e, eo_spinor_field[4], eo_spinor_field[0], Vhalf );
    spinor_field_norm_diff ( &norm_o, eo_spinor_field[5], eo_spinor_field[1], Vhalf );
    if ( g_cart_id == 0 ) fprintf ( stdout, "# [check_residuum_full] norm diff %2d e %16.7e o %16.76e\n", i, norm_e, norm_o );

  }  // end of loop on field components

  fini_2level_dtable ( &eo_spinor_field );
  fini_2level_dtable ( &eo_spinor_work );

  return(0);
}  // end of check_residuum_full


/**********************************************************/
/**********************************************************/

/**********************************************************
 * check residual for a source, propagator pair
 *
 * source, prop: eo spinor fields
 **********************************************************/
int check_residuum_eo ( double **source_e, double **source_o, double **prop_e, double **prop_o, double *gauge_field, double **mzz, double **mzzinv, int const nfields ) {

  const unsigned int Vhalf = VOLUME / 2;

  // temporary eo spinors without halo
  double **eo_spinor_field = init_2level_dtable ( 4, _GSI( (size_t)Vhalf ) );
  if ( eo_spinor_field == NULL ) {
    fprintf(stderr, "[check_residuum_eo] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  // temporary eo spinors  with halo
  double **eo_spinor_work = init_2level_dtable ( 2, _GSI( (size_t)( VOLUME+RAND)/2 ) );
  if ( eo_spinor_work == NULL ) {
    fprintf(stderr, "[check_residuum_eo] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  for ( int i = 0; i < nfields; i++ ) {

    // apply Dirac operator
    // Q_clover_phi_eo ( eo_spinor_field[0], eo_spinor_field[1], prop_e[i], prop_o[i], gauge_field, mutm, eo_spinor_work[0], mzz);

    // D in Schur decomp; apply D = g5 Q  = g5 A B
    Q_clover_eo_SchurDecomp_B ( eo_spinor_field[0], eo_spinor_field[1], prop_e[i],          prop_o[i],          gauge_field, mzz[1], mzzinv[0], eo_spinor_work[0] );
    Q_clover_eo_SchurDecomp_A ( eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[0], eo_spinor_field[1], gauge_field, mzz[0],            eo_spinor_work[0] );
    g5_phi( eo_spinor_field[2], VOLUME );

    // norm diff to source
    double norm_e = 0., norm_o = 0.;
    spinor_field_norm_diff ( &norm_e, eo_spinor_field[2], source_e[i], Vhalf );
    spinor_field_norm_diff ( &norm_o, eo_spinor_field[3], source_o[i], Vhalf );
    if ( g_cart_id == 0 ) fprintf ( stdout, "# [check_residuum_eo] norm diff %2d e %16.7e o %16.7e\n", i, norm_e, norm_o );

  }  // end of loop on field components

  fini_2level_dtable ( &eo_spinor_field );
  fini_2level_dtable ( &eo_spinor_work );

  return(0);
}  // end of check_residuum_eo

}  /* end of namespace cvc */
