/****************************************************
 * contract_cvc_tensor_mixed.cpp
 *
 * Wed Jul  5 07:54:34 CEST 2017
 *
 * PURPOSE:
 * - contractions with mixed propagtor and evec setup
 * DONE:
 * TODO:
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
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "cvc_complex.h"
#include "iblas.h"
#include "ilinalg.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "matrix_init.h"
#include "project.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor_mixed.h"

namespace cvc {

/************************************************************
 * calculate gsp using t-blocks
 *
          subroutine zgemm  (   character   TRANSA,
          V^+ Gamma(p) S
          eo - scalar product over even 0 / odd 1 sites

          V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)

          prop is nsf x (12 VOL3half) (C) = (12 VOL3half) x nsf (F)

          zgemm calculates
          V^H x [ (Gamma(p) x prop) ] which is numV x nsf (F) = nsf x numV (C)
 *
 ************************************************************/
int contract_vdag_gloc_spinor_field (
    double**prop_list_e, 
    double**prop_list_o, 
    int nsf,
    double**V, 
    int numV, 
    int momentum_number, 
    int (*momentum_list)[3], 
    int gamma_id_number, 
    int*gamma_id_list, 
    struct AffWriter_s*affw, 
    char*tag, int io_proc, 
    double*gauge_field, 
    double **mzz[2], 
    double**mzzinv[2] ) {
 
  const unsigned int Vhalf    = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field           = _GSI(  Vhalf )          * sizeof(double);
  const size_t sizeof_eo_spinor_field_with_halo = _GSI( (VOLUME+RAND)/2 ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_timeslice = _GSI(  VOL3half )       * sizeof(double);

  int exitstatus;
  double *spinor_work = NULL;
  double *spinor_aux  = NULL;
  double _Complex **phase_field = NULL;
  double _Complex **W           = NULL;
  double _Complex **V_ts        = NULL;
  double _Complex **prop_ts     = NULL;
  double _Complex **prop_phase  = NULL;
  double _Complex ***contr      = NULL;
  double _Complex *contr_allt_buffer = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_path[400];

  /* BLAS parameters for zgemm */
  char BLAS_TRANSA = 'C';
  char BLAS_TRANSB = 'N';
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  int BLAS_M = numV;
  int BLAS_K = 12*VOL3half;
  int BLAS_N = nsf;
  int BLAS_LDA = BLAS_K;
  int BLAS_LDB = BLAS_K;
  int BLAS_LDC = BLAS_M;

  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  exitstatus = init_2level_buffer ( (double***)(&prop_ts), nsf, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&prop_phase), nsf, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_3level_buffer ( (double****)(&contr), T, nsf, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  mcontr_buffer = (double*)malloc(numV*nsf*2*sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }

  /************************************************
   ************************************************
   **
   ** V, odd part
   **
   ************************************************
   ************************************************/

  /* loop on momenta */
  for( int im=0; im<momentum_number; im++ ) {
  
    /* make odd phase field */
    make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

    /* calculate the propagators including current Fourier phase */
    for( int i=0; i<nsf; i++) {
      spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_phase[i]), prop_list_o[i], (double*)(phase_field[0]), Vhalf);
    }

    for( int ig=0; ig<gamma_id_number; ig++ ) {

      for( int it=0; it<T; it++ ) {

        /* copy timslice of V  */
        unsigned int offset = _GSI(VOL3half) * it;
        for( int i=0; i<numV; i++ ) memcpy( V_ts[i], V[i]+offset, sizeof_eo_spinor_field_timeslice );

        /* calculate Gamma times spinor field timeslice 
         *
         *  prop_ts = g5 Gamma prop_list_o [it]
         *
         */
        for( int i=0; i<nsf; i++) {
          spinor_field_eq_gamma_ti_spinor_field( (double*)(prop_ts[i]), gamma_id_list[ig], (double*)(prop_phase[i])+offset, VOL3half);
        }
        g5_phi( (double*)(prop_ts[0]), nsf*VOL3half);

        /* matrix multiplication
         *
         * contr[it][i][k] = V_i^+ prop_ts_k
         *
         */

        BLAS_A = V_ts[0];
        BLAS_B = prop_ts[0];
        BLAS_C = contr[it][0];

         F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
         memcpy( mcontr_buffer,  contr[it][0], numV*nsf*sizeof(double _Complex) );
         exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*nsf, MPI_DOUBLE, MPI_SUM, g_ts_comm);
         if( exitstatus != MPI_SUCCESS ) {
           fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
           return(8);
         }
#endif
      }  /* end of loop on timeslices */

      /************************************************
       * write to file
       ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
      if ( io_proc == 2 ) {
        contr_allt_buffer = (double _Complex *)malloc(numV*nsf*T_global*sizeof(double _Complex) );
        if(contr_allt_buffer == NULL ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
          return(9);
        }
      }

      /* gather to root, which must be io_proc = 2 */
      if ( io_proc > 0 ) {
        int count = numV*nsf*2*T;
        exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(10);
         }
      }
#else
      if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
      return(1);
#endif
#else
      contr_allt_buffer = contr[0][0];
#endif
      if ( io_proc == 2 ) {
        sprintf(aff_path, "%s/v_dag_gloc_s/px%.2dpy%.2dpz%.2d/g%.2d", tag, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);
        /* fprintf(stdout, "# [contract_vdag_gloc_spinor_field] node = %s\n", aff_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_path);
        exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*nsf ) );
        if(exitstatus != 0) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(11);
        }

#ifdef HAVE_MPI
        free( contr_allt_buffer );
#endif
      }
    }  /* end of loop on Gamma structures */
  }  /* end of loop on momenta */

  if(g_cart_id == 0) fprintf(stdout, "# [contract_vdag_gloc_spinor_field] end of V part\n");
  fflush(stdout);
#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif
#if 0
#endif  /* of if 0 */


  /************************************************
   ************************************************
   **
   ** Xbar V, even part
   **
   ************************************************
   ************************************************/

  /************************************************
   * allocate auxilliary W
   ************************************************/
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  /* calculate Xbar V */
  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = Xbar V_o = -M_ee^-1[dn] M_eo V_o
     */
    memcpy( spinor_work, (double*)(V[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[1][0] );
  }
  free ( spinor_work ); spinor_work = NULL;

  /* loop on momenta */
  for( int im=0; im<momentum_number; im++ ) {
  
    /* make even phase field */
    make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

    /* calculate the propagators including current Fourier phase */
    for( int i=0; i<nsf; i++) {
      spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_phase[i]), prop_list_e[i], (double*)(phase_field[0]), Vhalf);
    }

    for( int ig=0; ig<gamma_id_number; ig++ ) {

      for( int it=0; it<T; it++ ) {

        /* copy timslice of V  */
        unsigned int offset = _GSI(VOL3half) * it;
        for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );

        /* calculate Gamma times spinor field timeslice 
         *
         *  prop_ts = g5 Gamma prop_list_o [it]
         *
         */
        for( int i=0; i<nsf; i++) {
          spinor_field_eq_gamma_ti_spinor_field( (double*)(prop_ts[i]), gamma_id_list[ig], (double*)(prop_phase[i])+offset, VOL3half);
        }
        g5_phi( (double*)(prop_ts[0]), nsf*VOL3half);

        /* matrix multiplication
         *
         * contr[it][i][k] = V_i^+ prop_ts_k
         *
         */

        BLAS_A = V_ts[0];
        BLAS_B = prop_ts[0];
        BLAS_C = contr[it][0];

         F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
         memcpy( mcontr_buffer,  contr[it][0], numV*nsf*sizeof(double _Complex) );
         exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*nsf, MPI_DOUBLE, MPI_SUM, g_ts_comm);
         if( exitstatus != MPI_SUCCESS ) {
           fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
           return(8);
         }
#endif

      }  /* end of loop on timeslices */

      /************************************************
       * write to file
       ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
      if ( io_proc == 2 ) {
        contr_allt_buffer = (double _Complex *)malloc(numV*nsf*T_global*sizeof(double _Complex) );
        if(contr_allt_buffer == NULL ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
          return(9);
        }
      }

      /* gather to root, which must be io_proc = 2 */
      if ( io_proc > 0 ) {
        int count = numV*nsf*2*T;
        exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(10);
         }
      }
#else
      if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
      return(1);
#endif
#else
     contr_allt_buffer = contr[0][0];
#endif
      if ( io_proc == 2 ) {
        sprintf(aff_path, "%s/xv_dag_gloc_s/px%.2dpy%.2dpz%.2d/g%.2d", tag, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        affdir = aff_writer_mkpath(affw, affn, aff_path);
        exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*nsf ) );
        if(exitstatus != 0) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(11);
        }
#ifdef HAVE_MPI
        free( contr_allt_buffer );
#endif
      }

    }  /* end of loop on Gamma structures */


  }  /* end of loop on momenta */


  if(g_cart_id == 0) fprintf(stdout, "# [contract_vdag_gloc_spinor_field] end of Xbar V part\n");
  fflush(stdout);
#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif
#if 0
#endif  /* of if 0 */

  /************************************************
   ************************************************
   **
   ** W, odd part
   **
   ************************************************
   ************************************************/
  /* calculate W from V and Xbar V */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  spinor_aux  = (double*)malloc( sizeof_eo_spinor_field );
  if ( spinor_work == NULL || spinor_aux == NULL ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W_o = Cbar V
     */
    memcpy( spinor_work, (double*)(W[i]), sizeof_eo_spinor_field );
    memcpy( (double*)(W[i]), (double*)(V[i]), sizeof_eo_spinor_field );
    C_clover_from_Xeo ( (double*)(W[i]), spinor_work, spinor_aux, gauge_field, mzz[1][1]);
                  
    // memcpy( spinor_aux, (double*)(V[i]), sizeof_eo_spinor_field );
    // C_clover_oo ( (double*)(W[i]), spinor_aux, gauge_field, spinor_work, mzz[1][1], mzzinv[1][0]);

  }
  free ( spinor_work ); spinor_work = NULL;
  free ( spinor_aux  ); spinor_aux = NULL;


  /* loop on momenta */
  for( int im=0; im<momentum_number; im++ ) {
  
    /* make odd phase field */
    make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

    /* calculate the propagators including current Fourier phase */
    for( int i=0; i<nsf; i++) {
      spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_phase[i]), prop_list_o[i], (double*)(phase_field[0]), Vhalf);
    }

    for( int ig=0; ig<gamma_id_number; ig++ ) {

      for( int it=0; it<T; it++ ) {

        /* copy timslice of V  */
        unsigned int offset = _GSI(VOL3half) * it;
        for( int i=0; i<numV; i++ ) memcpy( (double*)(V_ts[i]), (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );

        /* calculate Gamma times spinor field timeslice 
         *
         *  prop_ts = g5 Gamma prop_list_o [it]
         *
         */
        for( int i=0; i<nsf; i++) {
          spinor_field_eq_gamma_ti_spinor_field( (double*)(prop_ts[i]), gamma_id_list[ig], (double*)(prop_phase[i])+offset, VOL3half);
        }
        g5_phi( (double*)(prop_ts[0]), nsf*VOL3half);

        /* matrix multiplication
         *
         * contr[it][i][k] = V_i^+ prop_ts_k
         *
         */

        BLAS_A = V_ts[0];
        BLAS_B = prop_ts[0];
        BLAS_C = contr[it][0];

         F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
         memcpy( mcontr_buffer,  contr[it][0], numV*nsf*sizeof(double _Complex) );
         exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*nsf, MPI_DOUBLE, MPI_SUM, g_ts_comm);
         if( exitstatus != MPI_SUCCESS ) {
           fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
           return(8);
         }
#endif
      }  /* end of loop on timeslices */

      /************************************************
       * write to file
       ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
      if ( io_proc == 2 ) {
        contr_allt_buffer = (double _Complex *)malloc(numV*nsf*T_global*sizeof(double _Complex) );
        if(contr_allt_buffer == NULL ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
          return(9);
        }
      }

      /* gather to root, which must be io_proc = 2 */
      if ( io_proc > 0 ) {
        int count = numV*nsf*2*T;
        exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(10);
         }
      }
#else
      if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
      return(1);
#endif
#else
      contr_allt_buffer = contr[0][0];
#endif
      if ( io_proc == 2 ) {
        sprintf(aff_path, "%s/w_dag_gloc_s/px%.2dpy%.2dpz%.2d/g%.2d", tag, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        affdir = aff_writer_mkpath(affw, affn, aff_path);
        exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*nsf ) );
        if(exitstatus != 0) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(11);
        }

#if HAVE_MPI
        free( contr_allt_buffer );
#endif
      }
    }  /* end of loop on Gamma structures */
  }  /* end of loop on momenta */

  if(g_cart_id == 0) fprintf(stdout, "# [contract_vdag_gloc_spinor_field] end of W part\n");
  fflush(stdout);
#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif
#if 0
#endif  /* of if 0 */

  /************************************************
   ************************************************
   **
   ** XW, even part
   **
   ************************************************
   ************************************************/

  /* calculate X W */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = X W_o = -M_ee^-1[up] M_eo W_o
     */
    memcpy( spinor_work, (double*)(W[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[0][0] );
  }
  free ( spinor_work ); spinor_work = NULL;

  /* loop on momenta */
  for( int im=0; im<momentum_number; im++ ) {
  
    /* make even phase field */
    make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

    /* calculate the propagators including current Fourier phase */
    for( int i=0; i<nsf; i++) {
      spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_phase[i]), prop_list_e[i], (double*)(phase_field[0]), Vhalf);
    }

    for( int ig=0; ig<gamma_id_number; ig++ ) {

      for( int it=0; it<T; it++ ) {

        /* copy timslice of V  */
        unsigned int offset = _GSI(VOL3half) * it;
        for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );

        /* calculate Gamma times spinor field timeslice 
         *
         *  prop_ts = g5 Gamma prop_list_o [it]
         *
         */
        for( int i=0; i<nsf; i++) {
          spinor_field_eq_gamma_ti_spinor_field( (double*)(prop_ts[i]), gamma_id_list[ig], (double*)(prop_phase[i])+offset, VOL3half);
        }
        g5_phi( (double*)(prop_ts[0]), nsf*VOL3half);

        /* matrix multiplication
         *
         * contr[it][i][k] = V_i^+ prop_ts_k
         *
         */

        BLAS_A = V_ts[0];
        BLAS_B = prop_ts[0];
        BLAS_C = contr[it][0];

         F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
         memcpy( mcontr_buffer,  contr[it][0], numV*nsf*sizeof(double _Complex) );
         exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*nsf, MPI_DOUBLE, MPI_SUM, g_ts_comm);
         if( exitstatus != MPI_SUCCESS ) {
           fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
           return(8);
         }
#endif
      }  /* end of loop on timeslices */

      /************************************************
       * write to file
       ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
      if ( io_proc == 2 ) {
        contr_allt_buffer = (double _Complex *)malloc(numV*nsf*T_global*sizeof(double _Complex) );
        if(contr_allt_buffer == NULL ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
          return(9);
        }
      }

      /* gather to root, which must be io_proc = 2 */
      if ( io_proc > 0 ) {
        int count = numV*nsf*2*T;
        exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(10);
         }
      }
#else
      if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
      return(1);
#endif
#else
      contr_allt_buffer = contr[0][0];
#endif

      if ( io_proc == 2 ) {
        sprintf(aff_path, "%s/xw_dag_gloc_s/px%.2dpy%.2dpz%.2d/g%.2d", tag, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        affdir = aff_writer_mkpath(affw, affn, aff_path);
        exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*nsf ) );
        if(exitstatus != 0) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(11);
        }
#ifdef HAVE_MPI
        free( contr_allt_buffer );
#endif
      }
    }  /* end of loop on Gamma structures */
  }  /* end of loop on momenta */

  if(g_cart_id == 0) fprintf(stdout, "# [contract_vdag_gloc_spinor_field] end of X W part\n");
  fflush(stdout);
#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif

#if 0
#endif  /* of if 0 */

  fini_2level_buffer ( (double***)(&W) );

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&prop_ts) );
  fini_2level_buffer ( (double***)(&prop_phase) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_vdag_gloc_spinor_field] time for contract_vdag_gloc_spinor_field = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_gloc_spinor_field */


/*******************************************************************************************
 * calculate V^+ cvc-vertex S
 *
          subroutine zgemm  (   character   TRANSA,
          V^+ Gamma(p) S
          eo - scalar product over even 0 / odd 1 sites

          V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)

          prop is nsf x (12 VOL3half) (C) = (12 VOL3half) x nsf (F)

          zgemm calculates
          V^H x [ (Gamma(p) x prop) ] which is numV x nsf (F) = nsf x numV (C)
 *
 *******************************************************************************************/
int contract_vdag_cvc_spinor_field (double**prop_list_e, double**prop_list_o, int nsf, double**V, int numV, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size ) {
 
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double);

  int exitstatus;
  double **eo_spinor_work = NULL, **eo_spinor_aux = NULL;
  double _Complex **phase_field = NULL;
  double _Complex **W = NULL;
  double _Complex **V_ts = NULL;
  double _Complex **prop_ts = NULL, **prop_vertex = NULL;
  double _Complex ***contr = NULL;
  double _Complex *contr_allt_buffer = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_path[200];

  /* BLAS parameters for zgemm */
  char BLAS_TRANSA = 'C';
  char BLAS_TRANSB = 'N';
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  int BLAS_M = numV;
  int BLAS_K = 12*VOL3half;
  int BLAS_N = block_size;
  int BLAS_LDA = BLAS_K;
  int BLAS_LDB = BLAS_K;
  int BLAS_LDC = BLAS_M;

  int block_num = (int)(nsf / block_size);
  if ( block_num * block_size != nsf ) {
    fprintf(stderr, "[contract_v_dag_cvc_spinor_field] Error, nsf must be divisible by block size %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( (double***)(&eo_spinor_work), 1, _GSI( (VOLUME+RAND)/2) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  exitstatus = init_2level_buffer ( (double***)(&prop_ts), block_size, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&prop_vertex), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_3level_buffer ( (double****)(&contr), T, block_size, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  mcontr_buffer = (double*)malloc(numV * block_size * 2 * sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }


  /************************************************
   ************************************************
   **
   ** V, odd part
   **
   ************************************************
   ************************************************/

  for ( int iblock=0; iblock < block_num; iblock++ ) {

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* copy propagator to field with halo */
          memcpy( eo_spinor_work[0], prop_list_e[iblock*block_size + i], sizeof_eo_spinor_field );
          /* apply vertex for ODD target field */
          apply_cvc_vertex_eo((double*)(prop_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          for( int it=0; it<T; it++ ) {

            /* copy timslice of V  */
            unsigned int offset = _GSI(VOL3half) * it;
            for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(V[i])+offset, sizeof_eo_spinor_field_timeslice );

            /*
             *
             *
             */
            for( int i=0; i<block_size; i++) {
              spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_ts[i]), (double*)(prop_vertex[i])+offset, (double*)(phase_field[it]), VOL3half);
            }
            g5_phi( (double*)(prop_ts[0]), block_size*VOL3half);

            /* matrix multiplication
             *
             * contr[it][i][k] = V_i^+ prop_ts_k
             *
             */

            BLAS_A = V_ts[0];
            BLAS_B = prop_ts[0];
            BLAS_C = contr[it][0];

            F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
            memcpy( mcontr_buffer,  contr[it][0], numV*block_size*sizeof(double _Complex) );
            exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*block_size, MPI_DOUBLE, MPI_SUM, g_ts_comm);
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(8);
            }
#endif
          }  /* end of loop on timeslices */

          /************************************************
           * write to file
           ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
          if ( io_proc == 2 ) {
            contr_allt_buffer = (double _Complex *)malloc(numV*block_size*T_global*sizeof(double _Complex) );
            if(contr_allt_buffer == NULL ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
              return(9);
            }
          }

          /* gather to root, which must be io_proc = 2 */
          if ( io_proc > 0 ) {
            int count = numV * block_size * 2 * T;
            exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(10);
             }
          }
#else
          if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
          return(1);
#endif
#else
          contr_allt_buffer = contr[0][0];
#endif
          if ( io_proc == 2 ) {
            sprintf(aff_path, "%s/v_dag_cvc_s/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);
    
            affdir = aff_writer_mkpath(affw, affn, aff_path);
            exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*block_size ) );
            if(exitstatus != 0) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(11);
            }
#ifdef HAVE_MPI
            free( contr_allt_buffer );
#endif
          }
        }  /* end of loop on momenta */
      }   /* end of loop on momenta */
    }  /* end of loop on shift directions */
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */

  /************************************************
   ************************************************
   **
   ** Xbar V, even part
   **
   ************************************************
   ************************************************/

  /************************************************
   * allocate auxilliary W
   ************************************************/
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }


  /* calculate Xbar V */
  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = Xbar V_o = -M_ee^-1[dn] M_eo V_o
     */
    memcpy( eo_spinor_work[0], (double*)(V[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), eo_spinor_work[0], gauge_field, mzzinv[1][0] );
  }

  for( int iblock=0; iblock < block_num; iblock++ ) {

    for ( int mu=0; mu<4; mu++ ) {

      for ( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* copy propagator to field with halo */
          memcpy( eo_spinor_work[0], prop_list_o[iblock*block_size + i], sizeof_eo_spinor_field );
          /* apply vertex for ODD target field */
          apply_cvc_vertex_eo((double*)(prop_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

    
        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
      
          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);
    
          for( int it=0; it<T; it++ ) {
    
            /* copy timslice of V  */
            unsigned int offset = _GSI(VOL3half) * it;
            for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );
    
            /*
             *
             *
             */
            for( int i=0; i<block_size; i++) {
              spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_ts[i]), (double*)(prop_vertex[i])+offset, (double*)(phase_field[it]), VOL3half);
            }
            g5_phi( (double*)(prop_ts[0]), block_size*VOL3half);
    
            /* matrix multiplication
             *
             * contr[it][i][k] = V_i^+ prop_ts_k
             *
             */
    
            BLAS_A = V_ts[0];
            BLAS_B = prop_ts[0];
            BLAS_C = contr[it][0];
    
            F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);
    
#ifdef HAVE_MPI
            memcpy( mcontr_buffer,  contr[it][0], numV*block_size*sizeof(double _Complex) );
            exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*block_size, MPI_DOUBLE, MPI_SUM, g_ts_comm);
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(8);
            }
#endif
          }  /* end of loop on timeslices */
    
          /************************************************
           * write to file
           ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
          if ( io_proc == 2 ) {
            contr_allt_buffer = (double _Complex *)malloc(numV*block_size*T_global*sizeof(double _Complex) );
            if(contr_allt_buffer == NULL ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
              return(9);
            }
          }
    
          /* gather to root, which must be io_proc = 2 */
          if ( io_proc > 0 ) {
            int count = numV*block_size*2*T;
            exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(10);
             }
          }
#else
          if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
          return(1);
#endif
#else
          contr_allt_buffer = contr[0][0];
#endif

          if ( io_proc == 2 ) {
            sprintf(aff_path, "%s/xv_dag_cvc_s/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);
    
            affdir = aff_writer_mkpath(affw, affn, aff_path);
            exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*block_size ) );
            if(exitstatus != 0) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(11);
            }
#ifdef HAVE_MPI
            free( contr_allt_buffer );
#endif
          }
        }  /* end of loop on momenta */  
      }  /* end of loop on fbwd */

    }  /* end of loop on shift directions mu */

  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */


  /************************************************
   ************************************************
   **
   ** W, odd part
   **
   ************************************************
   ************************************************/


  /* calculate W from V and Xbar V */
  exitstatus = init_2level_buffer ( (double***)(&eo_spinor_aux), 1, _GSI( Vhalf ) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
  for( int i=0; i<numV; i++ ) {
    /*
     * W_o = Cbar V
         */
    memcpy( eo_spinor_work[0], (double*)(W[i]), sizeof_eo_spinor_field );
    memcpy( (double*)(W[i]), (double*)(V[i]), sizeof_eo_spinor_field );
    C_clover_from_Xeo ( (double*)(W[i]), eo_spinor_work[0], eo_spinor_aux[0], gauge_field, mzz[1][1]);
  }
  fini_2level_buffer ( (double***)(&eo_spinor_aux) );

  for( int iblock=0; iblock < block_num; iblock++ ) {

    for ( int mu=0; mu<4; mu++ ) {

      for ( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* copy propagator to field with halo */
          memcpy( eo_spinor_work[0], prop_list_e[iblock*block_size + i], sizeof_eo_spinor_field );
          /* apply vertex for ODD target field */
          apply_cvc_vertex_eo((double*)(prop_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
      
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);
    
          for( int it=0; it<T; it++ ) {
    
            /* copy timslice of V  */
            unsigned int offset = _GSI(VOL3half) * it;
            for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );
    
            /* 
             *
             */
            for( int i=0; i<block_size; i++) {
              spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_ts[i]), (double*)(prop_vertex[i])+offset, (double*)(phase_field[it]), VOL3half);
            }
            g5_phi( (double*)(prop_ts[0]), block_size*VOL3half);
    
            /* matrix multiplication
             *
             * contr[it][i][k] = V_i^+ prop_ts_k
             *
             */
    
            BLAS_A = V_ts[0];
            BLAS_B = prop_ts[0];
            BLAS_C = contr[it][0];
    
             F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);
    
#ifdef HAVE_MPI
             memcpy( mcontr_buffer,  contr[it][0], numV*block_size*sizeof(double _Complex) );
             exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*block_size, MPI_DOUBLE, MPI_SUM, g_ts_comm);
             if( exitstatus != MPI_SUCCESS ) {
               fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
               return(8);
             }
#endif
          }  /* end of loop on timeslices */
    
          /************************************************
           * write to file
           ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
          if ( io_proc == 2 ) {
            contr_allt_buffer = (double _Complex *)malloc(numV*block_size*T_global*sizeof(double _Complex) );
            if(contr_allt_buffer == NULL ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
              return(9);
            }
          }
    
          /* gather to root, which must be io_proc = 2 */
          if ( io_proc > 0 ) {
            int count = numV*block_size*2*T;
            exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(10);
             }
          }

#else
          if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
          return(1);
#endif
#else
          contr_allt_buffer = contr[0][0];
#endif
          if ( io_proc == 2 ) {
            sprintf(aff_path, "%s/w_dag_cvc_s/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);
    
            affdir = aff_writer_mkpath(affw, affn, aff_path);
            exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*block_size ) );
            if(exitstatus != 0) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(11);
            }
    
#ifdef HAVE_MPI
            free( contr_allt_buffer );
#endif
          }

        }  /* end of loop on momenta */ 
      }  /* end of loop on fbwd */
    }  /* end of loop on shift directions mu */
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */


  /************************************************
   ************************************************
   **
   ** XW, even part
   **
   ************************************************
   ************************************************/

  /* calculate X W */
  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = X W_o = -M_ee^-1[up] M_eo W_o
     */
    memcpy( eo_spinor_work[0], (double*)(W[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), eo_spinor_work[0], gauge_field, mzzinv[0][0] );
  }

  for( int iblock=0; iblock < block_num; iblock++ ) {

    for ( int mu=0; mu<4; mu++ ) {

      for ( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* copy propagator to field with halo */
          memcpy( eo_spinor_work[0], prop_list_o[iblock*block_size + i], sizeof_eo_spinor_field );
          /* apply vertex for ODD target field */
          apply_cvc_vertex_eo((double*)(prop_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
      
          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);
    
          for( int it=0; it<T; it++ ) {
    
            /* copy timslice of V  */
            unsigned int offset = _GSI(VOL3half) * it;
            for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );
    
            /*
             *
             *
             */
            for( int i=0; i<block_size; i++) {
              spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_ts[i]), (double*)(prop_vertex[i])+offset, (double*)(phase_field[it]), VOL3half);
            }
            g5_phi( (double*)(prop_ts[0]), block_size*VOL3half);
    
            /* matrix multiplication
             *
             * contr[it][i][k] = V_i^+ prop_ts_k
             *
             */
    
            BLAS_A = V_ts[0];
            BLAS_B = prop_ts[0];
            BLAS_C = contr[it][0];
    
             F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);
    
#ifdef HAVE_MPI
             memcpy( mcontr_buffer,  contr[it][0], numV*block_size*sizeof(double _Complex) );
             exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*block_size, MPI_DOUBLE, MPI_SUM, g_ts_comm);
             if( exitstatus != MPI_SUCCESS ) {
               fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
               return(8);
             }
#endif
          }  /* end of loop on timeslices */
    
          /************************************************
           * write to file
           ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
          if ( io_proc == 2 ) {
            contr_allt_buffer = (double _Complex *)malloc(numV*block_size*T_global*sizeof(double _Complex) );
            if(contr_allt_buffer == NULL ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
              return(9);
            }
          }
    
          /* gather to root, which must be io_proc = 2 */
          if ( io_proc > 0 ) {
            int count = numV*block_size*2*T;
            exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(10);
             }
          }
#else
          if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
          return(1);
#endif
#else
          contr_allt_buffer = contr[0][0];
#endif

          if ( io_proc == 2 ) {
            sprintf(aff_path, "%s/xw_dag_cvc_s/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);
    
            affdir = aff_writer_mkpath(affw, affn, aff_path);
            exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*block_size ) );
            if(exitstatus != 0) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(11);
            }
#ifdef HAVE_MPI   
            free( contr_allt_buffer );
#endif
          }

        }  /* end of loop on momenta */  
      }  /* end of loop on fbwd */
    }  /* end of loop on shift directions mu */
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */

  fini_2level_buffer ( (double***)(&W) );

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&eo_spinor_work) );
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&prop_ts) );
  fini_2level_buffer ( (double***)(&prop_vertex) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_vdag_cvc_spinor_field] time for contract_vdag_gloc_spinor_field = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_cvc_spinor_field */

}  /* end of namespace cvc */
