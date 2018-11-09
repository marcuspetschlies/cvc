/****************************************************
 * contract_cvc_tensor_all2all.cpp
 *
 * Sun Feb  5 13:23:50 CET 2017
 *
 * PURPOSE:
 * - contractions for cvc-cvc tensor
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
#include "contract_cvc_tensor_all2all.h"

namespace cvc {


/************************************************************
 * vdag_w_reduce_write
 ************************************************************/
int vdag_w_reduce_write (
    double _Complex *** const contr,
    double _Complex ** const V, double _Complex ** const W, int const dimV, int const dimW,
    char * const aff_path, struct AffWriter_s * const affw, struct AffNode_s * const affn, int const io_proc,
    double _Complex **V_ts, double _Complex **W_ts, double *mcontr_buffer
    ) {

  static const unsigned int Vhalf = VOLUME / 2;
  static const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  static const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  static const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double);

  double _Complex ** V_ts = init_2level_ztable ( dimV, 12 * (size_t)VOL3half );
  double _Complex ** W_ts = init_2level_ztable ( dimV, 12 * (size_t)VOL3half );
  if ( V_ts == NULL || W_ts == NULL ) {
    fprintf(stderr, "[vdag_w_reduce_write] Error frominit_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  int exitstatus;
  /* BLAS parameters for zgemm */
  char CHAR_C = 'C';
  char CHAR_N = 'N';
  double _Complex Z_ONE   = 1.;
  double _Complex Z_ZERO  = 0.;
  int BLAS_M = dimV;
  int BLAS_K = 12*VOL3half;
  int BLAS_N = dimW;
  int BLAS_LDA = BLAS_K;
  int BLAS_LDB = BLAS_K;
  int BLAS_LDC = BLAS_M;
  double _Complex *BLAS_A = V_ts[0];
  double _Complex *BLAS_B = W_ts[0];
  double _Complex *BLAS_C = NULL;
  double _Complex *contr_allt_buffer = NULL;

  for( int it=0; it<T; it++ ) {

    /* copy timslice of V  */
    unsigned int offset = _GSI(VOL3half) * it;
    for( int i=0; i<dimV; i++ ) memcpy( V_ts[i], (double*)(V[i])+offset, sizeof_eo_spinor_field_timeslice );
    for( int i=0; i<dimW; i++ ) memcpy( W_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );

    BLAS_C = contr[it][0];

    F_GLOBAL(zgemm, ZGEMM) ( &CHAR_C, &CHAR_N, &BLAS_M, &BLAS_N, &BLAS_K, &Z_ONE, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &Z_ZERO, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
    memcpy( mcontr_buffer,  contr[it][0], dimV * dimW * sizeof(double _Complex) );
    exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*dimV*dimW, MPI_DOUBLE, MPI_SUM, g_ts_comm);
    if( exitstatus != MPI_SUCCESS ) {
      fprintf(stderr, "[vdag_w_reduce_write] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(8);
    }
#endif
  }  /* end of loop on timeslices */

  /************************************************
   * write to file
   ************************************************/
#ifdef HAVE_MPI
  if ( io_proc == 2 ) {
    contr_allt_buffer = (double _Complex *)malloc(dimV*dimW * T_global*sizeof(double _Complex) );
    if(contr_allt_buffer == NULL ) {
      fprintf(stderr, "[vdag_w_reduce_write] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(9);
    }
  }
#else
  contr_allt_buffer = contr[0][0];
#endif

#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  /* gather to root, which must be io_proc = 2 */
  if ( io_proc > 0 ) {
    int count = dimV * dimW * 2 * T;
    exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
    if( exitstatus != MPI_SUCCESS ) {
      fprintf(stderr, "[vdag_w_reduce_write] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(10);
    }
  }
#else
  if (io_proc == 2) {
      fprintf(stderr, "[vdag_w_reduce_write] Error, not implemented yet\n");
  }
  return(1);
#endif
#endif

  if ( io_proc == 2 ) {
    uint32_t count = dimV * dimW * T_global;
    struct AffNode_s *affdir = aff_writer_mkpath(affw, affn, aff_path);
    exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, count );
    if(exitstatus != 0) {
      fprintf(stderr, "[vdag_w_reduce_write] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(11);
    }

#ifdef HAVE_MPI
    free( contr_allt_buffer );
#endif
  }

  fini_2level_ztable ( &V_ts );
  fini_2level_ztable ( &W_ts );
  return(0);
}  /* end of vdag_w_reduce_write */

#if 0
/************************************************************
 * calculate gsp V^+ Gamma(p) W
 *
 *   local vertex Gamma(p)
 *   eo - scalar product over even 0 / odd 1 sites
 *
 *   V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)
 *
 *   Wb is block_size x (12 VOL3half) (C) = (12 VOL3half) x block_size (F)
 *
 *   XbarV, W, XW and their blocked versions all the same formats
 *
 *   zgemm calculates
 *   V^H x [ Gamma(p) x Wb ) ] which is numV x block_size (F) = block_size x numV (C)
 *
 ************************************************************/
int contract_vdag_gloc_w_blocked (double**V, int numV, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size ) {
 
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_with_halo = _GSI( (VOLUME+RAND)/2 ) * sizeof(double);
  /* const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double); */

  int exitstatus;
  double *spinor_work = NULL, *spinor_aux = NULL;
  double _Complex **phase_field = NULL;
  double _Complex **V_ts = NULL;
  double _Complex **W = NULL, **W_phase = NULL, **W_gamma_phase = NULL, **W_ts = NULL, **W_aux = NULL;
  double _Complex ***contr = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL;
  char aff_path[200];

  int block_num = (int)(numV / block_size);
  if ( block_num * block_size != numV ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error, numV must be divisible by block size %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;

  /**********************************
   * Fourier phase field
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /**********************************
   * auxilliary V-fields
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /**********************************
   * auxilliary W-fields
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&W_ts), block_size, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_gamma_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  /**********************************
   * fields for contractions
   **********************************/
  exitstatus = init_3level_buffer ( (double****)(&contr), T, block_size, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  mcontr_buffer = (double*)malloc(numV*block_size*2*sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }


  /************************************************
   ************************************************
   **
   ** V - V (odd part)
   **
   ************************************************
   ************************************************/

  /* loop on blocks of fields */
  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make odd phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++ ) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(V[iblock*block_size + i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        /* calculate Gamma times spinor field timeslice 
         *
         *  prop_ts = g5 Gamma prop_list_o [it]
         *
         */
        spinor_field_eq_gamma_ti_spinor_field( (double*)(W_gamma_phase[0]), gamma_id_list[ig], (double*)(W_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(W_gamma_phase[0]), block_size * Vhalf);

        /* prepare the AFF key */
        sprintf(aff_path, "%s/v_dag_gloc_v/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

/************************************************
 * NOTE: this casting may be dangerous
 *   (double _Complex**)V since V is double**
 *
 ************************************************/
        exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }
      }  /* end of loop on Gamma structures */


    }  /* end of loop on momenta */

  }  /* end of loops on blocks */



  /************************************************
   ************************************************
   **
   ** Xbar V - Xbar V (even part)
   **
   ************************************************
   ************************************************/


  /* calculate Xbar V */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = Xbar V_o = -M_ee^-1[dn] M_eo V_o
     */
    memcpy( spinor_work, (double*)(V[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[1][0] );
  }
  free ( spinor_work ); spinor_work = NULL;

  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W[iblock * block_size + i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {
 
        spinor_field_eq_gamma_ti_spinor_field( (double*)(W_gamma_phase[0]), gamma_id_list[ig], (double*)(W_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(W_gamma_phase[0]), block_size * Vhalf);

        sprintf(aff_path, "%s/xv_dag_gloc_xv/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  
  }  /* end of loop on blocks */
#if 0 
#endif  /* of if 0 */



  /************************************************
   ************************************************
   **
   ** V - W (odd part) and
   ** W - W (odd part)
   **
   ************************************************
   ************************************************/
  /* calculate W from V and Xbar V */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  spinor_aux  = (double*)malloc( sizeof_eo_spinor_field );
  if ( spinor_work == NULL || spinor_aux == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W_o = Cbar V
     */
    memcpy( spinor_work, (double*)(W[i]), sizeof_eo_spinor_field );
    memcpy( (double*)(W[i]), (double*)(V[i]), sizeof_eo_spinor_field );
    C_clover_from_Xeo ( (double*)(W[i]), spinor_work, spinor_aux, gauge_field, mzz[1][1]);
  }
  free ( spinor_work ); spinor_work = NULL;
  free ( spinor_aux  ); spinor_aux = NULL;

  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make odd phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W[iblock * block_size + i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        spinor_field_eq_gamma_ti_spinor_field( (double*)(W_gamma_phase[0]), gamma_id_list[ig], (double*)(W_phase[0]), block_size*Vhalf );
        g5_phi( (double*)(W_gamma_phase[0]), block_size * Vhalf );

        sprintf(aff_path, "%s/v_dag_gloc_w/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

/************************************************
 * NOTE: this casting may be dangerous
 *   (double _Complex**)V since V is double**
 *
 ************************************************/
        exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

        sprintf(aff_path, "%s/w_dag_gloc_w/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }


      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */


  /************************************************
   ************************************************
   **
   ** XW - XW ( even part)
   **
   ************************************************
   ************************************************/
  /* calculate X W */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
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

  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W[iblock*block_size + i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        spinor_field_eq_gamma_ti_spinor_field( (double*)(W_gamma_phase[0]), gamma_id_list[ig], (double*)(W_phase[0]), block_size*Vhalf );
        g5_phi( (double*)(W_gamma_phase[0]), block_size * Vhalf );

        sprintf(aff_path, "%s/xw_dag_gloc_xw/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }
      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */

  /************************************************
   ************************************************
   **
   ** XW - XV ( even part)
   **
   ************************************************
   ************************************************/

  /* calculate X W */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }
  exitstatus = init_2level_buffer ( (double***)(&W_aux), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }


  for( int iblock=0; iblock < block_num; iblock++ ) {

    for( int i=0; i<block_size; i++ ) {
      /*
       * Xbar V = Xbar V = -M_ee^-1[dn] M_eo V
       */
      memcpy( spinor_work, (double*)(V[iblock * block_size + i]), sizeof_eo_spinor_field );
      X_clover_eo ( (double*)(W_aux[i]), spinor_work, gauge_field, mzzinv[1][0] );
    }

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W_aux[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        spinor_field_eq_gamma_ti_spinor_field( (double*)(W_gamma_phase[0]), gamma_id_list[ig], (double*)(W_phase[0]), block_size*Vhalf );
        g5_phi( (double*)(W_gamma_phase[0]), block_size * Vhalf );

        sprintf(aff_path, "%s/xw_dag_gloc_xv/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }
      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */
  }  /* end of loop on blocks */
  free ( spinor_work ); spinor_work = NULL;
  fini_2level_buffer ( (double***)(&W_aux) );
#if 0
#endif  /* of if 0 */

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&W) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&W_ts) );
  fini_2level_buffer ( (double***)(&W_phase) );
  fini_2level_buffer ( (double***)(&W_gamma_phase) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_v_dag_gloc_w_blocked] time for contract_v_dag_gloc_w_blocked = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_gloc_w_blocked */
#endif  // of if 0

#if 0
/************************************************************
 * calculate gsp V^+ Gamma(p) Phi
 *
 *   V intended eigenvectors
 *   Phi intended stochastic propagators
 *   local vertex Gamma(p)
 *   eo - scalar product over even 0 / odd 1 sites
 *
 *   V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)
 *
 *   Wb is block_size x (12 VOL3half) (C) = (12 VOL3half) x block_size (F)
 *
 *   XbarV, W, XW and their blocked versions all the same formats
 *
 *   zgemm calculates
 *   V^H x [ Gamma(p) x Wb ) ] which is numV x block_size (F) = block_size x numV (C)
 *
 ************************************************************/
int contract_vdag_gloc_phi_blocked (double**V, double**Phi, int numV, int numPhi, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size ) {
 
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_with_halo = _GSI( (VOLUME+RAND)/2 ) * sizeof(double);
  /* const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double); */

  int exitstatus;
  double *spinor_work = NULL, *spinor_aux = NULL;
  double _Complex **phase_field = NULL;
  double _Complex **V_ts = NULL;
  double _Complex **W = NULL;
  double _Complex **Phi_phase = NULL, **Phi_gamma_phase = NULL, **Phi_ts = NULL,
         **Phi_aux1 = NULL, **Phi_aux2 = NULL;
  double _Complex ***contr = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL;
  char aff_path[200];

  int block_num = (int)(numPhi / block_size);
  if ( block_num * block_size != numPhi ) {
    fprintf(stderr, "[contract_vdag_gloc_phi_blocked] Error, numPhi must be divisible by block size %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;

  /**********************************
   * Fourier phase field
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /**********************************
   * auxilliary V-fields
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /**********************************
   * auxilliary Phi-fields
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&Phi_ts), block_size, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_gamma_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_aux1), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_aux2), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  /**********************************
   * fields for contractions
   **********************************/
  exitstatus = init_3level_buffer ( (double****)(&contr), T, block_size, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  mcontr_buffer = (double*)malloc(numV*block_size*2*sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }

  /************************************************
   ************************************************
   **
   ** V, W - Phi, Xi ( all four odd part)
   **
   ************************************************
   ************************************************/

  /* calculate W from V 
   * W = Cbar V
   * */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }
  for( int i = 0; i < numV; i++ ) {
    /* Cbar_oo = g5 ( Mbar_oo - M_oe Mbar_ee^-1 M_eo ) */
    C_clover_oo ( (double*)(W[i]), (double*)(V[i]), gauge_field, spinor_work, mzz[1][1], mzzinv[1][0]);
  }

  /* loop on blocks of fields */
  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* copy Phi */
    memcpy( Phi_aux1[0], Phi[iblock*block_size], block_size * sizeof_eo_spinor_field );
    /* calculate Xi  = Cbar Phi */
    for( int i = 0; i < block_size; i++ ) {
      /* Cbar = g5 ( Mbar_oo - M_oe Mbar_ee^-1 M_eo ) */
      C_clover_oo ( (double*)(Phi_aux2[i]), (double*)(Phi_aux1[i]), gauge_field, spinor_work, mzz[1][1], mzzinv[1][0]);
    }

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make odd phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++ ) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux1[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        /* calculate Gamma times spinor field timeslice 
         */
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        /* prepare the AFF key */
        sprintf(aff_path, "%s/v_dag_gloc_phi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

/************************************************
 * NOTE: dangerous type cast
 *   (double _Complex**)V from  double**V
 *
 ************************************************/
        exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

        /* prepare the AFF key */
        sprintf(aff_path, "%s/w_dag_gloc_phi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }
      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */

    for( int im=0; im<momentum_number; im++ ) {
  
      /* make odd phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++ ) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux2[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        /* calculate Gamma times spinor field timeslice 
         */
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        /* prepare the AFF key */
        sprintf(aff_path, "%s/v_dag_gloc_xi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

/************************************************
 * NOTE: dangerous type cast
 *   (double _Complex**)V from  double**V
 *
 ************************************************/
        exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

        /* prepare the AFF key */
        sprintf(aff_path, "%s/w_dag_gloc_xi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }
      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */

  }  /* end of loops on blocks */
  free( spinor_work );

  /************************************************
   ************************************************
   **
   ** Xbar V - Xbar Phi, X Xi (even part)
   **
   ************************************************
   ************************************************/

  /* calculate Xbar V */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  spinor_aux  = (double*)malloc( sizeof_eo_spinor_field );
  if ( spinor_work == NULL || spinor_aux == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W <- Xbar V = -M_ee^-1[dn] M_eo V
     */
    memcpy( spinor_work, (double*)(V[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[1][0] );
  }

  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate Xbar Phi 
     * calculate X Xi 
     */
    for( int i = 0; i < block_size; i++ ) {
      memcpy( spinor_work, Phi[iblock * block_size + i], sizeof_eo_spinor_field );
      X_clover_eo ( (double*)(Phi_aux1[i]), spinor_work, gauge_field, mzzinv[1][0] );

      /* calculate Xi  = Cbar from Phi, Xbar Phi */
      memcpy( Phi_aux2[i], Phi[iblock * block_size + i], sizeof_eo_spinor_field );
      memcpy( spinor_work, Phi_aux1[i], sizeof_eo_spinor_field );
      C_clover_from_Xeo ( (double*)(Phi_aux2[i]), spinor_work, spinor_aux, gauge_field, mzz[1][1]);

      /* calculate X Xi */
      memcpy(spinor_work, Phi_aux2[i], sizeof_eo_spinor_field );
      X_clover_eo ( (double*)(Phi_aux2[i]), spinor_work, gauge_field, mzzinv[0][0] );
    }

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux1[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {
 
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        sprintf(aff_path, "%s/xv_dag_gloc_xphi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux2[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {
 
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        sprintf(aff_path, "%s/xv_dag_gloc_xxi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  
  }  /* end of loop on blocks */
  free ( spinor_work );
  free ( spinor_aux  );

  /************************************************
   ************************************************
   **
   ** X W - Xbar Phi, X Xi (even part)
   **
   ************************************************
   ************************************************/

  /* calculate Xbar V */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  spinor_aux  = (double*)malloc( sizeof_eo_spinor_field );
  if ( spinor_work == NULL || spinor_aux == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /* aux <- Cbar V = W */
    C_clover_oo (spinor_aux, (double*)(V[i]), gauge_field, spinor_work, mzz[1][1], mzzinv[1][0] );

    /* work <- aux */
    memcpy( spinor_work, spinor_aux, sizeof_eo_spinor_field );
    /* W <- X aux */
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[0][0] );
  }

  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate Xbar Phi */
    for( int i = 0; i < block_size; i++ ) {
      /* work <- phi */
      memcpy( spinor_work, Phi[iblock * block_size + i], sizeof_eo_spinor_field );
      /* phi <- Xbar work */
      X_clover_eo ( (double*)(Phi_aux1[i]), spinor_work, gauge_field, mzzinv[1][0] );

      /* phi2 <- phi */
      memcpy( Phi_aux2[i], Phi[iblock * block_size + i], sizeof_eo_spinor_field );
      /* work <- Xbar phi */
      memcpy( spinor_work, Phi_aux1[i], sizeof_eo_spinor_field );
      /* phi2 <- Cbar( phi2, work) using aux */
      C_clover_from_Xeo ( (double*)(Phi_aux2[i]), spinor_work, spinor_aux, gauge_field, mzz[1][1]);

      /* work <- phi2 */
      memcpy(spinor_work, Phi_aux2[i], sizeof_eo_spinor_field );
      /* phi2 <- X work */
      X_clover_eo ( (double*)(Phi_aux2[i]), spinor_work, gauge_field, mzzinv[0][0] );
    }

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux1[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {
 
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        sprintf(aff_path, "%s/xw_dag_gloc_xphi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux2[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {
 
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        sprintf(aff_path, "%s/xw_dag_gloc_xxi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  
  }  /* end of loop on blocks */
  free ( spinor_work );
  free ( spinor_aux  );

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&W) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&Phi_ts) );
  fini_2level_buffer ( (double***)(&Phi_phase) );
  fini_2level_buffer ( (double***)(&Phi_gamma_phase) );
  fini_2level_buffer ( (double***)(&Phi_aux1) );
  fini_2level_buffer ( (double***)(&Phi_aux2) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_v_dag_gloc_phi_blocked] time for contract_v_dag_gloc_phi_blocked = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_gloc_phi */
#endif

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
int contract_vdag_cvc_w_blocked (
    double** const V, int const numV,
    int const momentum_number, int const (*momentum_list)[3],
    struct AffWriter_s * const affw, char * const tag, int const io_proc,
    double * const gauge_field, double ** const mzz[2], double ** const mzzinv[2],
    int const block_size 
    ) {
 
  unsigned int const Vhalf = VOLUME / 2;
  unsigned int const VOL3half = ( LX * LY * LZ ) / 2;
  size_t const sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  size_t const sizeof_eo_spinor_field_with_halo = _GSI( (VOLUME+RAND)/2 ) * sizeof(double);
  /* size_t const sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double); */

  int exitstatus;
  double **eo_spinor_work = NULL;
  double _Complex **phase_field = NULL;

  /* auxilliary V fields */
  double _Complex **V_ts = NULL;

  /* auxilliary W fields */
  double _Complex **W = NULL;
  double _Complex **W_ts = NULL, **W_vertex = NULL,**W_phase = NULL, **W_aux = NULL;

  /* fields for contractions */
  double _Complex ***contr = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL;
  char aff_path[200];

  int block_num = (int)( numV / block_size);
  if ( block_num * block_size != numV ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error, numV must be divisible by block size %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( (double***)(&eo_spinor_work), 1, _GSI( (VOLUME+RAND)/2) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /* Fourier phases */
  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /* auxilliary V fields */
  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /* auxilliary W fields  */
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_aux), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_ts), block_size, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_vertex), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  /* fields for contractions */
  exitstatus = init_3level_buffer ( (double****)(&contr), T, block_size, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  /* buffer for xchange */
  mcontr_buffer = (double*)malloc(numV * block_size * 2 * sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }

  /************************************************
   ************************************************
   **
   ** calculate W from V
   **
   ************************************************
   ************************************************/
  for( int i=0; i < numV; i++ ) {
    /* W = Cbar_oo V */
    C_clover_oo ( (double*)(W[i]), (double*)(V[i]), gauge_field, eo_spinor_work[0], mzz[1][1], mzzinv[1][0]);
  }

  /************************************************
   ************************************************
   **
   ** V - Xbar V
   ** V - X W
   ** W - Xbar V
   ** W - X W
   **
   ************************************************
   ************************************************/

  for ( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate a block of Xbar V */
    for( int i=0; i < block_size; i++ ) {
      /* spinor_work = V */
      memcpy( eo_spinor_work[0], V[iblock*block_size + i], sizeof_eo_spinor_field );
      /* W_aux = Xbar V */
      X_clover_eo ( (double*)(W_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[1][0]);
    }

    /************************************************
     *
     * V - Xbar V and
     * W - Xbar V
     *
     ************************************************/

    /************************************************
     * loop on directions mu
     ************************************************/
    for( int mu=0; mu<4; mu++ ) {

      /************************************************
       * loop on fwd / bwd
       ************************************************/
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /************************************************
         * apply the cvc vertex in direction mu,
         * fbwd to current block of fields
         ************************************************/
        for( int i=0; i < block_size; i++) {
          /* spinor_work = W_aux = Xbar V */
          memcpy( eo_spinor_work[0], (double*)(W_aux[i]), sizeof_eo_spinor_field );
          /* W_vertex = CVC(mu, fbwd) Xbar V
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(W_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /************************************************
         * loop on momenta
         ************************************************/
        for( int im=0; im<momentum_number; im++ ) {
  
          /************************************************
           * make odd phase field
           ************************************************/
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /************************************************
           * multiply by momentum phase
           ************************************************/
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(W_phase[0]), block_size*Vhalf);

          /************************************************
           * (1) V - Xbar V
           ************************************************/
          sprintf(aff_path, "%s/v_dag_cvc_xv/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

/******************************************
 * NOTE: DANGEROUS TYPECAST
 ******************************************/
          exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, W_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

          /************************************************
           * (2) W - Xbar V
           ************************************************/
          sprintf(aff_path, "%s/w_dag_cvc_xv/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, W_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 
      }  /* end of loop on fwd, bwd */ 
    }  /* end of loop on shift directions */

    /************************************************
     *
     * V - X W and
     * W - X W
     *
     ************************************************/

    /* calculate a block of X W */
    for( int i=0; i < block_size; i++ ) {
      /* spinor_work = W */
      memcpy( eo_spinor_work[0], W[iblock*block_size + i], sizeof_eo_spinor_field );
      /* W_aux = X W */
      X_clover_eo ( (double*)(W_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[0][0]);
    }

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = W_aux = X W */
          memcpy( eo_spinor_work[0], (double*)(W_aux[i]), sizeof_eo_spinor_field );
          /* W_vertex = CVC(mu, fbwd) X W
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(W_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(W_phase[0]), block_size*Vhalf);

          /************************************************
           * (3) V - X W
           ************************************************/
          sprintf(aff_path, "%s/v_dag_cvc_xw/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

/******************************************
 * NOTE: DANGEROUS TYPECAST
 ******************************************/
          exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, W_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

          /************************************************
           * (4) W - X W
           ************************************************/
          sprintf(aff_path, "%s/w_dag_cvc_xw/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, W_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 
      }  /* end of loop on fwd, bwd */ 
    }  /* end of loop on shift directions */

  }  /* end of loop on blocks */

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&eo_spinor_work) );
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&W) );
  fini_2level_buffer ( (double***)(&W_ts) );
  fini_2level_buffer ( (double***)(&W_aux) );
  fini_2level_buffer ( (double***)(&W_vertex) );
  fini_2level_buffer ( (double***)(&W_phase) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_vdag_cvc_w_blocked] time for contract_vdag_cvc_w_blocked = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_cvc_w_blocked */


/*******************************************************************************************
 * calculate V^+ cvc-vertex Phi
 *
          V^+ point-split vertex Phi
          eo - scalar product over even 0 / odd 1 sites
 *
 *******************************************************************************************/
int contract_vdag_cvc_phi_blocked (double**V, double**Phi, int numV, int numPhi, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size ) {
 
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  /* const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double); */

  int exitstatus;
  double **eo_spinor_work = NULL;
  double _Complex **phase_field = NULL;

  /* auxilliary V fields */
  double _Complex **V_ts = NULL;
  double _Complex **W = NULL;

  /* auxilliary Phi fields */
  double _Complex **Phi_ts = NULL, **Phi_vertex = NULL,**Phi_phase = NULL, **Phi_aux = NULL;

  /* fields for contractions */
  double _Complex ***contr = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL;
  char aff_path[200];

  int block_num = (int)( numPhi / block_size);
  if ( block_num * block_size != numPhi ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error, numPhi must be divisible by block size %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( (double***)(&eo_spinor_work), 2, _GSI( (VOLUME+RAND)/2) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /* Fourier phases */
  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /* auxilliary V fields */
  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /* auxilliary W fields  */
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  /* auxilliary fields for Phi */
  exitstatus = init_2level_buffer ( (double***)(&Phi_aux), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_ts), block_size, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_vertex), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  /* fields for contractions */
  exitstatus = init_3level_buffer ( (double****)(&contr), T, block_size, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  /* buffer for xchange */
  mcontr_buffer = (double*)malloc(numV * block_size * 2 * sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }

  /************************************************
   ************************************************
   **
   ** calculate Xbarm V from V
   **
   **   all vectors
   **
   ************************************************
   ************************************************/
  for( int i=0; i < numV; i++ ) {
    /* spinor_work = V */
    memcpy( eo_spinor_work[0], (double*)(V[i]), sizeof_eo_spinor_field );
    /* W = Xbar V */
    X_clover_eo ( (double*)(W[i]), eo_spinor_work[0], gauge_field, mzzinv[1][0]);
  }


  /************************************************
   ************************************************
   **
   ** Xbar V - Phi
   ** V      - Xbar Phi
   ** Xbar V - Sigma
   ** V      - X Sigma
   **
   ************************************************
   ************************************************/

  for ( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate a block of Xbar Phi */
    for( int i=0; i < block_size; i++ ) {
      /* spinor_work = Phi */
      memcpy( eo_spinor_work[0], (double*)(Phi[iblock*block_size + i]), sizeof_eo_spinor_field );
      /* Phi_aux = Xbar spinor_work = Xbar Phi */
      X_clover_eo ( (double*)(Phi_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[1][0]);
    }

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /************************************************
         * (1) Xbar V - Phi
         ************************************************/

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi */
          memcpy( eo_spinor_work[0], (double*)(Phi[iblock*block_size + i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Phi 
           *   EVEN target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);


          sprintf(aff_path, "%s/xv_dag_cvc_phi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 

        /************************************************
         * (2) V - Xbar Phi
         ************************************************/

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = Xbar Phi */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Xbar Phi
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);

          sprintf(aff_path, "%s/v_dag_cvc_xphi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

/************************************************
 * NOTE: DANGEROUS TYPECAST
 ************************************************/
          exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 

      }  /* end of loop on fwd, bwd */ 
    }  /* end of loop on shift directions */

    /************************************************
     * (3) Xbar V - Xi
     ************************************************/

    /* calculate block of Sigma from Xbar Phi and Phi */
    for( int i = 0; i < block_size; i++  ) {
      /* eo_spinor = Xbar Phi */
      memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
      /* Phi_aux = Phi */
      memcpy( (double*)(Phi_aux[i]), (double*)(Phi[iblock*block_size+i]) , sizeof_eo_spinor_field );
      /* Phi_aux = Cbar_oo ( Phi_aux, Phi ) = Sigma */
      C_clover_from_Xeo ( (double*)(Phi_aux[i]), eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1]);
    }

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = Sigma */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Sigma
           *   EVEN target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);


          sprintf(aff_path, "%s/xv_dag_cvc_xi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 

      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */
#if 0
#endif  /* of if 0*/


    /************************************************
     * (4) V - X Xi
     ************************************************/

    /* calculate block of X Sigma from Sigma */
    for( int i = 0; i < block_size; i++  ) {
      /* eo_spinor = Sigma */
      memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
      /* Phi_aux = X Sigma */
      X_clover_eo ( (double*)(Phi_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[0][0]);
    }


    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = X Sigma */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) X Sigma
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);


          sprintf(aff_path, "%s/v_dag_cvc_xxi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

/************************************************
 * NOTE: DANGEROUS TYPECAST
 ************************************************/
          exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 

      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */
#if 0
#endif  /* of if 0 */
  }  /* end of loop on blocks */



  /************************************************
   * W - Xbar Phi and
   * W - X Xi
   ************************************************/

  /* calculate W from V, Xbar V */
  for( int i = 0; i < numV; i++ ) {
    /* spinor_work = W = Xbar V */
    memcpy( eo_spinor_work[0], (double*)(W[i]), sizeof_eo_spinor_field );
    /* W = V */
    memcpy( (double*)(W[i]), (double*)(V[i]), sizeof_eo_spinor_field );
    /* W = Cbar_oo ( W, spinor_work )*/
    C_clover_from_Xeo ( (double*)(W[i]), eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1]);
  }

  /* loop on blocks of fields */
  for ( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate a block of Xbar Phi */
    for( int i=0; i < block_size; i++ ) {
      /* spinor_work = Phi */
      memcpy( eo_spinor_work[0], (double*)(Phi[iblock*block_size + i]), sizeof_eo_spinor_field );
      /* Phi_aux = Xbar Phi */
      X_clover_eo ( (double*)(Phi_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[1][0]);
    }

    /************************************************
     * (5) W - Xbar Phi
     ************************************************/

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = Xbar Phi */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Xbar Phi 
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);

          sprintf(aff_path, "%s/w_dag_cvc_xphi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        }  /* end of loop on momenta */
      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */

    /************************************************
     * (6) W - X Xi
     ************************************************/

    /* calculate a block of X Sigma */
    for( int i = 0; i < block_size; i++  ) {
      /* spinor_work = Phi_aux = Xbar Phi */
      memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
      /* Phi_aux = Phi */
      memcpy( (double*)(Phi_aux[i]), (double*)(Phi[iblock*block_size+i]) , sizeof_eo_spinor_field );
      /* Phi_aux = C_oo ( Phi_aux, Phi ) = Sigma */
      C_clover_from_Xeo ( (double*)(Phi_aux[i]), eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1]);
      /* spinor_work = Phi_aux = Sigma */
      memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
      /* Phi_aux = X spinor_work  = */
      X_clover_eo ( (double*)(Phi_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[0][0]);
    }

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = X Sigma */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) X Sigma
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {

          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);

          sprintf(aff_path, "%s/w_dag_cvc_xxi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }


        }  /* end of loop on momenta */
      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */

  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */


  /************************************************
   * X W - Phi and
   * X W - Xi
   ************************************************/

  /* calculate W <- X W from W */
  for( int i=0; i < numV; i++ ) {
    /* spinor_work = W */
    memcpy( eo_spinor_work[0], (double*)(W[i]), sizeof_eo_spinor_field );
    /* W = X spinor_work = X W */
    X_clover_eo ( (double*)(W[i]), eo_spinor_work[0], gauge_field, mzzinv[0][0]);
  }

  /* loop on blocks of fields */
  for ( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate a block of Sigma */
    for( int i = 0; i < block_size; i++  ) {
      /* Phi_aux = Cbar_oo Phi */
      C_clover_oo ( (double*)(Phi_aux[i]), (double*)(Phi[iblock * block_size + i]), gauge_field, eo_spinor_work[0], mzz[1][1], mzzinv[1][0] );
    }

    /************************************************
     * (7) X W - Phi
     ************************************************/

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi */
          memcpy( eo_spinor_work[0], (double*)(Phi[iblock*block_size+i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Phi
           *   EVEN target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {

          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);

          sprintf(aff_path, "%s/xw_dag_cvc_phi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        }  /* end of loop on momenta */
      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */

    /************************************************
     * (8) X W - Sigma
     ************************************************/

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = Sigma */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Sigma
           *   EVEN target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {

          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);

          sprintf(aff_path, "%s/xw_dag_cvc_xi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        }  /* end of loop on momenta */
      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&eo_spinor_work) );
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&W) );
  fini_2level_buffer ( (double***)(&Phi_ts) );
  fini_2level_buffer ( (double***)(&Phi_aux) );
  fini_2level_buffer ( (double***)(&Phi_vertex) );
  fini_2level_buffer ( (double***)(&Phi_phase) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_vdag_cvc_phi_blocked] time for contract_v_dag_cvc_phi_blocked = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_cvc_phi_blocked */


}  /* end of namespace cvc */
