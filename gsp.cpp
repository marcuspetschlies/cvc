/***************************************************
 * gsp.cpp
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

#if 0

/***********************************************************************************************
 ***********************************************************************************************
 **
 ** gsp_calculate_v_dag_gamma_p_w
 **
 ** calculate V^+ Gamma(p) W
 **
 ***********************************************************************************************
 ***********************************************************************************************/

int gsp_calculate_v_dag_gamma_p_w(double**V, double**W, int num, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, char*tag, int symmetric) {
  
  int status, iproc;
  int x0, ievecs, kevecs, k;
  int isource_momentum, isource_gamma_id, momentum[3];

  double *phase_e=NULL, *phase_o = NULL;
  double *****gsp = NULL;
  double *****gsp_buffer = NULL;
  double *buffer=NULL, *buffer2 = NULL;
  size_t items;
  double ratime, retime, momentum_ratime, momentum_retime, gamma_ratime, gamma_retime;
  unsigned int Vhalf = VOLUME / 2;
  char filename[200];

#ifdef HAVE_MPI
  int io_proc = 0;
  MPI_Status mstatus;
  int mcoords[4], mrank;
#endif

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex *aff_buffer = NULL;
  char aff_buffer_path[200];
/*  uint32_t aff_buffer_size; */
#else
  FILE *ofs = NULL;
#endif

  ratime = _GET_TIME;

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
#endif

  /***********************************************
   * even/odd phase field for Fourier phase
   ***********************************************/
  phase_e = (double*)malloc(Vhalf*2*sizeof(double));
  phase_o = (double*)malloc(Vhalf*2*sizeof(double));
  if(phase_e == NULL || phase_o == NULL) {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
    return(1);
  }

  buffer = (double*)malloc(2*Vhalf*sizeof(double));
  if(buffer == NULL)  {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
    return(2);
  }

  buffer2 = (double*)malloc(2*T*sizeof(double));
  if(buffer2 == NULL)  {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
    return(3);
  }

  /***********************************************
   * allocate gsp
   ***********************************************/
  status = gsp_init (&gsp, 1, 1, T, num);
  if(gsp == NULL) {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from gsp_init, status was %d\n", status);
    return(18);
  }


#ifdef HAVE_LHPC_AFF
  /***********************************************
   * open aff output file and allocate gsp buffer
   ***********************************************/
#ifdef HAVE_MPI
  if(io_proc == 2) {
#endif

    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] using aff version %s\n", aff_status_str);

    sprintf(filename, "%s.aff", tag);
    fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] writing gsp data from file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from aff_writer, status was %s\n", aff_status_str);
      return(4);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error, aff writer is not initialized\n");
      return(5);
    }

    aff_buffer = (double _Complex*)malloc(num*num*sizeof(double _Complex));
    if(aff_buffer == NULL) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
      return(6);
    }

#ifdef HAVE_MPI
    status = gsp_init (&gsp_buffer, 1, 1, T_global, num);
    if(gsp_buffer == NULL) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from gsp_init, status was %d\n", status);
      return(7);
    }
#else
    gsp_buffer = gsp;
#endif  /* of ifdef HAVE_MPI */

#ifdef HAVE_MPI
  }  /* end of if io_proc == 2 */
#endif

#endif  /* of ifdef HAVE_LHPC_AFF */

  /***********************************************
   * loop on momenta
   ***********************************************/
  for(isource_momentum=0; isource_momentum < momentum_number; isource_momentum++) {

    momentum[0] = momentum_list[isource_momentum][0];
    momentum[1] = momentum_list[isource_momentum][1];
    momentum[2] = momentum_list[isource_momentum][2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] using source momentum = (%d, %d, %d)\n", momentum[0], momentum[1], momentum[2]);
    }
    momentum_ratime = _GET_TIME;

    /* make phase field in eo ordering */
    gsp_make_eo_phase_field (phase_e, phase_o, momentum);

  /***********************************************
   * loop on gamma id's
   ***********************************************/
    for(isource_gamma_id=0; isource_gamma_id < gamma_id_number; isource_gamma_id++) {

      if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] using source gamma id %d\n", gamma_id_list[isource_gamma_id]);
      gamma_ratime = _GET_TIME;

      if(symmetric) {
        /* loop on eigenvectors */
        for(ievecs = 0; ievecs<num; ievecs++) {
          for(kevecs = ievecs; kevecs<num; kevecs++) {
  
            ratime = _GET_TIME;
            eo_spinor_dag_gamma_spinor((complex*)buffer, V[ievecs], gamma_id_list[isource_gamma_id], W[kevecs]);
            retime = _GET_TIME;
            /* if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for eo_spinor_dag_gamma_spinor = %e seconds\n", retime - ratime); */
  
            ratime = _GET_TIME;
            eo_gsp_momentum_projection ((complex*)buffer2, (complex*)buffer, (complex*)phase_o, 1);
            retime = _GET_TIME;
            /* if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for eo_gsp_momentum_projection = %e seconds\n", retime - ratime); */
  
            for(x0=0; x0<T; x0++) {
              memcpy(gsp[0][0][x0][ievecs] + 2*kevecs, buffer2+2*x0, 2*sizeof(double));
            }
  
          }
        }
  
        /* gsp[k,i] = sigma_Gamma gsp[i,k]^+ */
        for(x0=0; x0<T; x0++) {
          for(ievecs = 0; ievecs<num-1; ievecs++) {
            for(kevecs = ievecs+1; kevecs<num; kevecs++) {
              gsp[0][0][x0][kevecs][2*ievecs  ] =  gamma_adjoint_sign[gamma_id_list[isource_gamma_id]] * gsp[0][0][x0][ievecs][2*kevecs  ];
              gsp[0][0][x0][kevecs][2*ievecs+1] = -gamma_adjoint_sign[gamma_id_list[isource_gamma_id]] * gsp[0][0][x0][ievecs][2*kevecs+1];
            }
          }
        }

      } else {
        /* loop on eigenvectors */
        for(ievecs = 0; ievecs<num; ievecs++) {
          for(kevecs = 0; kevecs<num; kevecs++) {
  
            ratime = _GET_TIME;
            eo_spinor_dag_gamma_spinor((complex*)buffer, V[ievecs], gamma_id_list[isource_gamma_id], W[kevecs]);
            retime = _GET_TIME;
            /* if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for eo_spinor_dag_gamma_spinor = %e seconds\n", retime - ratime); */
  
            ratime = _GET_TIME;
            eo_gsp_momentum_projection ((complex*)buffer2, (complex*)buffer, (complex*)phase_o, 1);
            retime = _GET_TIME;
            /* if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for eo_gsp_momentum_projection = %e seconds\n", retime - ratime); */
  
            for(x0=0; x0<T; x0++) {
              memcpy(gsp[0][0][x0][ievecs] + 2*kevecs, buffer2+2*x0, 2*sizeof(double));
            }
  
          }
        }
      }  /* end of if symmetric */

      /***********************************************
       * write gsp to disk
       ***********************************************/
      for(iproc=0; iproc < g_nproc_t; iproc++) {
#ifdef HAVE_MPI
        ratime = _GET_TIME;
        if(iproc > 0) {
          /***********************************************
           * gather at root
           ***********************************************/
          k = 2*T*num*num; /* number of items to be sent and received */
          if(io_proc == 2) {
            mcoords[0] = iproc; mcoords[1] = 0; mcoords[2] = 0; mcoords[3] = 0;
            MPI_Cart_rank(g_cart_grid, mcoords, &mrank);
            /* fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] proc%.4d receiving from proc%.4d\n", g_cart_id, mrank); */
            /* receive gsp with tag iproc */
            status = MPI_Recv(gsp_buffer[0][0][0][0], k, MPI_DOUBLE, mrank, iproc, g_cart_grid, &mstatus);
            if(status != MPI_SUCCESS ) {
              fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] proc%.4d Error from MPI_Recv, status was %d\n", g_cart_id, status);
              return(19);
            }
          } else if (g_proc_coords[0] == iproc && io_proc == 1) {
            mcoords[0] = 0; mcoords[1] = 0; mcoords[2] = 0; mcoords[3] = 0;
            MPI_Cart_rank(g_cart_grid, mcoords, &mrank);
            /* fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] proc%.4d sending to proc%.4d\n", g_cart_id, mrank); */
            /* send correlator with tag 2*iproc */
            status = MPI_Send(gsp[0][0][0][0], k, MPI_DOUBLE, mrank, iproc, g_cart_grid);
            if(status != MPI_SUCCESS ) {
              fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] proc%.4d Error from MPI_Recv, status was %d\n", g_cart_id, status);
              return(19);
            }
          }
        } else {
          if(io_proc == 2) {
            k = 2*T*num*num; /* number of items to be copied */
            memcpy(gsp_buffer[0][0][0][0], gsp[0][0][0][0], k*sizeof(double));
          }
        }  /* end of if iproc > 0 */
        retime = _GET_TIME;
        /* if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for gsp exchange = %e seconds\n", retime     - ratime); */
#endif  /* of ifdef HAVE_MPI */

        /***********************************************
         * I/O process write to file
         ***********************************************/
#ifdef HAVE_MPI
        if(io_proc == 2) {
#endif
          ratime = _GET_TIME;
#ifdef HAVE_LHPC_AFF
          for(x0=0; x0<T; x0++) {
            sprintf(aff_buffer_path, "/%s/px%.2dpy%.2dpz%.2d/g%.2d/t%.2d", tag, momentum[0], momentum[1], momentum[2], gamma_id_list[isource_gamma_id], x0+iproc*T);
            /* fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] current aff path = %s\n", aff_buffer_path); */

            affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
            items = num*num;
            memcpy(aff_buffer, gsp_buffer[0][0][x0][0], 2*items*sizeof(double));
            status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)items); 
            if(status != 0) {
              fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from aff_node_put_double, status was %d\n", status);
              return(8);
            }
          }  /* end of loop on x0 */
#else
          sprintf(filename, "%s.px%.2dpy%.2dpz%.2d.g%.2d", tag, momentum[0], momentum[1], momentum[2], gamma_id_list[isource_gamma_id]);
          ofs = fopen(filename, "w");
          if(ofs == NULL) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error, could not open file %s for writing\n", filename);
            return(9);
          }
          items = 2 * (size_t)T * num*num;
          if( fwrite(gsp_buffer[0][0][0][0], sizeof(double), items, ofs) != items ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error, could not write proper amount of data to file %s\n", filename);
            return(10);
          }
          fclose(ofs);
#endif
          retime = _GET_TIME;
          /* fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for writing = %e seconds\n", retime     - ratime); */
#ifdef HAVE_MPI
        }  /* end of if io_proc == 2 */
#endif
      }  /* end of loop on iproc */

#ifdef HAVE_MPI
      if(io_proc == 2) {
        gsp_reset (&gsp_buffer, 1, 1, T, num);
      }
#endif

     gsp_reset (&gsp, 1, 1, T, num);
#ifdef HAVE_MPI
     MPI_Barrier(g_cart_grid);
#endif

     gamma_retime = _GET_TIME;
     if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for gamma id %d = %e seconds\n", gamma_id_list[isource_gamma_id], gamma_retime - gamma_ratime);

   }  /* end of loop on source gamma id */

   momentum_retime = _GET_TIME;
   if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for momentum (%d, %d, %d) = %e seconds\n",
       momentum[0], momentum[1], momentum[2], momentum_retime - momentum_ratime);


  }   /* end of loop on source momenta */

#ifdef HAVE_LHPC_AFF

#ifdef HAVE_MPI
  if(io_proc == 2) {
#endif
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from aff_writer_close, status was %s\n", aff_status_str);
      return(11);
    }
    if(aff_buffer != NULL) free(aff_buffer);
#ifdef HAVE_MPI
  }  /* end of if io_proc == 2 */
#endif

#endif  /* of ifdef HAVE_LHPC_AFF */

  if(phase_e != NULL) free(phase_e);
  if(phase_o != NULL) free(phase_o);
  if(buffer  != NULL) free(buffer);
  if(buffer2 != NULL) free(buffer2);
  gsp_fini(&gsp);

#ifdef HAVE_LHPC_AFF

#ifdef HAVE_MPI
  if(io_proc == 2) {
#endif
    gsp_fini(&gsp_buffer);
#ifdef HAVE_MPI
  }
#endif

#endif  /* of ifdef HAVE_LHPC_AFF */

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for = %e seconds\n", retime-ratime);

  return(0);

}  /* end of gsp_calculate_v_dag_gamma_p_w */

#endif

/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 * calculate gsp using t-blocks
 *
          subroutine zgemm  (   character   TRANSA,
            character   TRANSB,
            integer   M,
            integer   N,
            integer   K,
            complex*16    ALPHA,
            complex*16, dimension(lda,*)    A,
            integer   LDA,
            complex*16, dimension(ldb,*)    B,
            integer   LDB,
            complex*16    BETA,
            complex*16, dimension(ldc,*)    C,
            integer   LDC 
          )       
 *
 ***********************************************************************************************/
int gsp_calculate_v_dag_gamma_p_w_block(
    double**V, int numV,
    int momentum_number, int (*momentum_list)[3],
    int gamma_id_number, int*gamma_id_list,
    char*prefix, char*tag, int io_proc,
    double *gauge_field, double **mzz[2], double **mzzinv[2] 
    ) {
  
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = (LX*LY*LZ)/2;
  const size_t sizeof_eo_spinor_field           = _GSI(Vhalf)    * sizeof(double);
  const size_t sizeof_eo_spinor_field_timeslice = _GSI(VOL3half) * sizeof(double);
  const size_t write_count = numV * numV;
  const size_t write_bytes = 2*numV*numV*sizeof(double);


  int exitstatus;
  char filename[200];

  double ratime, retime, zgemm_ratime, zgemm_retime, aff_retime, aff_ratime, total_ratime, total_retime;

  /***********************************************
   *variables for blas interface
   ***********************************************/
  double _Complex Z_1 = 1.;
  double _Complex Z_0 = 0.;

  char CHAR_N = 'N', CHAR_C = 'C';
  int INT_M = numV, INT_N = numV, INT_K = 12*VOL3half;

#ifdef HAVE_LHPC_AFF
  AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_key[200], *aff_status_str;

  /***********************************************
   * writer for aff output file
   ***********************************************/
  if(io_proc >= 1) {
    sprintf(filename, "%s.%.4d.t%.2d.aff", prefix, Nconf, g_proc_coords[0]  );
    if ( io_proc == 2 ) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      return(15);
    }
  
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }

#endif

  total_ratime = _GET_TIME;

  /***********************************************/
  /***********************************************/

  /***********************************************
   * calculate W
   ***********************************************/
  double **W = NULL, **eo_spinor_work = NULL;

  exitstatus = init_2level_buffer ( &W, numV, _GSI(Vhalf) );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( &eo_spinor_work, 2, _GSI((VOLUME+RAND)/2) );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;
  for ( int i = 0; i < numV; i++ ) {
    memcpy ( eo_spinor_work[0], V[i], sizeof_eo_spinor_field );
    C_clover_oo ( W[i], eo_spinor_work[0], gauge_field, eo_spinor_work[1], mzz[1][1], mzzinv[1][0] );
  }
  retime = _GET_TIME;
  if ( io_proc == 2 ) fprintf ( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for W = %e seconds\n", retime-ratime );

  /***********************************************/
  /***********************************************/

  /***********************************************
   * auxilliary fields
   ***********************************************/
  double **Vts = NULL, **Xtsxp = NULL, **Xtsxpxg = NULL, **Wts = NULL;
  exitstatus = init_2level_buffer ( &Vts,     numV, _GSI(VOL3half) ) || 
               init_2level_buffer ( &Wts,     numV, _GSI(VOL3half) ) ||
               init_2level_buffer ( &Xtsxp,   numV, _GSI(VOL3half) ) ||
               init_2level_buffer ( &Xtsxpxg, numV, _GSI(VOL3half) );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /***********************************************/
  /***********************************************/

  /***********************************************
   * phases for momentum projection
   ***********************************************/
  double _Complex **phase = NULL;
  exitstatus = init_2level_zbuffer ( &phase, momentum_number, VOL3half );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /***********************************************/
  /***********************************************/

  /***********************************************
   * loop on timeslices
   ***********************************************/
  for ( int it = 0; it < T; it++ ) {

    /***********************************************
     * phases for momentum projection
     ***********************************************/
    make_eo_phase_field_timeslice ( phase, momentum_number, momentum_list, it, 1 );

    ratime = _GET_TIME;
    /***********************************************
     * copy timeslices of fields
     ***********************************************/
    size_t offset = it * _GSI(VOL3half);
    for ( int k = 0; k < numV; k++ ) {
      memcpy ( Vts[k], V[k] + offset, sizeof_eo_spinor_field_timeslice );
    }
    for ( int k = 0; k < numV; k++ ) {
      memcpy ( Wts[k], W[k] + offset, sizeof_eo_spinor_field_timeslice );
    }
    retime = _GET_TIME;
    if ( io_proc == 2 ) fprintf ( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for timeslice fields = %e seconds\n", retime-ratime );



    /***********************************************
     * loop on momenta
     ***********************************************/
    for ( int imom = 0; imom < momentum_number; imom++ ) {

      /***********************************************
       * multiply with momentum phase 
       ***********************************************/
      for ( int k = 0; k < numV; k++ ) {
        spinor_field_eq_spinor_field_ti_complex_field ( Xtsxp[k], Vts[k], (double*)(phase[imom]), VOL3half );
      }

      /***********************************************
       * loop on gamma matrices
       ***********************************************/
      for ( int igam = 0; igam < gamma_id_number; igam++ ) {

#ifndef HAVE_LHPC_AFF
        FILE *ofs = NULL;
        if ( io_proc >= 1 ) {
          sprintf( filename, "%s.t%.2d.px%.2dpy%.2dpz%.2d.g%.2d.dat", prefix, it+g_proc_coords[0]*T,  momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], gamma_id_list[igam] ); 
          ofs = fopen ( filename, "wb" );
          if( ofs == NULL ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from open for filename %s %s %d\n", filename, __FILE__, __LINE__);
            return(1);
          }
        }
#endif
        /***********************************************
         * multiply with gamma matrix
         ***********************************************/
        spinor_field_eq_gamma_ti_spinor_field( Xtsxpxg[0], gamma_id_list[igam], Xtsxp[0], numV*VOL3half );
        g5_phi ( Xtsxpxg[0], numV*VOL3half );

        /***********************************************
         * reduce in position space
         ***********************************************/
        double _Complex **vv = NULL;

        exitstatus = init_2level_zbuffer ( &vv, numV, numV );

        /***********************************************
         * V^+ Gp V
         ***********************************************/
        F_GLOBAL(zgemm, ZGEMM) ( &CHAR_C, &CHAR_N, &INT_M, &INT_N, &INT_K, &Z_1, (double _Complex*)(Vts[0]), &INT_K, (double _Complex*)(Xtsxpxg[0]), &INT_K, &Z_0, vv[0], &INT_M, 1, 1);
        
#ifdef HAVE_MPI
        ratime = _GET_TIME;
        /***********************************************
         * reduce within global timeslice
         ***********************************************/
        double *vvx = NULL;
        exitstatus = init_1level_buffer ( &vvx, 2*numV*numV );
        if(exitstatus != 0) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(5);
        }

        memcpy( vvx, vv[0], numV*numV*sizeof(double _Complex));
        exitstatus = MPI_Allreduce( vvx, vv[0], 2*numV*numV, MPI_DOUBLE, MPI_SUM, g_ts_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(1);
        }

        fini_1level_buffer ( &vvx );

        retime = _GET_TIME;
        if ( io_proc >= 1 ) fprintf ( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for timeslice reduction = %e seconds\n", retime-ratime );
#endif
        /***********************************************
         * write to AFF
         ***********************************************/
        if ( io_proc >= 1 ) {
          // aff_ratime = _GET_TIME;
#ifdef HAVE_LHPC_AFF
          sprintf ( aff_key, "%s/v-v/t%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, it+g_proc_coords[0]*T,  momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], gamma_id_list[igam] );
          
          affdir = aff_writer_mkpath(affw, affn, aff_key );

          exitstatus = aff_node_put_complex (affw, affdir, vv[0], numV*numV );
          if(exitstatus != 0) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(5);
          }
#else
          if ( fwrite ( vv[0], sizeof(double _Complex), write_count, ofs  ) != write_count ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from fwrite %s %d\n", __FILE__, __LINE__);
            return(5);
          }
#endif
          // aff_retime = _GET_TIME;
          // fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for writing = %e\n", aff_retime-aff_ratime);

        }


        zgemm_ratime = _GET_TIME;
        /***********************************************
         * W^+ Gp V
         ***********************************************/
        F_GLOBAL(zgemm, ZGEMM) ( &CHAR_C, &CHAR_N, &INT_M, &INT_N, &INT_K, &Z_1, (double _Complex*)(Wts[0]), &INT_K, (double _Complex*)(Xtsxpxg[0]), &INT_K, &Z_0, vv[0], &INT_M, 1, 1);

        zgemm_retime = _GET_TIME;
        if ( io_proc == 2 ) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for zgemm = %e seconds\n", zgemm_retime-zgemm_ratime);

#ifdef HAVE_MPI
        ratime = _GET_TIME;

        /***********************************************
         * reduce within global timeslice
         ***********************************************/
        exitstatus = init_1level_buffer ( &vvx, 2*numV*numV );
        if(exitstatus != 0) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(5);
        }
        memcpy( vvx, vv[0], numV*numV*sizeof(double _Complex));
        exitstatus = MPI_Allreduce( vvx, vv[0], 2*numV*numV, MPI_DOUBLE, MPI_SUM, g_ts_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(1);
        }

        fini_1level_buffer ( &vvx );

        retime = _GET_TIME;
        if ( io_proc >= 1 ) fprintf ( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for timeslice reduction = %e seconds\n", retime-ratime );
#endif
        /***********************************************
         * write to AFF
         ***********************************************/
        if ( io_proc >= 1 ) {
          // aff_ratime = _GET_TIME;
#ifdef HAVE_LHPC_AFF
          sprintf ( aff_key, "%s/w-v/t%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, it+g_proc_coords[0]*T, momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], gamma_id_list[igam] );

          affdir = aff_writer_mkpath(affw, affn, aff_key );

          exitstatus = aff_node_put_complex (affw, affdir, vv[0], numV*numV );
          if(exitstatus != 0) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(5);
          }
#else
          if ( fwrite ( vv[0], sizeof(double _Complex), write_count, ofs  ) != write_count ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from fwrite %s %d\n", __FILE__, __LINE__);
            return(5);
          }
#endif
          // aff_retime = _GET_TIME;
          // fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for writing = %e\n", aff_retime-aff_ratime);
        }


        fini_2level_zbuffer ( &vv );

#ifndef HAVE_LHPC_AFF
        if ( io_proc >= 1 ) {
          fflush ( ofs );
          fclose ( ofs );
          ofs = NULL;
        }
#endif
      }  /* end of loop on gamma ids */

    }  /* end of loop on momenta */



    /***********************************************
     * W^+ Gp W
     ***********************************************/
    for ( int imom = 0; imom < momentum_number; imom++ ) {

      /***********************************************
       * multiply with momentum phase 
       ***********************************************/
      for ( int k = 0; k < numV; k++ ) {
        spinor_field_eq_spinor_field_ti_complex_field ( Xtsxp[k], Wts[k], (double*)(phase[imom]), VOL3half );
      }

      /***********************************************
       * loop on gamma matrices
       ***********************************************/
      for ( int igam = 0; igam < gamma_id_number; igam++ ) {

#ifndef HAVE_LHPC_AFF
        FILE *ofs = NULL;
        if ( io_proc >= 1 ) {
          sprintf( filename, "%s.t%.2d.px%.2dpy%.2dpz%.2d.g%.2d.dat", prefix, it+g_proc_coords[0]*T,  momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], gamma_id_list[igam] ); 
          ofs = fopen ( filename, "a+b" );
          if( ofs == NULL ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from open for filename %s %s %d\n", filename, __FILE__, __LINE__);
            return(1);
          }
        }
#endif
        /***********************************************
         * multiply with gamma matrix
         ***********************************************/
        spinor_field_eq_gamma_ti_spinor_field( Xtsxpxg[0], gamma_id_list[igam], Xtsxp[0], numV*VOL3half );
        g5_phi( Xtsxpxg[0], numV*VOL3half );

        /***********************************************
         * reduce in position space
         ***********************************************/
        double _Complex **vv = NULL;

        exitstatus = init_2level_zbuffer ( &vv, numV, numV );
        if(exitstatus != 0) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(5);
        }

        zgemm_ratime = _GET_TIME;
        /***********************************************
         * W^+ Gp W
         ***********************************************/
        F_GLOBAL(zgemm, ZGEMM) ( &CHAR_C, &CHAR_N, &INT_M, &INT_N, &INT_K, &Z_1, (double _Complex*)(Wts[0]), &INT_K, (double _Complex*)(Xtsxpxg[0]), &INT_K, &Z_0, vv[0], &INT_M, 1, 1);
        
        zgemm_retime = _GET_TIME;
        if ( io_proc == 2 ) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for zgemm = %e seconds\n", zgemm_retime-zgemm_ratime);

#ifdef HAVE_MPI
        ratime = _GET_TIME;

        /***********************************************
         * reduce within global timeslice
         ***********************************************/
        double *vvx = NULL;
        exitstatus = init_1level_buffer ( &vvx, 2*numV*numV );

        memcpy( vvx, vv[0], numV*numV*sizeof(double _Complex));
        exitstatus = MPI_Allreduce( vvx, vv[0], 2*numV*numV, MPI_DOUBLE, MPI_SUM, g_ts_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(1);
        }

        fini_1level_buffer ( &vvx );

        retime = _GET_TIME;
        if ( io_proc >= 1 ) fprintf ( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for timeslice reduction = %e seconds\n", retime-ratime );
#endif
        /***********************************************
         * write to AFF
         ***********************************************/
        if ( io_proc >= 1 ) {
          // aff_ratime = _GET_TIME;
#ifdef HAVE_LHPC_AFF
          sprintf ( aff_key, "%s/w-w/t%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, it+g_proc_coords[0]*T, momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], gamma_id_list[igam] );
          
          affdir = aff_writer_mkpath(affw, affn, aff_key );

          exitstatus = aff_node_put_complex (affw, affdir, vv[0], numV*numV );
          if(exitstatus != 0) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(5);
          }
#else
          if ( fwrite ( vv[0], sizeof(double _Complex), write_count, ofs  ) != write_count ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from fwrite %s %d\n", __FILE__, __LINE__);
            return(5);
          }
#endif

          // aff_retime = _GET_TIME;
          // fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for writing = %e\n", aff_retime-aff_ratime);
        }

        fini_2level_zbuffer ( &vv );

#ifndef HAVE_LHPC_AFF
        if ( io_proc >= 1 ) {
          fflush ( ofs );
          fclose( ofs );
          ofs = NULL;
        }
#endif
      }  /* end of loop on gamma ids */

    }  /* end of loop on momenta */

  }  /* end of loop on timeslices */



  ratime = _GET_TIME;
  /***********************************************
   * calculate XV
   ***********************************************/
  for ( int i = 0; i < numV; i++ ) {
    memcpy ( eo_spinor_work[0], V[i], sizeof_eo_spinor_field );
    X_clover_eo ( V[i], eo_spinor_work[0], gauge_field,mzzinv[1][0] );
  }
  retime = _GET_TIME;
  if ( io_proc == 2 ) fprintf( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for XV = %e seconds\n", retime-ratime );

  ratime = _GET_TIME;
  /***********************************************
   * calculate XW
   ***********************************************/
  for ( int i = 0; i < numV; i++ ) {
    memcpy ( eo_spinor_work[0], W[i], sizeof_eo_spinor_field );
    X_clover_eo ( W[i], eo_spinor_work[0], gauge_field,mzzinv[0][0] );
  }
  retime = _GET_TIME;
  if ( io_proc == 2 ) fprintf( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for XW = %e seconds\n", retime-ratime );

  /***********************************************/
  /***********************************************/

  /***********************************************
   * loop on timeslices
   ***********************************************/
  for ( int it = 0; it < T; it++ ) {

    /***********************************************
     * phases for momentum projection
     * EVEN part of timeslice
     ***********************************************/
    make_eo_phase_field_timeslice ( phase, momentum_number, momentum_list, it, 0 );

    ratime = _GET_TIME;
    /***********************************************
     * copy timeslices of fields
     ***********************************************/
    size_t offset = it * _GSI(VOL3half);
    for ( int k = 0; k < numV; k++ ) {
      memcpy ( Vts[k], V[k] + offset, sizeof_eo_spinor_field_timeslice );
    }
    for ( int k = 0; k < numV; k++ ) {
      memcpy ( Wts[k], W[k] + offset, sizeof_eo_spinor_field_timeslice );
    }
    retime = _GET_TIME;
    if ( io_proc == 2 ) fprintf ( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for timeslice fields = %e seconds\n", retime-ratime );

    /***********************************************
     * loop on momenta
     ***********************************************/
    for ( int imom = 0; imom < momentum_number; imom++ ) {

      /***********************************************
       * multiply with momentum phase 
       ***********************************************/
      for ( int k = 0; k < numV; k++ ) {
        spinor_field_eq_spinor_field_ti_complex_field ( Xtsxp[k], Vts[k], (double*)(phase[imom]), VOL3half );
      }

      /***********************************************
       * loop on gamma matrices
       ***********************************************/
      for ( int igam = 0; igam < gamma_id_number; igam++ ) {


#ifndef HAVE_LHPC_AFF
        FILE *ofs = NULL;
        if ( io_proc >= 1 ) {
          sprintf( filename, "%s.t%.2d.px%.2dpy%.2dpz%.2d.g%.2d.dat", prefix, it+g_proc_coords[0]*T,  momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], gamma_id_list[igam] ); 
          ofs = fopen ( filename, "a+b" );
          if( ofs == NULL ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from open for filename %s %s %d\n", filename, __FILE__, __LINE__);
            return(1);
          }
        }
#endif

        /***********************************************
         * multiply with gamma matrix
         ***********************************************/
        spinor_field_eq_gamma_ti_spinor_field( Xtsxpxg[0], gamma_id_list[igam], Xtsxp[0], numV*VOL3half );
        g5_phi ( Xtsxpxg[0], numV*VOL3half );

        /***********************************************
         * reduce in position space
         ***********************************************/
        double _Complex **vv = NULL;

        exitstatus = init_2level_zbuffer ( &vv, numV, numV );
        if(exitstatus != 0) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_2level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(5);
        }

        zgemm_ratime = _GET_TIME;
        /***********************************************
         * V^+ Gp V
         ***********************************************/
        F_GLOBAL(zgemm, ZGEMM) ( &CHAR_C, &CHAR_N, &INT_M, &INT_N, &INT_K, &Z_1, (double _Complex*)(Vts[0]), &INT_K, (double _Complex*)(Xtsxpxg[0]), &INT_K, &Z_0, vv[0], &INT_M, 1, 1);
        
        zgemm_retime = _GET_TIME;
        if ( io_proc == 2 ) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for zgemm = %e seconds\n", zgemm_retime-zgemm_ratime);

#ifdef HAVE_MPI
        ratime = _GET_TIME;

        /***********************************************
         * reduce within global timeslice
         ***********************************************/
        double *vvx = NULL;
        exitstatus = init_1level_buffer ( &vvx, 2*numV*numV );
        if(exitstatus != 0) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(5);
        }

        memcpy( vvx, vv[0], numV*numV*sizeof(double _Complex));
        exitstatus = MPI_Allreduce( vvx, vv[0], 2*numV*numV, MPI_DOUBLE, MPI_SUM, g_ts_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(1);
        }

        fini_1level_buffer ( &vvx );

        retime = _GET_TIME;
        if ( io_proc >= 1 ) fprintf ( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for timeslice reduction = %e seconds\n", retime-ratime );
#endif
        /***********************************************
         * write to AFF
         ***********************************************/
        if ( io_proc >= 1 ) {
          // aff_ratime = _GET_TIME;
#ifdef HAVE_LHPC_AFF
          sprintf ( aff_key, "%s/xv-xv/t%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, it+g_proc_coords[0]*T, momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], gamma_id_list[igam] );
          
          affdir = aff_writer_mkpath(affw, affn, aff_key );

          exitstatus = aff_node_put_complex (affw, affdir, vv[0], numV*numV );
          if(exitstatus != 0) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(5);
          }
#else
          if ( fwrite ( vv[0], sizeof(double _Complex), write_count, ofs  ) != write_count ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from fwrite %s %d\n", __FILE__, __LINE__);
            return(5);
          }
#endif

          // aff_retime = _GET_TIME;
          // fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for writing = %e\n", aff_retime-aff_ratime);
        }

        zgemm_ratime = _GET_TIME;
        /***********************************************
         * W^+ Gp V
         ***********************************************/
        F_GLOBAL(zgemm, ZGEMM) ( &CHAR_C, &CHAR_N, &INT_M, &INT_N, &INT_K, &Z_1, (double _Complex*)(Wts[0]), &INT_K, (double _Complex*)(Xtsxpxg[0]), &INT_K, &Z_0, vv[0], &INT_M, 1, 1);

        zgemm_retime = _GET_TIME;
        if ( io_proc == 2 ) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for zgemm = %e seconds\n", zgemm_retime-zgemm_ratime);

#ifdef HAVE_MPI
        ratime = _GET_TIME;

        /***********************************************
         * reduce within global timeslice
         ***********************************************/
        exitstatus = init_1level_buffer ( &vvx, 2*numV*numV );
        if(exitstatus != 0) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(5);
        }
        memcpy( vvx, vv[0], numV*numV*sizeof(double _Complex));
        exitstatus = MPI_Allreduce( vvx, vv[0], 2*numV*numV, MPI_DOUBLE, MPI_SUM, g_ts_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(1);
        }

        fini_1level_buffer ( &vvx );

        retime = _GET_TIME;
        if ( io_proc >= 1 ) fprintf ( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for timeslice reduction = %e seconds\n", retime-ratime );
#endif
        /***********************************************
         * write to AFF
         ***********************************************/
        if ( io_proc >= 1 ) {
          // aff_ratime = _GET_TIME;
#ifdef HAVE_LHPC_AFF
          sprintf ( aff_key, "%s/xw-xv/t%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, it+g_proc_coords[0]*T, momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], gamma_id_list[igam] );

          affdir = aff_writer_mkpath(affw, affn, aff_key );

          exitstatus = aff_node_put_complex (affw, affdir, vv[0], numV*numV );
          if(exitstatus != 0) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(5);
          }
#else
          if ( fwrite ( vv[0], sizeof(double _Complex), write_count, ofs  ) != write_count ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from fwrite %s %d\n", __FILE__, __LINE__);
            return(5);
          }
#endif

          // aff_retime = _GET_TIME;
          // fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for writing = %e\n", aff_retime-aff_ratime);
        }


        fini_2level_zbuffer ( &vv );

#ifndef HAVE_LHPC_AFF
        if ( io_proc >= 1 ) {
          fflush ( ofs );
          fclose ( ofs );
        }
#endif

      }  /* end of loop on gamma ids */

    }  /* end of loop on momenta */

    /***********************************************
     * W^+ Gp W
     ***********************************************/
    for ( int imom = 0; imom < momentum_number; imom++ ) {

      /***********************************************
       * multiply with momentum phase 
       ***********************************************/
      for ( int k = 0; k < numV; k++ ) {
        spinor_field_eq_spinor_field_ti_complex_field ( Xtsxp[k], Wts[k], (double*)(phase[imom]), VOL3half );
      }

      /***********************************************
       * loop on gamma matrices
       ***********************************************/
      for ( int igam = 0; igam < gamma_id_number; igam++ ) {
#ifndef HAVE_LHPC_AFF
        FILE *ofs = NULL;
        if ( io_proc >= 1 ) {
          sprintf( filename, "%s.t%.2d.px%.2dpy%.2dpz%.2d.g%.2d.dat", prefix, it+g_proc_coords[0]*T,  momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], gamma_id_list[igam] ); 
          ofs = fopen ( filename, "a" );
          if( ofs == NULL ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from open for filename %s %s %d\n", filename, __FILE__, __LINE__);
            return(1);
          }
        }
#endif
        /***********************************************
         * multiply with gamma matrix
         ***********************************************/
        spinor_field_eq_gamma_ti_spinor_field( Xtsxpxg[0], gamma_id_list[igam], Xtsxp[0], numV*VOL3half );
        g5_phi ( Xtsxpxg[0], numV*VOL3half );

        /***********************************************
         * reduce in position space
         ***********************************************/
        double _Complex **vv = NULL;

        exitstatus = init_2level_zbuffer ( &vv, numV, numV );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(1);
        }

        zgemm_ratime = _GET_TIME;
        /***********************************************
         * W^+ Gp W
         ***********************************************/
        F_GLOBAL(zgemm, ZGEMM) ( &CHAR_C, &CHAR_N, &INT_M, &INT_N, &INT_K, &Z_1, (double _Complex*)(Wts[0]), &INT_K, (double _Complex*)(Xtsxpxg[0]), &INT_K, &Z_0, vv[0], &INT_M, 1, 1);

        zgemm_retime = _GET_TIME;
        if ( io_proc == 2 ) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for zgemm = %e seconds\n", zgemm_retime-zgemm_ratime);
        
#ifdef HAVE_MPI
        ratime = _GET_TIME;

        /***********************************************
         * reduce within global timeslice
         ***********************************************/
        double *vvx = NULL;
        exitstatus = init_1level_buffer ( &vvx, 2*numV*numV );
        if(exitstatus != 0) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from init_1level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(5);
        }

        memcpy( vvx, vv[0], numV*numV*sizeof(double _Complex));
        exitstatus = MPI_Allreduce( vvx, vv[0], 2*numV*numV, MPI_DOUBLE, MPI_SUM, g_ts_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(1);
        }

        fini_1level_buffer ( &vvx );

        retime = _GET_TIME;
        if ( io_proc >= 1 ) fprintf ( stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for timeslice reduction = %e seconds\n", retime-ratime );
#endif
        /***********************************************
         * write to AFF
         ***********************************************/
        if ( io_proc >= 1 ) {
          // aff_ratime = _GET_TIME;
#ifdef HAVE_LHPC_AFF
          sprintf ( aff_key, "%s/xw-xw/t%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, it+g_proc_coords[0]*T, momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], gamma_id_list[igam] );
          
          affdir = aff_writer_mkpath(affw, affn, aff_key );

          exitstatus = aff_node_put_complex (affw, affdir, vv[0], numV*numV );
          if(exitstatus != 0) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(5);
          }
#else
          if ( fwrite ( vv[0], sizeof(double _Complex), write_count, ofs  ) != write_count ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from fwrite %s %d\n", __FILE__, __LINE__);
            return(5);
          }
#endif

          // aff_retime = _GET_TIME;
          // fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block] time for writing = %e\n", aff_retime-aff_ratime);
        }

        fini_2level_zbuffer ( &vv );

#ifndef HAVE_LHPC_AFF
        if ( io_proc >= 1 ) fclose ( ofs );
#endif
      }  /* end of loop on gamma ids */

    }  /* end of loop on momenta */

  }  /* end of loop on timeslices */


  /***********************************************/
  /***********************************************/

#ifdef HAVE_LHPC_AFF
  /***********************************************************
  * close the AFF writer
  ***********************************************************/
  if( io_proc >= 1 ) {
    aff_status_str = (char*)aff_writer_close ( affw );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      return(32);
    }
  }  /* end of if io_proc >= 1 */
#endif

  /***********************************************/
  /***********************************************/

  /***********************************************
   * deallocate auxilliary fields
   ***********************************************/
  fini_2level_buffer ( &Vts );
  fini_2level_buffer ( &Wts );
  fini_2level_buffer ( &Xtsxp );
  fini_2level_buffer ( &Xtsxpxg );

  /***********************************************
   * deallocate momentum phase field
   ***********************************************/
  fini_2level_zbuffer ( &phase );

  /***********************************************
   * deallocate W and eo_spinor_work
   ***********************************************/
  fini_2level_buffer ( &W );
  fini_2level_buffer ( &eo_spinor_work );
  
  /***********************************************/
  /***********************************************/

#ifdef HAVE_MPI
  /***********************************************
   * mpi barrier and total time
   ***********************************************/
  if ( ( exitstatus =  MPI_Barrier ( g_cart_grid ) ) != MPI_SUCCESS ) {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block] Error from MPI_Barrier, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(6);
  }
#endif
  total_retime = _GET_TIME;
  if ( io_proc == 2 ) fprintf(stdout, "\n# [gsp_calculate_v_dag_gamma_p_w_block] time for gsp_calculate_v_dag_gamma_p_w_block = %e seconds\n", total_retime-total_ratime);
  fflush ( stdout );

  return(0);

}  /* end of gsp_calculate_v_dag_gamma_p_w_block */


/***********************************************************************************************/
/***********************************************************************************************/

#if 0

/******************************************************************************************************************
 * calculate gsp matrix times vector using t-blocks
 *
 * F_GLOBAL(zgemv, ZGEMV) ( char*TRANS, int *M, int *N, double _Complex *ALPHA, double _Complex *A, int *LDA, double _Complex *X,
 *     int *INCX, double _Complex *BETA, double _Complex * Y, int *INCY, int len_TRANS);
 *
 ******************************************************************************************************************/
int gsp_calculate_v_dag_gamma_p_xi_block(double**V, double*W, int num, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, char*tag) {
  
  double _Complex *V_ptr = (double _Complex *)V[0];
  double _Complex *W_ptr = (double _Complex *)W;
  const size_t sizeof_spinor_point = 12 * sizeof(double _Complex);
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = (LX*LY*LZ)/2;

  int status, iproc, gamma_id;
  int x0, ievecs, k;
  int i_momentum, i_gamma_id, momentum[3];
  int io_proc = 2;
  unsigned int ix;
  size_t items;

  double ratime, retime, momentum_ratime, momentum_retime, gamma_ratime, gamma_retime;
  double spinor1[24], spinor2[24];
  double _Complex **phase = NULL;
  double _Complex *V_buffer=NULL, *W_buffer = NULL, **Z_buffer = NULL;
  double _Complex *zptr=NULL, ztmp;
#ifdef HAVE_MPI
  double _Complex **mZ_buffer = NULL;
#endif
  char filename[200];

  /*variables for blas interface */
  double _Complex BLAS_ALPHA, BLAS_BETA;

  char BLAS_TRANS;
  int BLAS_M, BLAS_N, BLAS_LDA, BLAS_INCX, BLAS_INCY;
  double _Complex *BLAS_A, *BLAS_X, *BLAS_Y;

#ifdef HAVE_MPI
  MPI_Status mstatus;
  int mcoords[4], mrank;
#endif

  FILE *ofs = NULL;

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
  if(io_proc == 1) {
    mcoords[0] = 0; mcoords[1] = 0; mcoords[2] = 0; mcoords[3] = 0;
    MPI_Cart_rank(g_cart_grid, mcoords, &mrank);
  }
#endif

  /***********************************************
   * even/odd phase field for Fourier phase
   ***********************************************/
  phase = (double _Complex**)malloc(T*sizeof(double _Complex*));
  if(phase == NULL) {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
    return(1);
  }
  phase[0] = (double _Complex*)malloc(Vhalf*sizeof(double _Complex));
  if(phase[0] == NULL) {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
    return(2);
  }
  for(x0=1; x0<T; x0++) {
    phase[x0] = phase[x0-1] + VOL3half;
  }

  /***********************************************
   * buffer for result of matrix multiplication
   ***********************************************/
  Z_buffer = (double _Complex**)malloc(T*sizeof(double _Complex*));
  Z_buffer[0] = (double _Complex*)malloc(T*num*sizeof(double _Complex));
  if(Z_buffer[0] == NULL) {
   fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
   return(3);
  }
  for(k=1; k<T; k++) Z_buffer[k] = Z_buffer[k-1] + num;

#ifdef HAVE_MPI
  mZ_buffer = (double _Complex**)malloc(T*sizeof(double _Complex*));
  mZ_buffer[0] = (double _Complex*)malloc(T*num*sizeof(double _Complex));
  if(mZ_buffer[0] == NULL) {
   fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
   return(4);
  }
  for(k=1; k<T; k++) mZ_buffer[k] = mZ_buffer[k-1] + num;
#endif

  /***********************************************
   * buffer for input matrix and vector
   ***********************************************/
  V_buffer = (double _Complex*)malloc(VOL3half*12*num*sizeof(double _Complex));
  if(V_buffer == NULL) {
   fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
   return(5);
  }

  W_buffer = (double _Complex*)malloc(VOL3half*12*sizeof(double _Complex));
  if(W_buffer == NULL) {
   fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
   return(6);
  }

  /***********************************************
   * set BLAS zgemm parameters
   ***********************************************/
  /* this set leads to getting V^+ Gamma(p) W 
   * BUT: requires complex conjugation of V and (Gamma(p) W) */
  BLAS_TRANS = 'T';
  BLAS_M     = 12*VOL3half;
  BLAS_N     = num;
  BLAS_A     = V_ptr;
  BLAS_LDA   = BLAS_M;
  BLAS_INCX  = 1;
  BLAS_INCY  = 1;
  BLAS_ALPHA = 1.;
  BLAS_BETA  = 0.;
  BLAS_X     = W_buffer;

  /***********************************************
   * loop on momenta
   ***********************************************/
  for(i_momentum=0; i_momentum < momentum_number; i_momentum++) {

    momentum[0] = momentum_list[i_momentum][0];
    momentum[1] = momentum_list[i_momentum][1];
    momentum[2] = momentum_list[i_momentum][2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] using source momentum = (%d, %d, %d)\n", momentum[0], momentum[1], momentum[2]);
    }
    momentum_ratime = _GET_TIME;

    /* make phase field in eo ordering */
    gsp_make_o_phase_field_sliced3d (phase, momentum);

    /***********************************************
     * loop on gamma id's
     ***********************************************/
    for(i_gamma_id=0; i_gamma_id < gamma_id_number; i_gamma_id++) {

      gamma_id = gamma_id_list[i_gamma_id];
      if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] using source gamma id %d\n", gamma_id);
      gamma_ratime = _GET_TIME;


      /***********************************************
       * loop on timeslices
       ***********************************************/
      for(x0 = 0; x0 < T; x0++)
      {
        /* copy to timeslice of V to V_buffer */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ievecs) shared(x0)
#endif
        for(ievecs=0; ievecs<num; ievecs++) {
          memcpy(V_buffer+ievecs*12*VOL3half, V_ptr+ 12*(ievecs*Vhalf + x0*VOL3half), VOL3half * sizeof_spinor_point );
        }

        memcpy(W_buffer, W_ptr+ 12*(x0*VOL3half), VOL3half * sizeof_spinor_point);

        /***********************************************
         * apply Gamma(pvec) to W
         ***********************************************/
        ratime = _GET_TIME;

#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix,spinor1,spinor2,zptr,ztmp)
#endif
        for(ix=0; ix<VOL3half; ix++) {
          /* W <- conj( gamma exp ip W ) */
          zptr = W_buffer + 12 * ix;
          ztmp = phase[x0][ix % VOL3half];
          memcpy(spinor1, zptr, sizeof_spinor_point);
          _fv_eq_gamma_ti_fv(spinor2, gamma_id, spinor1);
          zptr[ 0] = conj(ztmp * (spinor2[ 0] + spinor2[ 1] * I));
          zptr[ 1] = conj(ztmp * (spinor2[ 2] + spinor2[ 3] * I));
          zptr[ 2] = conj(ztmp * (spinor2[ 4] + spinor2[ 5] * I));
          zptr[ 3] = conj(ztmp * (spinor2[ 6] + spinor2[ 7] * I));
          zptr[ 4] = conj(ztmp * (spinor2[ 8] + spinor2[ 9] * I));
          zptr[ 5] = conj(ztmp * (spinor2[10] + spinor2[11] * I));
          zptr[ 6] = conj(ztmp * (spinor2[12] + spinor2[13] * I));
          zptr[ 7] = conj(ztmp * (spinor2[14] + spinor2[15] * I));
          zptr[ 8] = conj(ztmp * (spinor2[16] + spinor2[17] * I));
          zptr[ 9] = conj(ztmp * (spinor2[18] + spinor2[19] * I));
          zptr[10] = conj(ztmp * (spinor2[20] + spinor2[21] * I));
          zptr[11] = conj(ztmp * (spinor2[22] + spinor2[23] * I));
        }
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for conj gamma p W = %e seconds\n", retime - ratime);

        ratime = _GET_TIME;

        /* the result vector BLAS_Y */
        BLAS_Y = Z_buffer[x0];


        /***********************************************
         * scalar products as matrix multiplication
         ***********************************************/
        ratime = _GET_TIME;
        F_GLOBAL(zgemv, ZGEMV) ( &BLAS_TRANS, &BLAS_M, &BLAS_N, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_X, &BLAS_INCX, &BLAS_BETA, BLAS_Y, &BLAS_INCY, 1);
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for zgemm = %e seconds\n", retime - ratime);

      }  /* end of loop on timeslice x0 */

      /* Z_buffer <- Z_buffer^* */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(x0) shared(Z_buffer,T,num)
#endif
      for(x0=0; x0<T*num; x0++) {
        Z_buffer[0][x0] = conj( Z_buffer[0][x0] );
      }

      ratime = _GET_TIME;
#ifdef HAVE_MPI
      /* reduce within global timeslice */
      memcpy(mZ_buffer[0], Z_buffer[0], T*num*sizeof(double _Complex));
      MPI_Allreduce(mZ_buffer[0], Z_buffer[0], 2*T*num, MPI_DOUBLE, MPI_SUM, g_ts_comm);
#endif
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for allreduce = %e seconds\n", retime - ratime);


      /***********************************************
       * write gsp to disk
       ***********************************************/
      if(io_proc == 2) {
        sprintf(filename, "%s.px%.2dpy%.2dpz%.2d.g%.2d", tag, momentum[0], momentum[1], momentum[2], gamma_id);
        ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error, could not open file %s for writing\n", filename);
          return(12);
        }
      }
      for(iproc=0; iproc < g_nproc_t; iproc++) {
#ifdef HAVE_MPI
        ratime = _GET_TIME;
        if(iproc > 0) {
          /***********************************************
           * gather at root
           ***********************************************/
          k = 2*T*num; /* number of items to be sent and received */
          if(io_proc == 2) {
            mcoords[0] = iproc; mcoords[1] = 0; mcoords[2] = 0; mcoords[3] = 0;
            MPI_Cart_rank(g_cart_grid, mcoords, &mrank);

            /* receive gsp with tag iproc; overwrite Z_buffer */
            status = MPI_Recv(Z_buffer[0], k, MPI_DOUBLE, mrank, iproc, g_cart_grid, &mstatus);
            if(status != MPI_SUCCESS ) {
              fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] proc%.4d Error from MPI_Recv, status was %d\n", g_cart_id, status);
              return(9);
            }
          } else if (g_proc_coords[0] == iproc && io_proc == 1) {
            /* send correlator with tag 2*iproc */
            status = MPI_Send(Z_buffer[0], k, MPI_DOUBLE, mrank, iproc, g_cart_grid);
            if(status != MPI_SUCCESS ) {
              fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] proc%.4d Error from MPI_Recv, status was %d\n", g_cart_id, status);
              return(10);
            }
          }
        }  /* end of if iproc > 0 */

        retime = _GET_TIME;
        if(io_proc == 2) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for exchange = %e seconds\n", retime-ratime);
#endif  /* of ifdef HAVE_MPI */

        /***********************************************
         * I/O process write to file
         ***********************************************/
        if(io_proc == 2) {
          ratime = _GET_TIME;
          items = (size_t)(T*num);

          if( fwrite(Z_buffer[0], sizeof(double _Complex), items, ofs) != items ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error, could not write proper amount of data to file %s\n", filename);
            return(13);
          }

          retime = _GET_TIME;
          fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for writing = %e seconds\n", retime-ratime);
        }  /* end of if io_proc == 2 */

      }  /* end of loop on iproc */
      if(io_proc == 2) fclose(ofs);

      gamma_retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for gamma id %d = %e seconds\n", gamma_id_list[i_gamma_id], gamma_retime - gamma_ratime);

   }  /* end of loop on gamma id */

   momentum_retime = _GET_TIME;
   if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for momentum (%d, %d, %d) = %e seconds\n",
       momentum[0], momentum[1], momentum[2], momentum_retime - momentum_ratime);

  }   /* end of loop on source momenta */


  if(phase != NULL) {
    if(phase[0] != NULL) free(phase[0]);
    free(phase);
  }
  if(V_buffer != NULL) free(V_buffer);
  if(W_buffer != NULL) free(W_buffer);
  if(Z_buffer != NULL) {
    if(Z_buffer[0] != NULL) free(Z_buffer[0]);
    free(Z_buffer);
  }
#ifdef HAVE_MPI
  if(mZ_buffer != NULL) {
    if(mZ_buffer[0] != NULL) free(mZ_buffer[0]);
    free(mZ_buffer);
  }
#endif

  return(0);

}  /* end of gsp_calculate_v_dag_gamma_p_xi_block */

#endif

/***********************************************************************************************/
/***********************************************************************************************/

#if 0

/***********************************************************************************************
 * calculate gsp using t-blocks
 *
          subroutine zgemm  (   character   TRANSA,
          V^+ Gamma(p) W
          eo - scalar product over even 0 / odd 1 sites

          V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)

          W is numW x (12 VOL3half) (C) = (12 VOL3half) x numW (F)

          zgemm calculates
          V^H x [ (Gamma(p) x W) ] which is numV x numW (F) = numW x numV (C)
 *
 ***********************************************************************************************/
int gsp_calculate_v_dag_gamma_p_w_block_asym(double**V, double**W, int numV, int numW, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, 
    AffWriter_s *affw, char*tag , int io_proc) {
  
  const double _Complex *V_ptr = (double _Complex *)V[0];
  const double _Complex *W_ptr = (double _Complex *)W[0];
  const size_t sizeof_spinor_point = 12 * sizeof(double _Complex);

  const unsigned int Vhalf    = VOLUME / 2;
  const unsigned int VOL3half = (LX*LY*LZ)/2;

  const size_t V_buffer_items = (size_t)VOL3half * 12* (size_t)numV;
  const size_t V_buffer_bytes = V_buffer_items *sizeof(double _Complex);

  const size_t W_buffer_items = (size_t)VOL3half * 12* (size_t)numW;
  const size_t W_buffer_bytes = W_buffer_items *sizeof(double _Complex);

  int exitstatus;
  size_t items, offset, bytes;

  double ratime, retime, momentum_ratime, momentum_retime, gamma_ratime, gamma_retime;
  double _Complex *V_buffer = NULL, *W_buffer = NULL, *Z_buffer = NULL;
  double _Complex *zptr=NULL, ztmp;
#ifdef HAVE_MPI
  double _Complex *mZ_buffer = NULL;
#endif
  char filename[200];

  /***********************************************
   *variables for blas interface
   ***********************************************/
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  char BLAS_TRANSA, BLAS_TRANSB;
  int BLAS_M, BLAS_N, BLAS_K;
  double _Complex *BLAS_A=NULL, *BLAS_B=NULL, *BLAS_C=NULL;
  int BLAS_LDA, BLAS_LDB, BLAS_LDC;

#ifdef HAVE_MPI
  MPI_Status mstatus;
  int mcoords[4], mrank;
#endif


  /***********************************************/
  /***********************************************/

  /***********************************************
   * loop on timeslices
   ***********************************************/
  for ( int it = 0; it < T; it++ ) {

    /***********************************************
     * calculate W from W
     ***********************************************/

    init_2level_buffer ( &


  /***********************************************
   * set BLAS zgemm parameters
   ***********************************************/
  BLAS_TRANSA = 'C';
  BLAS_TRANSB = 'N';
  BLAS_M     = numV;
  BLAS_K     = 12*VOL3half;
  BLAS_N     = numW;
  BLAS_LDA   = BLAS_K;
  BLAS_LDB   = BLAS_K;
  BLAS_LDC   = BLAS_M;

  /***********************************************
   * loop on momenta
   ***********************************************/
  for( int i_momentum = 0; i_momentum < momentum_number; i_momentum++ ) {

    int momentum[3] = { momentum_list[i_momentum][0], momentum_list[i_momentum][1], momentum_list[i_momentum][2] }

    if(g_cart_id == 0 && g_verbose > 2) {
      fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] using source momentum  %3d %3d %3d\n", momentum[0], momentum[1], momentum[2] );
    }
    momentum_ratime = _GET_TIME;

    /***********************************************
     * even/odd phase field for Fourier phase
     ***********************************************/
    double _Complex ***phase = NULL;
    exitstatus = init_3level_zbuffer ( &phase , 2, T, Vhalf );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error from init_3level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(3);
    }


    /***********************************************/
    /***********************************************/

    /***********************************************
     * make phase field in eo ordering
     ***********************************************/
    gsp_make_eo_phase_field_sliced3d (phase, momentum, eo);


    /***********************************************
     * loop on gamma id's
     ***********************************************/
    for(i_gamma_id=0; i_gamma_id < gamma_id_number; i_gamma_id++) {

      gamma_id = gamma_id_list[i_gamma_id];
      if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] using source gamma id %d\n", gamma_id);
      gamma_ratime = _GET_TIME;

      Z_buffer = ((double _Complex*)gsp_out) + ( i_momentum * gamma_id_number + i_gamma_id ) * T * numV * numW;

      /* buffer for input matrices */
      if( (V_buffer = (double _Complex*)malloc(V_buffer_bytes) ) == NULL ) {
       fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error from malloc\n");
       return(5);
      }
      
      if( (W_buffer = (double _Complex*)malloc(W_buffer_bytes)) == NULL ) {
       fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error from malloc\n");
       return(6);
      }

      BLAS_A = V_buffer;
      BLAS_B = W_buffer;

      /***********************************************
       * loop on timeslices
       ***********************************************/
      for(x0 = 0; x0 < T; x0++) {

        /* copy to timeslice of V to V_buffer */
#ifdef HAVE_OPENMP
#pragma omp parallel shared(x0)
{
#endif
        bytes  = VOL3half * sizeof_spinor_point;
        offset = 12 * VOL3half;
        items  = 12 * Vhalf;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ievecs=0; ievecs<numV; ievecs++) {
          memcpy(V_buffer+ievecs*offset, V_ptr+(ievecs*items + x0*offset), bytes );
        }

#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ievecs=0; ievecs<numW; ievecs++) {
          memcpy(W_buffer+ievecs*offset, W_ptr+(ievecs*items + x0*offset), bytes );
        }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
        /***********************************************
         * apply Gamma(pvec) to W
         ***********************************************/
        ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel private(zptr,ztmp)
{
#endif
       
        double spinor2[24];
        zptr = W_buffer;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ix=0; ix<numW*VOL3half; ix++) {
          /* W <- gamma exp ip W */
          ztmp = phase[x0][ix % VOL3half];
          _fv_eq_gamma_ti_fv(spinor2, gamma_id, (double*)zptr);
          zptr[ 0] = ztmp * (spinor2[ 0] + spinor2[ 1] * I);
          zptr[ 1] = ztmp * (spinor2[ 2] + spinor2[ 3] * I);
          zptr[ 2] = ztmp * (spinor2[ 4] + spinor2[ 5] * I);
          zptr[ 3] = ztmp * (spinor2[ 6] + spinor2[ 7] * I);
          zptr[ 4] = ztmp * (spinor2[ 8] + spinor2[ 9] * I);
          zptr[ 5] = ztmp * (spinor2[10] + spinor2[11] * I);
          zptr[ 6] = ztmp * (spinor2[12] + spinor2[13] * I);
          zptr[ 7] = ztmp * (spinor2[14] + spinor2[15] * I);
          zptr[ 8] = ztmp * (spinor2[16] + spinor2[17] * I);
          zptr[ 9] = ztmp * (spinor2[18] + spinor2[19] * I);
          zptr[10] = ztmp * (spinor2[20] + spinor2[21] * I);
          zptr[11] = ztmp * (spinor2[22] + spinor2[23] * I);
          zptr += 12;
        }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for conj gamma p W = %e seconds\n", retime - ratime);

        /***********************************************
         * scalar products as matrix multiplication
         ***********************************************/
        ratime = _GET_TIME;

        /* output buffer */
        BLAS_C = Z_buffer + x0 * numV*numW;

        F_ZGEMM(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC, 1, 1);

        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for zgemm = %e seconds\n", retime - ratime);

      }  /* end of loop on timeslice x0 */
      free(V_buffer); V_buffer = NULL;
      free(W_buffer); W_buffer = NULL;

#ifdef HAVE_MPI
      ratime = _GET_TIME;
      /* reduce within global timeslice */
      items = T * numV * numW;
      bytes = items * sizeof(double _Complex);
      if( (mZ_buffer = (double _Complex*)malloc( bytes )) == NULL ) {
        fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error, could not open file %s for writing\n", filename);
        return(5);
      }
      memcpy(mZ_buffer, Z_buffer, bytes);
      MPI_Allreduce(mZ_buffer, Z_buffer, 2*T*numV*numW, MPI_DOUBLE, MPI_SUM, g_ts_comm);
      free(mZ_buffer); mZ_buffer = NULL;
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for allreduce = %e seconds\n", retime - ratime);
#endif


      /***********************************************
       * write gsp to disk
       ***********************************************/
      if(tag != NULL) {
        if(io_proc == 2) {
          sprintf(filename, "%s.px%.2dpy%.2dpz%.2d.g%.2d", tag, momentum[0], momentum[1], momentum[2], gamma_id);
          ofs = fopen(filename, "w");
          if(ofs == NULL) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error, could not open file %s for writing\n", filename);
            return(12);
          }
        }
#ifdef HAVE_MPI
        items = T * numV * numW;
        bytes = items * sizeof(double _Complex);
        if(io_proc == 2) {
          if( (mZ_buffer = (double _Complex*)malloc( bytes )) == NULL ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error, could not open file %s for writing\n", filename);
            return(7);
          }
        }
#endif

        for(iproc=0; iproc < g_nproc_t; iproc++) {
#ifdef HAVE_MPI
          ratime = _GET_TIME;
          if(iproc > 0) {
            /***********************************************
             * gather at root
             ***********************************************/
            k = 2*T*numV*numW; /* number of items to be sent and received */
            if(io_proc == 2) {
              mcoords[0] = iproc; mcoords[1] = 0; mcoords[2] = 0; mcoords[3] = 0;
              MPI_Cart_rank(g_cart_grid, mcoords, &mrank);

              /* receive gsp with tag iproc; overwrite Z_buffer */
              status = MPI_Recv(mZ_buffer, k, MPI_DOUBLE, mrank, iproc, g_cart_grid, &mstatus);
              if(status != MPI_SUCCESS ) {
                fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] proc%.4d Error from MPI_Recv, status was %d\n", g_cart_id, status);
                return(9);
              }
            } else if (g_proc_coords[0] == iproc && io_proc == 1) {
              /* send correlator with tag 2*iproc */
              status = MPI_Send(Z_buffer, k, MPI_DOUBLE, mrank, iproc, g_cart_grid);
              if(status != MPI_SUCCESS ) {
                fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] proc%.4d Error from MPI_Recv, status was %d\n", g_cart_id, status);
                return(10);
              }
            }
          }  /* end of if iproc > 0 */

          retime = _GET_TIME;
          if(io_proc == 2) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for exchange = %e seconds\n", retime-ratime);
#endif  /* of ifdef HAVE_MPI */

          /***********************************************
           * I/O process write to file
           ***********************************************/
          if(io_proc == 2) {
            ratime = _GET_TIME;
#ifdef HAVE_MPI
            zptr = iproc == 0 ? Z_buffer : mZ_buffer;
#else
            zptr = Z_buffer;
#endif
            if( fwrite(zptr, sizeof(double _Complex), items, ofs) != items ) {
              fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error, could not write proper amount of data to file %s\n", filename);
              return(13);
            }

            retime = _GET_TIME;
            fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for writing = %e seconds\n", retime-ratime);
          }  /* end of if io_proc == 2 */

        }  /* end of loop on iproc */
        if(io_proc == 2) {
#ifdef HAVE_MPI
          free(mZ_buffer);
#endif
          fclose(ofs);
        }
      }  /* end of if tag != NULL */

      gamma_retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for gamma id %d = %e seconds\n", gamma_id_list[i_gamma_id], gamma_retime - gamma_ratime);

   }  /* end of loop on gamma id */

   momentum_retime = _GET_TIME;
   if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for momentum (%d, %d, %d) = %e seconds\n",
       momentum[0], momentum[1], momentum[2], momentum_retime - momentum_ratime);



    fini_3level_zbuffer ( &phase );
  }   /* end of loop on source momenta */

  }  /* end of loop on timeslices */

  return(0);

}  /* end of gsp_calculate_v_dag_gamma_p_w_block_asym */
#endif  /* of if 0 */


/***********************************************************************************************/
/***********************************************************************************************/

#if 0

/************************************************************
 * 
 *  input V: Nev x eo spinor field
 *        W: 60 x Nev coefficients
 *
          V is numV x 12 Vhalf (C) = 12 Vhalf x numV (F)
          W is numW x numV     (C) = numV     x numW (F)

          zgemm calculates V x W which is 12 Vhalf x numW (F) = numW x 12 Vhalf (C)
 *
 ************************************************************/
int gsp_calculate_v_w_block_asym(double*gsp_out, double**V, double**W, unsigned int numV, unsigned int numW) {

  const unsigned int Vhalf = VOLUME / 2;

  double ratime, retime;

  char BLAS_TRANSA, BLAS_TRANSB;
  int BLAS_M, BLAS_K, BLAS_N, BLAS_LDA, BLAS_LDB, BLAS_LDC;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  /***********************************************
   * set BLAS zgemm parameters
   ***********************************************/
  BLAS_TRANSA = 'N';
  BLAS_TRANSB = 'N';
  BLAS_M     = 12 * Vhalf;
  BLAS_K     = numV;
  BLAS_N     = numW;
  BLAS_A     = (double _Complex *)V[0];
  BLAS_B     = (double _Complex *)W[0];
  BLAS_C     = (double _Complex *)gsp_out;
  BLAS_LDA   = BLAS_M;
  BLAS_LDB   = BLAS_K;
  BLAS_LDC   = BLAS_M;


  /***********************************************
   * matrix multiplication
   ***********************************************/
  ratime = _GET_TIME;
  F_GLOBAL(zgemm, ZGEMM) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_w_block_asym] time for zgemm = %e seconds\n", retime - ratime);

  return(0);

}  /* end of gsp_calculate_v_w_block_asym */

#endif

/***********************************************************************************************/
/***********************************************************************************************/

}  /* end of namespace cvc */
