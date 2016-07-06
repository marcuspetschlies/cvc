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
#include "invert_Qtm.h"
#include "gsp.h"


namespace cvc {

#ifdef F_
#define _F(s) s##_
#else
#define _F(s) s
#endif

extern "C" void _F(zgemm) ( char*TRANSA, char*TRANSB, int *M, int *N, int *K, double _Complex *ALPHA, double _Complex *A, int *LDA, double _Complex *B,
    int *LDB, double _Complex *BETA, double _Complex * C, int *LDC, int len_TRANSA, int len_TRANSB);


const int gamma_adjoint_sign[16] = {
  /* the sequence is:
       0, 1, 2, 3, id, 5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
       0  1  2  3   4  5    6    7    8    9   10   11   12   13   14   15 */
       1, 1, 1, 1,  1, 1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1
};

/***********************************************
 * Np - number of momenta
 * Ng - number of gamma matrices
 * Nt - number of timeslices
 * Nv - number of eigenvectors
 ***********************************************/

int gsp_init (double ******gsp_out, int Np, int Ng, int Nt, int Nv) {

  double *****gsp;
  int i, k, l, c, x0;
  size_t bytes;

  /***********************************************
   * allocate gsp space
   ***********************************************/
  gsp = (double*****)malloc(Np * sizeof(double****));
  if(gsp == NULL) {
    fprintf(stderr, "[gsp_init] Error from malloc\n");
    return(1);
  }

  gsp[0] = (double****)malloc(Np * Ng * sizeof(double***));
  if(gsp[0] == NULL) {
    fprintf(stderr, "[gsp_init] Error from malloc\n");
    return(2);
  }
  for(i=1; i<Np; i++) gsp[i] = gsp[i-1] + Ng;

  gsp[0][0] = (double***)malloc(Nt * Np * Ng * sizeof(double**));
  if(gsp[0][0] == NULL) {
    fprintf(stderr, "[gsp_init] Error from malloc\n");
    return(3);
  }

  c = 0;
  for(i=0; i<Np; i++) {
    for(k=0; k<Ng; k++) {
      if (c == 0) {
        c++;
        continue;
      }
      gsp[i][k] = gsp[0][0] + c * Nt;
      c++;
    }
  }

  gsp[0][0][0] = (double**)malloc(Nv * Nt * Np * Ng * sizeof(double*));
  if(gsp[0][0][0] == NULL) {
    fprintf(stderr, "[gsp_init] Error from malloc\n");
    return(4);
  }

  c = 0;
  for(i=0; i<Np; i++) {
    for(k=0; k<Ng; k++) {
      for(x0=0; x0<Nt; x0++) {
        if (c == 0) {
          c++;
          continue;
        }
        gsp[i][k][x0] = gsp[0][0][0] + c * Nv;
        c++;
      }
    }
  }

  bytes = 2 * (size_t)(Nv * Nv * Nt * Np * Ng) * sizeof(double);
  /* fprintf(stdout, "# [gsp_init] bytes = %lu\n", bytes); */
  gsp[0][0][0][0] = (double*)malloc( bytes );

  if(gsp[0][0][0][0] == NULL) {
    fprintf(stderr, "[gsp_init] Error from malloc\n");
    return(5);
  }
  c = 0;
  for(i=0; i<Np; i++) {
    for(k=0; k<Ng; k++) {
      for(x0=0; x0<Nt; x0++) {
        for(l=0; l<Nv; l++) {
          if (c == 0) {
            c++;
            continue;
          }
          gsp[i][k][x0][l] = gsp[0][0][0][0] + c * 2*Nv;
          c++;
        }
      }
    }
  }
  memset(gsp[0][0][0][0], 0, bytes);

  *gsp_out = gsp;
  return(0);
} /* end of gsp_init */

/***********************************************
 * free gsp field
 ***********************************************/
int gsp_fini(double******gsp) {

  if( (*gsp) != NULL ) {
    if((*gsp)[0] != NULL ) {
      if((*gsp)[0][0] != NULL ) {
        if((*gsp)[0][0][0] != NULL ) {
          if((*gsp)[0][0][0][0] != NULL ) {
            free((*gsp)[0][0][0][0]);
            (*gsp)[0][0][0][0] = NULL;
          }
          free((*gsp)[0][0][0]);
          (*gsp)[0][0][0] = NULL;
        }
        free((*gsp)[0][0]);
        (*gsp)[0][0] = NULL;
      }
      free((*gsp)[0]);
      (*gsp)[0] = NULL;
    }
    free((*gsp));
    (*gsp) = NULL;
  }

  return(0);
}  /* end of gsp_fini */


/***********************************************
 * reset gsp field to zero
 ***********************************************/
int gsp_reset (double ******gsp, int Np, int Ng, int Nt, int Nv) {
  size_t bytes = 2 * (size_t)(Nv * Nv) * Nt * Np * Ng * sizeof(double);
  memset((*gsp)[0][0][0][0], 0, bytes);
  return(0);
}  /* end of gsp_reset */


void gsp_make_eo_phase_field (double*phase_e, double*phase_o, int *momentum) {

  const int nthreads = g_num_threads;

  int ix, iix;
  int x0, x1, x2, x3;
  int threadid = 0;
  double ratime, retime;
  double dtmp;


  if(g_cart_id == 0) {
    fprintf(stdout, "# [gsp_make_eo_phase_field] using phase momentum = (%d, %d, %d)\n", momentum[0], momentum[1], momentum[2]);
  }

  ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) private(ix,iix,x0,x1,x2,x3,dtmp,threadid) firstprivate(T,LX,LY,LZ) shared(phase_e, phase_o, momentum)
{
  threadid = omp_get_thread_num();
#endif
  /* make phase field in eo ordering */
  for(x0 = threadid; x0<T; x0 += nthreads) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix  = g_ipt[x0][x1][x2][x3];
      iix = g_lexic2eosub[ix];
      dtmp = 2. * M_PI * (
          (x1 + g_proc_coords[1]*LX) * momentum[0] / (double)LX_global +
          (x2 + g_proc_coords[2]*LY) * momentum[1] / (double)LY_global +
          (x3 + g_proc_coords[3]*LZ) * momentum[2] / (double)LZ_global );
      if(g_iseven[ix]) {
        phase_e[2*iix  ] = cos(dtmp);
        phase_e[2*iix+1] = sin(dtmp);
      } else {
        phase_o[2*iix  ] = cos(dtmp);
        phase_o[2*iix+1] = sin(dtmp);
      }
    }}}
  }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif


  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [gsp_make_eo_phase_field] time for making eo phase field = %e seconds\n", retime-ratime);
}  /* end of gsp_make_eo_phase_field */


/***********************************************
 * phase field on odd sublattice in sliced 3d
 * ordering (which I think is the same as odd
 * ordering)
 ***********************************************/
void gsp_make_o_phase_field_sliced3d (double _Complex**phase, int *momentum) {

  double ratime, retime;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [gsp_make_o_phase_field_sliced3d] using phase momentum = (%d, %d, %d)\n", momentum[0], momentum[1], momentum[2]);
  }

  ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) shared(phase, momentum)
{
#endif
  const double TWO_MPI = 2. * M_PI;
  double phase_part;
  double p[3];

  unsigned int ix, iix;
  int x0, x1, x2, x3;
  double _Complex dtmp;

  p[0] = TWO_MPI * momentum[0] / (double)LX_global;
  p[1] = TWO_MPI * momentum[1] / (double)LY_global;
  p[2] = TWO_MPI * momentum[2] / (double)LZ_global;
  
  phase_part = (g_proc_coords[1]*LX) * p[0] + (g_proc_coords[2]*LY) * p[1] + (g_proc_coords[3]*LZ) * p[2];
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  /* make phase field in o ordering */
  for(x0 = 0; x0<T; x0 ++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix  = g_ipt[x0][x1][x2][x3];
      iix = g_eosub2sliced3d[1][g_lexic2eosub[ix] ];
      dtmp = ( phase_part + x1*p[0] + x2*p[1] + x3*p[2] ) * I; 
      if(!g_iseven[ix]) {
        phase[x0][iix] = cexp(dtmp);
      }
    }}}
  }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [gsp_make_o_phase_field_sliced3d] time for making eo phase field = %e seconds\n", retime-ratime);
}  /* end of gsp_make_o_phase_field_sliced3d */

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

  return(0);

}  /* end of gsp_calculate_v_dag_gamma_p_w */

/***********************************************************************************************
 ***********************************************************************************************
 **
 ** gsp_read_node
 ** - read aff node from file or read binary file
 **
 ***********************************************************************************************
 ***********************************************************************************************/
int gsp_read_node (double ***gsp, int num, int momentum[3], int gamma_id, char*tag) {

  int status;
  size_t items;
  char filename[200];

#ifdef HAVE_LHPC_AFF
  int x0;
  struct AffReader_s *affr = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex *aff_buffer = NULL;
  char aff_buffer_path[200];
  /*  uint32_t aff_buffer_size; */
#else
  FILE *ifs = NULL;
  long int offset;
#endif

#ifdef HAVE_LHPC_AFF
  aff_status_str = (char*)aff_version();
  fprintf(stdout, "# [gsp_read_node] using aff version %s\n", aff_status_str);

  sprintf(filename, "%s.aff", tag);
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

  aff_buffer = (double _Complex*)malloc(2*num*num*sizeof(double _Complex));
  if(aff_buffer == NULL) {
    fprintf(stderr, "[gsp_read_node] Error from malloc\n");
    return(3);
  }
#endif

#ifdef HAVE_LHPC_AFF
  for(x0=0; x0<T; x0++) {
    sprintf(aff_buffer_path, "/%s/px%.2dpy%.2dpz%.2d/g%.2d/t%.2d", tag, momentum[0], momentum[1], momentum[2], gamma_id, 
        x0+g_proc_coords[0]*T);
    /* if(g_cart_id == 0) fprintf(stdout, "# [gsp_read_node] current aff path = %s\n", aff_buffer_path); */

    affdir = aff_reader_chpath(affr, affn, aff_buffer_path);
    items = num*num;
    status = aff_node_get_complex (affr, affdir, aff_buffer, (uint32_t)items);
    /* straightforward memcpy ?*/
    memcpy( gsp[x0][0], aff_buffer, 2*items*sizeof(double));

    if(status != 0) {
      fprintf(stderr, "[gsp_read_node] Error from aff_node_get_complex, status was %d\n", status);
      return(4);
    }
  }  /* end of loop on x0 */
#else

  sprintf(filename, "%s.px%.2dpy%.2dpz%.2d.g%.2d", tag, momentum[0], momentum[1], momentum[2], gamma_id);
  ifs = fopen(filename, "r");
  if(ifs == NULL) {
    fprintf(stderr, "[gsp_read_node] Error, could not open file %s for writing\n", filename);
    return(5);
  }
#ifdef HAVE_MPI
  offset = (long int)(g_proc_coords[0]*T) * (2*num*num) * sizeof(double);
  if( fseek ( ifs, offset, SEEK_SET ) != 0 ) {
    fprintf(stderr, "[] Error, could not seek to file position\n");
    return(6);
  }
#endif
  items = 2 * (size_t)T * num*num;
  if( fread(gsp[0][0], sizeof(double), items, ifs) != items ) {
    fprintf(stderr, "[gsp_read_node] Error, could not read proper amount of data to file %s\n", filename);
    return(7);
  }
  fclose(ifs);

  byte_swap64_v2(gsp[0][0], 2*T*num*num);


#endif

#ifdef HAVE_LHPC_AFF
  aff_reader_close (affr);
#endif

  return(0);

}  /* end of gsp_read_node */



int gsp_write_eval(double *eval, int num, char*tag) {
  
  int status;
  int ievecs;

  double ratime, retime;
  char filename[200];

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

  if(g_cart_id == 0) {
  /***********************************************
   * output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [gsp_write_eval] using aff version %s\n", aff_status_str);

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
    sprintf(filename, "%s", tag );
    ofs = fopen(filename, "w");
    if(ofs == NULL) {
      fprintf(stderr, "[gsp_write_eval] Error, could not open file %s for writing\n", filename);
      return(5);
    }
    for(ievecs=0; ievecs<num; ievecs++) {
      fprintf(ofs, "%25.16e\n", eval[ievecs] );
    }
    fclose(ofs);
#endif
  }  /* end of if g_cart_id == 0 */

  return(0);
}  /* end of gsp_write_eval */

int gsp_read_eval(double **eval, int num, char*tag) {
  
  int status;
  int ievecs;

  double ratime, retime;
  char filename[200];

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex *aff_buffer = NULL;
  char aff_buffer_path[200];
/*  uint32_t aff_buffer_size; */
#else
  FILE *ifs = NULL;
#endif

  /* allocate */
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
  aff_status_str = (char*)aff_version();
  fprintf(stdout, "# [gsp_read_eval] using aff version %s\n", aff_status_str);

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

  affdir = aff_reader_chpath(affr, affn, aff_buffer_path);
  status = aff_node_get_double (affr, affdir, *eval, (uint32_t)num); 
  if(status != 0) {
    fprintf(stderr, "[gsp_read_eval] Error from aff_node_put_double, status was %d\n", status);
    return(3);
  }
  aff_reader_close (affr);
#else
  sprintf(filename, "%s", tag );
  ifs = fopen(filename, "r");
  if(ifs == NULL) {
    fprintf(stderr, "[gsp_read_eval] Error, could not open file %s for reading\n", filename);
    return(5);
  }
  for(ievecs=0; ievecs<num; ievecs++) {
    fscanf(ifs, "%lf", (*eval)+ievecs );
  }
  fclose(ifs);
#endif

  return(0);
}  /* end of gsp_read_eval */

void co_eq_tr_gsp_ti_gsp (complex *w, double**gsp1, double**gsp2, double*lambda, int num) {

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

  _co_eq_zero(w);
#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel shared(w,gsp1,gsp2,lambda,num)
{
#endif
  int i, k;
  complex waccum, w2;
  complex z1, z2;
  double r;

  _co_eq_zero(&waccum);
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(i = 0; i < num; i++) {
  for(k = 0; k < num; k++) {

      r = lambda[i] * lambda[k];

      _co_eq_co(&z1, (complex*)(gsp1[i]+2*k));
      _co_eq_co(&z2, (complex*)(gsp2[k]+2*i));
      _co_eq_co_ti_co(&w2, &z1,&z2);

      /* multiply with real, diagonal Lambda matrix */
      _co_pl_eq_co_ti_re(&waccum, &w2, r);
  }}
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
  _co_pl_eq_co(w, &waccum);
  omp_unset_lock(&writelock);
}  /* end of parallel region */
  omp_destroy_lock(&writelock);
#else
  _co_eq_co(w, &waccum);
#endif

}  /* co_eq_tr_gsp_ti_gsp */

int gsp_printf (double ***gsp, int num, char *name, FILE*ofs) {

  int it, k, l;
  if(ofs == NULL) {
    fprintf(stderr, "[gsp_printf] Error, ofs is NULL\n");
    return(1);
  }
 
  fprintf(ofs, "%s <- array(dim=c(%d,%d,%d))\n", name, T_global,num,num);
  for(it=0; it<T; it++) {
    fprintf(ofs, "# [gsp_printf] %s t = %d\n", name, it + g_proc_coords[0]*T);
    for(k=0; k<num; k++) {
      for(l=0; l<num; l++) {
        fprintf(ofs, "%s[%2d,%4d,%4d] = \t%25.16e + %25.16e * 1.i\n", name, it + g_proc_coords[0]*T+1, k+1, l+1, gsp[it][k][2*l], gsp[it][k][2*l+1]);
      }
    }
  }  /* end of loop on time */

  return(0);
}  /* end of gsp_printf */

void co_eq_tr_gsp (complex *w, double**gsp1, double*lambda, int num) {

#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

  _co_eq_zero(w);
#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel shared(w,gsp1,lambda,num)
{
#endif
  int i;
  complex waccum;
  complex z1;
  double r;

  _co_eq_zero(&waccum);
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(i = 0; i < num; i++) {
      r = lambda[i];
      /* multiply with real, diagonal Lambda matrix */
      waccum.re += gsp1[i][2*i  ] *r;
      waccum.im += gsp1[i][2*i+1] *r;
  }
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
  _co_pl_eq_co(w, &waccum);
  omp_unset_lock(&writelock);
}  /* end of parallel region */
  omp_destroy_lock(&writelock);
#else
  _co_eq_co(w, &waccum);
#endif

}  /* co_eq_tr_gsp */

/***********************************************
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
 ***********************************************/
int gsp_calculate_v_dag_gamma_p_w_block(double**V, double**W, int num, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, char*tag) {
  
  const double _Complex *V_ptr = (double _Complex *)V[0];
  const double _Complex *W_ptr = (double _Complex *)W[0];
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
  double _Complex *V_buffer = NULL, *W_buffer = NULL, **Z_buffer = NULL;
  double _Complex *zptr=NULL, ztmp;
#ifdef HAVE_MPI
  double _Complex **mZ_buffer = NULL;
#endif
  char filename[200];

  /*variables for blas interface */
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  char BLAS_TRANSA = 'N', BLAS_TRANSB = 'C';
  int BLAS_M = num,  BLAS_N = num, BLAS_K = 12*VOL3half;
  double _Complex *BLAS_A, *BLAS_B, *BLAS_C;
  int BLAS_LDA = num;
  int BLAS_LDB = num;
  int BLAS_LDC = num;

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
  Z_buffer[0] = (double _Complex*)malloc(T*num*num*sizeof(double _Complex));
  if(Z_buffer[0] == NULL) {
   fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
   return(3);
  }
  for(k=1; k<T; k++) Z_buffer[k] = Z_buffer[k-1] + num*num;

#ifdef HAVE_MPI
  mZ_buffer = (double _Complex**)malloc(T*sizeof(double _Complex*));
  mZ_buffer[0] = (double _Complex*)malloc(T*num*num*sizeof(double _Complex));
  if(mZ_buffer[0] == NULL) {
   fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
   return(4);
  }
  for(k=1; k<T; k++) mZ_buffer[k] = mZ_buffer[k-1] + num*num;
#endif

  /***********************************************
   * buffer for input matrices
   ***********************************************/
  V_buffer = (double _Complex*)malloc(VOL3half*12*num*sizeof(double _Complex));
  if(V_buffer == NULL) {
   fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
   return(5);
  }
  BLAS_B = V_buffer;

  W_buffer = (double _Complex*)malloc(VOL3half*12*num*sizeof(double _Complex));
  if(V_buffer == NULL) {
   fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
   return(6);
  }
  BLAS_A = W_buffer;

  /***********************************************
   * set BLAS zgemm parameters
   ***********************************************/
/*
 * this set leads to getting ( V^+ Gamma(p) W )^T
  BLAS_TRANSA = 'C';
  BLAS_TRANSB = 'N';
  BLAS_M = num;
  BLAS_K = 12*VOL3half;
  BLAS_N = num;
  BLAS_A = V_buffer;
  BLAS_B = W_buffer;
  BLAS_C = Z_buffer;
  BLAS_LDA = BLAS_K;
  BLAS_LDB = BLAS_K;
  BLAS_LDC = num;
*/
  /* this set leads to getting V^+ Gamma(p) W 
   * BUT: requires complex conjugation of V and (Gamma(p) W) */
  BLAS_TRANSA = 'C';
  BLAS_TRANSB = 'N';
  BLAS_M   = num;
  BLAS_K   = 12*VOL3half;
  BLAS_N   = num;
  BLAS_A   = W_buffer;
  BLAS_B   = V_buffer;
  BLAS_LDA = BLAS_K;
  BLAS_LDB = BLAS_K;
  BLAS_LDC = BLAS_M;

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

    /* TEST */
/*
    for (x0=0; x0<T; x0++) {
      for(ix=0; ix<VOL3half; ix++) {
        fprintf(stdout, "# [] phase %3d %8d %25.16e %25.16e\n", x0, ix, creal(phase[x0][ix]), cimag(phase[x0][ix]) );
      }
    }
*/

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
#pragma omp parallel for shared(x0)
#endif
        for(ievecs=0; ievecs<num; ievecs++) {
          memcpy(V_buffer+ievecs*12*VOL3half, V_ptr+ 12*(ievecs*Vhalf + x0*VOL3half), VOL3half * sizeof_spinor_point );
        }

#ifdef HAVE_OPENMP
#pragma omp parallel for shared(x0)
#endif
        for(ievecs=0; ievecs<num; ievecs++) {
          memcpy(W_buffer+ievecs*12*VOL3half, W_ptr+ 12*(ievecs*Vhalf + x0*VOL3half), VOL3half * sizeof_spinor_point  );
        }

        /***********************************************
         * apply Gamma(pvec) to W
         ***********************************************/
        ratime = _GET_TIME;
#if 0 /* this worked, gives the transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(spinor1,spinor2,zptr,ztmp)
#endif
        for(ix=0; ix<num*VOL3half; ix++) {
          zptr = W_buffer + 12 * ix;
          ztmp = phase[x0][ix % VOL3half];
          memcpy(spinor1, zptr, sizeof_spinor_point);
          _fv_eq_gamma_ti_fv(spinor2, gamma_id, spinor1);
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
        }
#endif  /* of if 0 */

#ifdef HAVE_OPENMP
#pragma omp parallel for private(spinor1,spinor2,zptr,ztmp)
#endif
        for(ix=0; ix<num*VOL3half; ix++) {
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
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ztmp)
#endif
        for(ix=0; ix<12*num*VOL3half; ix++) {
          /* V <- conj( V )*/
          ztmp = V_buffer[ix];
          V_buffer[ix] = conj(ztmp);
        }
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for conj V = %e seconds\n", retime - ratime);

        /***********************************************
         * scalar products as matrix multiplication
         ***********************************************/
        ratime = _GET_TIME;
        BLAS_C = Z_buffer[x0];
        _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for zgemm = %e seconds\n", retime - ratime);

      }  /* end of loop on timeslice x0 */

      ratime = _GET_TIME;
#ifdef HAVE_MPI
      /* reduce within global timeslice */
      memcpy(mZ_buffer[0], Z_buffer[0], T*num*num*sizeof(double _Complex));
      MPI_Allreduce(mZ_buffer[0], Z_buffer[0], 2*T*num*num, MPI_DOUBLE, MPI_SUM, g_ts_comm);
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
          k = 2*T*num*num; /* number of items to be sent and received */
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
          items = (size_t)(T*num*num);

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

}  /* end of gsp_calculate_v_dag_gamma_p_w_block */

void gsp_pl_eq_gsp (double _Complex **gsp1, double _Complex **gsp2, int num) {

  int i;

#ifdef HAVE_OPENMP
#pragma omp parallel for shared(gsp1,gsp2)
#endif
  for(i = 0; i < num*num; i++) {
      gsp1[0][i] += gsp2[0][i];
  }
}  /* gsp_pl_eq_gsp */

}  /* end of namespace cvc */
