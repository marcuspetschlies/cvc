/***************************************************
 * gsp.cpp
 ***************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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
#include "global.h"
#include "ilinalg.h"
#include "cvc_geometry.h"
#include "read_input_parser.h"
#include "invert_Qtm.h"
#include "gsp.h"

namespace cvc {


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

  gsp[0][0][0][0] = (double*)calloc(2 * Nv * Nv * Nt * Np * Ng, sizeof(double));
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
          }
          free((*gsp)[0][0][0]);
        }
        free((*gsp)[0][0]);
      }
      free((*gsp)[0]);
    }
    free((*gsp));
  }

  return(0);
}  /* end of gsp_fini */

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
#pragma omp parallel default(shared) private(ix,iix,x0,x1,x2,x3,dtmp,threadid) firstprivate(nthreads,T,LX,LY,LZ) shared(phase_e, phase_o, momentum)
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


/***********************************************************************************************
 ***********************************************************************************************
 **
 ** gsp_calculate_v_dag_gamma_p_w
 **
 ** calculate V^+ Gamma(p) W
 **
 ***********************************************************************************************
 ***********************************************************************************************/

int gsp_calculate_v_dag_gamma_p_w(double *****gsp, double**V, double**W, int num, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, char*tag, int symmetric) {
  
  int status;
  int x0, ievecs, kevecs, k;
  int isource_momentum, isource_gamma_id, momentum[3];

  double *phase_e=NULL, *phase_o = NULL;
  double *****gsp_buffer = NULL;
  double *buffer=NULL, *buffer2 = NULL;
  size_t items;
  double ratime, retime, momentum_ratime, momentum_retime, gamma_ratime, gamma_retime;
  unsigned int Vhalf = VOLUME / 2;
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

  /***********************************************
   * even/odd phase field for Fourier phase
   ***********************************************/
  phase_e = (double*)malloc(Vhalf*2*sizeof(double));
  phase_o = (double*)malloc(Vhalf*2*sizeof(double));
  if(phase_e == NULL || phase_o == NULL) {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
    EXIT(14);
  }

  buffer = (double*)malloc(2*Vhalf*sizeof(double));
  if(buffer == NULL)  {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
    EXIT(19);
  }

  buffer2 = (double*)malloc(2*T*sizeof(double));
  if(buffer2 == NULL)  {
    fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
    EXIT(20);
  }

  /***********************************************
   * output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {

    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] using aff version %s\n", aff_status_str);


    sprintf(filename, "%s.aff", tag);
    fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] writing gsp data from file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from aff_writer, status was %s\n", aff_status_str);
      EXIT(102);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error, aff writer is not initialized\n");
      EXIT(103);
    }

    if(g_cart_id == 0) {
      aff_buffer = (double _Complex*)malloc(num*num*sizeof(double _Complex));
      if(aff_buffer == NULL) {
        fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from malloc\n");
        EXIT(22);
      }
    }
  }
#endif

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
              memcpy(gsp[isource_momentum][isource_gamma_id][x0][ievecs] + 2*kevecs, buffer2+2*x0, 2*sizeof(double));
            }
  
          }
        }
  
        /* gsp[k,i] = sigma_Gamma gsp[i,k]^+ */
        for(x0=0; x0<T; x0++) {
          for(ievecs = 0; ievecs<num-1; ievecs++) {
            for(kevecs = ievecs+1; kevecs<num; kevecs++) {
              gsp[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs  ] =  gamma_adjoint_sign[gamma_id_list[isource_gamma_id]] * gsp[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs  ];
              gsp[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs+1] = -gamma_adjoint_sign[gamma_id_list[isource_gamma_id]] * gsp[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs+1];
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
              memcpy(gsp[isource_momentum][isource_gamma_id][x0][ievecs] + 2*kevecs, buffer2+2*x0, 2*sizeof(double));
            }
  
          }
        }
      }  /* end of if symmetric */

      /***********************************************
       * write gsp to disk
       ***********************************************/

#ifdef HAVE_MPI
      status = gsp_init (&gsp_buffer, 1, 1, T_global, num);
      if(gsp_buffer == NULL) {
        fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from gsp_init\n");
        EXIT(18);
      }
      k = 2*T*num*num; /* number of items to be sent and received */
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)

      /* fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] proc%.2d g_tr_id = %d; g_tr_nproc =%d\n", g_cart_id, g_tr_id, g_tr_nproc);*/
      MPI_Allgather(gsp[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, g_tr_comm);
  
#else
      /* collect at 0 from all times */
      MPI_Gather(gsp[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, 0, g_cart_grid);
#endif

#else
      gsp_buffer = gsp;
#endif  /* of ifdef HAVE_MPI */

      if(g_cart_id == 0) {
#ifdef HAVE_LHPC_AFF

        for(x0=0; x0<T_global; x0++) {
          sprintf(aff_buffer_path, "/%s/px%.2dpy%.2dpz%.2d/g%.2d/t%.2d", tag, momentum[0], momentum[1], momentum[2], gamma_id_list[isource_gamma_id], x0);
          /* if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] current aff path = %s\n", aff_buffer_path); */

          affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
          items = num*num;
          memcpy(aff_buffer, gsp_buffer[0][0][x0][0], 2*items*sizeof(double));
          status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)items); 
          if(status != 0) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from aff_node_put_double, status was %d\n", status);
            EXIT(104);
          }
        }  /* end of loop on x0 */
#else

        sprintf(filename, "%s.px%.2dpy%.2dpz%.2d.g%.2d", tag, momentum[0], momentum[1], momentum[2],
            gamma_id_list[isource_gamma_id]);
        ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error, could not open file %s for writing\n", filename);
          EXIT(103);
        }
        items = 2 * (size_t)T * num*num;
        if( fwrite(gsp_buffer[0][0][0][0], sizeof(double), items, ofs) != items ) {
          fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error, could not write proper amount of data to file %s\n", filename);
          EXIT(104);
        }
        fclose(ofs);

#endif
      }  /* end of if g_cart_id == 0 */


#ifdef HAVE_MPI
      gsp_fini(&gsp_buffer);
#else
      gsp_buffer = NULL;
#endif

     gamma_retime = _GET_TIME;
     if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for gamma id %d = %e seconds\n", gamma_id_list[isource_gamma_id], gamma_retime - gamma_ratime);

   }  /* end of loop on source gamma id */

   momentum_retime = _GET_TIME;
   if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w] time for momentum (%d, %d, %d) = %e seconds\n",
       momentum[0], momentum[1], momentum[2], momentum_retime - momentum_ratime);

  }   /* end of loop on source momenta */

#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(104);
    }
  }
#endif

  if(phase_e != NULL) free(phase_e);
  if(phase_o != NULL) free(phase_o);
  if(buffer  != NULL) free(buffer);
  if(buffer2 != NULL) free(buffer2);

#ifdef HAVE_LHPC_AFF
  if(aff_buffer != NULL) free(aff_buffer);
#endif

  return(0);

}  /* end of calculate_gsp */

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
    if(g_cart_id == 0) fprintf(stdout, "# [gsp_read_node] current aff path = %s\n", aff_buffer_path);

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
    fscanf(ifs, "%lf", (*eval)[ievecs] );
  }
  fclose(ifs);
#endif

  return(0);
}  /* end of gsp_read_eval */

void co_eq_tr_gsp_ti_gsp (complex *w, double**gsp1, double**gsp2, double*lambda, int num) {

  const int nthreads = g_num_threads;

  int threadid = 0;
#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

  _co_eq_zero(w);
#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel private(threadid) firstprivate(nthreads) shared(w,gsp1,gsp2,lambda,num)
{
  threadid = omp_get_thread_num();
#endif
  int i, k;
  complex waccum, w2;
  complex z1, z2;
  double r;

  _co_eq_zero(&waccum);

  for(i = threadid; i < num; i += nthreads) {
    for(k=0; k<num; k++) {

      r = lambda[i] * lambda[k];

      _co_eq_co(&z1, (complex*)(gsp1[i]+2*k));
      _co_eq_co(&z2, (complex*)(gsp2[k]+2*i));
      _co_eq_co_ti_co(&w2, &z1,&z2);

      /* multiply with real, diagonal Lambda matrix */
      _co_pl_eq_co_ti_re(&waccum, &w2, r);
    }
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

}  /* co_eq_tr_gsp_ti_gsp */

int gsp_printf (double ***gsp, int num, char *name, FILE*ofs) {

  int it, k, l;
  if(ofs == NULL) {
    fprintf(stderr, "[] Error, ofs is NULL\n");
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

}  /* end of namespace cvc */
