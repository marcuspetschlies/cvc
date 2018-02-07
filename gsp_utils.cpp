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

#if 0
/***********************************************
 * phase field on even/odd sublattice in sliced 3d
 * ordering (which I think is the same as odd
 * ordering)
 * eo - even 0 / odd 1
 ***********************************************/
void gsp_make_eo_phase_field_sliced3d (double _Complex**phase, int *momentum, int eo) {

  double ratime, retime;
  int eo_iseven = (int)(eo == 0);

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
      if(g_iseven[ix] == eo_iseven) {
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

#endif

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
int gsp_read_node (double _Complex ****gsp, int numV, int momentum[3], int gamma_id, char *prefix, char*tag) {

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

  for( int x0 = 0; x0 < T; x0++) {
    for ( int igsp = 0; igsp < gsp_name_num; igsp++ ) {

      sprintf ( aff_buffer_path, "%s/%s/t%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, gsp_name_list[igsp], x0+g_proc_coords[0]*T, momentum[0], momentum[1], momentum[2], gamma_id );
      if(g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [gsp_read_node] current aff path = %s\n", aff_buffer_path);
      affdir = aff_reader_chpath(affr, affn, aff_buffer_path);
      if ( ( exitstatus = aff_node_get_complex (affr, affdir, gsp[x0][igsp][0], (uint32_t)items) ) != 0 ) {
        fprintf(stderr, "[gsp_read_node] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(4);
      }
    }  /* end of loop on gsp names */
  }  /* end of loop on x0 */
#else

  for ( int x0 = 0; x0 < T; x0++ ) {
    sprintf(filename, "%s.t%.2d.px%.2dpy%.2dpz%.2d.g%.2d.dat", prefix, x0+g_proc_coords[0]*T, momentum[0], momentum[1], momentum[2], gamma_id);
    FILE *ifs = fopen(filename, "r");
    if(ifs == NULL) {
      fprintf(stderr, "[gsp_read_node] Error, could not open file %s for writing\n", filename);
      return(5);
    }

    size_t items = 6 * numV * numV;


    if( fread( gsp[x0][0][0], sizeof(double _Complex), items, ifs) != items ) {
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

/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 *
 ***********************************************************************************************/
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

/***********************************************************************************************/
/***********************************************************************************************/

/********************************************************************************
 * multiply diagonals element-wise and sum
 ********************************************************************************/
void co_eq_gsp_diag_ti_gsp_diag (complex *w, double**gsp1, double**gsp2, double*lambda, int num) {

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
    _co_eq_co_ti_co(&z1, (complex*)(gsp1[i]+2*i),  (complex*)(gsp2[i]+2*i) );
    _co_pl_eq_co_ti_re(&waccum, &z1, r);
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

}  /* end of co_eq_gsp_diag_ti_gsp_diag */

/***********************************************************************************************/
/***********************************************************************************************/

/********************************************************************************
 * extract diagonal of gsp, no weight or sum
 ********************************************************************************/
void co_eq_gsp_diag (complex *w, double**gsp1, int num) {

#ifdef HAVE_OPENMP
#pragma omp parallel shared(w,gsp1,num)
{
#endif
  int i;
 
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(i = 0; i < num; i++) {
    _co_eq_co(w+i, (complex*)(gsp1[i]+2*i) );
  }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
}  /* end of co_eq_gsp_diag */

/***********************************************************************************************/
/***********************************************************************************************/

/********************************************************************************
 * extract diagonal of gsp, no weight or sum
 ********************************************************************************/
void co_pl_eq_gsp_diag (complex *w, double**gsp1, int num) {

#ifdef HAVE_OPENMP
#pragma omp parallel shared(w,gsp1,num)
{
#endif
  int i;
 
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(i = 0; i < num; i++) {
    _co_pl_eq_co(w+i, (complex*)(gsp1[i]+2*i) );
  }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
}  /* end of co_pl_eq_gsp_diag */


/***********************************************************************************************/
/***********************************************************************************************/

void gsp_pl_eq_gsp (double _Complex **gsp1, double _Complex **gsp2, int num) {

  int i;

#ifdef HAVE_OPENMP
#pragma omp parallel for private(i) shared(gsp1,gsp2)
#endif
  for(i = 0; i < num*num; i++) {
      gsp1[0][i] += gsp2[0][i];
  }
}  /* gsp_pl_eq_gsp */



/***********************************************************************************************/
/***********************************************************************************************/

/**********************************************************************************
 * calculate XV from V
 **********************************************************************************/
int gsp_calculate_xv_from_v (double **xv, double **v, double **work, int num, double mass, unsigned int N) {

  const size_t sizeof_field = 24 * N * sizeof(double);

  int i;
  double ratime, retime;

  if(xv == NULL || v == NULL || work == NULL || num<=0) {
    fprintf(stderr, "[gsp_calculate_xbarv_from_v] Error, insufficient input\n");
    return(1);
  }

  ratime = _GET_TIME;
  for(i= 0; i<num; i++) {
    /* work0 <- v */
    memcpy(work[0], v[i], sizeof_field);
    /* work1 <- X_eo work0 */
    X_eo (work[1], work[0], mass, g_gauge_field);
    /* xv <- work1 */
    memcpy(xv[i], work[1], sizeof_field);
    retime = _GET_TIME;
  }
  if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_xbarv_from_v] time for XV = %e seconds\n", retime-ratime);

  return(0);
}  /* end of gsp_calculate_xv_from_v */


/******************************************************************************************************************
 * w <- Cbar v = Cbar_from_Xeo(v, xv)
 * w and xv can be same memory region
 ******************************************************************************************************************/
int gsp_calculate_w_from_xv_and_v (double **w, double **xv, double **v, double **work, int num, double mass, unsigned int N) {

  const size_t sizeof_field = 24 * N * sizeof(double);

  int i;
  double ratime, retime;
  double norm;

  ratime = _GET_TIME;
  for(i = 0; i<num; i++) {
    /* work0 <- v */
    memcpy(work[0], v[i], sizeof_field);
    /* work1 <- xv */
    memcpy(work[1], xv[i], sizeof_field);
    /* work0 <- Cbar_from_Xeo (work0, work1; aux = work2 */
    C_from_Xeo (work[0], work[1], work[2], g_gauge_field, mass);
    /*square norm  < work0 | work0 > */
    spinor_scalar_product_re(&norm, work[0], work[0], N);
    if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_w_from_xv_and_v] eval %4d %25.16e\n", i, norm*4.*g_kappa*g_kappa);
    norm = 1./sqrt( norm );
    /* w <- work0 x norm */
    spinor_field_eq_spinor_field_ti_re (w[i],  work[0], norm, N);
  }
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_w_from_xv_and_v] time for W = %e seconds\n", retime-ratime);

  return(0);
}  /* end of gsp_calculate_w_from_xv_and_v */

/***********************************************************************************************/
/***********************************************************************************************/

}  /* end of namespace cvc */
