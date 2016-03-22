#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "global.h"
#include "laplace_linalg.h"
#define _LAPHS_UTILS
#include "laphs.h"
#undef _LAPHS_UTILS

#include "laphs_io.h"
#include "laphs_utils.h"
#include "laplace.h"

/**************************************************************************************************
 * Perambulator
 *
 *  time_snk_number = T;
 *  time_src_number = laphs_time_src_number;
 *  spin_snk_number = 4;
 *  spin_src_number = laphs_spin_src_number;
 *  evec_snk_number = laphs_eigenvector_number;
 *  evec_src_number = laphs_evec_src_number;
 **************************************************************************************************/

namespace cvc {

void init_perambulator (perambulator_type *peram ) {
  peram->nt_src = 0;
  peram->ns_src = 0;
  peram->nv_src = 0;

  peram->nt_snk = 0;
  peram->ns_snk = 0;
  peram->nv_snk = 0;
  peram->nc_snk = 0;

  peram->irnd = -1;

  peram->v = NULL;
  peram->p = NULL;

}  /* end of init_perambulator */


int alloc_perambulator (perambulator_type *peram, \
                        int time_src_number, int spin_src_number, int evec_src_number,\
                        int time_snk_number, int spin_snk_number, int evec_snk_number, int color_snk_number, \
   char *quark_type, char *snk_type, int irnd ) {

  int i, k, l;
  size_t count, ncol, nrow;

  peram->nt_src = time_src_number;
  peram->ns_src = spin_src_number;
  peram->nv_src = evec_src_number;
  peram->nt_snk = time_snk_number;
  peram->ns_snk = spin_snk_number;
  peram->nv_snk = evec_snk_number;
  peram->nc_snk = color_snk_number;
  if(quark_type == NULL) {
    strcpy( peram->quark_type, _default_perambulator_quark_type );
  } else {
    strcpy( peram->quark_type, quark_type );
  }
  if( snk_type == NULL) {
    strcpy( peram->snk_type, _default_perambulator_snk_type );
  } else {
    strcpy( peram->snk_type, snk_type );
  }
  peram->irnd = irnd;

  /* allocate space for perambulators */
  /* number of rows */
  nrow = (size_t)( evec_snk_number * spin_snk_number * time_snk_number );

  /* number of columns */
  ncol = (size_t)( evec_src_number * spin_src_number * time_src_number );

  count = nrow * ncol * 2;  /* number of real entries in perambulator matrix */

  peram->p  = (double*)malloc( count * sizeof(double));
  if( peram->p == NULL) {
    fprintf(stderr, "[init_perambulator] Error, perambulator field is NULL\n");
    return(1);
  }
  memset(peram->p, 0, count * sizeof(double));


  peram->v = (double****)malloc( time_src_number * sizeof(double***));
  if( peram->v == NULL) {
    fprintf(stderr, "[init_perambulator] Error from malloc\n");
    return(2);
  }

  peram->v[0] = (double***)malloc( time_src_number * spin_src_number * sizeof(double**));
  if( peram->v[0] == NULL) {
    fprintf(stderr, "[init_perambulator] Error from malloc\n");
    return(2);
  }

  for(i=1; i<time_src_number; i++) peram->v[i] = peram->v[i-1] + spin_src_number;

  peram->v[0][0] = (double**)malloc( time_src_number * spin_src_number * evec_src_number * sizeof(double*));
  if( peram->v[0][0] == NULL) {
    fprintf(stderr, "[init_perambulator] Error from malloc\n");
    return(2);
  }

  l = -1;
  for(i=0; i < time_src_number; i++) {
    for(k=0; k < spin_src_number; k++) {
      l++;
      if(l == 0) { continue; }
      peram->v[i][k] = peram->v[0][0] + l * evec_src_number;
    }
  }

  count = 0;
  for(i=0; i < time_src_number; i++) {
    for(k=0; k < spin_src_number; k++) {
      for(l=0; l < evec_src_number; l++) {
        peram->v[i][k][l] = peram->p + count;
        count += 2 * nrow;
  }}}

  return(0);
}  /* init_perambulator */

void fini_perambulator (perambulator_type *peram) {

  if(peram->p != NULL) {
    free(peram->p);
    peram->p = NULL;
  }
  if(peram->v != NULL) {
    if(peram->v[0] != NULL) {
      if(peram->v[0][0] != NULL) {
        free(peram->v[0][0]);
        peram->v[0][0] = NULL;
      }
      free(peram->v[0]);
      peram->v[0] = NULL;
    }
    free(peram->v);
    peram->v = NULL;
  }

  peram->nt_snk = 0;
  peram->ns_snk = 0;
  peram->nv_snk = 0;
  peram->nc_snk = 0;

  peram->nt_src = 0;
  peram->ns_src = 0;
  peram->nv_src = 0;
  strcpy(peram->quark_type, "");
  strcpy(peram->snk_type, "");

}  /* fini_perambulator */


int print_perambulator_info (perambulator_type *peram) {

  fprintf(stdout, "# [print_perambulator_info] nt_src     = %4d\n", peram->nt_src);
  fprintf(stdout, "# [print_perambulator_info] ns_src     = %4d\n", peram->ns_src);
  fprintf(stdout, "# [print_perambulator_info] nv_src     = %4d\n", peram->nv_src);

  fprintf(stdout, "# [print_perambulator_info] nt_snk     = %4d\n", peram->nt_snk);
  fprintf(stdout, "# [print_perambulator_info] ns_snk     = %4d\n", peram->ns_snk);
  fprintf(stdout, "# [print_perambulator_info] nv_snk     = %4d\n", peram->nv_snk);
  fprintf(stdout, "# [print_perambulator_info] nc_snk     = %4d\n", peram->nc_snk);
  fprintf(stdout, "# [print_perambulator_info] i-rnd      = %4d\n", peram->irnd);

  fprintf(stdout, "# [print_perambulator_info] quark-type = %s\n", peram->quark_type);
  fprintf(stdout, "# [print_perambulator_info] snk-type   = %s\n", peram->snk_type);

  return(0);
}  /* end of print_perambulator_info */




/**************************************************************************************************
 * Eigensystem
 *   eigenvectors
 *   eigenvalues
 *   phases
 **************************************************************************************************/

void init_eigensystem (eigensystem_type *es) {
  es->nt    = 0;
  es->nv    = 0;
  es->evec  = NULL;
  es->v     = NULL;
  es->eval  = NULL;
  es->phase = NULL;

}  /* init_eigensystem */

int alloc_eigensystem (eigensystem_type *es, unsigned int nt, unsigned nv) {
  
  int i, k, l;
  size_t count = (size_t)(LX*LY*LZ) * 6;

  es->nt = nt;
  es->nv = nv;

  es->evec  = (double*)malloc(nt*nv * count * sizeof(double));
  es->eval  = (double*)malloc(nt*nv * sizeof(double));
  es->phase = (double*)malloc(nt*nv * sizeof(double));

  if(es->evec == NULL) {
    fprintf(stderr, "[init_eigensystem] Error from malloc\n");
    return(1);
  }

  if(es->eval == NULL) {
    fprintf(stderr, "[init_eigensystem] Error from malloc\n");
    return(2);
  }

  if(es->phase == NULL) {
    fprintf(stderr, "[init_eigensystem] Error from malloc\n");
    return(3);
  }

  es->v = (double***)malloc(nt * sizeof(double**));
  if(es->v == NULL) {
    fprintf(stderr, "[init_eigensystem] Error from malloc\n");
    return(4);
  }

  es->v[0] = (double**)malloc(nt*nv * sizeof(double*));
  if(es->v[0] == NULL) {
    fprintf(stderr, "[init_eigensystem] Error from malloc\n");
    return(5);
  }

  for(i=1; i<nt; i++) {
    es->v[i] = es->v[i-1] + nv;
  }

  l = -1;
  for(i=0; i < nt; i++) {
    for(k=0; k < nv; k++) {
      l++;
      es->v[i][k] = es->evec + l*count;
    }
  }

  memset(es->evec, 0, nt*nv*count*sizeof(double));
  memset(es->eval, 0, nt*nv*sizeof(double));
  memset(es->phase, 0, nt*nv*sizeof(double));
  return(0);

}  /* end of init_eigensystem */

void fini_eigensystem (eigensystem_type *es) {
  if( es->nt > 0 && es->nv>0 ) {

    if(es->evec != NULL) {
      free(es->evec);
      es->evec = NULL;
    }
    if(es->eval != NULL) {
      free(es->eval);
      es->eval = NULL;
    }
    if(es->phase != NULL) {
      free(es->phase);
      es->phase = NULL;
    }
    if(es->v != NULL) {
      if(es->v[0]!=NULL) {
        free(es->v[0]);
      }
      free(es->v);
      es->v = NULL;
    }
  }
  es->nt = 0;
  es->nv = 0;
  
}  /* end of fini_eigensystem */

int print_eigensystem_info (eigensystem_type *es) {
  fprintf(stdout, "# [print_eigensystem_info] nt = %4d\n", es->nt);
  fprintf(stdout, "# [print_eigensystem_info] nv = %4d\n", es->nv);
  return(0);
}  /* end of print_eigensystem_info */


/*************************************************************************
 * random vector
 *
 *
 *************************************************************************/
void init_randomvector (randomvector_type *rv) {
  rv->nt = 0;
  rv->ns = 0;
  rv->nv = 0;
  rv->rvec = NULL;
}  /* end of init_randomvector */

int alloc_randomvector (randomvector_type *rv, int nt, int ns, int nv) {

  int i;
  size_t count = (size_t)nv * (size_t)nt * ns * 2;

  rv->nt = nt;
  rv->ns = ns;
  rv->nv = nv;

  rv->rvec = (double*)malloc(count * sizeof(double));
  if( rv->rvec == NULL ) {
    fprintf(stderr, "[init_random_vector] Error from malloc\n");
    return(1);
  }
  return(0);
}  /* end of init_randomvector */

void fini_randomvector (randomvector_type *rv) {
  if(rv->rvec!= NULL) {
    free( rv->rvec);
    rv->rvec = NULL;
  }
}  /* end of fini_random_vector */

void print_randomvector_info (randomvector_type *rv) {
  fprintf(stdout, "# [print_randomvector_info] nt = %4d\n", rv->nt);
  fprintf(stdout, "# [print_randomvector_info] ns = %4d\n", rv->ns);
  fprintf(stdout, "# [print_randomvector_info] nv = %4d\n", rv->nv);
  return;
}  /* end of print_randomvector_info */

void print_randomvector (randomvector_type *rv, FILE*fp) {

  unsigned int ix;
  int x0, is, i;

  if(fp == NULL) {
    fprintf(stderr, "[print_randomvector] Error, file pointer is NULL\n");
    return;
  }

  if(rv->rvec == NULL) {
    fprintf(stderr, "[print_randomvector] Error, rvec fro randomvector is NULL\n");
    return;
  }

  fprintf(fp, "# [print_randomvector_info] nt = %4d\n", rv->nt);
  fprintf(fp, "# [print_randomvector_info] ns = %4d\n", rv->ns);
  fprintf(fp, "# [print_randomvector_info] nv = %4d\n", rv->nv);
  fprintf(fp, "# [print_randomvector] randomvector\n");
  ix = 0;
  for(x0 = 0; x0 < rv->nt; x0++) {
    for(is = 0; is < rv->ns; is++) {
      for(i  = 0; i  < rv->nv; i++) {
        fprintf(fp, "\t%4d%4d%4d%25.16e%25.16e\n", x0, is, i, rv->rvec[2*ix], rv->rvec[2*ix+1]);
        ix++;
      }
    }
  }
  return;
}  /* end of print_random_vector */



/*********************************************************************************************
 * Project a random vector
 *********************************************************************************************/

int project_randomvector (randomvector_type *prv, randomvector_type *rv, int bt, int bs, int bv) {

  unsigned int idx;
  int nt = rv->nt;
  int ns = rv->ns;
  int nv = rv->nv;
  int it, iv, is;
  int t_num=0, t_stride=0, t_offset=0;
  int v_num=0, v_stride=0, v_offset=0;
  int s_num=0, s_stride=0, s_offset=0;
  int t, v, s;

  if(rv->rvec ==NULL || prv->rvec == NULL) {
    fprintf(stderr, "[project_randomvector] Error, rvec is NULL\n");
    return(1);
  }

  if(strcmp(laphs_time_proj_type, "interlace") == 0 ) {
    t_offset = bt;                                        /* 0 <= bt < interlace */
    t_stride = laphs_time_src_number;                     /* move stride = interlace from one t to the next */
    t_num    = nt / laphs_time_src_number;                /*  nt / interlace */
  } else if(strcmp(laphs_time_proj_type, "block") == 0 ) {
    t_offset = bt * ( nt / laphs_time_src_number );       /* bt * block length */
    t_stride = 1;                                         /* move with stirde 1 inside block */
    t_num    = nt / laphs_time_src_number;                /* numnber of blocks */
  }

  if(strcmp(laphs_evec_proj_type, "interlace") == 0 ) {
    v_offset = bv;                                   
    v_stride = laphs_evec_src_number;               
    v_num    = nv / laphs_evec_src_number;         
  } else if(strcmp(laphs_evec_proj_type, "block") == 0 ) {
    v_offset = bv * ( nv / laphs_evec_src_number );
    v_stride = 1;
    v_num    = nv / laphs_evec_src_number;
  }

  if(strcmp(laphs_spin_proj_type, "interlace") == 0 ) {
    s_offset = bs;                                   
    s_stride = laphs_spin_src_number;               
    s_num    = ns / laphs_spin_src_number;         
  } else if(strcmp(laphs_spin_proj_type, "block") == 0 ) {
    s_offset = bs * ( ns / laphs_spin_src_number );
    s_stride = 1;
    s_num    = ns / laphs_spin_src_number;
  }


  /* TEST */
  fprintf(stdout, "# [project_randomvector] time: %12s %4d %4d %4d\n", laphs_time_proj_type, t_offset, t_stride, t_num);
  fprintf(stdout, "# [project_randomvector] spin: %12s %4d %4d %4d\n", laphs_spin_proj_type, s_offset, s_stride, s_num);
  fprintf(stdout, "# [project_randomvector] evec: %12s %4d %4d %4d\n", laphs_evec_proj_type, v_offset, v_stride, v_num);

  t = t_offset;
  for(it = 0; it < t_num; it++) {

    s = s_offset;
    for(is = 0; is < s_num; is++) {
      
      v = v_offset + iv * v_stride;
      for(iv = 0; iv < v_num; iv++) {

        v = v_offset + iv * v_stride;

        idx = ( t * ns + s ) * nv + v;

        prv->rvec[2*idx  ] = rv->rvec[2*idx  ];
        prv->rvec[2*idx+1] = rv->rvec[2*idx+1];
  
        v += v_stride;
      }  /* end of loop on eigenvectors */

     s += s_stride;
    }    /* end of loop on spin */

    t += t_stride;
  }      /* end of loop on time */

  return(0);
}  /* end of project_random_vector */



/****************************************************************************
 * test an eigensystem
 *
 ****************************************************************************/
int test_eigensystem (eigensystem_type *es, double* gauge_field) {
  
  int mu, i, x0;
  unsigned int ix;
  unsigned int VOL3 = LX*LY*LZ;
  double dtmp[2];
  double norm;

  double *ev_field2=NULL;
  complex w, w1, w2;

  ev_field2 = (double*)malloc(6*VOL3*sizeof(double));
  if(ev_field2 == NULL) {
    fprintf(stderr, "[test_eigensystem] Error from malloc\n");
    EXIT(3);
  }

  norm = 1. / (double)(VOL3*3);

  /**************************************************
   * test Laplace v = lambda v
   **************************************************/
  for(x0=0; x0 < es->nt; x0++)
  {

    for(i=0; i < es->nv; i++)
    {
      dtmp[0]= 0.;
      dtmp[1]= 0.;
      for(ix=0; ix<VOL3; ix++) {
        _co_eq_cv_dag_ti_cv(&w, es->v[x0][i]+_GVI(ix), es->v[x0][i]+_GVI(ix));
        dtmp[0] += w.re;
        dtmp[1] += w.im;
      }
      fprintf(stdout, "# [test_eigensystem] norm of vector no. %d = %e + %ei\n", i, dtmp[0], dtmp[1]);

      /* apply laplace operator */
      cv_eq_laplace_cv(ev_field2, gauge_field,  es->v[x0][i], x0);

      memset(dtmp, 0, 2*sizeof(double));
      
      for(ix=0; ix<VOL3; ix++) {
        for(mu=0; mu<3; mu++) {
          w1.re = es->v[x0][i][_GVI(ix)+2*mu  ];
          w1.im = es->v[x0][i][_GVI(ix)+2*mu+1];
          w2.re = ev_field2[_GVI(ix)+2*mu  ];
          w2.im = ev_field2[_GVI(ix)+2*mu+1];
          _co_eq_co_ti_co_inv(&w, &w2, &w1);
          dtmp[0] += w.re;
          dtmp[1] += w.im;
        }
      }

      dtmp[0] *= norm;
      dtmp[1] *= norm;

      fprintf(stdout, "# [test_eigensystem] averaged eigenvalue %4d %4d %25.16e %25.16e input eigenvalue %25.16e\n", x0, i, dtmp[0], dtmp[1], es->eval[x0*es->nv+i]);

    }  /* end of loop on eigenvectors */

  }  /* of loop on timeslices */

  /**************************************************
   * TODO: test orthogonality
   **************************************************/

  if(ev_field2 != NULL) free(ev_field2);

  return(0);
}  /* end of test_eigensystem */


/************************************************************************************************
 * s = V x r
 * output: fermion vector s
 * input:  eigensystem V, randomvector r
 ************************************************************************************************/

int fv_eq_eigensystem_ti_randomvector (double*s, eigensystem_type *v, randomvector_type*r) {

  int iv, it, is;
  int nv = v->nv;
  int nt = v->nt;
  int ns = r->ns;
  unsigned int ix, iix;
  unsigned int VOL3 = LX*LY*LZ;
  int r_idx;
  double *phi = NULL;
  complex w, w1;

  if(nv != r->nv) {
    fprintf(stderr, "[fv_eq_eigensystem_ti_randomvector] Error inconsistent nv in eigensystem and randomvector\n");
    return(1);
  }

  if(nt != r->nt ) {
    fprintf(stderr, "[fv_eq_eigensystem_ti_randomvector] Error inconsistent nt in eigensystem and randomvector\n");
    return(2);
  }

  if(nt != T ) {
    fprintf(stderr, "[fv_eq_eigensystem_ti_randomvector] Error eigensystem has to few t-values\n");
    return(3);
  }

  if(ns != 4 ) {
    fprintf(stderr, "[fv_eq_eigensystem_ti_randomvector] Error spin value in randomvector too low\n");
    return(4);
  }
  
  memset(s, 0, 24*VOLUME*sizeof(double));

  for(it = 0; it<T; it++) {
    for(iix = 0; iix<VOL3; iix++) {

      ix = it * VOL3 + iix;

      for(is=0; is < ns; is++) {

        phi = s + _GSI(ix) + 6*is;

        for(iv=0; iv<nv; iv++) {


          r_idx = ( it * ns + is ) * nv + iv;

          w1.re = r->rvec[2*r_idx  ];
          w1.im = r->rvec[2*r_idx+1];

          /* 1st color component */
          w.re = v->v[it][iv][6*iix+ 0];
          w.im = v->v[it][iv][6*iix+ 1];


          _co_pl_eq_co_ti_co( (complex*)(phi+0), &w, &w1);

          /* 2nd color component */
          w.re = v->v[it][iv][6*iix+ 2];
          w.im = v->v[it][iv][6*iix+ 3];

          _co_pl_eq_co_ti_co( (complex*)(phi+2), &w, &w1);


          /* 3rd color component */
          w.re = v->v[it][iv][6*iix+ 4];
          w.im = v->v[it][iv][6*iix+ 5];

          _co_pl_eq_co_ti_co( (complex*)(phi+4), &w, &w1);


        }  /* end loop on eigenvectors */
      }    /* end of loop on spin */
    }      /* end of loop on 3-dim. volume */
  }        /* end of loop on time */

  return(0);
}  /* end of fv_eq_eigensystem_ti_randomvector */


/************************************************************************************************
 * perambulator = V^+ D^{-1} s
 * output: perambulator p
 * input:  eigensystem V, fermion vector s
 *         {it,is,iv}_src the src-component the perambulator is associated with
 *
 * TODO: change to color vector scalar product
 ************************************************************************************************/

int perambulator_eq_eigensystem_dag_ti_fv (perambulator_type*p, eigensystem_type*v, double*s, int it_src, int is_src, int iv_src) {

  int it, is, iv;
  unsigned int ix, iix, p_idx;
  unsigned int VOL3 = LX*LY*LZ;
  int nt = p->nt_snk;
  int ns = p->ns_snk;
  int nv = p->nv_snk;
  double *phi=NULL;
  double *p_ptr = NULL;
  complex w, w1;


  if( nt != v->nt || nt != T) {
    fprintf(stderr, "[perambulator_eq_eigensystem_dag_ti_fv] Error, inconsistent nt from perambulator and eigensystem or nt not equal to T\n");
    exit(1);
  }

  if( ns != 4) {
    fprintf(stderr, "[perambulator_eq_eigensystem_dag_ti_fv] Error, need ns = 4 at sink to set perambulator from fermion vector\n");
    exit(2);
  }

  if( nv != v->nv ) {
    fprintf(stderr, "[perambulator_eq_eigensystem_dag_ti_fv] Error, inconsistent nv from perambulator and eigensystem\n");
    exit(3);
  }

  for(it=0; it<nt; it++) {
    for(is=0; is<ns; is++) {
      for(iv=0; iv<nv; iv++) {

        p_idx = (it * ns  + is ) * nv + iv;

        p_ptr = p->v[it_src][is_src][iv_src] + 2*p_idx;
        p_ptr[0] = 0.;
        p_ptr[1] = 0.;

        for(iix = 0; iix < VOL3; iix++) {
          ix = it * VOL3 + iix;

          phi = s + _GSI(ix) + 6*is;

          /* 1st color component */
          w.re  = v->v[it][iv][6*iix + 0];
          w.im  = v->v[it][iv][6*iix + 1];
          w1.re = phi[0];
          w1.im = phi[1];
          _co_pl_eq_co_ti_co_conj( (complex*)p_ptr, &w1, &w);
                    
          /* 2nd color component */
          w.re  = v->v[it][iv][6*iix + 2];
          w.im  = v->v[it][iv][6*iix + 3];
          w1.re = phi[2];
          w1.im = phi[3];
          _co_pl_eq_co_ti_co_conj( (complex*)p_ptr, &w1, &w);
                    
          /* 3rd color component */
          w.re  = v->v[it][iv][6*iix + 4];
          w.im  = v->v[it][iv][6*iix + 5];
          w1.re = phi[4];
          w1.im = phi[5];
          _co_pl_eq_co_ti_co_conj( (complex*)p_ptr, &w1, &w);



        }  /* end of loop on 3-dim. volume */
      }    /* end of loop on eigenvectors */
    }      /* end of loop on spin */
  }        /* end of loop on time */

  return(0);
}

#if 0
#endif
}
