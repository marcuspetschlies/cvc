/****************************************************
 * make_q2orbits.c
 *
 * Thu Oct  8 20:04:05 CEST 2009
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include <getopt.h>

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "make_q2orbits.h"


int make_qid_lists(int *q2id, int *qhat2id, double **q2list, double **qhat2list, int *q2count, int *qhat2count) {

  int x0, x1, x2, x3, ix;
  int y0, y1, y2, y3;
  int count, i, j;
  double q[4], qhat[4], q2, qhat2, *qlist_aux=(double*)NULL;


  qlist_aux = (double*)malloc(VOLUME/8*sizeof(double));
  if(qlist_aux == (double*)NULL) {
    fprintf(stderr, "could not allocate memory for qlist_aux\n");
    return(201);
  }

  for(ix=0; ix<VOLUME/8; ix++) qlist_aux[ix] = -1.;

  count = -1;
  for(x0=0; x0<T; x0++) {
    if(x0+Tstart>T_global/2) { y0 = x0+Tstart - T_global; }
    else                     { y0 = x0+Tstart; }
    q[0] = 2. * M_PI / (double)T_global * (double)(y0);
  for(x1=0; x1<LX; x1++) {
    if(x1 > LX/2) { y1 = x1 - LX; }
    else          { y1 = x1; }
    q[1] = 2. * M_PI / (double)LX       * (double)(y1);
  for(x2=0; x2<LX; x2++) {
    if(x2 > LY/2) { y2 = x2 - LY; }
    else          { y2 = x2; }
    q[2] = 2. * M_PI / (double)LY       * (double)(y2);
  for(x3=0; x3<LX; x3++) {
    if(x3 > LZ/2) { y3 = x3 - LZ; }
    else          { y3 = x3; }
    q[3] = 2. * M_PI / (double)LZ       * (double)(y3);
    ix   = g_ipt[x0][x1][x2][x3];
    q2   = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    /* fprintf(stdout, "----------------------------------------------\n"); */
    /* fprintf(stdout, "before assignment: count=%3d\tq2 = %e\t ix=%6d\n", count, q2, ix); */
    if(count==-1) {
      /* first value */
      /* fprintf(stdout, "first value\n"); */
      qlist_aux[0] = q2;
      count++;
      continue;
    }
    i=0;
    while( q2-qlist_aux[i]>_Q2EPS && qlist_aux[i] != -1.) i++;
    /* fprintf(stdout, "i=%3d\tcount=%3d\tqlist_aux[i]=%21.12e\tqlist_aux[i-1]=%21.12e\n", i, count, qlist_aux[i], qlist_aux[i-1]); */
    if(qlist_aux[i] == -1.) {
      /* new (largest) value  */
      /* fprintf(stdout, "new largest value\n"); */
      qlist_aux[i] = q2;
      count++;
      continue;
    }
    if(fabs(q2-qlist_aux[i]) <= _Q2EPS) {
      /* value exists */
      /* fprintf(stdout, "value exists\n"); */
      continue;
    }
    if(qlist_aux[i]-q2 > _Q2EPS) {
      /* new intermediate value */
      /* fprintf(stdout, "new intermediate value\n"); */
      for(j=count; j>=i; j--) qlist_aux[j+1] = qlist_aux[j];
      qlist_aux[i] = q2;
      count++;
    }
  }
  }
  }
  }

  for(x0=0; x0<T; x0++) {
    if(x0+Tstart>T_global/2) { y0 = x0+Tstart - T_global; }
    else                     { y0 = x0+Tstart; }
    q[0] = 2. * M_PI / (double)T_global * (double)(y0);
  for(x1=0; x1<LX; x1++) {
    if(x1 > LX/2) { y1 = x1 - LX; }
    else          { y1 = x1; }
    q[1] = 2. * M_PI / (double)LX       * (double)(y1);
  for(x2=0; x2<LX; x2++) {
    if(x2 > LY/2) { y2 = x2 - LY; }
    else          { y2 = x2; }
    q[2] = 2. * M_PI / (double)LY       * (double)(y2);
  for(x3=0; x3<LX; x3++) {
    if(x3 > LZ/2) { y3 = x3 - LZ; }
    else          { y3 = x3; }
    q[3] = 2. * M_PI / (double)LZ       * (double)(y3);
    ix   = g_ipt[x0][x1][x2][x3];
    q2   = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];

    i = -1;
    while(fabs(q2-qlist_aux[++i]) > _Q2EPS);
/*  {
      fprintf(stdout, "i=%8d, q2=%20.12e, qlist=%20.12e\n", i, q2, qlist_aux[i]);
    }
*/
    
    q2id[ix] = i;
  }
  }
  }
  }

  *q2list = (double*)malloc((count+1)*sizeof(double));
  memcpy((void*)(*q2list), (void*)qlist_aux, (count+1)*sizeof(double));
  *q2count = count+1;

  /* test: print list */
  /* for(i=0; i<*q2count; i++) fprintf(stdout, "%3d%21.12e\n", i, (*q2list)[i]); */
  /* for(i=0; i<VOLUME; i++) fprintf(stdout, "%3d%6d\t%21.12e\n", i, q2id[i], (*q2list)[q2id[i]]); */ 

  for(ix=0; ix<VOLUME/8; ix++) qlist_aux[ix] = -1.;

  count = -1;
  for(x0=0; x0<T; x0++) {
    qhat[0] = 2. * sin( M_PI / (double)T_global * (double)(x0+Tstart) );
  for(x1=0; x1<LX; x1++) {
    qhat[1] = 2. * sin( M_PI / (double)LX       * (double)(x1) );
  for(x2=0; x2<LX; x2++) {
    qhat[2] = 2. * sin( M_PI / (double)LY       * (double)(x2) );
  for(x3=0; x3<LX; x3++) {
    qhat[3] = 2. * sin( M_PI / (double)LZ       * (double)(x3) );
    ix   = g_ipt[x0][x1][x2][x3];
    qhat2   = qhat[0]*qhat[0] + qhat[1]*qhat[1] + qhat[2]*qhat[2] + qhat[3]*qhat[3];
    if(count==-1) {
      /* first value */
      qlist_aux[0] = qhat2;
      count++;
      continue;
    }
    i=0;
    while( qhat2-qlist_aux[i]>_Q2EPS && qlist_aux[i] != -1.) i++;
    if(qlist_aux[i] == -1.) {
      /* new (largest) value  */
      qlist_aux[i] = qhat2;
      count++;
      continue;
    }
    if(fabs(qhat2-qlist_aux[i]) <= _Q2EPS) {
      /* value exists */
      continue;
    }
    if(qlist_aux[i]-qhat2 > _Q2EPS) {
      /* new intermediate value */
      for(j=count; j>=i; j--) qlist_aux[j+1] = qlist_aux[j];
      qlist_aux[i] = qhat2;
      count++;
    }
  }
  }
  }
  }

  for(x0=0; x0<T; x0++) {
    q[0] = 2. * sin( M_PI / (double)T_global * (double)(x0+Tstart) );
  for(x1=0; x1<LX; x1++) {
    q[1] = 2. * sin( M_PI / (double)LX       * (double)(x1) );
  for(x2=0; x2<LX; x2++) {
    q[2] = 2. * sin( M_PI / (double)LY       * (double)(x2) );
  for(x3=0; x3<LX; x3++) {
    q[3] = 2. * sin( M_PI / (double)LZ       * (double)(x3) );
    ix   = g_ipt[x0][x1][x2][x3];
    q2   = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];

    i = -1;
    while(fabs(q2-qlist_aux[++i]) > _Q2EPS);
    qhat2id[ix] = i;
  }
  }
  }
  }

  *qhat2list = (double*)malloc((count+1)*sizeof(double));
  memcpy((void*)(*qhat2list), (void*)qlist_aux, (count+1)*sizeof(double));
  *qhat2count = count+1;
 
  /* test: print list */
  /* for(i=0; i<*qhat2count; i++) fprintf(stdout, "%3d%21.12e\n", i, (*qhat2list)[i]); */
  /* for(i=0; i<VOLUME; i++) fprintf(stdout, "%3d%6d\t%21.12e\n", i, qhat2id[i], (*qhat2list)[qhat2id[i]]); */

  free(qlist_aux);
 
  return(0);
}

/***********************************************************************/

int make_q2orbits(int **q2_id, int ***q2_list, double **q2_val, int **q2_count, int *q2_nc, double **h3_val, int h3_nc) {

  const int max_orbits_per_q2=20;
  int count, i, j, ix;

  if(((*q2_id) = (int*)malloc(h3_nc*sizeof(int))) ==(int*)NULL) return(201);
  for(ix=0; ix<h3_nc; ix++) (*q2_id)[ix] = -1;

  if(((*q2_val) = (double*)malloc(h3_nc*sizeof(double))) ==(double*)NULL)
    return(202);
  for(ix=0; ix<h3_nc; ix++) (*q2_val)[ix] = -1.;

  if(((*q2_count) = (int*)malloc(h3_nc*sizeof(int))) ==(int*)NULL) return(203);
  for(ix=0; ix<h3_nc; ix++) (*q2_count)[ix] = 0;
  
  if(((*q2_list) = (int**)malloc(h3_nc*sizeof(int*))) ==(int**)NULL) return(204);
  if((*(*q2_list) = (int*)malloc(max_orbits_per_q2*h3_nc*sizeof(int))) ==(int*)NULL) return(205);
  for(ix=1; ix<h3_nc; ix++) (*q2_list)[ix] = (*q2_list)[ix-1]+max_orbits_per_q2;
  for(ix=0; ix<max_orbits_per_q2*h3_nc; ix++) (*(*q2_list))[ix] = -1;
  
  count = -1;
  for(ix=0; ix<h3_nc; ix++) {
    if(count==-1) {
      fprintf(stdout, "first value\n");
      count++;
      (*q2_val)[count] = h3_val[0][ix];
      continue;
    }
    i=0;
    while(h3_val[0][ix] - (*q2_val)[i] > _Q2EPS && (*q2_val)[i]!=-1.) i++;
    if( (*q2_val)[i]==-1. ) {
      fprintf(stdout, "ix=%d, i=%d, q2=%f, new largest value\n", ix, i, h3_val[0][ix]);
      count++;
      (*q2_val)[i] = h3_val[0][ix];
      continue;
    }
    if( fabs( (*q2_val)[i]-h3_val[0][ix]) <= _Q2EPS ) {
      fprintf(stdout, "ix=%d, i=%d, q2=%f, value exists\n", ix, i, h3_val[0][ix]);
      continue;
    }
    if( (*q2_val)[i] - h3_val[0][ix] > _Q2EPS) {
      fprintf(stdout, "ix=%d, i=%d, q2=%f, new intermediate value, shift\n", ix, i, h3_val[0][ix]);
      for(j=count; j>=i; j--) {
        (*q2_val)[j+1] = (*q2_val)[j];
      }
      (*q2_val)[i] = h3_val[0][ix];
      count++;
    }
    else {
      fprintf(stdout, "ix=%d, q2=%f, nothing of the above\n", ix, h3_val[0][ix]);
    }
  }

  *q2_nc = count+1;
  fprintf(stdout, "q2_nc = %d\n", *q2_nc);

  /* set the id's */
  for(ix=0; ix<h3_nc; ix++) {
    i=0;
    while(fabs( (*q2_val)[i]-h3_val[0][ix]) > _Q2EPS ) i++;
    (*q2_id)[ix] = i;
  }

  fprintf(stdout, "# id's and val's:\n");
  for(ix=0; ix<h3_nc; ix++) {
    fprintf(stdout, "i=%6d, h3_val=%12.5e, q2_id=%6d, q2_val=%12.5e\n", ix, h3_val[0][ix], (*q2_id)[ix], (*q2_val)[(*q2_id)[ix]]);
  }


  for(ix=0; ix<h3_nc; ix++) {
    (*q2_list)[(*q2_id)[ix]][(*q2_count)[(*q2_id)[ix]]] = ix;
    (*q2_count)[(*q2_id)[ix]]++;
  }


  /* print the lists */
  
  fprintf(stdout, "# counts:\n");
  for(ix=0; ix<*q2_nc; ix++) {
    fprintf(stdout, "i=%6d, q2_count=%6d\n", ix, (*q2_count)[ix]);
  }

  fprintf(stdout, "# list members:\n");
  for(ix=0; ix<*q2_nc; ix++) {
    for(i=0; i<(*q2_count)[ix]; i++) {
      fprintf(stdout, "class=%6d, no of mem=%4d, h3 orbit of mem=%4d\n", ix, i, (*q2_list)[ix][i]);
    }
  }



  return(0);
}

/***********************************************************************/

int make_q4orbits(int ***q4_id, double ***q4_val, int ***q4_count, int **q4_nc, 
  int **q2_list, int *q2_count, int q2_nc, double **h3_val) {

  int n, m, i, j, count;

  m=0;
  for(n=0; n<q2_nc; n++) m += q2_count[n];
  fprintf(stdout, "sum q2_count = %d\n", m);

  *q4_id = (int**)malloc(q2_nc*sizeof(int*));
  *(*q4_id) = (int*)malloc(m*sizeof(int));
  for(n=1; n<q2_nc; n++) (*q4_id)[n] = (*q4_id)[n-1] + q2_count[n-1];
  for(n=1; n<m; n++) (*(*q4_id))[n] = -1.;

  *q4_count = (int**)malloc(q2_nc*sizeof(int*));
  *(*q4_count) = (int*)malloc(m*sizeof(int));
  for(n=1; n<q2_nc; n++) (*q4_count)[n] = (*q4_count)[n-1] + q2_count[n-1];
  for(n=1; n<m; n++) (*(*q4_count))[n] = 0;

  *q4_val = (double**)malloc(q2_nc*sizeof(double*));
  *(*q4_val) = (double*)malloc(m*sizeof(double));
  for(n=1; n<q2_nc; n++) (*q4_val)[n] = (*q4_val)[n-1] + q2_count[n-1];
  for(n=1; n<m; n++) (*(*q4_val))[n] = -1.;

  *q4_nc = (int*)malloc(q2_nc*sizeof(int));

  for(n=0; n<q2_nc; n++) {

    fprintf(stdout, "q2 class n = %d, q2_count = %d\n", n, q2_count[n]);

    count=-1;
    for(m=0; m<q2_count[n]; m++) {
      fprintf(stdout, "m = %d, h3_val[1] = %f\n", m, h3_val[1][q2_list[n][m]]);
      if(m==0) {
        (*q4_val)[n][0] = h3_val[1][q2_list[n][0]];
        count++;
        fprintf(stdout, "m=%d, first value\n", m);
        continue;
      }
      i=0;
      while(h3_val[1][q2_list[n][m]] > (*q4_val)[n][i]+_Q2EPS && (*q4_val)[n][i]!=-1.) i++;
      if((*q4_val)[n][i] == -1.) {
        (*q4_val)[n][i] = h3_val[1][q2_list[n][m]];
        count++;
        fprintf(stdout, "i=%d, new largest value\n", i);
        continue;
      }
      if(fabs(h3_val[1][q2_list[n][m]]-(*q4_val)[n][i]) <= _Q2EPS) {
        fprintf(stdout, "i=%d, value exists\n", i);
        continue;
      }
      if((*q4_val)[n][i]-h3_val[1][q2_list[n][m]] > _Q2EPS) {
        for(j=count; j>=i; j--) {
          (*q4_val)[n][j+1] = (*q4_val)[n][j];
        }
        (*q4_val)[n][i] = h3_val[1][q2_list[n][m]];
        count++;
        fprintf(stdout, "i=%d, new intermediate value\n", i);
        continue;
      }
      fprintf(stdout, "WARNING: none of the above\n");
    }

    (*q4_nc)[n] = count+1;
    fprintf(stdout, "q4_nc[%d] = %d\n", n, (*q4_nc)[n]);


    for(m=0; m<q2_count[n]; m++) {
      i=0;
      while(fabs( (*q4_val)[n][i]-h3_val[1][q2_list[n][m]]) > _Q2EPS ) i++;
      (*q4_id)[n][m] = i;
      (*q4_count)[n][m]++;
    }


  }

/*
  for(n=0; n<q2_nc; n++) {
    for(m=0; m<q2_count[n]; m++) {
      fprintf(stdout, "q4_val[%d][%d] = %f\n", n, m, (*q4_val)[n][m]);
    }
  }
*/

  fprintf(stdout, "print the lists:\n");
  for(n=0; n<q2_nc; n++) {
    fprintf(stdout, "q2 orbit number %d\n", n);
    for(m=0; m<(*q4_nc)[n]; m++) {
      fprintf(stdout, "%d\t%25.16e\t%d\n", m, (*q4_val)[n][m], (*q4_count)[n][m]);
    }
  }

  for(n=0; n<q2_nc; n++) {
    for(m=0; m<q2_count[n]; m++) {
      fprintf(stdout, "n=%d, m=%d, q4_id = %d\n", n, m, (*q4_id)[n][m]);
    }
  }
  return(0);
}

/***************************************************************************
 * make_rid_list
 ***************************************************************************/
int make_rid_list(int **rid, double **rlist, int *rcount, double Rmin, double Rmax) {

  int x0, x1, x2, x3, ix;
  int y0, y1, y2, y3;
  int count, i, j;
  int V3 = LX*LY*LZ;
  double q[4], qhat[4], q2, qhat2, *qlist_aux=(double*)NULL;

  qlist_aux = (double*)malloc(V3*sizeof(double));
  if(qlist_aux == (double*)NULL) {
    fprintf(stderr, "could not allocate memory for qlist_aux\n");
    return(201);
  }
  for(ix=0; ix<V3; ix++) qlist_aux[ix] = -1.;

  if(Rmax==-1.) Rmax = sqrt( (double)(LX*LX + LY*LY + LZ*LZ) ) + 0.5;

  count = -1;
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix = g_ipt[0][x1][x2][x3];
    q2 = sqrt( (double)(x1*x1 + x2*x2 + x3*x3) );
    if(q2<Rmin-_Q2EPS || q2>Rmax+_Q2EPS) continue;

    /* fprintf(stdout, "----------------------------------------------\n"); */
    /* fprintf(stdout, "before assignment: count=%3d\tq2 = %e\t ix=%6d\n", count, q2, ix); */
    if(count==-1) {
      /* first value */
      /* fprintf(stdout, "first value\n"); */
      qlist_aux[0] = q2;
      count++;
      continue;
    }
    i=0;
    while( q2-qlist_aux[i]>_Q2EPS && qlist_aux[i] != -1.) i++;
    /* fprintf(stdout, "i=%3d\tcount=%3d\tqlist_aux[i]=%21.12e\tqlist_aux[i-1]=%21.12e\n", i, count, qlist_aux[i], qlist_aux[i-1]); */
    if(qlist_aux[i] == -1.) {
      /* new (largest) value  */
      /* fprintf(stdout, "new largest value\n"); */
      qlist_aux[i] = q2;
      count++;
      continue;
    }
    if(fabs(q2-qlist_aux[i]) <= _Q2EPS) {
      /* value exists */
      /* fprintf(stdout, "value exists\n"); */
      continue;
    }
    if(qlist_aux[i]-q2 > _Q2EPS) {
      /* new intermediate value */
      /* fprintf(stdout, "new intermediate value\n"); */
      for(j=count; j>=i; j--) qlist_aux[j+1] = qlist_aux[j];
      qlist_aux[i] = q2;
      count++;
    }
  }
  }
  }

  fprintf(stdout, "# number of ||r||s: count = %d\n", count);

/*  for(i=0; i<=count; i++) fprintf(stdout, "%6d%25.16e\n", i, qlist_aux[i]); */

  *rid = (int*)malloc(V3*sizeof(int));
  for(ix=0; ix<V3; ix++) (*rid)[ix] = -1;
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    q2   = sqrt( (double)(x1*x1 + x2*x2 + x3*x3) );
    if(q2<Rmin-_Q2EPS || q2>Rmax+_Q2EPS) continue;
    ix = g_ipt[0][x1][x2][x3];

    i = -1;
    while(fabs(q2-qlist_aux[++i]) > _Q2EPS);
/*  {
      fprintf(stdout, "i=%8d, q2=%20.12e, rlist=%20.12e\n", i, q2, qlist_aux[i]);
    }
*/
    (*rid)[ix] = i;
  }
  }
  }

  *rlist = (double*)malloc((count+1)*sizeof(double));
  memcpy((void*)(*rlist), (void*)qlist_aux, (count+1)*sizeof(double));
  *rcount = count+1;

  /* test: print list */
/*
  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    fprintf(stdout, "%3d%3d%3d%6d%25.16e\n", x1, x2, x3, (*rid)[g_ipt[0][x1][x2][x3]], (*rlist)[(*rid)[g_ipt[0][x1][x2][x3]]]);
  }}}
  fprintf(stdout, "---------------\n");
  for(i=0; i<*rcount; i++) fprintf(stdout, "%3d%21.12e\n", i, (*rlist)[i]);
  fprintf(stdout, "---------------\n");
  for(i=0; i<V3; i++) fprintf(stdout, "%3d%6d\t%21.12e\n", i, (*rid)[i], (*rlist)[(*rid)[i]]); 
*/

  free(qlist_aux);
 
  return(0);
}

/***********************************************************************/

