/************************************************
 * gamma.cpp
 *
 * Mon Jun  5 08:37:15 CDT 2017
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * CHANGES:
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <getopt.h>

#include "cvc_linalg.h"
#include "iblas.h"
#include "global.h"
/* #include "cvc_geometry.h" */
#include "gamma.h"
#include "gamma_mult_table.h"

namespace cvc {

static int gamma_mult_table_is_initialized = 0;

int gamma_mult_table[16][16];
double gamma_mult_sign[16][16];
double gamma_adjoint_sign[16];
double gamma_transposed_sign[16];

void init_gamma_matrix (void) {
  if( gamma_mult_table_is_initialized == 0 ) {
    init_gamma_mult_table ();
    gamma_mult_table_is_initialized = 1;
  } 
}

void gamma_matrix_init ( gamma_matrix_type *g) {
  gamma_matrix_zero ( g );
  g->m[0] = g->v +  0;
  g->m[1] = g->v +  4;
  g->m[2] = g->v +  8;
  g->m[3] = g->v + 12;
  g->id = -1;
  g->s  = 1.;
}  /* end of gamma_matrix_init */
  
void gamma_matrix_zero ( gamma_matrix_type *g) {
  memset(g->v , 0, 16*sizeof(double _Complex));
}  /* end of gamma_matrix_zero */

void gamma_matrix_fill ( gamma_matrix_type *g) {

  int id = g->id;
  int p[4], isimag;
  double s[4];

  p[0] = gamma_permutation[id][ 0] / 6;
  p[1] = gamma_permutation[id][ 6] / 6;
  p[2] = gamma_permutation[id][12] / 6;
  p[3] = gamma_permutation[id][18] / 6;
  isimag = gamma_permutation[id][ 0] % 2;
  
  s[0] =  gamma_sign[id][ 0];
  s[1] =  gamma_sign[id][ 6];
  s[2] =  gamma_sign[id][12];
  s[3] =  gamma_sign[id][18];

  gamma_matrix_zero ( g );

  g->m[0][p[0]] = g->s * s[0] * ( isimag ? -I : 1. );
  g->m[1][p[1]] = g->s * s[1] * ( isimag ? -I : 1. );
  g->m[2][p[2]] = g->s * s[2] * ( isimag ? -I : 1. );
  g->m[3][p[3]] = g->s * s[3] * ( isimag ? -I : 1. );

}  /* end of gamma_matrix_fill */

void gamma_matrix_set ( gamma_matrix_type *g, int id, double s ) {
  gamma_matrix_init ( g );
  g->id = id;
  g->s  = s;
  gamma_matrix_fill ( g );
}  /* end of gamma_matrix_set */

void gamma_matrix_printf (gamma_matrix_type *g, char*name, FILE*ofs) {

  fprintf(ofs, "%s <- array(dim=c(4,4))\n", name);
  for (int i=0; i<4; i++) {
  for (int k=0; k<4; k++) {
    fprintf(ofs, "%s[%d,%d] <- %25.16e + %25.16e * 1.i\n", name, i+1, k+1, creal(g->m[i][k]), cimag(g->m[i][k]));
  }}
}  /* end of gamma_matrix_printf */

void gamma_matrix_mult ( gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3 ) {
  g1->id = gamma_mult_table[g2->id][g3->id];
  g1->s  = g2->s * g3->s * gamma_mult_sign[g2->id][g3->id];
  // fprintf(stdout, "# [gamma_matrix_mult] id1 %2d id2 %2d id3 %2d\n", g1->id, g2->id, g3->id);
  gamma_matrix_fill (g1);
}  /* gamma_matrix_mult */

void gamma_matrix_assign ( gamma_matrix_type *g , gamma_matrix_type *p) {
  memcpy( g->v, p->v, 16*sizeof(double _Complex) );
  g->id = p->id;
  g->s = p->s;
}  /* end of gamma_matrix_assign */

void gamma_matrix_transposed (gamma_matrix_type *g, gamma_matrix_type *p) {
  g->id = p->id;
  g->s  = p->s * gamma_transposed_sign[p->id];
  gamma_matrix_fill ( g );
}  /* end of gamma_matrix_transposed */

void gamma_matrix_eq_gamma_matrix_transposed (gamma_matrix_type *g, gamma_matrix_type *p) {
  double _Complex _v[16];
  memcpy( _v, p->v, 16*sizeof(double _Complex));
  g->v[ 0] = _v[ 0];
  g->v[ 1] = _v[ 4];
  g->v[ 2] = _v[ 8];
  g->v[ 3] = _v[12];
  g->v[ 4] = _v[ 1];
  g->v[ 5] = _v[ 5];
  g->v[ 6] = _v[ 9];
  g->v[ 7] = _v[13];
  g->v[ 8] = _v[ 2];
  g->v[ 9] = _v[ 6];
  g->v[10] = _v[10];
  g->v[11] = _v[14];
  g->v[12] = _v[ 3];
  g->v[13] = _v[ 7];
  g->v[14] = _v[11];
  g->v[15] = _v[15];
  g->id = -1;
  g->s  = 0;
}  /* end of gamma_matrix_eq_gamma_matrix_transposed */

void gamma_matrix_adjoint ( gamma_matrix_type *g, gamma_matrix_type *p) {
  g->id = p->id;
  g->s  = p->s * gamma_adjoint_sign[p->id];
  gamma_matrix_fill ( g );
}  /* end of gamma_matrix_adjoint */

void gamma_matrix_eq_gamma_matrix_adjoint ( gamma_matrix_type *g, gamma_matrix_type *p) {
  double _Complex _v[16];
  memcpy( _v, p->v, 16*sizeof(double _Complex));
  g->v[ 0] = conj(_v[ 0]);
  g->v[ 1] = conj(_v[ 4]);
  g->v[ 2] = conj(_v[ 8]);
  g->v[ 3] = conj(_v[12]);
  g->v[ 4] = conj(_v[ 1]);
  g->v[ 5] = conj(_v[ 5]);
  g->v[ 6] = conj(_v[ 9]);
  g->v[ 7] = conj(_v[13]);
  g->v[ 8] = conj(_v[ 2]);
  g->v[ 9] = conj(_v[ 6]);
  g->v[10] = conj(_v[10]);
  g->v[11] = conj(_v[14]);
  g->v[12] = conj(_v[ 3]);
  g->v[13] = conj(_v[ 7]);
  g->v[14] = conj(_v[11]);
  g->v[15] = conj(_v[15]);
  g->id = -1;
  g->s  = 0;
}  /* end of gamma_matrix_eq_gamma_matrix_adjoint */

void gamma_matrix_norm (double *norm, gamma_matrix_type *g) {
  double _norm = 0., _re, _im;
  for ( int i=0; i<16; i++) { 
    _re = creal( g->v[i] );
    _im = cimag( g->v[i] );
    _norm += _re * _re + _im * _im;
  }
  *norm = sqrt( _norm );
}  /* end of gamma_matrix_norm */

void gamma_matrix_eq_gamma_matrix_mi_gamma_matrix (gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3) {
  g1->v[ 0] = g2->v[ 0] - g3->v[ 0];
  g1->v[ 1] = g2->v[ 1] - g3->v[ 1];
  g1->v[ 2] = g2->v[ 2] - g3->v[ 2];
  g1->v[ 3] = g2->v[ 3] - g3->v[ 3];
  g1->v[ 4] = g2->v[ 4] - g3->v[ 4];
  g1->v[ 5] = g2->v[ 5] - g3->v[ 5];
  g1->v[ 6] = g2->v[ 6] - g3->v[ 6];
  g1->v[ 7] = g2->v[ 7] - g3->v[ 7];
  g1->v[ 8] = g2->v[ 8] - g3->v[ 8];
  g1->v[ 9] = g2->v[ 9] - g3->v[ 9];
  g1->v[10] = g2->v[10] - g3->v[10];
  g1->v[11] = g2->v[11] - g3->v[11];
  g1->v[12] = g2->v[12] - g3->v[12];
  g1->v[13] = g2->v[13] - g3->v[13];
  g1->v[14] = g2->v[14] - g3->v[14];
  g1->v[15] = g2->v[15] - g3->v[15];
  g1->id = -1;
  g1->s  = 0;
}  /* end of gamma_matrix_eq_gamma_matrix_mi_gamma_matrix */

void gamma_matrix_eq_gamma_matrix_pl_gamma_matrix_ti_re (gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3, double r) {
  g1->v[ 0] = g2->v[ 0] + g3->v[ 0] * r;
  g1->v[ 1] = g2->v[ 1] + g3->v[ 1] * r;
  g1->v[ 2] = g2->v[ 2] + g3->v[ 2] * r;
  g1->v[ 3] = g2->v[ 3] + g3->v[ 3] * r;
  g1->v[ 4] = g2->v[ 4] + g3->v[ 4] * r;
  g1->v[ 5] = g2->v[ 5] + g3->v[ 5] * r;
  g1->v[ 6] = g2->v[ 6] + g3->v[ 6] * r;
  g1->v[ 7] = g2->v[ 7] + g3->v[ 7] * r;
  g1->v[ 8] = g2->v[ 8] + g3->v[ 8] * r;
  g1->v[ 9] = g2->v[ 9] + g3->v[ 9] * r;
  g1->v[10] = g2->v[10] + g3->v[10] * r;
  g1->v[11] = g2->v[11] + g3->v[11] * r;
  g1->v[12] = g2->v[12] + g3->v[12] * r;
  g1->v[13] = g2->v[13] + g3->v[13] * r;
  g1->v[14] = g2->v[14] + g3->v[14] * r;
  g1->v[15] = g2->v[15] + g3->v[15] * r;
  g1->id = -1;
  g1->s  = 0;
}  /* end of gamma_matrix_eq_gamma_matrix_pl_gamma_matrix_ti_re */

void gamma_matrix_eq_gamma_matrix_ti_gamma_matrix ( gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3 ) {
  char CHAR_N = 'N';
  int INT_N = 4;
  double _Complex Z_1 = 1., Z_0 = 0.;
  /* _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1); */
  _F(zgemm) ( &CHAR_N, &CHAR_N, &INT_N, &INT_N, &INT_N, &Z_1, g3->v, &INT_N, g2->v, &INT_N, &Z_0, g1->v, &INT_N, 1, 1);
  g1->id = -1;
  g1->s  = 0;
}  /* end of gamma_matrix_eq_gamma_matrix_ti_gamma_matrix */

}  /* end of namespace cvc */
