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

#if 0
int gamma_mult_table[16][16];
double gamma_mult_sign[16][16];
double gamma_adjoint_sign[16];
double gamma_transposed_sign[16];
#endif  // of if 0

/************************************************/
/************************************************/

void init_gamma_matrix (void) {
  if( gamma_mult_table_is_initialized == 0 ) {
    init_gamma_mult_table ();
    gamma_mult_table_is_initialized = 1;
  } 
}

/************************************************/
/************************************************/

void gamma_matrix_init ( gamma_matrix_type *g) {
  gamma_matrix_zero ( g );
  g->m[0] = g->v +  0;
  g->m[1] = g->v +  4;
  g->m[2] = g->v +  8;
  g->m[3] = g->v + 12;
  g->id = -1;
  g->s  = 1.;
  g->z  = 1.;
}  /* end of gamma_matrix_init */
  
/************************************************/
/************************************************/
 
void gamma_matrix_zero ( gamma_matrix_type *g) {
  memset(g->v , 0, 16*sizeof(double _Complex));
}  /* end of gamma_matrix_zero */

/************************************************/
/************************************************/
 
void gamma_matrix_fill ( gamma_matrix_type *g) {

  int id = g->id;
  int p[4], isimag;
  double s[4];

  if ( id < 0 || id > 15 ) {
    /* fprintf ( stderr, "[gamma_matrix_fill] gamma id to be filled is outside basis range\n" ); */
    return;
  }

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

  /* fprintf(stdout, "# [gamma_matrix_fill] filling gamma matrix with p = (%d,%d,%d,%d), s = (%f, %f, %f, %f) isimag = %d\n",
      p[0], p[1], p[2], p[3], s[0], s[1], s[2], s[3], isimag); */

  g->m[0][p[0]] = g->s * s[0] * ( isimag ? -I : 1. );
  g->m[1][p[1]] = g->s * s[1] * ( isimag ? -I : 1. );
  g->m[2][p[2]] = g->s * s[2] * ( isimag ? -I : 1. );
  g->m[3][p[3]] = g->s * s[3] * ( isimag ? -I : 1. );

}  /* end of gamma_matrix_fill */

/************************************************/
/************************************************/

void gamma_matrix_set ( gamma_matrix_type *g, int id, double s ) {
  gamma_matrix_init ( g );
  g->id = id;
  g->s  = s;
  gamma_matrix_fill ( g );
}  /* end of gamma_matrix_set */

/************************************************/
/************************************************/

void gamma_matrix_printf (gamma_matrix_type *g, char*name, FILE*ofs) {

  fprintf(ofs, "# [gamma_matrix_printf] %s id %2d sign %16.7e z %16.7e %16.7e\n", name, g->id, g->s, creal(g->z) , cimag(g->z) );
  fprintf(ofs, "%s <- array(dim=c(4,4))\n", name);
  for (int i=0; i<4; i++) {
  for (int k=0; k<4; k++) {
    fprintf(ofs, "%s[%d,%d] <- %25.16e + %25.16e * 1.i\n", name, i+1, k+1, creal(g->m[i][k]), cimag(g->m[i][k]));
  }}
}  /* end of gamma_matrix_printf */

/************************************************/
/************************************************/

void gamma_matrix_mult ( gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3 ) {
  g1->id = g_gamma_mult_table[g2->id][g3->id];
  g1->s  = g2->s * g3->s * g_gamma_mult_sign[g2->id][g3->id];
  // fprintf(stdout, "# [gamma_matrix_mult] id1 %2d id2 %2d id3 %2d\n", g1->id, g2->id, g3->id);
  gamma_matrix_fill (g1);
}  /* gamma_matrix_mult */

/************************************************/
/************************************************/

void gamma_matrix_assign ( gamma_matrix_type *g , gamma_matrix_type *p) {
  memcpy( g->v, p->v, 16*sizeof(double _Complex) );
  g->id = p->id;
  g->s = p->s;
}  /* end of gamma_matrix_assign */

/************************************************/
/************************************************/

void gamma_matrix_transposed (gamma_matrix_type *g, gamma_matrix_type *p) {
  g->id = p->id;
  g->s  = p->s * g_gamma_transposed_sign[p->id];
  gamma_matrix_fill ( g );
}  /* end of gamma_matrix_transposed */

/************************************************/
/************************************************/

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

/************************************************/
/************************************************/

void gamma_matrix_adjoint ( gamma_matrix_type *g, gamma_matrix_type *p) {
  g->id = p->id;
  g->s  = p->s * g_gamma_adjoint_sign[p->id];
  gamma_matrix_fill ( g );
}  /* end of gamma_matrix_adjoint */

/************************************************/
/************************************************/

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

/************************************************/
/************************************************/

void gamma_matrix_eq_gamma_matrix_conj ( gamma_matrix_type *g, gamma_matrix_type *p) {
  g->v[ 0] = conj(p->v[ 0]);
  g->v[ 1] = conj(p->v[ 1]);
  g->v[ 2] = conj(p->v[ 2]);
  g->v[ 3] = conj(p->v[ 3]);
  g->v[ 4] = conj(p->v[ 4]);
  g->v[ 5] = conj(p->v[ 5]);
  g->v[ 6] = conj(p->v[ 6]);
  g->v[ 7] = conj(p->v[ 7]);
  g->v[ 8] = conj(p->v[ 8]);
  g->v[ 9] = conj(p->v[ 9]);
  g->v[10] = conj(p->v[10]);
  g->v[11] = conj(p->v[11]);
  g->v[12] = conj(p->v[12]);
  g->v[13] = conj(p->v[13]);
  g->v[14] = conj(p->v[14]);
  g->v[15] = conj(p->v[15]);
  g->id = -1;
  g->s  = 0;
}  /* end of gamma_matrix_eq_gamma_matrix_adjoint */
/************************************************/
/************************************************/

void gamma_matrix_norm (double *norm, gamma_matrix_type *g) {
  double _norm = 0., _re, _im;
  for ( int i=0; i<16; i++) { 
    _re = creal( g->v[i] );
    _im = cimag( g->v[i] );
    _norm += _re * _re + _im * _im;
  }
  *norm = sqrt( _norm );
}  /* end of gamma_matrix_norm */

/************************************************/
/************************************************/

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

/************************************************/
/************************************************/

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


/********************************************************************************/
/********************************************************************************/

void gamma_matrix_eq_gamma_matrix_pl_gamma_matrix_ti_co (gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3, double _Complex c ) {

  g1->v[ 0] = g2->v[ 0] + g3->v[ 0] * c;
  g1->v[ 1] = g2->v[ 1] + g3->v[ 1] * c;
  g1->v[ 2] = g2->v[ 2] + g3->v[ 2] * c;
  g1->v[ 3] = g2->v[ 3] + g3->v[ 3] * c;
  g1->v[ 4] = g2->v[ 4] + g3->v[ 4] * c;
  g1->v[ 5] = g2->v[ 5] + g3->v[ 5] * c;
  g1->v[ 6] = g2->v[ 6] + g3->v[ 6] * c;
  g1->v[ 7] = g2->v[ 7] + g3->v[ 7] * c;
  g1->v[ 8] = g2->v[ 8] + g3->v[ 8] * c;
  g1->v[ 9] = g2->v[ 9] + g3->v[ 9] * c;
  g1->v[10] = g2->v[10] + g3->v[10] * c;
  g1->v[11] = g2->v[11] + g3->v[11] * c;
  g1->v[12] = g2->v[12] + g3->v[12] * c;
  g1->v[13] = g2->v[13] + g3->v[13] * c;
  g1->v[14] = g2->v[14] + g3->v[14] * c;
  g1->v[15] = g2->v[15] + g3->v[15] * c;
  g1->id = -1;
  g1->s  = 0;
}  /* end of gamma_matrix_eq_gamma_matrix_pl_gamma_matrix_ti_co */

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 * g1 <- g1 * r
 ********************************************************************************/
void gamma_matrix_ti_eq_re ( gamma_matrix_type * const g1, double const r ) {

  g1->v[ 0] *= r;
  g1->v[ 1] *= r;
  g1->v[ 2] *= r;
  g1->v[ 3] *= r;
  g1->v[ 4] *= r;
  g1->v[ 5] *= r;
  g1->v[ 6] *= r;
  g1->v[ 7] *= r;
  g1->v[ 8] *= r;
  g1->v[ 9] *= r;
  g1->v[10] *= r;
  g1->v[11] *= r;
  g1->v[12] *= r;
  g1->v[13] *= r;
  g1->v[14] *= r;
  g1->v[15] *= r;

  g1->id = -1;
  g1->s  = 0;
}  /* end of gamma_matrix_ti_eq_re */

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 * g1 <- g1 * z
 ********************************************************************************/
void gamma_matrix_ti_eq_co ( gamma_matrix_type * const g1, double _Complex const z ) {

  g1->v[ 0] *= z;
  g1->v[ 1] *= z;
  g1->v[ 2] *= z;
  g1->v[ 3] *= z;
  g1->v[ 4] *= z;
  g1->v[ 5] *= z;
  g1->v[ 6] *= z;
  g1->v[ 7] *= z;
  g1->v[ 8] *= z;
  g1->v[ 9] *= z;
  g1->v[10] *= z;
  g1->v[11] *= z;
  g1->v[12] *= z;
  g1->v[13] *= z;
  g1->v[14] *= z;
  g1->v[15] *= z;

  g1->id = -1;
  g1->s  = 0;
}  /* end of gamma_matrix_ti_eq_co */

/********************************************************************************/
/********************************************************************************/

/************************************************
 * g1 = g2 x g3
 * save for g1 = g2 or g2 = g3 or g1 = g3 in memory
 ************************************************/
void gamma_matrix_eq_gamma_matrix_ti_gamma_matrix ( gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3 ) {
  char CHAR_N = 'N';
  int INT_N = 4;
  double _Complex Z_1 = 1., Z_0 = 0.;
  double _Complex v2[16], v3[16];
  memcpy ( v2, g2->v , 16*sizeof(double _Complex ) );
  memcpy ( v3, g3->v , 16*sizeof(double _Complex ) );

  /* _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1); */
  _F(zgemm) ( &CHAR_N, &CHAR_N, &INT_N, &INT_N, &INT_N, &Z_1, v3, &INT_N, v2, &INT_N, &Z_0, g1->v, &INT_N, 1, 1);

  g1->id = -1;
  g1->s  = 0;
}  /* end of gamma_matrix_eq_gamma_matrix_ti_gamma_matrix */


/************************************************/
/************************************************/

/************************************************
 * gamma matrix in qlua gamma basis
 * and in qlua binary counting
 ************************************************/
void gamma_matrix_qlua_binary ( gamma_matrix_type *g, int n  ) {

  gamma_matrix_type gx, gy, gz, gt;
  gamma_matrix_set ( &gt, 0, -1. );
  gamma_matrix_set ( &gx, 1, -1. );
  gamma_matrix_set ( &gy, 2,  1. );
  gamma_matrix_set ( &gz, 3, -1. );

  int gamma_bin[4] = {0,0,0,0};
  int n0 = n;
  gamma_bin[0] = n0 % 2;
  n0 = n0 >> 1;
  gamma_bin[1] = n0 % 2;
  n0 = n0 >> 1;
  gamma_bin[2] = n0 % 2;
  n0 = n0 >> 1;
  gamma_bin[3] = n0 % 2;

  gamma_matrix_set ( g, 4, 1. );

  if ( gamma_bin[0] ) { gamma_matrix_eq_gamma_matrix_ti_gamma_matrix ( g, g, &gx ); }
  if ( gamma_bin[1] ) { gamma_matrix_eq_gamma_matrix_ti_gamma_matrix ( g, g, &gy ); }
  if ( gamma_bin[2] ) { gamma_matrix_eq_gamma_matrix_ti_gamma_matrix ( g, g, &gz ); }
  if ( gamma_bin[3] ) { gamma_matrix_eq_gamma_matrix_ti_gamma_matrix ( g, g, &gt ); }

  g->id = n;
  g->s  = 1;

  return;
}  /* end of gamma_matrix_qlua_binary */

/************************************************/
/************************************************/

/************************************************
 * initialize gamma signs
 ************************************************/
int get_gamma_signs ( char * const mode , int const gid ) {

  static int signs_initialized = 0;

  int *res_ptr = NULL;

  static int res_Ct[16];
  static int res_g0[16];
  static int res_g05[16];
  static int res_g2t[16];
  static int res_g25t[16];
  static int res_g5[16];
  static int res_g13t[16];
  static int res_g5d[16];
  static int res_g13c[16];
  static int res_g05d[16];
  static int res_g0d[16];
  static int res_g25c[16];
  static int res_g2c[16];
  static int res_gd[16];
  static int res_Cc[16];
  static int res_gt[16];


  if ( ( ! signs_initialized ) || ( strcmp ( mode, "init" ) == 0 ) ) {
    // initialize sign arrays

    if ( g_cart_id == 0 ) fprintf ( stdout, "# [get_gamma_signs] initializing gamma signs\n" );

//  Tue Jul 24 16:06:16 2018 
res_Ct[ 0] =  -1;
res_Ct[ 1] =  -1;
res_Ct[ 2] =  -1;
res_Ct[ 3] =  -1;
res_Ct[ 4] =   1;
res_Ct[ 5] =   1;
res_Ct[ 6] =   1;
res_Ct[ 7] =   1;
res_Ct[ 8] =   1;
res_Ct[ 9] =   1;
res_Ct[10] =  -1;
res_Ct[11] =  -1;
res_Ct[12] =  -1;
res_Ct[13] =  -1;
res_Ct[14] =  -1;
res_Ct[15] =  -1;


res_g0[ 0] =   1;
res_g0[ 1] =  -1;
res_g0[ 2] =  -1;
res_g0[ 3] =  -1;
res_g0[ 4] =   1;
res_g0[ 5] =  -1;
res_g0[ 6] =  -1;
res_g0[ 7] =   1;
res_g0[ 8] =   1;
res_g0[ 9] =   1;
res_g0[10] =  -1;
res_g0[11] =  -1;
res_g0[12] =  -1;
res_g0[13] =   1;
res_g0[14] =   1;
res_g0[15] =   1;


res_g05[ 0] =  -1;
res_g05[ 1] =   1;
res_g05[ 2] =   1;
res_g05[ 3] =   1;
res_g05[ 4] =   1;
res_g05[ 5] =  -1;
res_g05[ 6] =   1;
res_g05[ 7] =  -1;
res_g05[ 8] =  -1;
res_g05[ 9] =  -1;
res_g05[10] =  -1;
res_g05[11] =  -1;
res_g05[12] =  -1;
res_g05[13] =   1;
res_g05[14] =   1;
res_g05[15] =   1;


res_g2t[ 0] =  -1;
res_g2t[ 1] =   1;
res_g2t[ 2] =   1;
res_g2t[ 3] =   1;
res_g2t[ 4] =   1;
res_g2t[ 5] =  -1;
res_g2t[ 6] =  -1;
res_g2t[ 7] =   1;
res_g2t[ 8] =   1;
res_g2t[ 9] =   1;
res_g2t[10] =   1;
res_g2t[11] =   1;
res_g2t[12] =   1;
res_g2t[13] =  -1;
res_g2t[14] =  -1;
res_g2t[15] =  -1;


res_g25t[ 0] =   1;
res_g25t[ 1] =  -1;
res_g25t[ 2] =  -1;
res_g25t[ 3] =  -1;
res_g25t[ 4] =   1;
res_g25t[ 5] =  -1;
res_g25t[ 6] =   1;
res_g25t[ 7] =  -1;
res_g25t[ 8] =  -1;
res_g25t[ 9] =  -1;
res_g25t[10] =   1;
res_g25t[11] =   1;
res_g25t[12] =   1;
res_g25t[13] =  -1;
res_g25t[14] =  -1;
res_g25t[15] =  -1;


res_g5[ 0] =  -1;
res_g5[ 1] =  -1;
res_g5[ 2] =  -1;
res_g5[ 3] =  -1;
res_g5[ 4] =   1;
res_g5[ 5] =   1;
res_g5[ 6] =  -1;
res_g5[ 7] =  -1;
res_g5[ 8] =  -1;
res_g5[ 9] =  -1;
res_g5[10] =   1;
res_g5[11] =   1;
res_g5[12] =   1;
res_g5[13] =   1;
res_g5[14] =   1;
res_g5[15] =   1;


res_g13t[ 0] =   1;
res_g13t[ 1] =   1;
res_g13t[ 2] =   1;
res_g13t[ 3] =   1;
res_g13t[ 4] =   1;
res_g13t[ 5] =   1;
res_g13t[ 6] =  -1;
res_g13t[ 7] =  -1;
res_g13t[ 8] =  -1;
res_g13t[ 9] =  -1;
res_g13t[10] =  -1;
res_g13t[11] =  -1;
res_g13t[12] =  -1;
res_g13t[13] =  -1;
res_g13t[14] =  -1;
res_g13t[15] =  -1;


res_g5d[ 0] =  -1;
res_g5d[ 1] =  -1;
res_g5d[ 2] =  -1;
res_g5d[ 3] =  -1;
res_g5d[ 4] =   1;
res_g5d[ 5] =   1;
res_g5d[ 6] =   1;
res_g5d[ 7] =   1;
res_g5d[ 8] =   1;
res_g5d[ 9] =   1;
res_g5d[10] =  -1;
res_g5d[11] =  -1;
res_g5d[12] =  -1;
res_g5d[13] =  -1;
res_g5d[14] =  -1;
res_g5d[15] =  -1;


res_g13c[ 0] =   1;
res_g13c[ 1] =   1;
res_g13c[ 2] =   1;
res_g13c[ 3] =   1;
res_g13c[ 4] =   1;
res_g13c[ 5] =   1;
res_g13c[ 6] =   1;
res_g13c[ 7] =   1;
res_g13c[ 8] =   1;
res_g13c[ 9] =   1;
res_g13c[10] =   1;
res_g13c[11] =   1;
res_g13c[12] =   1;
res_g13c[13] =   1;
res_g13c[14] =   1;
res_g13c[15] =   1;


res_g05d[ 0] =  -1;
res_g05d[ 1] =   1;
res_g05d[ 2] =   1;
res_g05d[ 3] =   1;
res_g05d[ 4] =   1;
res_g05d[ 5] =  -1;
res_g05d[ 6] =  -1;
res_g05d[ 7] =   1;
res_g05d[ 8] =   1;
res_g05d[ 9] =   1;
res_g05d[10] =   1;
res_g05d[11] =   1;
res_g05d[12] =   1;
res_g05d[13] =  -1;
res_g05d[14] =  -1;
res_g05d[15] =  -1;


res_g0d[ 0] =   1;
res_g0d[ 1] =  -1;
res_g0d[ 2] =  -1;
res_g0d[ 3] =  -1;
res_g0d[ 4] =   1;
res_g0d[ 5] =  -1;
res_g0d[ 6] =   1;
res_g0d[ 7] =  -1;
res_g0d[ 8] =  -1;
res_g0d[ 9] =  -1;
res_g0d[10] =   1;
res_g0d[11] =   1;
res_g0d[12] =   1;
res_g0d[13] =  -1;
res_g0d[14] =  -1;
res_g0d[15] =  -1;


res_g25c[ 0] =   1;
res_g25c[ 1] =  -1;
res_g25c[ 2] =  -1;
res_g25c[ 3] =  -1;
res_g25c[ 4] =   1;
res_g25c[ 5] =  -1;
res_g25c[ 6] =  -1;
res_g25c[ 7] =   1;
res_g25c[ 8] =   1;
res_g25c[ 9] =   1;
res_g25c[10] =  -1;
res_g25c[11] =  -1;
res_g25c[12] =  -1;
res_g25c[13] =   1;
res_g25c[14] =   1;
res_g25c[15] =   1;


res_g2c[ 0] =  -1;
res_g2c[ 1] =   1;
res_g2c[ 2] =   1;
res_g2c[ 3] =   1;
res_g2c[ 4] =   1;
res_g2c[ 5] =  -1;
res_g2c[ 6] =   1;
res_g2c[ 7] =  -1;
res_g2c[ 8] =  -1;
res_g2c[ 9] =  -1;
res_g2c[10] =  -1;
res_g2c[11] =  -1;
res_g2c[12] =  -1;
res_g2c[13] =   1;
res_g2c[14] =   1;
res_g2c[15] =   1;


res_gd[ 0] =   1;
res_gd[ 1] =   1;
res_gd[ 2] =   1;
res_gd[ 3] =   1;
res_gd[ 4] =   1;
res_gd[ 5] =   1;
res_gd[ 6] =  -1;
res_gd[ 7] =  -1;
res_gd[ 8] =  -1;
res_gd[ 9] =  -1;
res_gd[10] =  -1;
res_gd[11] =  -1;
res_gd[12] =  -1;
res_gd[13] =  -1;
res_gd[14] =  -1;
res_gd[15] =  -1;


res_Cc[ 0] =  -1;
res_Cc[ 1] =  -1;
res_Cc[ 2] =  -1;
res_Cc[ 3] =  -1;
res_Cc[ 4] =   1;
res_Cc[ 5] =   1;
res_Cc[ 6] =  -1;
res_Cc[ 7] =  -1;
res_Cc[ 8] =  -1;
res_Cc[ 9] =  -1;
res_Cc[10] =   1;
res_Cc[11] =   1;
res_Cc[12] =   1;
res_Cc[13] =   1;
res_Cc[14] =   1;
res_Cc[15] =   1;


res_gt[ 0] =   1;
res_gt[ 1] =  -1;
res_gt[ 2] =   1;
res_gt[ 3] =  -1;
res_gt[ 4] =   1;
res_gt[ 5] =   1;
res_gt[ 6] =  -1;
res_gt[ 7] =   1;
res_gt[ 8] =  -1;
res_gt[ 9] =   1;
res_gt[10] =   1;
res_gt[11] =  -1;
res_gt[12] =   1;
res_gt[13] =   1;
res_gt[14] =  -1;
res_gt[15] =   1;

    signs_initialized = 1;


  // } else if ( signs_initialized || ( strcmp ( mode, "fini" ) == 0 ) ) {
  //   // deallocate
  }

  if      ( strcmp( mode, "Ct"   ) == 0 ) res_ptr = res_Ct;
  else if ( strcmp( mode, "g0"   ) == 0 ) res_ptr = res_g0;
  else if ( strcmp( mode, "g05"  ) == 0 ) res_ptr = res_g05;
  else if ( strcmp( mode, "g2t"  ) == 0 ) res_ptr = res_g2t;
  else if ( strcmp( mode, "g25t" ) == 0 ) res_ptr = res_g25t;
  else if ( strcmp( mode, "g5"   ) == 0 ) res_ptr = res_g5;
  else if ( strcmp( mode, "g13t" ) == 0 ) res_ptr = res_g13t;
  else if ( strcmp( mode, "g5d"  ) == 0 ) res_ptr = res_g5d;
  else if ( strcmp( mode, "g13c" ) == 0 ) res_ptr = res_g13c;
  else if ( strcmp( mode, "g05d" ) == 0 ) res_ptr = res_g05d;
  else if ( strcmp( mode, "g0d"  ) == 0 ) res_ptr = res_g0d;
  else if ( strcmp( mode, "g25c" ) == 0 ) res_ptr = res_g25c;
  else if ( strcmp( mode, "g2c"  ) == 0 ) res_ptr = res_g2c;
  else if ( strcmp( mode, "gd"   ) == 0 ) res_ptr = res_gd;
  else if ( strcmp( mode, "Cc"   ) == 0 ) res_ptr = res_Cc;
  else if ( strcmp( mode, "gt"   ) == 0 ) res_ptr = res_Cc;

  if ( ( gid >= 0 ) && ( gid < 16 ) ) {
    /* gamma id within range of basis elements */
    return ( res_ptr[gid] );
  } else {
    /* outside basis, return 0 */
    return ( 0 );
  }


}  // end of get_gamma_signs

/************************************************
 * gout = gop1^OP1 x gin x gop2^OP2
 * save for gout = gin in memory
 ************************************************/
void gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( gamma_matrix_type *gout, gamma_matrix_type *gop1, char op1, gamma_matrix_type *gin, gamma_matrix_type *gop2, char op2 ) {
  char CHAR_N = 'N';
  int INT_N = 4;
  double _Complex Z_1 = 1., Z_0 = 0.;
  double _Complex v[16];


  /************************************************
   * check, whether gin has been initialized to
   * one of the 16 basis elements
   *
   * if not, i.e. if gin id is -1, 
   * then return by default with gout id -1
   * and gout norm factor 1.
   ************************************************/
  if ( gin->id == -1 ) {
    gout->id = -1;
    gout->s  =  1.;
    fprintf ( stdout, "# [gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op] Warning, input gamma gin is not set\n" );
    return;
  }

  gamma_matrix_type gl, gr;
  
  gamma_matrix_init ( &gl );
  gamma_matrix_init ( &gr );

  if ( op1 == 'N' ) {
    gamma_matrix_assign ( &gl, gop1 );
  } else if ( op1 == 'T' ) {
    gamma_matrix_eq_gamma_matrix_transposed ( &gl, gop1 );
  } else if ( op1 == 'H' ) {
    gamma_matrix_eq_gamma_matrix_adjoint ( &gl, gop1 );
  } else if ( op1 == 'C' ) {
    gamma_matrix_eq_gamma_matrix_conj ( &gl, gop1 );
  }

  if ( op2 == 'N' ) {
    gamma_matrix_assign ( &gr, gop2 );
  } else if ( op2 == 'T' ) {
    gamma_matrix_eq_gamma_matrix_transposed ( &gr, gop2 );
  } else if ( op2 == 'H' ) {
    gamma_matrix_eq_gamma_matrix_adjoint ( &gr, gop2 );
  } else if ( op2 == 'C' ) {
    gamma_matrix_eq_gamma_matrix_conj ( &gr, gop2 );
  }

  /* _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1); */
  _F(zgemm) ( &CHAR_N, &CHAR_N, &INT_N, &INT_N, &INT_N, &Z_1, gr.v, &INT_N, gin->v, &INT_N, &Z_0, v,       &INT_N, 1, 1);
  _F(zgemm) ( &CHAR_N, &CHAR_N, &INT_N, &INT_N, &INT_N, &Z_1, v,    &INT_N, gl.v,   &INT_N, &Z_0, gout->v, &INT_N, 1, 1);

  gamma_matrix_get_id_sign ( &(gout->id), &(gout->s), gout );

  return;

}  // end of gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op

/************************************************/
/************************************************/

/************************************************
 *
 ************************************************/
void gamma_matrix_get_id_sign ( int * id, double * s, gamma_matrix_type * g ) {

  double const eps = 1.e-14;
  int matches = 0;

  for( int i = 0; i < 16; i++ ) {

    gamma_matrix_type h;

    gamma_matrix_set ( &h, i, 1 );

    gamma_matrix_adjoint ( &h, &h );

    gamma_matrix_eq_gamma_matrix_ti_gamma_matrix ( &h, &h, g );

    double _Complex const z = ( h.m[0][0] + h.m[1][1] + h.m[2][2] + h.m[3][3] ) / 4.;
 
    // fprintf ( stdout, "# [gamma_matrix_get_id_sign] z = %f + I %f\n", creal(z), cimag(z) );
    if ( cabs (z) > eps ) {
      *id = i;
      if ( cabs( cimag(z) ) > eps ) {
        fprintf ( stderr, "[gamma_matrix_get_id_sign] Error, coefficient has imaginary part\n" );
        *id = -1;
        *s = 0;
        return;
      }
      *s  = creal( z );
      matches++;
    }

  }  // end of loop on gamma matrix base

  if ( matches == 0 ||  matches > 1 ) {
    fprintf ( stderr, "[gamma_matrix_get_id_sign] number of matches not 1 \n" );
    *id = -1;
    *s = 0;
  }
  return;
}  // end of gamma_matrix_get_id_sign

}  // end of namespace cvc
