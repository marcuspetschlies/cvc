#ifndef ZM4X4_H
#define ZM4X4_H


namespace cvc {

static inline void zm_eq_zm_transposed_4x4_array (double _Complex *r, double _Complex *s ) {
  double _Complex b[16];
  memcpy( b, s, 16*sizeof(double _Complex));
  r[ 0] = b[ 0];
  r[ 1] = b[ 4];
  r[ 2] = b[ 8];
  r[ 3] = b[12];
  r[ 4] = b[ 1];
  r[ 5] = b[ 5];
  r[ 6] = b[ 9];
  r[ 7] = b[13];
  r[ 8] = b[ 2];
  r[ 9] = b[ 6];
  r[10] = b[10];
  r[11] = b[14];
  r[12] = b[ 3];
  r[13] = b[ 7];
  r[14] = b[11];
  r[15] = b[15];
}  /* end of zm_eq_zm_transposed_4x4_array */

/***********************************************************/
/***********************************************************/

static inline void zm_pl_eq_zm_transposed_4x4_array (double _Complex *r, double _Complex *s ) {
  double _Complex b[16];
  memcpy( b, s, 16*sizeof(double _Complex));
  r[ 0] += b[ 0];
  r[ 1] += b[ 4];
  r[ 2] += b[ 8];
  r[ 3] += b[12];
  r[ 4] += b[ 1];
  r[ 5] += b[ 5];
  r[ 6] += b[ 9];
  r[ 7] += b[13];
  r[ 8] += b[ 2];
  r[ 9] += b[ 6];
  r[10] += b[10];
  r[11] += b[14];
  r[12] += b[ 3];
  r[13] += b[ 7];
  r[14] += b[11];
  r[15] += b[15];
}  /* end of zm_pl_eq_zm_transposed_4x4_array */


/***********************************************************/
/***********************************************************/

static inline int init_zm4x4 ( double _Complex ***z ){
  const size_t bytes = 16*sizeof(double _Complex );
  if(*z != NULL) { return(1); }

  /* 1st, outer level */
  (*z) = (double _Complex**)malloc(4 * sizeof(double _Complex*));
  if( *z == NULL ) { return(2); }

  /* 2nd, inner level */
  (*z)[0] = (double _Complex*)malloc( bytes );
  if( (*z)[0] == NULL ) { return(3); }
  (*z)[1] = (*z)[0] + 4;
  (*z)[2] = (*z)[1] + 4;
  (*z)[3] = (*z)[2] + 4;
  memset( (*z)[0], 0, bytes);
  return(0);
}  /* end of init_zm4x4 */

static inline int fini_zm4x4 (double _Complex***z) {
  if(*z!= NULL) {
    if( (*z)[0] != NULL) { free( (*z)[0] ); }
    free( *z);
    *z = NULL;
  }
  return(0);
}  /* end of fini_zm4x4 */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_eq_zm4x4 (double _Complex **r, double _Complex **s ) {
  memcpy( r[0], s[0], 16*sizeof(double _Complex) );
}  /* end of zm4x4_eq_zm4x4 */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_eq_zm4x4_transposed (double _Complex **r, double _Complex **s ) {
  double _Complex b[16];
  memcpy( b, s[0], 16*sizeof(double _Complex) );
  r[0][0] = b[ 0];
  r[0][1] = b[ 4];
  r[0][2] = b[ 8];
  r[0][3] = b[12];
  r[1][0] = b[ 1];
  r[1][1] = b[ 5];
  r[1][2] = b[ 9];
  r[1][3] = b[13];
  r[2][0] = b[ 2];
  r[2][1] = b[ 6];
  r[2][2] = b[10];
  r[2][3] = b[14];
  r[3][0] = b[ 3];
  r[3][1] = b[ 7];
  r[3][2] = b[11];
  r[3][3] = b[15];
}  /* end of zm4x4_eq_zm4x4_transposed */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_eq_zm4x4_adjoint (double _Complex **r, double _Complex **s ) {
  double _Complex b[16];
  memcpy( b, s[0], 16*sizeof(double _Complex) );
  r[0][0] = conj(b[ 0]);
  r[0][1] = conj(b[ 4]);
  r[0][2] = conj(b[ 8]);
  r[0][3] = conj(b[12]);
  r[1][0] = conj(b[ 1]);
  r[1][1] = conj(b[ 5]);
  r[1][2] = conj(b[ 9]);
  r[1][3] = conj(b[13]);
  r[2][0] = conj(b[ 2]);
  r[2][1] = conj(b[ 6]);
  r[2][2] = conj(b[10]);
  r[2][3] = conj(b[14]);
  r[3][0] = conj(b[ 3]);
  r[3][1] = conj(b[ 7]);
  r[3][2] = conj(b[11]);
  r[3][3] = conj(b[15]);
}  /* end of zm4x4_eq_zm4x4_adjoint */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_pl_eq_zm4x4_transposed (double _Complex **r, double _Complex **s ) {
  double _Complex buffer[16];
  memcpy( buffer, s[0], 16*sizeof(double _Complex) );
  r[0][0] += buffer[ 0];
  r[0][1] += buffer[ 4];
  r[0][2] += buffer[ 8];
  r[0][3] += buffer[12];
  r[1][0] += buffer[ 1];
  r[1][1] += buffer[ 5];
  r[1][2] += buffer[ 9];
  r[1][3] += buffer[13];
  r[2][0] += buffer[ 2];
  r[2][1] += buffer[ 6];
  r[2][2] += buffer[10];
  r[2][3] += buffer[14];
  r[3][0] += buffer[ 3];
  r[3][1] += buffer[ 7];
  r[3][2] += buffer[11];
  r[3][3] += buffer[15];
}  /* end of zm4x4_pl_eq_zm4x4_transposed */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_pl_eq_zm4x4 (double _Complex **r, double _Complex **s ) {
  r[0][ 0] += s[0][ 0];
  r[0][ 1] += s[0][ 1];
  r[0][ 2] += s[0][ 2];
  r[0][ 3] += s[0][ 3];
  r[0][ 4] += s[0][ 4];
  r[0][ 5] += s[0][ 5];
  r[0][ 6] += s[0][ 6];
  r[0][ 7] += s[0][ 7];
  r[0][ 8] += s[0][ 8];
  r[0][ 9] += s[0][ 9];
  r[0][10] += s[0][10];
  r[0][11] += s[0][11];
  r[0][12] += s[0][12];
  r[0][13] += s[0][13];
  r[0][14] += s[0][14];
  r[0][15] += s[0][15];
}  /* end of zm4x4_pl_eq_zm4x4 */

/***********************************************************/
/***********************************************************/
static inline void zm4x4_ti_eq_re (double _Complex **r, double c ) {
  r[0][ 0] *= c;
  r[0][ 1] *= c;
  r[0][ 2] *= c;
  r[0][ 3] *= c;
  r[0][ 4] *= c;
  r[0][ 5] *= c;
  r[0][ 6] *= c;
  r[0][ 7] *= c;
  r[0][ 8] *= c;
  r[0][ 9] *= c;
  r[0][10] *= c;
  r[0][11] *= c;
  r[0][12] *= c;
  r[0][13] *= c;
  r[0][14] *= c;
  r[0][15] *= c;
}  /* end of zm4x4_ti_eq_re */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_ti_eq_co (double _Complex **r, double _Complex c ) {
  r[0][ 0] *= c;
  r[0][ 1] *= c;
  r[0][ 2] *= c;
  r[0][ 3] *= c;
  r[0][ 4] *= c;
  r[0][ 5] *= c;
  r[0][ 6] *= c;
  r[0][ 7] *= c;
  r[0][ 8] *= c;
  r[0][ 9] *= c;
  r[0][10] *= c;
  r[0][11] *= c;
  r[0][12] *= c;
  r[0][13] *= c;
  r[0][14] *= c;
  r[0][15] *= c;
}  /* end of zm4x4_ti_eq_co */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_eq_zm4x4_ti_zm4x4 (double _Complex **r, double _Complex **s, double _Complex **t ) {
  double _Complex ss[16], tt[16];
  memcpy( ss, s[0], 16*sizeof(double _Complex) );
  memcpy( tt, t[0], 16*sizeof(double _Complex) );
  r[0][ 0] = ss[ 0] * tt[ 0] + ss[ 1] * tt[ 4] + ss[ 2] * tt[ 8] + ss[ 3] * tt[12];
  r[0][ 1] = ss[ 0] * tt[ 1] + ss[ 1] * tt[ 5] + ss[ 2] * tt[ 9] + ss[ 3] * tt[13];
  r[0][ 2] = ss[ 0] * tt[ 2] + ss[ 1] * tt[ 6] + ss[ 2] * tt[10] + ss[ 3] * tt[14];
  r[0][ 3] = ss[ 0] * tt[ 3] + ss[ 1] * tt[ 7] + ss[ 2] * tt[11] + ss[ 3] * tt[15];
  r[0][ 4] = ss[ 4] * tt[ 0] + ss[ 5] * tt[ 4] + ss[ 6] * tt[ 8] + ss[ 7] * tt[12];
  r[0][ 5] = ss[ 4] * tt[ 1] + ss[ 5] * tt[ 5] + ss[ 6] * tt[ 9] + ss[ 7] * tt[13];
  r[0][ 6] = ss[ 4] * tt[ 2] + ss[ 5] * tt[ 6] + ss[ 6] * tt[10] + ss[ 7] * tt[14];
  r[0][ 7] = ss[ 4] * tt[ 3] + ss[ 5] * tt[ 7] + ss[ 6] * tt[11] + ss[ 7] * tt[15];
  r[0][ 8] = ss[ 8] * tt[ 0] + ss[ 9] * tt[ 4] + ss[10] * tt[ 8] + ss[11] * tt[12];
  r[0][ 9] = ss[ 8] * tt[ 1] + ss[ 9] * tt[ 5] + ss[10] * tt[ 9] + ss[11] * tt[13];
  r[0][10] = ss[ 8] * tt[ 2] + ss[ 9] * tt[ 6] + ss[10] * tt[10] + ss[11] * tt[14];
  r[0][11] = ss[ 8] * tt[ 3] + ss[ 9] * tt[ 7] + ss[10] * tt[11] + ss[11] * tt[15];
  r[0][12] = ss[12] * tt[ 0] + ss[13] * tt[ 4] + ss[14] * tt[ 8] + ss[15] * tt[12];
  r[0][13] = ss[12] * tt[ 1] + ss[13] * tt[ 5] + ss[14] * tt[ 9] + ss[15] * tt[13];
  r[0][14] = ss[12] * tt[ 2] + ss[13] * tt[ 6] + ss[14] * tt[10] + ss[15] * tt[14];
  r[0][15] = ss[12] * tt[ 3] + ss[13] * tt[ 7] + ss[14] * tt[11] + ss[15] * tt[15];
}  /* end of zm4x4_eq_zm4x4_ti_zm4x4 */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_eq_zm4x4_ti_zm4x4_pl_zm4x4_ti_co (double _Complex **r, double _Complex **s, double _Complex **t, double _Complex **u, double _Complex c ) {
  double _Complex ss[16], tt[16];
  memcpy( ss, s[0], 16*sizeof(double _Complex) );
  memcpy( tt, t[0], 16*sizeof(double _Complex) );
  r[0][ 0] = ss[ 0] * tt[ 0]  + ss[ 1] * tt[ 4]  + ss[ 2] * tt[ 8]  + ss[ 3] * tt[12] + u[0][ 0] * c;
  r[0][ 1] = ss[ 0] * tt[ 1]  + ss[ 1] * tt[ 5]  + ss[ 2] * tt[ 9]  + ss[ 3] * tt[13] + u[0][ 1] * c;
  r[0][ 2] = ss[ 0] * tt[ 2]  + ss[ 1] * tt[ 6]  + ss[ 2] * tt[10]  + ss[ 3] * tt[14] + u[0][ 2] * c;
  r[0][ 3] = ss[ 0] * tt[ 3]  + ss[ 1] * tt[ 7]  + ss[ 2] * tt[11]  + ss[ 3] * tt[15] + u[0][ 3] * c;
  r[0][ 4] = ss[ 4] * tt[ 0]  + ss[ 5] * tt[ 4]  + ss[ 6] * tt[ 8]  + ss[ 7] * tt[12] + u[0][ 4] * c;
  r[0][ 5] = ss[ 4] * tt[ 1]  + ss[ 5] * tt[ 5]  + ss[ 6] * tt[ 9]  + ss[ 7] * tt[13] + u[0][ 5] * c;
  r[0][ 6] = ss[ 4] * tt[ 2]  + ss[ 5] * tt[ 6]  + ss[ 6] * tt[10]  + ss[ 7] * tt[14] + u[0][ 6] * c;
  r[0][ 7] = ss[ 4] * tt[ 3]  + ss[ 5] * tt[ 7]  + ss[ 6] * tt[11]  + ss[ 7] * tt[15] + u[0][ 7] * c;
  r[0][ 8] = ss[ 8] * tt[ 0]  + ss[ 9] * tt[ 4]  + ss[10] * tt[ 8]  + ss[11] * tt[12] + u[0][ 8] * c;
  r[0][ 9] = ss[ 8] * tt[ 1]  + ss[ 9] * tt[ 5]  + ss[10] * tt[ 9]  + ss[11] * tt[13] + u[0][ 9] * c;
  r[0][10] = ss[ 8] * tt[ 2]  + ss[ 9] * tt[ 6]  + ss[10] * tt[10]  + ss[11] * tt[14] + u[0][10] * c;
  r[0][11] = ss[ 8] * tt[ 3]  + ss[ 9] * tt[ 7]  + ss[10] * tt[11]  + ss[11] * tt[15] + u[0][11] * c;
  r[0][12] = ss[12] * tt[ 0]  + ss[13] * tt[ 4]  + ss[14] * tt[ 8]  + ss[15] * tt[12] + u[0][12] * c;
  r[0][13] = ss[12] * tt[ 1]  + ss[13] * tt[ 5]  + ss[14] * tt[ 9]  + ss[15] * tt[13] + u[0][13] * c;
  r[0][14] = ss[12] * tt[ 2]  + ss[13] * tt[ 6]  + ss[14] * tt[10]  + ss[15] * tt[14] + u[0][14] * c;
  r[0][15] = ss[12] * tt[ 3]  + ss[13] * tt[ 7]  + ss[14] * tt[11]  + ss[15] * tt[15] + u[0][15] * c;
}  /* end of zm4x4_eq_zm4x4_ti_zm4x4_pl_zm4x4_ti_co */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_eq_zm4x4_pl_zm4x4 (double _Complex **r, double _Complex **s, double _Complex **t ) {
  r[0][ 0] = s[0][ 0] + t[0][ 0];
  r[0][ 1] = s[0][ 1] + t[0][ 1];
  r[0][ 2] = s[0][ 2] + t[0][ 2];
  r[0][ 3] = s[0][ 3] + t[0][ 3];
  r[0][ 4] = s[0][ 4] + t[0][ 4];
  r[0][ 5] = s[0][ 5] + t[0][ 5];
  r[0][ 6] = s[0][ 6] + t[0][ 6];
  r[0][ 7] = s[0][ 7] + t[0][ 7];
  r[0][ 8] = s[0][ 8] + t[0][ 8];
  r[0][ 9] = s[0][ 9] + t[0][ 9];
  r[0][10] = s[0][10] + t[0][10];
  r[0][11] = s[0][11] + t[0][11];
  r[0][12] = s[0][12] + t[0][12];
  r[0][13] = s[0][13] + t[0][13];
  r[0][14] = s[0][14] + t[0][14];
  r[0][15] = s[0][15] + t[0][15];
}  /* end of zm4x4_eq_zm4x4_pl_zm4x4 */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_eq_zm4x4_pl_zm4x4_ti_re (double _Complex **r, double _Complex **s, double _Complex **t, double c ) {
  r[0][ 0] = s[0][ 0] + t[0][ 0] * c;
  r[0][ 1] = s[0][ 1] + t[0][ 1] * c;
  r[0][ 2] = s[0][ 2] + t[0][ 2] * c;
  r[0][ 3] = s[0][ 3] + t[0][ 3] * c;
  r[0][ 4] = s[0][ 4] + t[0][ 4] * c;
  r[0][ 5] = s[0][ 5] + t[0][ 5] * c;
  r[0][ 6] = s[0][ 6] + t[0][ 6] * c;
  r[0][ 7] = s[0][ 7] + t[0][ 7] * c;
  r[0][ 8] = s[0][ 8] + t[0][ 8] * c;
  r[0][ 9] = s[0][ 9] + t[0][ 9] * c;
  r[0][10] = s[0][10] + t[0][10] * c;
  r[0][11] = s[0][11] + t[0][11] * c;
  r[0][12] = s[0][12] + t[0][12] * c;
  r[0][13] = s[0][13] + t[0][13] * c;
  r[0][14] = s[0][14] + t[0][14] * c;
  r[0][15] = s[0][15] + t[0][15] * c;
}  /* end of zm4x4_eq_zm4x4_pl_zm4x4_ti_re */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_eq_zm4x4_ti_co (double _Complex **r, double _Complex **s, double _Complex c ) {
  r[0][ 0] = s[0][ 0] * c;
  r[0][ 1] = s[0][ 1] * c;
  r[0][ 2] = s[0][ 2] * c;
  r[0][ 3] = s[0][ 3] * c;
  r[0][ 4] = s[0][ 4] * c;
  r[0][ 5] = s[0][ 5] * c;
  r[0][ 6] = s[0][ 6] * c;
  r[0][ 7] = s[0][ 7] * c;
  r[0][ 8] = s[0][ 8] * c;
  r[0][ 9] = s[0][ 9] * c;
  r[0][10] = s[0][10] * c;
  r[0][11] = s[0][11] * c;
  r[0][12] = s[0][12] * c;
  r[0][13] = s[0][13] * c;
  r[0][14] = s[0][14] * c;
  r[0][15] = s[0][15] * c;
}   /* end of zm4x4_eq_zm4x4_ti_co */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_eq_zm4x4_pl_zm4x4_ti_co (double _Complex **r, double _Complex **s, double _Complex **t, double _Complex c ) {
  r[0][ 0] = s[0][ 0] + t[0][ 0] * c;
  r[0][ 1] = s[0][ 1] + t[0][ 1] * c;
  r[0][ 2] = s[0][ 2] + t[0][ 2] * c;
  r[0][ 3] = s[0][ 3] + t[0][ 3] * c;
  r[0][ 4] = s[0][ 4] + t[0][ 4] * c;
  r[0][ 5] = s[0][ 5] + t[0][ 5] * c;
  r[0][ 6] = s[0][ 6] + t[0][ 6] * c;
  r[0][ 7] = s[0][ 7] + t[0][ 7] * c;
  r[0][ 8] = s[0][ 8] + t[0][ 8] * c;
  r[0][ 9] = s[0][ 9] + t[0][ 9] * c;
  r[0][10] = s[0][10] + t[0][10] * c;
  r[0][11] = s[0][11] + t[0][11] * c;
  r[0][12] = s[0][12] + t[0][12] * c;
  r[0][13] = s[0][13] + t[0][13] * c;
  r[0][14] = s[0][14] + t[0][14] * c;
  r[0][15] = s[0][15] + t[0][15] * c;
}  /* end of zm4x4_eq_zm4x4_pl_zm4x4_ti_co */

/***********************************************************/
/***********************************************************/

static inline void co_eq_tr_zm4x4 (double _Complex *c, double _Complex **r ) {
  *c = r[0][0] + r[1][1] + r[2][2] + r[3][3];
}

/***********************************************************/
/***********************************************************/

static inline void zm4x4_eq_spin_parity_projection_zm4x4 (double _Complex **r, double _Complex **s, double c) {
  const double norm = 1. / ( 1. + fabs(c) );
  double _Complex b[16];
  memcpy( b, s[0], 16*sizeof(double _Complex ) );
  r[0][ 0] = ( b[ 0] - b[ 8] * c ) * norm;
  r[0][ 1] = ( b[ 1] - b[ 9] * c ) * norm;
  r[0][ 2] = ( b[ 2] - b[10] * c ) * norm;
  r[0][ 3] = ( b[ 3] - b[11] * c ) * norm;
  r[0][ 4] = ( b[ 4] - b[12] * c ) * norm;
  r[0][ 5] = ( b[ 5] - b[13] * c ) * norm;
  r[0][ 6] = ( b[ 6] - b[14] * c ) * norm;
  r[0][ 7] = ( b[ 7] - b[15] * c ) * norm;
  r[0][ 8] = ( b[ 8] - b[ 0] * c ) * norm;
  r[0][ 9] = ( b[ 9] - b[ 1] * c ) * norm;
  r[0][10] = ( b[10] - b[ 2] * c ) * norm;
  r[0][11] = ( b[11] - b[ 3] * c ) * norm;
  r[0][12] = ( b[12] - b[ 4] * c ) * norm;
  r[0][13] = ( b[13] - b[ 5] * c ) * norm;
  r[0][14] = ( b[14] - b[ 6] * c ) * norm;
  r[0][15] = ( b[15] - b[ 7] * c ) * norm;
}  /* end of zm4x4_eq_spin_parity_projection_zm4x4 */


/***********************************************************/
/***********************************************************/


/* [spin_projection] Fri Jul 14 16:05:32 2017*/


static inline void zm4x4_eq_spin_projection_zm4x4_11 (double _Complex **r, double _Complex **s, double a, double b ) { 
  double _Complex ss[16];
  memcpy(ss, s[0], 16*sizeof(double _Complex));
  double c = b * 0.3333333333333333333333333;
  r[0][0] =  a * ss[0] +     c * ss[0];
  r[0][1] =  a * ss[1] +     c * ss[1];
  r[0][2] =  a * ss[2] +     c * ss[2];
  r[0][3] =  a * ss[3] +     c * ss[3];
  r[0][4] =  a * ss[4] +     c * ss[4];
  r[0][5] =  a * ss[5] +     c * ss[5];
  r[0][6] =  a * ss[6] +     c * ss[6];
  r[0][7] =  a * ss[7] +     c * ss[7];
  r[0][8] =  a * ss[8] +     c * ss[8];
  r[0][9] =  a * ss[9] +     c * ss[9];
  r[0][10] =  a * ss[10] +     c * ss[10];
  r[0][11] =  a * ss[11] +     c * ss[11];
  r[0][12] =  a * ss[12] +     c * ss[12];
  r[0][13] =  a * ss[13] +     c * ss[13];
  r[0][14] =  a * ss[14] +     c * ss[14];
  r[0][15] =  a * ss[15] +     c * ss[15];
} /* end of zm4x4_eq_spin_projection_zm4x4_11 */


static inline void zm4x4_eq_spin_projection_zm4x4_12 (double _Complex **r, double _Complex **s, double a, double b ) { 
  double _Complex ss[16];
  memcpy(ss, s[0], 16*sizeof(double _Complex));
  double c = b * 0.3333333333333333333333333;
  r[0][0] =  a * ss[0] + I * c * ss[0];
  r[0][1] =  a * ss[1] + I * c * ss[1];
  r[0][2] =  a * ss[2] + I * c * ss[2];
  r[0][3] =  a * ss[3] + I * c * ss[3];
  r[0][4] =  a * ss[4] - I * c * ss[4];
  r[0][5] =  a * ss[5] - I * c * ss[5];
  r[0][6] =  a * ss[6] - I * c * ss[6];
  r[0][7] =  a * ss[7] - I * c * ss[7];
  r[0][8] =  a * ss[8] + I * c * ss[8];
  r[0][9] =  a * ss[9] + I * c * ss[9];
  r[0][10] =  a * ss[10] + I * c * ss[10];
  r[0][11] =  a * ss[11] + I * c * ss[11];
  r[0][12] =  a * ss[12] - I * c * ss[12];
  r[0][13] =  a * ss[13] - I * c * ss[13];
  r[0][14] =  a * ss[14] - I * c * ss[14];
  r[0][15] =  a * ss[15] - I * c * ss[15];
} /* end of zm4x4_eq_spin_projection_zm4x4_12 */


static inline void zm4x4_eq_spin_projection_zm4x4_13 (double _Complex **r, double _Complex **s, double a, double b ) { 
  double _Complex ss[16];
  memcpy(ss, s[0], 16*sizeof(double _Complex));
  double c = b * 0.3333333333333333333333333;
  r[0][0] =  a * ss[0] -     c * ss[4];
  r[0][1] =  a * ss[1] -     c * ss[5];
  r[0][2] =  a * ss[2] -     c * ss[6];
  r[0][3] =  a * ss[3] -     c * ss[7];
  r[0][4] =  a * ss[4] +     c * ss[0];
  r[0][5] =  a * ss[5] +     c * ss[1];
  r[0][6] =  a * ss[6] +     c * ss[2];
  r[0][7] =  a * ss[7] +     c * ss[3];
  r[0][8] =  a * ss[8] -     c * ss[12];
  r[0][9] =  a * ss[9] -     c * ss[13];
  r[0][10] =  a * ss[10] -     c * ss[14];
  r[0][11] =  a * ss[11] -     c * ss[15];
  r[0][12] =  a * ss[12] +     c * ss[8];
  r[0][13] =  a * ss[13] +     c * ss[9];
  r[0][14] =  a * ss[14] +     c * ss[10];
  r[0][15] =  a * ss[15] +     c * ss[11];
} /* end of zm4x4_eq_spin_projection_zm4x4_13 */


static inline void zm4x4_eq_spin_projection_zm4x4_21 (double _Complex **r, double _Complex **s, double a, double b ) { 
  double _Complex ss[16];
  memcpy(ss, s[0], 16*sizeof(double _Complex));
  double c = b * 0.3333333333333333333333333;
  r[0][0] =  a * ss[0] - I * c * ss[0];
  r[0][1] =  a * ss[1] - I * c * ss[1];
  r[0][2] =  a * ss[2] - I * c * ss[2];
  r[0][3] =  a * ss[3] - I * c * ss[3];
  r[0][4] =  a * ss[4] + I * c * ss[4];
  r[0][5] =  a * ss[5] + I * c * ss[5];
  r[0][6] =  a * ss[6] + I * c * ss[6];
  r[0][7] =  a * ss[7] + I * c * ss[7];
  r[0][8] =  a * ss[8] - I * c * ss[8];
  r[0][9] =  a * ss[9] - I * c * ss[9];
  r[0][10] =  a * ss[10] - I * c * ss[10];
  r[0][11] =  a * ss[11] - I * c * ss[11];
  r[0][12] =  a * ss[12] + I * c * ss[12];
  r[0][13] =  a * ss[13] + I * c * ss[13];
  r[0][14] =  a * ss[14] + I * c * ss[14];
  r[0][15] =  a * ss[15] + I * c * ss[15];
} /* end of zm4x4_eq_spin_projection_zm4x4_21 */


static inline void zm4x4_eq_spin_projection_zm4x4_22 (double _Complex **r, double _Complex **s, double a, double b ) { 
  double _Complex ss[16];
  memcpy(ss, s[0], 16*sizeof(double _Complex));
  double c = b * 0.3333333333333333333333333;
  r[0][0] =  a * ss[0] +     c * ss[0];
  r[0][1] =  a * ss[1] +     c * ss[1];
  r[0][2] =  a * ss[2] +     c * ss[2];
  r[0][3] =  a * ss[3] +     c * ss[3];
  r[0][4] =  a * ss[4] +     c * ss[4];
  r[0][5] =  a * ss[5] +     c * ss[5];
  r[0][6] =  a * ss[6] +     c * ss[6];
  r[0][7] =  a * ss[7] +     c * ss[7];
  r[0][8] =  a * ss[8] +     c * ss[8];
  r[0][9] =  a * ss[9] +     c * ss[9];
  r[0][10] =  a * ss[10] +     c * ss[10];
  r[0][11] =  a * ss[11] +     c * ss[11];
  r[0][12] =  a * ss[12] +     c * ss[12];
  r[0][13] =  a * ss[13] +     c * ss[13];
  r[0][14] =  a * ss[14] +     c * ss[14];
  r[0][15] =  a * ss[15] +     c * ss[15];
} /* end of zm4x4_eq_spin_projection_zm4x4_22 */


static inline void zm4x4_eq_spin_projection_zm4x4_23 (double _Complex **r, double _Complex **s, double a, double b ) { 
  double _Complex ss[16];
  memcpy(ss, s[0], 16*sizeof(double _Complex));
  double c = b * 0.3333333333333333333333333;
  r[0][0] =  a * ss[0] + I * c * ss[4];
  r[0][1] =  a * ss[1] + I * c * ss[5];
  r[0][2] =  a * ss[2] + I * c * ss[6];
  r[0][3] =  a * ss[3] + I * c * ss[7];
  r[0][4] =  a * ss[4] + I * c * ss[0];
  r[0][5] =  a * ss[5] + I * c * ss[1];
  r[0][6] =  a * ss[6] + I * c * ss[2];
  r[0][7] =  a * ss[7] + I * c * ss[3];
  r[0][8] =  a * ss[8] + I * c * ss[12];
  r[0][9] =  a * ss[9] + I * c * ss[13];
  r[0][10] =  a * ss[10] + I * c * ss[14];
  r[0][11] =  a * ss[11] + I * c * ss[15];
  r[0][12] =  a * ss[12] + I * c * ss[8];
  r[0][13] =  a * ss[13] + I * c * ss[9];
  r[0][14] =  a * ss[14] + I * c * ss[10];
  r[0][15] =  a * ss[15] + I * c * ss[11];
} /* end of zm4x4_eq_spin_projection_zm4x4_23 */


static inline void zm4x4_eq_spin_projection_zm4x4_31 (double _Complex **r, double _Complex **s, double a, double b ) { 
  double _Complex ss[16];
  memcpy(ss, s[0], 16*sizeof(double _Complex));
  double c = b * 0.3333333333333333333333333;
  r[0][0] =  a * ss[0] +     c * ss[4];
  r[0][1] =  a * ss[1] +     c * ss[5];
  r[0][2] =  a * ss[2] +     c * ss[6];
  r[0][3] =  a * ss[3] +     c * ss[7];
  r[0][4] =  a * ss[4] -     c * ss[0];
  r[0][5] =  a * ss[5] -     c * ss[1];
  r[0][6] =  a * ss[6] -     c * ss[2];
  r[0][7] =  a * ss[7] -     c * ss[3];
  r[0][8] =  a * ss[8] +     c * ss[12];
  r[0][9] =  a * ss[9] +     c * ss[13];
  r[0][10] =  a * ss[10] +     c * ss[14];
  r[0][11] =  a * ss[11] +     c * ss[15];
  r[0][12] =  a * ss[12] -     c * ss[8];
  r[0][13] =  a * ss[13] -     c * ss[9];
  r[0][14] =  a * ss[14] -     c * ss[10];
  r[0][15] =  a * ss[15] -     c * ss[11];
} /* end of zm4x4_eq_spin_projection_zm4x4_31 */


static inline void zm4x4_eq_spin_projection_zm4x4_32 (double _Complex **r, double _Complex **s, double a, double b ) { 
  double _Complex ss[16];
  memcpy(ss, s[0], 16*sizeof(double _Complex));
  double c = b * 0.3333333333333333333333333;
  r[0][0] =  a * ss[0] - I * c * ss[4];
  r[0][1] =  a * ss[1] - I * c * ss[5];
  r[0][2] =  a * ss[2] - I * c * ss[6];
  r[0][3] =  a * ss[3] - I * c * ss[7];
  r[0][4] =  a * ss[4] - I * c * ss[0];
  r[0][5] =  a * ss[5] - I * c * ss[1];
  r[0][6] =  a * ss[6] - I * c * ss[2];
  r[0][7] =  a * ss[7] - I * c * ss[3];
  r[0][8] =  a * ss[8] - I * c * ss[12];
  r[0][9] =  a * ss[9] - I * c * ss[13];
  r[0][10] =  a * ss[10] - I * c * ss[14];
  r[0][11] =  a * ss[11] - I * c * ss[15];
  r[0][12] =  a * ss[12] - I * c * ss[8];
  r[0][13] =  a * ss[13] - I * c * ss[9];
  r[0][14] =  a * ss[14] - I * c * ss[10];
  r[0][15] =  a * ss[15] - I * c * ss[11];
} /* end of zm4x4_eq_spin_projection_zm4x4_32 */


static inline void zm4x4_eq_spin_projection_zm4x4_33 (double _Complex **r, double _Complex **s, double a, double b ) { 
  double _Complex ss[16];
  memcpy(ss, s[0], 16*sizeof(double _Complex));
  double c = b * 0.3333333333333333333333333;
  r[0][0] =  a * ss[0] +     c * ss[0];
  r[0][1] =  a * ss[1] +     c * ss[1];
  r[0][2] =  a * ss[2] +     c * ss[2];
  r[0][3] =  a * ss[3] +     c * ss[3];
  r[0][4] =  a * ss[4] +     c * ss[4];
  r[0][5] =  a * ss[5] +     c * ss[5];
  r[0][6] =  a * ss[6] +     c * ss[6];
  r[0][7] =  a * ss[7] +     c * ss[7];
  r[0][8] =  a * ss[8] +     c * ss[8];
  r[0][9] =  a * ss[9] +     c * ss[9];
  r[0][10] =  a * ss[10] +     c * ss[10];
  r[0][11] =  a * ss[11] +     c * ss[11];
  r[0][12] =  a * ss[12] +     c * ss[12];
  r[0][13] =  a * ss[13] +     c * ss[13];
  r[0][14] =  a * ss[14] +     c * ss[14];
  r[0][15] =  a * ss[15] +     c * ss[15];
} /* end of zm4x4_eq_spin_projection_zm4x4_33 */

/***********************************************************/
/***********************************************************/

static inline void re_eq_zm4x4_norm_diff (double *p, double _Complex **r, double _Complex **s ) {

  double _Complex c;
  double pp = 0.;
  c   = r[0][ 0] - s[0][ 0];  pp += creal( c * conj(c) );
  c   = r[0][ 1] - s[0][ 1];  pp += creal( c * conj(c) );
  c   = r[0][ 2] - s[0][ 2];  pp += creal( c * conj(c) );
  c   = r[0][ 3] - s[0][ 3];  pp += creal( c * conj(c) );
  c   = r[0][ 4] - s[0][ 4];  pp += creal( c * conj(c) );
  c   = r[0][ 5] - s[0][ 5];  pp += creal( c * conj(c) );
  c   = r[0][ 6] - s[0][ 6];  pp += creal( c * conj(c) );
  c   = r[0][ 7] - s[0][ 7];  pp += creal( c * conj(c) );
  c   = r[0][ 8] - s[0][ 8];  pp += creal( c * conj(c) );
  c   = r[0][ 9] - s[0][ 9];  pp += creal( c * conj(c) );
  c   = r[0][10] - s[0][10];  pp += creal( c * conj(c) );
  c   = r[0][11] - s[0][11];  pp += creal( c * conj(c) );
  c   = r[0][12] - s[0][12];  pp += creal( c * conj(c) );
  c   = r[0][13] - s[0][13];  pp += creal( c * conj(c) );
  c   = r[0][14] - s[0][14];  pp += creal( c * conj(c) );
  c   = r[0][15] - s[0][15];  pp += creal( c * conj(c) );
  (*p) = sqrt(pp);
}  /* end of re_eq_zm4x4_norm_diff */

/***********************************************************/
/***********************************************************/

static inline void zm4x4_printf (double _Complex **r, char*name, FILE*ofs ) {

  fprintf(ofs, "%s <- array(dim=c(4,4))\n", name);
  for ( int i = 0; i< 4; i++ ) {
    fprintf(ofs, "%s[%d,] <- c( (%25.16e + %25.16e*1.i),  (%25.16e + %25.16e*1.i),  (%25.16e + %25.16e*1.i),  (%25.16e + %25.16e*1.i) )\n",
        name, i+1,
        creal( r[i][0] ), cimag( r[i][0] ), 
        creal( r[i][1] ), cimag( r[i][1] ), 
        creal( r[i][2] ), cimag( r[i][2] ), 
        creal( r[i][3] ), cimag( r[i][3] ) );
  }
}  /* end of zm4x4_printf */


}  /* end of namespace cvc */

#endif
