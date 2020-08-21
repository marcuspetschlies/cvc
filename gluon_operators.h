#ifndef _GLUON_OPERATORS_H
#define _GLUON_OPERATORS_H

namespace cvc {

/********************************************************************/
/********************************************************************/

inline void d2_eq_tr_cm ( double *c, double *A ) {
  c[0] = (A)[ 0] + (A)[ 8] + (A)[16];
  c[1] = (A)[ 1] + (A)[ 9] + (A)[17];
}  /* end of d2_eq_tr_cm */

inline void d2_pl_eq_tr_cm ( double *c, double *A ) {
  c[0] += (A)[ 0] + (A)[ 8] + (A)[16];
  c[1] += (A)[ 1] + (A)[ 9] + (A)[17];
}  /* end of d2_pl_eq_tr_cm */

/********************************************************************
 *
 ********************************************************************/
inline void project_to_generators ( double * const p, double * const A ) {

  const double sq1  = 0.8164965809277260;  /* sqrt ( 2 / 3 ) */
  const double sq2  = 0.5773502691896257;  /* sqrt ( 1 / 3 ) */

  /* unit matrix */
  p[0] = ( A[1] + A[9] + A[17] ) * sq1;

  /* lambda_1 */
  p[1] = A[3] + A[ 7];

  /* lambda_2 */
  p[2] = A[2] - A[ 6];

  /* lambda_3 */
  p[3] = A[1] - A[ 9];

  /* lambda_4 */
  p[4] = A[5] + A[13];

  /* lambda_5 */
  p[5] = A[4] - A[12];

  /* lambda_6 */
  p[6] = A[11] + A[15];

  /* lambda_7 */
  p[7] = A[10] - A[14];

  /* lambda_2 / 2 */
  p[8] = ( A[1] + A[9] - 2 * A[17] ) * sq2;
}  /* end of project_to_generators */

/********************************************************************
 *
 ********************************************************************/
inline void restore_from_generators ( double * const A, double * const p ) {
  const double one_over_sqrt3  = 0.5773502691896258;
  const double one_over_sqrt6  = 0.4082482904638631;

  A[ 0] = 0.;
  A[ 1] = p[0] * one_over_sqrt6 + p[3] * 0.5 + p[8] * one_over_sqrt3 * 0.5;

  A[ 2] =  p[2] * 0.5;
  A[ 3] =  p[1] * 0.5;

  A[ 6] = -p[2] * 0.5;
  A[ 7] =  p[1] * 0.5;

  A[ 4] =  p[5] * 0.5;
  A[ 5] =  p[4] * 0.5;

  A[12] = -p[5] * 0.5;
  A[13] =  p[4] * 0.5;

  A[ 8] = 0.;
  A[ 9] = p[0] * one_over_sqrt6 - p[3] * 0.5 + p[8] * one_over_sqrt3 * 0.5;

  A[10] =  p[7] * 0.5;
  A[11] =  p[6] * 0.5;

  A[14] = -p[7] * 0.5;
  A[15] =  p[6] * 0.5;

  A[16] = 0.;
  A[17] = p[0] * one_over_sqrt6 - p[8] * one_over_sqrt3;
}  /* end of restore_from_generators */

/********************************************************************/
/********************************************************************/

/********************************************************************
 *
 ********************************************************************/
inline void project_to_generators_hermitean ( double * const p, double * const A ) {

  const double sq1  = 0.8164965809277260;  /* sqrt ( 2 / 3 ) */
  const double sq2  = 0.5773502691896257;  /* sqrt ( 1 / 3 ) */

  /* unit matrix */
  p[0] = ( A[0] + A[8] + A[16] ) * sq1;

  /* lambda_1 */
  p[1] =  A[2] + A[ 6];

  /* lambda_2 */
  p[2] = -A[3] + A[ 7];

  /* lambda_3 */
  p[3] = A[0] - A[ 8];

  /* lambda_4 */
  p[4] =  A[4] + A[12];

  /* lambda_5 */
  p[5] = -A[5] + A[13];

  /* lambda_6 */
  p[6] =  A[10] + A[14];

  /* lambda_7 */
  p[7] = -A[11] + A[15];

  /* lambda_8 / 2 */
  p[8] = ( A[0] + A[8] - 2 * A[16] ) * sq2;
}  /* end of project_to_generators_hermitean */

/********************************************************************
 *
 ********************************************************************/
inline void restore_from_generators_hermitean ( double * const A, double * const p ) {
  const double one_over_sqrt3  = 0.5773502691896258;
  const double one_over_sqrt6  = 0.4082482904638631;

  A[ 0] = p[0] * one_over_sqrt6 + p[3] * 0.5 + p[8] * one_over_sqrt3 * 0.5;
  A[ 1] = 0.;

  A[ 2] = +p[1] * 0.5;
  A[ 3] = -p[2] * 0.5;
  A[ 6] = +p[1] * 0.5;
  A[ 7] = +p[2] * 0.5;

  A[ 4] = +p[4] * 0.5;
  A[ 5] = -p[5] * 0.5;
  A[12] = +p[4] * 0.5;
  A[13] = +p[5] * 0.5;

  A[ 8] = p[0] * one_over_sqrt6 - p[3] * 0.5 + p[8] * one_over_sqrt3 * 0.5;
  A[ 9] = 0.;

  A[10] = +p[6] * 0.5;
  A[11] = -p[7] * 0.5;
  A[14] = +p[6] * 0.5;
  A[15] = +p[7] * 0.5;

  A[16] = p[0] * one_over_sqrt6 - p[8] * one_over_sqrt3;
  A[17] = 0.;

}  /* end of restore_from_generators_hermitean */

/********************************************************************/
/********************************************************************/

int G_plaq_rect ( double *** Gp, double *** Gr, double * const gauge_field);

int gluonic_operators ( double ** op, double * const gfield );

int gluonic_operators_projected ( double ** const op, double *** const G );

int gluonic_operators_eo_from_fst ( double ** op, double *** const G );

int G_plaq ( double *** Gp, double * const gauge_field, int const antihermitean );

int G_rect ( double *** Gr, double * const gauge_field, int const antihermitean );

int gluonic_operators_eo_from_fst_projected ( double ** op, double *** const G, int const traceless );

int gluonic_operators_qtop_from_fst_projected ( double * op, double *** const G, int const traceless );

int gluonic_operators_gg_from_fst_projected ( double ** op, double *** const G, int const traceless );

}  /* end of namespace cvc */
#endif
