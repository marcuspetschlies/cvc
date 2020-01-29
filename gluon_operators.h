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

/********************************************************************/
/********************************************************************/

int G_plaq_rect ( double *** Gp, double *** Gr, double * const gauge_field);

int gluonic_operators ( double ** op, double * const gfield );

int gluonic_operators_eo_from_fst ( double ** op, double *** const G );

}  /* end of namespace cvc */
#endif
