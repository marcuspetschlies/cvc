#ifndef _LAPLACE_H
#define _LAPLACE_H

namespace cvc {

void cv_eq_laplace_cv_4d ( double * const v_out, double * const g, double * const v_in );

void cv_eq_laplace_cv_3d ( double * const v_out, double * const g, double * const v_in, int const ts);


}

#endif
