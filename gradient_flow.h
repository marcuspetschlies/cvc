#ifndef _GRADIENT_FLOW_H
#define _GRADIENT_FLOW_H

namespace cvc {

void ZX ( double * const z, double * const g, double const a, double const b );

void apply_ZX ( double * const g, double * const z, double const dt );

void apply_laplace ( double * const s, double * const r_in, double * const g );

void flow_fwd_gauge_spinor_field ( double * const g, double * const phi, unsigned int const niter, double const dt, int const flow_gauge, int const flow_spinor );

void flow_adjoint_step_gauge_spinor_field ( double * const g, double * const chi, unsigned int const niter, double const dt, int const flow_gauge, int const flow_spinor );

void flow_adjoint_gauge_spinor_field ( double ** const g, double * const chi, double const dt, unsigned int const mb, unsigned int const nb, int const store );

}

#endif
