#ifndef _GRADIENT_FLOW_H

namespace cvc {

void ZX ( double * const z, double * const g, double const a, double const b );

void apply_ZX ( double * const g, double * const z, double const dt );

int apply_laplace ( double * const s, double * const r_in, double * const g );

void flow_fwd_gauge_spinor_field ( double * const g, double * const phi, unsigned int const niter, double const dt );

}

#endif
