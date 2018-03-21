#ifndef _SCALAR_PRODUCTS_H
#define _SCALAR_PRODUCTS_H

namespace cvc {

void spinor_scalar_product_re(double *r, double *xi, double *phi, int V);
void spinor_scalar_product_co(complex *w, double *xi, double *phi, int V);

void eo_spinor_spatial_scalar_product_co( double _Complex * w, double * const xi, double * const phi, int const eo);

void eo_spinor_dag_gamma_spinor(complex * const gsp, double * const xi, int const gid, double * const phi);

void eo_gsp_momentum_projection (complex * const gsp_p, complex * const gsp_x, complex * const phase, int const eo);

}
#endif
