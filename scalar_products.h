#ifndef _SCALAR_PRODUCTS_H
#define _SCALAR_PRODUCTS_H

namespace cvc {

void spinor_scalar_product_re(double *r, double *xi, double *phi, unsigned int V);
void spinor_scalar_product_co(complex *w, double *xi, double *phi, unsigned int V);
void eo_spinor_spatial_scalar_product_co(complex *w, double *xi, double *phi, int eo);

void eo_spinor_dag_gamma_spinor(complex*gsp, double*xi, int gid, double*phi);
void eo_gsp_momentum_projection (complex *gsp_p, complex *gsp_x, complex *phase, int eo);

}
#endif
