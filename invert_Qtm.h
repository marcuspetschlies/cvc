#ifndef _INVERT_QTM_H
#define _INVERT_QTM_H

namespace cvc {
extern double solver_precision;
extern int niter_max;

void spinor_scalar_product_re(double *r, double *xi, double *phi, int V);
void spinor_scalar_product_co(complex *w, double *xi, double *phi, int V);
void eo_spinor_spatial_scalar_product_co(complex *w, double *xi, double *phi, int eo);

void eo_spinor_dag_gamma_spinor(complex*gsp, double*xi, int gid, double*phi);
void eo_gsp_momentum_projection (complex *gsp_p, complex *gsp_x, complex *phase, int eo);

int invert_Qtm(double *xi, double *phi, int kwork);
int invert_Qtm_her(double *xi, double *phi, int kwork);
int invert_Q_Wilson(double *xi, double *phi, int kwork);
int invert_Q_Wilson_her(double *xi, double *phi, int kwork);
int invert_Q_DW_Wilson(double *xi, double *phi, int kwork);
int invert_Q_DW_Wilson_her(double *xi, double *phi, int kwork);
}
#endif
