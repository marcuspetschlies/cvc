#ifndef _Q_PHI_H
#define _Q_PHI_H

/********************/

/* Computes xi = Q phi, where Q is the light tm Dirac operator with twisted boundary conditions. */

namespace cvc {

void Q_phi_tbc(double *xi, double *phi);

void Hopping(double *xi, double *phi, double*gauge_field);
void Hopping_eo(double *s, double *r, double *gauge_field, int EO);

void M_zz (double*s, double*r, double mass);
void M_zz_inv (double*s, double*r, double mass);
  
void C_oo (double*s, double*r, double *gauge_field, double mass, double *s_aux);

void Q_phi(double *xi, double *phi, double *gauge_field, const double mutm);

void Q_phi_eo (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux);

void Q_eo_SchurDecomp_A (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux);
void Q_eo_SchurDecomp_Ainv (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux);
void Q_eo_SchurDecomp_B (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux);
void Q_eo_SchurDecomp_Binv (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux);

void X_eo (double *even, double *odd, double mu, double *gauge_field);
void X_oe (double *odd, double *even, double mu, double *gauge_field);

void C_with_Xeo (double *r, double *s, double *gauge_field, double mu, double *r_aux);

void C_from_Xeo (double *t, double *s, double *r, double *gauge_field, double mu);

void apply_cvc_vertex_eo(double *s, double *r, int mu, int fbwd, double *gauge_field, int EO);

void apply_cvc_vertex(double *s, double *r, int mu, int fbwd, double *gauge_field);

void apply_cvc_vertex_propagator_eo ( fermion_propagator_type *s, fermion_propagator_type *r, int mu, int fbwd, double *gauge_field, int EO);

void apply_propagator_constant_cvc_vertex ( fermion_propagator_type *s, fermion_propagator_type *r, int mu, int fbwd, double U[18], const unsigned int N );

int Q_invert (double*prop, double*source, double*gauge_field, double mass, int op_id);

int spinor_field_eq_cov_displ_spinor_field ( double * const s, double * const r_in, int const mu, int const fbwd, double * const gauge_field );

}  /* end of namespace cvc */
#endif
