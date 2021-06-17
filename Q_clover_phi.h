#ifndef _Q_CLOVER_PHI_H
#define _Q_CLOVER_PHI_H

namespace cvc {

void clover_term_init (double***s, int nmat);
void clover_term_fini (double***s);

void clover_term_eo (double**s, double*gauge_field);
void Q_clover_phi_eo (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux, double **cl);
void M_clover_zz (double*s, double*r, double mass, double*cl);

void clover_mzz_matrix (double**mzz, double**cl, double mu, double csw);
void clover_mzz_inv_matrix (double**mzzinv, double**mzz);
void M_clover_zz_matrix (double*s, double*r, double*mzz);
void M_clover_zz_inv_matrix (double*s, double*r, double *mzzinv);
void Q_clover_phi_matrix_eo (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double *aux, double**mzz);
void C_clover_oo (double*s, double*r, double *gauge_field, double *s_aux, double*mzz, double*mzzinv);
void X_clover_eo (double *even, double *odd, double *gauge_field, double*mzzinv);
void C_clover_from_Xeo (double *t, double *s, double *r, double *gauge_field, double*mzz);

void Q_clover_eo_SchurDecomp_A    (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double *mzz,   double *aux);
void Q_clover_eo_SchurDecomp_Ainv (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double*mzzinv, double *aux);
void Q_clover_eo_SchurDecomp_B    (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double*mzz,    double*mzzinv, double *aux);
void Q_clover_eo_SchurDecomp_Binv (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double*mzzinv, double *aux);

int Q_clover_invert (double*prop, double*source, double*gauge_field, double *mzzinv, int op_id);

int Q_clover_eo_invert (double*prop_e, double*prop_o, double*source_e, double*source_o, double*gauge_field, double *mzzinv, int op_id);

int Q_clover_invert_subspace ( double**prop, double**source, int nsf, double*evecs, double*evecs_norm, int nev, double*gauge_field, double **mzz[2], double **mzzinv[2], int flavor_id);

int Q_clover_eo_invert_subspace ( double**prop_e,   double**prop_o, double**source_e, double**source_o,
                                  int nsf, double*evecs, double*evecs_norm, int nev, double*gauge_field, double **mzz[2], double **mzzinv[2], int flavor_id, double**eo_spinor_aux);


int Q_clover_eo_invert_subspace_stochastic ( double**prop_e, double**prop_o, double**source_e, double**source_o,
                                             int nsf, double*sample_prop, double*sample_source, int nsample,
                                             double*gauge_field, double**mzz[2], double**mzzinv[2], int flavor_id, double**eo_spinor_aux );

int Q_clover_eo_invert_subspace_stochastic_timeslice ( 
  double**prop_e, double**prop_o,
  double**source_e, double**source_o,
  int nsf,
  double***sample_prop, double***sample_source, int nsample,
  double*gauge_field, double**mzz[2], double**mzzinv[2],
  int timeslice, int flavor_id );


}  /* end of namespace cvc */
#endif
