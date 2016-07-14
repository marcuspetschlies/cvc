/*************************************************************
 * cvc_utils.h                                               *
 *************************************************************/
#ifndef _CVC_UTIL_H
#define _CVC_UTIL_H

namespace cvc {

int read_input (char *filename);

int alloc_gauge_field(double **gauge, const int V);
int alloc_gauge_field_dbl(double **gauge, const int N);
int alloc_gauge_field_flt(float **gauge, const int N);

int alloc_spinor_field(double **s, const int V);
int alloc_spinor_field_flt(float **s, const int N);
int alloc_spinor_field_dbl(double **s, const int N);

void plaquette(double*);
void plaquette2(double *pl, double*gfield);

void xchange_gauge(void);
void xchange_gauge_field(double*);

void xchange_field(double*);

void xchange_eo_field(double *phi, int eo);

void xchange_contraction(double *phi, int N);

int write_contraction (double *s, int *nsource, char *filename, int Nmu,
                       int write_ascii, int append);
int read_contraction(double *s, int *nsource, char *filename, int Nmu);

void init_gamma(void);

int printf_gauge_field( double *gauge, FILE *ofs);
int printf_spinor_field(double *s,     FILE *ofs);
int printf_spinor_field_5d(double *s,     FILE *ofs);

void set_default_input_values(void);

void init_gauge_trafo(double **g, double heat);
void apply_gt_gauge(double *g);
void apply_gt_prop(double *g, double *phi, int is, int ic, int mu, char *basename, int source_location);
void get_filename(char *filename, const int nu, const int sc, const int sign);
int wilson_loop(complex *w, const int xstart, const int dir, const int Ldir);

int IRand(int min, int max);
double Random_Z2();
int ranz2(double * y, unsigned int NRAND);
void random_gauge_field(double *gfield, double h);
void random_gauge_point(double **gauge_point, double heat);
void random_gauge_field2(double *gfield);
int read_pimn(double *pimn, const int read_flag);

void random_spinor_field (double *s, unsigned int V);

int init_hpe_fields(int ***loop_tab, int ***sigma_tab, int ***shift_start, double **tcf, double **tcb);
int free_hpe_fields(int ***loop_tab, int ***sigma_tab, int ***shift_start, double **tcf, double **tcb);
int rangauss (double * y1, unsigned int NRAND);

void cm_proj(double *A);
void contract_twopoint(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c);
void contract_twopoint_snk_momentum(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c, int*snk_mom);
void contract_twopoint_snk_momentum_trange(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c, int* snk_mom, int tmin, int tmax);

void contract_twopoint_xdep(void*contr, const int idsource, const int idsink, void*chi, void*phi, int n_c, int stride, double factor, size_t prec);
void contract_twopoint_xdep_timeslice(void*contr, const int idsource, const int idsink, void*chi, void*phi, int n_c, int stride, double factor, size_t prec);

int decompress_gauge(double*gauge_aux, float*gauge_field_flt);
int compress_gauge(float*gauge_field_flt, double *gauge_aux);
int set_temporal_gauge(double*gauge_transform);
int apply_gauge_transform(double*gauge_new, double*gauge_transform, double*gauge_old);

void free_sp_field(spinor_propagator_type **fp);

void free_fp_field(fermion_propagator_type **fp);
  
int unit_gauge_field(double*g, unsigned int N);
int write_contraction2 (double *s, char *filename, int Nmu, unsigned int items, int write_ascii, int append);
void printf_fp(fermion_propagator_type f, char*name, FILE*ofs);
void printf_sp(spinor_propagator_type f, char*name, FILE*ofs);
void norm2_sp(spinor_propagator_type f, double*res);

int init_rng_stat_file (unsigned int seed, char*filename);
int init_rng_state (int seed, int **rng_state);
int fini_rng_state (int **rng_state);
int sync_rng_state(int id, int reset);


int shift_spinor_field (double *s, double *r, int *d);
int printf_SU3_link (double *u, FILE*ofs);

void check_source(double *sf, double*work, double mass, unsigned int location, int sc);

void reunit(double *A);
void su3_proj_step (double *A, double *B);
void cm_proj_iterate(double *A, double *B, int maxiter, double tol);

void spinor_field_lexic2eo (double *r_lexic, double*r_e, double *r_o);
void spinor_field_eo2lexic (double *r_lexic, double*r_e, double *r_o);
void spinor_field_unpack_lexic2eo (double *r_lexic, double*r_o1, double *r_o2);
int printf_eo_spinor_field(double *s, int use_even, FILE *ofs);

void spinor_field_mi_eq_spinor_field_ti_re(double*r, double*s, double c, unsigned int N);
void spinor_field_ti_eq_re (double *r, double c, unsigned int N);
void spinor_field_eq_spinor_field_ti_re (double *r, double *s, double c, unsigned int N);
void spinor_field_norm_diff (double*d, double *r, double *s, unsigned int N);

}  /* end of namespace cvc */
#endif

