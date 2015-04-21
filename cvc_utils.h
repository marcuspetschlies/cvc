/*************************************************************
 * cvc_utils.h                                               *
 *************************************************************/
#ifndef _CVC_UTIL_H
#define _CVC_UTIL_H

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
void xchange_gauge_field_timeslice(double *);
void xchange_field(double*);
void xchange_field_timeslice(double *);

int write_contraction (double *s, int *nsource, char *filename, int Nmu,
                       int write_ascii, int append);
int read_contraction(double *s, int *nsource, char *filename, int Nmu);

void init_gamma(void);

int printf_gauge_field( double *gauge, FILE *ofs);
int printf_spinor_field(double *s,     FILE *ofs);

void set_default_input_values(void);

void init_gauge_trafo(double **g, double heat);
void apply_gt_gauge(double *g);
void apply_gt_prop(double *g, double *phi, int is, int ic, int mu, char *basename, int source_location);
void get_filename(char *filename, const int nu, const int sc, const int sign);
int wilson_loop(complex *w, const int xstart, const int dir, const int Ldir);

int IRand(int min, int max);
double Random_Z2();
int ranz2(double * y, int NRAND);
void cvc_random_gauge_field(double *gfield, double h);
void random_gauge_point(double **gauge_point, double heat);
void cvc_random_gauge_field2(double *gfield);
int read_pimn(double *pimn, const int read_flag);

int init_hpe_fields(int ***loop_tab, int ***sigma_tab, int ***shift_start, double **tcf, double **tcb);
int free_hpe_fields(int ***loop_tab, int ***sigma_tab, int ***shift_start, double **tcf, double **tcb);
int rangauss (double * y1, int NRAND);

void cm_proj(double *A);
void contract_twopoint(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c);

void contract_twopoint_xdep(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c, int stride, double factor);

int decompress_gauge(double*gauge_aux, float*gauge_field_flt);
int compress_gauge(float*gauge_field_flt, double *gauge_aux);
int set_temporal_gauge(double*gauge_transform);
int apply_gauge_transform(double*gauge_new, double*gauge_transform, double*gauge_old);
#endif

