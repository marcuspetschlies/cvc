/*************************************************************
 * cvc_utils.h                                               *
 *************************************************************/
#ifndef _CVC_UTIL_H
#define _CVC_UTIL_H
#include <sys/time.h>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include "uwerr.h"

namespace cvc {

int read_input (char *filename);

int alloc_gauge_field( double **gauge, unsigned int const V );
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

void xchange_spinor_field_bnd2(double*);

void xchange_eo_field(double *phi, int eo);

void xchange_contraction(double *phi, int N);

int write_contraction (double *s, int *nsource, char *filename, int Nmu,
                       int write_ascii, int append);
int read_contraction(double *s, int *nsource, char *filename, int Nmu);

void init_gamma(void);

int printf_gauge_field( double *gauge, FILE *ofs);
int printf_spinor_field(double *s, int print_halo, FILE *ofs);
int printf_spinor_field_5d(double *s,     FILE *ofs);
void printf_cm( double*A, char*name, FILE*file);


void set_default_input_values(void);

void init_gauge_trafo(double **g, double heat);
void apply_gt_gauge(double *g, double*gauge_field);
void apply_gt_prop(double *g, double *phi, int is, int ic, int mu, char *basename, int source_location);
void get_filename(char *filename, const int nu, const int sc, const int sign);
int wilson_loop(complex *w, double*gauge_field, const int xstart, const int dir, const int Ldir);

int IRand(int min, int max);
double Random_Z2();

int ranz2(double * y, unsigned int NRAND);

int ranz3 ( double * const y, unsigned int const NRAND );

int ranbinary(double * const y, unsigned int const NRAND);

void random_gauge_field(double *gfield, double h);
void random_gauge_point(double **gauge_point, double heat);
void random_cm(double *A, double heat);

void random_gauge_field2(double *gfield);
int read_pimn(double *pimn, const int read_flag);

void random_spinor_field (double *s, unsigned int V);

int rangauss (double * y1, unsigned int NRAND);

void cm_proj(double *A);
void contract_twopoint(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c);

void contract_twopoint_snk_momentum ( double * const contr, int const idsource, int const idsink, double ** const chi, double ** const phi, unsigned int const n_s, unsigned int const n_c, int const snk_mom[3], int const reduce );


void contract_twopoint_snk_momentum_trange(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c, int* snk_mom, int tmin, int tmax);

void contract_twopoint_xdep(void*contr, const int idsource, const int idsink, void * const chi, void * const phi, int const n_s, int const n_c, unsigned int const stride, double const factor, size_t const prec);

void contract_twopoint_xdep_timeslice(void*contr, const int idsource, const int idsink, void*chi, void*phi, int n_c, int stride, double factor, size_t prec);

int decompress_gauge(double*gauge_aux, float*gauge_field_flt);
int compress_gauge(float*gauge_field_flt, double *gauge_aux);
int set_temporal_gauge(double*gauge_transform, double*gauge_field);
int apply_gauge_transform(double*gauge_new, double*gauge_transform, double*gauge_old);

void free_sp_field(spinor_propagator_type **fp);

void free_fp_field(fermion_propagator_type **fp);
  
int unit_gauge_field(double * const g, unsigned int const N);

int write_contraction2 (double *s, char *filename, int Nmu, unsigned int items, int write_ascii, int append);
void printf_fp(fermion_propagator_type f, char*name, FILE*ofs);
void printf_sp(spinor_propagator_type f, char*name, FILE*ofs);
void norm2_sp(spinor_propagator_type f, double*res);

int init_rng_stat_file (unsigned int seed, char*filename);

int init_rng_state (int const seed, int **rng_state);
int fini_rng_state (int **rng_state);
int sync_rng_state ( int *rng_state_in, int const id, int const reset);
int save_rng_state ( int const id, char * filename );
int read_rng_state ( int **rng_state, int const id, char * filename );
int get_rng_state ( int *rng_state );


int shift_spinor_field (double *s, double *r, int *d);
int printf_SU3_link (double *u, FILE*ofs);

void check_source(double *sf, double*work, double*gauge_field, double mass, unsigned int location, int sc);

void reunit(double *A);
void su3_proj_step (double *A, double *B);
void cm_proj_iterate(double *A, double *B, int maxiter, double tol);

void spinor_field_lexic2eo (double *r_lexic, double*r_e, double *r_o);
void spinor_field_eo2lexic (double *r_lexic, double*r_e, double *r_o);

void spinor_field_unpack_lexic2eo (double *r_lexic, double*r_o1, double *r_o2);

void complex_field_lexic2eo (double *r_lexic, double*r_e, double *r_o);
void complex_field_eo2lexic (double *r_lexic, double*r_e, double *r_o);


int printf_eo_spinor_field(double *s, int use_even, int print_halo, FILE *ofs);

void spinor_field_eq_spinor_field_pl_spinor_field_ti_re(double*r, double*s, double *t, double c, unsigned int N);
void spinor_field_eq_spinor_field_ti_re (double *r, double *s, double c, unsigned int N);
void spinor_field_eq_spinor_field_mi_spinor_field(double*r, double*s, double*t, unsigned int N);
void spinor_field_eq_spinor_field_pl_spinor_field(double*r, double*s, double*t, unsigned int N);

void spinor_field_eq_gamma_ti_spinor_field ( double * const r, int const gid, double * const s, unsigned int const N);


void spinor_field_mi_eq_spinor_field_ti_re(double*r, double*s, double c, unsigned int N);
void spinor_field_ti_eq_re (double *r, double c, unsigned int N);
void spinor_field_pl_eq_spinor_field(double*r, double*s, unsigned int N);

void spinor_field_eq_spinor_field_ti_real_field (double*r, double*s, double *c, unsigned int N);
void spinor_field_eq_spinor_field_ti_complex_field ( double * const r, double * const s, double * const c, unsigned int const N);


void spinor_field_norm_diff (double * const d, double * const r, double * const s, unsigned int const N);

void g5_phi(double * const phi, unsigned int const N);

spinor_propagator_type *create_sp_field(size_t N);
fermion_propagator_type *create_fp_field(size_t N);
int fermion_propagator_field_tm_rotation(fermion_propagator_type *s, fermion_propagator_type *r, int sign, int fermion_type, unsigned int N);
int spinor_field_tm_rotation(double*s, double*r, int sign, int fermion_type, unsigned int N);

int check_cvc_wi_position_space (double *conn);

int assign_fermion_propagator_from_spinor_field (fermion_propagator_type *s, double**prop_list, unsigned int N);
int assign_spinor_field_from_fermion_propagator (double**prop_list, fermion_propagator_type *s, unsigned int N);
int assign_spinor_field_from_fermion_propagator_component (double*spinor_field, fermion_propagator_type *s, int icomp, unsigned int N);

void spinor_field_eq_spinor_field_ti_co (double*r, double*s, complex w, unsigned int N);
void spinor_field_pl_eq_spinor_field_ti_co (double*r, double*s, complex w, unsigned int N);


void xchange_eo_propagator ( fermion_propagator_type *fp, int eo, int dir);

int get_point_source_info (int const gcoords[4], int lcoords[4], int * const proc_id);

int get_timeslice_source_info (int gts, int *lts, int*proc_id );

void complex_field_ti_eq_re (double *r, double c, unsigned int N);

void complex_field_eq_complex_field_conj_ti_re (double *r, double c, unsigned int N);

void complex_field_eq_complex_field_pl_complex_field_conj_ti_re (double*r, double*s, double*t, double c, unsigned int N);

int check_point_source_propagator_clover_eo(double**prop_e, double**prop_o, double**work, double*gf, double**mzz, double**mzzinv, int gcoords[4], int nf );

int check_oo_propagator_clover_eo(double**prop_o, double**source, double**work, double*gf, double**mzz, double**mzzinv, int nf );

int check_subspace_propagator_clover_eo(
  double **prop_e, double**prop_o,
  double**source_e, double**source_o,
  int nf, double *eo_evecs_block, double *evecs_norm, int nev, double*gauge_field, double**mzz[2], double**mzzinv[2], int flavor_id );

int check_residual_clover ( double ** const prop, double ** const source, double * const gauge_field, double ** const mzz, double ** const mzzinv, int const nf  );

int plaquetteria  (double*gauge_field );

int gauge_field_eq_gauge_field_ti_phase (double**gauge_field_with_phase, double*gauge_field, complex co_phase[4] );

int gauge_field_eq_gauge_field_ti_bcfactor (double ** const gauge_field_with_bc, double * const gauge_field, double _Complex const bcfactor );

void co_field_eq_fv_dag_ti_fv (double * const c, double * const r, double * const s, unsigned int const N );

void co_field_pl_eq_fv_dag_ti_fv (double*c, double*r, double*s, unsigned int N );

void co_field_mi_eq_fv_dag_ti_fv (double*c, double*r, double*s, unsigned int N );

void co_field_eq_fv_dag_ti_gamma_ti_fv (double*c, double*r, int gid, double*s, unsigned int N );

void spinor_field_eq_cm_field_ti_spinor_field (double*r, double *u, double*s, unsigned int N);

void spinor_field_eq_gauge_field_ti_spinor_field (double*r, double *gf, double*s, int mu, unsigned int N);

void spinor_field_eq_gauge_field_dag_ti_spinor_field (double*r, double *gf, double*s, int mu, unsigned int N);

void spinor_field_eq_gauge_field_fbwd_ti_spinor_field (double*r, double *gf, double*s, int mu, int fbwd, unsigned int N);

void fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fermion_propagator_type *r, int gid, fermion_propagator_type *s, unsigned int N);

void fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fermion_propagator_type *r, int gid, fermion_propagator_type *s, unsigned int N);

void fermion_propagator_field_eq_fermion_propagator_field_ti_re (fermion_propagator_type*r, fermion_propagator_type*s, double c, unsigned int N);

int fix_eigenvector_phase ( double **evecs_field, int num );

int check_eigenpairs ( double ** const eo_evecs_field, double ** evecs_eval, unsigned int const evecs_num, double * const gauge_field, double ** const mzz[2], double ** const mzzinv[2] );

int get_io_proc (void);

unsigned int * sort_by_dvalue_mapping ( double * const value, unsigned int const nv );

int sort_dfield_by_map ( double * const v, unsigned int const nv, unsigned int * const map, unsigned int const N );


int vdag_gloc_w_scalar_product ( double _Complex ***** const vw_mat, double *** const veo,  int const nv, double ** const weo, int const iweo,
    int const momentum_number, int  (* const momentum_list)[3] , int const gamma_id_number, int * const gamma_id_list );

int vdag_gloc_w_scalar_product_pt ( double _Complex **** const vw_mat, double *** const veo,  int const nv, double ** const weo, int const iweo,
    int const pt_number, int  (* const pt_list)[4] , int const gamma_id_number, int * const gamma_id_list );

/***************************************************************************
 * set number of openmp threads
 ***************************************************************************/
inline void set_omp_number_threads (void) {
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [set_omp_number_threads] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [set_omp_number_threads] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[set_omp_number_threads] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif
}  /* end of set_omp_number_threads */

/***************************************************************************
 * calculate elapsed wall-time
 ***************************************************************************/
inline void show_time ( struct timeval * const ta, struct timeval * const tb, char * tag, char * timer, int const io ) {

  long int seconds =  tb->tv_sec  - ta->tv_sec;
  long int useconds = tb->tv_usec - ta->tv_usec;
  if ( useconds < 0 ) {
    useconds += 1000000;
    seconds--;
  }
  /* if ( io ) fprintf ( stdout, "# [%s] time for %s %ld sec %ld usec\n", tag, timer, seconds, useconds ); */
  if ( io ) fprintf ( stdout, "# [%s] time for %s %e sec\n", tag, timer, (double)seconds + (double)useconds/1000000.0 );

}  /* end of show_time */

int check_momentum_space_wi_tpvec ( double *** const hvp , int const pvec[3] );

int hvp_ft ( double *** const hvp, int const type );

int hvp_ft_single_component ( double * const hvp, int const shift );

int apply_uwerr_real ( double * const data, unsigned int const nmeas, unsigned int const ndata, unsigned int const ipo_first, unsigned int const ipo_stride, char * obs_name );

int apply_uwerr_func ( double * const data, unsigned int const nmeas, unsigned int const ndata, unsigned int const nset, int const narg, int * const arg_first, int * const arg_stride, char * obs_name,
   dquant func, dquant dfunc );

int read_source_coords_list ( int *** const conf_src_list, int const num_conf, int const num_src_per_conf, char * const ensemble_name );

}  /* end of namespace cvc */
#endif

