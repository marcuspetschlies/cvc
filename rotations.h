#ifndef _ROTATIONS_H
#define _ROTATIONS_H

#include "ilinalg.h"
#include "table_init_z.h"

namespace cvc {

typedef struct {
  int n[3];
  double w;
  double a[3];
  char name[40];
} rotation_type; 

/********************************************************/
/********************************************************/

typedef struct {
  double _Complex **m;
  int d;
} rotation_matrix_type;

/********************************************************/
/********************************************************/

/********************************************************
 *
 ********************************************************/
inline double dgeps (double const a, double const eps ) {
  double t = fabs ( a );
  return( t > eps ? a : 0. );
}  // end of dgeps

/********************************************************/
/********************************************************/

void rot_init_block_params (void);

extern rotation_type cubic_group_double_cover_rotations[48];
extern rotation_type cubic_group_rotations[24];
extern rotation_type cubic_group_rotations_v2[24];
extern int cubic_group_double_cover_identification_table[24][2];


void rot_init_rotation_table (void);

long unsigned int factorial (int n);

void rot_mat_unity ( double _Complex **R, int N );

void rot_mat_zero ( double _Complex **R, int N );

void rot_axis2polar ( double*theta, double*phi, int n[3] );

void rot_rotation_matrix_spherical_basis ( double _Complex**R, int J2, int n[3], double w);

void rot_spherical2cartesian_3x3 (double _Complex **C, double _Complex **S);

void rot_cartesian_to_spherical_contravariant_mat (double _Complex ***S, double _Complex ***C, int M, int N );


void rot_mat_adj (double _Complex **C, double _Complex **R, int N);

void rot_mat_conj (double _Complex ** const C, double _Complex ** const R, unsigned int const N);

void rot_mat_trans (double _Complex ** const C, double _Complex ** const R, unsigned int const N);

void rot_mat_assign (double _Complex **C, double _Complex **R, int N);

void rot_mat_ti_mat (double _Complex ** const C, double _Complex ** const A, double _Complex ** const B, int const N);

void rot_mat_ti_mat_adj (double _Complex ** const C, double _Complex ** const A, double _Complex ** const B, int const N);

void rot_mat_adj_ti_mat (double _Complex **C, double _Complex **A, double _Complex **B, int N);

void rot_printf_matrix (double _Complex **R, int N, char *A, FILE*ofs );

void rot_printf_matrix_comp (double _Complex **R, double _Complex **S, int N, char *A, FILE*ofs );

double rot_mat_norm2 (double _Complex **R, int N);

int rot_mat_check_is_sun (double _Complex **R, int N);

void rot_point ( int nrot[3], int const nn[3], double _Complex ** const R);

void rot_point_inv ( int nrot[3], int n[3], double _Complex **R);

int rot_mat_check_is_real_int (double _Complex **R, int N );

void rot_mat_ti_eq_re (double _Complex **R, double c, int N);

void rot_mat_pl_eq_mat_ti_co (double _Complex ** const R, double _Complex ** const S, double _Complex const c, int const N);


void rot_global_point_mod ( int nrot[3], int n[3], double _Complex **R);

void rot_center_global_point ( int nrot[3], int n[3], double _Complex **R);

void rot_center_global_point_inv ( int nrot[3], int n[3], double _Complex **R);

void rot_center_local_point ( int nrot[3], int n[3], double _Complex **R, int l[3]);

void rot_center_local_point_inv ( int nrot[3], int n[3], double _Complex **R, int l[3]);

int rot_gauge_field ( double*gf_rot, double *gf, double _Complex **R);

void rot_printf_rint_matrix (double _Complex **R, int N, char *A, FILE*ofs );

int rot_spinor_field ( double*sf_rot, double *sf, double _Complex **R);

int rot_bispinor_rotation_matrix_spherical_basis ( double _Complex **ASpin, int n[3], double w );

void rot_bispinor_mat_ti_spinor_field (double *sf_rot, double _Complex **R, double *sf, unsigned int N);

void rot_bispinor_mat_ti_fp_field( fermion_propagator_type *fp_rot, double _Complex ** R, fermion_propagator_type *fp, unsigned int N );

void rot_fp_field_ti_bispinor_mat ( fermion_propagator_type *fp_rot, double _Complex ** R, fermion_propagator_type *fp, unsigned int N);

void rot_spinor_field_ti_bispinor_mat ( double**sf_rot, double _Complex ** R, double**sf, unsigned int N );

void rot_bispinor_mat_ti_sp_field ( spinor_propagator_type *sp_rot, double _Complex ** R, spinor_propagator_type *sp, unsigned int N );

void rot_sp_field_ti_bispinor_mat ( spinor_propagator_type *sp_rot, double _Complex ** R, spinor_propagator_type *sp, unsigned int N );

void rot_inversion_matrix_spherical_basis ( double _Complex**R, int J2, int bispinor );

double rot_mat_diff_norm2 (double _Complex **R, double _Complex **S , int N );

double rot_mat_diff_norm (double _Complex **R, double _Complex **S , int N );

void wigner_d (double **wd, double b, int J2 );

int rot_rotation_matrix_spherical_basis_Wigner_D ( double _Complex**R, int J2, double a[3] );

double rot_mat_norm_diff ( double _Complex **R, double _Complex **S, int N );

void rot_mat_get_euler_angles ( double a[3], int n[3], double w );
 
int rot_mat_spin1_cartesian ( double _Complex **R, int n[3], double omega );

void rot_mat_ti_vec (double _Complex * const w, double _Complex ** const A, double _Complex * const v, int const N);

void rot_mat_transpose_ti_vec (double _Complex *w, double _Complex **A, double _Complex *v, int N);

void rot_mat_adjoint_ti_vec (double _Complex *w, double _Complex **A, double _Complex *v, int N);

void rot_vec_pl_eq_rot_vec_ti_co ( double _Complex * const w, double _Complex * const v, double _Complex const c, int const N);

void rot_vec_accum_vec_ti_co_pl_mat_ti_vec_ti_co (double _Complex *w, double _Complex **A, double _Complex *v, double _Complex cv, double _Complex cw, int N);

void rot_vec_accum_vec_ti_co_pl_mat_transpose_ti_vec_ti_co (double _Complex *w, double _Complex **A, double _Complex *v, double _Complex cv, double _Complex cw, int N);

int rot_mat_spin1_2_spherical ( double _Complex **R, int n[3], double omega );

void rot_mat_eq_mat_pl_mat (double _Complex **R, double _Complex **S1, double _Complex **S2, int N);

double _Complex rot_mat_trace ( double _Complex ** const R, int const N );

double _Complex rot_mat_trace_weight_re ( double _Complex ** const R, double * const weight, int const N );

double _Complex co_eq_trace_mat_ti_mat_weight_re ( double _Complex ** const A, double _Complex ** const B, double * const weight, int const num );

double _Complex co_eq_trace_mat_ti_weight_ti_mat_ti_weight_re ( double _Complex ** const A, double * const w1, double _Complex ** const B, double * const w2, int const num );

void rot_vec_pl_eq_vec_ti_co ( double _Complex*v, double _Complex*w, double _Complex c , int N );

void rot_vec_normalize ( double _Complex *v, int N );

void rot_mat_eq_diag_re_ti_mat ( double _Complex **A, double*lambda, double _Complex **B, int num );

void rot_mat_eq_mat_ti_diag_re ( double _Complex **A, double*lambda, double _Complex **B, int num );

void rot_mat_eq_diag_re_ti_mat_ti_diag_re ( double _Complex **A, double*lambda, double _Complex **B, int num );

void rot_vec_eq_mat_diag ( double _Complex *w, double _Complex **A, int num );

void co_pl_eq_mat_diag ( double _Complex * const w, double _Complex ** const A, int num );

void rot_vec_printf (double _Complex * const v, int const N, char *A, FILE*ofs );

/***********************************************************
 * check boundary status of a point
 ***********************************************************/
inline int rot_check_point_bnd ( int n[3] ) {
  int bnd = 0;
  if ( n[0] == LX || n[0] == -1) bnd++;
  if ( n[1] == LY || n[1] == -1) bnd++;
  if ( n[2] == LZ || n[2] == -1) bnd++;
  if (bnd > 0 && g_verbose > 2 && g_cart_id == 0 ) {
    fprintf(stderr, "# [rot_check_point_bnd] *******************************************************\n"\
                    "# [rot_check_point_bnd] * WARNING bnd = %d for n = (%3d, %3d, %3d)\n"\
                    "# [rot_check_point_bnd] *******************************************************\n",
                    bnd, n[0], n[1], n[2]);
    fflush(stdout);
  }
  return(bnd > 2);
}  /* end of rot_check_point_bnd */

/***********************************************************
 * reduce point coordinates according to MPI boundary
 ***********************************************************/
inline void rot_reduce_point_bnd ( int nred[3], int n[3] ) {
#ifndef HAVE_MPI
  nred[0] = ( n[0] + LX ) % LX;
  nred[1] = ( n[1] + LY ) % LY;
  nred[2] = ( n[2] + LZ ) % LZ;
#else
  nred[0] = n[0];
  nred[1] = n[1];
  nred[2] = n[2];
# if !( defined PARALLELTXYZ ) /* z-direction NOT parallelized */
  nred[2] = ( n[2] + LZ ) % LZ;
#  endif

#  if !( defined PARALLELTXY ) && !( defined PARALLELTXYZ )  /* y-direction NOT parallelized */
  nred[1] = ( n[1] + LY ) % LY;
#  endif

#  if !( defined PARALLELTX ) && !( defined PARALLELTXY ) && !( defined PARALLELTXYZ ) /* x-direction NOT parallelized */
  nred[0] = ( n[0] + LX ) % LX;
#  endif
#endif
  return;
}  /* end of rot_reduce_point_bnd */

/***********************************************************
 * allocate / deallocate rotation matrix
 ***********************************************************/
inline double _Complex **rot_init_rotation_matrix (int N ) {

  double _Complex ** SSpin = init_2level_ztable ( N, N );
  if ( SSpin == NULL ) {
    fprintf(stderr, "[rot_init_rotation_matrix] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(NULL);
  }
  return(SSpin);
}  /* end of rot_init_rotation_matrix */

inline double _Complex **rot_fini_rotation_matrix ( double _Complex ***R ) {
  fini_2level_ztable ( R );
  return(NULL);
}  /* end of rot_fini_rotation_matrix */


/***********************************************************
 * spin-rotate fermion vector
 *
 * safe, if _r == _s
 ***********************************************************/
inline void rot_bispinor_mat_ti_fv( double * _r, double _Complex ** _R, double * _s) {
  double _Complex _zspinor1[12];
  _zspinor1[ 0] = 0.;
  _zspinor1[ 0] += (_R)[0][0] * ( (_s)[ 0] + (_s)[ 1] * I);
  _zspinor1[ 0] += (_R)[0][1] * ( (_s)[ 6] + (_s)[ 7] * I);
  _zspinor1[ 0] += (_R)[0][2] * ( (_s)[12] + (_s)[13] * I);
  _zspinor1[ 0] += (_R)[0][3] * ( (_s)[18] + (_s)[19] * I);
  _zspinor1[ 1] = 0.;
  _zspinor1[ 1] += (_R)[0][0] * ( (_s)[ 2] + (_s)[ 3] * I);
  _zspinor1[ 1] += (_R)[0][1] * ( (_s)[ 8] + (_s)[ 9] * I);
  _zspinor1[ 1] += (_R)[0][2] * ( (_s)[14] + (_s)[15] * I);
  _zspinor1[ 1] += (_R)[0][3] * ( (_s)[20] + (_s)[21] * I);
  _zspinor1[ 2] = 0.;
  _zspinor1[ 2] += (_R)[0][0] * ( (_s)[ 4] + (_s)[ 5] * I);
  _zspinor1[ 2] += (_R)[0][1] * ( (_s)[10] + (_s)[11] * I);
  _zspinor1[ 2] += (_R)[0][2] * ( (_s)[16] + (_s)[17] * I);
  _zspinor1[ 2] += (_R)[0][3] * ( (_s)[22] + (_s)[23] * I);
  _zspinor1[ 3] = 0.;
  _zspinor1[ 3] += (_R)[1][0] * ( (_s)[ 0] + (_s)[ 1] * I);
  _zspinor1[ 3] += (_R)[1][1] * ( (_s)[ 6] + (_s)[ 7] * I);
  _zspinor1[ 3] += (_R)[1][2] * ( (_s)[12] + (_s)[13] * I);
  _zspinor1[ 3] += (_R)[1][3] * ( (_s)[18] + (_s)[19] * I);
  _zspinor1[ 4] = 0.;
  _zspinor1[ 4] += (_R)[1][0] * ( (_s)[ 2] + (_s)[ 3] * I);
  _zspinor1[ 4] += (_R)[1][1] * ( (_s)[ 8] + (_s)[ 9] * I);
  _zspinor1[ 4] += (_R)[1][2] * ( (_s)[14] + (_s)[15] * I);
  _zspinor1[ 4] += (_R)[1][3] * ( (_s)[20] + (_s)[21] * I);
  _zspinor1[ 5] = 0.;
  _zspinor1[ 5] += (_R)[1][0] * ( (_s)[ 4] + (_s)[ 5] * I);
  _zspinor1[ 5] += (_R)[1][1] * ( (_s)[10] + (_s)[11] * I);
  _zspinor1[ 5] += (_R)[1][2] * ( (_s)[16] + (_s)[17] * I);
  _zspinor1[ 5] += (_R)[1][3] * ( (_s)[22] + (_s)[23] * I);
  _zspinor1[ 6] = 0.;
  _zspinor1[ 6] += (_R)[2][0] * ( (_s)[ 0] + (_s)[ 1] * I);
  _zspinor1[ 6] += (_R)[2][1] * ( (_s)[ 6] + (_s)[ 7] * I);
  _zspinor1[ 6] += (_R)[2][2] * ( (_s)[12] + (_s)[13] * I);
  _zspinor1[ 6] += (_R)[2][3] * ( (_s)[18] + (_s)[19] * I);
  _zspinor1[ 7] = 0.;
  _zspinor1[ 7] += (_R)[2][0] * ( (_s)[ 2] + (_s)[ 3] * I);
  _zspinor1[ 7] += (_R)[2][1] * ( (_s)[ 8] + (_s)[ 9] * I);
  _zspinor1[ 7] += (_R)[2][2] * ( (_s)[14] + (_s)[15] * I);
  _zspinor1[ 7] += (_R)[2][3] * ( (_s)[20] + (_s)[21] * I);
  _zspinor1[ 8] = 0.;
  _zspinor1[ 8] += (_R)[2][0] * ( (_s)[ 4] + (_s)[ 5] * I);
  _zspinor1[ 8] += (_R)[2][1] * ( (_s)[10] + (_s)[11] * I);
  _zspinor1[ 8] += (_R)[2][2] * ( (_s)[16] + (_s)[17] * I);
  _zspinor1[ 8] += (_R)[2][3] * ( (_s)[22] + (_s)[23] * I);
  _zspinor1[ 9] = 0.;
  _zspinor1[ 9] += (_R)[3][0] * ( (_s)[ 0] + (_s)[ 1] * I);
  _zspinor1[ 9] += (_R)[3][1] * ( (_s)[ 6] + (_s)[ 7] * I);
  _zspinor1[ 9] += (_R)[3][2] * ( (_s)[12] + (_s)[13] * I);
  _zspinor1[ 9] += (_R)[3][3] * ( (_s)[18] + (_s)[19] * I);
  _zspinor1[10] = 0.;
  _zspinor1[10] += (_R)[3][0] * ( (_s)[ 2] + (_s)[ 3] * I);
  _zspinor1[10] += (_R)[3][1] * ( (_s)[ 8] + (_s)[ 9] * I);
  _zspinor1[10] += (_R)[3][2] * ( (_s)[14] + (_s)[15] * I);
  _zspinor1[10] += (_R)[3][3] * ( (_s)[20] + (_s)[21] * I);
  _zspinor1[11] = 0.;
  _zspinor1[11] += (_R)[3][0] * ( (_s)[ 4] + (_s)[ 5] * I);
  _zspinor1[11] += (_R)[3][1] * ( (_s)[10] + (_s)[11] * I);
  _zspinor1[11] += (_R)[3][2] * ( (_s)[16] + (_s)[17] * I);
  _zspinor1[11] += (_R)[3][3] * ( (_s)[22] + (_s)[23] * I);
  memcpy( (_r), _zspinor1, 24*sizeof(double) );
}  /* end of rot_bispinor_mat_ti_fv */


/***********************************************************
 * spin-rotate fermion propagator from the left
 *
 * safe, if _r == _s
 ***********************************************************/
inline void rot_bispinor_mat_ti_fp( fermion_propagator_type _r, double _Complex ** _R, fermion_propagator_type _s) {
  rot_bispinor_mat_ti_fv( (_r)[ 0], (_R), (_s)[ 0] );
  rot_bispinor_mat_ti_fv( (_r)[ 1], (_R), (_s)[ 1] );
  rot_bispinor_mat_ti_fv( (_r)[ 2], (_R), (_s)[ 2] );
  rot_bispinor_mat_ti_fv( (_r)[ 3], (_R), (_s)[ 3] );
  rot_bispinor_mat_ti_fv( (_r)[ 4], (_R), (_s)[ 4] );
  rot_bispinor_mat_ti_fv( (_r)[ 5], (_R), (_s)[ 5] );
  rot_bispinor_mat_ti_fv( (_r)[ 6], (_R), (_s)[ 6] );
  rot_bispinor_mat_ti_fv( (_r)[ 7], (_R), (_s)[ 7] );
  rot_bispinor_mat_ti_fv( (_r)[ 8], (_R), (_s)[ 8] );
  rot_bispinor_mat_ti_fv( (_r)[ 9], (_R), (_s)[ 9] );
  rot_bispinor_mat_ti_fv( (_r)[10], (_R), (_s)[10] );
  rot_bispinor_mat_ti_fv( (_r)[11], (_R), (_s)[11] );
}  /* rot_bispinor_mat_ti_fp */


/***********************************************************
 * spin-rotate fermion propagator from the right
 *
 * safe, if _r == _s
 ***********************************************************/
inline void rot_fp_ti_bispinor_mat ( fermion_propagator_type _r, double _Complex ** _R, fermion_propagator_type _s) {
  double _Complex _c;
  fermion_propagator_type _fp;
  create_fp( &_fp );
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][ 0] + (_s)[ 6][ 1] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][ 0] + (_s)[ 9][ 1] * I) *  (_R)[3][0];
  (_fp)[ 0][ 0] = creal(_c);
  (_fp)[ 0][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][ 2] + (_s)[ 6][ 3] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][ 2] + (_s)[ 9][ 3] * I) *  (_R)[3][0];
  (_fp)[ 0][ 2] = creal(_c);
  (_fp)[ 0][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][ 4] + (_s)[ 6][ 5] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][ 4] + (_s)[ 9][ 5] * I) *  (_R)[3][0];
  (_fp)[ 0][ 4] = creal(_c);
  (_fp)[ 0][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][ 6] + (_s)[ 6][ 7] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][ 6] + (_s)[ 9][ 7] * I) *  (_R)[3][0];
  (_fp)[ 0][ 6] = creal(_c);
  (_fp)[ 0][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 8] + (_s)[ 0][ 9] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][ 8] + (_s)[ 3][ 9] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][ 8] + (_s)[ 6][ 9] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][ 8] + (_s)[ 9][ 9] * I) *  (_R)[3][0];
  (_fp)[ 0][ 8] = creal(_c);
  (_fp)[ 0][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][10] + (_s)[ 0][11] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][10] + (_s)[ 3][11] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][10] + (_s)[ 6][11] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][10] + (_s)[ 9][11] * I) *  (_R)[3][0];
  (_fp)[ 0][10] = creal(_c);
  (_fp)[ 0][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][12] + (_s)[ 0][13] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][12] + (_s)[ 3][13] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][12] + (_s)[ 6][13] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][12] + (_s)[ 9][13] * I) *  (_R)[3][0];
  (_fp)[ 0][12] = creal(_c);
  (_fp)[ 0][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][14] + (_s)[ 0][15] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][14] + (_s)[ 3][15] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][14] + (_s)[ 6][15] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][14] + (_s)[ 9][15] * I) *  (_R)[3][0];
  (_fp)[ 0][14] = creal(_c);
  (_fp)[ 0][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][16] + (_s)[ 0][17] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][16] + (_s)[ 3][17] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][16] + (_s)[ 6][17] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][16] + (_s)[ 9][17] * I) *  (_R)[3][0];
  (_fp)[ 0][16] = creal(_c);
  (_fp)[ 0][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][18] + (_s)[ 0][19] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][18] + (_s)[ 3][19] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][18] + (_s)[ 6][19] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][18] + (_s)[ 9][19] * I) *  (_R)[3][0];
  (_fp)[ 0][18] = creal(_c);
  (_fp)[ 0][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][20] + (_s)[ 0][21] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][20] + (_s)[ 3][21] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][20] + (_s)[ 6][21] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][20] + (_s)[ 9][21] * I) *  (_R)[3][0];
  (_fp)[ 0][20] = creal(_c);
  (_fp)[ 0][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][22] + (_s)[ 0][23] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][22] + (_s)[ 3][23] * I) *  (_R)[1][0];
  _c += ( (_s)[ 6][22] + (_s)[ 6][23] * I) *  (_R)[2][0];
  _c += ( (_s)[ 9][22] + (_s)[ 9][23] * I) *  (_R)[3][0];
  (_fp)[ 0][22] = creal(_c);
  (_fp)[ 0][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][ 0] + (_s)[ 4][ 1] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][ 0] + (_s)[ 7][ 1] * I) *  (_R)[2][0];
  _c += ( (_s)[10][ 0] + (_s)[10][ 1] * I) *  (_R)[3][0];
  (_fp)[ 1][ 0] = creal(_c);
  (_fp)[ 1][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][ 2] + (_s)[ 4][ 3] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][ 2] + (_s)[ 7][ 3] * I) *  (_R)[2][0];
  _c += ( (_s)[10][ 2] + (_s)[10][ 3] * I) *  (_R)[3][0];
  (_fp)[ 1][ 2] = creal(_c);
  (_fp)[ 1][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][ 4] + (_s)[ 4][ 5] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][ 4] + (_s)[ 7][ 5] * I) *  (_R)[2][0];
  _c += ( (_s)[10][ 4] + (_s)[10][ 5] * I) *  (_R)[3][0];
  (_fp)[ 1][ 4] = creal(_c);
  (_fp)[ 1][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][ 6] + (_s)[ 4][ 7] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][ 6] + (_s)[ 7][ 7] * I) *  (_R)[2][0];
  _c += ( (_s)[10][ 6] + (_s)[10][ 7] * I) *  (_R)[3][0];
  (_fp)[ 1][ 6] = creal(_c);
  (_fp)[ 1][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 8] + (_s)[ 1][ 9] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][ 8] + (_s)[ 4][ 9] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][ 8] + (_s)[ 7][ 9] * I) *  (_R)[2][0];
  _c += ( (_s)[10][ 8] + (_s)[10][ 9] * I) *  (_R)[3][0];
  (_fp)[ 1][ 8] = creal(_c);
  (_fp)[ 1][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][10] + (_s)[ 1][11] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][10] + (_s)[ 4][11] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][10] + (_s)[ 7][11] * I) *  (_R)[2][0];
  _c += ( (_s)[10][10] + (_s)[10][11] * I) *  (_R)[3][0];
  (_fp)[ 1][10] = creal(_c);
  (_fp)[ 1][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][12] + (_s)[ 1][13] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][12] + (_s)[ 4][13] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][12] + (_s)[ 7][13] * I) *  (_R)[2][0];
  _c += ( (_s)[10][12] + (_s)[10][13] * I) *  (_R)[3][0];
  (_fp)[ 1][12] = creal(_c);
  (_fp)[ 1][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][14] + (_s)[ 1][15] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][14] + (_s)[ 4][15] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][14] + (_s)[ 7][15] * I) *  (_R)[2][0];
  _c += ( (_s)[10][14] + (_s)[10][15] * I) *  (_R)[3][0];
  (_fp)[ 1][14] = creal(_c);
  (_fp)[ 1][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][16] + (_s)[ 1][17] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][16] + (_s)[ 4][17] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][16] + (_s)[ 7][17] * I) *  (_R)[2][0];
  _c += ( (_s)[10][16] + (_s)[10][17] * I) *  (_R)[3][0];
  (_fp)[ 1][16] = creal(_c);
  (_fp)[ 1][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][18] + (_s)[ 1][19] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][18] + (_s)[ 4][19] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][18] + (_s)[ 7][19] * I) *  (_R)[2][0];
  _c += ( (_s)[10][18] + (_s)[10][19] * I) *  (_R)[3][0];
  (_fp)[ 1][18] = creal(_c);
  (_fp)[ 1][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][20] + (_s)[ 1][21] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][20] + (_s)[ 4][21] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][20] + (_s)[ 7][21] * I) *  (_R)[2][0];
  _c += ( (_s)[10][20] + (_s)[10][21] * I) *  (_R)[3][0];
  (_fp)[ 1][20] = creal(_c);
  (_fp)[ 1][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][22] + (_s)[ 1][23] * I) *  (_R)[0][0];
  _c += ( (_s)[ 4][22] + (_s)[ 4][23] * I) *  (_R)[1][0];
  _c += ( (_s)[ 7][22] + (_s)[ 7][23] * I) *  (_R)[2][0];
  _c += ( (_s)[10][22] + (_s)[10][23] * I) *  (_R)[3][0];
  (_fp)[ 1][22] = creal(_c);
  (_fp)[ 1][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][ 0] + (_s)[ 5][ 1] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][ 0] + (_s)[ 8][ 1] * I) *  (_R)[2][0];
  _c += ( (_s)[11][ 0] + (_s)[11][ 1] * I) *  (_R)[3][0];
  (_fp)[ 2][ 0] = creal(_c);
  (_fp)[ 2][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][ 2] + (_s)[ 5][ 3] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][ 2] + (_s)[ 8][ 3] * I) *  (_R)[2][0];
  _c += ( (_s)[11][ 2] + (_s)[11][ 3] * I) *  (_R)[3][0];
  (_fp)[ 2][ 2] = creal(_c);
  (_fp)[ 2][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][ 4] + (_s)[ 5][ 5] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][ 4] + (_s)[ 8][ 5] * I) *  (_R)[2][0];
  _c += ( (_s)[11][ 4] + (_s)[11][ 5] * I) *  (_R)[3][0];
  (_fp)[ 2][ 4] = creal(_c);
  (_fp)[ 2][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][ 6] + (_s)[ 5][ 7] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][ 6] + (_s)[ 8][ 7] * I) *  (_R)[2][0];
  _c += ( (_s)[11][ 6] + (_s)[11][ 7] * I) *  (_R)[3][0];
  (_fp)[ 2][ 6] = creal(_c);
  (_fp)[ 2][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 8] + (_s)[ 2][ 9] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][ 8] + (_s)[ 5][ 9] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][ 8] + (_s)[ 8][ 9] * I) *  (_R)[2][0];
  _c += ( (_s)[11][ 8] + (_s)[11][ 9] * I) *  (_R)[3][0];
  (_fp)[ 2][ 8] = creal(_c);
  (_fp)[ 2][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][10] + (_s)[ 2][11] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][10] + (_s)[ 5][11] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][10] + (_s)[ 8][11] * I) *  (_R)[2][0];
  _c += ( (_s)[11][10] + (_s)[11][11] * I) *  (_R)[3][0];
  (_fp)[ 2][10] = creal(_c);
  (_fp)[ 2][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][12] + (_s)[ 2][13] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][12] + (_s)[ 5][13] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][12] + (_s)[ 8][13] * I) *  (_R)[2][0];
  _c += ( (_s)[11][12] + (_s)[11][13] * I) *  (_R)[3][0];
  (_fp)[ 2][12] = creal(_c);
  (_fp)[ 2][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][14] + (_s)[ 2][15] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][14] + (_s)[ 5][15] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][14] + (_s)[ 8][15] * I) *  (_R)[2][0];
  _c += ( (_s)[11][14] + (_s)[11][15] * I) *  (_R)[3][0];
  (_fp)[ 2][14] = creal(_c);
  (_fp)[ 2][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][16] + (_s)[ 2][17] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][16] + (_s)[ 5][17] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][16] + (_s)[ 8][17] * I) *  (_R)[2][0];
  _c += ( (_s)[11][16] + (_s)[11][17] * I) *  (_R)[3][0];
  (_fp)[ 2][16] = creal(_c);
  (_fp)[ 2][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][18] + (_s)[ 2][19] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][18] + (_s)[ 5][19] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][18] + (_s)[ 8][19] * I) *  (_R)[2][0];
  _c += ( (_s)[11][18] + (_s)[11][19] * I) *  (_R)[3][0];
  (_fp)[ 2][18] = creal(_c);
  (_fp)[ 2][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][20] + (_s)[ 2][21] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][20] + (_s)[ 5][21] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][20] + (_s)[ 8][21] * I) *  (_R)[2][0];
  _c += ( (_s)[11][20] + (_s)[11][21] * I) *  (_R)[3][0];
  (_fp)[ 2][20] = creal(_c);
  (_fp)[ 2][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][22] + (_s)[ 2][23] * I) *  (_R)[0][0];
  _c += ( (_s)[ 5][22] + (_s)[ 5][23] * I) *  (_R)[1][0];
  _c += ( (_s)[ 8][22] + (_s)[ 8][23] * I) *  (_R)[2][0];
  _c += ( (_s)[11][22] + (_s)[11][23] * I) *  (_R)[3][0];
  (_fp)[ 2][22] = creal(_c);
  (_fp)[ 2][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][ 0] + (_s)[ 6][ 1] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][ 0] + (_s)[ 9][ 1] * I) *  (_R)[3][1];
  (_fp)[ 3][ 0] = creal(_c);
  (_fp)[ 3][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][ 2] + (_s)[ 6][ 3] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][ 2] + (_s)[ 9][ 3] * I) *  (_R)[3][1];
  (_fp)[ 3][ 2] = creal(_c);
  (_fp)[ 3][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][ 4] + (_s)[ 6][ 5] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][ 4] + (_s)[ 9][ 5] * I) *  (_R)[3][1];
  (_fp)[ 3][ 4] = creal(_c);
  (_fp)[ 3][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][ 6] + (_s)[ 6][ 7] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][ 6] + (_s)[ 9][ 7] * I) *  (_R)[3][1];
  (_fp)[ 3][ 6] = creal(_c);
  (_fp)[ 3][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 8] + (_s)[ 0][ 9] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][ 8] + (_s)[ 3][ 9] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][ 8] + (_s)[ 6][ 9] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][ 8] + (_s)[ 9][ 9] * I) *  (_R)[3][1];
  (_fp)[ 3][ 8] = creal(_c);
  (_fp)[ 3][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][10] + (_s)[ 0][11] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][10] + (_s)[ 3][11] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][10] + (_s)[ 6][11] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][10] + (_s)[ 9][11] * I) *  (_R)[3][1];
  (_fp)[ 3][10] = creal(_c);
  (_fp)[ 3][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][12] + (_s)[ 0][13] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][12] + (_s)[ 3][13] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][12] + (_s)[ 6][13] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][12] + (_s)[ 9][13] * I) *  (_R)[3][1];
  (_fp)[ 3][12] = creal(_c);
  (_fp)[ 3][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][14] + (_s)[ 0][15] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][14] + (_s)[ 3][15] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][14] + (_s)[ 6][15] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][14] + (_s)[ 9][15] * I) *  (_R)[3][1];
  (_fp)[ 3][14] = creal(_c);
  (_fp)[ 3][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][16] + (_s)[ 0][17] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][16] + (_s)[ 3][17] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][16] + (_s)[ 6][17] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][16] + (_s)[ 9][17] * I) *  (_R)[3][1];
  (_fp)[ 3][16] = creal(_c);
  (_fp)[ 3][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][18] + (_s)[ 0][19] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][18] + (_s)[ 3][19] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][18] + (_s)[ 6][19] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][18] + (_s)[ 9][19] * I) *  (_R)[3][1];
  (_fp)[ 3][18] = creal(_c);
  (_fp)[ 3][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][20] + (_s)[ 0][21] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][20] + (_s)[ 3][21] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][20] + (_s)[ 6][21] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][20] + (_s)[ 9][21] * I) *  (_R)[3][1];
  (_fp)[ 3][20] = creal(_c);
  (_fp)[ 3][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][22] + (_s)[ 0][23] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][22] + (_s)[ 3][23] * I) *  (_R)[1][1];
  _c += ( (_s)[ 6][22] + (_s)[ 6][23] * I) *  (_R)[2][1];
  _c += ( (_s)[ 9][22] + (_s)[ 9][23] * I) *  (_R)[3][1];
  (_fp)[ 3][22] = creal(_c);
  (_fp)[ 3][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][ 0] + (_s)[ 4][ 1] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][ 0] + (_s)[ 7][ 1] * I) *  (_R)[2][1];
  _c += ( (_s)[10][ 0] + (_s)[10][ 1] * I) *  (_R)[3][1];
  (_fp)[ 4][ 0] = creal(_c);
  (_fp)[ 4][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][ 2] + (_s)[ 4][ 3] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][ 2] + (_s)[ 7][ 3] * I) *  (_R)[2][1];
  _c += ( (_s)[10][ 2] + (_s)[10][ 3] * I) *  (_R)[3][1];
  (_fp)[ 4][ 2] = creal(_c);
  (_fp)[ 4][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][ 4] + (_s)[ 4][ 5] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][ 4] + (_s)[ 7][ 5] * I) *  (_R)[2][1];
  _c += ( (_s)[10][ 4] + (_s)[10][ 5] * I) *  (_R)[3][1];
  (_fp)[ 4][ 4] = creal(_c);
  (_fp)[ 4][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][ 6] + (_s)[ 4][ 7] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][ 6] + (_s)[ 7][ 7] * I) *  (_R)[2][1];
  _c += ( (_s)[10][ 6] + (_s)[10][ 7] * I) *  (_R)[3][1];
  (_fp)[ 4][ 6] = creal(_c);
  (_fp)[ 4][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 8] + (_s)[ 1][ 9] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][ 8] + (_s)[ 4][ 9] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][ 8] + (_s)[ 7][ 9] * I) *  (_R)[2][1];
  _c += ( (_s)[10][ 8] + (_s)[10][ 9] * I) *  (_R)[3][1];
  (_fp)[ 4][ 8] = creal(_c);
  (_fp)[ 4][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][10] + (_s)[ 1][11] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][10] + (_s)[ 4][11] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][10] + (_s)[ 7][11] * I) *  (_R)[2][1];
  _c += ( (_s)[10][10] + (_s)[10][11] * I) *  (_R)[3][1];
  (_fp)[ 4][10] = creal(_c);
  (_fp)[ 4][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][12] + (_s)[ 1][13] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][12] + (_s)[ 4][13] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][12] + (_s)[ 7][13] * I) *  (_R)[2][1];
  _c += ( (_s)[10][12] + (_s)[10][13] * I) *  (_R)[3][1];
  (_fp)[ 4][12] = creal(_c);
  (_fp)[ 4][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][14] + (_s)[ 1][15] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][14] + (_s)[ 4][15] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][14] + (_s)[ 7][15] * I) *  (_R)[2][1];
  _c += ( (_s)[10][14] + (_s)[10][15] * I) *  (_R)[3][1];
  (_fp)[ 4][14] = creal(_c);
  (_fp)[ 4][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][16] + (_s)[ 1][17] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][16] + (_s)[ 4][17] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][16] + (_s)[ 7][17] * I) *  (_R)[2][1];
  _c += ( (_s)[10][16] + (_s)[10][17] * I) *  (_R)[3][1];
  (_fp)[ 4][16] = creal(_c);
  (_fp)[ 4][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][18] + (_s)[ 1][19] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][18] + (_s)[ 4][19] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][18] + (_s)[ 7][19] * I) *  (_R)[2][1];
  _c += ( (_s)[10][18] + (_s)[10][19] * I) *  (_R)[3][1];
  (_fp)[ 4][18] = creal(_c);
  (_fp)[ 4][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][20] + (_s)[ 1][21] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][20] + (_s)[ 4][21] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][20] + (_s)[ 7][21] * I) *  (_R)[2][1];
  _c += ( (_s)[10][20] + (_s)[10][21] * I) *  (_R)[3][1];
  (_fp)[ 4][20] = creal(_c);
  (_fp)[ 4][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][22] + (_s)[ 1][23] * I) *  (_R)[0][1];
  _c += ( (_s)[ 4][22] + (_s)[ 4][23] * I) *  (_R)[1][1];
  _c += ( (_s)[ 7][22] + (_s)[ 7][23] * I) *  (_R)[2][1];
  _c += ( (_s)[10][22] + (_s)[10][23] * I) *  (_R)[3][1];
  (_fp)[ 4][22] = creal(_c);
  (_fp)[ 4][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][ 0] + (_s)[ 5][ 1] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][ 0] + (_s)[ 8][ 1] * I) *  (_R)[2][1];
  _c += ( (_s)[11][ 0] + (_s)[11][ 1] * I) *  (_R)[3][1];
  (_fp)[ 5][ 0] = creal(_c);
  (_fp)[ 5][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][ 2] + (_s)[ 5][ 3] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][ 2] + (_s)[ 8][ 3] * I) *  (_R)[2][1];
  _c += ( (_s)[11][ 2] + (_s)[11][ 3] * I) *  (_R)[3][1];
  (_fp)[ 5][ 2] = creal(_c);
  (_fp)[ 5][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][ 4] + (_s)[ 5][ 5] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][ 4] + (_s)[ 8][ 5] * I) *  (_R)[2][1];
  _c += ( (_s)[11][ 4] + (_s)[11][ 5] * I) *  (_R)[3][1];
  (_fp)[ 5][ 4] = creal(_c);
  (_fp)[ 5][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][ 6] + (_s)[ 5][ 7] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][ 6] + (_s)[ 8][ 7] * I) *  (_R)[2][1];
  _c += ( (_s)[11][ 6] + (_s)[11][ 7] * I) *  (_R)[3][1];
  (_fp)[ 5][ 6] = creal(_c);
  (_fp)[ 5][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 8] + (_s)[ 2][ 9] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][ 8] + (_s)[ 5][ 9] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][ 8] + (_s)[ 8][ 9] * I) *  (_R)[2][1];
  _c += ( (_s)[11][ 8] + (_s)[11][ 9] * I) *  (_R)[3][1];
  (_fp)[ 5][ 8] = creal(_c);
  (_fp)[ 5][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][10] + (_s)[ 2][11] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][10] + (_s)[ 5][11] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][10] + (_s)[ 8][11] * I) *  (_R)[2][1];
  _c += ( (_s)[11][10] + (_s)[11][11] * I) *  (_R)[3][1];
  (_fp)[ 5][10] = creal(_c);
  (_fp)[ 5][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][12] + (_s)[ 2][13] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][12] + (_s)[ 5][13] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][12] + (_s)[ 8][13] * I) *  (_R)[2][1];
  _c += ( (_s)[11][12] + (_s)[11][13] * I) *  (_R)[3][1];
  (_fp)[ 5][12] = creal(_c);
  (_fp)[ 5][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][14] + (_s)[ 2][15] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][14] + (_s)[ 5][15] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][14] + (_s)[ 8][15] * I) *  (_R)[2][1];
  _c += ( (_s)[11][14] + (_s)[11][15] * I) *  (_R)[3][1];
  (_fp)[ 5][14] = creal(_c);
  (_fp)[ 5][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][16] + (_s)[ 2][17] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][16] + (_s)[ 5][17] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][16] + (_s)[ 8][17] * I) *  (_R)[2][1];
  _c += ( (_s)[11][16] + (_s)[11][17] * I) *  (_R)[3][1];
  (_fp)[ 5][16] = creal(_c);
  (_fp)[ 5][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][18] + (_s)[ 2][19] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][18] + (_s)[ 5][19] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][18] + (_s)[ 8][19] * I) *  (_R)[2][1];
  _c += ( (_s)[11][18] + (_s)[11][19] * I) *  (_R)[3][1];
  (_fp)[ 5][18] = creal(_c);
  (_fp)[ 5][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][20] + (_s)[ 2][21] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][20] + (_s)[ 5][21] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][20] + (_s)[ 8][21] * I) *  (_R)[2][1];
  _c += ( (_s)[11][20] + (_s)[11][21] * I) *  (_R)[3][1];
  (_fp)[ 5][20] = creal(_c);
  (_fp)[ 5][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][22] + (_s)[ 2][23] * I) *  (_R)[0][1];
  _c += ( (_s)[ 5][22] + (_s)[ 5][23] * I) *  (_R)[1][1];
  _c += ( (_s)[ 8][22] + (_s)[ 8][23] * I) *  (_R)[2][1];
  _c += ( (_s)[11][22] + (_s)[11][23] * I) *  (_R)[3][1];
  (_fp)[ 5][22] = creal(_c);
  (_fp)[ 5][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][ 0] + (_s)[ 6][ 1] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][ 0] + (_s)[ 9][ 1] * I) *  (_R)[3][2];
  (_fp)[ 6][ 0] = creal(_c);
  (_fp)[ 6][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][ 2] + (_s)[ 6][ 3] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][ 2] + (_s)[ 9][ 3] * I) *  (_R)[3][2];
  (_fp)[ 6][ 2] = creal(_c);
  (_fp)[ 6][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][ 4] + (_s)[ 6][ 5] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][ 4] + (_s)[ 9][ 5] * I) *  (_R)[3][2];
  (_fp)[ 6][ 4] = creal(_c);
  (_fp)[ 6][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][ 6] + (_s)[ 6][ 7] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][ 6] + (_s)[ 9][ 7] * I) *  (_R)[3][2];
  (_fp)[ 6][ 6] = creal(_c);
  (_fp)[ 6][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 8] + (_s)[ 0][ 9] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][ 8] + (_s)[ 3][ 9] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][ 8] + (_s)[ 6][ 9] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][ 8] + (_s)[ 9][ 9] * I) *  (_R)[3][2];
  (_fp)[ 6][ 8] = creal(_c);
  (_fp)[ 6][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][10] + (_s)[ 0][11] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][10] + (_s)[ 3][11] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][10] + (_s)[ 6][11] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][10] + (_s)[ 9][11] * I) *  (_R)[3][2];
  (_fp)[ 6][10] = creal(_c);
  (_fp)[ 6][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][12] + (_s)[ 0][13] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][12] + (_s)[ 3][13] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][12] + (_s)[ 6][13] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][12] + (_s)[ 9][13] * I) *  (_R)[3][2];
  (_fp)[ 6][12] = creal(_c);
  (_fp)[ 6][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][14] + (_s)[ 0][15] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][14] + (_s)[ 3][15] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][14] + (_s)[ 6][15] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][14] + (_s)[ 9][15] * I) *  (_R)[3][2];
  (_fp)[ 6][14] = creal(_c);
  (_fp)[ 6][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][16] + (_s)[ 0][17] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][16] + (_s)[ 3][17] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][16] + (_s)[ 6][17] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][16] + (_s)[ 9][17] * I) *  (_R)[3][2];
  (_fp)[ 6][16] = creal(_c);
  (_fp)[ 6][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][18] + (_s)[ 0][19] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][18] + (_s)[ 3][19] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][18] + (_s)[ 6][19] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][18] + (_s)[ 9][19] * I) *  (_R)[3][2];
  (_fp)[ 6][18] = creal(_c);
  (_fp)[ 6][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][20] + (_s)[ 0][21] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][20] + (_s)[ 3][21] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][20] + (_s)[ 6][21] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][20] + (_s)[ 9][21] * I) *  (_R)[3][2];
  (_fp)[ 6][20] = creal(_c);
  (_fp)[ 6][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][22] + (_s)[ 0][23] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][22] + (_s)[ 3][23] * I) *  (_R)[1][2];
  _c += ( (_s)[ 6][22] + (_s)[ 6][23] * I) *  (_R)[2][2];
  _c += ( (_s)[ 9][22] + (_s)[ 9][23] * I) *  (_R)[3][2];
  (_fp)[ 6][22] = creal(_c);
  (_fp)[ 6][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][ 0] + (_s)[ 4][ 1] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][ 0] + (_s)[ 7][ 1] * I) *  (_R)[2][2];
  _c += ( (_s)[10][ 0] + (_s)[10][ 1] * I) *  (_R)[3][2];
  (_fp)[ 7][ 0] = creal(_c);
  (_fp)[ 7][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][ 2] + (_s)[ 4][ 3] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][ 2] + (_s)[ 7][ 3] * I) *  (_R)[2][2];
  _c += ( (_s)[10][ 2] + (_s)[10][ 3] * I) *  (_R)[3][2];
  (_fp)[ 7][ 2] = creal(_c);
  (_fp)[ 7][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][ 4] + (_s)[ 4][ 5] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][ 4] + (_s)[ 7][ 5] * I) *  (_R)[2][2];
  _c += ( (_s)[10][ 4] + (_s)[10][ 5] * I) *  (_R)[3][2];
  (_fp)[ 7][ 4] = creal(_c);
  (_fp)[ 7][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][ 6] + (_s)[ 4][ 7] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][ 6] + (_s)[ 7][ 7] * I) *  (_R)[2][2];
  _c += ( (_s)[10][ 6] + (_s)[10][ 7] * I) *  (_R)[3][2];
  (_fp)[ 7][ 6] = creal(_c);
  (_fp)[ 7][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 8] + (_s)[ 1][ 9] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][ 8] + (_s)[ 4][ 9] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][ 8] + (_s)[ 7][ 9] * I) *  (_R)[2][2];
  _c += ( (_s)[10][ 8] + (_s)[10][ 9] * I) *  (_R)[3][2];
  (_fp)[ 7][ 8] = creal(_c);
  (_fp)[ 7][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][10] + (_s)[ 1][11] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][10] + (_s)[ 4][11] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][10] + (_s)[ 7][11] * I) *  (_R)[2][2];
  _c += ( (_s)[10][10] + (_s)[10][11] * I) *  (_R)[3][2];
  (_fp)[ 7][10] = creal(_c);
  (_fp)[ 7][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][12] + (_s)[ 1][13] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][12] + (_s)[ 4][13] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][12] + (_s)[ 7][13] * I) *  (_R)[2][2];
  _c += ( (_s)[10][12] + (_s)[10][13] * I) *  (_R)[3][2];
  (_fp)[ 7][12] = creal(_c);
  (_fp)[ 7][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][14] + (_s)[ 1][15] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][14] + (_s)[ 4][15] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][14] + (_s)[ 7][15] * I) *  (_R)[2][2];
  _c += ( (_s)[10][14] + (_s)[10][15] * I) *  (_R)[3][2];
  (_fp)[ 7][14] = creal(_c);
  (_fp)[ 7][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][16] + (_s)[ 1][17] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][16] + (_s)[ 4][17] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][16] + (_s)[ 7][17] * I) *  (_R)[2][2];
  _c += ( (_s)[10][16] + (_s)[10][17] * I) *  (_R)[3][2];
  (_fp)[ 7][16] = creal(_c);
  (_fp)[ 7][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][18] + (_s)[ 1][19] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][18] + (_s)[ 4][19] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][18] + (_s)[ 7][19] * I) *  (_R)[2][2];
  _c += ( (_s)[10][18] + (_s)[10][19] * I) *  (_R)[3][2];
  (_fp)[ 7][18] = creal(_c);
  (_fp)[ 7][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][20] + (_s)[ 1][21] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][20] + (_s)[ 4][21] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][20] + (_s)[ 7][21] * I) *  (_R)[2][2];
  _c += ( (_s)[10][20] + (_s)[10][21] * I) *  (_R)[3][2];
  (_fp)[ 7][20] = creal(_c);
  (_fp)[ 7][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][22] + (_s)[ 1][23] * I) *  (_R)[0][2];
  _c += ( (_s)[ 4][22] + (_s)[ 4][23] * I) *  (_R)[1][2];
  _c += ( (_s)[ 7][22] + (_s)[ 7][23] * I) *  (_R)[2][2];
  _c += ( (_s)[10][22] + (_s)[10][23] * I) *  (_R)[3][2];
  (_fp)[ 7][22] = creal(_c);
  (_fp)[ 7][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][ 0] + (_s)[ 5][ 1] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][ 0] + (_s)[ 8][ 1] * I) *  (_R)[2][2];
  _c += ( (_s)[11][ 0] + (_s)[11][ 1] * I) *  (_R)[3][2];
  (_fp)[ 8][ 0] = creal(_c);
  (_fp)[ 8][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][ 2] + (_s)[ 5][ 3] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][ 2] + (_s)[ 8][ 3] * I) *  (_R)[2][2];
  _c += ( (_s)[11][ 2] + (_s)[11][ 3] * I) *  (_R)[3][2];
  (_fp)[ 8][ 2] = creal(_c);
  (_fp)[ 8][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][ 4] + (_s)[ 5][ 5] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][ 4] + (_s)[ 8][ 5] * I) *  (_R)[2][2];
  _c += ( (_s)[11][ 4] + (_s)[11][ 5] * I) *  (_R)[3][2];
  (_fp)[ 8][ 4] = creal(_c);
  (_fp)[ 8][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][ 6] + (_s)[ 5][ 7] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][ 6] + (_s)[ 8][ 7] * I) *  (_R)[2][2];
  _c += ( (_s)[11][ 6] + (_s)[11][ 7] * I) *  (_R)[3][2];
  (_fp)[ 8][ 6] = creal(_c);
  (_fp)[ 8][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 8] + (_s)[ 2][ 9] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][ 8] + (_s)[ 5][ 9] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][ 8] + (_s)[ 8][ 9] * I) *  (_R)[2][2];
  _c += ( (_s)[11][ 8] + (_s)[11][ 9] * I) *  (_R)[3][2];
  (_fp)[ 8][ 8] = creal(_c);
  (_fp)[ 8][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][10] + (_s)[ 2][11] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][10] + (_s)[ 5][11] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][10] + (_s)[ 8][11] * I) *  (_R)[2][2];
  _c += ( (_s)[11][10] + (_s)[11][11] * I) *  (_R)[3][2];
  (_fp)[ 8][10] = creal(_c);
  (_fp)[ 8][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][12] + (_s)[ 2][13] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][12] + (_s)[ 5][13] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][12] + (_s)[ 8][13] * I) *  (_R)[2][2];
  _c += ( (_s)[11][12] + (_s)[11][13] * I) *  (_R)[3][2];
  (_fp)[ 8][12] = creal(_c);
  (_fp)[ 8][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][14] + (_s)[ 2][15] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][14] + (_s)[ 5][15] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][14] + (_s)[ 8][15] * I) *  (_R)[2][2];
  _c += ( (_s)[11][14] + (_s)[11][15] * I) *  (_R)[3][2];
  (_fp)[ 8][14] = creal(_c);
  (_fp)[ 8][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][16] + (_s)[ 2][17] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][16] + (_s)[ 5][17] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][16] + (_s)[ 8][17] * I) *  (_R)[2][2];
  _c += ( (_s)[11][16] + (_s)[11][17] * I) *  (_R)[3][2];
  (_fp)[ 8][16] = creal(_c);
  (_fp)[ 8][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][18] + (_s)[ 2][19] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][18] + (_s)[ 5][19] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][18] + (_s)[ 8][19] * I) *  (_R)[2][2];
  _c += ( (_s)[11][18] + (_s)[11][19] * I) *  (_R)[3][2];
  (_fp)[ 8][18] = creal(_c);
  (_fp)[ 8][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][20] + (_s)[ 2][21] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][20] + (_s)[ 5][21] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][20] + (_s)[ 8][21] * I) *  (_R)[2][2];
  _c += ( (_s)[11][20] + (_s)[11][21] * I) *  (_R)[3][2];
  (_fp)[ 8][20] = creal(_c);
  (_fp)[ 8][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][22] + (_s)[ 2][23] * I) *  (_R)[0][2];
  _c += ( (_s)[ 5][22] + (_s)[ 5][23] * I) *  (_R)[1][2];
  _c += ( (_s)[ 8][22] + (_s)[ 8][23] * I) *  (_R)[2][2];
  _c += ( (_s)[11][22] + (_s)[11][23] * I) *  (_R)[3][2];
  (_fp)[ 8][22] = creal(_c);
  (_fp)[ 8][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][ 0] + (_s)[ 6][ 1] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][ 0] + (_s)[ 9][ 1] * I) *  (_R)[3][3];
  (_fp)[ 9][ 0] = creal(_c);
  (_fp)[ 9][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][ 2] + (_s)[ 6][ 3] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][ 2] + (_s)[ 9][ 3] * I) *  (_R)[3][3];
  (_fp)[ 9][ 2] = creal(_c);
  (_fp)[ 9][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][ 4] + (_s)[ 6][ 5] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][ 4] + (_s)[ 9][ 5] * I) *  (_R)[3][3];
  (_fp)[ 9][ 4] = creal(_c);
  (_fp)[ 9][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][ 6] + (_s)[ 6][ 7] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][ 6] + (_s)[ 9][ 7] * I) *  (_R)[3][3];
  (_fp)[ 9][ 6] = creal(_c);
  (_fp)[ 9][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 8] + (_s)[ 0][ 9] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][ 8] + (_s)[ 3][ 9] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][ 8] + (_s)[ 6][ 9] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][ 8] + (_s)[ 9][ 9] * I) *  (_R)[3][3];
  (_fp)[ 9][ 8] = creal(_c);
  (_fp)[ 9][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][10] + (_s)[ 0][11] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][10] + (_s)[ 3][11] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][10] + (_s)[ 6][11] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][10] + (_s)[ 9][11] * I) *  (_R)[3][3];
  (_fp)[ 9][10] = creal(_c);
  (_fp)[ 9][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][12] + (_s)[ 0][13] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][12] + (_s)[ 3][13] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][12] + (_s)[ 6][13] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][12] + (_s)[ 9][13] * I) *  (_R)[3][3];
  (_fp)[ 9][12] = creal(_c);
  (_fp)[ 9][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][14] + (_s)[ 0][15] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][14] + (_s)[ 3][15] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][14] + (_s)[ 6][15] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][14] + (_s)[ 9][15] * I) *  (_R)[3][3];
  (_fp)[ 9][14] = creal(_c);
  (_fp)[ 9][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][16] + (_s)[ 0][17] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][16] + (_s)[ 3][17] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][16] + (_s)[ 6][17] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][16] + (_s)[ 9][17] * I) *  (_R)[3][3];
  (_fp)[ 9][16] = creal(_c);
  (_fp)[ 9][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][18] + (_s)[ 0][19] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][18] + (_s)[ 3][19] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][18] + (_s)[ 6][19] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][18] + (_s)[ 9][19] * I) *  (_R)[3][3];
  (_fp)[ 9][18] = creal(_c);
  (_fp)[ 9][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][20] + (_s)[ 0][21] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][20] + (_s)[ 3][21] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][20] + (_s)[ 6][21] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][20] + (_s)[ 9][21] * I) *  (_R)[3][3];
  (_fp)[ 9][20] = creal(_c);
  (_fp)[ 9][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][22] + (_s)[ 0][23] * I) *  (_R)[0][3];
  _c += ( (_s)[ 3][22] + (_s)[ 3][23] * I) *  (_R)[1][3];
  _c += ( (_s)[ 6][22] + (_s)[ 6][23] * I) *  (_R)[2][3];
  _c += ( (_s)[ 9][22] + (_s)[ 9][23] * I) *  (_R)[3][3];
  (_fp)[ 9][22] = creal(_c);
  (_fp)[ 9][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][ 0] + (_s)[ 4][ 1] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][ 0] + (_s)[ 7][ 1] * I) *  (_R)[2][3];
  _c += ( (_s)[10][ 0] + (_s)[10][ 1] * I) *  (_R)[3][3];
  (_fp)[10][ 0] = creal(_c);
  (_fp)[10][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][ 2] + (_s)[ 4][ 3] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][ 2] + (_s)[ 7][ 3] * I) *  (_R)[2][3];
  _c += ( (_s)[10][ 2] + (_s)[10][ 3] * I) *  (_R)[3][3];
  (_fp)[10][ 2] = creal(_c);
  (_fp)[10][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][ 4] + (_s)[ 4][ 5] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][ 4] + (_s)[ 7][ 5] * I) *  (_R)[2][3];
  _c += ( (_s)[10][ 4] + (_s)[10][ 5] * I) *  (_R)[3][3];
  (_fp)[10][ 4] = creal(_c);
  (_fp)[10][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][ 6] + (_s)[ 4][ 7] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][ 6] + (_s)[ 7][ 7] * I) *  (_R)[2][3];
  _c += ( (_s)[10][ 6] + (_s)[10][ 7] * I) *  (_R)[3][3];
  (_fp)[10][ 6] = creal(_c);
  (_fp)[10][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 8] + (_s)[ 1][ 9] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][ 8] + (_s)[ 4][ 9] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][ 8] + (_s)[ 7][ 9] * I) *  (_R)[2][3];
  _c += ( (_s)[10][ 8] + (_s)[10][ 9] * I) *  (_R)[3][3];
  (_fp)[10][ 8] = creal(_c);
  (_fp)[10][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][10] + (_s)[ 1][11] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][10] + (_s)[ 4][11] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][10] + (_s)[ 7][11] * I) *  (_R)[2][3];
  _c += ( (_s)[10][10] + (_s)[10][11] * I) *  (_R)[3][3];
  (_fp)[10][10] = creal(_c);
  (_fp)[10][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][12] + (_s)[ 1][13] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][12] + (_s)[ 4][13] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][12] + (_s)[ 7][13] * I) *  (_R)[2][3];
  _c += ( (_s)[10][12] + (_s)[10][13] * I) *  (_R)[3][3];
  (_fp)[10][12] = creal(_c);
  (_fp)[10][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][14] + (_s)[ 1][15] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][14] + (_s)[ 4][15] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][14] + (_s)[ 7][15] * I) *  (_R)[2][3];
  _c += ( (_s)[10][14] + (_s)[10][15] * I) *  (_R)[3][3];
  (_fp)[10][14] = creal(_c);
  (_fp)[10][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][16] + (_s)[ 1][17] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][16] + (_s)[ 4][17] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][16] + (_s)[ 7][17] * I) *  (_R)[2][3];
  _c += ( (_s)[10][16] + (_s)[10][17] * I) *  (_R)[3][3];
  (_fp)[10][16] = creal(_c);
  (_fp)[10][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][18] + (_s)[ 1][19] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][18] + (_s)[ 4][19] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][18] + (_s)[ 7][19] * I) *  (_R)[2][3];
  _c += ( (_s)[10][18] + (_s)[10][19] * I) *  (_R)[3][3];
  (_fp)[10][18] = creal(_c);
  (_fp)[10][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][20] + (_s)[ 1][21] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][20] + (_s)[ 4][21] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][20] + (_s)[ 7][21] * I) *  (_R)[2][3];
  _c += ( (_s)[10][20] + (_s)[10][21] * I) *  (_R)[3][3];
  (_fp)[10][20] = creal(_c);
  (_fp)[10][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][22] + (_s)[ 1][23] * I) *  (_R)[0][3];
  _c += ( (_s)[ 4][22] + (_s)[ 4][23] * I) *  (_R)[1][3];
  _c += ( (_s)[ 7][22] + (_s)[ 7][23] * I) *  (_R)[2][3];
  _c += ( (_s)[10][22] + (_s)[10][23] * I) *  (_R)[3][3];
  (_fp)[10][22] = creal(_c);
  (_fp)[10][23] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][ 0] + (_s)[ 5][ 1] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][ 0] + (_s)[ 8][ 1] * I) *  (_R)[2][3];
  _c += ( (_s)[11][ 0] + (_s)[11][ 1] * I) *  (_R)[3][3];
  (_fp)[11][ 0] = creal(_c);
  (_fp)[11][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][ 2] + (_s)[ 5][ 3] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][ 2] + (_s)[ 8][ 3] * I) *  (_R)[2][3];
  _c += ( (_s)[11][ 2] + (_s)[11][ 3] * I) *  (_R)[3][3];
  (_fp)[11][ 2] = creal(_c);
  (_fp)[11][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][ 4] + (_s)[ 5][ 5] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][ 4] + (_s)[ 8][ 5] * I) *  (_R)[2][3];
  _c += ( (_s)[11][ 4] + (_s)[11][ 5] * I) *  (_R)[3][3];
  (_fp)[11][ 4] = creal(_c);
  (_fp)[11][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][ 6] + (_s)[ 5][ 7] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][ 6] + (_s)[ 8][ 7] * I) *  (_R)[2][3];
  _c += ( (_s)[11][ 6] + (_s)[11][ 7] * I) *  (_R)[3][3];
  (_fp)[11][ 6] = creal(_c);
  (_fp)[11][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 8] + (_s)[ 2][ 9] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][ 8] + (_s)[ 5][ 9] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][ 8] + (_s)[ 8][ 9] * I) *  (_R)[2][3];
  _c += ( (_s)[11][ 8] + (_s)[11][ 9] * I) *  (_R)[3][3];
  (_fp)[11][ 8] = creal(_c);
  (_fp)[11][ 9] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][10] + (_s)[ 2][11] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][10] + (_s)[ 5][11] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][10] + (_s)[ 8][11] * I) *  (_R)[2][3];
  _c += ( (_s)[11][10] + (_s)[11][11] * I) *  (_R)[3][3];
  (_fp)[11][10] = creal(_c);
  (_fp)[11][11] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][12] + (_s)[ 2][13] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][12] + (_s)[ 5][13] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][12] + (_s)[ 8][13] * I) *  (_R)[2][3];
  _c += ( (_s)[11][12] + (_s)[11][13] * I) *  (_R)[3][3];
  (_fp)[11][12] = creal(_c);
  (_fp)[11][13] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][14] + (_s)[ 2][15] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][14] + (_s)[ 5][15] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][14] + (_s)[ 8][15] * I) *  (_R)[2][3];
  _c += ( (_s)[11][14] + (_s)[11][15] * I) *  (_R)[3][3];
  (_fp)[11][14] = creal(_c);
  (_fp)[11][15] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][16] + (_s)[ 2][17] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][16] + (_s)[ 5][17] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][16] + (_s)[ 8][17] * I) *  (_R)[2][3];
  _c += ( (_s)[11][16] + (_s)[11][17] * I) *  (_R)[3][3];
  (_fp)[11][16] = creal(_c);
  (_fp)[11][17] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][18] + (_s)[ 2][19] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][18] + (_s)[ 5][19] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][18] + (_s)[ 8][19] * I) *  (_R)[2][3];
  _c += ( (_s)[11][18] + (_s)[11][19] * I) *  (_R)[3][3];
  (_fp)[11][18] = creal(_c);
  (_fp)[11][19] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][20] + (_s)[ 2][21] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][20] + (_s)[ 5][21] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][20] + (_s)[ 8][21] * I) *  (_R)[2][3];
  _c += ( (_s)[11][20] + (_s)[11][21] * I) *  (_R)[3][3];
  (_fp)[11][20] = creal(_c);
  (_fp)[11][21] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][22] + (_s)[ 2][23] * I) *  (_R)[0][3];
  _c += ( (_s)[ 5][22] + (_s)[ 5][23] * I) *  (_R)[1][3];
  _c += ( (_s)[ 8][22] + (_s)[ 8][23] * I) *  (_R)[2][3];
  _c += ( (_s)[11][22] + (_s)[11][23] * I) *  (_R)[3][3];
  (_fp)[11][22] = creal(_c);
  (_fp)[11][23] = cimag(_c);
  _fp_eq_fp( _r, _fp);
  free_fp( &_fp);
}  /* end of rot_fp_bispinor_mat */


/***********************************************************
 * spin-rotate fermion vector list from the left
 ***********************************************************/
inline void rot_fv_ti_bispinor_mat ( double** _r, double _Complex ** _R, double** _s, unsigned int _ix) {
  fermion_propagator_type _fp;
  create_fp( &_fp );
  _assign_fp_point_from_field( _fp, _s, _ix);
  rot_fp_ti_bispinor_mat ( _fp, _R, _fp);
  _assign_fp_field_from_point( _r, _fp, _ix);
  free_fp( &_fp );
}  /* end of rot_fv_ti_bispinor_mat */


inline void rot_sp_ti_bispinor_mat ( spinor_propagator_type _r, double _Complex ** _R, spinor_propagator_type _s) {
  double _Complex _c;
  spinor_propagator_type _sp;
  create_sp( &_sp );
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[0][0];
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[1][0];
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[2][0];
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[3][0];
  (_sp)[ 0][ 0] = creal(_c);
  (_sp)[ 0][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[0][0];
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[1][0];
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[2][0];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[3][0];
  (_sp)[ 0][ 2] = creal(_c);
  (_sp)[ 0][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[0][0];
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[1][0];
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[2][0];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[3][0];
  (_sp)[ 0][ 4] = creal(_c);
  (_sp)[ 0][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[0][0];
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[1][0];
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[2][0];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[3][0];
  (_sp)[ 0][ 6] = creal(_c);
  (_sp)[ 0][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[0][1];
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[1][1];
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[2][1];
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[3][1];
  (_sp)[ 1][ 0] = creal(_c);
  (_sp)[ 1][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[0][1];
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[1][1];
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[2][1];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[3][1];
  (_sp)[ 1][ 2] = creal(_c);
  (_sp)[ 1][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[0][1];
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[1][1];
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[2][1];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[3][1];
  (_sp)[ 1][ 4] = creal(_c);
  (_sp)[ 1][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[0][1];
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[1][1];
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[2][1];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[3][1];
  (_sp)[ 1][ 6] = creal(_c);
  (_sp)[ 1][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[0][2];
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[1][2];
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[2][2];
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[3][2];
  (_sp)[ 2][ 0] = creal(_c);
  (_sp)[ 2][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[0][2];
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[1][2];
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[2][2];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[3][2];
  (_sp)[ 2][ 2] = creal(_c);
  (_sp)[ 2][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[0][2];
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[1][2];
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[2][2];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[3][2];
  (_sp)[ 2][ 4] = creal(_c);
  (_sp)[ 2][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[0][2];
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[1][2];
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[2][2];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[3][2];
  (_sp)[ 2][ 6] = creal(_c);
  (_sp)[ 2][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[0][3];
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[1][3];
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[2][3];
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[3][3];
  (_sp)[ 3][ 0] = creal(_c);
  (_sp)[ 3][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[0][3];
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[1][3];
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[2][3];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[3][3];
  (_sp)[ 3][ 2] = creal(_c);
  (_sp)[ 3][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[0][3];
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[1][3];
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[2][3];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[3][3];
  (_sp)[ 3][ 4] = creal(_c);
  (_sp)[ 3][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[0][3];
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[1][3];
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[2][3];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[3][3];
  (_sp)[ 3][ 6] = creal(_c);
  (_sp)[ 3][ 7] = cimag(_c);
  _sp_eq_sp( _r, _sp);
  free_sp( &_sp);
}  /* end of rot_sp_ti_bispinor_mat */
inline void rot_bispinor_mat_ti_sp ( spinor_propagator_type _r, double _Complex ** _R, spinor_propagator_type _s) {
  double _Complex _c;
  spinor_propagator_type _sp;
  create_sp( &_sp );
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[0][0];
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[0][1];
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[0][2];
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[0][3];
  (_sp)[ 0][ 0] = creal(_c);
  (_sp)[ 0][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[1][0];
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[1][1];
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[1][2];
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[1][3];
  (_sp)[ 0][ 2] = creal(_c);
  (_sp)[ 0][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[2][0];
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[2][1];
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[2][2];
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[2][3];
  (_sp)[ 0][ 4] = creal(_c);
  (_sp)[ 0][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 0][ 0] + (_s)[ 0][ 1] * I) *  (_R)[3][0];
  _c += ( (_s)[ 0][ 2] + (_s)[ 0][ 3] * I) *  (_R)[3][1];
  _c += ( (_s)[ 0][ 4] + (_s)[ 0][ 5] * I) *  (_R)[3][2];
  _c += ( (_s)[ 0][ 6] + (_s)[ 0][ 7] * I) *  (_R)[3][3];
  (_sp)[ 0][ 6] = creal(_c);
  (_sp)[ 0][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[0][0];
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[0][1];
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[0][2];
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[0][3];
  (_sp)[ 1][ 0] = creal(_c);
  (_sp)[ 1][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[1][0];
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[1][1];
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[1][2];
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[1][3];
  (_sp)[ 1][ 2] = creal(_c);
  (_sp)[ 1][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[2][0];
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[2][1];
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[2][2];
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[2][3];
  (_sp)[ 1][ 4] = creal(_c);
  (_sp)[ 1][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 1][ 0] + (_s)[ 1][ 1] * I) *  (_R)[3][0];
  _c += ( (_s)[ 1][ 2] + (_s)[ 1][ 3] * I) *  (_R)[3][1];
  _c += ( (_s)[ 1][ 4] + (_s)[ 1][ 5] * I) *  (_R)[3][2];
  _c += ( (_s)[ 1][ 6] + (_s)[ 1][ 7] * I) *  (_R)[3][3];
  (_sp)[ 1][ 6] = creal(_c);
  (_sp)[ 1][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[0][0];
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[0][1];
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[0][2];
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[0][3];
  (_sp)[ 2][ 0] = creal(_c);
  (_sp)[ 2][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[1][0];
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[1][1];
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[1][2];
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[1][3];
  (_sp)[ 2][ 2] = creal(_c);
  (_sp)[ 2][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[2][0];
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[2][1];
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[2][2];
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[2][3];
  (_sp)[ 2][ 4] = creal(_c);
  (_sp)[ 2][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 2][ 0] + (_s)[ 2][ 1] * I) *  (_R)[3][0];
  _c += ( (_s)[ 2][ 2] + (_s)[ 2][ 3] * I) *  (_R)[3][1];
  _c += ( (_s)[ 2][ 4] + (_s)[ 2][ 5] * I) *  (_R)[3][2];
  _c += ( (_s)[ 2][ 6] + (_s)[ 2][ 7] * I) *  (_R)[3][3];
  (_sp)[ 2][ 6] = creal(_c);
  (_sp)[ 2][ 7] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[0][0];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[0][1];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[0][2];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[0][3];
  (_sp)[ 3][ 0] = creal(_c);
  (_sp)[ 3][ 1] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[1][0];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[1][1];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[1][2];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[1][3];
  (_sp)[ 3][ 2] = creal(_c);
  (_sp)[ 3][ 3] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[2][0];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[2][1];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[2][2];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[2][3];
  (_sp)[ 3][ 4] = creal(_c);
  (_sp)[ 3][ 5] = cimag(_c);
  _c = 0.;
  _c += ( (_s)[ 3][ 0] + (_s)[ 3][ 1] * I) *  (_R)[3][0];
  _c += ( (_s)[ 3][ 2] + (_s)[ 3][ 3] * I) *  (_R)[3][1];
  _c += ( (_s)[ 3][ 4] + (_s)[ 3][ 5] * I) *  (_R)[3][2];
  _c += ( (_s)[ 3][ 6] + (_s)[ 3][ 7] * I) *  (_R)[3][3];
  (_sp)[ 3][ 6] = creal(_c);
  (_sp)[ 3][ 7] = cimag(_c);
  _sp_eq_sp( _r, _sp);
  free_sp( &_sp);
}  /* end of rot_bispinor_mat_ti_sp */


}  /* end of namespace cvc */

#endif
