#ifndef _GROUP_PROJECTION_H
#define _GROUP_PROJECTION_H

#include "ilinalg.h"
#include "group_projection_applicator.h"


namespace cvc {

/********************************************************/
/********************************************************/

typedef struct {
  int d[3];
  int nr;
  int nrm;
  char parent_name[10];
  char name[10];
  int *r;
  int *rm;
  int nirrep;
  char **lirrep;
} little_group_type; 

/********************************************************/
/********************************************************/

typedef struct {

  int n;
  int dim;
  char group[100];
  char irrep[100];
  int *rid;
  int *rmid;
  double _Complex ***R;
  double _Complex ***IR;

} rot_mat_table_type;

/********************************************************/
/********************************************************/

typedef struct {
  int n;
  rot_mat_table_type *rspin;
  rot_mat_table_type *rtarget;
  rot_mat_table_type *rp;
  int P[3];
  int**p;
  /* double _Complex **c; */
  char name[200];
  int *ref_row_spin;
  int ref_row_target;
  int row_target;
  int *parity;
  int refframerot;
} little_group_projector_type;

/********************************************************/
/********************************************************/
 
extern little_group_type *little_groups;

int little_group_read (little_group_type *lg, const char *lg_name, const char * filename );

int little_group_read_list (little_group_type **lg, const char * filename );
 
int little_group_init ( little_group_type *lg );

void little_group_fini ( little_group_type *lg );

void little_group_show ( little_group_type *lg, FILE*ofs, int n);

void init_rot_mat_table (rot_mat_table_type *t );

void fini_rot_mat_table (rot_mat_table_type *t );

int alloc_rot_mat_table ( rot_mat_table_type *t, const char*group, const char*irrep, const int dim, const int n );


int set_rot_mat_table_spin ( rot_mat_table_type *t, int J2, int bispinor );

int set_rot_mat_table_spin_single_cover ( rot_mat_table_type *t, int J2, int const version , int const setby );


int set_rot_mat_table_cubic_group_double_cover ( rot_mat_table_type *t, const char * group, const char * irrep );

int set_rot_mat_table_cubic_group_single_cover ( rot_mat_table_type *t, const char * group, const char * irrep );



void rot_mat_table_printf ( rot_mat_table_type *t, char*name, FILE*ofs );

int rot_mat_table_copy (rot_mat_table_type *t, rot_mat_table_type *s );

int rot_mat_mult_table ( int***mtab, rot_mat_table_type *t );

int rot_mat_table_is_lg ( rot_mat_table_type *t, int d[3] );

int rot_mat_table_get_lg ( rot_mat_table_type *t, int d[3] );

int rot_mat_table_get_d2d ( rot_mat_table_type *t, int d1[3], int d2[3] );

int rot_mat_table_character ( double _Complex ***rc, rot_mat_table_type *t );

int rot_mat_table_orthogonality ( rot_mat_table_type *t1, rot_mat_table_type *t2 );

int init_little_group_projector (little_group_projector_type *p );
int fini_little_group_projector (little_group_projector_type *p );

int little_group_projector_show (little_group_projector_type *p, FILE*ofs, int with_mat );

int little_group_projector_copy (little_group_projector_type *p, little_group_projector_type *q );


little_group_projector_applicator_type ** little_group_projector_apply ( little_group_projector_type *p , FILE*ofs);

int spin_vector_asym_printf ( double _Complex **sv, int n, int*dim, char*name, FILE*ofs );

double spin_vector_asym_norm2 ( double _Complex **sv, int n, int *dim );

double spin_vector_asym_list_norm2 ( double _Complex ***sv, int nc, int n, int *dim );

int spin_vector_asym_list_normalize ( double _Complex ***sv, int nc, int n, int *dim );

int spin_vector_asym_normalize ( double _Complex **sv, int n, int *dim );

void spin_vector_pl_eq_spin_vector_ti_co_asym ( double _Complex **sv1, double _Complex **sv2, double _Complex c, int n, int *dim );

int rot_mat_table_rotate_multiplett ( rot_mat_table_type * const rtab, rot_mat_table_type * const rapply, rot_mat_table_type * const rtarget, int const parity, FILE*ofs );

int irrep_multiplicity (rot_mat_table_type * const rirrep, rot_mat_table_type * const rspin, int const parity );

int little_group_projector_apply_product ( little_group_projector_type *p , FILE*ofs);

void product_vector_project_accum ( double _Complex *v, rot_mat_table_type*r, int rid, int rmid, double _Complex *v0, double _Complex c1, double _Complex c2, int *dim , int n );

void product_mat_pl_eq_mat_ti_co ( double _Complex **R, rot_mat_table_type *r, int rid, int rmid, double _Complex c, int*dim, int n );

int product_mat_printf ( double _Complex **R, int *dim, int n, char *name, FILE*ofs );

void product_vector_printf ( double _Complex *v, int*dim, int n, char*name, FILE*ofs );

void rot_mat_eq_product_mat_ti_rot_mat ( double _Complex **R, rot_mat_table_type *r, int rid, int rmid, double _Complex **S, int n );

int rot_mat_table_rotate_multiplett_product ( rot_mat_table_type *rtab, rot_mat_table_type *rapply, rot_mat_table_type *rtarget, int n, int with_IR, FILE*ofs );

void rot_mat_table_eq_product_mat_table ( rot_mat_table_type *r, rot_mat_table_type *s, int const n );

int rot_mat_table_get_spin2 ( rot_mat_table_type *t );

int rot_mat_table_get_bispinor ( rot_mat_table_type *t );

int get_reference_rotation ( int pref[3], int *Rref, int const p[3] );

inline int comp_int3_abs ( int const a[3] , int const b[3] );

/***********************************************************/
/***********************************************************/

static inline void product_vector_index2coords ( int idx, int *coords, int *dim, int n ) {
  int ll = 1;
  for ( int i = n-1; i >= 0; i-- ) {
    coords[i] = (idx % (ll*dim[i])) / ll;
    idx      -= coords[i] * ll;
    ll       *= dim[i];
  }
}  /* end of product_vector_index2coords */

/***********************************************************/
/***********************************************************/

static inline int product_vector_coords2index ( int *coords, int *dim, int n ) {
  int idx = coords[0];
  for ( int i = 1; i < n; i++ ) {
    idx = dim[i] * idx + coords[i];
  }
  return(idx);
}  /* end of product_vector_coords2index */


/***********************************************************/
/***********************************************************/

static inline void product_vector_set_element ( double _Complex*v, double _Complex c, int *coords, int *dim, int n ) {
  v[ product_vector_coords2index ( coords, dim, n ) ] = c;
}  /* end of product_vector_set_element */


}  /* end of namespace cvc */

#endif
