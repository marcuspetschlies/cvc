#ifndef _GROUP_PROJECTION_H
#define _GROUP_PROJECTION_H

#include "ilinalg.h"

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
  char correlator_name[200];
  int *ref_row_spin;
  int ref_row_target;
  int row_target;
} little_group_projector_type;

/********************************************************/
/********************************************************/

extern little_group_type *little_groups;

int little_group_read_list (little_group_type **lg, char *filename );
 
void little_group_init ( little_group_type **lg, int n );

void little_group_fini ( little_group_type **lg, int n );

void little_group_show ( little_group_type *lg, FILE*ofs, int n);

void init_rot_mat_table (rot_mat_table_type *t );

void fini_rot_mat_table (rot_mat_table_type *t );

int alloc_rot_mat_table ( rot_mat_table_type *t, char*group, char*irrep, int dim, int n );

int set_rot_mat_table_spin ( rot_mat_table_type *t, int J2, int bispinor );

int set_rot_mat_table_cubic_group_double_cover ( rot_mat_table_type *t, char *group, char*irrep );

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

int little_group_projector_set (
  little_group_projector_type *p,
  little_group_type *lg,
  char*irrep , int row_target, int interpolator_num,
  int *interpolator_J2_list, int **interpolator_momentum_list, int *interpolator_bispinor_list,    
  int ref_row_target, int *ref_row_spin, char*correlator_name );

int little_group_projector_apply ( little_group_projector_type *p , FILE*ofs );

int spin_vector_asym_printf ( double _Complex **sv, int n, int*dim, char*name, FILE*ofs );

double spin_vector_asym_norm2 ( double _Complex **sv, int n, int *dim );

double spin_vector_asym_list_norm2 ( double _Complex ***sv, int nc, int n, int *dim );

int spin_vector_asym_list_normalize ( double _Complex ***sv, int nc, int n, int *dim );

int spin_vector_asym_normalize ( double _Complex **sv, int n, int *dim );


}  /* end of namespace cvc */

#endif
