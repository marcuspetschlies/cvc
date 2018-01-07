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

int rot_mat_mult_table ( int***mtab, rot_mat_table_type *t );

int rot_mat_table_is_lg ( rot_mat_table_type *t, int d[3] );

int rot_mat_table_get_lg ( rot_mat_table_type *t, int d[3] );

int rot_mat_table_get_d2d ( rot_mat_table_type *t, int d1[3], int d2[3] );

int rot_mat_table_character ( double _Complex ***rc, rot_mat_table_type *t );

int rot_mat_table_orthogonality ( rot_mat_table_type *t1, rot_mat_table_type *t2 );

}  /* end of namespace cvc */

#endif
