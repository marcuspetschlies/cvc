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
} little_group_type; 

typedef struct {

  int n;
  char group[20]
  char irrep[20];
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

int set_rot_mat_table_spin ( rot_mat_table_type *t, int J2, int bispinor );

}  /* end of namespace cvc */

#endif
