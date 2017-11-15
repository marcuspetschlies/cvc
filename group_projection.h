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

/********************************************************/
/********************************************************/

extern little_group_type *little_groups;

int little_group_read_list (little_group_type **lg, char *filename );
 
void little_group_init ( little_group_type **lg, int n );

void little_group_fini ( little_group_type **lg, int n );

void little_group_show ( little_group_type *lg, FILE*ofs, int n);

}  /* end of namespace cvc */

#endif
