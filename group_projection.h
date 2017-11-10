#ifndef _GROUP_PROJECTION_H
#define _GROUP_PROJECTION_H

#include "ilinalg.h"

namespace cvc {

/********************************************************/
/********************************************************/

typedef struct {
  int d[3];
  int n;
  char parent_name[10]
  char name[10]
  int *r;
  int *rm;
} little_group_type; 

/********************************************************/
/********************************************************/

extern little_group_type *little_groups;

int little_group_read_table (little_group_type **lg, char *filename );
 
}  /* end of namespace cvc */

#endif
