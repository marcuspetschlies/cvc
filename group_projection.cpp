/****************************************************
 * group_projection.cpp
 *
 * Fr 10. Nov 16:15:09 CET 2017
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif

#include "cvc_complex.h"
#include "iblas.h"
#include "ilinalg.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "mpi_init.h"
#include "matrix_init.h"
#include "group_projection.h"
#include "cvc_utils.h"

namespace cvc {

little_group_type *little_groups = NULL;

/***********************************************************/
/***********************************************************/

int little_group_read_list (little_group_type **lg, char *filename )  {
  
  char comma[] = ",";

  char line[200];
  int nline = 0;
  char r_str[100], rm_str[100];

  FILE *ofs = fopen(filename, "r");
  if ( ofs == NULL ) {
    fprintf(stderr, "[little_group_read_table] Error from fopen trying to open file %s\n", filename);
    return(1);
  }
            

  while ( fgets ( line, 100, ofs) != NULL) {
    nline++;
  }
              
  if ( *lg != NULL ) free ( *lg );
  *lg = (little_group_type* )malloc( nline * sizeof (little_group_type) );
  if ( *lg == NULL ) {
    fprintf(stderr, "[little_group_read_table] Error from malloc\n");
    return(2);
  }
                    
  rewind( ofs );
  for ( int i = 0; i < nline; i++ ) {
    fscanf(ofs, "%s %d %d %d %s %d %s %s",  
        (*lg)[i].parent_name,
        (*lg)[i].d, (*lg)[i].d+1, (*lg)[i].d+2,
        (*lg)[i].name,
        &((*lg)[i].n),
        r_str, rm_str );

    if ( ( (*lg)[i].r = (int*)malloc( (*lg)[i].n * sizeof(int) ) ) == NULL ) {
      fprintf(stderr, "[little_group_read_table] Error from malloc\n");
      return(3);
    }

    /* extract the list of rotations Rd = d */
    char *ptr = strtok( r_str, comma );
    (*lg)[i].r[0] = atoi(ptr);
    for ( int k = 1; k <= (*lg)[i].n; k++ ) {
      ptr = strtok( NULL, comma );
      if ( ptr == NULL ) {
        fprintf(stderr, "[little_group_read_table] Error from strtok\n");
        return(4);
      }
      (*lg)[i].r[k] = atoi(ptr);
    }
      
    /* extract the list of rotations Rd = -d */
    ptr = strtok( rm_str, comma );
    (*lg)[i].rm[0] = atoi(ptr);
    for ( int k = 1; k <= (*lg)[i].n; k++ ) {
      ptr = strtok( NULL, comma );
      if ( ptr == NULL ) {
        fprintf(stderr, "[little_group_read_table] Error from strtok\n");
        return(4);
      }
      (*lg)[i].rm[k] = atoi(ptr);
    }
  }
  return(0);

}  /* end of little_group_read_list */

/***********************************************************/
/***********************************************************/

void little_group_init ( little_group_type **lg, int n ) {
  if( n == 0 ) return;
  if ( *lg != NULL ) little_group_fini ( lg );

  *lg = (little_group_type* )malloc( n * sizeof (little_group_type) );
  if ( *lg == NULL ) {
    fprintf(stderr, "[little_group_init] Error from malloc\n");
    return(2);
  }

  for ( int i = 0; i < n; i++ ) {
    (*lg)[i].r  = NULL;
    (*lg)[i].rm = NULL;
  }
  return;
}

/***********************************************************/
/***********************************************************/

void little_group_fini ( little_group_type **lg, int n ) {
  if ( *lg == NULL || n == 0 ) return;

  for ( int i = 0; i < n; i++ ) {
    if ( (*lg)[i].r  != NULL ) free ( (*lg)[i].r  );
    if ( (*lg)[i].rm != NULL ) free ( (*lg)[i].rm );
  }

  free ( *lg ); *lg = NULL;
  return;
}

void little_group_show ( little_group_type *lg, FILE*ofs, int n) {
}

/***********************************************************/
/***********************************************************/

}  /* end of namespace cvc */
