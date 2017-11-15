/****************************************************
 * group_projection.cpp
 *
 * Fr 10. Nov 16:15:09 CET 2017
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * - mapping 
 *   little group for total momentum d LG(d) x irrep Gamma x rotation R -> T_Gamma(R)
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
  char r_str[100], rm_str[100], r_aux[100];

  FILE *ofs = fopen(filename, "r");
  if ( ofs == NULL ) {
    fprintf(stderr, "[little_group_read_list] Error from fopen trying to open file %s\n", filename);
    return(-1);
  }
   
  /***********************************************************/
  /***********************************************************/

  while ( fgets ( line, 100, ofs) != NULL) {
    nline++;
  }
  fprintf(stdout, "# [little_group_read_list] number of entries = %d\n", nline);

              
  /***********************************************************
   * allocate lg list
   ***********************************************************/
  if ( *lg != NULL ) free ( *lg );
  *lg = (little_group_type* )malloc( nline * sizeof (little_group_type) );
  if ( *lg == NULL ) {
    fprintf(stderr, "[little_group_read_table] Error from malloc\n");
    return(-2);
  }

  /***********************************************************
   * read line-wise from file
   ***********************************************************/
  rewind( ofs );
  for ( int i = 0; i < nline; i++ ) {
    fscanf(ofs, "%s %d %d %d %s %s %s",  
        (*lg)[i].parent_name,
        (*lg)[i].d, (*lg)[i].d+1, (*lg)[i].d+2,
        (*lg)[i].name,
        r_str, rm_str );

    /* TEST */
    if ( g_verbose > 3 ) {
      fprintf(stdout, "# [little_group_read_list] %s --- %d --- %d --- %d --- %s --- %s --- %s\n", (*lg)[i].parent_name, 
          (*lg)[i].d[0], (*lg)[i].d[1], (*lg)[i].d[2], (*lg)[i].name, r_str, rm_str);
    }


    /***********************************************************
     * extract the list of rotations Rd = d
     *
     * subtract 1 from list entry
     ***********************************************************/
    strcpy ( r_aux, r_str );
    char *ptr = strtok( r_aux, comma );
    (*lg)[i].nr = 1;
    while ( (ptr = strtok( NULL, comma ) ) != NULL ) (*lg)[i].nr++;
    fprintf(stdout, "# [little_group_read_list] %d nr %d\n", i, (*lg)[i].nr);

    if ( ( (*lg)[i].r = (int*)malloc( (*lg)[i].nr * sizeof(int) ) ) == NULL ) {
      fprintf(stderr, "[little_group_read_list] Error from malloc\n");
      return(-3);
    }

    ptr = strtok( r_str, comma );
    /* fprintf(stdout, "# [little_group_read_list] 0 ptr = %s\n", ptr);  */

    (*lg)[i].r[0] = atoi(ptr) - 1;
    for ( int k = 1; k < (*lg)[i].nr; k++ ) {
      ptr = strtok( NULL, comma );
      /* fprintf(stdout, "# [little_group_read_list] %d ptr = %s\n", k, ptr); */
      if ( ptr == NULL ) {
        fprintf(stderr, "[little_group_read_list] Error from strtok\n");
        return(-5);
      }
      (*lg)[i].r[k] = atoi(ptr) - 1;
    }


    /***********************************************************
     * extract the list of rotations Rd = -d
     *
     * subtract 1 from list entry
     ***********************************************************/
    strcpy ( r_aux, rm_str );
    ptr = strtok( r_aux, comma );
    (*lg)[i].nrm = 1;
    while ( ( ptr = strtok( NULL, comma ) ) != NULL ) (*lg)[i].nrm++;
    fprintf(stdout, "# [little_group_read_list] %d nrm %d\n", i, (*lg)[i].nrm);

    if ( ( (*lg)[i].rm = (int*)malloc( (*lg)[i].nrm * sizeof(int) ) ) == NULL ) {
      fprintf(stderr, "[little_group_read_list] Error from malloc\n");
      return(-4);
    }
    ptr = strtok( rm_str, comma );
    (*lg)[i].rm[0] = atoi(ptr) - 1;
    for ( int k = 1; k < (*lg)[i].nrm; k++ ) {
      ptr = strtok( NULL, comma );
      if ( ptr == NULL ) {
        fprintf(stderr, "[little_group_read_list] Error from strtok\n");
        return(-6);
      }
      (*lg)[i].rm[k] = atoi(ptr) - 1;
    }
  }
  return(nline);

}  /* end of little_group_read_list */

/***********************************************************/
/***********************************************************/

void little_group_init ( little_group_type **lg, int n ) {
  if( n == 0 ) return;
  if ( *lg != NULL ) {
    fprintf(stderr, "[little_group_init] Error, lg is not NULL\n");
    *lg = NULL;
    return;
  }

  *lg = (little_group_type* )malloc( n * sizeof (little_group_type) );
  if ( *lg == NULL ) {
    fprintf(stderr, "[little_group_init] Error from malloc\n");
    return;
  }

  for ( int i = 0; i < n; i++ ) {
    (*lg)[i].r  = NULL;
    (*lg)[i].rm = NULL;
  }
  return;
}  /* end of little_group_init */

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
}  /*  little_group_fini */

/***********************************************************/
/***********************************************************/

void little_group_show ( little_group_type *lg, FILE*ofs, int n) {

  for ( int i = 0; i < n; i++ ) {
    fprintf(stdout, "# [little_group_show]\n# [little_group_show]\n# [little_group_show]\n# [little_group_show] %4d %s\n", i, lg[i].name );
    fprintf(stdout, "# [little_group_show] parent name %s\n", lg[i].parent_name );
    fprintf(stdout, "# [little_group_show] d-vector (%3d, %3d, %3d)\n", lg[i].d[0], lg[i].d[1], lg[i].d[2] );
    /* fprintf(stdout, "# [little_group_show] %4d %s\n", i, lg[i].name ); */
    fprintf(stdout, "# [little_group_show] number of elements Rd = +d %2d\n", lg[i].nr );
    for ( int k = 0; k < lg[i].nr; k++ ) {
      fprintf(stdout, "# [little_group_show]   r[%2d]  = %3d\n", k, lg[i].r[k] );
    }
    fprintf(stdout, "# [little_group_show] number of elements Rd = -d %2d\n", lg[i].nrm );
    for ( int k = 0; k < lg[i].nrm; k++ ) {
      fprintf(stdout, "# [little_group_show]   rm[%2d] = %3d\n", k, lg[i].rm[k] );
    }
  }
  return;
}  /* end of little_group_fini */

/***********************************************************/
/***********************************************************/

}  /* end of namespace cvc */
