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

void init_rot_mat_table (rot_mat_table_type *t ) {
  strcpy( t->group, "NA" );
  strcpy( t->irrep, "NA" );
  t->n   = 0;
  t->rid = NULL;
  t->rmid = NULL;
  t->R   = NULL;
  t->IR  = NULL;
  return;
}  /* end of init_rot_mat_table */

/***********************************************************/
/***********************************************************/

void fini_rot_mat_table (rot_mat_table_type *t ) {
  strcpy( t->group, "NA" );
  strcpy( t->irrep, "NA" );
  t->n   = 0;
  if ( t->rid  != NULL ) { free ( t->rid );  t->rid  = NULL; }
  if ( t->rmid != NULL ) { free ( t->rmid ); t->rmid = NULL; }
  if ( t->R    != NULL ) { fini_3level_buffer( &(t->R) ); }
  if ( t->IR   != NULL ) { fini_3level_buffer( &(t->IR) ); }
  return;
}  /* end of fini_rot_mat_table */

/***********************************************************/
/***********************************************************/

int alloc_rot_mat_table ( rot_mat_table_type *t, char*group, char*irrep, int dim, int n ) {
  int exitstatus;
  
  if ( t->n != 0 ) {
    fprintf(stdout, "# [alloc_rot_mat_table] deleting existing rotation matrix table\n");
    fini_rot_mat_table ( t );
  }
  strcpy ( t->group, group );
  strcpy ( t->irrep, irrep )
  t->n = n;
  if ( ( t->rid = (int*)malloc( n * sizeof(int) ) ) == NULL ) {
    fprintf(stderr, "[alloc_rot_mat_table] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  if ( ( t->rmid = (int*)malloc( n * sizeof(int) ) ) == NULL ) {
    fprintf(stderr, "[alloc_rot_mat_table] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  if ( ( exitstatus = init_3level_buffer ( &(t->R), n, dim, dim ) ) != 0 ) {
    fprintf(stderr, "[alloc_rot_mat_table] Error from init_3level_buffer, exitstatus was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }
  if ( ( exitstatus = init_3level_buffer ( &(t->IR), n, dim, dim ) ) != 0 ) {
    fprintf(stderr, "[alloc_rot_mat_table] Error from init_3level_buffer, exitstatus was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

  return(0);

}  /* end of alloc_rot_mat_table */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * J2 = 2 x spin
 ***********************************************************/
int set_rot_mat_table_spin ( rot_mat_table_type *t, int J2, int bispinor ) {

  int exitstatus;
  char name[20];
  rotation_matrix_type P;

  if ( bispinor && J2 != 1 ) {
    fprintf(stderr, "[set_rot_mat_table_spin] bispinor works only for spin J = 1/2\n");
    return(1);
  }

  if ( bspinor ) {
    sprintf(name, "spin1_2+1_2");
  } else {
    if ( J2 % 2 == 0 ) {
      sprintf(name, "spin%d", J2/2);
    } else {
      sprintf(name, "spin%d_2", J2);
    }
  }
  fprintf(stdout, "# [set_rot_mat_table_spin] name = %s\n", name);

  if ( ( exitstatus = alloc_rot_mat_table ( t, "SU2", name, (1+bispinor)*(J2+1), 48) ) != 0 ) {
    fprintf(stderr, "[set_rot_mat_table_spin] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
  init_2level_buffer ( &P, (1+bispinor)*(J2+1), (1+bispinor)*(J2+1) );

  /* set parity matrix if applicable */
  rot_inversion_matrix_spherical_basis ( P, J2, bispinor );


  if  ( !bispinor ) {
    for ( int i = 0; i < 48; i++ ) {
      t->rid[i] = i;
      rot_rotation_matrix_spherical_basis ( t->R[i], J2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      rot_mat_ti_mat ( t->IR[i], P, t->R[i], J2+1);
    }
  } else {

    for ( int i = 0; i < 48; i++ ) {
      t->rid[i] = i;
      rot_bispinor_rotation_matrix_spherical_basis ( t->r[i], cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      rot_mat_ti_mat ( t->IR[i], P, t->R[i], J2+1);
    }
  }

  fini_2level_buffer ( &P );
  return(0);
}  /* end of set_rot_mat_table_spin */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * rotation matrices for double cover
 ***********************************************************/
int set_rot_mat_table_cubic_group_double_cover ( rot_mat_table_type *t, char *group, char*irrep ) {

  if ( strcmp ( group, "2Oh" ) == 0 ) {
    
    /* LG 2Oh irrep A1 */
    if        ( strcmp ( irrep, "A1" ) == 0 ) {
      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, 48) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < 48; i++ ) {
        t->rid[i] = i;
        t->rmid[i] = i;
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

    /* LG 2Oh irrep A2 */
    } else if ( strcmp ( irrep, "A2" ) == 0 ) {
      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, 48) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < 48; i++ ) { 
        t->rid[i]      = i; 
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

      /* 6C8', 6C8 */
      for ( int i = 7; i <= 18; i++ ) {
        t->R[i][0][0]  = -1.;
        t->IR[i][0][0] = -1.;
      }
     
      /* 12C4' */
      for ( int i = 35; i <= 46; i++ ) {
        t->R[i][0][0]  = -1.;
        t->IR[i][0][0] = -1.;
      }

    /* LG 2Oh irrep E */
    } else if ( strcmp ( irrep, "E" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, 48) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < 48; i++ ) { t->rid[i] = i; }

      /* R1, R48 */
      t->R[ 0][0][0] = 1.; t->R[ 0][1][1] = 1.;
      t->R[47][0][0] = 1.; t->R[47][1][1] = 1.;
      /* R2, R5 */
      t->R[ 1][0][0] = 1.; t->R[ 1][1][1] = 1.;
      t->R[ 4][0][0] = 1.; t->R[ 4][1][1] = 1.;
      /* R3, R6 */
      t->R[ 2][0][0] = 1.; t->R[ 2][1][1] = 1.;
      t->R[ 5][0][0] = 1.; t->R[ 5][1][1] = 1.;
      /* R4, R7 */
      t->R[ 3][0][0] = 1.; t->R[ 3][1][1] = 1.;
      t->R[ 6][0][0] = 1.; t->R[ 6][1][1] = 1.;

      /* R13, R16 sigma_3 */
      t->R[12][0][0] = 1.; t->R[12][1][1] = -1.;
      t->R[15][0][0] = 1.; t->R[15][1][1] = -1.;
      /* R10, R19 sigma_3 */
      t->R[ 9][0][0] = 1.; t->R[ 9][1][1] = -1.;
      t->R[18][0][0] = 1.; t->R[18][1][1] = -1.;
      /* R38, R44 sigma_3 */
      t->R[37][0][0] = 1.; t->R[37][1][1] = -1.;
      t->R[43][0][0] = 1.; t->R[43][1][1] = -1.;
      /* R39, R45 sigma_3 */
      t->R[38][0][0] = 1.; t->R[38][1][1] = -1.;
      t->R[44][0][0] = 1.; t->R[44][1][1] = -1.;
      
      /* -1/2 1 + i sqrt(3)/2 sigma_2 */

      /* R24, R28 */
      t->R[23][0][0] = -ONE_HALF;   t->R[23][1][1] = -ONE_HALF;
      t->R[23][0][1] =  SQRT3_HALF; t->R[23][1][0] = -SQRT3_HALF;
      t->R[27][0][0] = -ONE_HALF;   t->R[27][1][1] = -ONE_HALF;
      t->R[27][0][1] =  SQRT3_HALF; t->R[27][1][0] = -SQRT3_HALF;
      /* R21, R33 */
      t->R[20][0][0] = -ONE_HALF;   t->R[20][1][1] = -ONE_HALF;
      t->R[20][0][1] =  SQRT3_HALF; t->R[20][1][0] = -SQRT3_HALF;
      t->R[32][0][0] = -ONE_HALF;   t->R[32][1][1] = -ONE_HALF;
      t->R[32][0][1] =  SQRT3_HALF; t->R[32][1][0] = -SQRT3_HALF;
      /* R26, R30 */
      t->R[25][0][0] = -ONE_HALF;   t->R[25][1][1] = -ONE_HALF;
      t->R[25][0][1] =  SQRT3_HALF; t->R[25][1][0] = -SQRT3_HALF;
      t->R[29][0][0] = -ONE_HALF;   t->R[29][1][1] = -ONE_HALF;
      t->R[29][0][1] =  SQRT3_HALF; t->R[29][1][0] = -SQRT3_HALF;
      /* R23, R35 */
      t->R[22][0][0] = -ONE_HALF;   t->R[22][1][1] = -ONE_HALF;
      t->R[22][0][1] =  SQRT3_HALF; t->R[22][1][0] = -SQRT3_HALF;
      t->R[34][0][0] = -ONE_HALF;   t->R[34][1][1] = -ONE_HALF;
      t->R[34][0][1] =  SQRT3_HALF; t->R[34][1][0] = -SQRT3_HALF;
   
      /* -1/2 1 - i sqrt(3)/2 sigma_2 */

      /* R20, R32 */
      t->R[19][0][0] = -ONE_HALF;   t->R[19][1][1] = -ONE_HALF;
      t->R[19][0][1] = -SQRT3_HALF; t->R[19][1][0] =  SQRT3_HALF;
      t->R[31][0][0] = -ONE_HALF;   t->R[31][1][1] = -ONE_HALF;
      t->R[31][0][1] = -SQRT3_HALF; t->R[31][1][0] =  SQRT3_HALF;
      /* R25, R29 */
      t->R[24][0][0] = -ONE_HALF;   t->R[24][1][1] = -ONE_HALF;
      t->R[24][0][1] = -SQRT3_HALF; t->R[24][1][0] =  SQRT3_HALF;
      t->R[28][0][0] = -ONE_HALF;   t->R[28][1][1] = -ONE_HALF;
      t->R[28][0][1] = -SQRT3_HALF; t->R[28][1][0] =  SQRT3_HALF;
      /* R22, R34 */
      t->R[22][0][0] = -ONE_HALF;   t->R[22][1][1] = -ONE_HALF;
      t->R[22][0][1] = -SQRT3_HALF; t->R[22][1][0] =  SQRT3_HALF;
      t->R[33][0][0] = -ONE_HALF;   t->R[33][1][1] = -ONE_HALF;
      t->R[33][0][1] = -SQRT3_HALF; t->R[33][1][0] =  SQRT3_HALF;
      /* R27, R31 */
      t->R[26][0][0] = -ONE_HALF;   t->R[26][1][1] = -ONE_HALF;
      t->R[26][0][1] = -SQRT3_HALF; t->R[26][1][0] =  SQRT3_HALF;
      t->R[30][0][0] = -ONE_HALF;   t->R[30][1][1] = -ONE_HALF;
      t->R[30][0][1] = -SQRT3_HALF; t->R[30][1][0] =  SQRT3_HALF;

      /* -cos(pi/3) sigma_3 - sin(pi/3) sigma_1 */

      /* R11, R14 */
      t->R[10][0][0] = -ONE_HALF;   t->R[10][1][1] =  ONE_HALF;
      t->R[10][0][1] = -SQRT3_HALF; t->R[10][1][0] = -SQRT3_HALF;
      t->R[13][0][0] = -ONE_HALF;   t->R[13][1][1] =  ONE_HALF;
      t->R[13][0][1] = -SQRT3_HALF; t->R[13][1][0] = -SQRT3_HALF;
      /* R8, R17 */
      t->R[ 7][0][0] = -ONE_HALF;   t->R[ 7][1][1] =  ONE_HALF;
      t->R[ 7][0][1] = -SQRT3_HALF; t->R[ 7][1][0] = -SQRT3_HALF;
      t->R[16][0][0] = -ONE_HALF;   t->R[16][1][1] =  ONE_HALF;
      t->R[16][0][1] = -SQRT3_HALF; t->R[16][1][0] = -SQRT3_HALF;
      /* R36, R42 */
      t->R[35][0][0] = -ONE_HALF;   t->R[35][1][1] =  ONE_HALF;
      t->R[35][0][1] = -SQRT3_HALF; t->R[35][1][0] = -SQRT3_HALF;
      t->R[41][0][0] = -ONE_HALF;   t->R[41][1][1] =  ONE_HALF;
      t->R[41][0][1] = -SQRT3_HALF; t->R[41][1][0] = -SQRT3_HALF;
      /* R37, R43 */
      t->R[36][0][0] = -ONE_HALF;   t->R[36][1][1] =  ONE_HALF;
      t->R[36][0][1] = -SQRT3_HALF; t->R[36][1][0] = -SQRT3_HALF;
      t->R[42][0][0] = -ONE_HALF;   t->R[42][1][1] =  ONE_HALF;
      t->R[42][0][1] = -SQRT3_HALF; t->R[42][1][0] = -SQRT3_HALF;

      /* -cos(pi/3) sigma_3 + sin(pi/3) sigma_1 */

      /* R12, R15 */
      t->R[11][0][0] = -ONE_HALF;   t->R[11][1][1] =  ONE_HALF;
      t->R[11][0][1] =  SQRT3_HALF; t->R[11][1][0] =  SQRT3_HALF;
      t->R[14][0][0] = -ONE_HALF;   t->R[14][1][1] =  ONE_HALF;
      t->R[14][0][1] =  SQRT3_HALF; t->R[14][1][0] =  SQRT3_HALF;
      /* R9, R18 */
      t->R[ 8][0][0] = -ONE_HALF;   t->R[ 8][1][1] =  ONE_HALF;
      t->R[ 8][0][1] =  SQRT3_HALF; t->R[ 8][1][0] =  SQRT3_HALF;
      t->R[17][0][0] = -ONE_HALF;   t->R[17][1][1] =  ONE_HALF;
      t->R[17][0][1] =  SQRT3_HALF; t->R[17][1][0] =  SQRT3_HALF;
      /* R40, R46 */
      t->R[39][0][0] = -ONE_HALF;   t->R[39][1][1] =  ONE_HALF;
      t->R[39][0][1] =  SQRT3_HALF; t->R[39][1][0] =  SQRT3_HALF;
      t->R[45][0][0] = -ONE_HALF;   t->R[45][1][1] =  ONE_HALF;
      t->R[45][0][1] =  SQRT3_HALF; t->R[45][1][0] =  SQRT3_HALF;
      /* R41, R47 */
      t->R[40][0][0] = -ONE_HALF;   t->R[40][1][1] =  ONE_HALF;
      t->R[40][0][1] =  SQRT3_HALF; t->R[40][1][0] =  SQRT3_HALF;
      t->R[46][0][0] = -ONE_HALF;   t->R[46][1][1] =  ONE_HALF;
      t->R[46][0][1] =  SQRT3_HALF; t->R[46][1][0] =  SQRT3_HALF;

      memcpy ( t->rmid, t->rmid, 48*sizeof(int) );
      memcpy ( t->IR[0][0], t->R[0][0], 48*4*sizeof(double _Complex ) );

    /* LG 2Oh irrp T1 */
    } else if ( strcmp ( irrep, "T1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 3, 48) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < 48; i++ ) {
        t->rid[i] = i;
        rot_rotation_matrix_spherical_basis ( t->R[i], 2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      }
      
      memcpy ( t->rmid, t->rmid, 48*sizeof(int) );
      memcpy ( t->IR[0][0], t->R[0][0], 48*9*sizeof(double _Complex ) );

    /* LG 2Oh irrep T2 */
    } else if ( strcmp ( irrep, "T2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 3, 48) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < 48; i++ ) {
        t->rid[i] = i;
        rot_rotation_matrix_spherical_basis ( t->R[i], 2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      }
  
      /* 6C8, 6C8' additional minus sign, R8 to R19 */
      for ( int i =  7; i <= 18; i++ ) { rot_mat_ti_eq_re ( t->R[i], -1., 3 ); }

      /* 12C4' additional minus sign, R36 to R47 */
      for ( int i = 35; i <= 46; i++ ) { rot_mat_ti_eq_re ( t->R[i], -1., 3 ); }
          
      memcpy ( t->rmid, t->rmid, 48*sizeof( int ) );
      memcpy ( t->IR[0][0], t->R[0][0], 48*9*sizeof(double _Complex ) );

    /* LG 2Oh irrep G1 */
    } else if ( strcmp ( irrep, "G1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, 48) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < 48; i++ ) {
        t->rid[i] = i;
        rot_rotation_matrix_spherical_basis ( t->R[i], 2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      }

      memcpy ( t->rmid, t->rmid, 48*sizeof(int) );
      memcpy ( t->IR[0][0], t->R[0][0], 48*4*sizeof(double _Complex ) );

    /* LG 2Oh irrep G2 */
    } else if ( strcmp ( irrep, "G2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, 48) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < 48; i++ ) {
        t->rid[i] = i;
        rot_rotation_matrix_spherical_basis ( t->R[i], 2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      }

      /* 6C8, 6C8' additional sign */
      for ( int i =  7; i <= 18; i++ ) { rot_mat_ti_eq_re ( t->R[i], -1., 2 ); }

      /* 12C4' additional sign */
      for ( int i = 35; i <= 46; i++ ) { rot_mat_ti_eq_re ( t->R[i], -1., 2 ); }


      memcpy ( t->rmid, t->rmid, 48*sizeof(int) );
      memcpy ( t->IR[0][0], t->R[0][0], 48*4*sizeof(double _Complex ) );

    } else if ( strcmp ( irrep, "H" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 4, 48) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < 48; i++ ) {
        t->rid[i] = i;
        rot_rotation_matrix_spherical_basis ( t->R[i], 3, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      }

      memcpy ( t->rmid, t->rmid, 48*sizeof(int) );
      memcpy ( t->IR[0][0], t->R[0][0], 48*16*sizeof(double _Complex ) );

    } else {
      fprintf(stderr, "[set_rot_mat_table_cubic_double_cover] unknown irrep name %s\n", irrep );
      return(1);
    }

  /* LG 2C4v */
  } else if ( strcmp ( group, "2C4v" ) == 0 ) {

    int rid[8]  = {  0,  3,  6,  9, 12, 15, 18, 47 };
    int rmid[8] = {  1,  2,  4,  5, 37, 38, 43, 44 };

    /* LG 2C4v irrep A1 */
    if ( strcmp ( irrep, "A1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, 8) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  8*sizeof(int) );
      memcpy ( t->rmid, r,id, 8*sizeof(int) );

      for ( int i = 0; i < 8; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

    /* LG 2C4v irrep A2 */
    } else if ( strcmp ( irrep, "A2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, 8) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  8*sizeof(int) );
      memcpy ( t->rmid, r,id, 8*sizeof(int) );

      for ( int i = 0; i < 8; i++ ) {
        t->R[i][0][0]  =  1.;
        t->IR[i][0][0] = -1.;
      }

    /* LG 2C4v irrep B1 */
    } else if ( strcmp ( irrep, "B1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, 8) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  8*sizeof(int) );
      memcpy ( t->rmid, r,id, 8*sizeof(int) );

      for ( int i = 0; i < 8; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

      /* 2C8' */
      t->R[3][0][0]  = -1.;
      t->R[4][0][0]  = -1.;
      /* 2C8 */
      t->R[5][0][0]  = -1.;
      t->R[6][0][0]  = -1.;

      /* 4IC4' */
      t->IR[4][0][0] = -1.;
      t->IR[5][0][0] = -1.;
      t->IR[6][0][0] = -1.;
      t->IR[7][0][0] = -1.;

    /* LG 2C4v irrep B2 */
    } else if ( strcmp ( irrep, "B2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, 8) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  8*sizeof(int) );
      memcpy ( t->rmid, r,id, 8*sizeof(int) );

      for ( int i = 0; i < 8; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

      /* 2C8' */
      t->R[3][0][0]  = -1.;
      t->R[4][0][0]  = -1.;
      /* 2C8 */
      t->R[5][0][0]  = -1.;
      t->R[6][0][0]  = -1.;

      /* 4IC4' */
      t->IR[0][0][0] = -1.;
      t->IR[1][0][0] = -1.;
      t->IR[2][0][0] = -1.;
      t->IR[3][0][0] = -1.;

    /* LG 2C4v irrep E */
    } else if ( strcmp ( irrep, "E" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, 8) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  8*sizeof(int) );
      memcpy ( t->rmid, r,id, 8*sizeof(int) );





  /* LG 2C2v */
  } else if ( strcmp ( group, "2C2v" ) == 0 ) {


  /* LG 2C3v */
  } else if ( strcmp ( group, "2C3v" ) == 0 ) {

  } else if {
    fprintf(stderr, "[set_rot_mat_table_cubic_double_cover] unknown group name %s\n", group );
    return(1);
  }
  



}  /* end of set_rot_mat_table_cubic_group_double_cover */

/***********************************************************/
/***********************************************************/

}  /* end of namespace cvc */
