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
#include "ranlxd.h"

#include "cvc_complex.h"
#include "fotran_name_mangling.h"
#include "ilinalg.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "mpi_init.h"
#include "cvc_utils.h"
#include "rotations.h"
#include "clebsch_gordan.h"
#include "table_init_i.h"
#include "table_init_c.h"
#include "table_init_z.h"
#include "table_init_asym_z.h"
#include "group_projection.h"

namespace cvc {

little_group_type *little_groups = NULL;

/***********************************************************/
/***********************************************************/

int little_group_read_list (little_group_type **lg, const char * filename )  {
 
  char comma[] = ",";

  char line[1000];
  int nline = 0;
  char r_str[100], rm_str[1000], r_aux[1000], lirrep_str[1000];

  FILE *ofs = fopen(filename, "r");
  if ( ofs == NULL ) {
    fprintf(stderr, "[little_group_read_list] Error from fopen trying to open file %s\n", filename);
    return(-1);
  }
   
  /***********************************************************/
  /***********************************************************/

  while ( fgets ( line, 1000, ofs) != NULL) {
    nline++;
  }
  fprintf(stdout, "# [little_group_read_list] number of entries = %d\n", nline);

  /***********************************************************/
  /***********************************************************/
              
  /***********************************************************
   * allocate lg list
   ***********************************************************/
  little_group_init ( lg, nline );
  if ( *lg == NULL ) {
    fprintf(stderr, "[little_group_read_table] Error from little_group_init\n");
    return(-2);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * read line-wise from file
   ***********************************************************/
  rewind( ofs );
  for ( int i = 0; i < nline; i++ ) {
    fscanf(ofs, "%s %d %d %d %s %s %s %s",  
        (*lg)[i].parent_name,
        (*lg)[i].d, (*lg)[i].d+1, (*lg)[i].d+2,
        (*lg)[i].name,
        r_str, rm_str, lirrep_str );

    /* TEST */
    if ( g_verbose > 3 ) {
      fprintf(stdout, "# [little_group_read_list] %s --- %d --- %d --- %d --- %s --- %s --- %s --- %s\n", (*lg)[i].parent_name, 
          (*lg)[i].d[0], (*lg)[i].d[1], (*lg)[i].d[2], (*lg)[i].name, r_str, rm_str, lirrep_str);
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



    /***********************************************************
     * extract the list of irreps for current little group
     ***********************************************************/
    strcpy ( r_aux, lirrep_str );
    ptr = strtok( r_aux, comma );
    (*lg)[i].nirrep = 1;
    while ( (ptr = strtok( NULL, comma ) ) != NULL ) (*lg)[i].nirrep++;
    fprintf(stdout, "# [little_group_read_list] %d nirrep %d\n", i, (*lg)[i].nirrep);

    (*lg)[i].lirrep = init_2level_ctable ( (*lg)[i].nirrep, 20 );
    if ( (*lg)[i].lirrep == NULL ) {
      fprintf(stderr, "[little_group_read_list] Error from init_2level_ctable %s %d\n", __FILE__, __LINE__);
      return(-3);
    }

    strcpy ( r_aux, lirrep_str );
    ptr = strtok( r_aux, comma );
    /* fprintf(stdout, "# [little_group_read_list] 0 ptr = %s\n", ptr);  */

    strcpy ( (*lg)[i].lirrep[0] , ptr );
    for ( int k = 1; k < (*lg)[i].nirrep; k++ ) {
      ptr = strtok( NULL, comma );
      /* fprintf(stdout, "# [little_group_read_list] %d ptr = %s\n", k, ptr); */
      if ( ptr == NULL ) {
        fprintf(stderr, "[little_group_read_list] Error from strtok\n");
        return(-5);
      }
      strcpy( (*lg)[i].lirrep[k], ptr );
    }
    
    fprintf(stdout, "# [little_group_read_list]\n");

  }  /* end of loop on lines */

  /***********************************************************/
  /***********************************************************/

  fclose( ofs );

  /***********************************************************/
  /***********************************************************/

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

    (*lg)[i].nirrep = 0;
    (*lg)[i].lirrep = NULL;
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

    fini_2level_ctable ( &((*lg)[i].lirrep) );
  }

  free ( *lg ); *lg = NULL;
  return;
}  /*  little_group_fini */

/***********************************************************/
/***********************************************************/

void little_group_show ( little_group_type *lg, FILE*ofs, int n) {

  for ( int i = 0; i < n; i++ ) {
    fprintf(stdout, "# [little_group_show]\n");
    fprintf(stdout, "# [little_group_show] ----------------------------------------------------------------------------------\n");
    fprintf(stdout, "# [little_group_show]\n# [little_group_show] %4d %s\n", i, lg[i].name );
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
    for ( int k = 0; k < lg[i].nirrep; k++ ) {
      fprintf(stdout, "# [little_group_show] irrep[%d] = %20s\n", k, lg[i].lirrep[k] );
    }
  }
  return;
}  /* end of little_group_show */

/***********************************************************/
/***********************************************************/

void init_rot_mat_table (rot_mat_table_type *t ) {
  strcpy( t->group, "NA" );
  strcpy( t->irrep, "NA" );
  t->n    = 0;
  t->dim  = 0;
  t->rid  = NULL;
  t->rmid = NULL;
  t->R    = NULL;
  t->IR   = NULL;
  return;
}  /* end of init_rot_mat_table */

/***********************************************************/
/***********************************************************/

int rot_mat_table_copy (rot_mat_table_type *t, rot_mat_table_type *s ) {
  strcpy( t->group, s->group );
  strcpy( t->irrep, s->irrep );
  t->n    = s->n;
  t->dim  = 0;
  if ( ( t->rid = (int*)malloc ( t->n * sizeof(int)) ) == NULL ) {
    fprintf( stderr, "[rot_mat_table_copy] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  memcpy ( t->rid, s->rid, t->n * sizeof(int) );
  if ( ( t->rmid = (int*)malloc ( t->n * sizeof(int)) ) == NULL ) {
    fprintf( stderr, "[rot_mat_table_copy] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  memcpy ( t->rmid, s->rmid, t->n * sizeof(int) );
                
  t->R = init_3level_ztable ( (size_t)(t->n), (size_t)(t->dim), (size_t)(t->dim) );
  if ( t->R == NULL ) {
    fprintf (stderr, "[rot_mat_table_copy] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
    return(2);
  }
  memcpy ( t->R[0][0], s->R[0][0], t->n * t->dim*t->dim * sizeof(double _Complex) );
  
  t->IR = init_3level_ztable ( (size_t)(t->n), (size_t)(t->dim), (size_t)(t->dim) );
  if ( t->IR == NULL ) {
    fprintf (stderr, "[rot_mat_table_copy] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
    return(3);
  }
  memcpy ( t->IR[0][0], s->IR[0][0], t->n * t->dim*t->dim * sizeof(double _Complex) );
  return(0);
}  // end of rot_mat_table_copy


/***********************************************************/
/***********************************************************/

void fini_rot_mat_table (rot_mat_table_type *t ) {
  strcpy( t->group, "NA" );
  strcpy( t->irrep, "NA" );
  t->n   = 0;
  t->dim = 0;
  if ( t->rid  != NULL ) { free ( t->rid );  t->rid  = NULL; }
  if ( t->rmid != NULL ) { free ( t->rmid ); t->rmid = NULL; }
  if ( t->R    != NULL ) { fini_3level_ztable ( &(t->R) ); }
  if ( t->IR   != NULL ) { fini_3level_ztable ( &(t->IR) ); }
  return;
}  /* end of fini_rot_mat_table */

/***********************************************************/
/***********************************************************/

int alloc_rot_mat_table ( rot_mat_table_type *t, const char*group, const char*irrep, const int dim, const int n ) {
  
  if ( t->n != 0 ) {
    fprintf(stdout, "# [alloc_rot_mat_table] deleting existing rotation matrix table\n");
    fini_rot_mat_table ( t );
  }
  strcpy ( t->group, group );
  strcpy ( t->irrep, irrep );
  t->n   = n;
  t->dim = dim;
  if ( ( t->rid = (int*)malloc( n * sizeof(int) ) ) == NULL ) {
    fprintf(stderr, "[alloc_rot_mat_table] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  if ( ( t->rmid = (int*)malloc( n * sizeof(int) ) ) == NULL ) {
    fprintf(stderr, "[alloc_rot_mat_table] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  t->R = init_3level_ztable ( (size_t)n, (size_t)dim, (size_t)dim );
  if ( t->R == NULL ) {
    fprintf(stderr, "[alloc_rot_mat_table] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }
  t->IR = init_3level_ztable ( (size_t)n, (size_t)dim, (size_t)dim );
  if ( t->IR == NULL ) {
    fprintf(stderr, "[alloc_rot_mat_table] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  return(0);

}  // end of alloc_rot_mat_table

/***********************************************************/
/***********************************************************/

/***********************************************************
 * 
 ***********************************************************/
int rot_mat_table_get_spin2 ( rot_mat_table_type *t ) {

  char irrep[20];

  if ( strcmp( t->group , "SU2" ) != 0 ) {
    fprintf(stderr, "[rot_mat_table_get_spin] Error, only for SU2 irreps\n");
    return(-1);
  }

  sscanf ( t->irrep, "spin%s", irrep );

  if ( strcmp( irrep, "1_2+1_2" ) == 0 ) {
    return(1);
  } else {
    if ( strchr ( irrep, '_' ) ==  NULL ) {
      return( 2*atoi(irrep) );
    } else {
      char a[10], b[10];
      sscanf( irrep, "%s_%s", a, b);
      return( atoi(a) );
    }
  }
  return(-1);
}  /* end of rot_mat_table_get_spin2 */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * 
 ***********************************************************/
int rot_mat_table_get_bispinor ( rot_mat_table_type *t ) {

  if ( strchr ( t->irrep, '+' ) ==  NULL ) {
    return(0);
  } else {
    return(1);
  }
}  /* end of rot_mat_table_get_bispinor */

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

  if ( bispinor ) {
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
  P.d = (1 + bispinor) * (J2 + 1);
  P.m = NULL;

  P.m = init_2level_ztable ( (size_t)(P.d), (size_t)(P.d) );
  if ( P.m == NULL ) {
    fprintf(stderr, "[set_rot_mat_table_spin] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  /* set parity matrix if applicable */
  rot_inversion_matrix_spherical_basis ( P.m, J2, bispinor );


  if  ( !bispinor ) {
    for ( int i = 0; i < 48; i++ ) {
      t->rid[i]  = i;
      t->rmid[i] = i;
      rot_rotation_matrix_spherical_basis ( t->R[i], J2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      rot_mat_ti_mat ( t->IR[i], P.m, t->R[i], J2+1);
    }
  } else {

    // fprintf(stdout, "# [set_rot_mat_table_spin] setting spin rotations with bispinor %d\n", bispinor );

    for ( int i = 0; i < 48; i++ ) {
      t->rid[i]  = i;
      t->rmid[i] = i;
      rot_bispinor_rotation_matrix_spherical_basis ( t->R[i], cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      rot_mat_ti_mat ( t->IR[i], P.m, t->R[i], 2*(J2+1) );
    }
  }

  fini_2level_ztable ( &(P.m) );
  return(0);
}  /* end of set_rot_mat_table_spin */


/***********************************************************/
/***********************************************************/

/***********************************************************
 * J2 = 2 x spin
 ***********************************************************/
int set_rot_mat_table_spin_single_cover ( rot_mat_table_type *t, int J2, int const version , int const setby ) {

  int exitstatus;
  char name[20];
  rotation_matrix_type P;
  rotation_type *rotlist;

  if ( J2 % 2 != 0 ) {
    fprintf ( stderr, "[set_rot_mat_table_spin_single_cover] Error, only for integer spin\n" );
    return(1);
  }
 
  sprintf(name, "spin%d", J2/2);
  fprintf(stdout, "# [set_rot_mat_table_spin_single_cover] name = %s\n", name);

  if ( ( exitstatus = alloc_rot_mat_table ( t, "SU2", name, (J2+1), 24 ) ) != 0 ) {
    fprintf(stderr, "[set_rot_mat_table_spin_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
  P.d = (J2 + 1);
  P.m = NULL;

  P.m = init_2level_ztable ( (size_t)(P.d), (size_t)(P.d) );
  if ( P.m == NULL ) {
    fprintf(stderr, "[set_rot_mat_table_spin_single_cover] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  /* set parity matrix if applicable */
  rot_inversion_matrix_spherical_basis ( P.m, J2, 0 );

  switch  (version)  {
    case 0:
    default:
      rotlist = cubic_group_rotations;
      break;
    case 1:
      rotlist = cubic_group_rotations_v2;
      break;
  }

  // loop on rotation group elements
  for ( int i = 0; i < 24; i++ ) {
    t->rid[i]  = i;
    t->rmid[i] = i;

    if ( setby == 0 ) {
      rot_rotation_matrix_spherical_basis ( t->R[i], J2, rotlist[i].n, rotlist[i].w );
    } else if ( setby == 1 ) {
      rot_rotation_matrix_spherical_basis_Wigner_D ( t->R[i], J2, rotlist[i].a );
    }

    rot_mat_ti_mat ( t->IR[i], P.m, t->R[i], J2+1);
  }

  fini_2level_ztable ( &(P.m) );
  return(0);
}  /* end of set_rot_mat_table_spin_single_cover */


/***********************************************************/
/***********************************************************/

/***********************************************************
 * irrep matrices for double cover
 ***********************************************************/
// #include "set_rot_mat_table_cubic_group_double_cover.cpp"
#include "set_rot_mat_table_cubic_group_double_cover_gu.cpp"

#include "set_rot_mat_table_cubic_group_single_cover_v2.cpp"

/***********************************************************/
/***********************************************************/

/***********************************************************
 * rot_mat_table_printf
 ***********************************************************/
void rot_mat_table_printf ( rot_mat_table_type *t, char*name, FILE*ofs ) {

  char name_full[100];
  fprintf( ofs, "# [rot_mat_table_printf] %s.group = %s\n", name, t->group );
  fprintf( ofs, "# [rot_mat_table_printf] %s.irrep = %s\n", name, t->irrep );
  fprintf( ofs, "# [rot_mat_table_printf] %s.n     = %d\n", name, t->n );
  fprintf( ofs, "# [rot_mat_table_printf] %s.dim   = %d\n", name, t->dim );
  fprintf( ofs, "%s_R <- list()\n", name );
  fprintf( ofs, "%s_IR <- list()\n", name );

  for ( int i = 0; i < t->n; i++ ) {
    sprintf( name_full, "%s_R[[%2d]]", name, t->rid[i]+1 );
    rot_printf_matrix ( t->R[i], t->dim, name_full, ofs );
    fprintf( ofs, "\n");
  }
  fprintf( ofs, "\n\n");

  for ( int i = 0; i < t->n; i++ ) {
    sprintf( name_full, "%s_IR[[%2d]]", name, t->rmid[i]+1 );
    rot_printf_matrix ( t->IR[i], t->dim, name_full, ofs );
    fprintf( ofs, "\n");
  }
  fprintf( ofs, "\n\n");
  return;
}  /* end of rot_mat_table_printf */

/***********************************************************/
/***********************************************************/

int rot_mat_mult_table ( int***mtab, rot_mat_table_type *t ) {

  const double eps = 5.e-15;

  if ( *mtab == NULL ) {
    *mtab = init_2level_itable ( t->n, t->n );
    if ( *mtab == NULL ) {
      fprintf(stderr, "[rot_mat_mult_table] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }
  double _Complex ** A = init_2level_ztable ( (size_t)(t->dim), (size_t)(t->dim) );
  if ( A == NULL ) {
    fprintf(stderr, "[rot_mat_mult_table] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  for ( int i = 0; i < t->n; i++ ) {
  for ( int k = 0; k < t->n; k++ ) {
    rot_mat_ti_mat ( A, t->R[i], t->R[k], t->dim );

#if 0
    char name[20];
    sprintf(name, "A%.2d", t->rid[i] );
    rot_printf_matrix ( t->R[i], t->dim, name, stdout );
    sprintf(name, "B%.2d", t->rid[k] );
    rot_printf_matrix ( t->R[k], t->dim, name, stdout );
    sprintf(name, "A%.2dxB%.2d", t->rid[i], t->rid[k] );
    rot_printf_matrix ( A, t->dim, name, stdout );
#endif  /* if 0 */

    int found_match = 0;
    for ( int l = 0; l < t->n; l++ ) {
      double diff_norm = rot_mat_diff_norm ( A, t->R[l], t->dim );
      fprintf(stdout, "# [rot_mat_mult_table] | %2d x %2d - %2d | = %e\n", i, k, l, diff_norm );
      if ( diff_norm < eps ) {
        found_match = 1;
        fprintf(stdout, "# [rot_mat_mult_table] %2d x %2d matches %2d at %e\n", i, k, l, diff_norm);
        (*mtab)[i][k] = t->rid[l];
        break;
      }
    }
    if ( ! found_match ) {
      fprintf(stderr, "[rot_mat_mult_table] Error, no match found for %2d x %2d %s %d\n", i, k, __FILE__, __LINE__);
      return(3);
    }
  }}

  fini_2level_ztable ( &A );

  if ( g_verbose > 2 ) {
    /* print the table */
    fprintf(stdout, "   *|");
    for ( int i = 0; i < t->n; i++ ) fprintf(stdout, " %2d", t->rid[i]);
    fprintf(stdout,"\n");
    fprintf(stdout, "-----");
    for ( int i = 0; i < t->n; i++ ) fprintf(stdout, "---" );
    fprintf(stdout,"\n");

    for ( int i = 0; i < t->n; i++ ) {
      fprintf(stdout, "  %2d|", t->rid[i]);
    for ( int k = 0; k < t->n; k++ ) {
      fprintf(stdout, " %2d", (*mtab)[i][k]);
    }
      fprintf(stdout, "\n");
    }
  }

  return(0);
}  /* end of rot_mat_mult_table */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * d = P L / (2 pi) integer 3-vector
 ***********************************************************/
int rot_mat_table_is_lg ( rot_mat_table_type *t, int d[3] ) {

  int Rd[3];
  int is_lg = 1;

  double _Complex **R = init_2level_ztable ( 3, 3);
  if ( R == NULL ) {
    fprintf(stderr, "[rot_mat_table_is_lg] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(-1);
  }
  double _Complex ** A = init_2level_ztable (3, 3);
  if ( A == NULL ) {
    fprintf(stderr, "[rot_mat_table_is_lg] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(-1);
  }

  /***********************************************************
   * check Rd = d
   ***********************************************************/
  for ( int i = 0; i < t->n; i++ ) {

    rot_rotation_matrix_spherical_basis ( R, 2, cubic_group_double_cover_rotations[t->rid[i]].n, cubic_group_double_cover_rotations[t->rid[i]].w );

    rot_spherical2cartesian_3x3 ( A, R );

    if ( ! rot_mat_check_is_real_int ( A, 3 ) ) {
      fprintf(stderr, "# [rot_mat_table_is_lg] R %2d was not real int\n", t->rid[i] );
      return(-2);
    }
    rot_point ( Rd, d, A );

    is_lg = is_lg &&  ( ( abs( d[0] - Rd[0] ) + abs( d[1] - Rd[1] ) + abs( d[2] - Rd[2] ) )  == 0 );
  }

  /***********************************************************
   * check Rd = -d
   ***********************************************************/
  for ( int i = 0; i < t->n; i++ ) {

    rot_rotation_matrix_spherical_basis ( R, 2, cubic_group_double_cover_rotations[t->rmid[i]].n, cubic_group_double_cover_rotations[t->rmid[i]].w );

    rot_spherical2cartesian_3x3 ( A, R );

    if ( ! rot_mat_check_is_real_int ( A, 3 ) ) {
      fprintf(stderr, "# [rot_mat_table_is_lg] R %2d was not real int\n", t->rmid[i] );
      return(-2);
    }

    rot_point ( Rd, d, A );

    is_lg = is_lg &&  ( ( abs( d[0] + Rd[0] ) + abs( d[1] + Rd[1] ) + abs( d[2] + Rd[2] ) )  == 0 );
  }

  fini_2level_ztable ( &R );
  fini_2level_ztable ( &A );

  return(is_lg);
}  /* end of rot_mat_table_is_lg */

/***********************************************************/
/***********************************************************/

/***********************************************************
 ***********************************************************/

int rot_mat_table_get_lg ( rot_mat_table_type *t, int d[3] ) {

  int exitstatus;
  int Rd[3];
  int rid_tmp[48];
  char group_name[40], irrep_name[40];

  if ( t == NULL ) {
    fprintf(stderr, "[rot_mat_table_get_lg] need (empty) rot mat table as input\n");
    return(1);
  }

  double _Complex ***R_tmp = init_3level_ztable ( 48, 3, 3);
  if ( R_tmp == NULL ) {
    fprintf(stderr, "[rot_mat_table_get_lg] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(-1);
  }
  double _Complex ** A = init_2level_ztable ( 3, 3);
  if ( A == NULL ) {
    fprintf(stderr, "[rot_mat_table_get_lg] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(-1);
  }

  /***********************************************************
   * fill in R : Rd = d
   ***********************************************************/
  int countR = 0;
  for ( int i = 0; i < 48; i++ ) {

    rot_rotation_matrix_spherical_basis ( A, 2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );

    rot_spherical2cartesian_3x3 ( A, A );

    if ( ! rot_mat_check_is_real_int ( A, 3 ) ) {
      fprintf(stderr, "# [rot_mat_table_get_lg] R %2d was not real int\n", i );
      return(-2);
    }
    rot_point ( Rd, d, A );

    if ( ( abs( d[0] - Rd[0] ) + abs( d[1] - Rd[1] ) + abs( d[2] - Rd[2] ) )  == 0  ) {
      rid_tmp[countR] = i;
      rot_mat_assign ( R_tmp[countR], A, 3 );
      countR++;
    }
  }

  sprintf( group_name, "LG_dx%ddy%ddz%d", d[0], d[1], d[2] );
  sprintf( irrep_name, "%s", "spin1_kartesian" );

  if ( ( exitstatus = alloc_rot_mat_table ( t, group_name, irrep_name, 3, countR ) ) != 0 ) {
    fprintf(stderr, "[rot_mat_table_get_lg] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }
  memcpy( t->rid, rid_tmp, countR * sizeof( int ) );
  memcpy( t->R[0][0], R_tmp[0][0], countR * 3*3 * sizeof(double _Complex ) );

  /***********************************************************
   * fill in R : Rd = -d
   ***********************************************************/
  countR = 0;
  memset ( R_tmp[0][0], 0, 48 * 3*3 * sizeof(double _Complex ) );
  memset ( rid_tmp, 0, 48 * sizeof( int ) );

  for ( int i = 0; i < 48; i++ ) {

    rot_rotation_matrix_spherical_basis ( A, 2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );

    rot_spherical2cartesian_3x3 ( A, A );

    if ( ! rot_mat_check_is_real_int ( A, 3 ) ) {
      fprintf(stderr, "# [rot_mat_table_get_lg] R %2d was not real int\n", i );
      return(-2);
    }

    rot_point ( Rd, d, A );

    if ( ( abs( d[0] + Rd[0] ) + abs( d[1] + Rd[1] ) + abs( d[2] + Rd[2] ) )  == 0 ) {
      rid_tmp[countR] = i;
      rot_mat_assign ( R_tmp[countR], A, 3 );
      countR++;
    }
  }

  if ( t->n != countR ) {
    fprintf(stderr, "[rot_mat_table_get_lg] Error, number of R:Rd=d %d differs from number of R:Rd=-d %d\n", t->n, countR );
    return(3);
  }
  memcpy( t->rmid, rid_tmp, countR * sizeof( int ) );
  memcpy( t->IR[0][0], R_tmp[0][0], countR * 3*3 * sizeof(double _Complex ) );

  fini_3level_ztable ( &R_tmp );
  fini_2level_ztable ( &A );

  return(0);
}  /* end of rot_mat_table_get_lg */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * find rot d1 to d2
 ***********************************************************/
int rot_mat_table_get_d2d ( rot_mat_table_type *t, int d1[3], int d2[3] ) {

  int exitstatus;
  int Rd[3];
  int rid_tmp[48];
  char group_name[40], irrep_name[40];

  if ( t == NULL ) {
    fprintf(stderr, "[rot_mat_table_get_lg] need (empty) rot mat table as input\n");
    return(1);
  }

  double _Complex *** R_tmp = init_3level_ztable ( 48, 3, 3);
  if ( R_tmp == NULL ) {
    fprintf(stderr, "[rot_mat_table_get_lg] Error from init_3level_ztable %s %d\n",  __FILE__, __LINE__);
    return(-1);
  }
  double _Complex ** A = init_2level_ztable ( 3, 3);
  if ( A == NULL ) {
    fprintf(stderr, "[rot_mat_table_get_lg] Error from init_2level_ztabe  %s %d\n", __FILE__, __LINE__);
    return(-1);
  }

  /***********************************************************
   * find R : Rd1 = d2
   ***********************************************************/
  int countR = 0;
  for ( int i = 0; i < 48; i++ ) {

    rot_rotation_matrix_spherical_basis ( A, 2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );

    rot_spherical2cartesian_3x3 ( A, A );

    if ( ! rot_mat_check_is_real_int ( A, 3 ) ) {
      fprintf(stderr, "# [rot_mat_table_get_lg] R %2d was not real int\n", i );
      return(-2);
    }
    rot_point ( Rd, d1, A );

    if ( ( abs( d2[0] - Rd[0] ) + abs( d2[1] - Rd[1] ) + abs( d2[2] - Rd[2] ) )  == 0  ) {
      rid_tmp[countR] = i;
      rot_mat_assign ( R_tmp[countR], A, 3 );
      countR++;
    }
  }

  sprintf( group_name, "dx%ddy%ddz%d_to_dx%ddy%ddz%d", d1[0], d1[1], d1[2], d2[0], d2[1], d2[2] );
  sprintf( irrep_name, "%s", "spin1_kartesian" );

  if ( ( exitstatus = alloc_rot_mat_table ( t, group_name, irrep_name, 3, countR ) ) != 0 ) {
    fprintf(stderr, "[rot_mat_table_get_lg] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }
  memcpy( t->rid, rid_tmp, countR * sizeof( int ) );
  memcpy( t->R[0][0], R_tmp[0][0], countR * 3*3 * sizeof(double _Complex ) );

  fini_3level_ztable ( &R_tmp );
  fini_2level_ztable ( &A );

  return(0);
}  /* end of rot_mat_table_get_d2d */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * function to make character table
 ***********************************************************/
int rot_mat_table_character ( double _Complex ***rc, rot_mat_table_type *t ) {
  
  if ( *rc == NULL ) {
    *rc = init_2level_ztable ( 2, (size_t)(t->n) );
    if ( *rc == NULL ) {
      fprintf(stdout, "[rot_mat_table_character] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
      return(1);
    }
  }

  memset( (*rc)[0], 0, 2*t->n*sizeof(double _Complex ) );
  for ( int i = 0; i < t->n; i++ ) {
    for ( int k = 0; k < t->dim; k++ ) {
      (*rc)[0][i] += t->R[i][k][k];
    }
    for ( int k = 0; k < t->dim; k++ ) {
      (*rc)[1][i] += t->IR[i][k][k];
    }
  }  /* end of loop on rotations */

  return(0);
}  /* end of rot_mat_table_character */


/***********************************************************/
/***********************************************************/

/***********************************************************
 * test orthogonality relation for irrep matrices
 ***********************************************************/
int rot_mat_table_orthogonality ( rot_mat_table_type *t1, rot_mat_table_type *t2 ) {

  const double eps = 5.e-14;

  double _Complex zexp = 0;
  if ( t1->n != t2->n ) {
    fprintf( stderr, "[rot_mat_table_orthogonality] Error, number of elements must be the same\n");
    return(1);
  }

  if ( strcmp( t1->group, t2->group ) != 0 ) { 
    fprintf( stderr, "[rot_mat_table_orthogonality] Error, groups must be the same\n");
    return(2);
  }

  fprintf (stdout, "# [rot_mat_table_orthogonality] checking group %s irrep %s vs irrep %s\n", t1->group, t1->irrep, t2->irrep );

  /* loop on matrix elements of t1 */
  for ( int i1 = 0; i1 < t1->dim; i1++ ) {
  for ( int i2 = 0; i2 < t1->dim; i2++ ) {

    /* loop on matrix elements of t2 */
    for ( int k1 = 0; k1 < t2->dim; k1++ ) {
    for ( int k2 = 0; k2 < t2->dim; k2++ ) {
 
      if ( strcmp( t1->irrep, t2->irrep ) != 0 ) {
        zexp = 0.;
      } else {
        if ( i1 == k1 && i2 == k2 ) {
          /* 2 x times number of rotations / dimension */
          /* times two because we consider R and IR */
          zexp = 2. * (double)t1->n / (double)t1->dim;
        } else {
          zexp = 0.;
        }
      }

      double _Complex z = 0., zi = 0;

      /* loop on rotations with group */     
      for ( int l = 0; l < t1->n; l++ ) {
        if ( t1->rid[l] != t2->rid[l] ) {
          fprintf(stderr, "[rot_mat_table_orthogonality] Error, rids do not match %d %d\n", t1->rid[l], t2->rid[l] );
          return(3);
        }

        z  += t1->R[l][i1][i2]  * conj ( t2->R[l][k1][k2] );
        // z  += t1->R[l][i1][i2]  * t2->R[l][k1][k2];

        if ( t1->rmid[l] != t2->rmid[l] ) {
          fprintf(stderr, "[rot_mat_table_orthogonality] Error, rmids do not match %d %d\n", t1->rmid[l], t2->rmid[l] );
          return(4);
        }

        zi += t1->IR[l][i1][i2] * conj ( t2->IR[l][k1][k2] );
        // zi += t1->IR[l][i1][i2] * t2->IR[l][k1][k2];

      }  /* end of loop on rotations */
      int orthogonal = (int)(cabs(z+zi-zexp) < eps);
      if ( !orthogonal ) {
        fprintf( stderr, "[rot_mat_table_orthogonality] group %s irreps %s --- %s elements (%2d %2d) ---  (%2d %2d) not orthogonal\n", t1->group, t1->irrep, t2->irrep, i1, i2, k1, k2);
        // return(5);
      }
      fprintf ( stdout, "# [rot_mat_table_orthogonality] %d %d    %d %d z %16.7e %16.7e zi %16.7e %16.7e z+zi %16.7e %16.7e zexp %16.7e orthogonal %d\n", i1, i2, k1, k2, 
          creal(z), cimag(z), creal(zi), cimag(zi), creal(z+zi), cimag(z+zi), creal(zexp), orthogonal);

    }}  /* end of loop on t2 matrix elements */
  }}  /* end of loop on t1 matrix elements */

  return(0);
}  /* end of rot_mat_table_orthogonality */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/

int init_little_group_projector (little_group_projector_type *p ) {

  p->n              = 0;
  p->rspin          = NULL;
  p->rtarget        = NULL;
  p->rp             = NULL;
  p->P[0]           = 0;
  p->P[1]           = 0;
  p->P[2]           = 0;
  p->p              = NULL;
  /* p->c = NULL; */
  p->ref_row_target = -1;
  p->row_target     = -1;
  p->ref_row_spin   = NULL;
  p->parity         = NULL;
  p->refframerot    = -1;
  strcpy ( p->name, "NA" );
  
  return(0);
}   /* end of init_little_group_projector */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
int fini_little_group_projector (little_group_projector_type *p ) {

  if ( p->rspin != NULL ) {
    for ( int i = 0; i < p->n; i++ ) {
      fini_rot_mat_table ( &(p->rspin[i]) );
    }
    free ( p->rspin );
    p->rspin = NULL;
  }
  if ( p->rtarget != NULL ) {
    fini_rot_mat_table ( p->rtarget );
    free ( p->rtarget );
    p->rtarget = NULL;
  }
  if ( p->rp != NULL ) {
    fini_rot_mat_table ( p->rp );
    free ( p->rp );
    p->rp = NULL;
  }

  p->refframerot = -1;
  
  fini_2level_itable ( &(p->p) );

#if 0
  if ( p->c != NULL  ) {
    if ( p->c[0] != NULL  ) free ( p->c[0] );
    free ( p->c );
    p->c = NULL;
  }
#endif
  strcpy ( p->name, "NA" );
  p->n              = 0;
  p->P[0]           = 0;
  p->P[1]           = 0;
  p->P[2]           = 0;
  p->ref_row_target = -1;
  p->row_target     = -1;
  fini_1level_itable ( &(p->ref_row_spin) );
  fini_1level_itable ( &(p->parity) );

  return(0);
}   /* end of fini_little_group_projector */

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/

int little_group_projector_show (little_group_projector_type *p, FILE*ofs, int with_mat ) {

  fprintf( ofs, "\n\n# [little_group_projector_show] begin of projector\n" );
  fprintf( ofs, "# [little_group_projector_show] number of interpolators = %d\n", p->n);
  fprintf( ofs, "# [little_group_projector_show] total momentum P = %2d %2d %2d\n", p->P[0], p->P[1], p->P[2] );
 
  for ( int i = 0; i < p->n; i++ ) {
    fprintf( ofs, "# [little_group_projector_show] p[%i] = %2d %2d %2d\n", i, p->p[i][0], p->p[i][1], p->p[i][2] );
  }

#if 0
  fprintf( ofs, "# [little_group_projector_show] linear combination for spin-matrix row\n" );
  for ( int i = 0; i < p->n; i++ ) {
    for ( int k = 0; k < p->rspin[i].dim; k++ ) {
      fprintf( ofs, "# [little_group_projector_show] c[%d][%d] = %16.7e %16.7e\n", i, k, creal(p->c[i][k]), cimag(p->c[i][k] ));
    }
  }
#endif
  fprintf( ofs, "# [little_group_projector_show] correlator name    = %s\n", p->name );
  fprintf( ofs, "# [little_group_projector_show] row target         = %d\n", p->row_target );
  fprintf( ofs, "# [little_group_projector_show] ref row target     = %d\n", p->ref_row_target );

  fprintf( ofs, "# [little_group_projector_show] ref row for spins\n" );
  for ( int i = 0; i < p->n; i++ ) {
    fprintf( ofs, "# [little_group_projector_show]   spin(%d) ref row = %d\n", i, p->ref_row_spin[i] );
  }

  fprintf( ofs, "# [little_group_projector_show] interpolator-intrinsic parity\n" );
  for ( int i = 0; i < p->n; i++ ) {
    fprintf( ofs, "# [little_group_projector_show]   parity(%d) = %d\n", i, p->parity[i] );
  }

  if ( !with_mat ) {
    fprintf( ofs, "# [little_group_projector_show] target group     = %s\n", p->rtarget->group );
    fprintf( ofs, "# [little_group_projector_show] target irrep     = %s\n", p->rtarget->irrep );
    fprintf( ofs, "# [little_group_projector_show] target elements  = %d\n", p->rtarget->n );
    fprintf( ofs, "# [little_group_projector_show] target dim       = %d\n", p->rtarget->dim );

    for ( int i = 0; i < p->n; i++ ) {
      fprintf( ofs, "# [little_group_projector_show] spin(%d) group   = %s\n", i, p->rspin[i].group );
      fprintf( ofs, "# [little_group_projector_show] spin(%d) irrep   = %s\n", i, p->rspin[i].irrep );
    }
  }
  fprintf( ofs, "# [little_group_projector_show] reference frame rotation = %d\n", p->refframerot );
   
  if ( with_mat ) {
    /***********************************************************
     * show the rotation matrices
     ***********************************************************/
    rot_mat_table_printf ( p->rtarget, "rtarget", ofs );
    for ( int i = 0; i < p->n; i++ ) {
      char name[200];
      sprintf (name, "rspin%d", i);
      rot_mat_table_printf ( &(p->rspin[i]), name, ofs );
    }
    rot_mat_table_printf ( p->rp, "rp", ofs );
  }
  
  fprintf( ofs, "# [little_group_projector_show] end of projector\n\n" );
  return(0);
}  /* end of little_group_projector_show */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * p <- q
 ***********************************************************/
int little_group_projector_copy (little_group_projector_type *p, little_group_projector_type *q ) {

  p->n = q->n;
  if ( ( p->rspin = (rot_mat_table_type *)malloc ( p->n * sizeof( rot_mat_table_type ) ) ) == NULL ) {
    fprintf( stderr, "[little_group_projector_copy] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  for ( int i = 0; i < q->n; i++ ) {
    if ( rot_mat_table_copy ( &(p->rspin[i]), &(q->rspin[i]) ) != 0 ) {
      fprintf( stderr, "[little_group_projector_copy] Error from rot_mat_table_copy %s %d\n", __FILE__, __LINE__);
      return(2);
    }
  }

  if ( ( p->rtarget= (rot_mat_table_type *)malloc (  sizeof( rot_mat_table_type ) ) ) == NULL ) {
    fprintf( stderr, "[little_group_projector_copy] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  if ( rot_mat_table_copy ( p->rtarget, q->rtarget ) != 0 ) {
    fprintf( stderr, "[little_group_projector_copy] Error from rot_mat_table_copy %s %d\n", __FILE__, __LINE__);
    return(3);
  }

  if ( ( p->rp = (rot_mat_table_type *)malloc ( sizeof( rot_mat_table_type ) ) ) == NULL ) {
    fprintf( stderr, "[little_group_projector_copy] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  if ( rot_mat_table_copy ( p->rp, q->rp ) != 0 ) {
    fprintf( stderr, "[little_group_projector_copy] Error from rot_mat_table_copy %s %d\n", __FILE__, __LINE__);
    return(4);
  }

  memcpy ( p->P, q->P, 3*sizeof(int));
  p->p = init_2level_itable ( p->n, 3 );
  if ( p->p == NULL ) {
    fprintf( stderr, "[little_group_projector_copy] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    return(5);
  }
  memcpy ( p->p[0], q->p[0], 3*p->n*sizeof(int) );

#if 0
  if ( ( p->c = (double _Complex**)malloc ( p->n * sizeof(double _Complex*) ) ) == NULL ) {
    fprintf( stderr, "[little_group_projector_copy] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
  int items = 0;
  for ( int i = 0; i < p->n ; i++) items += p->rspin[i].dim;
  if ( ( p->c[0] = (double _Complex*)malloc ( items * sizeof(double _Complex) ) ) == NULL ) {
    fprintf( stderr, "[little_group_projector_copy] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(7);
  }
  for ( int i = 1; i < p->n ; i++) p->c[i] = p->c[i-1] + p->rspin[i-1].dim;
  memcpy ( p->c[0], q->c[0], items * sizeof(double _Complex) );
#endif

  p->ref_row_spin = init_1level_itable ( p->n );
  if ( p->ref_row_spin == NULL ) {
    fprintf(stderr, "[little_group_projector_copy] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  memcpy ( p->ref_row_spin, q->ref_row_spin, p->n * sizeof(int) );

  p->parity = init_1level_itable ( p->n );
  if ( p->parity == NULL ) {
    fprintf(stderr, "[little_group_projector_copy] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  memcpy ( p->parity, q->parity, p->n * sizeof(int) );

  p->row_target     = q->row_target;
  p->ref_row_target = q->ref_row_target;
  strcpy ( p->name, q->name );
  
  p->refframerot = q->refframerot;

  return(0);
}   /* end of little_group_projector_copy */

/***********************************************************/
/***********************************************************/


/***********************************************************
 *
 ***********************************************************/
#include "little_group_projector_set.cpp"

/***********************************************************/
/***********************************************************/

/***********************************************************
 * print direct product of spin vectors
 ***********************************************************/
int spin_vector_asym_printf ( double _Complex **sv, int n, int*dim, char*name, FILE*ofs ) {

  const double eps = 9.e-15;

  fprintf( ofs, "# [spin_vector_asym_printf] %s\n", name );
  for ( int i = 0; i < n; i++ ) { 
    fprintf( ofs, "# [spin_vector_asym_printf]   %s(%d)\n", name, i );

    for ( int k = 0; k < dim[i]; k++ ) {
      fprintf( ofs, "   (%16.7e  +  %16.7e*1.i)\n", dgeps( creal (sv[i][k]), eps ) , dgeps ( cimag(sv[i][k]), eps ) );
      /* fprintf( ofs, "   %16.7e    %16.7e\n", creal (sv[i][k]), cimag(sv[i][k]) ); */
    }
  }
  return(0);
}  /* end of spin_vector_asym_prinf */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * norm of direct spin-vector product
 ***********************************************************/
double spin_vector_asym_norm2 ( double _Complex **sv, int n, int *dim ) {

  double norm2 = 1.;

  for ( int i = 0; i < n; i++ ) {
    double dtmp = 0.;

    for ( int k = 0; k < dim[i]; k++ ) {
      dtmp += creal ( sv[i][k] * conj ( sv[i][k] ) );
    }
    norm2 *= dtmp;
  }
  return(norm2);
}  /* end of spin_vector_asym_norm2 */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * norm of direct spin-vector product
 ***********************************************************/
void spin_vector_pl_eq_spin_vector_ti_co_asym ( double _Complex **sv1, double _Complex **sv2, double _Complex c, int n, int *dim ) {

  for ( int i = 0; i < n; i++ ) {

    for ( int k = 0; k < dim[i]; k++ ) {
      sv1[i][k] += sv2[i][k] * c;
    }
  }
  return;
}  /* end of spin_vector_pl_eq_spin_vector_ti_co_asym */


/***********************************************************/
/***********************************************************/

/***********************************************************
 * norm of direct spin-vector product
 ***********************************************************/
int spin_vector_asym_normalize ( double _Complex **sv, int n, int *dim ) {
  const double eps = 1.e-14;

  double normi = sqrt ( spin_vector_asym_norm2 ( sv, n, dim ) );
  fprintf ( stdout, "# [spin_vector_asym_normalize] norm = %16.7e\n", normi );
  normi = fabs ( normi ) < eps ? 0. : 1./normi;

  for ( int k = 0; k < n; k++ ) {
    for ( int l = 0; l < dim[k]; l++ ) {
      sv[k][l] *= normi;
    }
  }
  return( 0 );
}  /* end of spin_vector_asym_normalize */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * norm of direct spin-vector product
 ***********************************************************/
double spin_vector_asym_list_norm2 ( double _Complex ***sv, int nc, int n, int *dim ) {

  double norm2 = 0.;

  for ( int i = 0; i < nc; i++ ) {
    norm2 += spin_vector_asym_norm2 ( sv[i], n, dim );
  }
  return(norm2);
}  /* end of spin_vector_asym_list_norm2 */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * norm of direct spin-vector product
 ***********************************************************/
int spin_vector_asym_list_normalize ( double _Complex ***sv, int nc, int n, int *dim ) {
  const double eps = 5.e-15;

  double normi = sqrt ( spin_vector_asym_list_norm2 ( sv, nc, n, dim ) );
  fprintf ( stdout, "# [spin_vector_asym_list_normalize] norm = %16.7e\n", normi );
  normi = fabs ( normi ) < eps ? 0. : 1./normi;

  for ( int i = 0; i < nc; i++ ) {
    for ( int k = 0; k < n; k++ ) {
      for ( int l = 0; l < dim[k]; l++ ) {
        sv[i][k][l] *= normi;
      }
    }
  }
  return( 0 );
}  /* end of spin_vector_asym_list_normalize */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * apply the projector to a basis vector
 *
 * app should be an arry of p->rtarget->dim = number of target
 * irrep rows applicators
 ***********************************************************/
little_group_projector_applicator_type ** little_group_projector_apply ( little_group_projector_type *p , FILE*ofs ) {

  int exitstatus;
  char name[20];

  little_group_projector_applicator_type ** app = (little_group_projector_applicator_type **) malloc ( p->rtarget->dim * sizeof( little_group_projector_applicator_type * ) );
  if ( app == NULL ) {
    fprintf ( stderr, "[little_group_projector_apply] Error from malloc %s %d\n", __FILE__, __LINE__ );
    return( NULL );
  }

  /***********************************************************
   * allocate spin vectors, to which spin rotations are applied
   ***********************************************************/
  int * spin_dimensions = init_1level_itable ( p->n );
  if ( spin_dimensions == NULL ) {
    fprintf ( stderr, "[little_group_projector_apply] Error from init_1level_itable %s %d\n", __FILE__, __LINE__ );
  }
  for ( int i = 0; i < p->n; i++ ) spin_dimensions[i] = p->rspin[i].dim;

  double _Complex ** sv0 = init_2level_ztable_asym( 1, spin_dimensions, p->n );
  if ( sv0 == NULL ) {
    fprintf ( stderr, "[little_group_projector_apply] Error from init_2level_ztable_asym %s %d\n", __FILE__, __LINE__);
    return( NULL );
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * initialize spin vectors according to ref_row_spin
   ***********************************************************/
  for ( int i = 0; i < p->n; i++ ) { 
    if ( p->ref_row_spin[i] >= 0 && p->ref_row_spin[i] < p->rspin[i].dim ) {
      /***********************************************************
       * basis vector no. ref_row_spin
       ***********************************************************/
      sv0[i][p->ref_row_spin[i]] = 1.; 

    } else if ( p->ref_row_spin[i] == -1 ) {
      /***********************************************************
       * random vector
       ***********************************************************/
      // non-zero real and imag
      ranlxd ( (double*)(sv0[i]), 2*p->rspin[i].dim );
/*
      // non-zero real, zero imag
      for ( int k = 0; k < p->rspin[i].dim; k++ ) {
        double dtmp;
        ranlxd ( &dtmp, 1);
        sv0[i][k] = dtmp;
      }
*/
    }
  }  // end of loop on interpolators

  // print inital vector v0 for each interpolator
  if ( ofs != NULL ) spin_vector_asym_printf ( sv0, p->n, spin_dimensions, "v0",  ofs );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set of subduction matrices, one for each row of the
   * target irrep
   ***********************************************************/
  rot_mat_table_type RR;
  if ( p->n == 1 ) {
    init_rot_mat_table ( &RR );
    exitstatus = alloc_rot_mat_table ( &RR, "NA", "NA", p->rspin[0].dim, p->rtarget->dim );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[little_group_projector_apply] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    }
  }  // end of if p->n == 1

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * loop on rows of target irrep
   ***********************************************************/
  for ( int row_target = 0; row_target < p->rtarget->dim; row_target++ ) {

    if ( ofs != NULL ) fprintf ( ofs, "# [little_group_projector_apply] lg %s irrep %s row %d\n", p->rtarget->group, p->rtarget->irrep, row_target );

    /***********************************************************
     * allocate fields
     ***********************************************************/
    double _Complex **** sv1 = init_4level_ztable_asym ( 2, p->rtarget->n, p->n, spin_dimensions );
    if ( sv1 == NULL ) {
      fprintf ( stderr, "[little_group_projector_apply] Error from init_4level_ztable_asym for %s %d\n", __FILE__, __LINE__ );
      return ( NULL );
    }

    int **** prot = init_4level_itable ( 2, p->rtarget->n, p->n, 3 );
    if ( prot == NULL ) {
      fprintf ( stderr, "[little_group_projector_apply] Error from init_4level_itable %s %d\n", __FILE__, __LINE__ );
      return ( NULL );
    }

    double _Complex ** z_irrep_matrix_coeff = init_2level_ztable ( 2, (size_t)(p->rtarget->n) );
    if ( z_irrep_matrix_coeff == NULL ) {
      fprintf ( stderr, "[little_group_projector_apply] Error from init_1level_ztable %s %d\n", __FILE__, __LINE__ );
      return ( NULL );
    }

    /***********************************************************
     * intialize the applicator
     * using sv1, prot and z_irrep_matrix_coeff above
     ***********************************************************/
    app[row_target] = init_little_group_projector_applicator ();
    app[row_target]->rotation_n = p->rtarget->n;
    app[row_target]->interpolator_n   = p->n;
    app[row_target]->interpolator_dim = init_1level_itable ( p->n );
    memcpy ( app[row_target]->interpolator_dim, spin_dimensions, p->n * sizeof(int) );
    app[row_target]->P[0] = p->P[0];
    app[row_target]->P[1] = p->P[1];
    app[row_target]->P[2] = p->P[2];
    app[row_target]->prot = prot;
    app[row_target]->v = sv1;
    app[row_target]->c = z_irrep_matrix_coeff;
    strcpy ( app[row_target]->name, p->name );

 
    /***********************************************************
     * if only 1 interpolator field, sum up to projected
     * vector and calculate projection matrices
     ***********************************************************/
    double _Complex * Rsv = NULL, * IRsv = NULL, **R = NULL, **IR = NULL;
    if ( p-> n == 1 ) {
      Rsv  = init_1level_ztable ( (size_t)(spin_dimensions[0]) );
      IRsv = init_1level_ztable ( (size_t)(spin_dimensions[0]) );
      R    = rot_init_rotation_matrix ( p->rspin[0].dim );
      IR   = rot_init_rotation_matrix ( p->rspin[0].dim );
      if ( Rsv == NULL || IRsv == NULL || R == NULL || IR == NULL ) {
        fprintf ( stderr, "[little_group_projector_apply] Error from init_level_ztable_asym %s %d\n", __FILE__, __LINE__ );
        return ( NULL );
      }
    }  // end of if 1 interpolator

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * set projector norm
     ***********************************************************/
    double const projector_norm = (double)p->rtarget->dim / (double)p->rtarget->n * 0.5;

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * loop on rotation group elements R
     ***********************************************************/
    for ( int irot = 0; irot < p->rtarget->n; irot++ ) {

      int rid = p->rtarget->rid[irot];
      if ( g_verbose > 2 ) fprintf ( stdout, "# [little_group_projector_apply] lg %s irrep %s irot %2d rid %2d\n", p->rtarget->group, p->rtarget->irrep, irot, rid );

      // This is choice according to my notes
      double _Complex ztmp = conj ( p->rtarget->R[irot][row_target][p->ref_row_target] ) * projector_norm;

      // double _Complex ztmp = conj ( p->rtarget->R[irot][p->ref_row_target][row_target] );

      // This is the standard choice according to paper
      // double _Complex ztmp =  p->rtarget->R[irot][row_target][p->ref_row_target];

      // double _Complex ztmp =  p->rtarget->R[irot][p->ref_row_target][row_target];

      // set the complex multiplicative coefficient for this rotation
      z_irrep_matrix_coeff[0][ irot ] = ztmp;
      if ( g_verbose > 4 ) {
        fprintf(stdout, "# [little_group_projector_apply] T Gamma (R) coeff rot %2d = %25.16e %25.16e\n", rid, creal(z_irrep_matrix_coeff[0][irot]), cimag(z_irrep_matrix_coeff[0][irot]) );
      }

      /***********************************************************
       * include the name of the rotation
       ***********************************************************/
#if defined CUBIC_GROUP_DOUBLE_COVER
      strcpy ( app[row_target]->rotation_name[0][irot], cubic_group_double_cover_rotations[rid].name );
#elif defined CUBIC_GROUP_SINGLE_COVER
      strcpy ( app[row_target]->rotation_name[0][irot], cubic_group_rotations_v2[rid].name );
#endif

      /***********************************************************
       * loop on interpolators
       ***********************************************************/
      for ( int k = 0; k < p->n; k++ ) {

        // We decompose the following line
        // rot_vec_accum_vec_ti_co_pl_mat_ti_vec_ti_co ( Rsv[k], p->rspin[k].R[irot], sv0[k], ztmp, 1., spin_dimensions[k] );

        // into those two:
        // sv1 = R sv0 for interpolator k using rotation no. irot
        rot_mat_ti_vec ( sv1[0][irot][k], p->rspin[k].R[irot], sv0[k], spin_dimensions[k] );

        if ( p->n == 1 ) {
          // Rsv += sv1[irot], accumulate R sv0 in Rsv
          rot_vec_pl_eq_rot_vec_ti_co ( Rsv, sv1[0][irot][0], ztmp, spin_dimensions[0] );
        }    

        // DO NOT TRANSPOSE, this is wrong
        // DO NOT USE THE FOLLOWING
        // rot_vec_accum_vec_ti_co_pl_mat_transpose_ti_vec_ti_co ( Rsv[k], p->rspin[k].R[irot], sv0[k], ztmp, 1., spin_dimensions[k] );


        // add rotated inital momentum vector for interpolator no. k to list
        rot_point ( prot[0][irot][k], p->p[k], p->rp->R[irot] );


      }  // end of loop on interpolators

      /***********************************************************
       * add to projection matrix if only single interpolator
       ***********************************************************/
      if ( p->n == 1 ) {
        rot_mat_pl_eq_mat_ti_co ( R, p->rspin[0].R[irot], z_irrep_matrix_coeff[0][irot], p->rspin[0].dim );
      }

    }  // end of loop on rotations R

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * normalize and show Rsv;
     * show R
     ***********************************************************/
    if ( p->n == 1 ) {
      // print Rsv
      sprintf ( name, "Rsv[[%d]]", row_target+1 );
      if ( ofs != NULL ) rot_vec_printf ( Rsv, spin_dimensions[0], name, ofs );

      // print current subduction matrix
      sprintf ( name, "Rsub[[%d]]", row_target+1 );
      if ( ofs != NULL ) rot_printf_matrix ( R, p->rspin[0].dim, name, ofs );

      // RR <- R
      rot_mat_assign ( RR.R[row_target], R, p->rspin[0].dim);

      // deallocate R
      rot_fini_rotation_matrix ( &R );

    }  // end of if p->n == 1

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * not center of mass frame, include IR rotations
     ***********************************************************/

    if ( g_verbose > 2 )  fprintf( stdout, "# [little_group_projector_apply] including IR rotations\n");

    /***********************************************************
     * loop on rotation group elements IR
     ***********************************************************/
    for ( int irot = 0; irot < p->rtarget->n; irot++ ) {

      int rmid = p->rtarget->rmid[irot];
      if ( g_verbose > 2 ) fprintf ( stdout, "# [little_group_projector_apply] lg %s irrep %s irot %2d rmid %2d\n", p->rtarget->group, p->rtarget->irrep, irot, rmid );

      // This is choice according to my notes
      double _Complex ztmp = conj ( p->rtarget->IR[irot][row_target][p->ref_row_target] ) * projector_norm;

      // This is the standard choice according to paper
      // double _Complex ztmp =  p->rtarget->IR[irot][row_target][p->ref_row_target];

      // set and show the complex multiplicative coefficient for this rotation
      z_irrep_matrix_coeff[1][irot] = ztmp;
      if ( g_verbose > 4 ) {
        fprintf(stdout, "# [little_group_projector_apply] T Gamma (IR) coeff rot %2d = %25.16e %25.16e\n", rmid, creal(z_irrep_matrix_coeff[1][irot]), cimag(z_irrep_matrix_coeff[1][irot]) );
      }

      /***********************************************************
       * include the name of the rotation
       ***********************************************************/
#if defined CUBIC_GROUP_DOUBLE_COVER
      sprintf ( app[row_target]->rotation_name[1][irot], "I %s", cubic_group_double_cover_rotations[rmid].name );
#elif defined CUBIC_GROUP_SINGLE_COVER
      sprintf ( app[row_target]->rotation_name[1][irot], "I %s", cubic_group_rotations_v2[rmid].name );
#endif

      /***********************************************************
       * loop on interpolators
       ***********************************************************/
      for ( int k = 0; k < p->n; k++ ) {

        // We decompose the following line
        // rot_vec_accum_vec_ti_co_pl_mat_ti_vec_ti_co ( IRsv[k], p->rspin[k].IR[irot], sv0[k], p->parity[k] * ztmp, 1., spin_dimensions[k] );

        // into those two:
        // sv1 = R sv0 for interpolator k using rotation no. irot
        rot_mat_ti_vec ( sv1[1][irot][k], p->rspin[k].IR[irot], sv0[k], spin_dimensions[k] );

        if ( p->n == 1 ) {
          // Rsv += sv1[irot], accumulate R sv0 in Rsv
          rot_vec_pl_eq_rot_vec_ti_co ( IRsv, sv1[1][irot][0], p->parity[0] * ztmp, spin_dimensions[0] );
        }

        // multiply irrep matrix coefficient with intrinsic parity factor for interpolator no. k
        z_irrep_matrix_coeff[1][irot] *= p->parity[k];

        // add rotated inital momentum vector for interpolator no. k to list
        rot_point ( prot[1][irot][k], p->p[k], p->rp->IR[irot] );

      }  // end of loop on interpolators

      // show the complex multiplicative coefficient for this rotation including instrinsic parity
      if ( g_verbose > 4 ) {
        fprintf(stdout, "# [little_group_projector_apply] T Gamma (IR) coeff incl. parity rot %2d = %25.16e %25.16e\n",
            rmid, creal(z_irrep_matrix_coeff[1][irot]), cimag(z_irrep_matrix_coeff[1][irot]) );
      }

      if ( p->n == 1 ) {
        /***********************************************************
         * add IR to projection matrix if only single interpolator
         ***********************************************************/
        rot_mat_pl_eq_mat_ti_co ( IR, p->rspin[0].IR[irot], z_irrep_matrix_coeff[1][irot], p->rspin[0].dim );
      }

    }  // end of loop on rotations IR

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * combine Rsv and IRsv
     ***********************************************************/
    if ( p->n == 1 ) {

      // show IRsv
      sprintf ( name, "IRsv[[%d]]", row_target+1 );
      if ( ofs != NULL ) rot_vec_printf ( IRsv, spin_dimensions[0], name, ofs );
 
      // Rsv = Rsv + IRsv
      rot_vec_pl_eq_rot_vec_ti_co ( Rsv, IRsv, 1., spin_dimensions[0] );

      // normalize Rsv
      rot_vec_normalize ( Rsv, spin_dimensions[0] );

      // show Rsv
      sprintf ( name, "vsub[[%d]]", row_target+1 );
      if ( ofs != NULL ) rot_vec_printf ( Rsv, spin_dimensions[0], name, ofs );

      /***********************************************************/
      /***********************************************************/

      // print current IR
      sprintf ( name, "IRsub[[%d]]", row_target+1 );
      if ( ofs != NULL ) rot_printf_matrix ( IR, p->rspin[0].dim, name, ofs );

      // RR.R = RR.R + IR
      rot_mat_pl_eq_mat_ti_co ( RR.R[row_target], IR, 1.0,  p->rspin[0].dim );

      // normalize RR.R *= target dim / ( # rotations + # rotations-reflections )
      rot_mat_ti_eq_re ( RR.R[row_target], p->rtarget->dim /( 2. * p->rtarget->n ), RR.dim );

      // show RR.R
      sprintf ( name, "RRsub[[%d]]", row_target+1 );
      if ( ofs != NULL ) rot_printf_matrix ( RR.R[row_target], p->rspin[0].dim, name, ofs );

      // deallocate IR
      rot_fini_rotation_matrix ( &IR );

    }  // end of if p->n == 1

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * now print the complete projector information
     * for this row of the target representation
     ***********************************************************/

    show_little_group_projector_applicator ( app[row_target], ofs );

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * don't deallocate those, they're used in app[row_target]
     ***********************************************************/
    // fini_2level_ztable ( &z_irrep_matrix_coeff );
    // fini_4level_ztable_asym( &sv1 );
    // fini_4level_itable ( &prot );

    fini_1level_ztable ( &Rsv );
    fini_1level_ztable ( &IRsv );

  }  // end of loop on row_target

  /***********************************************************/
  /***********************************************************/

  if ( p->n == 1 ) {

    /***********************************************************
     * check rotation property of RR
     ***********************************************************/
    exitstatus = rot_mat_table_rotate_multiplett ( &RR, &(p->rspin[0]), p->rtarget, p->parity[0], ofs);
    if ( exitstatus != 0 ) {
      fprintf( stderr, "[little_group_projector_apply] Error from rot_mat_table_rotate_multiplett, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return( NULL );
    }

    // deallocate RR
    fini_rot_mat_table ( &RR );

  }  // end of if p->n == 1

  /***********************************************************/
  /***********************************************************/

  if ( p->n == 1 ) {
    /***********************************************************
     * calculate multiplicity or spin representation
     * in target representation
     ***********************************************************/
    fprintf ( stdout, "# [little_group_projector_apply] multiplicity of %s in %s is %d\n",
        p->rtarget->irrep, p->rspin[0].irrep, irrep_multiplicity ( p->rtarget, &p->rspin[0], p->parity[0] ) );
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * deallocate sv1
   ***********************************************************/
  fini_1level_itable ( &spin_dimensions );
  fini_2level_ztable_asym( &sv0 );
  return( app );

}  // end of little_group_projector_apply

/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
#if 0
int little_group_projector_key (char*key,  little_group_projector_type *p_snk, little_group_projector_type *p_src, int irot_snk, int irot_src ) {

  if ( irot_snk < 0 || irot_snk >= p_snk->rtarget->n ) {
    fprintf ( stderr, "[little_group_projection_key] Error, irot snk outside allowed range\n" );
    return(1);
  }

  if ( irot_src < 0 || irot_src >= p_src->rtarget->n ) {
    fprintf ( stderr, "[little_group_projection_key] Error, irot src outside allowed range\n" );
    return(1);
  }

  if ( key == NULL ) {
    fprintf ( stderr, "[little_group_projection_key] Error, key is NULL\n" );
    return(2);
  }

  return(0);
}  /* end of little_group_projector_key */
#endif

/***********************************************************/
/***********************************************************/

/***********************************************************
 * rotate a projection matrix multiplett
 ***********************************************************/

int rot_mat_table_rotate_multiplett ( rot_mat_table_type * const rtab, rot_mat_table_type * const rapply, rot_mat_table_type * const rtarget, int const parity, FILE*ofs ) {

  if ( ofs == NULL ) ofs = stdout;

  if ( rtab->dim != rapply->dim ) {
    fprintf(stderr, "[rot_mat_table_rotate_multiplett] Error, incompatible dimensions\n");
    return(1);
  }

  if ( rtab->n != rtarget->dim ) {
    fprintf(stderr, "[rot_mat_table_rotate_multiplett] Error, incompatible number of rotations in rtab and matrix dimension in rtarget\n");
    return(2);
  }
  
  /***********************************************************/
  /***********************************************************/
  fprintf ( ofs, "# [rot_mat_table_rotate_multiplett] using rapply %s / %s for rtarget %s / %s\n", rapply->group, rapply->irrep, rtarget->group, rtarget->irrep );
  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * loop on elements in rtab
   ***********************************************************/
  for ( int ia = 0; ia < rtab->n; ia++ ) {

    /***********************************************************
     * loop on rotations = elements of rtarget
     *
     * R rotations
     ***********************************************************/
    for ( int irot = 0; irot < rtarget->n; irot++ ) {

      double _Complex **R2 = rot_init_rotation_matrix ( rtab->dim );
      double _Complex **R3 = rot_init_rotation_matrix ( rtab->dim );

      rot_mat_ti_mat (R2, rapply->R[irot], rtab->R[ia], rtab->dim );

      for ( int k = 0; k < rtarget->dim; k++ ) {
        rot_mat_pl_eq_mat_ti_co ( R3, rtab->R[k], rtarget->R[irot][k][ia], rtab->dim );
      }

      char name[100];
      sprintf( name, "R2[[%2d]]", rtarget->rid[irot] );
      rot_printf_matrix ( R2, rtab->dim, name, ofs );
      fprintf(ofs, "\n");
      sprintf( name, "R3[[%2d]]", rtarget->rid[irot] );
      rot_printf_matrix ( R3, rtab->dim, name, ofs );
      double norm = rot_mat_norm_diff ( R2, R3, rtab->dim );
      double norm2 = sqrt( rot_mat_norm2 ( R2, rtab->dim ) );

      fprintf(ofs, "# [rot_mat_table_rotate_multiplett]  R[[%2d]] rid  %2d norm diff = %16.7e / %16.7e\n\n", irot+1,
          rtarget->rid[irot], norm, norm2 );

      rot_fini_rotation_matrix ( &R2 );
      rot_fini_rotation_matrix ( &R3 );

    }  // end of loop on rotations

    /***********************************************************
     * IR rotations
     ***********************************************************/
    for ( int irot = 0; irot < rtarget->n; irot++ ) {

      double _Complex **R2 = rot_init_rotation_matrix ( rtab->dim );
      double _Complex **R3 = rot_init_rotation_matrix ( rtab->dim );

      rot_mat_ti_mat (R2, rapply->IR[irot], rtab->R[ia], rtab->dim );
      // multiply with pairty
      rot_mat_ti_eq_re ( R2, parity, rtab->dim );


      for ( int k = 0; k < rtarget->dim; k++ ) {
        rot_mat_pl_eq_mat_ti_co ( R3, rtab->R[k], rtarget->IR[irot][k][ia], rtab->dim );
      }

      char name[100];
      sprintf( name, "IR2[[%2d]]", rtarget->rmid[irot] );
      rot_printf_matrix ( R2, rtab->dim, name, ofs );
      fprintf(ofs, "\n");
      sprintf( name, "IR3[[%2d]]", rtarget->rmid[irot] );
      rot_printf_matrix ( R3, rtab->dim, name, ofs );
      double norm = rot_mat_norm_diff ( R2, R3, rtab->dim );
      double norm2 = sqrt( rot_mat_norm2 ( R2, rtab->dim ) );
      fprintf(ofs, "# [rot_mat_table_rotate_multiplett] IR[[%2d]] rmid %2d norm diff = %16.7e / %16.7e\n\n", irot+1,
          rtarget->rmid[irot], norm, norm2 );

      rot_fini_rotation_matrix ( &R2 );
      rot_fini_rotation_matrix ( &R3 );
    }  // end of loop on rotation-reflections

  }  // end of loop on elements of rtab

  return(0);
}  /* end of loop on rot_mat_table_rotate_multiplett */


/***********************************************************/
/***********************************************************/

/***********************************************************
 * calculate multiplicity of occurence of irrep given by
 * rirrep in spin irrep given by rspin
 ***********************************************************/
int irrep_multiplicity (rot_mat_table_type * const rirrep, rot_mat_table_type * const rspin, int const parity ) {

  double _Complex s = 0.;
  int nelem = rirrep->n;
  // loop on rotations
  for ( int irot = 0; irot < rirrep->n; irot++ ) {
    s +=          rot_mat_trace ( rirrep->R[irot],  rirrep->dim ) * rot_mat_trace ( rspin->R[irot],  rspin->dim );
  }

  // loop on rotations-reflections
  for ( int irot = 0; irot < rirrep->n; irot++ ) {
    s += parity * rot_mat_trace ( rirrep->IR[irot],  rirrep->dim ) * rot_mat_trace ( rspin->IR[irot],  rspin->dim );
  }
  nelem *= 2;

  // TEST
  if ( g_verbose > 2 ) fprintf(stdout, "# [irrep_multiplicity] s = %25.16e %25.16e g = %2d\n", creal(s), cimag(s), rirrep->n );

  return( (int)( round( creal(s) ) / nelem ) );
}  // end of irrep multiplicty

/***********************************************************/
/***********************************************************/

void product_vector_printf ( double _Complex *v, int*dim, int n, char*name, FILE*ofs ) {

  const double eps = 9.e-15;
  int pdim = 1;
  for ( int i = 0; i < n; i++ ) pdim*=dim[i];
  int * coords = init_1level_itable ( n );

  fprintf( ofs, "# [product_vector_printf] %s\n", name);
  fprintf( ofs, "   %s <- array( dim=c( %d", name, dim[0]);
  for ( int i = 1; i < n; i++ ) fprintf( ofs, ", %d ", dim[i] );
  fprintf( ofs, ") )\n");

  for ( int idx = 0; idx < pdim; idx++ ) {
    product_vector_index2coords ( idx, coords, dim, n );
    fprintf( ofs, "   %s[ %d", name, coords[0]+1);
    for ( int i = 1; i < n; i++ ) fprintf( ofs, ", %2d ", coords[i]+1 );
    fprintf( ofs, "] <- %25.16e + %25.16e*1.i\n", dgeps( creal(v[idx]), eps), dgeps( cimag(v[idx]),eps ) );
  }

  fini_1level_itable ( &coords );
  return;
}  /* end of function product_vector_printf */


/***********************************************************/
/***********************************************************/

void product_vector_project_accum ( double _Complex *v, rot_mat_table_type*r, int rid, int rmid, double _Complex *v0, double _Complex c1, double _Complex c2, int *dim , int n ) {
  
  int pdim =1;

  for ( int i = 0; i < n; i++ ) pdim *= dim[i];

  int ** coords = init_2level_itable ( pdim, n );
  for ( int i=0; i < pdim; i++ ) product_vector_index2coords ( i, coords[i], dim, n );

  for ( int idx = 0; idx < pdim; idx++ ) {

    double _Complex res = 0.;

    for ( int kdx = 0; kdx < pdim; kdx++ ) {
      double _Complex a = 1.;
      if ( rid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].R[rid][coords[idx][l]][coords[kdx][l]];
      } else if ( rmid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].IR[rmid][coords[idx][l]][coords[kdx][l]];
      } else { a = 0.; }

      res += a * v0[kdx];
    }
    v[idx] = c2 * v[idx] + c1 * res;
  }

  fini_2level_itable ( &coords );
  return;
}  /* end of product_vector_project_accum */


/***********************************************************/
/***********************************************************/

void product_mat_pl_eq_mat_ti_co ( double _Complex **R, rot_mat_table_type *r, int rid, int rmid, double _Complex c, int*dim, int n ) {
  
  int pdim =1;

  for ( int i = 0; i < n; i++ ) pdim *= dim[i];

  int ** coords = init_2level_itable ( pdim, n );
  for ( int i=0; i < pdim; i++ ) product_vector_index2coords ( i, coords[i], dim, n );

  for ( int idx  = 0; idx < pdim; idx++ ) {
    for ( int kdx  = 0; kdx < pdim; kdx++ ) {
      double _Complex a = 1.;
      if ( rid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].R[rid][coords[idx][l]][coords[kdx][l]];
        /* fprintf ( stdout, "# [product_mat_pl_eq_mat_ti_co] idx %3d kdx %3d a %25.16e %25.16e\n", idx, kdx, creal(a), cimag(a) ); */
      } else if ( rmid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].IR[rmid][coords[idx][l]][coords[kdx][l]];
      } else { a = 0.;}
      R[idx][kdx] += c * a;
    }
  }
  
  fini_2level_itable ( &coords );
  return;
}  /* end of product_mat_pl_eq_mat_ti_co */

/***********************************************************/
/***********************************************************/

void rot_mat_table_eq_product_mat_table ( rot_mat_table_type *r, rot_mat_table_type *s, int n ) {
  
  int dim[n];
  int pdim =1;

  for ( int i = 0; i < n; i++ ) {
    dim[i] = s[i].dim;
    pdim *= dim[i];

  }
  int nrot = s[0].n;

  init_rot_mat_table ( r );

  alloc_rot_mat_table ( r, "NA", "NA", pdim, nrot );

  int ** coords = init_2level_itable ( pdim, n );
  for ( int i=0; i < pdim; i++ ) product_vector_index2coords ( i, coords[i], dim, n );

  for ( int irot = 0; irot < nrot; irot++ ) {

    for ( int idx  = 0; idx < pdim; idx++ ) {
      for ( int kdx  = 0; kdx < pdim; kdx++ ) {
        double _Complex a = 1.;
        for ( int l = 0; l < n; l++ ) a *= s[l].R[irot][coords[idx][l]][coords[kdx][l]];
        r->R[irot][idx][kdx] = a;
      }
    }

    for ( int idx  = 0; idx < pdim; idx++ ) {
      for ( int kdx  = 0; kdx < pdim; kdx++ ) {
        double _Complex a = 1.;
        for ( int l = 0; l < n; l++ ) a *= s[l].IR[irot][coords[idx][l]][coords[kdx][l]];
        r->IR[irot][idx][kdx] = a;
      }
    }
  }
  
  fini_2level_itable ( &coords );
  return;
}  /* end of product_mat_pl_eq_mat_ti_co */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * mixed product
 * R is rotation matrix, dim = pdim
 * r is list of spin matrices, dim = dim[i]
 * s is rotation matrix, dim = pdim
 ***********************************************************/
void rot_mat_eq_product_mat_ti_rot_mat ( double _Complex **R, rot_mat_table_type *r, int rid, int rmid, double _Complex **S, int n ) {

  int pdim =1;

  int * dim = init_1level_itable ( n );
  for ( int i = 0; i < n; i++ ) {
    dim[i] = r[i].dim;
    pdim *= r[i].dim;
  }

  int ** coords = init_2level_itable ( pdim, n );
  for ( int i=0; i < pdim; i++ ) product_vector_index2coords ( i, coords[i], dim, n );
  memset ( R[0], 0, pdim*pdim*sizeof(double _Complex) );

  for ( int idx  = 0; idx < pdim; idx++ ) {
    for ( int kdx  = 0; kdx < pdim; kdx++ ) {
      double _Complex a = 1.;
      if ( rid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].R[rid][coords[idx][l]][coords[kdx][l]];
      } else if ( rmid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].IR[rmid][coords[idx][l]][coords[kdx][l]];
      } else { a = 0.;}

      for ( int ldx  = 0; ldx < pdim; ldx++ ) {
        R[idx][ldx] += a * S[kdx][ldx];
      }

    }
  }
  
  fini_2level_itable ( &coords );
  fini_1level_itable ( &dim );
  return;
}  /* end of product_mat_ti_mat */


/***********************************************************/
/***********************************************************/

/***********************************************************
 * print a direct product matrix
 ***********************************************************/
int product_mat_printf ( double _Complex **R, int *dim, int n, char *name, FILE*ofs ) {

  const double eps = 9.e-15;
  int pdim =1;
  for ( int i = 0; i < n; i++ ) pdim *= dim[i];

  int ** coords = init_2level_itable ( pdim, n );
  if ( coords == NULL ) {
    fprintf (stderr, "[product_mat_printf] Error from init_2level_itable\n");
    return(1);
  }
  for ( int i=0; i < pdim; i++ ) product_vector_index2coords ( i, coords[i], dim, n );

  fprintf ( ofs, "# [product_mat_printf] %s\n", name);
  fprintf ( ofs, "%s <- array( dim=c( %d", name, dim[0] );
  for ( int i = 1; i < n; i++ ) fprintf( ofs, ", %d ", dim[i] );
  for ( int i = 0; i < n; i++ ) fprintf( ofs, ", %d ", dim[i] );
  fprintf ( ofs, "))\n" );

  for ( int idx  = 0; idx < pdim; idx++ ) {

    for ( int kdx  = 0; kdx < pdim; kdx++ ) {

      fprintf ( ofs, "%s[%2d", name, coords[idx][0]+1 );
      for ( int i = 1; i < n; i++ ) fprintf( ofs, ", %2d ", coords[idx][i]+1 );
      for ( int i = 0; i < n; i++ ) fprintf( ofs, ", %2d ", coords[kdx][i]+1 );
      fprintf ( ofs, "] <- %25.16e + %25.16e*1.i\n", 
          dgeps ( creal( R[idx][kdx] ), eps), dgeps ( cimag( R[idx][kdx] ), eps) );
    }
  }

  fini_2level_itable ( &coords );
  return(0);
}  /* end of product_mat_printf */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * apply the projector to a product state
 ***********************************************************/
int little_group_projector_apply_product ( little_group_projector_type *p , FILE*ofs) {

  int *spin_dimensions = NULL;
  int exitstatus;
  char name[20];
  rot_mat_table_type RR;
  int frame_is_cmf = ( p->P[0] == 0 && p->P[1] == 0 && p->P[2] == 0 );
  int pdim = 1;


  /***********************************************************
   * allocate spin vectors, to which spin rotations are applied
   ***********************************************************/
  int * spin_dimension = init_1level_itable ( p->n );
  if ( spin_dimension == NULL ) {
    fprintf ( stderr, "# [little_group_projector_apply_product] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  for ( int i = 0; i < p->n; i++ ) {
    spin_dimensions[i] = p->rspin[i].dim;
    pdim *= p->rspin[i].dim;
  }
  fprintf ( stdout, "# [little_group_projector_apply_product] spinor product dimension = %d\n", pdim );

  double _Complex * sv0 = init_1level_ztable( (size_t)pdim );
  if ( sv0 == NULL ) {
    fprintf ( stderr, "# [little_group_projector_apply_product] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * initialize spin vectors according to ref_row_spin
   ***********************************************************/
  if ( ( p->n == 2  ) && ( p->ref_row_spin[0] < 0 ) && ( p->ref_row_spin[1] <= 0 ) ) {
    int J2_1 = rot_mat_table_get_spin2 ( &(p->rspin[0]) );
    int J2_2 = rot_mat_table_get_spin2 ( &(p->rspin[1]) );

    int J2_3 = -p->ref_row_spin[0];
    int M2_3 = J2_3 + 2*p->ref_row_spin[1];

    int bispinor[2] = {
      rot_mat_table_get_bispinor ( &(p->rspin[0]) ),
      rot_mat_table_get_bispinor ( &(p->rspin[1]) ) };

    /***********************************************************
     * use Clebsch-Gordan coefficients
     ***********************************************************/
    for ( int i1 = 0; i1 <= J2_1; i1++ ) {
      int M2_1 = J2_1 - 2*i1;
    for ( int i2 = 0; i2 <= J2_2; i2++ ) {
      int M2_2 = J2_2 - 2*i2;
      fprintf ( ofs, "# [little_group_projector_apply_product] J2_1 = %2d M2_1 = %2d   J2_2 = %2d M2_2 = %2d   J2_3 = %2d M2_3 = %2d  bispinor %d %d\n", J2_1, M2_1, J2_2, M2_2, J2_3, M2_3,
         bispinor[0], bispinor[1] );
      for ( int j1 = 0; j1 <= bispinor[0]; j1++ ) {
        for ( int j2 = 0; j2 <= bispinor[1]; j2++ ) {
          int coords[2] = {i1+j1*(J2_1+1), i2+j2*(J2_2+1)};
          sv0[ product_vector_coords2index ( coords, spin_dimensions, 2 ) ] = clebsch_gordan_coeff ( J2_3, M2_3, J2_1, M2_1, J2_2, M2_2 );
        }
      }
    }}
  } else {
    product_vector_set_element ( sv0, 1.0, p->ref_row_spin, spin_dimensions, p->n );
  }

  product_vector_printf ( sv0, spin_dimensions, p->n,  "v0", ofs );

  /***********************************************************
   * TEST
   ***********************************************************/
  init_rot_mat_table ( &RR );
  exitstatus = alloc_rot_mat_table ( &RR, "NA", "NA", pdim, p->rtarget->dim );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[little_group_projector_apply_product] Error from alloc_rot_mat_table, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }
  /***********************************************************
   * END OF TEST
   ***********************************************************/


  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * allocate sv1
   ***********************************************************/
  double _Complex ** sv1 = init_2level_ztable ( (size_t)( p->rtarget->dim), (size_t)pdim );
  if ( sv1 == NULL ) {
    fprintf ( stderr, "[little_group_projector_apply_product] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * loop on rows of target irrep
   ***********************************************************/
  for ( int row_target = 0; row_target < p->rtarget->dim; row_target++ ) {

    double _Complex *Rsv = init_1level_ztable ( (size_t)pdim );

    /***********************************************************
     * TEST
     ***********************************************************/
    double _Complex **R = rot_init_rotation_matrix ( pdim );
    /***********************************************************
     * END OF TEST
     ***********************************************************/

    /***********************************************************
     * loop on rotation group elements R
     ***********************************************************/
    for ( int irot = 0; irot < p->rtarget->n; irot++ ) {

      fprintf ( stdout, "# [little_group_projector_apply_product] lg %s irrep %s irot %2d rid %2d\n", p->rtarget->group, p->rtarget->irrep, irot,
          p->rtarget->rid[irot] );

      /* This is choice according to my notes */
      double _Complex z_irrep_matrix_coeff = conj ( p->rtarget->R[irot][row_target][p->ref_row_target] );


      /***********************************************************
       * add spin rotation x sv0 as product
       ***********************************************************/
      product_vector_project_accum ( Rsv, p->rspin, irot, -1, sv0, z_irrep_matrix_coeff, 1., spin_dimensions, p->n );

      /***********************************************************
       * TEST
       ***********************************************************/
      product_mat_pl_eq_mat_ti_co ( R, p->rspin, irot, -1, z_irrep_matrix_coeff, spin_dimensions, p->n );
      /***********************************************************
       * END OF TEST
       ***********************************************************/

    }  /* end of loop on rotations R */

    /***********************************************************
     * TEST
     ***********************************************************/
    sprintf ( name, "vsub[[%d]]", row_target+1 );
    product_vector_printf ( Rsv, spin_dimensions, p->n, name,  ofs  );
    /***********************************************************
     * END OF TEST
     ***********************************************************/


    /***********************************************************
     * TEST
     ***********************************************************/
    sprintf ( name, "Rsub[[%d]]", row_target+1 );
    product_mat_printf ( R, spin_dimensions, p->n, name, ofs );

    rot_mat_assign ( RR.R[row_target], R, pdim );
    rot_fini_rotation_matrix ( &R );

    /***********************************************************
     * END OF TEST
     ***********************************************************/

    /***********************************************************
     * not center of mass frame, include IR rotations
     ***********************************************************/
    // if ( !frame_is_cmf )  { 
      if ( g_verbose > 2 ) fprintf( stdout, "# [little_group_projector_apply_product] including IR rotations\n");

      double _Complex *IRsv = init_1level_ztable( (size_t)pdim );

      /***********************************************************
       * TEST
       ***********************************************************/
      double _Complex **IR = rot_init_rotation_matrix ( pdim );
      /***********************************************************
       * END OF TEST
       ***********************************************************/


      /***********************************************************
       * loop on rotation group elements IR
       ***********************************************************/
      for ( int irot = 0; irot < p->rtarget->n; irot++ ) {

        fprintf ( stdout, "# [little_group_projector_apply_product] lg %s irrep %s irot %2d rmid %2d\n",
            p->rtarget->group, p->rtarget->irrep, irot, p->rtarget->rmid[irot] );

        /* This is choice according to my notes */
        double _Complex z_irrep_matrix_coeff = conj ( p->rtarget->IR[irot][row_target][p->ref_row_target] );

        /* TEST */
        /* fprintf(stdout, "# [little_group_projector_apply_product] T Gamma (IR) coeff rot %2d = %25.16e %25.16e\n", rmid, creal(z_irrep_matrix_coeff), cimag(z_irrep_matrix_coeff) ); */

        /***********************************************************
         * add rotation-reflection applied to sv0 as product
         ***********************************************************/
        //STOPPED HERE
        //  include intrinsic parity
        product_vector_project_accum ( IRsv, p->rspin, -1, irot, sv0, z_irrep_matrix_coeff, 1., spin_dimensions, p->n );

        /***********************************************************
         * TEST
         ***********************************************************/
        product_mat_pl_eq_mat_ti_co ( IR, p->rspin, -1, irot, z_irrep_matrix_coeff, spin_dimensions, p->n );
        /***********************************************************
         * END OF TEST
         ***********************************************************/

      }  /* end of loop on rotations IR */

      /***********************************************************
       * TEST
       ***********************************************************/
      sprintf ( name, "Ivsub[[%d]]", row_target+1 );
      product_vector_printf ( IRsv, spin_dimensions, p->n, name,  ofs  );
      /***********************************************************
       * END OF TEST
       ***********************************************************/

      /***********************************************************
       * add IRsv to Rsv, normalize
       ***********************************************************/

      rot_vec_pl_eq_vec_ti_co ( Rsv, IRsv, 1.0, pdim );


      /***********************************************************
       * TEST
       ***********************************************************/
      sprintf ( name, "IRsub[[%d]]", row_target+1 );
      product_mat_printf ( IR, spin_dimensions, p->n, name, ofs );

      rot_mat_pl_eq_mat_ti_co ( RR.R[row_target], IR, 1.0,  pdim );

      rot_fini_rotation_matrix ( &IR );
      /***********************************************************
       * END OF TEST
       ***********************************************************/

      fini_1level_ztable( &IRsv );
    // }  /* end of if not center of mass frame */

    /***********************************************************
     * normalize Rsv+IRsv, show Cvsub
     ***********************************************************/
    rot_vec_normalize ( Rsv, pdim );
    sprintf ( name, "Cvsub[[%d]]", row_target+1 );
    product_vector_printf ( Rsv, spin_dimensions, p->n, name, ofs );

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * TEST
     ***********************************************************/

    rot_mat_ti_eq_re ( RR.R[row_target], p->rtarget->dim /( 2. * p->rtarget->n ), pdim);
    sprintf ( name, "RRsub[[%d]]", row_target+1 );
    product_mat_printf ( RR.R[row_target], spin_dimensions, p->n, name, ofs );


    /***********************************************************
     * END OF TEST
     ***********************************************************/

    fini_1level_ztable( &Rsv );

  }  /* end of loop on row_target */

  /***********************************************************/
  /***********************************************************/


  /***********************************************************
   * check rotation properties of RR, deallocate RR
   ***********************************************************/
  exitstatus = rot_mat_table_rotate_multiplett_product ( &RR, p->rspin, p->rtarget, p->n, !frame_is_cmf, ofs );
  if ( exitstatus != 0 ) {
    fprintf( stderr, "[little_group_projector_apply_product] Error from rot_mat_table_rotate_multiplett_product, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  fini_rot_mat_table ( &RR );


  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * deallocate sv1
   ***********************************************************/
  fini_2level_ztable ( &sv1 );

  fini_1level_itable ( &spin_dimensions );
  fini_1level_ztable( &sv0 );
  return(0);

}  /* end of little_group_projector_apply_product */

/***********************************************************/
/***********************************************************/


/***********************************************************
 * rotate a projection matrix multiplett for product matrix
 ***********************************************************/
int rot_mat_table_rotate_multiplett_product ( 
    rot_mat_table_type *rtab,
    rot_mat_table_type *rapply,
    rot_mat_table_type *rtarget,
    int n, int with_IR, FILE*ofs
) {

  const double eps = 2.e-14;
  int pdim = 1;
  for ( int i = 0; i < n; i++ ) pdim *= rapply[i].dim;

  if ( rtab->dim != pdim ) {
    fprintf(stderr, "[rot_mat_table_rotate_multiplett_product] Error, incompatible dimensions\n");
    return(1);
  }

  if ( rtab->n != rtarget->dim ) {
    fprintf(stderr, "[rot_mat_table_rotate_multiplett_product] Error, incompatible number of rotations in rtab and matrix dimension in rtarget\n");
    return(2);
  }
  
  /***********************************************************/
  /***********************************************************/

  for ( int i = 0; i < n; i++ ) 
    fprintf ( ofs, "# [rot_mat_table_rotate_multiplett_product] using rapply(%d) %s / %s for rtarget %s / %s\n", i, rapply[i].group, rapply[i].irrep, rtarget->group, rtarget->irrep );
  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * loop on elements in rtab
   ***********************************************************/
  for ( int ia = 0; ia < rtab->n; ia++ ) {

    /***********************************************************
     * loop on rotations = elements of rtarget
     *
     * R rotations
     ***********************************************************/
    for ( int irot = 0; irot < rtarget->n; irot++ ) {

      double _Complex **R2 = rot_init_rotation_matrix ( pdim );
      double _Complex **R3 = rot_init_rotation_matrix ( pdim );

      rot_mat_eq_product_mat_ti_rot_mat ( R2, rapply, irot, -1, rtab->R[ia], n );

      for ( int k = 0; k < rtarget->dim; k++ ) {
        rot_mat_pl_eq_mat_ti_co ( R3, rtab->R[k], rtarget->R[irot][k][ia], rtab->dim );
      }

      char name[100];
      sprintf( name, "R2[[%2d]]", rtarget->rid[irot] );
      rot_printf_matrix ( R2, rtab->dim, name, ofs );
      fprintf(ofs, "\n");
      sprintf( name, "R3[[%2d]]", rtarget->rid[irot] );
      rot_printf_matrix ( R3, rtab->dim, name, ofs );
      double norm = rot_mat_norm_diff ( R2, R3, rtab->dim );
      double norm2 = sqrt( rot_mat_norm2 ( R2, rtab->dim ) );

      fprintf(ofs, "# [rot_mat_table_rotate_multiplett_product] irot %2d rid  %2d norm diff = %16.7e / %16.7e   %d\n\n",
          irot, rtarget->rid[irot], norm, norm2, fabs(norm)<eps );

      rot_fini_rotation_matrix ( &R2 );
      rot_fini_rotation_matrix ( &R3 );
    }

    /***********************************************************
     * IR rotations
     ***********************************************************/
    if ( with_IR ) {
      for ( int irot = 0; irot < rtarget->n; irot++ ) {

        double _Complex **R2 = rot_init_rotation_matrix ( rtab->dim );
        double _Complex **R3 = rot_init_rotation_matrix ( rtab->dim );

        rot_mat_eq_product_mat_ti_rot_mat ( R2, rapply, -1, irot, rtab->R[ia], n );

        for ( int k = 0; k < rtarget->dim; k++ ) {
          rot_mat_pl_eq_mat_ti_co ( R3, rtab->R[k], rtarget->IR[irot][k][ia], rtab->dim );
        }

        char name[100];
        sprintf( name, "IR2[[%2d]]", rtarget->rmid[irot] );
        rot_printf_matrix ( R2, rtab->dim, name, ofs );
        fprintf(ofs, "\n");
        sprintf( name, "IR3[[%2d]]", rtarget->rmid[irot] );
        rot_printf_matrix ( R3, rtab->dim, name, ofs );
        double norm = rot_mat_norm_diff ( R2, R3, rtab->dim );
        double norm2 = sqrt( rot_mat_norm2 ( R2, rtab->dim ) );
        fprintf(ofs, "# [rot_mat_table_rotate_multiplett_product] irot %2d rmid %2d norm diff = %16.7e / %16.7e   %d\n\n",
            irot, rtarget->rmid[irot], norm, norm2, fabs(norm)<eps );

        rot_fini_rotation_matrix ( &R2 );
        rot_fini_rotation_matrix ( &R3 );
      }
    }

  }  /* end of loop on elements of rtab */

  return(0);
}  /* end of rot_mat_table_rotate_multiplett_product */

/***********************************************************/
/***********************************************************/

/***********************************************************
 ***********************************************************/
int little_group_projector_data_key ( char *key,  little_group_projector_type const * const p, int const ielem ) {

  int pvec[p->n][3];

  for ( int i = 0; i < p->n; i++ ) { memcpy ( pvec[i], p->p[i], 3*sizeof(int) ); }
  


  return(0);

}  /* end of little_group_projector_data_key */


}  /* end of namespace cvc */
