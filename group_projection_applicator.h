/****************************************************
 * group_projection_applicator.h
 *
 * Mi 18. Apr 09:59:48 CEST 2018
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

#include "table_init_i.h"
#include "table_init_c.h"
#include "table_init_z.h"
#include "table_init_asym_z.h"

namespace cvc {

inline int compare_momentum_configuration ( int ** const p, int ** const q, int const n );
inline int compare_spinor_configuration ( double _Complex ** const v, double _Complex ** const w, int const n, int * const dim );
inline int compare_spinor_pm_configuration ( double _Complex ** const v, double _Complex ** const w, int const n, int * const dim );

double const eps = 9.e-15;

typedef struct {
  int rotation_n;
  int interpolator_n;
  int *interpolator_dim;
  int P[3];
  int **** prot;
  double _Complex ****v;
  double _Complex **c;
  char name[200];
  char rotation_name[2][48][20];
} little_group_projector_applicator_type;

/***********************************************************
 * abs max
 ***********************************************************/
inline double __dgeps (double const a, double const eps ) {
  double t = fabs ( a );
  return( t > eps ? a : 0. );
}  // end of __dgeps



/***********************************************************/
/***********************************************************/

inline  little_group_projector_applicator_type * init_little_group_projector_applicator ( void ) {
  
  little_group_projector_applicator_type *a = (little_group_projector_applicator_type *) malloc ( sizeof( little_group_projector_applicator_type ) );
  if ( a == NULL ) return( NULL );

  a->rotation_n   = 0;
  a->interpolator_n   = 0;
  a->interpolator_dim = NULL;
  a->P[0] = 0;
  a->P[1] = 0;
  a->P[2] = 0;
  a->prot = NULL;
  a->v = NULL;
  a->c = NULL;
  strcpy ( a->name, "NA" );
  return(a);
}

/***********************************************************/
/***********************************************************/

inline little_group_projector_applicator_type * fini_little_group_projector_applicator ( little_group_projector_applicator_type*a) {
  a->rotation_n = 0;
  a->interpolator_n = 0;
  fini_1level_itable ( &(a->interpolator_dim) );
  a->P[0] = 0;
  a->P[1] = 0;
  a->P[2] = 0;
  fini_4level_itable ( &(a->prot) );
  fini_4level_ztable_asym ( &(a->v) );
  fini_2level_ztable ( &(a->c) );
  strcpy ( a->name, "NA" );
  return( a );
}

/***********************************************************/
/***********************************************************/

/***********************************************************
 * a <- b
 ***********************************************************/
inline little_group_projector_applicator_type * copy_little_group_projector_applicator ( little_group_projector_applicator_type * b ) {

  little_group_projector_applicator_type *a = ( little_group_projector_applicator_type *) malloc ( sizeof (little_group_projector_applicator_type) );
  if ( a == NULL ) {
    fprintf ( stderr, "[copy_little_group_projector_applicator] Error from malloc %s %d\n", __FILE__, __LINE__ );
    return(NULL);
  }

  a->rotation_n = b->rotation_n;
  a->interpolator_n = b->interpolator_n;
  a->interpolator_dim = init_1level_itable ( a->interpolator_n );
  if ( a->interpolator_dim == NULL ) {
    fprintf ( stderr, "[copy_little_group_projector_applicator] Error from init_1level_itable %s %d\n", __FILE__, __LINE__ );
    return(NULL);
  }

  a->P[0] = b->P[0];
  a->P[1] = b->P[1];
  a->P[2] = b->P[2];

  a->prot = init_4level_itable ( 2, a->rotation_n, a->interpolator_n, 3 );
  if ( a->prot == NULL ) {
    fprintf ( stderr, "[copy_little_group_projector_applicator] Error from init_4level_itable %s %d\n", __FILE__, __LINE__ );
    return(NULL);
  }

  a->c = init_2level_ztable ( 2, a->rotation_n );
  if ( a->c == NULL ) {
    fprintf ( stderr, "[copy_little_group_projector_applicator] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
    return(NULL);
  }

  a->v = init_4level_ztable_asym ( 2, a->rotation_n, a->interpolator_n, a->interpolator_dim );
  if ( a->c == NULL ) {
    fprintf ( stderr, "[copy_little_group_projector_applicator] Error from init_4level_ztable_asym %s %d\n", __FILE__, __LINE__ );
    return(NULL);
  }

  strcpy ( a->name, b->name );

  return( a );
}  // end of copy_little_group_projector_applicator


/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
inline little_group_projector_applicator_type * reduce_little_group_projector_applicator ( little_group_projector_applicator_type * const a, int const keep_direction ) {

  /***********************************************************
   * set coefficients to zero if they are below eps in absolute
   * value
   ***********************************************************/
  for ( int i = 0; i < 2*a->rotation_n; i++ ) {
    if ( cabs ( a->c[0][i] ) < eps ) a->c[0][i] = 0.;
  }

#if 0
  // loop on parity parts
  for ( int iparity = 0; iparity < 2; iparity++ ) {

    // loop on rotations
    for ( int i = 0; i < a->rotation_n-1; i++ ) {
      if ( a->c[iparity][i] == 0 ) continue;
      for ( int k = i+1; k < a->rotation_n; k++ ) {
        if ( a->c[iparity][k] == 0. ) continue;
        int const mom_config_is_equal = compare_momentum_configuration ( a->prot[iparity][i], a->prot[iparity][k], a->interpolator_n );
        if ( mom_config_is_equal == 0 ) continue;


        int const spinor_config_is_equal = keep_direction ? \
                                           compare_spinor_configuration ( a->v[iparity][i], a->v[iparity][k], a->interpolator_n, a->interpolator_dim ) : \
                                           compare_spinor_pm_configuration ( a->v[iparity][i], a->v[iparity][k], a->interpolator_n, a->interpolator_dim );

        if ( spinor_config_is_equal ) {
          a->c[iparity][i] += a->c[iparity][k] * spinor_config_is_equal;
          a->c[iparity][k] = 0.;
        }
      }  // end of loop on to-compare-to rotations
    }  // end of loop on rotations
  }  // end loop on parity parts
#endif

  // loop on rotations
  for ( int i = 0; i < 2*a->rotation_n-1; i++ ) {
    if ( a->c[0][i] == 0 ) continue;
      for ( int k = i+1; k < 2*a->rotation_n; k++ ) {
        if ( a->c[0][k] == 0. ) continue;
        int const mom_config_is_equal = compare_momentum_configuration ( a->prot[0][i], a->prot[0][k], a->interpolator_n );
        if ( mom_config_is_equal == 0 ) continue;


        int const spinor_config_is_equal = keep_direction ? \
                                           compare_spinor_configuration ( a->v[0][i], a->v[0][k], a->interpolator_n, a->interpolator_dim ) : \
                                           compare_spinor_pm_configuration ( a->v[0][i], a->v[0][k], a->interpolator_n, a->interpolator_dim );

        if ( spinor_config_is_equal ) {
          a->c[0][i] += a->c[0][k] * spinor_config_is_equal;
          a->c[0][k] = 0.;
        }
      }  // end of loop on to-compare-to rotations
    }  // end of loop on rotations

  return( a );
}  // end of reduce_little_group_projector_applicator

/***********************************************************/
/***********************************************************/

inline int compare_momentum_configuration ( int ** const p, int ** const q, int const n ) {
  int isequal = 1;

  for ( int i = 0; i < n; i++ ) {
    isequal &= ( p[i][0] == q[i][0] ) && ( p[i][1] == q[i][1] ) && ( p[i][2] == q[i][2] );
  }
  return( isequal );
}

/***********************************************************/
/***********************************************************/

/***********************************************************
 * returns
 * 1    if v[i] == w[i] for all spinors i
 * +/-2 if v[i] == +/- w[i] for all spinors i;
 *   sign is the product of all sign changes
 ***********************************************************/
inline int compare_spinor_configuration ( double _Complex ** const v, double _Complex ** const w, int const n, int * const dim ) {
  int isequal = 1;

  for ( int i = 0; i < n; i++ ) {
    int iep = 1;
    for ( int k = 0; k < dim[i]; k++ ) {
      iep &= cabs ( v[i][k] - w[i][k] ) < eps;
    }  // end of loop on components
    isequal &= iep;
  }  // end of loop on spinors

  return( isequal );
}  // end of compare_spinor_configuration

/***********************************************************/
/***********************************************************/

/***********************************************************
 * returns
 * +/-1 if v[i] == +/- w[i] for all spinors i;
 *   sign is the product of all sign changes
 ***********************************************************/
inline int compare_spinor_pm_configuration ( double _Complex ** const v, double _Complex ** const w, int const n, int * const dim ) {
  int isequalpm = 1;
  int sign = 1;

  for ( int i = 0; i < n; i++ ) {
    int iep = 1;
    int iem = 1;
    for ( int k = 0; k < dim[i]; k++ ) {
      iep &= cabs ( v[i][k] - w[i][k] ) < eps;
      iem &= cabs ( v[i][k] + w[i][k] ) < eps;
    }  // end of loop on components
    if ( iem ) sign = -sign;
    isequalpm &= iem || iep;
  }  // end of loop on spinors

  return ( sign*isequalpm );
}  // end of compare_spinor_pm_configuration


/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
inline little_group_projector_applicator_type * show_little_group_projector_applicator ( little_group_projector_applicator_type * a, FILE*ofs ) {


  FILE * myofs = ofs != NULL ? ofs : stdout;

  fprintf ( myofs, "\n\n# [show_little_group_projector_applicator] =================================\n");
  fprintf ( myofs, "# [show_little_group_projector_applicator] = BEGIN lg projector applicator = \n");
  fprintf ( myofs, "# [show_little_group_projector_applicator] =================================\n");
  fprintf ( myofs, "# [show_little_group_projector_applicator] lg projector applicator %s\n", a->name );
  fprintf ( myofs, "# [show_little_group_projector_applicator] P             %3d %3d %3d\n", a->P[0], a->P[1], a->P[2] );
  fprintf ( myofs, "# [show_little_group_projector_applicator] rotations     %3d\n", a->rotation_n );
  fprintf ( myofs, "# [show_little_group_projector_applicator] interpolators %3d\n", a->interpolator_n );

  /***********************************************************
   * reduce the applicator
   ***********************************************************/
  // reduce_little_group_projector_applicator ( a, 1 );
  reduce_little_group_projector_applicator ( a, 0 );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * now print the complete projector information
   * for this row of the target representation
   ***********************************************************/

  // show the information for rotations

  for ( int iparity = 0; iparity < 2; iparity++ ) {

    for ( int irot = 0; irot < a->rotation_n; irot++ ) {

      // show the overall coefficient
      if ( cabs ( a->c[iparity][irot] ) < eps ) continue;
      // fprintf ( myofs, "# [show_little_group_projector_applicator] rot %2d par %2d elem %-8s", irot+1 , 1 - 2*iparity, a->rotation_name[iparity][irot] );
      fprintf ( myofs, "     rot %2d par %2d elem %-8s", irot+1 , 1 - 2*iparity, a->rotation_name[iparity][irot] );
      // fprintf ( myofs, "  c  %25.16e %25.16e\n", __dgeps( creal( a->c[iparity][irot]), eps ), __dgeps( cimag( a->c[iparity][irot]), eps ) );

      for ( int iop = 0; iop < a->interpolator_n; iop++ ) {
        fprintf ( myofs, "  p%d (%2d %2d %2d)", iop+1, a->prot[iparity][irot][iop][0], a->prot[iparity][irot][iop][1], a->prot[iparity][irot][iop][2] );
      }

      for ( int iop = 0; iop < a->interpolator_n; iop++ ) {
        // fprintf ( myofs, "    operator %d p  %3d %3d %3d v ", iop+1, a->prot[0][irot][iop][0], a->prot[0][irot][iop][1], a->prot[0][irot][iop][2] );
        // fprintf ( myofs, "  p%d  (%3d %3d %3d) v ", iop+1, a->prot[0][irot][iop][0], a->prot[0][irot][iop][1], a->prot[0][irot][iop][2] );
        // show the rotated basis vectors
        fprintf ( myofs, "  v%d (", iop+1);
        for ( int k = 0; k < a->interpolator_dim[iop]; k++ ) {
          double const dre = __dgeps ( creal( a->v[iparity][irot][iop][k]), eps );
          double const dim = __dgeps ( cimag( a->v[iparity][irot][iop][k]), eps );
          fprintf ( myofs, " %16.7e +I %16.7e,", dre, dim );
              // creal( a->v[0][irot][iop][k]), cimag( a->v[0][irot][iop][k] ) );
        }
        fprintf ( myofs, ")" );


      }  // end of loop on operators

      fprintf ( myofs, "  c  %25.16e %25.16e\n", __dgeps( creal( a->c[iparity][irot]), eps ), __dgeps( cimag( a->c[iparity][irot]), eps ) );

    }  // end of loop on rotations

    fprintf ( myofs, "# [show_little_group_projector_applicator] \n\n");

  }  // end of loop on rotations
  fprintf ( myofs, "# [show_little_group_projector_applicator] ===============================\n");
  fprintf ( myofs, "# [show_little_group_projector_applicator] = END lg projector applicator =\n");
  fprintf ( myofs, "# [show_little_group_projector_applicator] ===============================\n\n\n");

  return( a );
}  // end of show_little_group_projector_applicator

/***********************************************************/
/***********************************************************/

}  // end of namespace cvc
