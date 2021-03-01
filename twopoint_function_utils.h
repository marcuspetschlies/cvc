#ifndef TWOPOINT_FUNCTION_UTILS_H
#define TWOPOINT_FUNCTION_UTILS_H

#include "types.h"
#include "group_projection.h"

namespace cvc {

void twopoint_function_init ( twopoint_function_type *p );
void twopoint_function_print ( twopoint_function_type *p, char *name, FILE*ofs );
void twopoint_function_print_diagram_key (char*key, twopoint_function_type *p, int id );
void twopoint_function_print_correlator_key (char*key, twopoint_function_type *p );

void twopoint_function_correlator_phase (double _Complex * const c, twopoint_function_type *p, unsigned int const N);

void twopoint_function_print_correlator_data ( double _Complex * const c, twopoint_function_type *p, FILE*ofs, int const N );
  
void twopoint_function_get_aff_filename_prefix (char*filename, twopoint_function_type*p);

void twopoint_function_copy ( twopoint_function_type *p, twopoint_function_type *r , int const copy_data );

double twopoint_function_get_diagram_norm ( twopoint_function_type *p, int const id );

void twopoint_function_get_diagram_name (char *diagram_name,  twopoint_function_type *p, int const id );

double _Complex twopoint_function_get_correlator_phase ( twopoint_function_type *p );

int twopoint_function_accumulate_diagrams ( double _Complex *** const diagram, twopoint_function_type *p, int const N, struct AffReader_s *affr );

void twopoint_function_fini ( twopoint_function_type *p );

void * twopoint_function_allocate ( twopoint_function_type * p );

void twopoint_function_show_data ( twopoint_function_type *p, FILE*ofs );

int twopoint_function_fill_data ( twopoint_function_type *p, char *datafile_prefix );

int twopoint_function_write_data ( twopoint_function_type *p );

int twopoint_function_data_location_identifier ( char * udli, twopoint_function_type *p, char * const datafile_prefix, int const ids, char * const sep );

int twopoint_function_fill_data_from_udli ( twopoint_function_type *p, char * udli, int const io_proc );

int twopoint_function_apply_diagram_norm ( twopoint_function_type *p );

int twopoint_function_accum_diagrams ( double _Complex *** const z, twopoint_function_type *p );

int twopoint_function_correlator_from_h5_file ( twopoint_function_type * const p, int const io_proc );

void twopoint_function_check_reference_rotation ( twopoint_function_type * const tp, little_group_projector_type * const pr , double const deps );

void twopoint_function_check_reference_rotation_vector_spinor ( twopoint_function_type * const tp, little_group_projector_type * const pr, double const deps );

/***************************************************************************/
/***************************************************************************/

/***************************************************************************
 * coords to index
 ***************************************************************************/
inline unsigned int coords_to_index ( const int * coords, const unsigned int * dims, const unsigned int n ) {

  unsigned int res = coords[0];
  for ( unsigned int i = 1; i < n; i++ ) {
    res = res * dims[i] + coords[i];
  }
  return ( res );
}  /* end of coords_to_index */

/***************************************************************************
 * momentum filter
 ***************************************************************************/
inline int momentum_filter ( int * const pf, int * const pc, int * const pi1, int * const pi2, int const pp_max ) {

  /* check mometnum conservation  */
  if ( pf == NULL || pc == NULL ) return ( 1 == 0 );

  if ( pi2 == NULL && pc == NULL ) {

    int const is_conserved = ( pi1[0] + pf[0] == 0 ) && ( pi1[1] + pf[1] == 0 ) && ( pi1[2] + pf[2] == 0 );

    int const is_lessequal = \
            ( pi1[0] * pi1[0] + pi1[1] * pi1[1] + pi1[2] * pi1[2] <= pp_max ) \
        &&  ( pf[0]  * pf[0]  + pf[1]  * pf[1]  + pf[2]  * pf[2]  <= pp_max );

    return ( is_conserved && is_lessequal );

  } else if ( pc != NULL ) {

    if ( pi2 == NULL ) {

      int const is_conserved = ( pi1[0] + pf[0] + pc[0] == 0 ) && \
                               ( pi1[1] + pf[1] + pc[1] == 0 ) && \
                               ( pi1[2] + pf[2] + pc[2] == 0 );

      int const is_lessequal = \
                               ( pi1[0] * pi1[0] + pi1[1] * pi1[1] + pi1[2] * pi1[2]  <= pp_max ) \
                            && ( pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]  <= pp_max ) \
                            && ( pf[0] * pf[0] + pf[1] * pf[1] + pf[2] * pf[2]  <= pp_max );

      return ( is_conserved && is_lessequal );

    } else {

      int const is_conserved = ( pi1[0] + pi2[0] + pf[0] + pc[0] == 0 ) &&
                               ( pi1[1] + pi2[1] + pf[1] + pc[1] == 0 ) &&
                               ( pi1[2] + pi2[2] + pf[2] + pc[2] == 0 );

      int const is_lessequal = \
                               ( pi1[0] * pi1[0] + pi1[1] * pi1[1] + pi1[2] * pi1[2]  <= pp_max ) \
                            && ( pi2[0] * pi2[0] + pi2[1] * pi2[1] + pi2[2] * pi2[2]  <= pp_max ) \
                            && ( pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]  <= pp_max ) \
                            && ( pf[0] * pf[0] + pf[1] * pf[1] + pf[2] * pf[2]  <= pp_max );

      return ( is_conserved && is_lessequal );
    }
  } else {
    return ( 1 == 0 );
  }

} /* end of mometnum_filter */


}  // end of namespace cvc

#endif
