#ifndef TWOPOINT_FUNCTION_UTILS_H
#define TWOPOINT_FUNCTION_UTILS_H

#include "types.h"
namespace cvc {

void twopoint_function_init ( twopoint_function_type *p );
void twopoint_function_print ( twopoint_function_type *p, char *name, FILE*ofs );
void twopoint_function_print_diagram_key (char*key, twopoint_function_type *p, int id );
void twopoint_function_print_correlator_key (char*key, twopoint_function_type *p );

void twopoint_function_correlator_phase (double _Complex * const c, twopoint_function_type *p, unsigned int const N);

void twopoint_function_print_correlator_data ( double _Complex *c, twopoint_function_type *p, FILE*ofs );

void twopoint_function_get_aff_filename_prefix (char*filename, twopoint_function_type*p);

void twopoint_function_copy ( twopoint_function_type *p, twopoint_function_type *r );

void twopoint_function_get_diagram_norm ( double*r, twopoint_function_type *p, int id );

double _Complex twopoint_function_get_correlator_phase ( twopoint_function_type *p );

}  /* end of namespace cvc */


#endif
