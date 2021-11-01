#ifndef _CLOVER_H
#define _CLOVER_H

namespace cvc {

int init_clover ( double *** clover_term, double **(*mzz)[2], double **(*mzzinv)[2], double*gauge_field, double const mass, double const csw );


void fini_clover ( double **(*mzz)[2], double **(*mzzinv)[2] );

}
#endif
