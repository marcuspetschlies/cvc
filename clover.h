#ifndef _CLOVER_H
#define _CLOVER_H

namespace cvc {

int init_clover ( double **(*mzz)[2], double **(*mzzinv)[2], double*gauge_field );

void fini_clover(void);

}
#endif
