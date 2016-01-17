#ifndef _LAPHS_IO_H
#define _LAPHS_IO_H

#include "laphs.h"

namespace cvc {


int read_perambulator(perambulator_type *peram);

int read_eigensystem(eigensystem_type *es);
int read_eigensystem_timeslice(eigensystem_type *es, unsigned int it);

int read_eigenvalue_timeslice(eigensystem_type *es, unsigned int it);
int read_phase_timeslice(eigensystem_type *es, unsigned int it);


int read_randomvector(randomvector_type *rv, char*quark_type, int irnd);

}
#endif
