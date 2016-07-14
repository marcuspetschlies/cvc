#ifndef _PREPARE_SOURCE_H
#define _PREPARE_SOURCE_H

namespace cvc {

int prepare_volume_source(double *s, unsigned int V);

int project_spinor_field(double *s, double * r, int parallel, double *V, int num, unsigned int N);

}  /* end of namespace cvc */

#endif
