#ifndef _PREPARE_SOURCE_H
#define _PREPARE_SOURCE_H

namespace cvc {

int prepare_volume_source(double *s, unsigned int V);

int project_spinor_field(double *s, double * r, int parallel, double *V, int num, unsigned int N);

int init_eo_spincolor_pointsource_propagator(double *s_even, double *s_odd, int global_source_coords[4], int isc, int sign, int have_source, double *work0);
int fini_eo_spincolor_pointsource_propagator(double *s_even, double *s_odd, double *phi_even, double *phi_odd, int sign, double *work0);

}  /* end of namespace cvc */

#endif
