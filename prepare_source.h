#ifndef _PREPARE_SOURCE_H
#define _PREPARE_SOURCE_H

namespace cvc {

int prepare_volume_source(double *s, unsigned int V);

int init_eo_spincolor_pointsource_propagator(double *s_even, double *s_odd, int global_source_coords[4], int isc, int sign, int have_source, double *work0);

int fini_eo_propagator(double *s_even, double *s_odd, double *phi_even, double *phi_odd, int sign, double *work0);

int init_eo_sequential_source(double *s_even, double *s_odd, double *p_even, double *p_odd, int tseq, int sign, int pseq[3], int gseq, double *work0);


int init_clover_eo_spincolor_pointsource_propagator(double *s_even, double *s_odd, int global_source_coords[4], int isc, double*mzzinv, int have_source, double *work0);

int fini_clover_eo_propagator(double *p_even, double *p_odd, double *r_even, double *r_odd , double*mzzinv, double *work0);

int init_clover_eo_sequential_source(double *s_even, double *s_odd, double *p_even, double *p_odd, int tseq, double*mzzinv, int pseq[3], int gseq, double *work0);

int check_vvdagger_locality (double** V, int numV, int gcoords[4], char*tag, double **sp);

int init_sequential_source(double *s, double *p, int tseq, int pseq[3], int gseq);

}  /* end of namespace cvc */

#endif
