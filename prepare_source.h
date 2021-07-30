#ifndef _PREPARE_SOURCE_H
#define _PREPARE_SOURCE_H

#include "gamma.h"

namespace cvc {

int prepare_volume_source ( double * const s, unsigned int const V);

int init_eo_spincolor_pointsource_propagator(double *s_even, double *s_odd, int global_source_coords[4], int isc, double*gauge_field, int sign, int have_source, double *work0);

int fini_eo_propagator(double *p_even, double *p_odd, double *r_even,   double *r_odd , double*gauge_field, int sign, double *work0);

int init_eo_sequential_source(double *s_even, double *s_odd, double *p_even, double *p_odd, int tseq, double*gauge_field, int sign, int pseq[3], int gseq, double *work0);


int init_clover_eo_spincolor_pointsource_propagator(double *s_even, double *s_odd, int global_source_coords[4], int isc, double *gauge_field, double*mzzinv, int have_source, double *work0);

int fini_clover_eo_propagator(double *p_even, double *p_odd, double *r_even, double *r_odd , double*gauge_field, double*mzzinv, double *work0);

int init_clover_eo_sequential_source(double *s_even, double *s_odd, double *p_even, double *p_odd, int tseq, double*gauge_field, double*mzzinv, int pseq[3], int gseq, double *work0);

int check_vvdagger_locality (double** V, int numV, int gcoords[4], char*tag, double **sp);

int init_sequential_source(double *s, double *p, int tseq, int pseq[3], int gseq);

int init_coherent_sequential_source(double *s, double **p, int tseq, int ncoh, int pseq[3], int gseq);

int init_timeslice_source_oet ( double ** const s, int const tsrc, int * const momentum, int const spin_dilution, int const color_dilution, int const init);

int init_timeslice_source_z3_oet ( double ** const s, int const  tsrc, int const momentum[3], int const init );

int prepare_sequential_fht_loop_source ( double ** const seq_source, double _Complex *** const loop, double ** const prop, gamma_matrix_type * const gamma_mat, int const gamma_num, double _Complex * const ephase, int const type, gamma_matrix_type * const g5herm  );

int prepare_sequential_fht_twinpeak_source ( double ** const seq_source, double ** const prop, double * const scalar_field, int const gamma_id, double _Complex * const ephase );

}  /* end of namespace cvc */

#endif
