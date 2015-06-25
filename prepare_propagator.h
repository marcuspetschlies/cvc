#ifndef _PREPARE_PROPAGATOR_H
#define _PREPARE_PROPAGATOR_H

namespace cvc {

int prepare_seqprop_point_from_stochastic_oneend (double**fp_out, double **phi, double **chi,
        double **prop, const int idsource, int idsink, int ncol, double*phase_field, unsigned int N);

int prepare_prop_point_from_stochastic (double**fp_out, double**phi, double**chi,
        double**prop, int idsink, double*phase_field, unsigned int N);

}
#endif
