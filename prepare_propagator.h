#ifndef _PREPARE_PROPAGATOR_H
#define _PREPARE_PROPAGATOR_H

namespace cvc {
#if 0
int prepare_seqprop_point_from_stochastic_oneend (double**fp_out, double **phi, double **chi,
        double **prop, const int idsource, int idsink, int ncol, double*phase_field, unsigned int N);

int prepare_prop_point_from_stochastic (double**fp_out, double**phi, double**chi,
        double**prop, int idsink, double*phase_field, unsigned int N);
#endif
int prepare_seqn_stochastic_vertex_propagator_sliced3d (double**seq_prop, double**stoch_prop, double**stoch_source,
        double**prop, int nstoch, int nprop, int momentum[3], int gid);

int prepare_seqn_stochastic_vertex_propagator_sliced3d_oet (double**seq_prop, double**stoch_prop_0, double**stoch_prop_p,
    double**prop, int momentum[3], int gid, int idsource);
  

}  /* end of namespace cvc */
#endif
