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

int stochastic_source_ti_vertex_ti_propagator (double*** seq_stochastic_source, double**stoch_source, 
    double**prop, int nstoch, int nprop, int momentum[3], int gid);

int prepare_seqn_stochastic_vertex_propagator_sliced3d_oet (double**seq_prop, double**stoch_prop_0, double**stoch_prop_p,
    double**prop, int momentum[3], int gid, int idsource);
 
int prepare_seq_stochastic_vertex_stochastic_oet (double**seq_prop, double**stoch_prop_0, double**stoch_prop_p,
        int gid, int source_coords[4], int nsample);

int point_source_propagator (double ** const prop, int const gsx[4], int const op_id, int const smear_source, int const smear_sink, double * const gauge_field_smeared, int const check_residual, double * const gauge_field, double ** mzz[2] );


int prepare_propagator_from_source ( double ** const prop, double ** const source , int const nsc, int const op_id,
    int smear_source, int smear_sink, double *gauge_field_smeared, 
    int const check_residual, double * const gauge_field, double ** mzz[2], char * prefix );

}  /* end of namespace cvc */
#endif
