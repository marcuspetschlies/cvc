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
 
int prepare_seq_stochastic_vertex_stochastic_oet (double**seq_prop, double**stoch_prop_0, double**stoch_prop_p,
        int gid, int source_coords[4], int nsample);

int point_to_all_fermion_propagator_clover_eo ( double **eo_spinor_field_e, double **eo_spinor_field_o,  int op_id,
        int global_source_coords[4], double *gauge_field, double **mzz, double **mzzinv, int check_propagator_residual, double **eo_spinor_work );

int prepare_clover_eo_stochastic_timeslice_propagator ( double***eo_stochastic_source_allt, double***eo_stochastic_propagator_allt,
        double ***eo_stochastic_source, double ***eo_stochastic_propagator, int sequential_source_timeslice, int ns, int adj );

int prepare_clover_eo_stochastic_timeslice_propagator (
    double**prop, double*source, 
    double *eo_evecs_block, double*evecs_lambdainv, int evecs_num, 
    double*gauge_field_with_phase, double **mzz[2], double **mzzinv[2],
    int op_id, int check_propagator_residual);

int prepare_clover_eo_sequential_propagator_timeslice (
  double *sequential_propagator_e, double *sequential_propagator_o,
  double *forward_propagator_e,    double *forward_propagator_o,
  int momentum[3], int gamma_id, int timeslice,
  double*gauge_field, double**mzz, double**mzzinv);

int point_to_all_fermion_propagator_clover_full2eo ( double **eo_spinor_field_e, double **eo_spinor_field_o,  int op_id,
    int global_source_coords[4], double *gauge_field, double **mzz, double **mzzinv, int check_propagator_residual );

int check_residuum_full ( double **source, double **prop, double *gauge_field, double const mutm, double **mzz, int const nfields );

int check_residuum_eo ( double **source_e, double **source_o, double **prop_e, double **prop_o, double *gauge_field, double const mutm, double **mzz, int const nfields );

}  /* end of namespace cvc */
#endif
