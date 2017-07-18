#ifndef _DEFAULT_INPUT_H
#define _DEFAULT_INPUT_H

namespace cvc {

#define _default_T_global 0
#define _default_LX 0
#define _default_LY 0
#define _default_LZ 0
#define _default_L5 0
#define _default_Nconf 0
#define _default_kappa (0.7071068)
#define _default_kappa5d (0.)
#define _default_mu (0.)
#define _default_mubar (0.)
#define _default_epsbar (0.)
#define _default_musigma (0.)
#define _default_mudelta (0.)
#define _default_sourceid  0
#define _default_sourceid2 0
#define _default_sourceid_step 1
#define _default_Nsave 1
#define _default_format 0
#define _default_BCangleT (1.)
#define _default_BCangleX (0.)
#define _default_BCangleY (0.)
#define _default_BCangleZ (0.)
#define _default_filename_prefix "prop."
#define _default_filename_prefix2 "prop."
#define _default_filename_prefix3 "prop."

#define _default_sequential_filename_prefix "seq_prop."
#define _default_sequential_filename_prefix2 "seq_prop."

#define _default_gaugefilename_prefix "conf."
#define _default_outfile_prefix "out."
#define _default_path_prefix "./"
#define _default_resume 0
#define _default_subtract 0
#define _default_source_location 0
#define _default_niter_max 1000
#define _default_solver_precision 1.e-08
#define _default_reliable_delta 1.e-01
#define _default_gaugeid 0
#define _default_gaugeid2 0
#define _default_gauge_step 1
#define _default_source_type 1
#define _default_noise_type 1
#define _default_seed 12345
#define _default_hpe_order_min -1
#define _default_hpe_order_max -1
#define _default_hpe_order -1

#define _default_cutangle -1.
#define _default_cutradius -1.
#define _default_cutdirT 1
#define _default_cutdirX 1
#define _default_cutdirY 1
#define _default_cutdirZ 1
#define _default_rmin -1.
#define _default_rmax -1.

#define _default_avgT 0
#define _default_avgL 0

#define _default_model_dcoeff_re 0.
#define _default_model_dcoeff_im 0.
#define _default_model_mrho 0.
#define _default_ft_rmax 0.
#define _default_prop_normsqr 1.
#define _default_qhatsqr_min 0.
#define _default_qhatsqr_max -1.

#define _default_Nlong -1
#define _default_N_ape -1
#define _default_N_Jacobi -1

#define _default_alpha_ape 0.
#define _default_kappa_Jacobi 0.

#define _default_alpha_hyp 0.
#define _default_N_hyp 0

#define _default_source_timeslice 0
#define _default_sequential_source_timeslice -1
#define _default_sequential_source_timeslice_number 0

#define _default_sequential_source_location_x 0
#define _default_sequential_source_location_y 0
#define _default_sequential_source_location_z 0


#define _default_no_extra_masses 0
#define _default_no_light_masses 1
#define _default_no_strange_masses 0
#define _default_nproc_t 1
#define _default_nproc_x 1
#define _default_nproc_y 1
#define _default_nproc_z 1

#define _default_ts_nb_x_up 0
#define _default_ts_nb_x_dn 0
#define _default_ts_nb_y_up 0
#define _default_ts_nb_y_dn 0
#define _default_ts_nb_z_up 0
#define _default_ts_nb_z_dn 0

#define _default_ts_id 0
#define _default_xs_id 0
#define _default_ys_id 0

#define _default_Tstart 0
#define _default_LXstart 0
#define _default_LYstart 0
#define _default_LZstart 0

#define _default_local_local     0
#define _default_local_smeared   0
#define _default_smeared_local   0
#define _default_smeared_smeared 0
#define _default_rotate_ETMC_UKQCD 0
#define _default_propagator_position 0

#define _default_gpu_device_number 0
#define _default_gpu_per_node -1

#define _default_coherent_source 0
#define _default_coherent_source_base 0
#define _default_coherent_source_delta 0

#define _default_gauge_file_format 0
#define _default_rng_filename "ranlxd_state"
#define _default_source_index_min 0
#define _default_source_index_max -1
#define _default_propagator_bc_type 0
#define _default_propagator_gamma_basis 0
#define _default_propagator_precision 32
#define _default_write_source 0
#define _default_read_source 0

#define _default_write_propagator 0
#define _default_read_propagator 1
#define _default_read_sequential_propagator 0
#define _default_write_sequential_source 0
#define _default_write_sequential_propagator 0

#define _default_nsample 1
#define _default_nsample_oet 1
#define _default_num_threads 1
#define _default_source_momentum_x 0
#define _default_source_momentum_y 0
#define _default_source_momentum_z 0
#define _default_source_momentum_set 0

#define _default_source_momentum_number 0

#define _default_sink_momentum_x 0
#define _default_sink_momentum_y 0
#define _default_sink_momentum_z 0
#define _default_sink_momentum_set 0
#define _default_sink_momentum_number 0

#define _default_seq_source_momentum_x 0
#define _default_seq_source_momentum_y 0
#define _default_seq_source_momentum_z 0
#define _default_seq_source_momentum_set 0

#define _default_seq_source_momentum_number 0

#define _default_rng_state NULL
#define _default_verbose  0
#define _default_m0 0.
#define _default_m5 0.

#define _default_cpu_prec 2
#define _default_gpu_prec 2
#define _default_gpu_prec_sloppy 1
#define _default_inverter_type_name "none"

#define _default_space_dilution_depth 0
#define _default_mms_id -1
#define _default_check_inversion 0


#define _default_src_snk_time_separation 0

#define _default_seq_source_gamma_id -1
#define _default_seq_source_gamma_id_number 0

#define _default_source_gamma_id -1
#define _default_source_gamma_id_number 0

#define _default_csw 0

#define _default_fermion_type -1
#define _default_source_location_number 0
#define _default_coherent_source_number 1

#define _default_total_momentum_number 0

#define _default_twopoint_function_number 0

#define _default_zero 0

}
#endif
