#!/bin/bash

file="read_input_parser.l"
tmpFile="read_input_parser_cvc.l"
auxFile="tmp.l"

varList=("alpha_ape" "alpha_hyp" "avgL" "avgT" "BCangle" "filename_prefix" "format" "ft_rmax" "g_as_over_a" "gaugefilename_prefix" "g_check_inversion" "g_coherent_source" "g_cpu_prec" "g_cutangle" "g_cutdir" "g_cutradius" "g_epsbar" "g_gauge_file_format" "g_gaugeid" "g_gauge_step" "g_gpu_device_number" "g_gpu_per_node" "g_gpu_prec" "g_kappa" "g_local_local" "g_local_smeared" "g_m0" "g_m5" "g_mms_id" "g_mu" "g_no_extra_masses" "g_noise_type" "g_no_light_masses" "g_no_strange_masses" "g_nproc_t" "g_nproc_x" "g_nproc_y" "g_nsample" "g_num_threads" "g_outfile_prefix" "g_path_prefix" "g_propagator_bc_type" "g_propagator_gamma_basis" "g_propagator_position" "g_propagator_precision" "g_prop_normsqr" "g_qhatsqr_max" "g_qhatsqr_min" "g_read_source" "g_resume" "g_rmax" "g_rmin" "g_rng_filename" "g_rotate_ETMC_UKQCD" "g_seed" "g_seq_source_momentum" "g_sequential_source_timeslice" "g_sink_momentum" "g_smeared_local" "g_smeared_smeared" "g_sourceid" "g_source_index" "g_source_location" "g_source_momentum" "g_source_timeslice" "g_source_type" "g_subtract" "g_write_source" "hpe_order" "kappa_Jacobi" "L" "model_dcoeff_im" "model_dcoeff_re" "model_mrho" "N_ape" "Nconf" "N_hyp" "niter_max" "N_Jacobi" "Nlong" "Nsave" "reliable_delta" "solver_precision" "T_global" "myverbose" "line_of_file" "comment_caller" "name_caller" "g_inverter_type_name")

cp $file $tmpFile

for v in ${varList[*]}; do
  cp $tmpFile $auxFile
  cat $auxFile | awk '
    BEGIN{w=0}
    /start namespace inclusion here/ {w=1}
    /end namespace inclusion here/ {w=0}
    w==1 && !/^</ && !/[a-zA-Z_0-9]'$v'/ {gsub(/'$v'/,"cvc::'$v'",$0); print; next}
    w==1 && ( /^</ || /[a-zA-Z_0-9]'$v'/ ) {print; next}
    w==0 {print}' > $tmpFile
done

exit 0
