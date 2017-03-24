


/*struct gamma_component_information_type{
  int num_component;
  int** gamma_component;
  double ** gamma_component_sign;
};

struct gamma_components_type{
  gamma_component_information_type piN_piN_component;
  gamma_component_information_type N_N_component;
  gamma_component_information_type D_D_component;
  gamma_component_information_type piN_D_component;
  int num_component_max;
};*/

extern const int n_c;
extern const int n_s;

typedef char pathname_type[255];

struct program_instruction_type{
  int filename_set;
	char filename[255];

  int io_proc;

  double *gauge_field_smeared;

  unsigned int VOL3;
  
  size_t sizeof_spinor_field;
  size_t sizeof_spinor_field_timeslice;

  int argc;
  char **argv;

  int op_id_up;
  int op_id_dn;
  int source_proc_id;
  double ratime, retime;
  double *spinor_work[2];
  spinor_propagator_type **conn_X;
  double ****buffer;
  int icomp, iseq_mom, iseq2_mom;

  int max_num_diagram;
};

struct global_source_location_type{
  int x[4];
};

struct local_source_location_type{
  int x[4];
  int proc_id;
};

struct three_momentum_type{
  int p[3];
};

struct information_needed_for_source_phase_type{
  bool add_source_phase;
  three_momentum_type pi2;
  three_momentum_type pf2;
};

typedef double**** FT_WDc_contractions_type;
typedef double**** gathered_FT_WDc_contractions_type;

typedef double** propagator_pointer_list_type;

typedef double** general_propagator_tffi_type;
typedef double** general_propagator_pffii_type;

struct global_and_local_stochastic_source_timeslice_type{
  int t_src;
  int local_t_src;
  int local_grid_contains_t_src;
};

struct forward_propagators_type{
  double **propagator_list_up;
  double **propagator_list_dn;
  int no_fields;
};

struct sequential_propagators_type{
  double **propagator_list;
  int no_fields;
};

struct stochastic_sources_and_propagators_type{
  double **propagator_list;
  double **source_list;
  int no_fields;
};

struct cvc_and_tmLQCD_information_type{
  double **data;
};

struct contraction_writer_type{
  struct AffWriter_s *affw;
  struct AffNode_s *affn, *affdir;
  double _Complex *aff_buffer;
};
