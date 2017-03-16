


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

struct N_N_correlators_type{
  double **data;
  int size;
};

struct N_N_Wick_contractions_type{
  double **data;
  int size;
};

struct N_N_final_side_contracted_type{
  double **data;
};

struct forward_propagators_type{
  double **propagator_list_up;
  int no_fields;
};

struct sequential_propagators_type{
  double **data;
};

struct stochastic_propagators_type{
  double **data;
};

struct cvc_and_tmLQCD_information_type{
  double **data;
};


