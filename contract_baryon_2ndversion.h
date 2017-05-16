
namespace cvc{

void compute_b_or_w_diagram_from_V2(gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,int diagram,b_1_xi_type *b_1_xi,w_1_xi_type *w_1_xi,V2_for_b_and_w_diagrams_type *V2_for_b_and_w_diagrams,program_instruction_type *program_instructions,int num_component_f1,int *component_f1,int num_component_i1,int *component_i1,int num_components,int(*component)[2],int tsrc);

void compute_V2_for_b_and_w_diagrams(int i_src,int i_coherent,int ncomp,int *comp_list,double*comp_list_sign,int nsample,V2_for_b_and_w_diagrams_type *V2_for_b_and_w_diagrams,program_instruction_type *program_instructions,double **uprop_list,double **tfii_list,double **phi_list);

}
