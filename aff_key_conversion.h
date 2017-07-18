#ifndef AFF_KEY_CONVERSION_H
#define AFF_KEY_CONVERSION_H

namespace cvc {

void aff_key_conversion (char*key, char*tag, int i_sample, int pi2[3], int pf1[3], int pf2[3], int source_coords[4], int gamma_id, int i_spin );

int v2_key_index_conversion ( double _Complex *buffer, int perm[4], int N, int LL[4] );

int vn_oet_read_key ( double _Complex *key_buffer, char*tag, int i_sample, int pi2[3], int pf1[3], int pf2[3], int source_coords[4], int gamma_id, struct AffReader_s *affr );

void aff_key_conversion_diagram (  char*key, char*tag, int pi1[3], int pi2[3], int pf1[3], int pf2[3], int gi1, int gi2, int gf1, int gf2, int source_coords[4], char*diag_name, int diag_id );

void gamma_name_to_gamma_signed_id (int *id, double*sign, char *name );


}  /* end of namespace cvc */

#endif
