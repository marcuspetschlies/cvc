#ifndef AFF_KEY_CONVERSION_H
#define AFF_KEY_CONVERSION_H

namespace cvc {

void aff_key_conversion (char*key, char * const tag, int const i_sample, int const pi2[3], int const pf1[3], int const pf2[3], int const source_coords[4], int const gamma_id, int const C_gamma_id,  int const i_spin );

int v2_key_index_conversion ( double _Complex *buffer, int perm[4], int N, int LL[4] );

int vn_oet_read_key ( double _Complex *key_buffer, char*tag, int const i_sample, int const pi2[3], int const pf1[3], int const pf2[3], int const source_coords[4], int const gamma_id, int const C_gamma_id, struct AffReader_s *affr );

void aff_key_conversion_diagram (  char*key, char * const tag, int const pi1[3], int const pi2[3], int const pf1[3], int const pf2[3], 
        int const gi1, int const gi2, int const gf1, int const gf2, int const source_coords[4], char * const diag_name, int const diag_id , int const gx1_mult_C);

void gamma_name_to_gamma_signed_id (int *id, double*sign, char *name );


}  /* end of namespace cvc */

#endif
