#ifndef _CONTRACT_BARYON_H
#define _CONTRACT_BARYON_H

namespace cvc {

int contract_piN_piN (spinor_propagator_type **res, double**uprop_list, double**dprop_list, double**tfii_list, double**tffi_list, double**pffii_list, 
        int ncomp, int(*comp_list)[2], double*comp_list_sign);

int contract_piN_piN_oet (spinor_propagator_type **res, double**uprop_list, double**dprop_list, double**pfifi_list, int ncomp, int(*comp_list)[2], double*comp_list_sign);


int contract_N_N (spinor_propagator_type **res, double**uprop_list, double**dprop_list, int ncomp, int(*comp_list)[2], double*comp_list_sign);

int contract_D_D (spinor_propagator_type **res, double**uprop_list, double**dprop_list, int ncomp, int(*comp_list)[2], double*comp_list_sign);

int contract_piN_D (spinor_propagator_type **res, double**uprop_list, double**dprop_list, double**tfii_list, int ncomp, int(*comp_list)[2], double*comp_list_sign);

int add_baryon_boundary_phase (spinor_propagator_type*sp, int tsrc, int ncomp);

int add_source_phase (double****connt, int pi2[3], int pf2[3], int source_coords[3], int ncomp);

}  /* end of namespace cvc */
#endif

