#ifndef _CONTRACT_PIN2PIN_H
#define _CONTRACT_PIN2PIN_H

namespace cvc {

int contract_piN2piN (spinor_propagator_type *res, double**uprop_list, double**dprop_list, double**tfii_list, double**tffi_list, double**pffii_list, 
        int ncomp, int(*comp_list)[2], double*comp_list_sign);

}  /* end of namespace cvc */
#endif

