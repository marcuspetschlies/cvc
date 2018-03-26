#ifndef _VDAG_W_UTILS_H
#define _VDAG_W_UTILS_H

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

namespace cvc {

/***********************************************************
 *
 * dimV must be integer multiple of dimW
 ***********************************************************/
int vdag_w_spin_color_reduction ( double ***contr, double ** const V, double ** const W, unsigned int const dimV, unsigned int const dimW, int const t );

/***********************************************************/
/***********************************************************/

/***********************************************************
 * momentum projection
 ***********************************************************/
int vdag_w_momentum_projection ( double _Complex ***contr_p, double *** const contr_x, int const dimV, int const dimW, const int (*momentum_list)[3], int const momentum_number, int const t, int const ieo, int const mu ); 

/***********************************************************/
/***********************************************************/

/***********************************************************
 * write to AFF file
 ***********************************************************/
int vdag_w_write_to_aff_file ( double _Complex *** const contr_tp, unsigned int const nv, unsigned int const nw, struct AffWriter_s*affw, char * const tag, const int (*momentum_list)[3], unsigned int const momentum_number, unsigned int const io_proc);
 
}  // end of namespace cvc

#endif
