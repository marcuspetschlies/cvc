#ifndef _MAKE_X_ORBITS_H
#define _MAKE_X_ORBITS_H

namespace cvc {

int init_x_orbits_4d(unsigned int **xid, unsigned int **xid_count, double **xid_val, unsigned int *xid_nc, unsigned int ***xid_member, int gcoords[4]);

int fini_x_orbits_4d(unsigned int **xid, unsigned int **xid_count, double **xid_val, unsigned int ***xid_member);

}  /* of namespace cvc */

#endif
