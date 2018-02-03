#ifndef _CVC_GEOMETRY_3D_H
#define _CVC_GEOMETRY_3D_H

namespace cvc {

// 4-dim. arrays
unsigned long int get_index_3d( const int x, const int y, const int z);

void geometry_3d(void);

int init_geometry_3d(void);

void free_geometry_3d(void);

}
#endif
