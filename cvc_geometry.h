#ifndef _CVC_GEOMETRY_H
#define _CVC_GEOMETRY_H

namespace cvc {

// 4-dim. arrays
unsigned long int get_index(const int t, const int x, const int y, const int z);

void geometry(void);

int init_geometry(void);

void free_geometry(void);

// 5-dim. arrays
unsigned long int get_index_5d(const int s, const int t, const int x, const int y, const int z);
void geometry_5d(void);
int init_geometry_5d(void);
void free_geometry_5d(void);

}
#endif
