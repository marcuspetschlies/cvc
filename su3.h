#ifndef _SU3_H
#define _SU3_H

namespace cvc {

typedef struct
{
   _Complex double c00, c01, c02, c10, c11, c12, c20, c21, c22;
} su3;

typedef su3 su3_tuple[4];

extern double _Complex *** lambda_gm;

void init_lambda_gm (void);
void fini_lambda_gm (void);

}
#endif
