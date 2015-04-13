/*
      stat5.h                                   B Bunk 2/2005
                                                rev    3/2006
*/
#include <stdio.h>            /* for FILE */

#define INT  int
#define REAL double

/* prototypes for stat5 functions */
void  clear5(INT nvar, INT nbmax);
void  accum5(INT ivar, REAL value);
REAL  aver5(INT ivar);
REAL  var5(INT ivar);
REAL  sigma5(INT ivar);
REAL  cov5(INT ivar, INT jvar);
REAL  covar5(INT ivar, INT jvar);
REAL  tau5(INT ivar);
REAL  rsq5(INT ivar);
REAL  tauint5(INT ivar);
void  jackout5(INT ivar1, INT *nb, REAL bj[]);
void  jackeval5(INT nb, REAL fj[], REAL *aver, REAL *sigma);
void  jack5(REAL (*fct)(INT nvar, REAL a[]), REAL *aver, REAL *sigma);
void  save5(FILE *file);
void  savef5(FILE *file);
void  get5(FILE *file);
void  getf5(FILE *file);

#undef INT
#undef REAL

