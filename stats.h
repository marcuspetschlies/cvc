#ifndef _STATS_H
#define _STATS_H

#if defined MAIN_PROGRAM
#  define EXTERN
#else
#  define EXTERN extern
#endif 

EXTERN int nreplica, *n_r, npara, ipo;
EXTERN double *para, s_tau, **data;
EXTERN dquant func;
EXTERN char obsname[800];
EXTERN int nalpha;

#endif
