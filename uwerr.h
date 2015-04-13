#ifndef _UWERR_H
#define _UWERR_H

#include "dquant.h"

/* typedef int (*dquant_ptr)(int, double*, int, double*, double*); */

#ifndef TINY
#   define TINY (1e-15)
#endif

#ifndef SQR
#   define SQR(_a) ((_a)*(_a))
#endif

#ifndef CUB
#   define CUB(_a) ((_a)*(_a)*(_a))
#endif

#ifndef ARITHMEAN
#   define ARITHMEAN(_data_pointer, _data_number, _res_pointer) { \
              int _i;                                             \
              *(_res_pointer) = 0.0;                              \
              for(_i=0; _i<(_data_number); _i++) {                \
                 *(_res_pointer) += *((_data_pointer) + _i);      \
              }                                                   \
              *(_res_pointer) = *(_res_pointer) /                 \
                 (double)(_data_number);                          \
           }
#endif

#ifndef WEIGHEDMEAN
#   define WEIGHEDMEAN(_data_pointer, _data_number, _res_pointer, \
                       _weights_pointer) {                        \
              int _i;                                             \
              double _n=0.0;                                      \
              *(_res_pointer) = 0.0;                              \
              for(_i=0; _i<(_data_number); _i++) {                \
                 *(_res_pointer) += *((_data_pointer)+_i) *       \
                    (double)(*(_weights_pointer+_i));             \
                 _n += (double)(*(_weights_pointer+_i));          \
              }                                                   \
              *(_res_pointer) = *(_res_pointer) / _n;             \
           }
#endif

#ifndef STDDEV
#   define STDDEV(_data_pointer, _data_number, _res_pointer) {    \
              int _j;                                             \
              double _mean=0.0;                                   \
              *(_res_pointer) = 0.0;                              \
              ARITHMEAN(_data_pointer, _data_number, &_mean);     \
              for(_j=0; _j<(_data_number); _j++) {                \
                 *(_res_pointer) = *(_res_pointer) +              \
                    SQR(*((_data_pointer)+_j)-_mean);             \
              }                                                   \
              *(_res_pointer) = sqrt(*(_res_pointer)) /           \
                 (double)(_data_number);                          \
           }
#endif

#ifndef VAR
#   define VAR(_data_pointer, _data_number, _res_pointer) {    \
              int _j;                                             \
              double _mean=0.0;                                   \
              *(_res_pointer) = 0.0;                              \
              ARITHMEAN(_data_pointer, _data_number, &_mean);     \
              for(_j=0; _j<(_data_number); _j++) {                \
                 *(_res_pointer) = *(_res_pointer) +              \
                    SQR(*((_data_pointer)+_j)-_mean);             \
              }                                                   \
              *(_res_pointer) = *(_res_pointer)/                  \
                 (double)((_data_number) - 1);                    \
           }
#endif

#ifndef SUM
#   define SUM(_data_pointer, _data_number, _res_pointer) {       \
              int _i;                                             \
              *(_res_pointer) = 0.0;                              \
              for(_i=0; _i<(_data_number); _i++) {                \
                 *(_res_pointer) += *((_data_pointer) + _i);      \
              }                                                   \
           }
#endif

#ifndef ASSIGN
#   define ASSIGN(_data_pointer1, _data_pointer2, _data_number) { \
              int _i;                                             \
              for(_i=0; _i<(_data_number); _i++) {                \
                 *(_data_pointer1 + _i) = *(_data_pointer2 + _i); \
              }                                                   \
           }
#endif

#ifndef ADD
#   define ADD(_data_pointer1, _data_pointer2, _data_number) {        \
              int _i;                                                 \
              for(_i=0; _i<(_data_number); _i++) {                    \
                 *(_data_pointer1 + _i) = *(_data_pointer1 + _i) +    \
                    *(_data_pointer2 + _i);                           \
              }                                                       \
           }
#endif

#ifndef ADD_ASSIGN
#   define ADD_ASSIGN(_data_pointer1, _data_pointer2, _data_pointer3, \
                      _data_number) {                                 \
              int _i;                                                 \
              for(_i=0; _i<(_data_number); _i++) {                    \
                 *(_data_pointer1 + _i) = *(_data_pointer2 + _i) +    \
                    *(_data_pointer3 + _i);                           \
              }                                                       \
           }
#endif

#ifndef SUB
#   define SUB(_data_pointer1, _data_pointer2, _data_number) {        \
              int _i;                                                 \
              for(_i=0; _i<(_data_number); _i++) {                    \
                 *(_data_pointer1 + _i) = *(_data_pointer1 + _i) -    \
                    *(_data_pointer2 + _i);                           \
              }                                                       \
           }
#endif

#ifndef SUB_ASSIGN
#   define SUB_ASSIGN(_data_pointer1, _data_pointer2, _data_pointer3, \
                      _data_number) {                                 \
              int _i;                                                 \
              for(_i=0; _i<(_data_number); _i++) {                    \
                 *(_data_pointer1 + _i) = *(_data_pointer2 + _i) -    \
                    *(_data_pointer3 + _i);                           \
              }                                                       \
           }
#endif

#ifndef SET_TO
#   define SET_TO(_data_pointer, _data_number, _value) { \
              int _i;                                    \
              for(_i=0; _i<(_data_number); _i++) {       \
                 *(_data_pointer + _i) = _value;         \
              }                                          \
           }
#endif

#ifndef PROD
#   define PROD(_data_pointer, _data_number, _factor) {  \
              int _i;                                    \
              for(_i=0; _i<(_data_number); _i++) {       \
                 *(_data_pointer + _i) *= _factor;       \
              }                                          \
           }
#endif

#ifndef PROD_ASSIGN
#   define PROD_ASSIGN(_data_pointer1, _data_pointer2, _data_number, _factor) {  \
              int _i;                                    \
              for(_i=0; _i<(_data_number); _i++) {       \
                 *(_data_pointer1 + _i) = _factor *      \
                    *(_data_pointer2 + _i);              \
              }                                          \
           }
#endif

#ifndef DIV
#   define DIV(_data_pointer, _data_number, _factor) {   \
              int _i;                                    \
              for(_i=0; _i<(_data_number); _i++) {       \
                 *(_data_pointer + _i) /= _factor;       \
              }                                          \
           }
#endif

#ifndef DIV_ASSIGN
#   define DIV_ASSIGN(_data_pointer1, _data_pointer2, _data_number, _factor) {  \
              int _i;                                    \
              for(_i=0; _i<(_data_number); _i++) {       \
                 *(_data_pointer1 + _i) =                \
                    *(_data_pointer2 + _i) / _factor;    \
              }                                          \
           }
#endif

#ifndef SUB_PROD
#   define SUB_PROD(_data_pointer1, _data_pointer2,                   \
                    _data_number, _factor) {                          \
              int _i;                                                 \
              for(_i=0; _i<(_data_number); _i++) {                    \
                 *(_data_pointer1 + _i) = *(_data_pointer1 + _i) -    \
                    *(_data_pointer2 + _i) * _factor;                 \
              }                                                       \
           }
#endif

#ifndef SUB_PROD_ASSIGN
#   define SUB_PROD_ASSIGN(_data_pointer1, _data_pointer2,            \
                           _data_pointer3, _data_number, _factor) {   \
              int _i;                                                 \
              for(_i=0; _i<(_data_number); _i++) {                    \
                 *(_data_pointer1 + _i) = *(_data_pointer2 + _i) -    \
                    *(_data_pointer3 + _i) * _factor;                 \
              }                                                       \
           }
#endif

#ifndef SCALAR_PROD
#   define SCALAR_PROD(_data_pointer1, _data_pointer2, _data_number, \
                       _res_pointer) {                           \
              int _i;                                            \
              *(_res_pointer) = 0.0;                             \
              for(_i=0; _i<(_data_number); _i++) {               \
                 *(_res_pointer) +=                              \
                    *(_data_pointer1+_i) * *(_data_pointer2+_i); \
              }                                                  \
           }
#endif

#ifndef MIN_INT
#   define MIN_INT(_int_pointer, _int_number, _res_pointer) { \
              int _i;                                         \
              *(_res_pointer) = *(_int_pointer);              \
              for(_i=1; _i<(_int_number); _i++) {             \
                 if(*(_int_pointer+_i)<*(_res_pointer)) {     \
		    *(_res_pointer) = *((_int_pointer)+_i);   \
                 }                                            \
              }                                               \
           }
#endif

#ifndef ABS_MAX_DBL
#   define ABS_MAX_DBL(_dbl_pointer, _dbl_number, _res_pointer) {   \
              int _i;                                               \
              *(_res_pointer) = fabs(*(_dbl_pointer));              \
              for(_i=1; _i<(_dbl_number); _i++) {                   \
                 if(*(_res_pointer) < fabs(*((_dbl_pointer)+_i))) { \
		    *(_res_pointer) = fabs(*((_dbl_pointer)+_i));   \
                 }                                                  \
              }                                                     \
           }
#endif

int uwerr (char*);


#endif
