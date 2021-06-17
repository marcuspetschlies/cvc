#######################################################################
#
# make_sp_code_inline.sh
#
# Wed Dec 14 14:08:04 EET 2011
#
#######################################################################
#!/bin/bash

source dimensions.in
echo "#ifndef _SP_LINALG_INLINE_H"
echo "#define _SP_LINALG_INLINE_H"

echo "#include <math.h>"
echo "#include \"cvc_complex.h\""
echo "#include \"cvc_linalg.h\""
echo ""
echo "// spinor dimension: $N_SPINOR_DIM"
echo "// color dimension: $N_COLOR_DIM"
echo ""


# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _sp_eq_zero(spinor_propagator_type _r) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] = 0.;"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] = 0.;"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _sp_eq_sp(spinor_propagator_type _r, spinor_propagator_type _s) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] = (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))];"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] = (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))];"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _sp_eq_sp_transposed(spinor_propagator_type _r, spinor_propagator_type _s) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] = (_s)[$(printf "%2d" $alpha)][$(printf "%2d" $((2*$beta)))];"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] = (_s)[$(printf "%2d" $alpha)][$(printf "%2d" $((2*$beta+1)))];"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _sp_pl_eq_sp(spinor_propagator_type _r, spinor_propagator_type _s) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] += (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))];"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] += (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))];"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _sp_mi_eq_sp(spinor_propagator_type _r, spinor_propagator_type _s) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] -= (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))];"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] -= (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))];"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _sp_eq_sp_ti_re(spinor_propagator_type _r, spinor_propagator_type _s, double _c) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] = (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] * (_c);"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] = (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] * (_c);"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _sp_pl_eq_sp_ti_re(spinor_propagator_type _r, spinor_propagator_type _s, double _c) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] += (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] * (_c);"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] += (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] * (_c);"
  done
done
echo -e "}\n\n"


# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _sp_eq_sp_ti_im(spinor_propagator_type _r, spinor_propagator_type _s, double _c) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] = -(_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] * (_c);"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] = +(_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] * (_c);"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _sp_pl_eq_sp_ti_im(spinor_propagator_type _r, spinor_propagator_type _s, double _c) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] += -(_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] * (_c);"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] += +(_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] * (_c);"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _sp_eq_sp_ti_co(spinor_propagator_type _r, spinor_propagator_type _s, complex _c) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] = +(_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] * (_c).re - (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] * (_c).im;"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] = +(_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] * (_c).im + (_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] * (_c).re;"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------


echo "static inline void _sp_eq_gamma_ti_sp(spinor_propagator_type _r, int _mu, spinor_propagator_type _s) {"
echo "int __perm[$N_SPINOR_DIM], __isimag;"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  echo "__perm[$(printf "%2d" $beta)] = gamma_permutation[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$beta)))]/$((2*N_COLOR_DIM));"
done
echo "__isimag = gamma_permutation[(_mu)][0]%2;"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] = (_s)[$(printf "%2d" $beta)][2*__perm[$(printf "%2d" $alpha)]  +__isimag] * gamma_sign[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$alpha)))];"
    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] = (_s)[$(printf "%2d" $beta)][2*__perm[$(printf "%2d" $alpha)]+1-__isimag] * gamma_sign[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$alpha+1)))];"
  done
done
echo -e "}\n\n"

#echo "static inline void _sp_eq_gamma_ti_sp(spinor_propagator_type _r, int _mu, spinor_propagator_type _s) {"
#echo "int __perm[$N_SPINOR_DIM], __isimag;"
#for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
#  echo "__perm[$(printf "%2d" $beta)] = gamma_permutation[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$beta)))]/$((2*N_COLOR_DIM));"
#done
#echo "__isimag = gamma_permutation[(_mu)][0]%2;"
#echo "__one_mi_isimag = gamma_permutation[(_mu)][0]%2;"
#for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
#  for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
#    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha)))] = (_s)[$(printf "%2d" $beta)][2*__perm[$(printf "%2d" $alpha)]  +__isimag] * gamma_sign[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$alpha)))];"
#    echo "(_r)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$alpha+1)))] = (_s)[$(printf "%2d" $beta)][2*__perm[$(printf "%2d" $alpha)]+__one_mi_isimag] * gamma_sign[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$alpha+1)))];"
#  done
#done
#echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------


echo "static inline void _co_eq_tr_sp(complex* _c, spinor_propagator_type _s) {"
echo "(_c)->re = "
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  echo " +(_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$beta)))]"
done
echo ";"
echo "(_c)->im = "
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  echo " +(_s)[$(printf "%2d" $beta)][$(printf "%2d" $((2*$beta+1)))]"
done
echo ";"
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "// create, free spin propagator"
echo "static inline void create_sp(spinor_propagator_type *fp) {"
echo "  int i;"
echo "  *fp = (double**) malloc($N_SPINOR_DIM * sizeof(double*));"
echo "  if( *fp == NULL) return;"
echo "  (*fp)[0] = (double*)malloc($(($N_SPINOR_DIM*$N_SPINOR_DIM*2)) * sizeof(double));"
echo "  if ( (*fp)[0] == NULL) {"
echo "    free( *fp );"
echo "    *fp = NULL;"
echo "    return;"
echo "  }"
for((i=1;i<$N_SPINOR_DIM;i++)) ; do
  echo "(*fp)[$(printf "%2d" $i)] = (*fp)[$(printf "%2d" $(($i-1)))] + $(($N_SPINOR_DIM*2));"
done
echo "  _sp_eq_zero( *fp );"
echo "  return;"
echo -e "}\n\n"

echo "static inline void free_sp(spinor_propagator_type *fp) {"
echo "  int i;"
echo "  if( *fp != NULL ) {"
echo "    if( (*fp)[0] != NULL) free( (*fp)[0] );"
echo "    free(*fp);"
echo "    *fp = NULL;"
echo "  }"
echo "  return;"
echo -e "}\n\n"


# ----------------------------------------------------------------------------------------------------------------

echo "#endif"
exit 0


