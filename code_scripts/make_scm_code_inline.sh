#######################################################################
#
# make_scm_code_inline.sh
#
#######################################################################
#!/bin/bash

function fmt2 {
  printf "%2d" $1
}

source dimensions.in
SCM_TYPE="double _Complex"
cat << EOF
#ifndef _SCM_LINALG_INLINE_H
#define _SCM_LINALG_INLINE_H


#include <math.h>
#include <complex.h>

/* spinor dimension: $N_SPINOR_DIM */
/* color dimension: $N_COLOR_DIM */

namespace cvc {

EOF

echo "static inline void _co_eq_tr_scm( ${SCM_TYPE} * const _t, ${SCM_TYPE} ** const _r ) {"
echo "  ${SCM_TYPE} z = 0.;"
for(( ialpha=0; ialpha<${N_SPINOR_DIM}; ialpha++ )); do
for(( ia=0; ia<${N_COLOR_DIM}; ia++ )); do
  ka=$(( $N_COLOR_DIM * $ialpha + $ia))
  echo "  z += _r[$(fmt2 $ka)][$(fmt2 $ka)];"
done
done
echo "  *_t = z;"

echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _co_eq_tr_scm_ti_gamma( ${SCM_TYPE} * const _t, ${SCM_TYPE} ** const _r, ${SCM_TYPE} ** const _g ) {"
echo "  ${SCM_TYPE} z = 0.;"
for(( ialpha=0; ialpha<${N_SPINOR_DIM}; ialpha++ )); do
for(( ia=0; ia<${N_COLOR_DIM}; ia++ )); do
  ka=$(( $N_COLOR_DIM * $ialpha + $ia))
for(( ibeta=0; ibeta<${N_SPINOR_DIM}; ibeta++ )); do
  kb=$(( $N_COLOR_DIM * $ibeta + $ia))
  echo "  z += _r[$(fmt2 $ka)][$(fmt2 $kb)] * _g[$ibeta][$ialpha];"
done
done
done
echo "  *_t = z;"

echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _scm_eq_scm( ${SCM_TYPE} ** const _r,  ${SCM_TYPE} ** const _s ) {"
for(( ialpha=0; ialpha<${N_SPINOR_DIM}; ialpha++ )); do
for(( ia=0; ia<${N_COLOR_DIM}; ia++ )); do
  ka=$(( $N_COLOR_DIM * $ialpha + $ia))

  for(( ibeta=0; ibeta<${N_SPINOR_DIM}; ibeta++ )); do
  for(( ib=0; ib<${N_COLOR_DIM}; ib++ )); do
    kb=$(( $N_COLOR_DIM * $ibeta + $ib ))
    
    echo "  _r[$(fmt2 $ka)][$(fmt2 $kb)] = _s[$(fmt2 $ka)][$(fmt2 $kb)];"

  done
  done
done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _scm_pl_eq_gamma_ti_scm_ti_gamma( ${SCM_TYPE} ** const _r, ${SCM_TYPE} ** const _gl, ${SCM_TYPE} ** const _s, ${SCM_TYPE} ** const _gr ) {"
#  echo "  memset ( _r[0], 0, $(( $N_SPINOR_DIM * $N_SPINOR_DIM * $N_COLOR_DIM * $N_COLOR_DIM )) * sizeof( ${SCM_TYPE} ) );"
echo "  ${SCM_TYPE} z;"
for(( ialpha=0; ialpha<${N_SPINOR_DIM}; ialpha++ )); do
for(( ia=0; ia<${N_COLOR_DIM}; ia++ )); do
  ka=$(( $N_COLOR_DIM * $ialpha + $ia))

  for(( ibeta=0; ibeta<${N_SPINOR_DIM}; ibeta++ )); do
  for(( ib=0; ib<${N_COLOR_DIM}; ib++ )); do
    kb=$(( $N_COLOR_DIM * $ibeta + $ib ))

    echo "  z = 0.;"
    for(( igamma=0; igamma<${N_SPINOR_DIM}; igamma++ )); do
      kc=$(( $N_COLOR_DIM * $igamma + $ia ))
      for(( idelta=0; idelta<${N_SPINOR_DIM}; idelta++ )); do
      kd=$(( $N_COLOR_DIM * $idelta + $ib ))

      echo "  z += _gl[$ialpha][$igamma] * _s[$(fmt2 $kc)][$(fmt2 $kd)] * _gr[$idelta][$ibeta];"

    done  # of idelta
    done  # of igamma
    echo "  _r[$(fmt2 $ka)][$(fmt2 $kb)] += z;"
  done
  done
done
done #  end of loop on ia, ialpha
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _scm_eq_gamma_ti_scm_adj_ti_gamma( ${SCM_TYPE} ** const _r, ${SCM_TYPE} ** const _gl, ${SCM_TYPE} ** const _s, ${SCM_TYPE} ** const _gr ) {"
  echo "  memset ( _r[0], 0, $(( $N_SPINOR_DIM * $N_SPINOR_DIM * $N_COLOR_DIM * $N_COLOR_DIM )) * sizeof( ${SCM_TYPE} ) );"
  echo "  ${SCM_TYPE} z = 0.;"
for(( ialpha=0; ialpha<${N_SPINOR_DIM}; ialpha++ )); do
for(( ia=0; ia<${N_COLOR_DIM}; ia++ )); do
  ka=$(( $N_COLOR_DIM * $ialpha + $ia))

  for(( ibeta=0; ibeta<${N_SPINOR_DIM}; ibeta++ )); do
  for(( ib=0; ib<${N_COLOR_DIM}; ib++ )); do
    kb=$(( $N_COLOR_DIM * $ibeta + $ib ))

    echo "  z = 0.;"
    for(( igamma=0; igamma<${N_SPINOR_DIM}; igamma++ )); do
      kc=$(( $N_COLOR_DIM * $igamma + $ia ))
      for(( idelta=0; idelta<${N_SPINOR_DIM}; idelta++ )); do
      kd=$(( $N_COLOR_DIM * $idelta + $ib ))

      echo "  z += _gl[$ialpha][$igamma] * conj( _s[$(fmt2 $kd)][$(fmt2 $kc)] ) * _gr[$idelta][$ibeta];"

    done
    done
    echo "  _r[$(fmt2 $ka)][$(fmt2 $kb)] = z;"
  done
  done
done
done #  end of loop on ia, ialpha
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _scm_pl_eq_gamma_ti_scm_ti_gamma_perm ( ${SCM_TYPE} ** const  _r, int const _mul, ${SCM_TYPE} ** const _s, int const _mur ) {"
echo "  int const __lisimag = gamma_permutation[(_mul)][0] % 2;"
echo "  int const __risimag = gamma_permutation[(_mur)][0] % 2;"
echo "  ${SCM_TYPE} const __norm = (( 1 - (__lisimag) ) + (__lisimag)*I) * (( 1 - (__risimag) ) + (__risimag)*I);"
printf "  int const __lperm[$N_SPINOR_DIM] = {"
for((beta=0;beta<$(( $N_SPINOR_DIM -1)) ;beta++)); do printf "gamma_permutation[(_mul)][%2d]/%d , " $((2*$N_COLOR_DIM*$beta)) $((2*$N_COLOR_DIM)) ; done
printf "gamma_permutation[(_mul)][%2d]/%d };\n " $((2*$N_COLOR_DIM * ($N_SPINOR_DIM -1) )) $((2*$N_COLOR_DIM)) 

printf "  int const __rperm[$N_SPINOR_DIM] = {"
for((beta=0;beta<$(( $N_SPINOR_DIM -1)) ;beta++)); do printf "gamma_permutation[(_mur)][%2d]/%d , " $((2*$N_COLOR_DIM*$beta)) $((2*$N_COLOR_DIM)) ; done
printf "gamma_permutation[(_mur)][%2d]/%d };\n " $((2*$N_COLOR_DIM * ($N_SPINOR_DIM -1) )) $((2*$N_COLOR_DIM)) 

echo "  int __lid, __rid;"

for((beta=0;beta<$N_SPINOR_DIM;beta++)); do

  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))
    echo "  __lid = $N_COLOR_DIM * __lperm[$beta] + $b;"

    for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do

      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$alpha+$a))
        echo "  __rid = $N_COLOR_DIM * __rperm[$alpha] + $a;"

        # echo "  (_r)[$j][$i] += (_s)[__lid][__rid] * gamma_sign[(_mul)][2*(__lid)] * gamma_sign[(_mur)][2*(__rid)] * __norm;"
        echo "  (_r)[$j][$i] += (_s)[__lid][__rid] * gamma_sign[(_mul)][2*($j)+1] * gamma_sign[(_mur)][2*(__rid)+1] * __norm;"
      done
    done
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _co_eq_tr_gamma_ti_scm_perm ( ${SCM_TYPE} * const _t, ${SCM_TYPE} ** const _r, int const _mu ) {"
echo "  int const __isimag = gamma_permutation[(_mu)][0] % 2;"
echo "  ${SCM_TYPE} const __norm = (( 1 - (__isimag) ) + (__isimag)*I);"
printf "  int const __perm[$N_SPINOR_DIM] = {"
for((beta=0;beta<$(( $N_SPINOR_DIM -1)) ;beta++)); do printf "gamma_permutation[(_mu)][%2d]/%d , " $((2*$N_COLOR_DIM*$beta)) $((2*$N_COLOR_DIM)) ; done
printf "gamma_permutation[(_mu)][%2d]/%d };\n " $((2*$N_COLOR_DIM * ($N_SPINOR_DIM -1) )) $((2*$N_COLOR_DIM))

echo "  int __lid;"
echo "  ${SCM_TYPE} __tmp = 0.;"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do

  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))
    echo "  __lid = $N_COLOR_DIM * __perm[$beta] + $b;"
    echo "  __tmp += (_r)[__lid][$j] * gamma_sign[(_mu)][2*($j)+1];"
  done
done
echo "  *(_t) = __norm * __tmp;"
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _scm_eq_gamma_ti_scm_adj_ti_gamma_perm ( ${SCM_TYPE} ** const  _r, int const _mul, ${SCM_TYPE} ** const _s, int const _mur ) {"
echo "  int const __lisimag = gamma_permutation[(_mul)][0] % 2;"
echo "  int const __risimag = gamma_permutation[(_mur)][0] % 2;"
echo "  ${SCM_TYPE} const __norm = (( 1 - (__lisimag) ) + (__lisimag)*I) * (( 1 - (__risimag) ) + (__risimag)*I);"
printf "  int const __lperm[$N_SPINOR_DIM] = {"
for((beta=0;beta<$(( $N_SPINOR_DIM -1)) ;beta++)); do printf "gamma_permutation[(_mul)][%2d]/%d , " $((2*$N_COLOR_DIM*$beta)) $((2*$N_COLOR_DIM)) ; done
printf "gamma_permutation[(_mul)][%2d]/%d };\n " $((2*$N_COLOR_DIM * ($N_SPINOR_DIM -1) )) $((2*$N_COLOR_DIM))

printf "  int const __rperm[$N_SPINOR_DIM] = {"
for((beta=0;beta<$(( $N_SPINOR_DIM -1)) ;beta++)); do printf "gamma_permutation[(_mur)][%2d]/%d , " $((2*$N_COLOR_DIM*$beta)) $((2*$N_COLOR_DIM)) ; done
printf "gamma_permutation[(_mur)][%2d]/%d };\n " $((2*$N_COLOR_DIM * ($N_SPINOR_DIM -1) )) $((2*$N_COLOR_DIM))

echo "  int __lid, __rid;"

for((beta=0;beta<$N_SPINOR_DIM;beta++)); do

  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))
    echo "  __lid = $N_COLOR_DIM * __lperm[$beta] + $b;"

    for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do

      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$alpha+$a))
        echo "  __rid = $N_COLOR_DIM * __rperm[$alpha] + $a;"

        echo "  (_r)[$j][$i] = conj((_s)[__rid][__lid]) * gamma_sign[(_mul)][2*($j)+1] * gamma_sign[(_mur)][2*(__rid)+1] * __norm;"
      done
    done
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

cat << EOF

}

#endif
EOF

exit 0

