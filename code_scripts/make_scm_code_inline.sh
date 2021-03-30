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
echo "#ifndef _SCM_LINALG_INLINE_H"
echo "#define _SCM_LINALG_INLINE_H"

echo "#include <math.h>"
echo "#include <complex.h>"
echo ""
echo "/* spinor dimension: $N_SPINOR_DIM */"
echo "/* color dimension: $N_COLOR_DIM */"
echo ""

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

echo "#endif"

exit 0

