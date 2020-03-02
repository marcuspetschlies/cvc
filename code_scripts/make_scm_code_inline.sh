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

echo "static inline void _co_eq_tr_scm( ${SCM_TYPE} * _t, ${SCM_TYPE} ** _r ) {"
echo "  *_t = 0.;"
for(( ialpha=0; ialpha<${N_SPINOR_DIM}; ialpha++ )); do
for(( ia=0; ia<${N_COLOR_DIM}; ia++ )); do
  ka=$(( $N_COLOR_DIM * $ialpha + $ia))
  echo "  *_t += _r[$(fmt2 $ka)][$(fmt2 $ka)];"
done
done

echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------
echo "static inline void _scm_eq_scm( ${SCM_TYPE} ** _r,  ${SCM_TYPE} ** _s ) {"
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

echo "static inline void _scm_pl_eq_gamma_ti_scm_ti_gamma( ${SCM_TYPE} ** _r, ${SCM_TYPE} ** _gl, ${SCM_TYPE} ** _s, ${SCM_TYPE} ** _gr ) {"
#  echo "  memset ( _r[0], 0, $(( $N_SPINOR_DIM * $N_SPINOR_DIM * $N_COLOR_DIM * $N_COLOR_DIM )) * sizeof( ${SCM_TYPE} ) );"
for(( ialpha=0; ialpha<${N_SPINOR_DIM}; ialpha++ )); do
for(( ia=0; ia<${N_COLOR_DIM}; ia++ )); do
  ka=$(( $N_COLOR_DIM * $ialpha + $ia))

  for(( ibeta=0; ibeta<${N_SPINOR_DIM}; ibeta++ )); do
  for(( ib=0; ib<${N_COLOR_DIM}; ib++ )); do
    kb=$(( $N_COLOR_DIM * $ibeta + $ib ))

    for(( igamma=0; igamma<${N_SPINOR_DIM}; igamma++ )); do
      kc=$(( $N_COLOR_DIM * $igamma + $ia ))
      for(( idelta=0; idelta<${N_SPINOR_DIM}; idelta++ )); do
      kd=$(( $N_COLOR_DIM * $idelta + $ib ))

      echo "  _r[$(fmt2 $ka)][$(fmt2 $kb)] += _gl[$ialpha][$igamma] * _s[$(fmt2 $kc)][$(fmt2 $kd)] * _gr[$idelta][$ibeta];"

    done
    done
  done
  done
done
done #  end of loop on ia, ialpha
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _scm_eq_gamma_ti_scm_adj_ti_gamma( ${SCM_TYPE} ** _r, ${SCM_TYPE} ** _gl, ${SCM_TYPE} ** _s, ${SCM_TYPE} ** _gr ) {"
  echo "  memset ( _r[0], 0, $(( $N_SPINOR_DIM * $N_SPINOR_DIM * $N_COLOR_DIM * $N_COLOR_DIM )) * sizeof( ${SCM_TYPE} ) );"
for(( ialpha=0; ialpha<${N_SPINOR_DIM}; ialpha++ )); do
for(( ia=0; ia<${N_COLOR_DIM}; ia++ )); do
  ka=$(( $N_COLOR_DIM * $ialpha + $ia))

  for(( ibeta=0; ibeta<${N_SPINOR_DIM}; ibeta++ )); do
  for(( ib=0; ib<${N_COLOR_DIM}; ib++ )); do
    kb=$(( $N_COLOR_DIM * $ibeta + $ib ))

    for(( igamma=0; igamma<${N_SPINOR_DIM}; igamma++ )); do
      kc=$(( $N_COLOR_DIM * $igamma + $ia ))
      for(( idelta=0; idelta<${N_SPINOR_DIM}; idelta++ )); do
      kd=$(( $N_COLOR_DIM * $idelta + $ib ))

      echo "  _r[$(fmt2 $ka)][$(fmt2 $kb)] += _gl[$ialpha][$igamma] * conj( _s[$(fmt2 $kd)][$(fmt2 $kc)] ) * _gr[$idelta][$ibeta];"

    done
    done
  done
  done
done
done #  end of loop on ia, ialpha
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "#endif"

exit 0

