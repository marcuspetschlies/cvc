#######################################################################
#
# make_contractv123_inline.sh
#
# Mon May 22 16:15:27 CEST 2017
#
#######################################################################
#!/bin/bash
# source dimensions.in
N_SPINOR_DIM=4
N_COLOR_DIM=3

perm_tab_3[0]=1
perm_tab_3[1]=2
perm_tab_3[2]=0
#
perm_tab_3[3]=2
perm_tab_3[4]=0
perm_tab_3[5]=1
#
perm_tab_3[6]=0
perm_tab_3[7]=1
perm_tab_3[8]=2
#

if [ 1 -eq 0 ]; then

echo "#ifndef _CONTRACTVN_INLINE_H"
echo "#define _CONTRACTVN_INLINE_H"

echo "// spinor dimension: $N_SPINOR_DIM"
echo "// color dimension: $N_COLOR_DIM"
echo ""

echo -e "namespace cvc {\n\n"

fi

if [ 1 -eq 0 ]; then

echo "static inline void _v1_eq_fv_eps_fp( double *_v1, double *_fv, fermion_propagator_type _fp) {"
echo "  double _cre, _cim;"
for((m=0; m<$N_COLOR_DIM; m++ )); do

  for((p=0; p<$N_COLOR_DIM; p++)); do
    c=${perm_tab_3[$((3*$p+0))]}
    b=${perm_tab_3[$((3*$p+1))]}
    a=${perm_tab_3[$((3*$p+2))]}

    for((alpha2=0; alpha2<$N_SPINOR_DIM;  alpha2++)); do
      echo "  _cre = 0.;"
      echo "  _cim = 0.;"

      for((alpha1=0; alpha1<$N_SPINOR_DIM;  alpha1++)); do
        echo -e "  _cre += "
        echo -e "      _fv[$((2*($N_COLOR_DIM*$alpha1+$c)  ))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$b)  ))] "
        echo -e "    - _fv[$((2*($N_COLOR_DIM*$alpha1+$c)+1))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$b)+1))] "
        echo -e "    - _fv[$((2*($N_COLOR_DIM*$alpha1+$b)  ))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$c)  ))] "
        echo -e "    + _fv[$((2*($N_COLOR_DIM*$alpha1+$b)+1))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$c)+1))];"

        echo -e "  _cim += "
        echo -e "    + _fv[$((2*($N_COLOR_DIM*$alpha1+$c)  ))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$b)+1))] "
        echo -e "    + _fv[$((2*($N_COLOR_DIM*$alpha1+$c)+1))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$b)  ))] "
        echo -e "    - _fv[$((2*($N_COLOR_DIM*$alpha1+$b)  ))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$c)+1))] "
        echo -e "    - _fv[$((2*($N_COLOR_DIM*$alpha1+$b)+1))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$c)  ))];"
      done
      echo "  _v1[$(( 2* ( $N_SPINOR_DIM * ( $N_COLOR_DIM * $a + $m) +  $alpha2 )   ))] = _cre;"
      echo "  _v1[$(( 2* ( $N_SPINOR_DIM * ( $N_COLOR_DIM * $a + $m) +  $alpha2 )+1 ))] = _cim;"
    done
  done
done
echo -e "}   /* end of _v1_eq_fv_eps_fp */\n\n"


echo "static inline void _v2_eq_v1_eps_fp( double *_v2, double *_v1, fermion_propagator_type _fp) {"

echo "  double _cre, _cim;"
for((alpha1=0; alpha1<$N_SPINOR_DIM;  alpha1++)); do
  for((alpha2=0; alpha2<$N_SPINOR_DIM;  alpha2++)); do
    for((alpha3=0; alpha3<$N_SPINOR_DIM;  alpha3++)); do

      for((p=0; p<$N_COLOR_DIM; p++)); do
        m=${perm_tab_3[$((3*$p+0))]}
        l=${perm_tab_3[$((3*$p+1))]}
        n=${perm_tab_3[$((3*$p+2))]}

        echo "  _cre = 0.;"
        echo "  _cim = 0.;"
        for((a=0; a<$N_COLOR_DIM; a++ )); do

          echo -e "  _cre += "
          echo -e "    _v1[$(( 2 * ( $N_SPINOR_DIM * ( $N_COLOR_DIM * $a + $m ) + $alpha1 )  ))] * _fp[$(($N_COLOR_DIM*$alpha3+$l))][$((2*($N_COLOR_DIM*$alpha2+$a)  ))] "
          echo -e "  - _v1[$(( 2 * ( $N_SPINOR_DIM * ( $N_COLOR_DIM * $a + $m ) + $alpha1 )+1))] * _fp[$(($N_COLOR_DIM*$alpha3+$l))][$((2*($N_COLOR_DIM*$alpha2+$a)+1))] "
          echo -e "  - _v1[$(( 2 * ( $N_SPINOR_DIM * ( $N_COLOR_DIM * $a + $l ) + $alpha1 )  ))] * _fp[$(($N_COLOR_DIM*$alpha3+$m))][$((2*($N_COLOR_DIM*$alpha2+$a)  ))] "
          echo -e "  + _v1[$(( 2 * ( $N_SPINOR_DIM * ( $N_COLOR_DIM * $a + $l ) + $alpha1 )+1))] * _fp[$(($N_COLOR_DIM*$alpha3+$m))][$((2*($N_COLOR_DIM*$alpha2+$a)+1))];"

          echo -e "  _cim += "
          echo -e "  + _v1[$(( 2 * ( $N_SPINOR_DIM * ( $N_COLOR_DIM * $a + $m ) + $alpha1 )  ))] * _fp[$(($N_COLOR_DIM*$alpha3+$l))][$((2*($N_COLOR_DIM*$alpha2+$a)+1))] "
          echo -e "  + _v1[$(( 2 * ( $N_SPINOR_DIM * ( $N_COLOR_DIM * $a + $m ) + $alpha1 )+1))] * _fp[$(($N_COLOR_DIM*$alpha3+$l))][$((2*($N_COLOR_DIM*$alpha2+$a)  ))] "
          echo -e "  - _v1[$(( 2 * ( $N_SPINOR_DIM * ( $N_COLOR_DIM * $a + $l ) + $alpha1 )  ))] * _fp[$(($N_COLOR_DIM*$alpha3+$m))][$((2*($N_COLOR_DIM*$alpha2+$a)+1))] "
          echo -e "  - _v1[$(( 2 * ( $N_SPINOR_DIM * ( $N_COLOR_DIM * $a + $l ) + $alpha1 )+1))] * _fp[$(($N_COLOR_DIM*$alpha3+$m))][$((2*($N_COLOR_DIM*$alpha2+$a)  ))];"
        done
        echo "  _v2[$(( 2* ( $N_COLOR_DIM * (  $N_SPINOR_DIM * ( $N_SPINOR_DIM * $alpha1 + $alpha2) +  $alpha3 ) + $n )   ))] = _cre;"
        echo "  _v2[$(( 2* ( $N_COLOR_DIM * (  $N_SPINOR_DIM * ( $N_SPINOR_DIM * $alpha1 + $alpha2) +  $alpha3 ) + $n )+1 ))] = _cim;"
      done
    done
  done
done
echo -e "}   /* end of _v2_eq_v1_eps_fp */\n\n"

echo "static inline void _v3_eq_fv_dot_fp( double *_v3, double *_fv, fermion_propagator_type _fp) {"
echo "  double _cre, _cim;"
for((alpha2=0; alpha2<$N_SPINOR_DIM;  alpha2++)); do
  for((m=0; m<$N_COLOR_DIM; m++ )); do

    echo "  _cre = 0.;"
    echo "  _cim = 0.;"

    for((alpha1=0; alpha1<$N_SPINOR_DIM;  alpha1++)); do
      for((l=0; l<$N_COLOR_DIM; l++ )); do
        echo -e "  _cre += "
        echo -e "    + _fv[$((2*($N_COLOR_DIM*$alpha1+$l)  ))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$l)  ))] "
        echo -e "    + _fv[$((2*($N_COLOR_DIM*$alpha1+$l)+1))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$l)+1))];"

        echo -e "  _cim += "
        echo -e "    + _fv[$((2*($N_COLOR_DIM*$alpha1+$l)  ))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$l)+1))] "
        echo -e "    - _fv[$((2*($N_COLOR_DIM*$alpha1+$l)+1))] * _fp[$(($N_COLOR_DIM*$alpha2+$m))][$((2*($N_COLOR_DIM*$alpha1+$l)  ))];"
      done
    done
    echo "  _v3[$(( 2 * ( $N_COLOR_DIM * $alpha2 + $m )   ))] = _cre;"
    echo "  _v3[$(( 2 * ( $N_COLOR_DIM * $alpha2 + $m )+1 ))] = _cim;"
  done
done
echo "}   /* end of _v3_eq_fv_dot_fp */"

fi


echo "static inline void _v4_eq_fv_dot_fp( double *_v4, double *_fv, fermion_propagator_type _fp) {"
echo "  double _cre, _cim;"
for((l=0; l<$N_COLOR_DIM; l++ )); do
for((gamma=0; gamma<$N_SPINOR_DIM;  gamma++)); do
for((beta=0; beta<$N_SPINOR_DIM;  beta++)); do
for((alpha=0; alpha<$N_SPINOR_DIM;  alpha++)); do

    echo "  _cre = 0.;"
    echo "  _cim = 0.;"

    for((a=0; a<$N_COLOR_DIM; a++ )); do
        echo -e "  _cre += "
        echo -e "    + _fv[$((2*($N_COLOR_DIM*$alpha+$a)  ))] * _fp[$(($N_COLOR_DIM*$gamma+$l))][$((2*($N_COLOR_DIM*$beta+$a)  ))] "
        echo -e "    - _fv[$((2*($N_COLOR_DIM*$alpha+$a)+1))] * _fp[$(($N_COLOR_DIM*$gamma+$l))][$((2*($N_COLOR_DIM*$beta+$a)+1))];"

        echo -e "  _cim += "
        echo -e "    + _fv[$((2*($N_COLOR_DIM*$alpha+$a)  ))] * _fp[$(($N_COLOR_DIM*$gamma+$l))][$((2*($N_COLOR_DIM*$beta+$a)+1))] "
        echo -e "    + _fv[$((2*($N_COLOR_DIM*$alpha+$a)+1))] * _fp[$(($N_COLOR_DIM*$gamma+$l))][$((2*($N_COLOR_DIM*$beta+$a)  ))];"
    done

    echo "  _v4[$(( 2* ( $N_COLOR_DIM * (  $N_SPINOR_DIM * ( $N_SPINOR_DIM * $alpha + $beta) +  $gamma ) + $l )   ))] = _cre;"
    echo "  _v4[$(( 2* ( $N_COLOR_DIM * (  $N_SPINOR_DIM * ( $N_SPINOR_DIM * $alpha + $beta) +  $gamma ) + $l )+1 ))] = _cim;"
done
done
done
done
echo "}   /* end of _v4_eq_fv_dot_fp */"

if [ 1 -eq 0 ]; then

echo -e "\n}  /* end of namespace cvc */"
echo -e "\n#endif"

fi
exit 0
