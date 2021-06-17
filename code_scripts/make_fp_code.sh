#!/bin/bash

source dimensions.in
echo "#ifndef _FP_LINALG_H"
echo "#define _FP_LINALG_H"

echo "#include <math.h>"
echo "#include \"cvc_complex.h\""
echo "#include \"cvc_linalg.h\""

echo "// spinor dimension: $N_SPINOR_DIM"
echo "// color dimension: $N_COLOR_DIM"
echo ""


# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_eq_fp(_r, _s) {\\"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$s1+$a))
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = (_s)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))];\\"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = (_s)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))];\\"
      done
    done
  done
done
echo -e "}\n\n"


# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_eq_zero(_r) {\\"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$s1+$a))
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = 0.;\\"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = 0.;\\"
      done
    done
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_pl_eq_fp(_r, _s) {\\"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_pl_eq_fv( (_r)[$(printf "%2d" $j)], (_s)[$(printf "%2d" $j)] );\\"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_mi_eq_fp(_r, _s) {\\"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_mi_eq_fv( (_r)[$(printf "%2d" $j)], (_s)[$(printf "%2d" $j)] );\\"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_eq_fp_ti_re(_r, _s, _c) {\\"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_eq_fv_ti_re( (_r)[$(printf "%2d" $j)], (_s)[$(printf "%2d" $j)], (_c) );\\"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_eq_fp_ti_im(_r, _s, _c) {\\"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_eq_fv_ti_im( (_r)[$(printf "%2d" $j)], (_s)[$(printf "%2d" $j)], (_c) );\\"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_eq_fp_spin_transposed(_r, _s) {\\"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$s1+$a))
        k=$(($N_COLOR_DIM*$s1+$b))
        l=$(($N_COLOR_DIM*$s2+$a))
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = (_s)[$(printf "%2d" $k)][$(printf "%2d" $((2*$l)))];\\"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = (_s)[$(printf "%2d" $k)][$(printf "%2d" $((2*$l+1)))];\\"
      done
    done
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_eq_gamma_ti_fp(_r, _mu, _s) {\\"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_eq_gamma_ti_fv( (_r)[$(printf "%2d" $j)], (_mu), (_s)[$(printf "%2d" $j)] );\\"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_eq_gamma_transposed_ti_fp(_r, _mu, _s) {\\"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_eq_gamma_transposed_ti_fv( (_r)[$(printf "%2d" $j)], (_mu), (_s)[$(printf "%2d" $j)] );\\"
  done
done
echo -e "}\n\n"


# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_eq_fp_ti_gamma(_r, _mu, _s) {\\"
echo "int __perm, __isimag;\\"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  echo "__perm = gamma_permutation[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$beta)))]/$((2*$N_COLOR_DIM));\\"
  echo "__isimag = gamma_permutation[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$beta)))]%2;\\"
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))

    for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$alpha+$a))

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = (_s)[$N_COLOR_DIM*(__perm)+$b][$(printf "%2d" $((2*$i)))+(__isimag)] * gamma_sign[(_mu)][2*($N_COLOR_DIM*(__perm)+$b)];\\"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = (_s)[$N_COLOR_DIM*(__perm)+$b][$(printf "%2d" $((2*$i+1)))-(__isimag)] * gamma_sign[(_mu)][2*($N_COLOR_DIM*(__perm)+$b)+1];\\"

      done
    done
  done
done
echo -e "}\n\n"


# ----------------------------------------------------------------------------------------------------------------

echo "#define _fp_eq_fp_ti_gamma_transposed(_r, _mu, _s) {\\"
echo "int __perm, __isimag;\\"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  echo "__perm = gamma_permutation[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$beta)))]/$((2*$N_COLOR_DIM));\\"
  echo "__isimag = gamma_permutation[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$beta)))]%2;\\"

  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))
 
    for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$alpha+$a))

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = (_s)[$N_COLOR_DIM*(__perm)+$b][$(printf "%2d" $((2*$i)))+(__isimag)] * gamma_sign[(_mu)][$(printf "%2d" $((2*$j)))];\\"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = (_s)[$N_COLOR_DIM*(__perm)+$b][$(printf "%2d" $((2*$i+1)))-(__isimag)] * gamma_sign[(_mu)][$(printf "%2d" $((2*$j+1)))];\\"

      done
    done
  done
done

echo -e "}\n\n"


# -----------------------------------------------------------------------------------------------------

echo "#define _fp_eq_Cg5_ti_fp(_r, _s, _w) {\\"
echo "  _fp_eq_gamma_ti_fp( (_r), 5, (_s ));\\"
echo "  _fp_eq_gamma_ti_fp( (_w), 2, (_r ));\\"
echo "  _fp_eq_gamma_ti_fp( (_r), 0, (_w ));\\"
echo -e "}\n\n"

# -----------------------------------------------------------------------------------------------------

echo "#define _fp_eq_fp_ti_Cg5(_r, _s, _w) { \\"
echo "  _fp_eq_fp_ti_gamma( (_r), 0, (_s) );\\"
echo "  _fp_eq_fp_ti_gamma( (_w), 2, (_r) );\\"
echo "  _fp_eq_fp_ti_gamma( (_r), 5, (_w) );\\"
echo -e "}\n\n"


# -----------------------------------------------------------------------------------------------------

echo "#define _fp_eq_rot_ti_fp( _r, _s, _sign, _type, _w) {\\"
echo "  if( (_type) == _TM_FERMION ) {\\"
echo "    _fp_eq_gamma_ti_fp( (_r), 5, (_s) );\\"
echo "    _fp_eq_fp_ti_im( (_w), (_r), (_sign)*1.);\\"
echo "    _fp_pl_eq_fp( (_w), (_s) );\\"
echo "    _fp_eq_fp_ti_re( (_r), (_w), _ONE_OVER_SQRT2 );\\"
echo "  } else if ( (_type) == _WILSON_FERMION ) {\\"
echo "    _fp_eq_fp( (_r), (_s));\\"
echo "  } \\"
echo -e "}\n\n"

# -----------------------------------------------------------------------------------------------------

echo "#define _fp_eq_fp_ti_rot( _r, _s, _sign, _type, _w) {\\"
echo "  if( (_type) == _WILSON_FERMION )  {\\"
echo "    _fp_eq_fp( (_r), (_s) );\\"
echo "  } else if( (_type) == _TM_FERMION ) {\\"
echo "    _fp_eq_fp_ti_gamma( (_r), 5, (_s) );\\"
echo "    _fp_eq_fp_ti_im( (_w), (_r), (_sign)*1.);\\"
echo "    _fp_pl_eq_fp( (_w), (_s) );\\"
echo "    _fp_eq_fp_ti_re( (_r), (_w), _ONE_OVER_SQRT2 );\\"
echo "  }\\"
echo -e "}\n\n"


# -----------------------------------------------------------------------------------------------------

echo "#define _assign_fp_point_from_field(_r, _g, _i) {\\"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))
    echo "  _fv_eq_fv( (_r)[$j], (_g)[$j]+_GSI((_i)));\\"
  done
done
echo -e "}\n\n"

# -----------------------------------------------------------------------------------------------------

echo "#define _assign_fp_field_from_point(_g, _r, _i) {\\"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))
    echo "  _fv_eq_fv( (_g)[$j]+_GSI((_i)), (_r)[$j] );\\"
  done
done
echo -e "}\n\n"

echo "#endif"

exit 0
