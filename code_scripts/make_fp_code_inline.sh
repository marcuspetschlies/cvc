#######################################################################
#
# make_fp_code_inline.sh
#
# Wed Dec 14 14:08:04 EET 2011
#
#######################################################################
#!/bin/bash

source dimensions.in
echo "#ifndef _FP_LINALG_INLINE_H"
echo "#define _FP_LINALG_INLINE_H"

echo "#include <math.h>"
echo "#include \"cvc_complex.h\""
echo "#include \"cvc_linalg.h\""
echo ""
echo "// spinor dimension: $N_SPINOR_DIM"
echo "// color dimension: $N_COLOR_DIM"
echo ""


# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_fp(fermion_propagator_type _r, fermion_propagator_type _s) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$s1+$a))
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = (_s)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))];"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = (_s)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))];"
      done
    done
  done
done
echo -e "}\n\n"


# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_zero(fermion_propagator_type _r) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$s1+$a))
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = 0.;"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = 0.;"
      done
    done
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_pl_eq_fp(fermion_propagator_type _r, fermion_propagator_type _s) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_pl_eq_fv( (_r)[$(printf "%2d" $j)], (_s)[$(printf "%2d" $j)] );"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_mi_eq_fp(fermion_propagator_type _r, fermion_propagator_type _s) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_mi_eq_fv( (_r)[$(printf "%2d" $j)], (_s)[$(printf "%2d" $j)] );"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_fp_ti_re(fermion_propagator_type _r, fermion_propagator_type _s, double _c) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_eq_fv_ti_re( (_r)[$(printf "%2d" $j)], (_s)[$(printf "%2d" $j)], (_c) );"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_ti_eq_re(fermion_propagator_type _r, double _c) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_ti_eq_re( (_r)[$(printf "%2d" $j)], (_c) );"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_fp_ti_im(fermion_propagator_type _r, fermion_propagator_type _s, double _c) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_eq_fv_ti_im( (_r)[$(printf "%2d" $j)], (_s)[$(printf "%2d" $j)], (_c) );"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_fp_spin_transposed(fermion_propagator_type _r, fermion_propagator_type _s) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$s1+$a))
        k=$(($N_COLOR_DIM*$s1+$b))
        l=$(($N_COLOR_DIM*$s2+$a))
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = (_s)[$(printf "%2d" $k)][$(printf "%2d" $((2*$l)))];"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = (_s)[$(printf "%2d" $k)][$(printf "%2d" $((2*$l+1)))];"
      done
    done
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_gamma_ti_fp(fermion_propagator_type _r, int _mu, fermion_propagator_type _s) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_eq_gamma_ti_fv( (_r)[$(printf "%2d" $j)], (_mu), (_s)[$(printf "%2d" $j)] );"
  done
done
echo -e "}\n\n"

# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_gamma_transposed_ti_fp(fermion_propagator_type _r, int _mu, fermion_propagator_type _s) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_eq_gamma_transposed_ti_fv( (_r)[$(printf "%2d" $j)], (_mu), (_s)[$(printf "%2d" $j)] );"
  done
done
echo -e "}\n\n"


# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_fp_ti_gamma(fermion_propagator_type _r, int _mu, fermion_propagator_type _s) {"
echo "int __perm, __isimag;"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  echo "__perm = gamma_permutation[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$beta)))]/$((2*$N_COLOR_DIM));"
  echo "__isimag = gamma_permutation[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$beta)))]%2;"
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))

    for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$alpha+$a))

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = (_s)[$N_COLOR_DIM*(__perm)+$b][$(printf "%2d" $((2*$i)))+(__isimag)] * gamma_sign[(_mu)][2*($N_COLOR_DIM*(__perm)+$b)];"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = (_s)[$N_COLOR_DIM*(__perm)+$b][$(printf "%2d" $((2*$i+1)))-(__isimag)] * gamma_sign[(_mu)][2*($N_COLOR_DIM*(__perm)+$b)+1];"

      done
    done
  done
done
echo -e "}\n\n"


# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_fp_ti_gamma_transposed(fermion_propagator_type _r, int _mu, fermion_propagator_type _s) {"
echo "int __perm, __isimag;"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  echo "__perm = gamma_permutation[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$beta)))]/$((2*$N_COLOR_DIM));"
  echo "__isimag = gamma_permutation[(_mu)][$(printf "%2d" $((2*$N_COLOR_DIM*$beta)))]%2;"

  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))
 
    for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$alpha+$a))

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = (_s)[$N_COLOR_DIM*(__perm)+$b][$(printf "%2d" $((2*$i)))+(__isimag)] * gamma_sign[(_mu)][$(printf "%2d" $((2*$j)))];"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = (_s)[$N_COLOR_DIM*(__perm)+$b][$(printf "%2d" $((2*$i+1)))-(__isimag)] * gamma_sign[(_mu)][$(printf "%2d" $((2*$j+1)))];"

      done
    done
  done
done

echo -e "}\n\n"


# ----------------------------------------------------------------------------------------------------------------

echo "static inline void _fp_pl_eq_fp_spin_transposed(fermion_propagator_type _r, fermion_propagator_type _s) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$s1+$a))

        k=$(($N_COLOR_DIM*$s1+$b))
        l=$(($N_COLOR_DIM*$s2+$a))
        
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] += (_s)[$(printf "%2d" $k)][$(printf "%2d" $((2*$l)))];"
        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] += (_s)[$(printf "%2d" $k)][$(printf "%2d" $((2*$l+1)))];"
      done
    done
  done
done
echo -e "}\n\n"

# -----------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_Cg5_ti_fp(fermion_propagator_type _r, fermion_propagator_type _s, fermion_propagator_type _w) {"
echo "  _fp_eq_gamma_ti_fp( (_r), 5, (_s ));"
echo "  _fp_eq_gamma_ti_fp( (_w), 2, (_r ));"
echo "  _fp_eq_gamma_ti_fp( (_r), 0, (_w ));"
echo -e "}\n\n"

# -----------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_fp_ti_Cg5(fermion_propagator_type _r, fermion_propagator_type _s, fermion_propagator_type _w) { "
echo "  _fp_eq_fp_ti_gamma( (_r), 0, (_s) );"
echo "  _fp_eq_fp_ti_gamma( (_w), 2, (_r) );"
echo "  _fp_eq_fp_ti_gamma( (_r), 5, (_w) );"
echo -e "}\n\n"


# -----------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_rot_ti_fp( fermion_propagator_type _r, fermion_propagator_type _s, double _sign, int _type, fermion_propagator_type _w) {"
echo "  if( (_type) == _TM_FERMION ) {"
echo "    _fp_eq_gamma_ti_fp( (_r), 5, (_s) );"
echo "    _fp_eq_fp_ti_im( (_w), (_r), (_sign)*1.);"
echo "    _fp_pl_eq_fp( (_w), (_s) );"
echo "    _fp_eq_fp_ti_re( (_r), (_w), _ONE_OVER_SQRT2 );"
echo "  } else if ( (_type) == _WILSON_FERMION ) {"
echo "    _fp_eq_fp( (_r), (_s));"
echo "  } "
echo -e "}\n\n"

# -----------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_fp_ti_rot( fermion_propagator_type _r, fermion_propagator_type _s, double _sign, int _type, fermion_propagator_type _w) {"
echo "  if( (_type) == _WILSON_FERMION )  {"
echo "    _fp_eq_fp( (_r), (_s) );"
echo "  } else if( (_type) == _TM_FERMION ) {"
echo "    _fp_eq_fp_ti_gamma( (_r), 5, (_s) );"
echo "    _fp_eq_fp_ti_im( (_w), (_r), (_sign)*1.);"
echo "    _fp_pl_eq_fp( (_w), (_s) );"
echo "    _fp_eq_fp_ti_re( (_r), (_w), _ONE_OVER_SQRT2 );"
echo "  }"
echo -e "}\n\n"


# -----------------------------------------------------------------------------------------------------

echo "static inline void _assign_fp_point_from_field(fermion_propagator_type _r, fermion_propagator_type _g, unsigned int _i) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))
    echo "  _fv_eq_fv( (_r)[$j], (_g)[$j]+_GSI((_i)));"
  done
done
echo -e "}\n\n"

# -----------------------------------------------------------------------------------------------------

echo "static inline void _assign_fp_field_from_point(fermion_propagator_type _g, fermion_propagator_type _r, unsigned int _i) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))
    echo "  _fv_eq_fv( (_g)[$j]+_GSI((_i)), (_r)[$j] );"
  done
done
echo -e "}\n\n"


echo "static inline void _project_fp_to_basis(double *b, fermion_propagator_type _r, int icol) {"
echo "int idx;"
echo "double spinor[$((2*$N_SPINOR_DIM*$N_COLOR_DIM))];"
echo "double norm = $(echo $N_SPINOR_DIM | awk '{printf("%25.16e\n", 1./$1)}');"
for((mu=0;mu<32;mu++)); do
  echo "b[$mu] = 0.;"
done
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
    echo "idx = $(($N_COLOR_DIM*$beta)) + icol;"
    for((mu=0;mu<16;mu++)); do
      echo "  _fv_eq_gamma_ti_fv( spinor, $mu, (_r)[idx] );"
      if [ $mu -lt 6 ]; then
        echo "b[$(printf "%2d" $((2*$mu)))] +=  spinor[2*idx  ];"
        echo "b[$(printf "%2d" $((2*$mu+1)))] +=  spinor[2*idx+1];"
      else
        echo "b[$(printf "%2d" $((2*$mu)))] -=  spinor[2*idx  ];"
        echo "b[$(printf "%2d" $((2*$mu+1)))] -=  spinor[2*idx+1];"
      fi
    done
done
for((mu=0;mu<32;mu++)); do
  echo "b[$mu] *= norm;"
done
echo -e "}\n\n"


echo "static inline void _fp_eq_cm_ti_fp(fermion_propagator_type _r, double *_c, fermion_propagator_type _s) {"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "_fv_eq_cm_ti_fv( (_r)[$(printf "%2d" $j)], (_c), (_s)[$(printf "%2d" $j)] );"
  done
done
echo -e "}\n\n"


echo "static inline void _fp_eq_fp_ti_cm_dagger(fermion_propagator_type _r, double *_c, fermion_propagator_type _s) {"
for((beta=0;beta<$N_SPINOR_DIM;beta++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$beta+$b))

    for((alpha=0;alpha<$N_SPINOR_DIM;alpha++)); do
      for((a=0;a<$N_COLOR_DIM;a++)); do
        i=$(($N_COLOR_DIM*$alpha+$a))

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i)))] = "
        for((c=0;c<$N_COLOR_DIM;c++)); do
          k=$(($N_COLOR_DIM*$beta+$c))
          echo "+(_s)[ $(printf "%2d" $k)][$(printf "%2d" $((2*$i)))] * (_c)[$(printf "%2d" $((6*$b+2*$c)))] + (_s)[ $(printf "%2d" $k)][$(printf "%2d" $((2*$i+1)))] * (_c)[$(printf "%2d" $((6*$b+2*$c+1)))]"
        done
        echo ";"

        echo "(_r)[$(printf "%2d" $j)][$(printf "%2d" $((2*$i+1)))] = "
        for((c=0;c<$N_COLOR_DIM;c++)); do
          k=$(($N_COLOR_DIM*$beta+$c))
          echo "+(_s)[ $(printf "%2d" $k)][$(printf "%2d" $((2*$i+1)))] * (_c)[$(printf "%2d" $((6*$b+2*$c)))] - (_s)[ $(printf "%2d" $k)][$(printf "%2d" $((2*$i)))] * (_c)[$(printf "%2d" $((6*$b+2*$c+1)))]"
        done
        echo ";"
      done
    done
  done
done
echo -e "}\n\n" 


echo "// create, free fermion propagator"
echo "static inline void create_fp(fermion_propagator_type*fp) {"
echo "  int i;"
echo "  *fp = (double**) malloc($(($N_SPINOR_DIM*$N_COLOR_DIM)) * sizeof(double*));"
echo "  if( *fp == NULL) return;"
echo "  (*fp)[0] = (double*)malloc( $((2*$N_SPINOR_DIM*$N_COLOR_DIM*$N_SPINOR_DIM*$N_COLOR_DIM)) * sizeof(double));"
echo "  if ( (*fp)[0] == NULL) {"
echo "    free( *fp );"
echo "    *fp = NULL;"
echo "    return;"
echo "  }"
for((beta=1;beta<$(($N_SPINOR_DIM*$N_COLOR_DIM));beta++)) ; do
  echo "  (*fp)[$(printf "%2d" $beta)] = (*fp)[$(printf "%2d" $(($beta-1)))] + $((2*$N_SPINOR_DIM*$N_COLOR_DIM));"
done
echo "  _fp_eq_zero( *fp );"
echo "  return;"
echo -e "}\n\n"

echo "static inline void free_fp(fermion_propagator_type*fp) {"
echo "  int i;"
echo "  if( *fp != NULL ) {"
echo "    if( (*fp)[0] != NULL) free( (*fp)[0] );"
echo "    free(*fp);"
echo "    *fp = NULL;"
echo "  }"
echo "  return;"
echo -e "}\n\n"


# -----------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_gamma_rot_ti_fp( fermion_propagator_type _r, fermion_propagator_type _s, double _sign, fermion_propagator_type _w) {"
echo "  _fp_eq_gamma_ti_fp( (_r), 0, (_s) );"
echo "  _fp_eq_gamma_ti_fp( (_w), 5, (_r) );"
echo "  _fp_ti_eq_re( (_w), (_sign)*1. );"
echo "  _fp_pl_eq_fp( (_w), (_s) );"
echo "  _fp_eq_fp_ti_re( (_r), (_w), _ONE_OVER_SQRT2 );"
echo -e "}\n\n"

# -----------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_fp_ti_gamma_rot( fermion_propagator_type _r, fermion_propagator_type _s, double _sign, fermion_propagator_type _w) {"
echo "    _fp_eq_fp_ti_gamma( (_r), 5, (_s) );"
echo "    _fp_eq_fp_ti_gamma( (_w), 0, (_r) );"
echo "    _fp_ti_eq_re( (_w), (_sign)*1.);"
echo "    _fp_pl_eq_fp( (_w), (_s) );"
echo "    _fp_eq_fp_ti_re( (_r), (_w), _ONE_OVER_SQRT2 );"
echo -e "}\n\n"

# -----------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_gamma_rot2_ti_fp( fermion_propagator_type _r, fermion_propagator_type _s, double _sign, fermion_propagator_type _w) {"
echo "  _fp_eq_gamma_ti_fp( (_r), 0, (_s) );"
echo "  _fp_eq_gamma_ti_fp( (_w), 5, (_s) );"
echo "  _fp_ti_eq_re( (_w), (_sign) );"
echo "  _fp_pl_eq_fp( (_r), (_w) );"
echo "  _fp_ti_eq_re( (_r), _ONE_OVER_SQRT2 );"
echo -e "}\n\n"

# -----------------------------------------------------------------------------------------------------

echo "static inline void _fp_eq_fp_ti_gamma_rot2( fermion_propagator_type _r, fermion_propagator_type _s, double _sign, fermion_propagator_type _w) {"
echo "    _fp_eq_fp_ti_gamma( (_r), 0, (_s) );"
echo "    _fp_eq_fp_ti_gamma( (_w), 5, (_s) );"
echo "    _fp_ti_eq_re( (_w), (_sign) );"
echo "    _fp_pl_eq_fp( (_r), (_w) );"
echo "    _fp_ti_eq_re( (_r), _ONE_OVER_SQRT2 );"
echo -e "}\n\n"

echo "#endif"

exit 0
