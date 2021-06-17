#######################################################################
#
# make_rot_code_inline.sh
#
# Fri May 19 15:39:55 CEST 2017
#
#######################################################################
#!/bin/bash

source dimensions.in

if [ 0 -eq  1 ]; then

echo "inline void rot_bispinor_mat_ti_fv( double * _r, double _Complex ** _R, double * _s) {"
echo "  double _Complex _zspinor1[12];"
for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))
    echo "  _zspinor1[$(printf "%2d" $j )] = 0.;"

    for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
      a=$b
      k=$(($N_COLOR_DIM*$s1+$a))

      echo "  _zspinor1[$(printf "%2d" $j )] += (_R)[$s2][$s1] * ( (_s)[$(printf "%2d" $((2*$k)) )] + (_s)[$(printf "%2d" $((2*$k+1)) )] * I);"
    done
  done
done
echo "  memcpy( (_r), _zspinor1, 24*sizeof(double) );"
echo "}  /* end of rot_bispinor_mat_ti_fv */"

fi

if [ 0 -eq  1 ]; then

echo "inline void rot_fp_ti_bispinor_mat ( fermion_propagator_type _r, double _Complex ** _R, fermion_propagator_type _s) {"
echo "  double _Complex _c;"
echo "  fermion_propagator_type _fp;"
echo "  create_fp( &_fp );"

for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
  for((b=0;b<$N_COLOR_DIM;b++)); do
    j=$(($N_COLOR_DIM*$s2+$b))

    for((s3=0;s3<$N_SPINOR_DIM;s3++)); do
      for((c=0;c<$N_COLOR_DIM;c++)); do
        l=$(($N_COLOR_DIM*$s3+$c))
        echo "  _c = 0.;"

        for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
          a=$b
          k=$(($N_COLOR_DIM*$s1+$a))

          echo "  _c += ( (_s)[$(printf "%2d" $k)][$(printf "%2d" $((2*$l)) )] + (_s)[$(printf "%2d" $k )][$(printf "%2d" $((2*$l+1)) )] * I) *  (_R)[$s1][$s2];"

        done

        echo "  (_fp)[$(printf "%2d" $j )][$(printf "%2d" $((2*$l))  )] = creal(_c);"
        echo "  (_fp)[$(printf "%2d" $j )][$(printf "%2d" $((2*$l+1)))] = cimag(_c);"
      done
    done
  done
done
echo "  _fp_eq_fp( _r, _fp);"
echo "  free_fp( &_fp);"
echo "}  /* end of rot_fp_bispinor_mat */"

fi


if [ 0 -eq  1 ]; then

echo "inline void rot_fv_ti_bispinor_mat ( double** _r, double _Complex ** _R, double** _s, unsigned int _ix) {"
echo "  fermion_propagator_type _fp;"
echo "  create_fp( &_fp );"
echo "  _assign_fp_point_from_field( _fp, _s, _ix);"
echo "  rot_fp_ti_bispinor_mat ( _fp, _R, _fp);"
echo "  _assign_fp_field_from_point( _r, _fp, _ix);"
echo "  free_fp( &_fp );"
echo "}  /* end of rot_fv_ti_bispinor_mat */"

fi

# if [ 0 -eq  1 ]; then

echo "inline void rot_sp_ti_bispinor_mat ( spinor_propagator_type _r, double _Complex ** _R, spinor_propagator_type _s) {"
echo "  double _Complex _c;"
echo "  spinor_propagator_type _sp;"
echo "  create_sp( &_sp );"

for((s3=0;s3<$N_SPINOR_DIM;s3++)); do
  for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
    echo "  _c = 0.;"

    for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
      echo "  _c += ( (_s)[$(printf "%2d" $s1 )][$(printf "%2d" $((2*$s2)) )] + (_s)[$(printf "%2d" $s1 )][$(printf "%2d" $((2*$s2+1)) )] * I) *  (_R)[$s1][$s3];"
    done
    echo "  (_sp)[$(printf "%2d" $s3 )][$(printf "%2d" $((2*$s2))  )] = creal(_c);"
    echo "  (_sp)[$(printf "%2d" $s3 )][$(printf "%2d" $((2*$s2+1)))] = cimag(_c);"
  done
done
echo "  _sp_eq_sp( _r, _sp);"
echo "  free_sp( &_sp);"
echo "}  /* end of rot_sp_ti_bispinor_mat */"

# fi


# if [ 0 -eq  1 ]; then
echo "inline void rot_bispinor_mat_ti_sp ( spinor_propagator_type _r, double _Complex ** _R, spinor_propagator_type _s) {"
echo "  double _Complex _c;"
echo "  spinor_propagator_type _sp;"
echo "  create_sp( &_sp );"

for((s3=0;s3<$N_SPINOR_DIM;s3++)); do
  for((s2=0;s2<$N_SPINOR_DIM;s2++)); do
    echo "  _c = 0.;"

    for((s1=0;s1<$N_SPINOR_DIM;s1++)); do
      echo "  _c += ( (_s)[$(printf "%2d" $s3 )][$(printf "%2d" $((2*$s1)) )] + (_s)[$(printf "%2d" $s3 )][$(printf "%2d" $((2*$s1+1)) )] * I) *  (_R)[$s2][$s1];"
    done
    echo "  (_sp)[$(printf "%2d" $s3 )][$(printf "%2d" $((2*$s2))  )] = creal(_c);"
    echo "  (_sp)[$(printf "%2d" $s3 )][$(printf "%2d" $((2*$s2+1)))] = cimag(_c);"
  done
done
echo "  _sp_eq_sp( _r, _sp);"
echo "  free_sp( &_sp);"
echo "}  /* end of rot_bispinor_mat_ti_sp */"

# fi

