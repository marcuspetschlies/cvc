#######################################################################
#
# make_projection_code_inline.sh
#
# Thu Jun  1 08:58:54 CEST 2017
#
#######################################################################
#!/bin/bash
source perm_tab4.in

echo "static inline void _project_pseudotensor_to_scalar (double _Complex **_h, double _Complex *_s, double _p[4], double _q[4]) {"
  echo "  double _plat[4] = { 2.*sin( 0.5 * (_p)[0] ), 2.*sin( 0.5 * (_p)[1] ), 2.*sin( 0.5 * (_p)[2] ), 2.*sin( 0.5 * (_p)[3] ) };"
  echo "  double _qlat[4] = { 2.*sin( 0.5 * (_q)[0] ), 2.*sin( 0.5 * (_q)[1] ), 2.*sin( 0.5 * (_q)[2] ), 2.*sin( 0.5 * (_q)[3] ) };"
  echo "  double _pp = _plat[0] * _plat[0] + _plat[1] * _plat[1] + _plat[2] * _plat[2] + _plat[3] * _plat[3];"
  echo "  double _qq = _qlat[0] * _qlat[0] + _qlat[1] * _qlat[1] + _qlat[2] * _qlat[2] + _qlat[3] * _qlat[3];"
  echo "  double _pq = _plat[0] * _qlat[0] + _plat[1] * _qlat[1] + _plat[2] * _qlat[2] + _plat[3] * _qlat[3];"

  echo "  (*_s) = 0.;"
  i=0
  for((i=0; i<24; i++)); do
    i0=${perm_tab4[$((4*$i  ))]}
    i1=${perm_tab4[$((4*$i+1))]}
    i2=${perm_tab4[$((4*$i+2))]}
    i3=${perm_tab4[$((4*$i+3))]}
    sign=${perm_tab4_sign[$i]}
    if [ $sign -eq 1 ]; then
      echo "  (*_s) += (_h)[$i0][$i1] * _plat[$i2] * _qlat[$i3];"
    else
      echo "  (*_s) -= (_h)[$i0][$i1] * _plat[$i2] * _qlat[$i3];"
    fi
  done
  echo "  (*_s) /= _pp * _qq - _pq * _pq;"
echo "}  /* end of _project_pseudotensor_to_scalar */"

exit 0
