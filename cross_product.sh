#!/bin/bash


for i in 0 1 2; do

  j=$(( ($i+1) % 3))
  k=$(( ($j+1) % 3))

  printf '_a_re = ( (_v)[%d] * (_w[%d]) - (_v)[%d] * (_w)[%d] ) - ( (_v)[%d] * (_w[%d]) - (_v)[%d] * (_w)[%d] );\n'  $((2*$j)) $((2*$k)) $((2*$k)) $((2*$j))  $((2*$j+1)) $((2*$k+1)) $((2*$k+1)) $((2*$j+1)) 
  printf '_a_im = ( (_v)[%d] * (_w[%d]) - (_v)[%d] * (_w)[%d] ) + ( (_v)[%d] * (_w[%d]) - (_v)[%d] * (_w)[%d] );\n'  $((2*$j)) $((2*$k+1)) $((2*$k)) $((2*$j+1))  $((2*$j+1)) $((2*$k)) $((2*$k+1)) $((2*$j)) 

  printf '\n'
  
  printf '(_c)[0] += (_u)[%d] * _a_re - (_u)[%d] * _a_im;\n' $((2*$i)) $((2*$i+1))
  printf '(_c)[1] += (_u)[%d] * _a_re + (_u)[%d] * _a_im;\n' $((2*$i+1)) $((2*$i))


  printf '\n'

done

exit 0
