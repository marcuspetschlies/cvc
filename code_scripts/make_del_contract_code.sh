#!/bin/bash

what=$1
echo "#ifndef _DEL_CONTRACT_${what}_H"
echo "#define _DEL_CONTRACT_${what}_H"
echo "#define _sp_eq_fp_del_contract${what}_fp(_r, _u, _v) {\\"

# -------------------------------------------------------------------------------------------------

if [ "X$what" == "X13" ]; then
# contract 1 3
for((s2=0;s2<4;s2++)); do
  for((b3=0;b3<3;b3++)); do
 
    for((s1=0;s1<4;s1++)); do
      for((a3=0;a3<3;a3++)); do

        s1b3=$((3*$s1+$b3))
        s2b3=$((3*$s2+$b3))

        echo "(_r)[$(printf "%2d" $s2)][$(printf "%2d" $((2*$s1)))] += \\"
        for((k=0;k<4;k++)); do
          ra3=$((3*$k+$a3))
          echo "+(_u)[$(printf "%2d" $s1b3)][$(printf "%2d" $((2*$ra3)))] * (_v)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$ra3)))] - (_u)[$(printf "%2d" $s1b3)][$(printf "%2d" $((2*$ra3+1)))] * (_v)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$ra3+1)))] \\"
        done
        echo ";\\"

        echo "(_r)[$(printf "%2d" $s2)][$(printf "%2d" $((2*$s1+1)))] += \\"
        for((k=0;k<4;k++)); do
          ra3=$((3*$k+$a3))
          echo "+(_u)[$(printf "%2d" $s1b3)][$(printf "%2d" $((2*$ra3)))] * (_v)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$ra3+1)))] + (_u)[$(printf "%2d" $s1b3)][$(printf "%2d" $((2*$ra3+1)))] * (_v)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$ra3)))] \\"
        done
        echo ";\\"

      done
    done
  done
done
fi

# -------------------------------------------------------------------------------------------------

if [ "X$what" == "X14" ]; then
# contract 1 4
for((s2=0;s2<4;s2++)); do
  for((b3=0;b3<3;b3++)); do
 
    for((s1=0;s1<4;s1++)); do
      for((a3=0;a3<3;a3++)); do

        s1b3=$((3*$s1+$b3))
        s2a3=$((3*$s2+$a3))

        echo "(_r)[$(printf "%2d" $s2)][$(printf "%2d" $((2*$s1)))] += \\"
        for((k=0;k<4;k++)); do
          ra3=$((3*$k+$a3))
          rb3=$((3*$k+$b3))
          echo "+(_u)[$(printf "%2d" $s1b3)][$(printf "%2d" $((2*$ra3)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s2a3)))] - (_u)[$(printf "%2d" $s1b3)][$(printf "%2d" $((2*$ra3+1)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s2a3+1)))] \\"
        done
        echo ";\\"

        echo "(_r)[$(printf "%2d" $s2)][$(printf "%2d" $((2*$s1+1)))] += \\"
        for((k=0;k<4;k++)); do
          ra3=$((3*$k+$a3))
          rb3=$((3*$k+$b3))
          echo "+(_u)[$(printf "%2d" $s1b3)][$(printf "%2d" $((2*$ra3)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s2a3+1)))] + (_u)[$(printf "%2d" $s1b3)][$(printf "%2d" $((2*$ra3+1)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s2a3)))] \\"
        done
        echo ";\\"

      done
    done
  done
done
fi

# -------------------------------------------------------------------------------------------------

if [ "X$what" == "X23" ]; then
# contract 1 3
for((s2=0;s2<4;s2++)); do
  for((b3=0;b3<3;b3++)); do
 
    for((s1=0;s1<4;s1++)); do
      for((a3=0;a3<3;a3++)); do

        s1a3=$((3*$s1+$a3))
        s2b3=$((3*$s2+$b3))

        echo "(_r)[$(printf "%2d" $s2)][$(printf "%2d" $((2*$s1)))] += \\"
        for((k=0;k<4;k++)); do
          ra3=$((3*$k+$a3))
          rb3=$((3*$k+$b3))
          echo "+(_u)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s1a3)))] * (_v)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$ra3)))] - (_u)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s1a3+1)))] * (_v)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$ra3+1)))] \\"
        done
        echo ";\\"

        echo "(_r)[$(printf "%2d" $s2)][$(printf "%2d" $((2*$s1+1)))] += \\"
        for((k=0;k<4;k++)); do
          ra3=$((3*$k+$a3))
          rb3=$((3*$k+$b3))
          echo "+(_u)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s1a3)))] * (_v)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$ra3+1)))] + (_u)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s1a3+1)))] * (_v)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$ra3)))] \\"
        done
        echo ";\\"

      done
    done
  done
done
fi

# -------------------------------------------------------------------------------------------------

if [ "X$what" == "X24" ]; then
# contract 1 4
for((s2=0;s2<4;s2++)); do
  for((b3=0;b3<3;b3++)); do
 
    for((s1=0;s1<4;s1++)); do
      for((a3=0;a3<3;a3++)); do

        s1a3=$((3*$s1+$a3))
        s2a3=$((3*$s2+$a3))

        echo "(_r)[$(printf "%2d" $s2)][$(printf "%2d" $((2*$s1)))] += \\"
        for((k=0;k<4;k++)); do
          rb3=$((3*$k+$b3))
          echo "+(_u)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s1a3)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s2a3)))] - (_u)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s1a3+1)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s2a3+1)))] \\"
        done
        echo ";\\"

        echo "(_r)[$(printf "%2d" $s2)][$(printf "%2d" $((2*$s1+1)))] += \\"
        for((k=0;k<4;k++)); do
          rb3=$((3*$k+$b3))
          echo "+(_u)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s1a3)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s2a3+1)))] + (_u)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s1a3+1)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$s2a3)))] \\"
        done
        echo ";\\"

      done
    done
  done
done
fi

# -------------------------------------------------------------------------------------------------

if [ "X$what" == "X34" ]; then
# contract 3 4
for((s2=0;s2<4;s2++)); do
  for((b3=0;b3<3;b3++)); do
 
    for((s1=0;s1<4;s1++)); do
      for((a3=0;a3<3;a3++)); do

        s1a3=$((3*$s1+$a3))
        s2b3=$((3*$s2+$b3))

        echo "(_r)[$(printf "%2d" $s2)][$(printf "%2d" $((2*$s1)))] += \\"
        for((k=0;k<4;k++)); do
          ra3=$((3*$k+$a3))
          rb3=$((3*$k+$b3))
          echo "+(_u)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$s1a3)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$ra3)))] - (_u)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$s1a3+1)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$ra3+1)))] \\"
        done
        echo ";\\"

        echo "(_r)[$(printf "%2d" $s2)][$(printf "%2d" $((2*$s1+1)))] += \\"
        for((k=0;k<4;k++)); do
          ra3=$((3*$k+$a3))
          rb3=$((3*$k+$b3))
          echo "+(_u)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$s1a3)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$ra3+1)))] + (_u)[$(printf "%2d" $s2b3)][$(printf "%2d" $((2*$s1a3+1)))] * (_v)[$(printf "%2d" $rb3)][$(printf "%2d" $((2*$ra3)))] \\"
        done
        echo ";\\"

      done
    done
  done
done
fi

# -------------------------------------------------------------------------------------------------

echo "}"


echo "#endif"
exit 0
