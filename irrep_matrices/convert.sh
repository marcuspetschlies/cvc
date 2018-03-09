#!/bin/bash

group="Oh"
irrep_list=( "A1g" "A2g" "Eg" "T1g" "T2g" "A1u" "A2u" "Eu" "T1u" "T2u")
# irrep_list=( "A1g" "A2g" )

mapping_list=( 0 21 22 23 2 6 3 7 1 5 4 8 10 12 14 9 11 13 17 18 19 15 20 16 )


for irrep in ${irrep_list[*]}; do
  f="set_rot_mat_table_cubic_group_single_cover_${group}_${irrep}_v2.cpp"
  printf "// group %s\n" $gropu > $f
  printf "// irrep %s\n\n" $irrep >> $f
  awk -v var="${mapping_list[*]}" '
    BEGIN{r=-1; split(var,map," ");}
    /'$irrep'/ {
      if ( $(NF-2)==1 && $(NF-1)==1 ) {
        r++
        printf("  // element %2d %s\n", r+1, $2)
      }
      if ( r < 24 ) {
        printf ("  R[%2d][%d][%d] = %s;\n", map[r+1], $(NF-2)-1, $(NF-1)-1, $NF )
      } else {
        printf ("  IR[%2d][%d][%d] = %s;\n", map[r%24+1], $(NF-2)-1, $(NF-1)-1, $NF )
      }
    }' Oh.txt >> $f
done

exit 0
