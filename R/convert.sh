#!/bin/bash

#awk '
#  BEGIN{l=0}
#  /cubic_group_double_cover_rotations\[[0-9]+\]\.n\[0\]/ {l++; printf("cubic_group_double_cover_rotations[[%2d]] <- list(n=c(%d, ", l, $NF); next}
#  /cubic_group_double_cover_rotations\[[0-9]+\]\.n\[1\]/ { printf("%d, ", $NF); next}
#  /cubic_group_double_cover_rotations\[[0-9]+\]\.n\[2\]/ { printf("%d) ", $NF); next}
#  /cubic_group_double_cover_rotations\[[0-9]+\]\.w/ { printf(", w = %s ", $NF); next}
#  /cubic_group_double_cover_rotations\[[0-9]+\]\.name/ { printf(", name =\"NA\")\n\n"); next}' $1


awk -F= '
  BEGIN{l=0}
  /cubic_group_rotations_v2\[[0-9]+\]\.n\[0\]/ {l++; printf("cubic_group_rotations_v2[[%2d]] <- list(n=c(%s, ", l, "NA"); next}
  /cubic_group_rotations_v2\[[0-9]+\]\.n\[1\]/ { printf("%s, ", "NA"); next}
  /cubic_group_rotations_v2\[[0-9]+\]\.n\[2\]/ { printf("%s) ", "NA"); next}
  /cubic_group_rotations_v2\[[0-9]+\]\.w/ { printf(", w = %s ", "NA"); next}
  /cubic_group_rotations_v2\[[0-9]+\]\.a\[0\]/ {printf(", a=c(%s ", $NF); next}
  /cubic_group_rotations_v2\[[0-9]+\]\.a\[1\]/ {printf(", %s ", $NF); next}
  /cubic_group_rotations_v2\[[0-9]+\]\.a\[2\]/ {printf(", %s)", $NF); next}
  /cubic_group_rotations_v2\[[0-9]+\]\.name/ { printf(", name = %s)\n\n", $NF); next}' $1
