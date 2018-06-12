# ******************************************************************
# * little groups with their irreps
# ******************************************************************
source("little_groups_2Oh.R")
source("R_gamma.R")
source("rotations.R")
source("set_cubic_group_double_cover_elements.R")
source("set_rot_mat_table_cubic_group_double_cover_gu.R")
source("projector_set.R")


# ******************************************************************
# *
# ******************************************************************
test_projector <- function () {

  interpolator_num     <- 1
  interpolator_J2_list <- c( 2 )
  
  for ( gr in little_groups_2Oh )
  {

    for ( ir in gr$irreps )
    {

      cat( "# [test_projector] little group ", gr$group, " irrep ", ir, "\n" )
      p <- projector_set ( lg=gr, irrep = ir, interpolator_num = interpolator_num, interpolator_J2_list = interpolator_J2_list )






    }

  }

}
