
group_irrep_list <- vector ( mode = "list" )

group_irrep_list[["2Oh"]]  <- c( "A1g", "A1u", "A2g", "A2u", "Eg", "Eg", "T1g", "T1u", "T2g", "T2u", "G1g", "G1u", "G2g", "G2u", "Hg", "Hu" )

group_irrep_list[["2C4v"]] <- c( "A1","A2","B1","B2","E","G1","G2" )

group_irrep_list[["2C2v"]] <- c( "A1","A2","B1","B2","G1" )

group_irrep_list[["2C3v"]] <- c( "A1","A2","K1","K2","E","G1" )


test_irrep <- function () {

  for ( group in c( "2Oh", "2C4v", "2C2v", "2C3v" ) ) {

    t <- vector ( mode = "list" )

    for ( irrep in group_irrep_list[[group]] ) {
      t[[irrep]] <- set_rot_mat_table_cubic_group_double_cover( group = group, irrep = irrep )
    }

    for ( irrep1 in group_irrep_list[[group]] ) {
    for ( irrep2 in group_irrep_list[[group]] ) {


    }}


  }
}
