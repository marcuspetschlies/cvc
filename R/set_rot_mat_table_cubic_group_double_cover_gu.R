# ***********************************************************
# * set_rot_mat_table_cubic_group_double_cover_gu.cpp
# *
# * Do 19. Apr 08:44:50 CEST 2018
# *
# * Our standard reference JHEP08(2008)024 gives representation
# * matrices for A1, A2, E, T1, T2, G1, G2, H of 2O
# * i.e. no g/u parity projection
# *
# * Here we complete to A1g, A1u, ... for 2Oh by using
# *
# * T^{Xg]( R ) = +T^{Xg}( IR )
# * T^{Xu]( R ) = -T^{Xu}( IR )
# *
# * for all irreps X
# *
# ***********************************************************

# ***********************************************************
# * irrep matrices for double cover
# ***********************************************************
set_rot_mat_table_cubic_group_double_cover <- function( group, irrep ) {

  ONE_HALF   <- 0.5
  SQRT3_HALF <- 0.5 * sqrt(3.)

  t      <- list()
  t$rid  <- integer()
  t$rmid <- integer()
  t$R    <- vector( mode = "list" )
  t$IR   <- vector( mode = "list" )

  # ***********************************************************
  # * LG 2Oh
  # ***********************************************************
  if ( group == "2Oh" ) {
    nrot <- 48
    
    t$rid  <- 1:nrot
    t$rmid <- 1:nrot

    # ***********************************************************
    # * LG 2Oh irrep A1g
    # ***********************************************************
    if ( irrep ==  "A1g" ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array ( 1, dim = c(1, 1) )
        t$IR[[i]] <- array ( 1, dim = c(1, 1) )
      }

    # ***********************************************************
    # * LG 2Oh irrep A1u
    # ***********************************************************
    } else if ( irrep == "A1u" ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array (  1, dim = c(1, 1) )
        t$IR[[i]] <- array ( -1, dim = c(1, 1) )
      }

    # ***********************************************************
    # * LG 2Oh irrep A2g
    # ***********************************************************
    } else if ( irrep == "A2g" ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array (  1, dim = c(1, 1) )
        t$IR[[i]] <- array (  1, dim = c(1, 1) )
      }

      # * 6C8', 6C8
      for ( i in 8 :19 ) {
        t$R[[i]]  <- array ( -1, dim = c(1, 1) )
        t$IR[[i]] <- array ( -1, dim = c(1, 1) )
      }
     
      # * 12C4'
      for ( i in 36 : 47 ) {
        t$R[[i]]  <- array ( -1, dim = c(1, 1) )
        t$IR[[i]] <- array ( -1, dim = c(1, 1) )
      }

    # ***********************************************************
    # * LG 2Oh irrep A2u
    # ***********************************************************
    } else if ( irrep ==  "A2u" ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array (  1, dim = c(1, 1) )
        t$IR[[i]] <- array ( -1, dim = c(1, 1) )
      }

      # * 6C8', 6C8
      for ( i in 8 : 19 ) {
        t$R[[i]]  <- array ( -1, dim = c(1, 1) )
        t$IR[[i]] <- array (  1, dim = c(1, 1) )
      }
     
      # * 12C4'
      for ( i in 36 : 47 ) {
        t$R[[i]]  <- array ( -1, dim = c(1, 1) )
        t$IR[[i]] <- array (  1, dim = c(1, 1) )
      }


    # ***********************************************************
    # * LG 2Oh irrep E
    # ***********************************************************
    } else if ( ( irrep == "Eg" ) || ( irrep == "Eu" ) ) {

      for ( i in 1 : nrot ) t$R[[i]] <- array ( 0, dim = c(2, 2) )

      # * R1, R48
      t$R[[ 1]][1,1] <- 1.; t$R[[ 1]][2,2] <- 1.
      t$R[[48]][1,1] <- 1.; t$R[[48]][2,2] <- 1.
      # * R2, R5
      t$R[[ 2]][1,1] <- 1.; t$R[[ 2]][2,2] <- 1.
      t$R[[ 5]][1,1] <- 1.; t$R[[ 5]][2,2] <- 1.
      # * R3, R6
      t$R[[ 3]][1,1] <- 1.; t$R[[ 3]][2,2] <- 1.
      t$R[[ 6]][1,1] <- 1.; t$R[[ 6]][2,2] <- 1.
      # * R4, R7
      t$R[[ 4]][1,1] <- 1.; t$R[[ 4]][2,2] <- 1.
      t$R[[ 7]][1,1] <- 1.; t$R[[ 7]][2,2] <- 1.

      # * R13, R16 sigma_3
      t$R[[13]][1,1] <- 1.; t$R[[13]][2,2] <- -1.
      t$R[[16]][1,1] <- 1.; t$R[[16]][2,2] <- -1.
      # * R10, R19 sigma_3
      t$R[[10]][1,1] <- 1.; t$R[[10]][2,2] <- -1.
      t$R[[19]][1,1] <- 1.; t$R[[19]][2,2] <- -1.
      # * R38, R44 sigma_3
      t$R[[38]][1,1] <- 1.; t$R[[38]][2,2] <- -1.
      t$R[[44]][1,1] <- 1.; t$R[[44]][2,2] <- -1.
      # * R39, R45 sigma_3
      t$R[[39]][1,1] <- 1.; t$R[[39]][2,2] <- -1.
      t$R[[45]][1,1] <- 1.; t$R[[45]][2,2] <- -1.
      
      # * -1/2 1 + i sqrt(3)/2 sigma_2

      # * R24, R28
      t$R[[24]][1,1] <- -ONE_HALF;   t$R[[24]][2,2] <- -ONE_HALF
      t$R[[24]][1,2] <-  SQRT3_HALF; t$R[[24]][2,1] <- -SQRT3_HALF
      t$R[[28]][1,1] <- -ONE_HALF;   t$R[[28]][2,2] <- -ONE_HALF
      t$R[[28]][1,2] <-  SQRT3_HALF; t$R[[28]][2,1] <- -SQRT3_HALF
      # * R21, R33
      t$R[[21]][1,1] <- -ONE_HALF;   t$R[[21]][2,2] <- -ONE_HALF
      t$R[[21]][1,2] <-  SQRT3_HALF; t$R[[21]][2,1] <- -SQRT3_HALF
      t$R[[33]][1,1] <- -ONE_HALF;   t$R[[33]][2,2] <- -ONE_HALF
      t$R[[33]][1,2] <-  SQRT3_HALF; t$R[[33]][2,1] <- -SQRT3_HALF
      # * R26, R30
      t$R[[26]][1,1] <- -ONE_HALF;   t$R[[26]][2,2] <- -ONE_HALF
      t$R[[26]][1,2] <-  SQRT3_HALF; t$R[[26]][2,1] <- -SQRT3_HALF
      t$R[[30]][1,1] <- -ONE_HALF;   t$R[[30]][2,2] <- -ONE_HALF
      t$R[[30]][1,2] <-  SQRT3_HALF; t$R[[30]][2,1] <- -SQRT3_HALF
      # * R23, R35
      t$R[[23]][1,1] <- -ONE_HALF;   t$R[[23]][2,2] <- -ONE_HALF
      t$R[[23]][1,2] <-  SQRT3_HALF; t$R[[23]][2,1] <- -SQRT3_HALF
      t$R[[35]][1,1] <- -ONE_HALF;   t$R[[35]][2,2] <- -ONE_HALF
      t$R[[35]][1,2] <-  SQRT3_HALF; t$R[[35]][2,1] <- -SQRT3_HALF
   
      # * -1/2 1 - i sqrt(3)/2 sigma_2

      # * R20, R32
      t$R[[20]][1,1] <- -ONE_HALF;   t$R[[20]][2,2] <- -ONE_HALF
      t$R[[20]][1,2] <- -SQRT3_HALF; t$R[[20]][2,1] <-  SQRT3_HALF
      t$R[[32]][1,1] <- -ONE_HALF;   t$R[[32]][2,2] <- -ONE_HALF
      t$R[[32]][1,2] <- -SQRT3_HALF; t$R[[32]][2,1] <-  SQRT3_HALF
      # * R25, R29
      t$R[[25]][1,1] <- -ONE_HALF;   t$R[[25]][2,2] <- -ONE_HALF
      t$R[[25]][1,2] <- -SQRT3_HALF; t$R[[25]][2,1] <-  SQRT3_HALF
      t$R[[29]][1,1] <- -ONE_HALF;   t$R[[29]][2,2] <- -ONE_HALF
      t$R[[29]][1,2] <- -SQRT3_HALF; t$R[[29]][2,1] <-  SQRT3_HALF
      # * R22, R34
      t$R[[22]][1,1] <- -ONE_HALF;   t$R[[22]][2,2] <- -ONE_HALF
      t$R[[22]][1,2] <- -SQRT3_HALF; t$R[[22]][2,1] <-  SQRT3_HALF
      t$R[[34]][1,1] <- -ONE_HALF;   t$R[[34]][2,2] <- -ONE_HALF
      t$R[[34]][1,2] <- -SQRT3_HALF; t$R[[34]][2,1] <-  SQRT3_HALF
      # * R27, R31
      t$R[[27]][1,1] <- -ONE_HALF;   t$R[[27]][2,2] <- -ONE_HALF
      t$R[[27]][1,2] <- -SQRT3_HALF; t$R[[27]][2,1] <-  SQRT3_HALF
      t$R[[31]][1,1] <- -ONE_HALF;   t$R[[31]][2,2] <- -ONE_HALF
      t$R[[31]][1,2] <- -SQRT3_HALF; t$R[[31]][2,1] <-  SQRT3_HALF

      # * -cos(pi/3) sigma_3 - sin(pi/3) sigma_1

      # * R11, R14
      t$R[[11]][1,1] <- -ONE_HALF;   t$R[[11]][2,2] <-  ONE_HALF
      t$R[[11]][1,2] <- -SQRT3_HALF; t$R[[11]][2,1] <- -SQRT3_HALF
      t$R[[14]][1,1] <- -ONE_HALF;   t$R[[14]][2,2] <-  ONE_HALF
      t$R[[14]][1,2] <- -SQRT3_HALF; t$R[[14]][2,1] <- -SQRT3_HALF
      # * R8, R17
      t$R[[ 8]][1,1] <- -ONE_HALF;   t$R[[ 8]][2,2] <-  ONE_HALF
      t$R[[ 8]][1,2] <- -SQRT3_HALF; t$R[[ 8]][2,1] <- -SQRT3_HALF
      t$R[[17]][1,1] <- -ONE_HALF;   t$R[[17]][2,2] <-  ONE_HALF
      t$R[[17]][1,2] <- -SQRT3_HALF; t$R[[17]][2,1] <- -SQRT3_HALF
      # * R36, R42
      t$R[[36]][1,1] <- -ONE_HALF;   t$R[[36]][2,2] <-  ONE_HALF
      t$R[[36]][1,2] <- -SQRT3_HALF; t$R[[36]][2,1] <- -SQRT3_HALF
      t$R[[42]][1,1] <- -ONE_HALF;   t$R[[42]][2,2] <-  ONE_HALF
      t$R[[42]][1,2] <- -SQRT3_HALF; t$R[[42]][2,1] <- -SQRT3_HALF
      # * R37, R43
      t$R[[37]][1,1] <- -ONE_HALF;   t$R[[37]][2,2] <-  ONE_HALF
      t$R[[37]][1,2] <- -SQRT3_HALF; t$R[[37]][2,1] <- -SQRT3_HALF
      t$R[[43]][1,1] <- -ONE_HALF;   t$R[[43]][2,2] <-  ONE_HALF
      t$R[[43]][1,2] <- -SQRT3_HALF; t$R[[43]][2,1] <- -SQRT3_HALF

      # * -cos(pi/3) sigma_3 + sin(pi/3) sigma_1

      # * R12, R15
      t$R[[12]][1,1] <- -ONE_HALF;   t$R[[12]][2,2] <-  ONE_HALF
      t$R[[12]][1,2] <-  SQRT3_HALF; t$R[[12]][2,1] <-  SQRT3_HALF
      t$R[[15]][1,1] <- -ONE_HALF;   t$R[[15]][2,2] <-  ONE_HALF
      t$R[[15]][1,2] <-  SQRT3_HALF; t$R[[15]][2,1] <-  SQRT3_HALF
      # * R9, R18
      t$R[[ 9]][1,1] <- -ONE_HALF;   t$R[[ 9]][2,2] <-  ONE_HALF
      t$R[[ 9]][1,2] <-  SQRT3_HALF; t$R[[ 9]][2,1] <-  SQRT3_HALF
      t$R[[18]][1,1] <- -ONE_HALF;   t$R[[18]][2,2] <-  ONE_HALF
      t$R[[18]][1,2] <-  SQRT3_HALF; t$R[[18]][2,1] <-  SQRT3_HALF
      # * R40, R46
      t$R[[40]][1,1] <- -ONE_HALF;   t$R[[40]][2,2] <-  ONE_HALF
      t$R[[40]][1,2] <-  SQRT3_HALF; t$R[[40]][2,1] <-  SQRT3_HALF
      t$R[[46]][1,1] <- -ONE_HALF;   t$R[[46]][2,2] <-  ONE_HALF
      t$R[[46]][1,2] <-  SQRT3_HALF; t$R[[46]][2,1] <-  SQRT3_HALF
      # * R41, R47
      t$R[[41]][1,1] <- -ONE_HALF;   t$R[[41]][2,2] <-  ONE_HALF
      t$R[[41]][1,2] <-  SQRT3_HALF; t$R[[41]][2,1] <-  SQRT3_HALF
      t$R[[47]][1,1] <- -ONE_HALF;   t$R[[47]][2,2] <-  ONE_HALF
      t$R[[47]][1,2] <-  SQRT3_HALF; t$R[[47]][2,1] <-  SQRT3_HALF

      t$IR <- t$R

      # ***********************************************************
      # * multiply minus sign to IR irrep matrices
      # ***********************************************************
      if ( irrep == "Eu" ) {
        for ( i in 1 : nrot ) t$IR[[i]] <- -t$IR[[i]]
      }

    # ***********************************************************
    # * LG 2Oh irrep T1g , T1u
    # ***********************************************************
    } else if ( ( irrep == "T1g" ) || ( irrep ==  "T1u" ) ) {

      for ( i in 1 : nrot ) {
        t$R[[i]] <- rot_rotation_matrix_spherical_basis ( 2, cubic_group_double_cover_rotations[[i]]$n, cubic_group_double_cover_rotations[[i]]$w )
      }
      
      t$IR <- t$R

      # ***********************************************************
      # * multiply minus sign to IR irrep matrices
      # ***********************************************************
      if ( irrep == "T1u" ) {
        for ( i in 1 : nrot ) t$IR[[i]] <- -t$IR[[i]]
      }

    # ***********************************************************
    # * LG 2Oh irrep T2g, T2u
    # ***********************************************************
    } else if ( ( irrep == "T2g" ) || ( irrep == "T2u" ) ) { 


      for ( i in 1 : nrot ) {
        t$R[[i]] <- rot_rotation_matrix_spherical_basis ( 2, cubic_group_double_cover_rotations[[i]]$n, cubic_group_double_cover_rotations[[i]]$w )
      }
  
      # * 6C8, 6C8' additional minus sign, R8 to R19
      for ( i in  8 : 19 ) { t$R[[i]] <- -t$R[[i]] }

      # * 12C4' additional minus sign, R36 to R47
      for ( i in 36 : 47 ) { t$R[[i]] <- -t$R[[i]] }
          
      t$IR <- t$R

      # ***********************************************************
      # * multiply minus sign to IR irrep matrices
      # ***********************************************************
      if ( irrep ==  "T2u" ) {
        for ( i in 1 : nrot ) t$IR[[i]] <- -t$IR[[i]]
      }

    # ***********************************************************
    # * LG 2Oh irrep G1g, G1u
    # ***********************************************************
    } else if ( ( irrep == "G1g" ) || ( irrep == "G1u" ) ) {

      for ( i in 1 : nrot ) {
        t$R[[i]] <- rot_rotation_matrix_spherical_basis ( 1, cubic_group_double_cover_rotations[[i]]$n, cubic_group_double_cover_rotations[[i]]$w )
      }

      t$IR <- t$R

      # ***********************************************************
      # * multiply minus sign to IR irrep matrices
      # ***********************************************************
      if ( irrep == "G1u" ) {
        for ( i in 1 : nrot ) t$IR[[i]] <- -t$IR[[i]] 
      }

    # ***********************************************************
    # * LG 2Oh irrep G2g, G2u
    # ***********************************************************
    } else if ( ( irrep == "G2g" ) || ( irrep == "G2u" ) ) {

      for ( i in 1 : nrot ) {
        t$R[[i]] <- rot_rotation_matrix_spherical_basis ( 1, cubic_group_double_cover_rotations[[i]]$n, cubic_group_double_cover_rotations[[i]]$w )
      }

      # * 6C8, 6C8' additional sign
      for ( i in  8 : 19 ) { t$R[[i]] <- -t$R[[i]] }

      # * 12C4' additional sign
      for ( i in 36 : 47 ) { t$R[[i]] <- -t$R[[i]] }

      t$IR <- t$R

      # ***********************************************************
      # * multiply minus sign to IR irrep matrices
      # ***********************************************************
      if ( irrep ==  "G2u" ) {
        for ( i in 1 : nrot ) t$IR[[i]] <- -t$IR[[i]]
      }

    # ***********************************************************
    # * LG 2Oh irrep H
    # ***********************************************************
    } else if ( ( irrep  == "Hg" ) || ( irrep ==  "Hu" ) ) {

      for ( i in 1 : nrot ) {
        t$R[[i]] <- rot_rotation_matrix_spherical_basis ( 3, cubic_group_double_cover_rotations[[i]]$n, cubic_group_double_cover_rotations[[i]]$w )
      }

      t$IR <- t$R

      # ***********************************************************
      # * multiply minus sign to IR irrep matrices
      # ***********************************************************
      if ( irrep == "Hu" ) {
        for ( i in 1 : nrot ) t$IR[[i]] <- -t$IR[[i]]
      }

    } else {
      stop ( "[set_rot_mat_table_cubic_double_cover] unknown irrep name ", irrep )
      return( NULL )
    }

  # ***********************************************************
  # * LG 2C4v
  # ***********************************************************
  } else if ( group == "2C4v" ) {

    nrot <- 8
    t$rid  <- c(  1,  4,  7, 10, 13, 16, 19, 48 )
    t$rmid <- c(  2,  3,  5,  6, 38, 39, 44, 45 )

    # ***********************************************************
    # * LG 2C4v irrep A1
    # ***********************************************************
    if ( irrep == "A1" ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array ( 1, dim = c(1, 1) )
        t$IR[[i]] <- array ( 1, dim = c(1, 1) )
      }

    # ***********************************************************
    # * LG 2C4v irrep A2
    # ***********************************************************
    } else if ( irrep ==  "A2"  ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array (  1, dim = c(1, 1) )
        t$IR[[i]] <- array ( -1, dim = c(1, 1) )
      }

    # ***********************************************************
    # * LG 2C4v irrep B1
    # ***********************************************************
    } else if ( irrep == "B1" ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array (  1, dim = c(1, 1) )
        t$IR[[i]] <- array (  1, dim = c(1, 1) )
      }

      # * 2C8'
      t$R[[4]][1,1]  <- -1.
      t$R[[5]][1,1]  <- -1.
      # * 2C8
      t$R[[6]][1,1]  <- -1.
      t$R[[7]][1,1]  <- -1.

      # * 4IC4'
      t$IR[[5]][1,1] <- -1.
      t$IR[[6]][1,1] <- -1.
      t$IR[[7]][1,1] <- -1.
      t$IR[[8]][1,1] <- -1.

    # ***********************************************************
    # * LG 2C4v irrep B2
    # ***********************************************************
    } else if ( irrep ==  "B2" ) {

      for ( i  in 1 : nrot ) {
        t$R[[i]]  <- array (  1, dim = c(1, 1) )
        t$IR[[i]] <- array (  1, dim = c(1, 1) )
      }

      # * 2C8'
      t$R[[4]][1,1]  <- -1.
      t$R[[5]][1,1]  <- -1.
      # * 2C8
      t$R[[6]][1,1]  <- -1.
      t$R[[7]][1,1]  <- -1.

      # * 4IC4'
      t$IR[[1]][1,1] <- -1.
      t$IR[[2]][1,1] <- -1.
      t$IR[[3]][1,1] <- -1.
      t$IR[[4]][1,1] <- -1.

    # ***********************************************************
    # * LG 2C4v irrep E
    # ***********************************************************
    } else if ( irrep == "E" ) {

      for ( i  in 1 : nrot ) {
        t$R[[i]]  <- array ( 0, dim = c(2, 2) )
        t$IR[[i]] <- array ( 0, dim = c(2, 2) )
      }

      # * I R1, J R48 1
      t$R[[1]][1,1]  <-  1; t$R[[1]][2,2]  <-  1
      t$R[[8]][1,1]  <-  1; t$R[[8]][2,2]  <-  1

      # * 2C4 R4,R7 -1
      t$R[[2]][1,1]  <- -1; t$R[[2]][2,2]  <- -1
      t$R[[3]][1,1]  <- -1; t$R[[3]][2,2]  <- -1

      # * 2C8' R10,R13
      t$R[[4]][1,1]  <-  1.i; t$R[[4]][2,2]  <- -1.i
      t$R[[5]][1,1]  <- -1.i; t$R[[5]][2,2]  <-  1.i

      # * 2C8 R16,R19
      t$R[[6]][1,1]  <- -1.i; t$R[[6]][2,2]  <-  1.i
      t$R[[7]][1,1]  <-  1.i; t$R[[7]][2,2]  <- -1.i

      # * 4IC4 IR2,IR3,IR5,IR6
      t$IR[[1]][1,2] <-  1; t$IR[[1]][2,1]  <-  1
      t$IR[[2]][1,2] <- -1; t$IR[[2]][2,1]  <- -1
      t$IR[[3]][1,2] <-  1; t$IR[[3]][2,1]  <-  1
      t$IR[[4]][1,2] <- -1; t$IR[[4]][2,1]  <- -1

      # * 4IC4' IR38,IR39,IR44,IR45
      t$IR[[5]][1,2] <-  1.i; t$IR[[5]][2,1]  <- -1.i
      t$IR[[6]][1,2] <- -1.i; t$IR[[6]][2,1]  <-  1.i
      t$IR[[7]][1,2] <-  1.i; t$IR[[7]][2,1]  <- -1.i
      t$IR[[8]][1,2] <- -1.i; t$IR[[8]][2,1]  <-  1.i

    # ***********************************************************
    # * LG 2C4v irrep G1
    # ***********************************************************
    } else if ( irrep == "G1" ) {

      for ( i in 1 : nrot ) {
        k <- t$rid[i]
        t$R[[i]]  <-  rot_rotation_matrix_spherical_basis ( 1, cubic_group_double_cover_rotations[[k]]$n, cubic_group_double_cover_rotations[[k]]$w )
        k <- t$rmid[i]
        t$IR[[i]] <- -rot_rotation_matrix_spherical_basis ( 1, cubic_group_double_cover_rotations[[k]]$n, cubic_group_double_cover_rotations[[k]]$w )
      }

    # ***********************************************************
    # * LG 2C4v irrep G2
    # ***********************************************************
    } else if ( irrep == "G2" ) {

      for ( i  in 1 : nrot ) {
        k <- t$rid[i]
        t$R[[i]]  <- rot_rotation_matrix_spherical_basis ( 1, cubic_group_double_cover_rotations[[k]]$n, cubic_group_double_cover_rotations[[k]]$w )
        k <- t$rmid[i]
        t$IR[[i]] <- rot_rotation_matrix_spherical_basis ( 1, cubic_group_double_cover_rotations[[k]]$n, cubic_group_double_cover_rotations[[k]]$w )
      }

      # *          2C8'    2C8
      for ( i in c(4, 5,   6, 7) ) {
        t$R[[i]]  <- -t$R[[i]]
      }

      # *           4IC4'
      for ( i in c( 5, 6, 7, 8 ) ) {
        t$IR[[i]] <- -t$IR[[i]]
      }

    } else {
      stop ( "[set_rot_mat_table_cubic_double_cover] unknown irrep name ", irrep )
      return( NULL )
    }

  # ***********************************************************
  # * LG 2C2v
  # ***********************************************************
  } else if ( group == "2C2v" ) {

    nrot <- 4
    t$rid  <- c(  1,  38, 44, 48 )
    t$rmid <- c(  4,   7, 39, 45 )

    # ***********************************************************
    # * LG 2C2v irrep A1
    # ***********************************************************
    if ( irrep == "A1" ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array ( 1, dim = c (1, 1) )
        t$IR[[i]] <- array ( 1, dim = c (1, 1) )
      }

    # ***********************************************************
    # * LG 2C2v irrep A2
    # ***********************************************************
    } else if ( irrep == "A2" ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array (  1, dim = c (1, 1) )
        t$IR[[i]] <- array ( -1, dim = c (1, 1) )
      }

    # ***********************************************************
    # * LG 2C2v irrep B1
    # ***********************************************************
    } else if (  irrep == "B1" ) {

        t$R[[1]]  <- array (  1, dim = c (1, 1) )
        t$R[[2]]  <- array ( -1, dim = c (1, 1) )
        t$R[[3]]  <- array ( -1, dim = c (1, 1) )
        t$R[[4]]  <- array (  1, dim = c (1, 1) )

        t$IR[[1]] <- array ( -1, dim = c (1, 1) )
        t$IR[[2]] <- array ( -1, dim = c (1, 1) )
        t$IR[[3]] <- array (  1, dim = c (1, 1) )
        t$IR[[4]] <- array (  1, dim = c (1, 1) )

    # ***********************************************************
    # * LG 2C2v irrep B2
    # ***********************************************************
    } else if ( irrep == "B2"  ) {

        t$R[[1]]  <- array (  1, dim = c (1, 1) )
        t$R[[2]]  <- array ( -1, dim = c (1, 1) )
        t$R[[3]]  <- array ( -1, dim = c (1, 1) )
        t$R[[4]]  <- array (  1, dim = c (1, 1) )

        t$IR[[1]] <- array (  1, dim = c (1, 1) )
        t$IR[[2]] <- array (  1, dim = c (1, 1) )
        t$IR[[3]] <- array ( -1, dim = c (1, 1) )
        t$IR[[4]] <- array ( -1, dim = c (1, 1) )
    
    # ***********************************************************
    # * LG 2C2v irrep G1
    # ***********************************************************
    } else if ( irrep == "G1" ) {

      for ( i in 1 : nrot ) {
        k <- t$rid[i]
        t$R[[i]]  <-  rot_rotation_matrix_spherical_basis ( 1, cubic_group_double_cover_rotations[[k]]$n, cubic_group_double_cover_rotations[[k]]$w )
        k <- t$rmid[i]
        t$IR[[i]] <- -rot_rotation_matrix_spherical_basis ( 1, cubic_group_double_cover_rotations[[k]]$n, cubic_group_double_cover_rotations[[k]]$w )
      }
    } else {
      stop ( "[set_rot_mat_table_cubic_double_cover] unknown irrep name ", irrep )
      return( NULL )
    }

  # ***********************************************************
  # * LG 2C3v
  # ***********************************************************
  } else if ( group == "2C3v" ) {

    nrot <- 6
    t$rid  <- c(  1, 20, 24, 28, 32, 48 )
    t$rmid <- c( 37, 45, 47, 39, 41, 43 )

    # ***********************************************************
    # * LG 2C3v irrep A1
    # ***********************************************************
    if ( irrep == "A1"  ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array (  1, dim = c (1, 1) )
        t$IR[[i]] <- array (  1, dim = c (1, 1) )
      }

    # ***********************************************************
    # * LG 2C3v irrep A2
    # ***********************************************************
    } else if (  irrep == "A2" ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array (  1, dim = c (1, 1) )
        t$IR[[i]] <- array ( -1, dim = c (1, 1) )
      }

    # ***********************************************************
    # * LG 2C3v irrep K1
    # ***********************************************************
    } else if ( irrep == "K1"  ) {

      # * I
      t$R[[1]]  <- array (  1, dim = c (1, 1) )
      # * 2C6
      t$R[[2]]  <- array ( -1, dim = c (1, 1) )
      t$R[[3]]  <- array ( -1, dim = c (1, 1) )
      # * 2C3
      t$R[[4]]  <- array (  1, dim = c (1, 1) )
      t$R[[5]]  <- array (  1, dim = c (1, 1) )
      # * J
      t$R[[6]]  <- array ( -1, dim = c (1, 1) )

      # * 3IC4
      t$IR[[1]] <- array (  1.i, dim = c (1, 1) )
      t$IR[[2]] <- array (  1.i, dim = c (1, 1) )
      t$IR[[3]] <- array (  1.i, dim = c (1, 1) )
      # * 3IC4'
      t$IR[[4]] <- array ( -1.i, dim = c (1, 1) )
      t$IR[[5]] <- array ( -1.i, dim = c (1, 1) )
      t$IR[[6]] <- array ( -1.i, dim = c (1, 1) )

    # ***********************************************************
    # * LG 2C3v irrep K2
    # ***********************************************************
    } else if ( irrep == "K2" ) {

      # * I
      t$R[[1]]  <- array (  1, dim = c (1, 1) )
      # * 2C6
      t$R[[2]]  <- array ( -1, dim = c (1, 1) )
      t$R[[3]]  <- array ( -1, dim = c (1, 1) )
      # * 2C3
      t$R[[4]]  <- array (  1, dim = c (1, 1) )
      t$R[[5]]  <- array (  1, dim = c (1, 1) )
      # * J
      t$R[[6]]  <- array ( -1, dim = c (1, 1) )

      # * 3IC4
      t$IR[[1]] <- array ( -1.i, dim = c (1, 1) )
      t$IR[[2]] <- array ( -1.i, dim = c (1, 1) )
      t$IR[[3]] <- array ( -1.i, dim = c (1, 1) )
      # * 3IC4'
      t$IR[[4]] <- array (  1.i, dim = c (1, 1) )
      t$IR[[5]] <- array (  1.i, dim = c (1, 1) )
      t$IR[[6]] <- array (  1.i, dim = c (1, 1) )

    # ***********************************************************
    # * LG 2C3v irrep E
    # ***********************************************************
    } else if ( irrep == "E"  ) {

      for ( i in 1 : nrot ) {
        t$R[[i]]  <- array ( 0, dim = c(2, 2) )
        t$IR[[i]] <- array ( 0, dim = c(2, 2) )
      }

      # * I R1, J R48
      t$R[[1]][1,1] <-  1.
      t$R[[1]][2,2] <-  1.
      t$R[[6]][1,1] <-  1.
      t$R[[6]][2,2] <-  1.

      # * 2C6 R20,R24
      t$R[[2]][1,1]  <- -0.5
      t$R[[2]][2,2]  <- -0.5
      t$R[[2]][1,2]  <- -SQRT3_HALF
      t$R[[2]][2,1]  <-  SQRT3_HALF

      t$R[[3]][1,1]  <- -0.5
      t$R[[3]][2,2]  <- -0.5
      t$R[[3]][1,2]  <-  SQRT3_HALF
      t$R[[3]][2,1]  <- -SQRT3_HALF

      # * 2C3 R28,R32
      t$R[[4]][1,1]  <- -0.5
      t$R[[4]][2,2]  <- -0.5
      t$R[[4]][1,2]  <-  SQRT3_HALF
      t$R[[4]][2,1]  <- -SQRT3_HALF

      t$R[[5]][1,1]  <- -0.5
      t$R[[5]][2,2]  <- -0.5
      t$R[[5]][1,2]  <- -SQRT3_HALF
      t$R[[5]][2,1]  <-  SQRT3_HALF

      # * 3IC4 IR37,IR45,IR47
      t$IR[[1]][1,1]  <-  0.5
      t$IR[[1]][2,2]  <- -0.5
      t$IR[[1]][1,2]  <-  SQRT3_HALF
      t$IR[[1]][2,1]  <-  SQRT3_HALF

      t$IR[[2]][1,1]  <- -1.
      t$IR[[2]][2,2]  <-  1.

      t$IR[[3]][1,1]  <-  0.5
      t$IR[[3]][2,2]  <- -0.5
      t$IR[[3]][1,2]  <- -SQRT3_HALF
      t$IR[[3]][2,1]  <- -SQRT3_HALF

      t$IR[[4]][1,1]  <- -1.
      t$IR[[4]][2,2]  <-  1.

      t$IR[[5]][1,1]  <-  0.5
      t$IR[[5]][2,2]  <- -0.5
      t$IR[[5]][1,2]  <- -SQRT3_HALF
      t$IR[[5]][2,1]  <- -SQRT3_HALF

      t$IR[[6]][1,1]  <-  0.5
      t$IR[[6]][2,2]  <- -0.5
      t$IR[[6]][1,2]  <-  SQRT3_HALF
      t$IR[[6]][2,1]  <-  SQRT3_HALF

    # ***********************************************************
    # * LG 2C3v irrep G1
    # ***********************************************************
    } else if ( irrep == "G1" ) {

      for ( i in 1 : nrot ) {
        k <- t$rid[i]
        t$R[[i]]  <-  rot_rotation_matrix_spherical_basis ( 1, cubic_group_double_cover_rotations[[k]]$n, cubic_group_double_cover_rotations[[k]]$w )
        k <- t$rmid[i]
        t$IR[[i]] <- -rot_rotation_matrix_spherical_basis ( 1, cubic_group_double_cover_rotations[[k]]$n, cubic_group_double_cover_rotations[[k]]$w )
      }

    } else {
      stop ( "[set_rot_mat_table_cubic_double_cover] unknown irrep name ", irrep )
      return( NULL )
    }

  } else {
    stop ( "[set_rot_mat_table_cubic_double_cover] unknown group name ", group )
    return ( NULL )
  }
 
  return( t )
}  # end of set_rot_mat_table_cubic_group_double_cover
