# ***********************************************************
#
# ***********************************************************

gamma_basis <- vector(mode="list")

init_gamma_basis <- function () {
  gamma_basis[["tmlqcd"]] <-vector(mode="list")

  gamma_basis[["tmlqcd"]][["t"]] <- array(0, dim=c(4,4))
  gamma_basis[["tmlqcd"]][["t"]][1,3] = -1
  gamma_basis[["tmlqcd"]][["t"]][2,4] = -1
  gamma_basis[["tmlqcd"]][["t"]][3,1] = -1
  gamma_basis[["tmlqcd"]][["t"]][4,2] = -1
  
  gamma_basis[["tmlqcd"]][["x"]] <- array(0, dim=c(4,4))
  gamma_basis[["tmlqcd"]][["x"]][1,4] = -1.i
  gamma_basis[["tmlqcd"]][["x"]][2,3] = -1.i
  gamma_basis[["tmlqcd"]][["x"]][3,2] = +1.i
  gamma_basis[["tmlqcd"]][["x"]][4,1] = +1.i
  
  gamma_basis[["tmlqcd"]][["y"]] <- array(0, dim=c(4,4))
  gamma_basis[["tmlqcd"]][["y"]][1,4] = -1
  gamma_basis[["tmlqcd"]][["y"]][2,3] = +1
  gamma_basis[["tmlqcd"]][["y"]][3,2] = +1
  gamma_basis[["tmlqcd"]][["y"]][4,1] = -1
  
  gamma_basis[["tmlqcd"]][["z"]] <- array(0, dim=c(4,4))
  gamma_basis[["tmlqcd"]][["z"]][1,3] = -1.i
  gamma_basis[["tmlqcd"]][["z"]][2,4] = +1.i
  gamma_basis[["tmlqcd"]][["z"]][3,1] = +1.i
  gamma_basis[["tmlqcd"]][["z"]][4,2] = -1.i
  
  gamma_basis[["tmlqcd"]][["id"]] <- array(0, dim=c(4,4))
  gamma_basis[["tmlqcd"]][["id"]][1,1] = 1
  gamma_basis[["tmlqcd"]][["id"]][2,2] = 1
  gamma_basis[["tmlqcd"]][["id"]][3,3] = 1
  gamma_basis[["tmlqcd"]][["id"]][4,4] = 1
  
  gamma_basis[["tmlqcd"]][["5"]] <- array(0, dim=c(4,4))
  gamma_basis[["tmlqcd"]][["5"]][1,1] = +1
  gamma_basis[["tmlqcd"]][["5"]][2,2] = +1
  gamma_basis[["tmlqcd"]][["5"]][3,3] = -1
  gamma_basis[["tmlqcd"]][["5"]][4,4] = -1
}



# ***********************************************************
# *
# ***********************************************************
axis2polar <- function ( n ) {

  if ( missing(n) ) stop ("Error, need orientiation n")

  r <- sqrt( sum( n^2 ) )

  theta <- 0
  phi   <- 0

  if ( r > 0 ) {

    theta <- acos( n[3] / r )

    phi   <- atan2( n[2], n[1] )

    if ( phi < 0 ) phi = phi + 2*pi
  }
  return (list(theta=theta,phi=phi))

}  # end of axis2polar

# ***********************************************************
# ***********************************************************

# ***********************************************************
# * matrix U below is for cartesian -> spherical contravariant
# *
# * v_sph_con = U v_cart
# ***********************************************************/
rot_spherical2cartesian_3x3 <- function( S ) {

  r <- 1. / sqrt(2.)

  U <- array ( dim=c(3,3) )

  U[1,1] <- -r
  U[1,2] <-  1.i*r
  U[1,3] <-  0.
  U[2,1] <-  0.
  U[2,2] <-  0.
  U[2,3] <-  1.
  U[3,1] <-  r
  U[3,2] <-  1.i*r
  U[3,3] <-  0.

  return ( Conj(t(U)) %*% S %*% U )

}  # end of rot_spherical2cartesian_3x3

# **********************************************************
# **********************************************************

# ***********************************************************
# * input
# * J2 = 2 x J (J = 0, 1/2, 1, 3/2, ... )
# * n  = direction of rotation axis
# * w  = rotation angle
# *
# * output
# * R  = rotation matrix in spherical basis
# ***********************************************************

rot_rotation_matrix_spherical_basis <- function ( J2, n, w) {

  if ( missing (J2) || missing(n) || missing(w) ) stop( "Insufficient arguments to build rotation matrix" )

  R <- array( dim = c (J2+1,J2+1) )

  polang      <- axis2polar ( n )

  v           <- sin ( w / 2. ) * sin ( polang$theta )
  vsqr_mi_one <- v*v - 1

  u           <- cos ( w / 2. ) - 1.i * sin( w / 2.) * cos( polang$theta )

  for( ik in 0:J2 ) {
    k2 <- J2 - 2*ik

    J_mi_m1 <- ( J2 - k2 ) / 2
    J_pl_m1 <- ( J2 + k2 ) / 2

    J_mi_m1_fac <- factorial (J_mi_m1)
    J_pl_m1_fac <- factorial (J_pl_m1)

    for( il in 0:J2 ) {
      l2 <- J2 - 2 * il

      J_mi_m2 <- ( J2 - l2 ) / 2
      J_pl_m2 <- ( J2 + l2 ) / 2

      J_mi_m2_fac <- factorial (J_mi_m2)
      J_pl_m2_fac <- factorial (J_pl_m2)

      norm <- sqrt( J_pl_m1_fac * J_mi_m1_fac * J_pl_m2_fac * J_mi_m2_fac )

      m1_pl_m2 <- (k2 + l2 ) / 2
      m1_mi_m2 <- (k2 - l2 ) / 2

      if ( m1_pl_m2 >= 0 ) {

        smax <-  min( J_mi_m1, J_mi_m2 )

        ssum <- 0.
        if ( smax >= 0 ) {
          for( s in  0 : smax ) {
            ssum <- ssum +  v^(J2 - m1_pl_m2 - 2*s ) * vsqr_mi_one^s / ( factorial(s) * factorial(s + m1_pl_m2) * factorial(J_mi_m1 - s) * factorial( J_mi_m2 - s) )
          }
        }

        R[ik+1,il+1] <- ( cos( m1_mi_m2 * polang$phi) - sin( m1_mi_m2 * polang$phi) * 1.i ) * ssum * u^m1_pl_m2 *  (-1.i)^(J2 - m1_pl_m2) * norm


      } else {
        smax <- min( J_pl_m1, J_pl_m2 )

        ssum <- 0.
        if ( smax >= 0 ) {
          for( s in 0 : smax ) {
            ssum <- ssum + v^(J2 + m1_pl_m2 - 2*s ) * vsqr_mi_one^s / ( factorial(s) * factorial(s - m1_pl_m2) * factorial(J_pl_m1 - s) * factorial( J_pl_m2 - s) )
          }
        }

        R[ik+1, il+1 ] <- ( cos( m1_mi_m2 * polang$phi) - sin( m1_mi_m2 * polang$phi) * 1.i ) * ssum * Conj(u)^( -m1_pl_m2) * (-1.i)^(J2 + m1_pl_m2) * norm

      }

    }  # end of loop on m2
  }  # end of loop on m1

  return ( R )
}  # end of rot_rotation_matrix_spherical_basis

# ***********************************************************
# ***********************************************************

# ***********************************************************
# * bi-spinor rotation matrix as (1/2, 0) + (0, 1/2)
# ***********************************************************
rot_bispinor_rotation_matrix_spherical_basis <- function( n, w ) {

  SSpin <- rot_rotation_matrix_spherical_basis ( 1, n, w)

  ASpin <- array ( 0, dim=c(4,4) )

  ASpin[1,1] <- SSpin[1,1]
  ASpin[1,2] <- SSpin[1,2]
  ASpin[2,1] <- SSpin[2,1]
  ASpin[2,2] <- SSpin[2,2]
  ASpin[3,3] <- SSpin[1,1]
  ASpin[3,4] <- SSpin[1,2]
  ASpin[4,3] <- SSpin[2,1]
  ASpin[4,4] <- SSpin[2,2]

  return( ASpin )
}  # end of rot_bispinor_rotation_matrix_spherical_basis

# ***********************************************************
# ***********************************************************

# ***********************************************************
# * effect of parity on spherical basis state 
# ***********************************************************
rot_inversion_matrix_spherical_basis <- function ( J2, bispinor ) {

  if ( bispinor && ( J2 != 1 ) ) stop( "[rot_inversion_matrix_spherical_basis] unrecognized combination of J2 and bispinor")

  if ( ( ( J2 == 1 ) && bispinor ) || ( J2 == 3 ) ) {
    return ( gamma_basis[["tmlqcd"]][["t"]] )
  }

  if ( J2 %% 4 == 0 ) {
    # ***********************************************************
    # * spin 0, 2, 4, ...
    # ***********************************************************
    return ( diag ( rep ( 1, times=(J2+1 ) ) ) )

  } else if ( J2 %% 4 == 2 ) {
    # ***********************************************************
    # * spin 1, 3, 5, ...
    # ***********************************************************
    return ( diag ( rep ( -1, times=(J2+1 ) ) ) )

  } else if ( ( J2 %% 2 == 1 ) && !bispinor ) {

    # ***********************************************************
    # * spin 1/2, 3/2, 5/2, ...
    # ***********************************************************
    return ( diag ( rep ( 1, times=(J2+1 ) ) ) )
  }

  return ( NaN )

}  # end of rot_inversion_matrix_spherical_basis



# ***********************************************************
# ***********************************************************

# ***********************************************************
# * set Wigner d-function for all M, M'
# ***********************************************************
wigner_d <- function ( b, J2 ) {

  wd <- array( dim = c(J2+1, J2+1) )

  cbh <- cos ( 0.5 * b )
  sbh <- sin ( 0.5 * b )

  for ( im1 in 0 : J2 ) {

    J_mi_m1 <- im1
    J_pl_m1 <- J2 - im1

    J_mi_m1_fac <- factorial ( J_mi_m1 )
    J_pl_m1_fac <- factorial ( J_pl_m1 )

    for ( im2 in 0 : J2  ) {

      J_mi_m2  <- im2
      J_pl_m2  <- J2 - im2
      m1_pl_m2 <- J2 - im1 - im2

      sign     <- (-1)^J_mi_m2

      J_mi_m2_fac <- factorial ( J_mi_m2 )
      J_pl_m2_fac <- factorial ( J_pl_m2 )

      k_min <- max ( -m1_pl_m2, 0 )
      k_max <- min ( J_mi_m1, J_mi_m2 )

      f <- sign * sqrt( J_mi_m1_fac * J_pl_m1_fac * J_mi_m2_fac * J_pl_m2_fac )

      dtmp = 0.

      if ( k_max >= k_min ) {
        for ( k in k_min : k_max ) {

          k_sign <- (-1)^k 

          dtmp <- dtmp + k_sign * cbh^(m1_pl_m2 + 2*k ) * sbh^(J2 - m1_pl_m2 - 2*k ) /
              ( factorial ( k ) * factorial ( J_mi_m1 - k ) * factorial ( J_mi_m2 - k ) * factorial ( m1_pl_m2 + k ) )

        }
      }

      wd[im1+1, im2+1 ] <- f * dtmp

    }  # end of loop on im2
  }    # end of loop on im1

  return ( wd )
}  # end of wigner_d

# ***********************************************************
# ***********************************************************

# ***********************************************************
# * input
# * J2 = 2 x J (J = 0, 1/2, 1, 3/2, ... )
# * a,b,c Euler angles
# *
# * output
# * R  = rotation matrix in spherical basis
# ***********************************************************
rot_rotation_matrix_spherical_basis_Wigner_D <- function(  J2, a ) {

  R <- array ( dim = c(J2+1, J2+1) )
  wd <- wigner_d ( a[2], J2 )

  for ( im1 in 0 : J2  ) {
    m1 <- ( J2 - 2*im1 ) / 2.

    z1 <- cos ( m1 * a[1] ) - sin ( m1 * a[1] )*1.i

    for ( im2 in 0 : J2 ) {
      m2 <- ( J2 - 2*im2 ) / 2.

      z2 <- cos ( m2 * a[3] ) - sin ( m2 * a[3] )*1.i

      R[im1+1, im2+1] = z1 * z2 * wd[im1+1, im2+1]
    }
  }

  return( R )
}  # end of rot_rotation_matrix_spherical_basis_Wigner_D

# ***********************************************************
# ***********************************************************

# ***********************************************************
# * build spin-1 rotations in cartesian basis
# * directly from n and omega
# ***********************************************************
rot_mat_spin1_cartesian <- function( n, omega ) {

  R <- array ( 0, dim = c(3, 3) )

  if ( all(n  == 0 ) ) {
    R[1, 1] = 1
    R[2, 2] = 1
    R[3, 3] = 1
    return( R )
  }

  cos_omega        <- cos( omega )
  sin_omega        <- sin( omega )
  sin_omega_h      <- sin ( 0.5 * omega )
  one_mi_cos_omega <- 2. * sin_omega_h * sin_omega_h
  d                <- n / sqrt( sum( n^2 ) ) 

  R[1,1] <- cos_omega  + one_mi_cos_omega * d[1] * d[1]
  R[1,2] <- one_mi_cos_omega * d[1] * d[2] - sin_omega * d[3]
  R[1,3] <- one_mi_cos_omega * d[3] * d[1] + sin_omega * d[2]

  R[2,1] <- one_mi_cos_omega * d[1] * d[2] + sin_omega * d[3]
  R[2,2] <- cos_omega  + one_mi_cos_omega * d[2] * d[2]
  R[2,3] <- one_mi_cos_omega * d[2] * d[3] - sin_omega * d[1]

  R[3,1] <- one_mi_cos_omega * d[3] * d[1] - sin_omega * d[2]
  R[3,2] <- one_mi_cos_omega * d[2] * d[3] + sin_omega * d[1]
  R[3,3] <- cos_omega  + one_mi_cos_omega * d[3] * d[3]

  return( R )
}  # end of rot_mat_spin1_cartesian

# ***********************************************************
# ***********************************************************

# ***********************************************************
# * build spin-1/2 rotations in spherical basis
# * directly from n and omega
# ***********************************************************
rot_mat_spin1_2_spherical <- function( n, omega ) {

  R <- array ( dim = c(2, 2) )

  if ( all( n == 0 ) ) {
    # ***********************************************************
    # * For R = I and R = J set to any vector 
    # ***********************************************************
    n[0] = 1
    n[1] = 2
    n[2] = 3
  }

  cos_omega_half <- cos( omega / 2. )
  sin_omega_half <- sin( omega / 2. )
  d              <- n / sqrt(sum(n^2)) 

  R[1,1] <- cos_omega_half - 1.i * d[3] * sin_omega_half

  R[1,2] <- ( -1.i * d[1] - d[2] ) * sin_omega_half

  R[2,1] <- ( -1.i * d[1] + d[2] ) * sin_omega_half

  R[2,2] <- cos_omega_half + 1.i * d[3] * sin_omega_half

  return( R )
}  # end of rot_mat_spin1_2_spherical

# ***********************************************************
# ***********************************************************

# ***********************************************************
# *
# ***********************************************************
rot_mat_check_is_real_int <- function ( R , eps=1.e-12 ) {
  if ( missing(R) ) stop("Need input matrix")
  S <- round( Re(R) )
  if ( any( abs( R - S ) > eps ) ) stop("matrix is not real int")

  return( S );
}  # end of rot_mat_check_is_int
