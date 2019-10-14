
get_source_coords_table <- function( f ) {

  if ( !file.exists(f)) stop( "cannot find file ", f )

  d <- read.table (f)

  # config list
  c <- unique ( d$V1 )

  res <- list()

  for ( i in c ) {
    b <- d[which(d$V1==i),]

    res[[as.character(i)]] <- list()

    for ( s in 1:length(b$V1) ) {

      res[[as.character(i)]][[s]] <- c( b$V2[s], b$V3[s], b$V4[s], b$V5[s] ) 
    }  # end of source coords per config

  }  # end of loop on configs

  return ( invisible( res ) )

}  # end of get_source_coords_table

get_source_coords_tag <- function( c, i, sl ) {

  if ( missing ( c ) ) stop( "need config c=..." )
  if ( missing ( i ) ) stop( "need number of source i =..." )
  if ( missing ( sl ) ) stop( "need source location table sl =..." )

  b <- sl[[as.character(c)]]

  if ( i < 1 || i > length(b) ) stop( "source number out of range" )

  return( paste( "t", b[[i]][1], "x", b[[i]][2], "y", b[[i]][3], "z", b[[i]][4], sep=""  ) )
}  # end of get_source_coords_tag


get_source_coords_key_tag <- function( c, i, sl ) {

  if ( missing ( c ) ) stop( "need config c=..." )
  if ( missing ( i ) ) stop( "need number of source i =..." )
  if ( missing ( sl ) ) stop( "need source location table sl =..." )

  b <- sl[[as.character(c)]]

  if ( i < 1 || i > length(b) ) stop( "source number out of range" )

  return( paste( "T", b[[i]][1], "_X", b[[i]][2], "_Y", b[[i]][3], "_Z", b[[i]][4], sep=""  ) )
}  # end of get_source_coords_key_tag


