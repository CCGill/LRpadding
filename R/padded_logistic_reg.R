#' Padded Logistic Regression
#' 
#' basic_padded_logistic_reg function that performs padded logistic regression.  
#' This is equivalent to using fastLR in the RcppNumerical package having
#'  padded predictors and the response with 0's.   
#' 
#' @param x the design matrix, with predictors as columns.
#' @param y the response vector.
#' @param padding an integer defining the padding level to be used (default is zero).
#' @return vector of coefficients.
#' 
#' @export
basic_padded_logistic_reg<-function(x,y,padding = 0){
  
  stopifnot(is.integer(padding))
  
  return(basic_padding_logistic_reg_(x, y,padding))
}