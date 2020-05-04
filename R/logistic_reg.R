
#' Logistic Regression
#' 
#' logistic_reg function that performs logistic regression, implemented here for baseline unit testing.
#' 
#' @param x the design matrix, with predictors as columns.
#' @param y the response vector.
#' @return vector of coefficients.
#' @export
logistic_reg<-function(x,y){
  #print('Calling Rcpp function logistic_reg_')
  return(logistic_reg_(x, y))
}