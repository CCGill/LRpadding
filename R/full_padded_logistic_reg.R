
#' Padded Logistic Regression
#' 
#' padded_logistic_reg function that performs padded logistic regression.  
#' This is equivalent to using fastLR in the RcppNumerical package having
#'  padded predictors and the response with 0's. 
#' 
#' @param x the design matrix, with predictors as columns.
#' @param y the response vector.
#' @param start The initial guess of the coefficient vector.
#' @param eps_f Iteration stops if \eqn{|f-f'|/|f|<\epsilon_f}{|f-f'|/|f|<eps_f},
#'              where \eqn{f} and \eqn{f'} are the current and previous value
#'              of the objective function (negative log likelihood) respectively.
#' @param eps_g Iteration stops if
#'              \eqn{||g|| < \epsilon_g * \max(1, ||\beta||)}{||g|| < eps_g * max(1, ||beta||)},
#'              where \eqn{\beta}{beta} is the current coefficient vector and
#'              \eqn{g} is the gradient.
#' @param maxit Maximum number of iterations.
#' @param padding Number of zeros to pad the response and predictors with.
#' 
#' @return a list with the following components:
#' \item{coefficients}{Coefficient vector}
#' \item{fitted.values}{The fitted probability values}
#' \item{linear.predictors}{The fitted values of the linear part, i.e.,
#'                          \eqn{X\hat{\beta}}{X * beta_hat}}
#' \item{loglikelihood}{The maximized log likelihood}
#' \item{converged}{Whether the optimization algorithm has converged}
#'   
#' @export
padded_logistic_reg<-function(x,
                       y,
                       padding =0,
                       start= rep(0,ncol(x)),
                       eps_f = 1e-8,
                       eps_g = 1e-5,
                       maxit = 300){
  #print('Calling Rcpp function logistic_reg_')
  return(padded_logistic_reg_(x, y,padding,start,eps_f,eps_g,maxit))
}