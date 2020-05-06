#include "LR.h"
#include "basicLR.h"





// [[Rcpp::export]]
Rcpp::NumericVector basic_padding_logistic_reg_(Rcpp::NumericMatrix x, Rcpp::NumericVector y,int padding = 0)
{
  const MapMat xx = Rcpp::as<MapMat>(x);
  const MapVec yy = Rcpp::as<MapVec>(y);
  // Negative log likelihood
  basicPaddedLogisticReg nll(xx, yy, padding);
  // Initial guess
  Eigen::VectorXd beta(xx.cols());
  beta.setZero();
  
  double fopt;
  int status = optim_lbfgs(nll, beta, fopt);
  if(status < 0)
    Rcpp::stop("fail to converge");
  
  return Rcpp::wrap(beta);
}

// [[Rcpp::export]]
Rcpp::List padded_logistic_reg_(Rcpp::NumericMatrix x,
                                Rcpp::NumericVector y,
                                int padding,
                                Rcpp::NumericVector start,
                                double eps_f, double eps_g, int maxit)
{
  const MapMat xx = Rcpp::as<MapMat>(x);
  const MapVec yy = Rcpp::as<MapVec>(y);
  // Negative log likelihood
  PaddedLogisticReg nll(xx, yy, padding);
  // Initial guess
  Rcpp::NumericVector b = Rcpp::clone(start);
  MapVec beta(b.begin(), b.length());
  
  double fopt;
  int status = optim_lbfgs(nll, beta, fopt, maxit, eps_f, eps_g);
  if(status < 0)
    Rcpp::warning("algorithm did not converge");
  
  return Rcpp::List::create(
    Rcpp::Named("coefficients")      = b,
    Rcpp::Named("fitted.values")     = nll.current_p(),
    Rcpp::Named("linear.predictors") = nll.current_xb(),
    Rcpp::Named("loglikelihood")     = -fopt,
    Rcpp::Named("converged")         = (status >= 0)
  );
}

