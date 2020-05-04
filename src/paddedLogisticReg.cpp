#include <Rcpp.h>
#include <RcppNumerical.h>
#include <Rmath.h>

using namespace Rcpp;
using namespace Numer;

//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::depends(RcppNumerical)]]

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;

class paddedLogisticReg: public MFuncGrad
{
private:
  const MapMat X;
  const MapVec Y;
  const int padding;
public:
  paddedLogisticReg(const MapMat x_, const MapVec y_, const int pad_ = 0) : X(x_), Y(y_), padding(pad_) {} //constructor
  
  double f_grad(Constvec& beta, Refvec grad)
  {
    // Negative log likelihood
    //   sum(log(1 + exp(X * beta))) - y' * X * beta
    
    Eigen::VectorXd xbeta = X * beta;
    const double yxbeta = Y.dot(xbeta);
    // X * beta => exp(X * beta)
    xbeta = xbeta.array().exp();
    
    // relies on the intercept being the first entry of beta.
    double paddingterm = ( padding > 0 ) ? padding*R::log1pexp(beta[0]) : 0.0 ;
    
    const double f = (xbeta.array() + 1.0).log().sum() - yxbeta + paddingterm;
    // Gradient
    //   X' * (p - y), p = exp(X * beta) / (1 + exp(X * beta))
    
    // exp(X * beta) => p
    xbeta.array() /= (xbeta.array() + 1.0);
    grad.noalias() = X.transpose() * (xbeta - Y);
    
    if(padding>0) grad[0] +=padding*( 1.0/(1.0+exp(-beta[0]))) ; 
    
    return f;
  }
};

// [[Rcpp::export]]
Rcpp::NumericVector padding_logistic_reg_(Rcpp::NumericMatrix x, Rcpp::NumericVector y,int padding = 0)
{
  const MapMat xx = Rcpp::as<MapMat>(x);
  const MapVec yy = Rcpp::as<MapVec>(y);
  // Negative log likelihood
  paddedLogisticReg nll(xx, yy, padding);
  // Initial guess
  Eigen::VectorXd beta(xx.cols());
  beta.setZero();
  
  double fopt;
  int status = optim_lbfgs(nll, beta, fopt);
  if(status < 0)
    Rcpp::stop("fail to converge");
  
  return Rcpp::wrap(beta);
}
