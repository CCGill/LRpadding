#include <Rcpp.h>
#include <RcppNumerical.h>
#include <Rmath.h>

using namespace Rcpp;
using namespace Numer;

//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::depends(RcppNumerical)]]

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;
typedef Eigen::Map<Eigen::MatrixXd> RefMat;
typedef Eigen::Map<Eigen::VectorXd> RefVec;

class basicPaddedLogisticReg: public MFuncGrad
{
private:
  const MapMat X;
  const MapVec Y;
  const int padding;
public:
  basicPaddedLogisticReg(const MapMat x_, const MapVec y_, const int pad_ = 0) : X(x_), Y(y_), padding(pad_) {} //constructor
  
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


class PaddedLogisticReg: public MFuncGrad
{
private:
  const RefMat X;
  const RefVec Y;
  const int n; // length of response and number of rows of model matrix X
  Eigen::VectorXd xbeta; // linear predictors
  Eigen::VectorXd prob; // 
  const int padding;
  
public:
  PaddedLogisticReg(const RefMat x_, const RefVec y_,const int padding_) : //constructor
  X(x_),
  Y(y_),
  n(X.rows()),
  xbeta(n),
  prob(n),
  padding(padding_)
  {}
  
  double f_grad(Constvec& beta, Refvec grad) 
  { // this function evaluates the negative log-likelihood, while 
    //updating the gradient
    
    // Negative log likelihood
    //   sum(log(1 + exp(X * beta))) - y' * X * beta
    
    xbeta.noalias() = X * beta; // noalias updates xbeta without intermediate copies
    const double yxbeta = Y.dot(xbeta);
    
    for( int i =0; i<n ; i++){
      prob[i] = R::log1pexp(xbeta[i]); // avoids overflow
    }
    //at this point, note the returned prob vector is (xbeta - prob), updated later.
    
    // relies on the intercept being the first entry of beta.
    double paddingterm = ( padding > 0 ) ? padding*R::log1pexp(beta[0]) : 0.0 ;
    const double ret_value = prob.sum() - yxbeta + paddingterm ;
    
    //finally, update the prob and grad values for the optimization.
    
    // Gradient
    //   X' * (p - y), p = exp(X * beta) / (1 + exp(X * beta))
    //                   = exp(X*beta - log(1+exp(X*beta)))
    prob = (xbeta - prob).array().exp();
    
    grad.noalias() = X.transpose() * (prob - Y);
    if(padding>0) grad[0] +=padding*( 1.0/(1.0+exp(-beta[0]))) ;
    
    return ret_value;
  }
  
  Eigen::VectorXd current_xb() const { return xbeta; }
  Eigen::VectorXd current_p() const { return prob; }
  
};



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

