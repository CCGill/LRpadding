#include "basicLR.h"

basicLogisticReg::basicLogisticReg(const Eigen::Ref<Eigen::MatrixXd> x_, const Eigen::Ref<Eigen::VectorXd> y_) : X(x_), Y(y_) {}

double basicLogisticReg::f_grad(Constvec& beta, Refvec grad)
{
  // Negative log likelihood
  //   sum(log(1 + exp(X * beta))) - y' * X * beta
  
  Eigen::VectorXd xbeta = X * beta;
  const double yxbeta = Y.dot(xbeta);
  // X * beta => exp(X * beta)
  xbeta = xbeta.array().exp();
  const double f = (xbeta.array() + 1.0).log().sum() - yxbeta;
  
  // Gradient
  //   X' * (p - y), p = exp(X * beta) / (1 + exp(X * beta))
  
  // exp(X * beta) => p
  xbeta.array() /= (xbeta.array() + 1.0);
  grad.noalias() = X.transpose() * (xbeta - Y);
  
  return f;
}



basicPaddedLogisticReg::basicPaddedLogisticReg(const MapMat x_,
                                               const MapVec y_,
                                               const int pad_) : X(x_), Y(y_), padding(pad_) {} //constructor

double basicPaddedLogisticReg::f_grad(Constvec& beta, Refvec grad)
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

