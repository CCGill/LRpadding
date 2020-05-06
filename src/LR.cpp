#include "LR.h"

LogisticReg::LogisticReg(const RefMat x_, const RefVec y_) : //constructor
  X(x_),
  Y(y_),
  n(X.rows()),
  xbeta(n),
  prob(n)
{}

double LogisticReg::f_grad(Constvec& beta, Refvec grad) 
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
  const double ret_value = prob.sum() - yxbeta;
  
  //finally, update the prob and grad values for the optimization.
  
  // Gradient
  //   X' * (p - y), p = exp(X * beta) / (1 + exp(X * beta))
  //                   = exp(X*beta - log(1+exp(X*beta)))
  prob = (xbeta - prob).array().exp();
  
  grad.noalias() = X.transpose() * (prob - Y);
  
  return ret_value;
}

Eigen::VectorXd LogisticReg::current_xb() const { return xbeta; }

Eigen::VectorXd LogisticReg::current_p() const { return prob; }






PaddedLogisticReg::PaddedLogisticReg(const RefMat x_,
                                     const RefVec y_,
                                     const int padding_) : //constructor
  X(x_),
  Y(y_),
  n(X.rows()),
  xbeta(n),
  prob(n),
  padding(padding_)
{}

double PaddedLogisticReg::f_grad(Constvec& beta, Refvec grad) 
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

Eigen::VectorXd PaddedLogisticReg::current_xb() const { return xbeta; }
Eigen::VectorXd PaddedLogisticReg::current_p() const { return prob; }
  



