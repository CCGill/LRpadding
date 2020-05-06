#ifndef LOG_REG
#define LOG_REG

#include "mydefs.h" // includes typedefs and RcppNumerical, RcppEigen etc.




/* Full LR class */

class LogisticReg: public MFuncGrad
{
private:
  const RefMat X;
  const RefVec Y;
  const int n; // length of response and number of rows of model matrix X
  Eigen::VectorXd xbeta; // linear predictors
  Eigen::VectorXd prob; // 
  
public:
  LogisticReg(const RefMat x_, const RefVec y_); //constructor
  double f_grad(Constvec& beta, Refvec grad); 
  Eigen::VectorXd current_xb() const;
  Eigen::VectorXd current_p() const;
};


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
  PaddedLogisticReg(const RefMat x_, const RefVec y_,const int padding_ =0);
  double f_grad(Constvec& beta, Refvec grad); 
  Eigen::VectorXd current_xb() const ; 
  Eigen::VectorXd current_p() const ; 
  
};



#endif