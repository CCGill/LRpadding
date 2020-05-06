#ifndef BASICLOG_REG
#define BASICLOG_REG


#include <RcppNumerical.h>
#include <Rmath.h>
#include <RcppEigen.h>
#include <Rcpp.h>

//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::depends(RcppNumerical)]]


using namespace Rcpp;
using namespace Numer;

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;
typedef Eigen::Ref<Eigen::MatrixXd> RefMat;
typedef Eigen::Ref<Eigen::VectorXd> RefVec;

class basicLogisticReg: public MFuncGrad
{
private:
  const Eigen::MatrixXd X;
  const Eigen::VectorXd Y;
public:
  basicLogisticReg(const Eigen::Ref<Eigen::MatrixXd> x_, const Eigen::Ref<Eigen::VectorXd> y_);
  double f_grad(Constvec& beta, Refvec grad);
};






class basicPaddedLogisticReg: public MFuncGrad
{
private:
  const MapMat X;
  const MapVec Y;
  const int padding;
public:
  basicPaddedLogisticReg(const MapMat x_, const MapVec y_, const int pad_ = 0);
  double f_grad(Constvec& beta, Refvec grad);
};







#endif