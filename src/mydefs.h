#ifndef MYDEFS
#define MYDEFS


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



#endif