#include <Rcpp.h>
#include <RcppNumerical.h>

using namespace Rcpp;


using namespace Numer;

//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::depends(RcppNumerical)]]

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;
typedef Eigen::Map<Eigen::MatrixXd> RefMat;
typedef Eigen::Map<Eigen::VectorXd> RefVec;