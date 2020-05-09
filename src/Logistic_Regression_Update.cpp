#include "LR.h"
//#include <assert.h>

double LRloglikelihood(RefVec response,RefVec predictor,double intercept, int padding = 0){
  
  double tmp = 0.0;
  
  for(int i=0 ; i<response.size() ; i++)
    tmp += R::log1pexp(predictor[i]);
  
  if(padding > 0)
    tmp += padding*R::log1pexp(intercept);
  
  return (response.array()*predictor.array()).sum() - tmp ;
}

void update_logreg_coeffs_(RefMat model,
                           RefMat responsemat,
                           RefMat coeffmat,
                           RefMat Xb,
                           RefMat fitted,
                           RefVec cols_to_update,
                           int padding_zeros = 0)
{// note that this function is perfect for parallelizing...
  // come back to this.  We have also written this with a Ref class for parameters
  // so that it can be used in both c++ code behind the scenes, or with  or 
  



// we setup parameters to match the fastLR defaults.
  int maxit = 300;
  double eps_f = 1e-8;
  double eps_g = 1e-5;
  
  Eigen::VectorXd betavec(coeffmat.rows());
  
  
  for(int i =0 ; i<coeffmat.cols(); i++){
    if(cols_to_update[i]>0){
      
      betavec.setZero(); // set start vector to zero...
      
      double old_loglikelihood = 0.0;
      
      // calculate loglikelihood for comparison
      
      old_loglikelihood = LRloglikelihood(responsemat.col(i),
                                          Xb.col(i),
                                          coeffmat(0,i),
                                          padding_zeros);
      
      // fit new Logistic Regression with padding
      
      PaddedLogisticReg nll(model, responsemat.col(i), padding_zeros);
      double fopt;
      int status = optim_lbfgs(nll, betavec, fopt, maxit, eps_f, eps_g);
      if(status < 0)
        Rcpp::warning("algorithm did not converge");
      
      if(-fopt > old_loglikelihood){ // optimization improved on previous estimate
        Rcout << " Update accepted"<< std::endl;
        coeffmat.col(i) = betavec; // copy coefficients to matrix
        fitted.col(i) = nll.current_p();
        Xb.col(i) = nll.current_xb();
      }else{
        Rcout<< "Update rejected" << std::endl;
      }
    }  
  }
}


//  Next an Rcpp wrapper to make this available to R for testing.


//' LR Update Function
//'
//' Runs Logistic Regression on each response with the model and updates the
//' coefficients, predictors, and fitted values when the log-likelihood has
//' been improved.  Updates values in place.
//'
//' @param model  - model matrix.
//' @param responsemat matrix of response vectors in columns.
//' @param coeffmat initial coefficient matrix.
//' @param Xb matrix of linear predictors, one column for each response.
//' @param fitted matrix of fitted probabilities.
//' @param cols_to_update vector of zeros and ones (doubles) indicating which
//'  columns to update.
//' @param padding_zeros integer number of padding zeros to add to the response
//'  and predictors to update.
//'
//' @return list of coefficient matrix, predictor matrix, fitted_values matrix,
//' vector of old loglikelihoods (from inputs) and vector on new log likelihoods
//' based on the returned coefficients.
//' @export
//[[Rcpp::export]]   
void update_logreg_coeffs(Eigen::Map<Eigen::MatrixXd> model,
                          Eigen::Map<Eigen::MatrixXd> responsemat,
                          Eigen::Map<Eigen::MatrixXd> coeffmat,
                          Eigen::Map<Eigen::MatrixXd> Xb,
                          Eigen::Map<Eigen::MatrixXd> fitted,
                          Eigen::Map<Eigen::VectorXd> cols_to_update,
                          int padding_zeros = 0){
  
  update_logreg_coeffs_(model,
                        responsemat,
                        coeffmat,
                        Xb,
                        fitted,
                        cols_to_update,
                        padding_zeros );
  
}

  
  
  
  
  