
### These are baseline functions to optimize.  
### Expect that c++ version will be much faster and that
### changing the current padded logistic regression objects to output predictors
### and log likelihood will further speed things up.

partial_elbo_predictors <- function(vec,predictors){
  return( -sum( vec*qgam::log1pexp(-predictors) + (1-vec)*qgam::log1pexp(predictors)  )) 
}

simpler_log_likelihood <- function(vec,predictors){
  return(sum(vec*predictors - qgam::log1pexp(predictors)))
}



baseline_update_log_reg_coeffs<-function(model,
                                responsemat,
                                oldcoeffmat,
                                padding_zeros = 0,
                                cols_to_update){
### A function to fit a logistic regression model and
### improve on the given coefficients.
  
  ## iterate through the columns of responsemat, fit a logistic regression, and 
  ## if we improve the log-likelihood then record in the coeffmat, return coeffmat
  
  coeffmat = oldcoeffmat ## shallow-copy the coefficient matrix
  
  old_XB = model*oldcoeffmat
  
  loglikelihoods = colSums(responsemat*old_XB)  - colSums(qgam::log1pexp(old_XB))
  
  for(i in cols_to_update){
    
    tmp_coeffs <- basic_padded_logistic_reg(model,responsemat[,i],padding = padding_zeros)
    newll<-simpler_log_likelihood(responsemat[,i],model%*%tmp_coeffs)
    if(loglikelihoods[i] < newll){
      print('update accepted')
      coeffmat[,i] = tmp_coeffs
      loglikelihoods[i] = newll
    }else{  
      print('update rejected')
    }
  }
  print('logistic regression update finished')
  
  return(list(coeffs = coeffmat,
              loglikelihoods = loglikelihoods,
              predictors = model%*%coeffmat))
  
}
