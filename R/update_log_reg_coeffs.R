
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



#' Baseline Update Function
#'
#' Runs Logistic Regression on each response with the model and updates the
#' coefficients, predictors, and fitted values when the log-likelihood has
#' been improved.
#'
#' @param model  - model matrix.
#' @param responsemat matrix of response vectors in columns.
#' @param coeffmat initial coefficient matrix.
#' @param predictors matrix of linear predictors, one column for each response.
#' @param fitted_values matrix of fitted probabilities.
#' @param integer number of padding zeros to add to the response and predictors
#' @param cols_to_update vector of zeros and ones indicating which columns 
#' to update.
#' 
#' @return list of coefficient matrix, predictor matrix, fitted_values matrix, 
#' vector of old loglikelihoods (from inputs) and vector on new log likelihoods
#' based on the returned coefficients.
#' @export
baseline_update_log_reg_coeffs<-function(model,
                                         responsemat,
                                         coeffmat,
                                         predictors,
                                         fitted_values,
                                         padding_zeros = 0L,
                                         cols_to_update){
  ### A function to fit a logistic regression model and
  ### improve on the given coefficients.
  
  ## iterate through the columns of responsemat, fit a logistic regression, and 
  ## if we improve the log-likelihood then record in the coeffmat, return coeffmat
  
  
  
  old_XB = model %*% coeffmat
  
  old_loglikelihoods = colSums(responsemat*old_XB)  - colSums(qgam::log1pexp(old_XB)) 
  if(padding_zeros>0){ old_loglikelihoods = old_loglikelihoods - padding_zeros*qgam::log1pexp(coeffmat[1,])}
  ## this qgam saves a _lot_ of time.  as does calculating all the loglikelihoods together.
  loglikelihoods = rep(0,length(old_loglikelihoods))
  print(cols_to_update)
  for(i in 1:ncol(coeffmat)){
    print(cols_to_update[i])
    if(cols_to_update[i]>0){
      print(paste0('Calling padded_logistic_reg on column ',i))
      
      tmpmodel <- padded_logistic_reg(model,responsemat[,i],padding = padding_zeros)
      
      #newll<-simpler_log_likelihood(responsemat[,i],model%*%tmp_coeffs)
      
      if(old_loglikelihoods[i] < tmpmodel$loglikelihood){
        print('update accepted')
        coeffmat[,i] = tmpmodel$coefficients
        loglikelihoods[i] = tmpmodel$loglikelihood
        fitted_values[,i] = tmpmodel$fitted.values
        predictors[,i] = tmpmodel$linear.predictors
        
      }else{  
        print('update rejected')
      }
    }
  }
  print('logistic regression update finished')
  
  return(list(coeffs = coeffmat,
              old_loglikelihoods = old_loglikelihoods,
              loglikelihoods = loglikelihoods,
              predictors = predictors,
              fitted_values = fitted_values))
  
}



