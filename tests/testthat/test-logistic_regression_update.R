test_that("Logistic Regression Update Code agrees with R implementation", {
  mytest<-function(seed){
    ## this test relies on a function defined in R which is already well tested, so
    ## we use that R function as the baseline.
    numcomps = 3
    numresp = 20000
    numinds = 500
    set.seed(seed)
    Responsemat<-matrix(runif(numresp*numcomps),numresp,numcomps) # this is our S matrix
    
    modelmat<-cbind(rep(1,numresp),matrix(rnorm(numresp*numinds),numresp,numinds)) # this is X incl. intercept
    coeffmat<-matrix(rnorm(numcomps*(numinds+1)),numinds+1,numcomps) # betamat, a column for each component.
    
    
    predictor<-modelmat %*%coeffmat # Xb
    fitted = matrix(c(-1),numresp,numcomps)
    nonzerocoeffs<-rep(1,numcomps)
    nonzerocoeffs[2] = 0
    TestVec<-coeffmat[,2]
    paddingtest = 40000L
    set.seed(seed+123)
    res<-baseline_update_log_reg_coeffs(modelmat,
                                        Responsemat,
                                        coeffmat,
                                        predictor,
                                        fitted,
                                        paddingtest,
                                        nonzerocoeffs)
    set.seed(seed+123)
    update_logreg_coeffs(modelmat,
                         Responsemat,
                         coeffmat,
                         predictor,
                         fitted,
                         nonzerocoeffs,
                         paddingtest)
    # check dimensions
    #coeffs
    expect_equal(numcomps,ncol(coeffmat))
    expect_equal(numcomps,ncol(res$coeffs))
    expect_equal(numinds+1,nrow(coeffmat))
    expect_equal(numinds+1,nrow(res$coeffs))
    #predictor
    expect_equal(numcomps,ncol(predictor))
    expect_equal(numcomps,ncol(res$predictors))
    expect_equal(numresp,nrow(predictor))
    expect_equal(numresp,nrow(res$predictors))
    #fitted
    expect_equal(numcomps,ncol(fitted))
    expect_equal(numcomps,ncol(res$fitted_values))
    expect_equal(numresp,nrow(fitted))
    expect_equal(numresp,nrow(res$fitted_values))
    #check for NA
    expect_equal(sum(unlist(lapply(res,function(x){any(is.na(x))}))),0)
    expect_equal(sum(unlist(lapply(list(fitted,predictor,coeffmat),function(x){any(is.na(x))}))),0)
    #Check for NaN
    expect_equal(sum(unlist(lapply(res,function(x){any(is.nan(x))}))),0)
    expect_equal(sum(unlist(lapply(list(fitted,predictor,coeffmat),function(x){any(is.nan(x))}))),0)
    
  expect_equal(coeffmat,res$coeffs)
  expect_equal(coeffmat[,2],TestVec)
  expect_equal(max(fitted[,2]),-1)
  expect_equal(max(fitted[,2]),-1)
  expect_gte(min(fitted[,-c(2)]),0)
  expect_lte(min(fitted[,-c(2)]),1)
  expect_equal(predictor,res$predictors)
  expect_true(all(res$loglikelihoods>=res$old_loglikelihoods))
  }
  for(seed in c(123,42)){
    mytest(seed)
  }
})
