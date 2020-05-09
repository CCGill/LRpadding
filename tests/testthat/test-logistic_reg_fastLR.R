test_that('logistic reg function agrees with fastLR',{
  set.seed(123)
  result <- lapply( 1:ncol(TEST1_data$inputs$Response), function(q){
    logistic_reg(TEST1_data$inputs$model,
                 TEST1_data$inputs$Response[,q])
  })
  for(i in 1:ncol(TEST1_data$inputs$Response)){
    ## test ll
    expect_equal(TEST1_data$expected_results[[i]]$loglikelihood,result[[i]]$loglikelihood)
    ## test linear predictors
    expect_equal(TEST1_data$expected_results[[i]]$linear.predictors,result[[i]]$linear.predictors)
    ## test fitted values
    expect_equal(TEST1_data$expected_results[[i]]$fitted.values,result[[i]]$fitted.values)
    ## coefficients
    expect_equal(TEST1_data$expected_results[[i]]$coefficients,result[[i]]$coefficients)
  }
})
