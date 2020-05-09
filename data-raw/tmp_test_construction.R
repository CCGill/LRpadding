### script to generate data for three tests:

### 1st, test the Logistic_Reg function against the 
###   RcppNumerical::fastLR alternative

### 2nd, test the new update rule against the baseline version 
###   we already wrote and tested the baseline version against earlier scripts


#### 1 - Logistic_Reg  ####

# note that we already test equivalence of padded and basic implementations.
rm(list = ls())
seed = 12345
numcomps = 3
numresp = 200
numinds = 100

# setup inputs
set.seed(seed)
Responsemat <-
  matrix(runif(numresp * numcomps), numresp, numcomps) # this is our S matrix

modelmat <-
  cbind(rep(1, numresp), matrix(rnorm(numresp * numinds), numresp, numinds)) # this is X incl. intercept
coeffmat <-
  matrix(rnorm(numcomps * (numinds + 1)), numinds + 1, numcomps) # betamat, a column for each component.
#saveRDS(list(coeff<-coeffmat,))

set.seed(123)
fastLR_results <- lapply(1:ncol(Responsemat), function(q) {
  tmp <- RcppNumerical::fastLR(modelmat, Responsemat[, q])
  return(tmp)
})


TEST1_data = list(
  expected_results = fastLR_results,
  inputs = list(model = modelmat,
                Response = Responsemat)
)

# #### TEST1 ####
# test_that('logistic reg function agrees with fastLR', {
#   set.seed(123)
#   result <- lapply(1:ncol(TEST1_data$inputs$Response), function(q) {
#     logistic_reg(TEST1_data$inputs$model,
#                  TEST1_data$inputs$Response[, q])
#   })
#   for (i in 1:ncol(TEST1_data$inputs$Response)) {
#     ## test ll
#     expect_equal(TEST1_data$expected_results[[i]]$loglikelihood,
#                  result[[i]]$loglikelihood)
#     ## test linear predictors
#     expect_equal(TEST1_data$expected_results[[i]]$linear.predictors,
#                  result[[i]]$linear.predictors)
#     ## test fitted values
#     expect_equal(TEST1_data$expected_results[[i]]$fitted.values,
#                  result[[i]]$fitted.values)
#     ## coefficients
#     expect_equal(TEST1_data$expected_results[[i]]$coefficients,
#                  result[[i]]$coefficients)
#   }
# })





#### 2 - the test for the full update function ####
## for another test, we can apply our full update function to check it's still working.

set.seed(12345)

numcomps = 10
numresp = 1000
numinds = 50
set.seed(seed)
Responsemat<-matrix(runif(numresp*numcomps),numresp,numcomps) # this is our S matrix

modelmat<-cbind(rep(1,numresp),matrix(rnorm(numresp*numinds),numresp,numinds)) # this is X incl. intercept
coeffmat<-matrix(rnorm(numcomps*(numinds+1)),numinds+1,numcomps) # betamat, a column for each component.


predictor<-modelmat %*%coeffmat # Xb
fitted = matrix(c(-1),numresp,numcomps)
nonzerocoeffs<-rep(1,numcomps)
#set 3 of the coefficients to not be updated
ignore_these_vec<-sample(1:numcomps,3,replace = F)
nonzerocoeffs[ignore_these_vec] = 0
sum(nonzerocoeffs==0)
table(nonzerocoeffs)


TestVec<-coeffmat[,5]
paddingtest = 40000L
set.seed(123)


system.time(res<-baseline_update_log_reg_coeffs(modelmat,
                                                Responsemat,
                                                coeffmat,
                                                predictor,
                                                fitted,
                                                paddingtest,
                                                nonzerocoeffs))

TEST2_data = list(
  expected_results = res,
  inputs = list(model = modelmat,
                Response = Responsemat,
                coeffs = coeffmat,
                predictor = predictor,
                fitted = fitted,
                nonzerocoeffs = nonzerocoeffs,
                padding = paddingtest)
)

## want to save

#### TEST2   ####
# test_that('logistic regression update agrees with baseline implementation',
#           {
#             set.seed(123)
#             update_logreg_coeffs(
#               TEST2_data$inputs$model,
#               TEST2_data$inputs$Response,
#               TEST2_data$inputs$coeffs,
#               TEST2_data$inputs$predictor,
#               TEST2_data$inputs$fitted,
#               TEST2_data$inputs$nonzerocoeffs,
#               TEST2_data$inputs$padding
#             )
#             ## note that updates here are by reference only (i.e updated in place)
#             expect_equal(TEST2_data$expected_results$coeffs,
#                          TEST2_data$inputs$coeffs)
#             expect_equal(TEST2_data$expected_results$fitted_values,
#                          TEST2_data$inputs$fitted)
#             expect_equal(TEST2_data$expected_results$predictors,
#                          TEST2_data$inputs$predictor)
#             expect_equal(max(TEST2_data$expected_results$fitted_values[, TEST2_data$inputs$nonzerocoeffs <
#                                                                          0.5]), -1)
#           })




####  Save data   #### 
# save for tests to R/sysdata.rda, available only internally, not exported.
usethis::use_data(TEST1_data,TEST2_data,internal = TRUE)

