test_that('logistic regression update agrees with baseline implementation',
          {
            set.seed(123)
            update_logreg_coeffs(
              TEST2_data$inputs$model,
              TEST2_data$inputs$Response,
              TEST2_data$inputs$coeffs,
              TEST2_data$inputs$predictor,
              TEST2_data$inputs$fitted,
              TEST2_data$inputs$nonzerocoeffs,
              TEST2_data$inputs$padding
            )
            ## note that updates here are by reference only (i.e updated in place)
            expect_equal(TEST2_data$expected_results$coeffs,
                         TEST2_data$inputs$coeffs)
            expect_equal(TEST2_data$expected_results$fitted_values,
                         TEST2_data$inputs$fitted)
            expect_equal(TEST2_data$expected_results$predictors,
                         TEST2_data$inputs$predictor)
            expect_equal(max(TEST2_data$expected_results$fitted_values[, TEST2_data$inputs$nonzerocoeffs <
                                                                         0.5]), -1)
          })