test_that("padding is equivalent to naive implementation, full", {
  
  test_padding_equivalence<-function(myseed){
    set.seed(myseed)
    
    n = 5000
    p = 100L
    x = matrix(rnorm(n * p), n)
    x = cbind(rep(1,n),scale(x,scale = F,center = T)) ##attach the intercept
    beta = runif(p+1)
    xb = c(x %*% beta)
    probs = 1 / (1 + exp(-xb))
    y = rbinom(n, 1, probs)
    testpadding = 10000L
    
    padded_x <- rbind(x,t(matrix( rep.int(c(1,0), c(1,p)), p+1, testpadding)))
    padded_y <- c(y,rep(0,testpadding))
    
    res1 <- logistic_reg(padded_x, padded_y)
    res2 <- padded_logistic_reg(x,y,padding = testpadding)
    return(res1$coefficients - res2$coefficients)
  }
  expect_equal(test_padding_equivalence(myseed = 123),rep(0,101))
  expect_equal(test_padding_equivalence(myseed = 42),rep(0,101))
  expect_equal(test_padding_equivalence(myseed = 7531),rep(0,101))
})
