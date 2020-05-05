
<!-- README.md is generated from README.Rmd. Please edit that file -->
LRpadding
=========

<!-- badges: start -->
<!-- badges: end -->
LRpadding is a very short term project to implement a simple padding functionality for logistic regression, allowing the concatenation of arbitrarily many zeros on both the predictors and the response. This is part of a much larger project, this code is entirely for testing and development.

Without an intercept, padding in this way would result in no effect (simply multiplying the likelihood by a constant), so we assume we include an intercept as the first column of the model matrix.

Installation
------------

You can install the current version of LRpadding from GitHub with:

    remotes::install_github('CCGill/LRpadding')

We make use of the excellent RcppNumerical package (<http://cran.r-project.org/package=RcppNumerical>), and a simple example version of the RcppNumerical function fastLR, which is built on the L-BFGS algorithm for unconstrained minimization problems based on the LBFGS++ library.

Example
-------

This is an example demonstrating that a logistic\_reg implementation on padded response/model agrees with the implementation of the padding. It is based on a unit test for this package. Note that the basic\_logistic\_reg function is currently identical to the example version of fastLR from the RcppNumerical package, returning only the coefficients, and logistic\_reg is identical to the full fastLR function.

We also show that our result closely matches the corresponding glm call.

``` r
library(LRpadding)
## basic example code
set.seed(42)
    
    n = 20000
    p = 100L
    x = matrix(rnorm(n * p), n)
    x = cbind(rep(1,n),scale(x,scale = F,center = T)) ##attach the intercept
    beta = runif(p+1)
    xbeta = c(x %*% beta)
    probs = 1 / (1 + exp(-xbeta))
    y = rbinom(n, 1, probs)
    testpadding = 40000L
  
    padded_x <- rbind(x,t(matrix( rep.int(c(1,0), c(1,p)), p+1, testpadding)))
    padded_y <- c(y,rep(0,testpadding))
    
    res1 <- logistic_reg(padded_x, padded_y)
    res2 <- padded_logistic_reg(x,y,padding = testpadding)
    max(abs(res1$coefficients - res2$coefficients)) ## identical results
#> [1] 3.166356e-13

    res1$loglikelihood
#> [1] -13321.19
    res2$loglikelihood
#> [1] -13321.19
    
    res3 <- basic_logistic_reg(padded_x,padded_y)
    res4 <- basic_padded_logistic_reg(x,y,padding = testpadding)
    max(abs(res3 - res4)) ## identical results
#> [1] 2.273737e-12

    max(abs(res1$coefficients - res3))
#> [1] 0.0006703128
    max(abs(res2$coefficients - res4))
#> [1] 0.0006703128
```

A second example, for time.

``` r
    set.seed(42)
    system.time(res1 <- logistic_reg(padded_x, padded_y)$coefficients )
#>    user  system elapsed 
#>   0.100   0.001   0.102
 
    system.time(res2 <- padded_logistic_reg(x,y,padding = testpadding)$coefficients)
#>    user  system elapsed 
#>   0.035   0.001   0.036

    system.time(res3 <- glm.fit(padded_x,padded_y,family = binomial())$coefficients)
#>    user  system elapsed 
#>   5.189   0.171   5.364

    max(abs(res1 - res2)) ## identical results
#> [1] 3.166356e-13

    max(abs(res2 - res3))
#> [1] 6.953141e-05
```
