% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/update_log_reg_coeffs.R
\name{baseline_update_log_reg_coeffs}
\alias{baseline_update_log_reg_coeffs}
\title{Baseline Update Function}
\usage{
baseline_update_log_reg_coeffs(
  model,
  responsemat,
  coeffmat,
  predictors,
  fitted_values,
  padding_zeros = 0L,
  cols_to_update
)
}
\arguments{
\item{model}{- model matrix.}

\item{responsemat}{matrix of response vectors in columns.}

\item{coeffmat}{initial coefficient matrix.}

\item{predictors}{matrix of linear predictors, one column for each response.}

\item{fitted_values}{matrix of fitted probabilities.}

\item{padding_zeros}{integer number of padding zeros to add to the response
and predictors, defaults to zero.}

\item{cols_to_update}{vector of zeros and ones indicating which columns 
to update.}
}
\value{
list of coefficient matrix, predictor matrix, fitted_values matrix, 
vector of old loglikelihoods (from inputs) and vector on new log likelihoods
based on the returned coefficients.
}
\description{
Runs Logistic Regression on each response with the model and updates the
coefficients, predictors, and fitted values when the log-likelihood has
been improved.
}
