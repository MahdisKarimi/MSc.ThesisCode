#' Simulate FHT data
#'
#' Simulate FHT data
#'
#' @return \code{observations} A list of \code{survival_times} and corresponding \code{delta}
#' @return \code{true_parameters} A list of \code{beta} and \code{gamma}
#' @return \code{design_matrices} A list of \code{X} and \code{Z}
#'
#' @keywords keywords
#'
#' @export
#'
#' @examples
#' result <- generate_test_data()
#' observations <- result$observations
#' times <- observations$survival_times
#' delta <- observations$delta
#' # make

simulate_FHT_data <- function(N=1000, setup_type='small_dense', add_noise=FALSE, seed=2) {
  library(statmod)
  set.seed(seed)
  if (setup_type == 'small_dense') {
    #N <- 1000
    # y0, beta, X
    beta_ <- c(2, 0.1, 0.2)
    #beta_ <- c(4.6, 0.1, 0.05)
    d <- length(beta_)
    X0 <- rep(1, N)
    #X1 <- cbind(c(rep(1, 300), rep(2, 300), rep(-0.5, 400)))
    X1 <- scale(rnorm(N))
    X2 <- scale(rnorm(N))
    X_design_matrix <- cbind(X0, X1, X2)

    # mu, gamma, Z
    # with intercept and normalization
    gamma_ <- c(-1, -0.1, 0.1)
    p <- length(gamma_)
    Z0 <- rep(1, N)
    Z1 <- rnorm(N)
    Z2 <- rnorm(N)
    Zrest <- scale(cbind(Z1, Z2))
    Z_design_matrix <- cbind(Z0, Z1, Z2)

    exponential_rate <- 0.1
  } else if (setup_type == 'small_sparse') {
    beta_ <- c(2, rep(0, 10), rep(0.1, 10))
    d <- length(beta_)
    X0 <- rep(1, N)
    # rbinom(100, 1, 0.5) -- bernoulli
    Xrest <- 4*matrix(rbeta((d-1)*N, shape1=1, shape2=1), ncol=(d-1))
    # center and scale
    Xrest <- scale(Xrest)
    X_design_matrix <- cbind(X0, Xrest)

    gamma_ <- c(-1, rep(-0.1, 10), rep(0, 10))
    p <- length(gamma_)
    Z0 <- rep(1, N)
    Zrest <- 4*matrix(rbeta((p-1)*N, shape1=1, shape2=1), ncol=(p-1))
    # center and scale
    Zrest <- scale(Zrest)
    Z_design_matrix <- cbind(Z0, Zrest)
    exponential_rate <- 0.1
  } else if (setup_type == 'huge') {
    huge_d <- 10000
    informative_d <- 35
    beta_ <- c(2, rep(0.1, informative_d), rep(0, huge_d-informative_d))
    d <- length(beta_)
    X0 <- rep(1, N)
    # rbinom(100, 1, 0.5) -- bernoulli
    Xrest <- 4*matrix(rbeta((d-1)*N, shape1=1, shape2=1), ncol=(d-1))
    # center and scale
    Xrest <- scale(Xrest)
    X_design_matrix <- cbind(X0, Xrest)

    informative_p <- 5
    total_p <- 15
    gamma_ <- c(-1, rep(-0.1, informative_p), rep(0, total_p-informative_p))
    p <- length(gamma_)
    Z0 <- rep(1, N)
    Zrest <- 4*matrix(rbeta((p-1)*N, shape1=1, shape2=1), ncol=(p-1))
    # center and scale
    Zrest <- scale(Zrest)
    Z_design_matrix <- cbind(Z0, Zrest)
    exponential_rate <- 0.01
  } else if (setup_type == 'huge_clinical') {
    huge_d <- 10000
    informative_d <- 35
    beta_ <- c(2, rep(0.1, informative_d), rep(0, huge_d-informative_d))
    d <- length(beta_)
    X0 <- rep(1, N)

    informative_p <- 5
    total_p <- 15
    gamma_ <- c(-1, rep(-0.1, informative_p), rep(0, total_p-informative_p))
    p <- length(gamma_)
    Z0 <- rep(1, N)

    correlated <- generate_clinical(n.obs=N, n.clin=total_p, tot.genes=huge_d)

    Xrest <- scale(correlated$gene)
    X_design_matrix <- cbind(X0, Xrest)

    Zrest <- scale(correlated$clin)
    Z_design_matrix <- cbind(Z0, Zrest)
    exponential_rate <- 0.03 # 20% censoring
  } else if (setup_type == 'correlated') {
    huge_d <- 10000 # number of genes
    # 1000 blocks, 10 in each
    number_of_blocks <- 10
    correlation_in_gene <- rep(0.7, number_of_blocks)
    correlation_in_clinical <- rep(0.7, number_of_blocks)
    correlation_between_clinical_gene <- rep(0.7, number_of_blocks)
    number_of_genes_in_block <- rep(10, number_of_blocks)
    number_of_clinical_in_block <- c(rep(2, 5), rep(3, 5))
    correlated <- generate_clinical(
      n.obs=N, n.clin=number_of_clinical_in_block, n.gene=number_of_genes_in_block, n.groups=number_of_blocks,
      tot.genes=huge_d, rho.g=correlation_in_gene, rho.b=correlation_between_clinical_gene,
      rho.c=correlation_in_clinical
    )
    X0 <- rep(1, N)
    Xrest <- scale(correlated$gene)
    X_design_matrix <- cbind(X0, Xrest)
    Z0 <- rep(1, N)
    Zrest <- scale(correlated$clin)
    Z_design_matrix <- cbind(Z0, Zrest)

    # X (gene data)
    informative_d <- 35
    non_informative_d <- huge_d-informative_d
    beta_size <- 0.1
    beta_block_with_three_informative <- c(beta_size, beta_size, beta_size, rep(0, 7))
    beta_block_with_four_informative <- c(beta_size, beta_size, beta_size, beta_size, rep(0, 6))
    beta_ <- c(
      2,
      beta_block_with_three_informative,
      beta_block_with_four_informative,
      beta_block_with_three_informative,
      beta_block_with_four_informative,
      beta_block_with_three_informative,
      beta_block_with_four_informative,
      beta_block_with_three_informative,
      beta_block_with_four_informative,
      beta_block_with_three_informative,
      beta_block_with_four_informative,
      rep(0, 9900)
    )

    # Z (clinical)
    informative_p <- 10
    total_p <- 25
    gamma_size <- -0.1
    gamma_block_size_two <- c(gamma_size, 0)
    gamma_block_size_three <- c(gamma_size, 0, 0)
    gamma_ <- c(
      -0.5,
      rep(gamma_block_size_two, 5),
      rep(gamma_block_size_three, 5)
    )
    p <- length(gamma_)

    exponential_rate <- 0.03 # 20% censoring
  } else {
    stop("Nonexistant setup type!")
  }

  noise1 <- rep(0, N)
  noise2 <- rep(0, N)
  if (add_noise) {
    noise1 <- rnorm(N, mean=0, sd=0.1)
    noise2 <- rnorm(N, mean=0, sd=0.3)
  }
  y0_pre_noise <- exp(X_design_matrix %*% beta_)
  y0 <- y0_pre_noise * exp(noise1)
  mu_pre_noise <- Z_design_matrix %*% gamma_
  mu <- mu_pre_noise + noise2
  sigma_2 <- 1 ## NB

  # Transform parameters
  mu_IG <- y0/(-mu)
  lambda_IG <- y0^2/sigma_2

  # Draw survival times and censoring times
  survival_times_not_censored <- statmod::rinvgauss(N, mean=mu_IG, shape=lambda_IG)

  #censoring_times <- statmod::rinvgauss(N, mean=abs(mu_IG*2), shape=lambda_IG)
  censoring_times <- rexp(N, rate=exponential_rate) # 1 or N times?

  # plot(survival_times_not_censored)
  # points(censoring_times, col='red')

  observations <- censor_observations(survival_times_not_censored, censoring_times)
  censored_survival_times <- observations$times
  observed <- observations$delta
  observations <- list(survival_times=censored_survival_times, delta=observed)
  true_parameters <- list(beta=beta_, gamma=gamma_)
  design_matrices <- list(X_design_matrix=X_design_matrix, Z_design_matrix=Z_design_matrix)
  return(list(
    observations=observations,
    true_parameters=true_parameters,
    design_matrices=design_matrices
  ))
}
