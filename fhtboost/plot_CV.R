#' @export

plot_CV <- function(CV_errors_K) {
  plot(rowMeans(CV_errors_K), typ='l', lty=1,
       xlab = "Counter of iteration", ylab = "Mean CV")
  Ks <- dim(CV_errors_K)[2]
  for (k in 1:Ks) {
    lines(CV_errors_K[, k], lty=3)
  }
}
