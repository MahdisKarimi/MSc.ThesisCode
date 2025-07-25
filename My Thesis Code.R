# ==============================================================================
# Comprehensive Script for Simulation, Training, and Evaluation of FHTBoost Models (Advanced Version)
# ==============================================================================

# --- 0. Initial Setup and Package Loading ---
rm(list = ls())
graphics.off()

required_packages <- c("statmod", "mvtnorm", "corpcor", "ggplot2", "tidyr", 
                       "knitr", "survival", "viridis", "progress", "dplyr")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}

# !!! PLEASE MODIFY THE PATH BELOW !!!
SOURCE_CODE_DIRECTORY <- "C:/Users/mahdis/OneDrive/Desktop/MSc. Thesis/fhtboost-master/fhtboost-master/R/" 
setwd(SOURCE_CODE_DIRECTORY)

# --- 0.1. Sourcing all function files ---
print("--- Start sourcing function files ---")
tryCatch({
  source("generate_clinical.R"); source("censor_observations.R"); source("simulate_FHT_data.R"); source("simulate_normal_data.R")
  source("FHT_link_function.R"); source("FHT_parametric_density.R"); source("FHT_parametric_survival.R"); source("FHT_parametric_cumulative_hazard.R"); source("FHT_hazard.R"); source("FHT_loglikelihood_with_data.R"); source("FHT_loglikelihood_with_y0_mu.R"); source("survival_censored.R"); source("FHT_prob_infinity.R"); source("FHT_only_intercepts.R"); source("FHT_componentwise_minus_loglikelihood_with_parameters.R"); source("FHT_minus_loglikelihood_with_all_parameters.R")
  source("parameter_vector_to_list.R"); source("data_to_optimizable_function.R"); source("data_to_FHT_loss_function.R"); source("maximum_likelihood_intercepts.R")
  source("loss_function_derivative_mu.R"); source("loss_function_derivative_y0.R"); source("FHT_componentwise_loss_function_derivative_mu.R"); source("FHT_componentwise_loss_function_derivative_y0.R"); source("best_least_squares_update.R"); source("boosting_iteration_mu.R"); source("boosting_iteration_y0.R"); source("boosting_iteration_both.R"); source("boosting_run.R"); source("cyclic_boosting_run.R"); source("run_L2Boost.R")
  source("calculate_at_risk.R"); source("kaplan_meier.R"); source("kaplan_meier_estimate_of_censoring.R"); source("non_parametric_estimates.R"); source("kaplan_meier_plot.R"); source("brier_score_no_censoring.R"); source("brier_score_with_censoring.R"); source("brier_score_with_censoring_on_times.R"); source("brier_score_with_censoring_on_times_with_probabilities.R"); source("brier_score_on_uncensored_data.R"); source("brier_score_on_censored_data.R"); source("brier_r2.R"); source("brier_r2_with_censoring_on_times.R")
  source("create_folds.R"); source("create_folds_stratified.R"); source("get_kth_fold.R"); source("get_all_but_kth_fold.R"); source("run_CV.R"); source("run_CV_cyclic_helper_mu.R"); source("run_CV_cyclic_helper_y0.R")
  source("plot_CV.R"); source("make_filename.R") 
  source("draw_IG_data.R"); source("plot_IG_times.R"); source("plot_wiener_processes.R"); source("destandardize.R"); source("estimate_baseline_hazard.R"); source("non_null_parameters.R")
  print("--- All function files sourced successfully ---")
}, error = function(e) {
  stop(paste("Error in sourcing files:", e$message))
})


# --- 0.2. Definition of run_CV_cyclic_grid_search with Checkpointing and Progress Bar ---
run_CV_cyclic_grid_search <- function(
    M_y0_max, M_mu_max, K_fold_repetitions, K, X, Z, times, delta,
    boost_intercepts_continually = FALSE, metric = "loglik",
    checkpoint_file = NULL 
) {
  
  grid_search_results <- data.frame(
    m_stop_y0_candidate = integer(),
    m_stop_mu_candidate = integer(),
    mean_cv_error = numeric()
  )
  start_mu_candidate <- 1
  
  # Restore from checkpoint if the file exists
  if (!is.null(checkpoint_file) && file.exists(checkpoint_file)) {
    print(paste("Restoring grid search results from file:", checkpoint_file))
    saved_state <- readRDS(checkpoint_file)
    grid_search_results <- saved_state$grid_search_results
    start_mu_candidate <- saved_state$last_completed_mu_candidate + 1
    print(paste("Resuming grid search from m_stop_mu_candidate =", start_mu_candidate))
  }
  
  if (start_mu_candidate > M_mu_max) {
    print("Grid search has already been completed.")
  } else {
    print(paste("Starting CV grid search for m_stop_y0 (up to", M_y0_max, ") and m_stop_mu (from",start_mu_candidate,"to", M_mu_max, ")"))
    
    pb_mu <- progress_bar$new(
      format = "Grid Search (mu): [:bar] :percent eta: :eta (:current/:total mu_stops)",
      total = M_mu_max - start_mu_candidate + 1, clear = FALSE, width = 80
    )
    
    for (current_m_stop_mu in start_mu_candidate:M_mu_max) {
      pb_mu$tick()
      cv_results_for_y0 <- NULL
      tryCatch({
        cv_results_for_y0 <- run_CV_cyclic_helper_y0(
          M = M_y0_max, M_fixed = current_m_stop_mu,
          K_fold_repetitions = K_fold_repetitions, K = K,
          X = X, Z = Z, times = times, delta = delta,
          boost_intercepts_continually = boost_intercepts_continually
        )
      }, error = function(e) { print(paste("Error in run_CV_cyclic_helper_y0 for m_stop_mu =", current_m_stop_mu, ":", e$message))})
      
      mean_errors_for_current_y0_run <- NA
      if (!is.null(cv_results_for_y0)) {
        error_matrix_name <- if (metric == "loglik") "CV_errors_K_loglik" else "CV_errors_K_deviance"
        if (!is.null(cv_results_for_y0[[error_matrix_name]])) {
          mean_errors_for_current_y0_run <- rowMeans(cv_results_for_y0[[error_matrix_name]], na.rm = TRUE)
        } else {
          warning(paste("CV metric (", metric, ") for m_stop_mu =", current_m_stop_mu, "not found in helper output."))
        }
      }
      
      if (all(is.na(mean_errors_for_current_y0_run))) {
        warning(paste("All CV errors for m_stop_mu =", current_m_stop_mu, "were NA."))
      } else {
        for (current_m_stop_y0 in 1:length(mean_errors_for_current_y0_run)) {
          grid_search_results <- rbind(grid_search_results, data.frame(
            m_stop_y0_candidate = current_m_stop_y0,
            m_stop_mu_candidate = current_m_stop_mu,
            mean_cv_error = mean_errors_for_current_y0_run[current_m_stop_y0]
          ))
        }
      }
      # Save checkpoint after each iteration of the mu loop
      if (!is.null(checkpoint_file)) {
        saveRDS(list(grid_search_results = grid_search_results, last_completed_mu_candidate = current_m_stop_mu), file = checkpoint_file)
      }
    }
    print("--- CV grid search finished ---")
  }
  
  
  if (nrow(grid_search_results) > 0 && sum(!is.na(grid_search_results$mean_cv_error)) > 0) {
    best_combination_index <- NA
    valid_results <- grid_search_results[!is.na(grid_search_results$mean_cv_error), ]
    
    if (nrow(valid_results) > 0) {
      if (metric == "loglik") {
        best_combination_index <- which.min(valid_results$mean_cv_error)
      } else if (metric == "deviance") { 
        best_combination_index <- which.max(valid_results$mean_cv_error)
      }
      
      if(is.na(best_combination_index) || length(best_combination_index) == 0) {
        warning("Best combination index not found in grid search (after filtering NAs).")
        return(NULL)
      }
      
      best_m_stop_y0 <- valid_results$m_stop_y0_candidate[best_combination_index]
      best_m_stop_mu <- valid_results$m_stop_mu_candidate[best_combination_index]
      optimal_cv_error_value <- valid_results$mean_cv_error[best_combination_index]
      
      return(list(
        best_m_stop_y0 = best_m_stop_y0,
        best_m_stop_mu = best_m_stop_mu,
        optimal_cv_error = optimal_cv_error_value,
        all_grid_results = grid_search_results 
      ))
    } else {
      warning("All grid search results had NA errors.")
      return(NULL)
    }
  } else {
    warning("No valid results were obtained in the CV grid search.")
    return(NULL)
  }
}


# ==============================================================================
# Comprehensive function to run a full scenario analysis (with progress and checkpointing)
# ==============================================================================
run_full_scenario_analysis <- function(
    scenario_name,
    N_train, N_test, setup_type, seed_train, seed_test,
    add_noise_train = FALSE, add_noise_test = FALSE,
    M_y0_cv_cyclic, M_mu_cv_cyclic, K_folds_cv_cyclic, K_reps_cv_cyclic = 1,
    boost_intercepts_cv_cyclic = TRUE, metric_cyclic = "loglik",
    M_cv_combined, K_folds_cv_combined, K_reps_cv_combined = 1,
    boost_intercepts_cv_combined = TRUE, metric_combined = "loglik",
    output_directory = "Comprehensive_Scenario_Results_V3",
    generate_plots = TRUE, generate_tables = TRUE, generate_detailed_csv = TRUE,
    num_brier_time_points = 100,
    force_rerun_cv_cyclic = FALSE, # If TRUE, reruns the cyclic CV grid search even if a checkpoint file exists
    force_rerun_cv_combined = FALSE # If TRUE, reruns the Combined CV
) {
  
  print(paste("<<<<< Starting processing for scenario:", scenario_name, ">>>>>"))
  
  if (!dir.exists(output_directory)) dir.create(output_directory, recursive = TRUE)
  scenario_output_dir <- file.path(output_directory, scenario_name)
  if (!dir.exists(scenario_output_dir)) dir.create(scenario_output_dir, recursive = TRUE)
  
  checkpoint_cv_cyclic_file <- file.path(scenario_output_dir, paste0("checkpoint_cv_cyclic_", scenario_name, ".rds"))
  checkpoint_cv_combined_file <- file.path(scenario_output_dir, paste0("checkpoint_cv_combined_", scenario_name, ".rds"))
  checkpoint_final_models_file <- file.path(scenario_output_dir, paste0("checkpoint_final_models_", scenario_name, ".rds"))
  
  # --- 1. Simulate training and test data ---
  print("--- 1.1. Simulating training data ---")
  sim_data_train <- simulate_FHT_data(N = N_train, setup_type = setup_type, add_noise = add_noise_train, seed = seed_train)
  times_train <- sim_data_train$observations$survival_times
  delta_train <- sim_data_train$observations$delta
  X_matrix_train <- sim_data_train$design_matrices$X_design_matrix
  Z_matrix_train <- sim_data_train$design_matrices$Z_design_matrix
  true_beta_train <- sim_data_train$true_parameters$beta
  true_gamma_train <- sim_data_train$true_parameters$gamma
  if(is.null(names(true_beta_train)) && length(true_beta_train)>0) names(true_beta_train) <- paste0("beta", 0:(length(true_beta_train)-1))
  if(is.null(names(true_gamma_train)) && length(true_gamma_train)>0) names(true_gamma_train) <- paste0("gamma", 0:(length(true_gamma_train)-1))
  
  censoring_ratio_train <- sum(delta_train == 0) / length(delta_train)
  print(paste("Censoring percentage in training data:", round(censoring_ratio_train * 100, 1), "%"))
  
  print("--- 1.2. Simulating test data ---")
  sim_data_test <- simulate_FHT_data(N = N_test, setup_type = setup_type, add_noise = add_noise_test, seed = seed_test)
  times_test <- sim_data_test$observations$survival_times
  delta_test <- sim_data_test$observations$delta
  X_matrix_test <- sim_data_test$design_matrices$X_design_matrix
  Z_matrix_test <- sim_data_test$design_matrices$Z_design_matrix
  
  censoring_ratio_test <- sum(delta_test == 0) / length(delta_test)
  print(paste("Censoring percentage in test data:", round(censoring_ratio_test * 100, 1), "%"))
  
  # --- 2. Strategy 1: Cyclic Boosting ---
  print(paste("--- 2.1. CV for Cyclic Boosting (Scenario:", scenario_name, ") ---"))
  optimal_m_stops_cyclic_res <- NULL
  if (force_rerun_cv_cyclic || !file.exists(checkpoint_cv_cyclic_file)) {
    tryCatch({
      optimal_m_stops_cyclic_res <- run_CV_cyclic_grid_search(
        M_y0_max = M_y0_cv_cyclic, M_mu_max = M_mu_cv_cyclic,
        K_fold_repetitions = K_reps_cv_cyclic, K = K_folds_cv_cyclic,
        X = X_matrix_train, Z = Z_matrix_train, times = times_train, delta = delta_train,
        boost_intercepts_continually = boost_intercepts_cv_cyclic, metric = metric_cyclic,
        checkpoint_file = checkpoint_cv_cyclic_file 
      )
      if(!is.null(optimal_m_stops_cyclic_res)) saveRDS(optimal_m_stops_cyclic_res, checkpoint_cv_cyclic_file)
    }, error = function(e) { print(paste("Error in run_CV_cyclic_grid_search:", e$message))})
  } else {
    print(paste("Loading Cyclic CV results from checkpoint file:", checkpoint_cv_cyclic_file))
    optimal_m_stops_cyclic_res <- readRDS(checkpoint_cv_cyclic_file)
  }
  
  optimal_m_y0_cyclic <- NA; optimal_m_mu_cyclic <- NA; cv_metric_value_cyclic <- NA
  if (!is.null(optimal_m_stops_cyclic_res) && !is.null(optimal_m_stops_cyclic_res$best_m_stop_y0)) {
    optimal_m_y0_cyclic <- optimal_m_stops_cyclic_res$best_m_stop_y0
    optimal_m_mu_cyclic <- optimal_m_stops_cyclic_res$best_m_stop_mu
    cv_metric_value_cyclic <- optimal_m_stops_cyclic_res$optimal_cv_error
    
    if (generate_plots && !is.null(optimal_m_stops_cyclic_res$all_grid_results) && nrow(optimal_m_stops_cyclic_res$all_grid_results) > 0) {
      tryCatch({
        plot_data_cyclic_cv <- optimal_m_stops_cyclic_res$all_grid_results
        plot_data_cyclic_cv_valid <- plot_data_cyclic_cv[!is.na(plot_data_cyclic_cv$mean_cv_error), ]
        
        if(nrow(plot_data_cyclic_cv_valid) > 0){
          title_cyclic_cv <- paste("CV Grid - Cyclic -", scenario_name, "\nOptimal: y0=", optimal_m_y0_cyclic, "mu=", optimal_m_mu_cyclic)
          fill_direction <- ifelse(metric_cyclic=="loglik", 1, -1)
          
          p_cyclic_cv <- ggplot(plot_data_cyclic_cv_valid, aes(x = as.factor(m_stop_y0_candidate), y = as.factor(m_stop_mu_candidate), fill = mean_cv_error)) +
            geom_tile(color = "grey70", linewidth = 0.1) +
            scale_fill_viridis_c(option = "plasma", direction = fill_direction, name = paste("Mean CV", metric_cyclic), na.value = "grey90") +
            labs(x = "m_stop_y0 (β iterations)", y = "m_stop_mu (γ iterations)") +
            theme_minimal(base_size = 11) + theme(plot.title = element_text(hjust = 0.5, size=14), axis.text.x = element_text(angle = 45, hjust = 1,size = 10))
          if (!is.na(optimal_m_y0_cyclic) && !is.na(optimal_m_mu_cyclic)) {
            p_cyclic_cv <- p_cyclic_cv +
              geom_point(data = data.frame(x_opt = as.factor(optimal_m_y0_cyclic), y_opt = as.factor(optimal_m_mu_cyclic)),
                         aes(x = x_opt, y = y_opt), color = "cyan", size = 5, shape = "X", stroke=1.5, inherit.aes = FALSE)
          }
          ggsave(filename=file.path(scenario_output_dir, paste0("plot_cv_cyclic_grid_", metric_cyclic, "_", scenario_name, ".png")), plot=p_cyclic_cv, width=9, height=7, dpi=150)
        } else { print("No valid data to plot the Cyclic CV grid.")}
      }, error = function(e) { print(paste("Error plotting Cyclic CV grid:", e$message))})
    }
  } else {
    optimal_m_y0_cyclic <- M_y0_cv_cyclic %/% 2; optimal_m_mu_cyclic <- M_mu_cv_cyclic %/% 2
    print("Grid CV for Cyclic was unsuccessful or had no results, using default m_stop values.")
  }
  print(paste("Cyclic - Optimal m_y0:", optimal_m_y0_cyclic, "Optimal m_mu:", optimal_m_mu_cyclic, "CV Metric (",metric_cyclic,") at optimum:", if(!is.na(cv_metric_value_cyclic)) round(cv_metric_value_cyclic,4) else "NA"))
  
  model_cyclic_final <- NULL
  if (!is.na(optimal_m_y0_cyclic) && optimal_m_y0_cyclic > 0 && !is.na(optimal_m_mu_cyclic) && optimal_m_mu_cyclic > 0) {
    print("--- 2.2. Training final Cyclic Boosting model ---")
    model_cyclic_final <- cyclic_boosting_run(
      times = times_train, delta = delta_train, X = X_matrix_train, Z = Z_matrix_train,
      m_stop_y0 = optimal_m_y0_cyclic, m_stop_mu = optimal_m_mu_cyclic,
      boost_intercepts_continually = boost_intercepts_cv_cyclic, should_print = FALSE
    )
  } else {print(paste("Optimal m_stops for Cyclic are invalid (",optimal_m_y0_cyclic,",",optimal_m_mu_cyclic,"), model will not be trained."))}
  
  # --- 3. Strategy 2: Combined Boosting ---
  print(paste("--- 3.1. CV for Combined Boosting (Scenario:", scenario_name, ") ---"))
  cv_results_combined <- NULL
  if(force_rerun_cv_combined || !file.exists(checkpoint_cv_combined_file)){
    tryCatch({
      cv_results_combined <- run_CV(
        M = M_cv_combined, K_fold_repetitions = K_reps_cv_combined, K = K_folds_cv_combined,
        X = X_matrix_train, Z = Z_matrix_train, times = times_train, delta = delta_train,
        boost_intercepts_continually = boost_intercepts_cv_combined
      )
      if(!is.null(cv_results_combined)) saveRDS(cv_results_combined, checkpoint_cv_combined_file)
    }, error = function(e) { print(paste("Error in run_CV (Combined):", e$message))})
  } else {
    print(paste("Loading Combined CV results from checkpoint file:", checkpoint_cv_combined_file))
    cv_results_combined <- readRDS(checkpoint_cv_combined_file)
  }
  
  optimal_m_stop_combined <- NA; cv_metric_value_combined <- NA
  
  if (generate_plots && !is.null(cv_results_combined)) {
    plot_file_combined_cv <- file.path(scenario_output_dir, paste0("plot_cv_combined_all_metrics_", scenario_name, ".png"))
    png(filename=plot_file_combined_cv, width=1100, height=550)
    par(mfrow=c(1, 2), mar=c(5.1, 4.5, 4.1, 2.1), oma=c(0,0,3,0))
    
    opt_m_loglik_comb <- NA; val_loglik_comb <- NA
    if (!is.null(cv_results_combined$CV_errors_K_loglik)) {
      cv_loglik_matrix <- cv_results_combined$CV_errors_K_loglik
      if(is.matrix(cv_loglik_matrix) && nrow(cv_loglik_matrix) > 0 && ncol(cv_loglik_matrix) > 0 && sum(!is.na(cv_loglik_matrix)) > 0){
        mean_cv_loglik <- rowMeans(cv_loglik_matrix, na.rm=TRUE)
        opt_m_loglik_comb <- which.min(mean_cv_loglik)
        val_loglik_comb <- mean_cv_loglik[opt_m_loglik_comb]
        plot_CV(cv_loglik_matrix); title(main = "Combined CV - NegLogLik", cex.main=1.1)
        abline(v = opt_m_loglik_comb, col = "green4", lty = 2, lwd=2)
        mtext(paste("Optimal (LogLik):", opt_m_loglik_comb, "\nValue:", round(val_loglik_comb,3)), side=1, line=-2.5, adj=0.95, col="green4", cex=0.9)
      } else { plot(0,type='n',axes=FALSE,ann=FALSE); text(1,0,"LogLik CV data not valid", cex=1.2); title(main = "Combined CV - NegLogLik") }
    } else { plot(0,type='n',axes=FALSE,ann=FALSE); text(1,0,"LogLik CV data not available", cex=1.2); title(main = "Combined CV - NegLogLik") }
    
    opt_m_dev_comb <- NA; val_dev_comb <- NA
    if (!is.null(cv_results_combined$CV_errors_K_deviance)) {
      cv_deviance_matrix <- cv_results_combined$CV_errors_K_deviance
      if(is.matrix(cv_deviance_matrix) && nrow(cv_deviance_matrix) > 0 && ncol(cv_deviance_matrix) > 0 && sum(!is.na(cv_deviance_matrix)) > 0){
        mean_cv_deviance <- rowMeans(cv_deviance_matrix, na.rm=TRUE)
        opt_m_dev_comb <- which.max(mean_cv_deviance) # Larger Deviance is better
        val_dev_comb <- mean_cv_deviance[opt_m_dev_comb]
        plot_CV(cv_deviance_matrix); title(main = "Combined CV - Deviance", cex.main=1.1)
        abline(v = opt_m_dev_comb, col = "green4", lty = 2, lwd=2)
        mtext(paste("Optimal (Deviance):", opt_m_dev_comb, "\nValue:", round(val_dev_comb,3)), side=1, line=-2.5, adj=0.95, col="green4", cex=0.9)
      } else { plot(0,type='n',axes=FALSE,ann=FALSE); text(1,0,"Deviance CV data not valid", cex=1.2); title(main = "Combined CV - Deviance") }
    } else { plot(0,type='n',axes=FALSE,ann=FALSE); text(1,0,"Deviance CV data not available", cex=1.2); title(main = "Combined CV - Deviance") }
    dev.off()
    
    if (metric_combined == "loglik" && !is.na(opt_m_loglik_comb)) {
      optimal_m_stop_combined <- opt_m_loglik_comb
      cv_metric_value_combined <- val_loglik_comb
    } else if (metric_combined == "deviance" && !is.na(opt_m_dev_comb)) {
      optimal_m_stop_combined <- opt_m_dev_comb
      cv_metric_value_combined <- val_dev_comb
    }
  }
  if (is.na(optimal_m_stop_combined)) {
    optimal_m_stop_combined <- M_cv_combined %/% 2
    print(paste("Metric", metric_combined, "for Combined CV not found or plotting failed, using default m_stop value."))
  }
  print(paste("Combined - Optimal m_stop (based on", metric_combined, "):", optimal_m_stop_combined, "CV Metric Value:", if(!is.na(cv_metric_value_combined)) round(cv_metric_value_combined,4) else "NA"))
  
  model_combined_final <- NULL
  if (!is.na(optimal_m_stop_combined) && optimal_m_stop_combined > 0) {
    print("--- 3.2. Training final Combined Boosting model ---")
    model_combined_final <- boosting_run(
      times = times_train, delta = delta_train, X = X_matrix_train, Z = Z_matrix_train,
      m_stop = optimal_m_stop_combined,
      boost_intercepts_continually = boost_intercepts_cv_combined, should_print = FALSE
    )
  } else {print(paste("Optimal m_stop for Combined is invalid (",optimal_m_stop_combined,"), model will not be trained."))}
  

    if(!is.null(model_cyclic_final) || !is.null(model_combined_final)){
    saveRDS(list(model_cyclic = model_cyclic_final, model_combined = model_combined_final), file = checkpoint_final_models_file)
  }
  
  
  # --- 4. Generate coefficient trend tables ---
  if (generate_tables) {
    # Calculate MLE
    mle_row_df <- NULL
    tryCatch({
      minus_loglik_func_nlm <- data_to_optimizable_function(X_matrix_train, Z_matrix_train, times_train, delta_train)
      num_beta <- ncol(X_matrix_train)
      num_gamma <- ncol(Z_matrix_train)
      initial_intercepts_mle <- maximum_likelihood_intercepts(times_train, delta_train)
      initial_params_nlm <- c(initial_intercepts_mle[1], rep(0.01, num_beta - 1), initial_intercepts_mle[2], rep(0.01, num_gamma - 1))
      mle_res <- nlm(minus_loglik_func_nlm, initial_params_nlm, iterlim = 500, print.level = 0)
      if (mle_res$code <= 2) {
        mle_params_vec <- mle_res$estimate
        mle_params_list_s <- parameter_vector_to_list(mle_params_vec, num_beta, num_gamma)
        mle_row_df <- data.frame(
          Iteration = "MLE",
          matrix(ncol = num_beta + num_gamma, nrow = 1)
        )
        colnames(mle_row_df) <- c("Iteration", paste0("beta_", 1:num_beta), paste0("gamma_", 1:num_gamma))
        mle_row_df[1, ] <- c("MLE", mle_params_list_s$beta, mle_params_list_s$gamma)
      } else { print("MLE calculation for tables failed.") }
    }, error = function(e) { print(paste("Error in MLE calculation for tables:", e$message))})
    
    create_coeff_table <- function(model_final, model_name) {
      if (is.null(model_final)) return(NULL)
      beta_hist <- model_final$parameters$beta_hats
      gamma_hist <- model_final$parameters$gamma_hats
      num_beta <- ncol(beta_hist)
      num_gamma <- ncol(gamma_hist)
      
      actual_max_iter <- nrow(beta_hist)
      iters_show <- sort(unique(seq_len(actual_max_iter)))
      
      results_df <- data.frame(Iteration = character(), matrix(ncol = num_beta + num_gamma, nrow = 0), stringsAsFactors = FALSE)
      colnames(results_df) <- c("Iteration", paste0("beta_", 1:num_beta), paste0("gamma_", 1:num_gamma))
      
      for (iter_v in iters_show) {
        results_df[iter_v, ] <- c(as.character(iter_v), beta_hist[iter_v, ], gamma_hist[iter_v, ])
      }
      
      if (!is.null(mle_row_df)) results_df <- rbind(results_df, mle_row_df)
      
      print(paste("--- Coefficient Table", model_name, "-", scenario_name, "---"))
      if (requireNamespace("knitr", quietly = TRUE) && nrow(results_df) > 0) {
        print(knitr::kable(results_df, digits = 3, caption = paste("Coefficient Path -", model_name, "-", scenario_name)))
      } else {
        print(round(results_df, 3))
      }
      
      if (generate_detailed_csv && nrow(results_df) > 0) {
        write.csv(results_df, file = file.path(scenario_output_dir, paste0("table_coeffs_", tolower(model_name), "_", scenario_name, ".csv")), row.names = FALSE)
      }
    }
    
    
    create_coeff_table(model_cyclic_final, "Cyclic")
    create_coeff_table(model_combined_final, "Combined")
  }
  
  
  # --- 5. Evaluate and compare models on test data ---
  ibs_results_df <- data.frame(model=character(), ibs=numeric(), stringsAsFactors = FALSE)
  all_brier_curves_test <- list() # To store Brier curves
  
  # Cyclic Model
  if (!is.null(model_cyclic_final)) {
    beta_cyc <- model_cyclic_final$final_parameters$beta_hat_final
    gamma_cyc <- model_cyclic_final$final_parameters$gamma_hat_final
    pred_y0_test_cyc <- exp(X_matrix_test %*% beta_cyc)
    pred_mu_test_cyc <- Z_matrix_test %*% gamma_cyc
    brier_cyc_list <- brier_score_on_censored_data(times_test, delta_test, pred_y0_test_cyc, pred_mu_test_cyc, number_of_time_points = num_brier_time_points)
    ibs_cyc <- mean(brier_cyc_list$brier_scores, na.rm = TRUE)
    ibs_results_df <- rbind(ibs_results_df, data.frame(model="Cyclic", ibs=ibs_cyc))
    all_brier_curves_test[["Cyclic"]] <- brier_cyc_list
    print(paste("Test IBS - Cyclic (", scenario_name, "): ", round(ibs_cyc, 4)))
    if (generate_detailed_csv) write.csv(brier_cyc_list, file=file.path(scenario_output_dir, paste0("brier_curve_cyclic_",scenario_name,".csv")), row.names=FALSE)
  } else { print("Cyclic model is not available for evaluation.") }
  
  # Combined Model
  if (!is.null(model_combined_final)) {
    beta_comb <- model_combined_final$final_parameters$beta_hat_final
    gamma_comb <- model_combined_final$final_parameters$gamma_hat_final
    pred_y0_test_comb <- exp(X_matrix_test %*% beta_comb)
    pred_mu_test_comb <- Z_matrix_test %*% gamma_comb
    brier_comb_list <- brier_score_on_censored_data(times_test, delta_test, pred_y0_test_comb, pred_mu_test_comb, number_of_time_points = num_brier_time_points)
    ibs_comb <- mean(brier_comb_list$brier_scores, na.rm = TRUE)
    ibs_results_df <- rbind(ibs_results_df, data.frame(model="Combined", ibs=ibs_comb))
    all_brier_curves_test[["Combined"]] <- brier_comb_list
    print(paste("Test IBS - Combined (", scenario_name, "): ", round(ibs_comb, 4)))
    if (generate_detailed_csv) write.csv(brier_comb_list, file=file.path(scenario_output_dir, paste0("brier_curve_combined_",scenario_name,".csv")), row.names=FALSE)
  } else { print("Combined model is not available for evaluation.") }
  
  # Calculate and save final parameters and predictions
  if (generate_detailed_csv) {
    if(!is.null(model_cyclic_final)){
      write.csv(model_cyclic_final$final_parameters$beta_hat_final, file.path(scenario_output_dir, paste0("final_beta_cyclic_",scenario_name,".csv")))
      write.csv(model_cyclic_final$final_parameters$gamma_hat_final, file.path(scenario_output_dir, paste0("final_gamma_cyclic_",scenario_name,".csv")))
      if(exists("pred_y0_test_cyc")) write.csv(pred_y0_test_cyc, file.path(scenario_output_dir, paste0("pred_y0_test_cyclic_",scenario_name,".csv")))
      if(exists("pred_mu_test_cyc")) write.csv(pred_mu_test_cyc, file.path(scenario_output_dir, paste0("pred_mu_test_cyclic_",scenario_name,".csv")))
    }
    if(!is.null(model_combined_final)){
      write.csv(model_combined_final$final_parameters$beta_hat_final, file.path(scenario_output_dir, paste0("final_beta_combined_",scenario_name,".csv")))
      write.csv(model_combined_final$final_parameters$gamma_hat_final, file.path(scenario_output_dir, paste0("final_gamma_combined_",scenario_name,".csv")))
      if(exists("pred_y0_test_comb")) write.csv(pred_y0_test_comb, file.path(scenario_output_dir, paste0("pred_y0_test_combined_",scenario_name,".csv")))
      if(exists("pred_mu_test_comb")) write.csv(pred_mu_test_comb, file.path(scenario_output_dir, paste0("pred_mu_test_combined_",scenario_name,".csv")))
    }
  }
  
  # Brier Score comparison plot
  if (generate_plots && length(all_brier_curves_test) > 0) { 
    tryCatch({
      plot_df <- data.frame(time = numeric(), model_type = character(), brier_score = numeric())
      if(!is.null(all_brier_curves_test$Cyclic)) {
        plot_df <- rbind(plot_df, data.frame(time = all_brier_curves_test$Cyclic$brier_times, model_type = "Cyclic", brier_score = all_brier_curves_test$Cyclic$brier_scores))
      }
      if(!is.null(all_brier_curves_test$Combined)) {
        plot_df <- rbind(plot_df, data.frame(time = all_brier_curves_test$Combined$brier_times, model_type = "Combined", brier_score = all_brier_curves_test$Combined$brier_scores))
      }
      
      ibs_cyc_label <- ifelse(exists("ibs_cyc"), round(ibs_cyc,4), "N/A")
      ibs_comb_label <- ifelse(exists("ibs_comb"), round(ibs_comb,4), "N/A")
      
      plot_df$model_type <- factor(plot_df$model_type,
                                   levels = c("Cyclic", "Combined"),
                                   labels = c(paste("Cyclic (IBS:", ibs_cyc_label, ")"),
                                              paste("Combined (IBS:", ibs_comb_label, ")")))
      
      p_brier_comp <- ggplot(plot_df, aes(x=time, y=brier_score, color=model_type, linetype=model_type)) +
        geom_line(linewidth=1.1) +
        labs(x="Time", y="Brier Score") +
        theme_bw(base_size = 14) + 
        scale_color_manual(values=c(paste("Cyclic (IBS:", ibs_cyc_label, ")")="firebrick", paste("Combined (IBS:", ibs_comb_label, ")")="dodgerblue")) +
        scale_linetype_manual(values=c(paste("Cyclic (IBS:", ibs_cyc_label, ")")="solid", paste("Combined (IBS:", ibs_comb_label, ")")="dashed")) +
        theme(legend.title=element_blank(), legend.position="top", plot.title = element_text(hjust = 0.5))
      ggsave(filename=file.path(scenario_output_dir, paste0("plot_brier_comparison_", scenario_name, ".png")), plot=p_brier_comp, width=10, height=7)
      print(p_brier_comp)
    }, error = function(e) { print(paste("Error plotting Brier comparison:", e$message))})
  }
  
  if(generate_plots && !is.null(true_beta_train) && !is.null(true_gamma_train)){
    tryCatch({
      param_data_list <- list()
      if(!is.null(model_cyclic_final)) {
        param_data_list$Cyclic_Beta <- data.frame(param = names(true_beta_train), true = true_beta_train,
                                                  estimated = model_cyclic_final$final_parameters$beta_hat_final,
                                                  model="Cyclic", type="Beta")
        param_data_list$Cyclic_Gamma <- data.frame(param = names(true_gamma_train), true = true_gamma_train,
                                                   estimated = model_cyclic_final$final_parameters$gamma_hat_final,
                                                   model="Cyclic", type="Gamma")
      }
      if(!is.null(model_combined_final)) {
        param_data_list$Combined_Beta <- data.frame(param = names(true_beta_train), true = true_beta_train,
                                                    estimated = model_combined_final$final_parameters$beta_hat_final,
                                                    model="Combined", type="Beta")
        param_data_list$Combined_Gamma <- data.frame(param = names(true_gamma_train), true = true_gamma_train,
                                                     estimated = model_combined_final$final_parameters$gamma_hat_final,
                                                     model="Combined", type="Gamma")
      }
      
      if(length(param_data_list) > 0) {
        all_param_df <- do.call(rbind, param_data_list)
        
        
        df_true <- all_param_df[, c("param", "true", "type")]
        df_true$source <- "True Value"
        colnames(df_true)[2] <- "value"
        
        df_est <- all_param_df[, c("param", "estimated", "type", "model")]
        colnames(df_est)[2] <- "value"
        df_est$source <- df_est$model
        
        final_df <- rbind(df_true[, c("param", "value", "type", "source")],
                          df_est[, c("param", "value", "type", "source")])
        
        final_df$source <- factor(final_df$source,
                                  levels = c("True Value", "Cyclic", "Combined"))
        
        p_param_comp <- ggplot(final_df, aes(x=param, y=value, color=source, shape=source)) +
          geom_point(size=3, position=position_dodge(width=0.2)) +
          facet_wrap(~type, scales="free_x") +
          labs(x="Parameter", y="Value") +
          theme_light(base_size=12) +
          scale_color_manual(values=c("True Value"="black", "Cyclic"="firebrick", "Combined"="dodgerblue")) +
          scale_shape_manual(values=c("True Value"=17, "Cyclic"=16, "Combined"=16)) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1),
                legend.title = element_blank(),
                legend.position = "top",
                plot.title = element_text(hjust = 0.5))
        
        ggsave(filename=file.path(scenario_output_dir, paste0("plot_param_comparison_", scenario_name, ".png")),
               plot=p_param_comp, width=10, height=6)
        print(p_param_comp)
      }
    }, error = function(e) {print(paste("Error plotting parameter comparison:", e$message))})
  }
  
  
  # Predicted survival curve vs. Kaplan-Meier plot
  if(generate_plots && !is.null(sim_data_test) && ( !is.null(model_cyclic_final) || !is.null(model_combined_final) ) ){
    tryCatch({
      km_test_fit <- survfit(Surv(times_test, delta_test) ~ 1) 
      
      plot_data_surv <- data.frame(time = km_test_fit$time, km_survival = km_test_fit$surv)
      
      if(!is.null(model_cyclic_final)){
        beta_c <- model_cyclic_final$final_parameters$beta_hat_final
        gamma_c <- model_cyclic_final$final_parameters$gamma_hat_final
        X_avg_test_c <- matrix(c(1, rep(0, ncol(X_matrix_test)-1)), nrow=1)
        Z_avg_test_c <- matrix(c(1, rep(0, ncol(Z_matrix_test)-1)), nrow=1)
        y0_avg_pred_c <- exp(X_avg_test_c %*% beta_c)
        mu_avg_pred_c <- Z_avg_test_c %*% gamma_c
        plot_data_surv$cyclic_pred <- FHT_parametric_survival(plot_data_surv$time, mu_avg_pred_c[1,1], y0_avg_pred_c[1,1])
      }
      if(!is.null(model_combined_final)){
        beta_cb <- model_combined_final$final_parameters$beta_hat_final
        gamma_cb <- model_combined_final$final_parameters$gamma_hat_final
        X_avg_test_cb <- matrix(c(1, rep(0, ncol(X_matrix_test)-1)), nrow=1)
        Z_avg_test_cb <- matrix(c(1, rep(0, ncol(Z_matrix_test)-1)), nrow=1)
        y0_avg_pred_cb <- exp(X_avg_test_cb %*% beta_cb)
        mu_avg_pred_cb <- Z_avg_test_cb %*% gamma_cb
        plot_data_surv$combined_pred <- FHT_parametric_survival(plot_data_surv$time, mu_avg_pred_cb[1,1], y0_avg_pred_cb[1,1])
      }
      
      plot_df_surv_long <- pivot_longer(plot_data_surv, 
                                        cols = starts_with(c("km_", "cyclic_", "combined_")), 
                                        names_to = "method", values_to = "survival_probability")
      plot_df_surv_long$method <- factor(plot_df_surv_long$method,
                                         levels = c("km_survival", "cyclic_pred", "combined_pred"),
                                         labels = c("Kaplan-Meier (Test)", "Cyclic Predicted", "Combined Predicted"))
      
      p_surv_comp <- ggplot(plot_df_surv_long, aes(x=time, y=survival_probability , color=method, linetype=method)) +
        geom_step(linewidth=1) + 
        labs(x="Time", y="Survival Probability") +
        theme_classic(base_size = 14) +
        scale_color_manual(values=c("Kaplan-Meier (Test)"="black", 
                                    "Cyclic Predicted"="firebrick", 
                                    "Combined Predicted"="dodgerblue")) +
        scale_linetype_manual(values=c("Kaplan-Meier (Test)"="solid", 
                                       "Cyclic Predicted"="solid", 
                                       "Combined Predicted"="dashed"))+
        theme(legend.title=element_blank(), legend.position="top", plot.title = element_text(hjust = 0.5)) +
        ylim(0,1)
      ggsave(filename=file.path(scenario_output_dir, paste0("plot_survival_comparison_", scenario_name, ".png")), plot=p_surv_comp, width=10, height=7)
      print(p_surv_comp)
      
    }, error = function(e){print(paste("Error plotting survival comparison:", e$message))})
  }
  
  
  # Save IBS results
  if (generate_detailed_csv && nrow(ibs_results_df)>0) write.csv(ibs_results_df, file=file.path(scenario_output_dir, paste0("results_ibs_", scenario_name, ".csv")), row.names=FALSE)
  
  print(paste("<<<<< Finished processing scenario:", scenario_name, ">>>>>"))
  return(list(scenario_name=scenario_name, 
              ibs_results=ibs_results_df, 
              optimal_m_cyclic=list(y0=optimal_m_y0_cyclic, mu=optimal_m_mu_cyclic, cv_metric_value=cv_metric_value_cyclic, metric_type=metric_cyclic),
              optimal_m_combined=list(m=optimal_m_stop_combined, cv_metric_value=cv_metric_value_combined, metric_type=metric_combined),
              censoring_train_ratio = censoring_ratio_train,
              censoring_test_ratio = censoring_ratio_test
  ))
}

# ==============================================================================
# Call the comprehensive function for different scenarios (Example)
# ==============================================================================
# Ensure necessary packages for plotting are loaded
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")
if (!requireNamespace("knitr", quietly = TRUE)) install.packages("knitr")
if (!requireNamespace("survival", quietly = TRUE)) install.packages("survival") # for survfit
library(ggplot2)
library(tidyr)
library(knitr)
library(survival)


all_scenario_final_results_list <- list()

# Example: Run the first scenario with checkpointing capability
all_scenario_final_results_list[["SmallDense_N500"]] <- tryCatch({
  run_full_scenario_analysis(
    scenario_name = "SmallDense_N500",
    N_train = 500, N_test = 200, setup_type = 'small_dense',
    seed_train = 126, seed_test = 103, add_noise_train = FALSE,
    M_y0_cv_cyclic = 50, M_mu_cv_cyclic = 50, K_folds_cv_cyclic = 10, metric_cyclic = "loglik",
    M_cv_combined = 50, K_folds_cv_combined = 10, K_reps_cv_cyclic = 10, K_reps_cv_combined = 10, metric_combined = "loglik",
    output_directory = "Comprehensive_Results_Final", 
    generate_detailed_csv = FALSE, generate_plots = TRUE,
    force_rerun_cv_cyclic = FALSE, force_rerun_cv_combined = FALSE 
  )
}, error = function(e) { print(paste("Error running scenario SmallDense_N500:", e$message)); return(NULL) })


# Scenario 2: Medium dimensions (small_sparse), N_train=1000
all_scenario_final_results_list[["SmallSparse_N1000"]] <- tryCatch({
  run_full_scenario_analysis(
    scenario_name = "SmallSparse_N1000",
    N_train = 1000, N_test = 300, setup_type = 'small_sparse',
    seed_train = 123, seed_test = 13, add_noise_train = FALSE,
    M_y0_cv_cyclic = 150, M_mu_cv_cyclic = 150, K_folds_cv_cyclic = 10, metric_cyclic = "loglik",
    M_cv_combined = 200, K_folds_cv_combined = 10, K_reps_cv_cyclic = 10, K_reps_cv_combined = 10, metric_combined = "loglik",
    output_directory = "Comprehensive_Results_Final", 
    generate_detailed_csv = FALSE, generate_plots = TRUE,
    force_rerun_cv_cyclic = FALSE, force_rerun_cv_combined = FALSE 
  )
}, error = function(e) { print(paste("Error running scenario SmallSparse_N1000:", e$message)); return(NULL) })

# Scenario 3: High dimensions (huge), N_train=1000 
all_scenario_final_results_list[["Huge_N1000"]] <- tryCatch({
  run_full_scenario_analysis(
    scenario_name = "Huge_N1000",
    N_train = 1000, N_test = 300, setup_type = 'huge',
    seed_train = 203, seed_test = 103, add_noise_train = FALSE,
    M_y0_cv_cyclic = 60, M_mu_cv_cyclic = 60, K_folds_cv_cyclic = 5, metric_cyclic = "loglik",
    M_cv_combined = 60, K_folds_cv_combined = 5, metric_combined = "loglik",
    output_directory = "Comprehensive_Results_Final", 
    generate_detailed_csv = FALSE, generate_plots = TRUE,
    force_rerun_cv_cyclic = FALSE, force_rerun_cv_combined = FALSE 
  )
}, error = function(e) { print(paste("Error running scenario Huge_N1000:", e$message)); return(NULL) })

# --- Aggregate and summarize final results ---

print("--- Summary of IBS and optimal parameters for all executed scenarios ---")
summary_final_df <- data.frame(
  scenario = character(), model = character(), ibs = numeric(),
  m_stop_y0 = numeric(), m_stop_mu = numeric(), m_stop_combined = numeric(),
  cv_metric_value = numeric(), censoring_train = numeric(), censoring_test = numeric(),
  stringsAsFactors = FALSE
)

for (scen_name in names(all_scenario_final_results_list)) {
  res <- all_scenario_final_results_list[[scen_name]]
  if (!is.null(res) && !is.null(res$ibs_results) && nrow(res$ibs_results) > 0) {
    for(i in 1:nrow(res$ibs_results)){
      model_type <- res$ibs_results$model[i]
      ibs_val <- res$ibs_results$ibs[i]
      m_y0 <- ifelse(model_type=="Cyclic", res$optimal_m_cyclic$y0, NA)
      m_mu <- ifelse(model_type=="Cyclic", res$optimal_m_cyclic$mu, NA)
      m_comb <- ifelse(model_type=="Combined", res$optimal_m_combined$m, NA)
      cv_val <- ifelse(model_type=="Cyclic", res$optimal_m_cyclic$cv_metric, res$optimal_m_combined$cv_metric)
      
      summary_final_df <- rbind(summary_final_df, data.frame(
        scenario = scen_name, model = model_type, ibs = ibs_val,
        m_stop_y0 = m_y0, m_stop_mu = m_mu, m_stop_combined = m_comb,
        cv_metric_value = cv_val,
        censoring_train = res$censoring_train_ratio,
        censoring_test = res$censoring_test_ratio
      ))
    }
  }
}

if(nrow(summary_final_df) > 0){
  print(knitr::kable(summary_final_df, digits=4, caption="Final Summary of IBS and Optimal m_stops"))
  write.csv(summary_final_df, file=file.path("Comprehensive_Results_Final", "summary_all_scenarios_ibs.csv"), row.names=FALSE)
} else {
  print("No final IBS results available to display or save.")
}