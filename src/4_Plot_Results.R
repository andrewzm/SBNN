library(MASS)
library(dgumbel)
library(ggplot2)
library(fields)
library(rjson)
library(tidyr)
library(stringr)
library(latex2exp)
library(reticulate)
library(dplyr, warn.conflicts = FALSE)
library(Matrix, warn.conflicts = FALSE)
library(purrr)
library(yaml)
library(xtable)

set.seed(1)
np <- import('numpy') # Use reticulate to import numpy
source("src/utils.R")
FIG_DIR <- "src/figures/"
dir.create(FIG_DIR, showWarnings = FALSE)

###########
## FIGURE 1
###########

cat("Doing Figure 1\n")

s <- seq(-4, 4, length.out = 256)

Samples_all <- Mean_all <- NULL
for(layers in c(1,2,4,8)) {

    sel_samples <- sample(1:4000, 10)

    fname <- paste0('src/runs/Section2_BNN_degeneracy/output/std_prior_preds_', layers, 
                     '_layers.json')
    Samples <- rjson::fromJSON(file = fname)
    X_mean <- rowMeans(data.frame(Samples))
    X_mean <- data.frame(mean = X_mean,
                         s = s)
    X_mean$layers <- paste0("L = ", layers)
    Mean_all <- rbind(Mean_all, X_mean)

    X <- data.frame(Samples[sel_samples]) 
    names(X) <- paste0("Sim", 1:10)
    X$s <- s
    X$layers <- layers
    X$layers <- paste0("L = ", layers)
    Samples_all <- rbind(Samples_all, 
                         gather(X, key = "Sim", value = "y", 
                         -s, - layers))
}

p <- ggplot(Samples_all) + 
        geom_line(aes(s, y, group = Sim), linewidth = 0.2) + 
        geom_line(data = Mean_all, aes(s, mean, group = layers), 
                  color = "red") +
        facet_wrap(~layers) + 
        theme_bw() + 
        theme(text = element_text(size = 7),
              strip.background = element_rect(fill="white")) + 
        ylim(-3, 3) + 
        ylab("Y(s)") + xlab("s")

save_image(path = paste0(FIG_DIR, "Section2_BNN_degeneracy.png"),
               p, width_in = 6, height_in = 3, res = 300,
               device = "ggsave_png")

#######################
## Basis function plots
#######################

cat("Doing Basis Function Plot\n")

rbf <- function(d, tau = 1) {
  exp(-(d/tau)^2)
}
sgrid <- seq(-4, 4, by = 0.01)
centroids <- seq(-4, 4, length.out = 15)
rho <- fields::rdist(sgrid, centroids) %>% rbf()
rbf_df <- data.frame(cbind(sgrid, rho)) %>%
          gather(rbf_id, rho, -sgrid)
p <- ggplot(rbf_df) + 
     geom_line(aes(sgrid, rho, colour = rbf_id), linewidth = 0.3) + 
       theme_bw() + 
       theme(text = element_text(size = 7),
             legend.position = "none") +
       ylab(TeX(r"( ${\rho}((0, s_2)'; \tau = 1)$)")) +
       xlab(TeX(r"($s_2$)"))

save_image(path = paste0(FIG_DIR, "Section4_Basis_Functions.png"),
               p, width_in = 5, height_in = 1.5, res = 300, 
               device = "ggsave_png")

####################
## COVARIOGRAM PLOTS
####################

cat("Plotting covariograms\n")

process_names <- c("BNN-IP", "SBNN-IL", "SBNN-IP", 
                                            "SBNN-VL", "SBNN-VP")


for(process in process_names) {

  fnames <- c(
    dir('src/runs/Section4_1_BNN-IL/output', 
                pattern = 'covariogram', 
                full.names = TRUE),
    dir(paste0('src/runs/Section4_1_', process, '/output'), 
                pattern = 'covariogram', 
                full.names = TRUE)
  )

  Cvgrams_all <- NULL
  for(fname in fnames) {

      number <- str_extract_all(basename(fname), "\\d+")[[1]]
      if(length(number) > 0) {
        label = paste0(number, " steps")
      } else {
        label <- "GP"
        number <- NA
      }

      if(grepl("_BNN-IL/", fname)) {
        model <- "BNN-IL"
      } else {
        model <- process
      } 

      Cvgram <- t(data.frame(rjson::fromJSON(file = fname)))
      row.names(Cvgram) <- NULL
      Cvgram <- data.frame(h = Cvgram[, 1], 
                          Co = Cvgram[, 2],
                          number = number,
                          label = label,
                          model = model)
      Cvgrams_all <- rbind(Cvgrams_all, Cvgram)
  }

  Cvgrams_all$label <- as.factor(Cvgrams_all$label)
  nums <- as.numeric(setdiff(gsub(" steps", "", levels(Cvgrams_all$label)), "GP"))

  # Order by numbers
  ordered_levels <- c(levels(Cvgrams_all$label)[order(nums, na.last = TRUE)], "GP")

  Cvgrams_all$label <- factor(Cvgrams_all$label, 
                              levels = ordered_levels)

  linetypes <- c("dashed","dotted","dotdash",
                 "longdash","solid","solid")

  linecolours <- c("black","black", "black", 
                   "black","black", "red")

  p <- ggplot(Cvgrams_all) +
      geom_line(aes(h, Co, linetype = label,
                    colour = label), 
                linewidth = 0.2)  + 
      scale_linetype_manual(values = linetypes) +
      scale_colour_manual(values = linecolours) +   
      ylab(TeX(r"( $C^o(\| \textbf{s} - \textbf{r} \|)$)")) +
      xlab(TeX(r"( $\| \textbf{s} - \textbf{r} \|$)")) +
      facet_grid(~model) +
      theme_bw() + 
      theme(text = element_text(size = 7),
            legend.title = element_blank(),
            strip.background = element_rect(fill="white"))

  save_image(path = paste0(FIG_DIR, "Section4_1_", process, "_covariogram.png"),
               p, width_in = 6, height_in = 2.5, res = 300,device = "ggsave_png")

}

############################
## ALL OTHER PLOTS
############################

grid <- rjson::fromJSON(file = 'src/runs/Section4_1_SBNN-IL/output/cov_matrix_grid.json') %>%
        Reduce(rbind, .) %>%
        as.data.frame()
rownames(grid) <- NULL
names(grid) <- c("s1", "s2")

process_names <- c("GP", "BNN-IL", "BNN-IP", 
                          "SBNN-IL", "SBNN-IP", 
                          "SBNN-VL", "SBNN-VP")
section_names <- c("Section4_1_", "Section4_2_", "Section4_3_")

wass_losses <- NULL

for(section in section_names) {

cat(paste0("Doing plots for ", section, "\n"))
    
############################
## COVARIANCE HEAT MAPS
############################

  cat(paste0("...Covariance heat maps\n"))
    
  allcovs <- lapply(process_names,
                    function(x) { 
                      if(x == "GP") {
                        pth <- paste0("src/runs/", section, "SBNN-IL/output/Target_cov_matrix.npy")
                      } else {
                        pth <- paste0("src/runs/", section, x, 
                                    "/output/BNN_cov_matrix.npy")
                      }
                      np$load(pth)%>% as.matrix()
                    } ) %>% setNames(process_names)

  s1 <- unique(grid$s1)
  s2 <- unique(grid$s2)

  n_s1 <- length(s1)
  n_s2 <- length(s2)

  s1idx <- round(seq(1, n_s1, length.out = 6)[c(-1, -6)])
  s2idx <- round(seq(1, n_s2, length.out = 6)[c(-1, -6)])

  focal_axis <- data.frame(s1 = s1[s1idx],
                          s2 = s2[s2idx])

  focal_grid <- expand.grid(s1 = focal_axis$s1, s2 = focal_axis$s2)


  for(process in process_names) {

    all_cov <- list()
    
    for(i in 1:nrow(focal_grid)) {
      idx <- which.min((grid$s1 - focal_grid$s1[i])^2 +
                        (grid$s2 - focal_grid$s2[i])^2) 

      values <- allcovs[[process]][idx,]
      
      this_cov <- data.frame(s1o = grid$s1[idx] %>% round(2),
                              s2o = grid$s2[idx] %>% round(2),
                              s1 = grid$s1,
                              s2 = grid$s2,
                              cov = values)
      all_cov <- rbind(all_cov, this_cov)  

    }

    all_cov$s1o_factor <- factor(all_cov$s1o)
    all_cov$s2o_factor <- factor(all_cov$s2o, 
                                levels = rev(unique(all_cov$s2o)))

    {p <- ggplot(all_cov) + geom_tile(aes(s1, s2, fill = pmin(pmax(cov, 0), 1))) +
        geom_point(aes(s1o, s2o), pch = 4, colour = 'black',
                    size = 0.5) +
        facet_grid(s2o_factor ~ s1o_factor,
                  labeller = custom_labeller()) +
        ylab(TeX(r"($s_2$)")) +
        xlab(TeX(r"($s_1$)")) +
        theme_bw() + 
        scale_fill_distiller(palette = "PuBuGn", limits = c(0,1)) +
        theme(text = element_text(size = 7),
              legend.title = element_blank(),
              strip.background = element_rect(fill="white"),
              legend.key.width = unit(0.3, "cm"),
              legend.margin = margin(t=0,r=0,b=0,l=-0.2, unit="cm"),
              strip.text = element_text(
                margin = margin(t = 1, b = 1, l = 1, r = 1, unit = "pt"))) +
              coord_fixed()} %>% suppressWarnings()

    save_image(path = paste0(FIG_DIR, section, process, "_cov.png"),
               p, width_in = 3.1, height_in = 2.9, res = 300)
  }

  ############################
  ## PLOTS OF SAMPLES
  ############################

  cat(paste0("...Plots of samples\n"))
  
  allsamples <- lapply(process_names,
                    function(x) { 
                      if(x == "GP") {
                        pth <- paste0("src/runs/", section, "SBNN-IL/output/Target_prior_samples.npy")
                      } else {
                        output_fnames <- dir(paste0("src/runs/", section, x, "/output/"))
                        req_fname <- output_fnames[output_fnames %>% grepl("BNN_prior_samples_step",.) %>% which()]
                        pth <- paste0("src/runs/", section, x, "/output/", req_fname)
                      }
                      np$load(pth)%>% as.matrix()
                    } ) %>% setNames(process_names)

  nsim = ncol(allsamples[["GP"]])
  
  s1 <- s2 <- seq(-4, 4, length.out = 64L)
  sgrid <- expand.grid(s1 = s1, s2 = s2)

  ## Comparisong of sample paths
  for(process in process_names) {

    
    all_sims <- list()
    idx <- sample(1:nsim, 8)
    samples_df <-  as.data.frame(cbind(sgrid, allsamples[[process]][, idx]))

    samples_df <- gather(samples_df, sim, value, -s1, -s2)

    if (section == "Section4_3_") {
      minval <- 0.3
      maxval <- 3.5
    } else {
      minval <- -2.7
      maxval <- 2.7
    }

    p <- ggplot(samples_df) + geom_tile(aes(s1, s2, fill = pmin(pmax(value, minval), maxval))) +
        facet_wrap(~sim, ncol = 4) +
        ylab(TeX(r"($s_2$)")) +
        xlab(TeX(r"($s_1$)")) +
        theme_bw() + 
        scale_fill_distiller(palette = "Spectral", limits = c(minval, maxval)) +
        theme(text = element_text(size = 7),
              legend.title = element_blank(),
              legend.key.width = unit(0.3, "cm"),
              legend.margin = margin(t=0,r=0,b=0,l=-0.2, unit="cm"),
              strip.text = element_blank()) +
              coord_fixed()

    save_image(path = paste0(FIG_DIR, section, process, "_prior_samples.png"),
               p, width_in = 3.1, height_in = 1.5, res = 300)

  }

  ##########################
  # Density plots
  ##########################

  cat(paste0("...Density plots\n"))
  
  ## Comparison of marginal distributions
  for (process in setdiff(process_names, "GP")) {

    idx <- find_subgrid_indices(s1, s2, 4, 2)
    bnn_samples <- allsamples[[process]]
    gp_samples <- allsamples[["GP"]]
    
    marginal_samples_df <- 
    lapply(idx,
        function(x) {
          
          data.frame(s1o = round(sgrid[x, 1], 2), 
                    s2o = round(sgrid[x, 2], 2), 
                    gp = gp_samples[x, 1:1000], 
                    bnn = bnn_samples[x, ],
                    idx = x)
        }
    ) %>% Reduce(rbind, .)

    marginal_samples_df$s1o_factor <- factor(marginal_samples_df$s1o)
    marginal_samples_df$s2o_factor <- factor(marginal_samples_df$s2o, 
                                  levels = rev(unique(marginal_samples_df$s2o)))

    {p <- ggplot(marginal_samples_df) + 
      geom_density(aes(gp, colour = "1"), linewidth = 0.3) +
      geom_density(aes(bnn, colour = "2"), linewidth = 0.3) +
      scale_color_manual(labels = c("1" = "Target", "2" = process),
                         values = c("1" = "#00BFC4", "2" = "#F8766D")) +
      facet_grid(s2o_factor ~ s1o_factor,
                    labeller = custom_labeller()) +
      xlab("process value") +
      theme_bw() +
      theme(text = element_text(size = 7),
            legend.title = element_blank(),
            strip.background = element_rect(fill="white"),
            legend.key.width = unit(0.3, "cm"),
            legend.margin = margin(t=0,r=0,b=0,l=-0.2, unit="cm"),
            strip.text = element_text(
                  margin = margin(t = 1, b = 1, l = 1, r = 1, unit = "pt")))
    } %>% suppressWarnings() 


   save_image(path = paste0(FIG_DIR, section, process, "_marginal_histograms.png"),
               p, width_in = 6.2, height_in = 2.5, res = 300, device = "ggsave_png")

    ## Comparison of joint distributions
    idx <- find_subgrid_indices(s1, s2, 2, 1)
    idx <- c(idx, idx[1] + 1, idx[1] + 6)
    cov_combinations <- list(c(idx[1], idx[2]),
                              c(idx[1], idx[3]),
                              c(idx[1], idx[4]))

    joint_samples_df <- 
    lapply(cov_combinations,
        function(x) {
          data.frame(s11o = round(sgrid[x[1], 1], 2), 
                    s12o = round(sgrid[x[1], 2], 2), 
                    s21o = round(sgrid[x[2], 1], 2), 
                    s22o = round(sgrid[x[2], 2], 2), 
                    gp1 = gp_samples[x[1], 1:1000], 
                    gp2 = gp_samples[x[2], 1:1000], 
                    bnn1 = bnn_samples[x[1], ],
                    bnn2 = bnn_samples[x[2], ])
        }
    ) %>% Reduce(rbind, .)

    joint_samples_df$s1o_factor <- as.factor(joint_samples_df$s22o)
    joint_samples_df$s2o_factor <- as.factor(joint_samples_df$s21o)

    # Ensure contour plots have the same "levels"
    maxlevel <- joint_samples_df %>%
    summarise(.by = c(s1o_factor, s2o_factor),
              max_level = find_max_level(gp1, gp2, bnn1, bnn2)) %>%
    dplyr::select(max_level) %>%  max()
    levels <- pretty(range(0, maxlevel), 10)

    {p <- ggplot(joint_samples_df) + 
      geom_density_2d(aes(gp1, gp2, colour = "1"), linewidth = 0.3, breaks = levels) +
      geom_density_2d(aes(bnn1, bnn2, colour = "2"), linewidth = 0.3, breaks = levels) +
      scale_color_manual(labels = c("1" = "Target", "2" = process),
                         values = c("1" = "#00BFC4", "2" = "#F8766D")) +
      facet_grid(s1o_factor ~ s2o_factor, labeller = custom_labeller()) +
      xlab(paste0("process value at (", sgrid[idx[1],1] %>% round(2), 
                  ",", round(sgrid[idx[1],2], 2), ")")) +
      ylab("process value at second spatial point") +
      theme_bw() +
      theme(text = element_text(size = 7),
            legend.title = element_blank(),
            strip.background = element_rect(fill="white"),
            legend.key.width = unit(0.3, "cm"),
            legend.margin = margin(t=0,r=0,b=0,l=-0.2, unit="cm"),
            strip.text = element_text(
                  margin = margin(t = 1, b = 1, l = 1, r = 1, unit = "pt")))  +
      coord_fixed()} %>% suppressWarnings()

    save_image(path = paste0(FIG_DIR, section, process, "_joint_histograms.png"),
               p, width_in = 6.2, height_in = 2.3, res = 300, device = "ggsave_png")

    # Get contour levels
    levels <- get_contour_levels(p)
    nlevels <- length(levels)

    caption <- paste0("Contour levels (from the outer to the inner contours) correspond to ", 
        paste(print_2dec(levels[-nlevels]), collapse = ", "), 
        " and ", print_2dec(levels[nlevels]), ".")
    #cat(paste0("Figure 5 (", process, "): ", caption, "\n"))    
    writeLines(caption, paste0(FIG_DIR, section, process, "_contour_levels.tex"))
  }

  #######################
  ## WASSERSTEIN LOSSES
  ######################
  for(process in setdiff(process_names, "GP")) {
     wss_path <- paste0("src/runs/", section, process, 
                                    "/output/wsr_values.log")

     these_losses <- readLines(wss_path)   %>% as.numeric()

     wass_losses <- rbind(wass_losses,
                         data.frame(t = 1:length(these_losses),
                                    wss = these_losses,
                                    process = process,
                                    model = case_when(
            str_detect(wss_path, "Section4_1") ~ "Stationary GP",
            str_detect(wss_path, "Section4_2") ~ "Nonstationary GP",
            str_detect(wss_path, "Section4_3") ~ "Log-GP")))
  
  }

}

##########################
## PLOT WASSERSTEIN LOSSES
##########################
wass_losses$model <- factor(wass_losses$model, 
                            levels = c("Stationary GP",
                                      "Nonstationary GP",
                                      "Log-GP"))
p <- ggplot(wass_losses) + geom_line(aes(x = t, y = wss)) +
            facet_grid(process ~ model, scales = "free_x")  +
            ylab(TeX(r"(W_1(\cdot))")) +
            xlab("Outer loop iteration") +
            theme_bw() + 
            theme(text = element_text(size = 7),
                  strip.background = element_rect(fill="white"),
                  strip.text = element_text(
                          margin = margin(t = 1, b = 1, l = 1, r = 1, unit = "pt")))

      save_image(path = paste0("./src/figures/Section4_Wasserstein_distances.png"),
               p, width_in = 6, height_in = 6, res = 300, device = "ggsave_png")


###############################
## Posterior samples plot
###############################

cat(paste0("Figures of posterior distribution\n"))

s1 <- s2 <- seq(-4, 4, length.out = 64L)
sgrid <- expand.grid(s1 = s1, s2 = s2)
fnames <- c("Section4_1_SBNN-IL", "Section5_2_SBNN-IP")
models <- c("GP", "Max-Stable")

for(i in seq_along(fnames)) {

  sgrid$y_true <- np$load(paste0("src/runs/", fnames[i], "/output/true_field.npy"))

  p <- ggplot(sgrid) + 
            geom_tile(aes(s1, s2, fill = y_true)) +
            scale_fill_distiller(palette = "Spectral") + 
            ylab(TeX(r"($s_2$)")) +
            xlab(TeX(r"($s_1$)")) +
            theme_bw() + 
            theme(text = element_text(size = 7),
                  legend.title = element_blank(),
                  legend.key.width = unit(0.3, "cm"),
                  legend.margin = margin(t=0,r=0,b=0,l=-0.2, unit="cm"),
                  strip.text = element_blank()) +
                  coord_fixed()

        save_image(path = paste0("./src/figures/Section4_4_", models[i], "_true_field.png"),
                  p, width_in = 3.1, height_in = 2.9, res = 300)

  for(N in c(100, 1000, 5000)) {

    obs <- np$load(paste0("src/runs/", fnames[i], "/output/obs_N", N, ".npy")) %>%
                      as.data.frame()
    names(obs) <- c("s1", "s2")
    obs$z <- np$load(paste0("src/runs/", fnames[i], "/output/obs_N100_vals.npy")) 
    
    if(models[i] == "GP") {
      process_list <- c("GP", "SBNN-IL")
    } else {
      process_list <- c("SBNN-IP")
    }

    for(process in process_list) {

      if(process %in% c("SBNN-IL", "SBNN-IP")) {

        posterior_mean <- np$load(paste0("/home/azm/bnn-spatial/src/runs/", fnames[i], "/output/BNN_pred_mean_N", N, ".npy")) %>%
                      as.matrix() %>% t()

        posterior_sd <- np$load(paste0("/home/azm/bnn-spatial/src/runs/", fnames[i], "/output/BNN_pred_sd_N", N, ".npy")) %>%
                      as.matrix() %>% t()
      } else {
        posterior_mean <- np$load(paste0("/home/azm/bnn-spatial/src/runs/", fnames[i], "/output/Target_pred_mean_N", N, ".npy")) %>%
                    as.matrix() %>% t()

        posterior_sd <- np$load(paste0("/home/azm/bnn-spatial/src/runs/", fnames[i], "/output/Target_pred_sd_N", N, ".npy")) %>%
                    as.matrix() %>% t()
      }


      for(plotvar in c("mean", "sd")) {
        
        if(plotvar == "mean") {
          sgrid$values <- pmin(pmax(c(posterior_mean), -2.5), 2.5)
          cbar <- scale_fill_distiller(palette = "Spectral", limits = c(-2.5, 2.5))
        } else {
          sgrid$values <- pmin(pmax(c(posterior_sd), 0), 0.6)
          cbar <- scale_fill_distiller(palette = "BrBG", limits = c(0, 0.6))
        }

        p <- ggplot(sgrid) + 
            geom_tile(aes(s1, s2, fill = values)) +
            geom_point(data = obs, aes(s1, s2), size = 0.3) + 
            cbar + 
            ylab(TeX(r"($s_2$)")) +
            xlab(TeX(r"($s_1$)")) +
            theme_bw() + 
            theme(text = element_text(size = 7),
                  legend.title = element_blank(),
                  legend.key.width = unit(0.3, "cm"),
                  legend.margin = margin(t=0,r=0,b=0,l=-0.2, unit="cm"),
                  strip.text = element_blank()) +
                  coord_fixed()

        save_image(path = paste0("./src/figures/Section4_4_", models[i], "_Posterior_", 
                             process, "_", plotvar, "_N", N, ".png"),
                  p, width_in = 3.1, height_in = 2.9, res = 300)

      }
    }

    ###########################
    
    bnn_vanilla_samples <- np$load(paste0("src/runs/", fnames[i], "/output/BNN_Vanilla_posterior_samples_N", N, ".npy")) %>%
                    as.array() %>% aperm(c(1,3,2)) %>% matrix(ncol = 64^2) %>% t()
    bnn_samples <- np$load(paste0("src/runs/", fnames[i], "/output/BNN_posterior_samples_N", N, ".npy")) %>%
                    as.array() %>% aperm(c(1,3,2)) %>% matrix(ncol = 64^2) %>% t()

    if(models[i] == "GP") {
      gp_samples <- np$load(paste0("src/runs/", fnames[i], "/output/Target_posterior_samples_N", N, ".npy")) %>%
                      as.array() %>% aperm(c(1,3,2)) %>% matrix(ncol = 64^2) %>% t()
    } else {
       gp_samples <- bnn_samples
       warning("Setting GP samples to BNN samples for now")
    }

    nsim = min(ncol(gp_samples), ncol(bnn_samples))
   
    ## Comparison of marginal distributions
    idx <- find_subgrid_indices(s1, s2, 4, 2)

    marginal_samples_df <- 
    lapply(idx,
        function(x) {
          data.frame(s1o = round(sgrid[x, 1], 2), 
                    s2o = round(sgrid[x, 2], 2), 
                    y_true = sgrid[x,3],
                    gp = gp_samples[x, 1:nsim],
                    bnn_vanilla =  bnn_vanilla_samples[x, 1:nsim],
                    bnn = bnn_samples[x, 1:nsim],
                    idx = x)
        }
    ) %>% Reduce(rbind, .)

    marginal_samples_df$s1o_factor <- factor(marginal_samples_df$s1o)
    marginal_samples_df$s2o_factor <- factor(marginal_samples_df$s2o, 
                                  levels = rev(unique(marginal_samples_df$s2o)))

    p <- ggplot(marginal_samples_df) + 
          geom_density(aes(bnn, colour = "2"), linewidth = 0.3) +
      facet_grid(s2o_factor ~ s1o_factor,
                    labeller = custom_labeller()) +
      geom_vline(aes(xintercept = y_true), linetype = "dashed") +
      xlab("process value") +
      theme_bw() +
      theme(text = element_text(size = 7),
            legend.title = element_blank(),
            strip.background = element_rect(fill="white"),
            legend.key.width = unit(0.3, "cm"),
            legend.margin = margin(t=0,r=0,b=0,l=-0.2, unit="cm"),
            strip.text = element_text(
                  margin = margin(t = 1, b = 1, l = 1, r = 1, unit = "pt"))) 

    if(models[i] == "GP") {
        p <- p + geom_density(aes(gp, colour = "1"), linewidth = 0.3) +
          scale_color_manual(labels = c("1" = "Target", "2" = "SBNN-IL", "3" = "BNN (untuned)"),
                            values = c("1" = "#00BFC4", "2" = "#F8766D", "3" = "#FFD700")) 
    } else {
        p <- p + scale_color_manual(labels = c("2" = "SBNN-IP", "3" = "BNN (untuned)"),
                            values = c("2" = "#F8766D", "3" = "#FFD700")) 
    }

    

    save_image(path = paste0("./src/figures/Section4_4_", 
                            models[i], "_marginal_histograms_N", N, ".png"),
                  p, width_in = 6.2, height_in = 2.5, res = 300, device = "ggsave_png")

    ## Comparison of joint distributions
    idx <- find_subgrid_indices(s1, s2, 2, 1)
    idx <- c(idx, idx[1] + 1, idx[1] + 6)
    cov_combinations <- list(c(idx[1], idx[2]),
                              c(idx[1], idx[3]),
                              c(idx[1], idx[4]))

    joint_samples_df <- 
    lapply(cov_combinations,
        function(x) {
          data.frame(s11o = round(sgrid[x[1], 1], 2), 
                    s12o = round(sgrid[x[1], 2], 2), 
                    s21o = round(sgrid[x[2], 1], 2), 
                    s22o = round(sgrid[x[2], 2], 2), 
                    y_true1 = sgrid[x[1], 3],
                    y_true2 = sgrid[x[2], 3],
                    gp1 = gp_samples[x[1], 1:nsim], 
                    gp2 = gp_samples[x[2], 1:nsim], 
                    bnn_vanilla1 = bnn_vanilla_samples[x[1], 1:nsim],
                    bnn_vanilla2 = bnn_vanilla_samples[x[2], 1:nsim],
                    bnn1 = bnn_samples[x[1], 1:nsim],
                    bnn2 = bnn_samples[x[2], 1:nsim])
        }
    ) %>% Reduce(rbind, .)

    joint_samples_df$s1o_factor <- as.factor(joint_samples_df$s22o)
    joint_samples_df$s2o_factor <- as.factor(joint_samples_df$s21o)

    # Ensure contour plots have the same "levels"
    maxlevel <- joint_samples_df %>%
      summarise(.by = c(s1o_factor, s2o_factor),
                  max_level = find_max_level(bnn1, bnn2, gp1, gp2)) %>%
      dplyr::select(max_level) %>%  max()
    minmaxlevel <- joint_samples_df %>%
      summarise(.by = c(s1o_factor, s2o_factor),
                  minmax_level = find_min_max_level(bnn1, bnn2, gp1, gp2)) %>%
      dplyr::select(minmax_level) %>%  min()
    levels <- pretty(range(0, maxlevel), max(ceiling(maxlevel*1.5/minmaxlevel), 10))

    p <- ggplot(joint_samples_df) + 
      geom_density_2d(aes(bnn1, bnn2, colour = "2"), linewidth = 0.3, breaks = levels) +
      #geom_density_2d(aes(bnn_vanilla1, bnn_vanilla2, colour = "3"), linewidth = 0.3, breaks = levels) +
      facet_grid(s1o_factor ~ s2o_factor, labeller = custom_labeller(var_num = 2)) +
      xlab(paste0("process value at (", sgrid[idx[1],1] %>% round(2), 
                  ",", round(sgrid[idx[1],2], 2), ")")) +
      ylab("process value at second spatial point") +
      geom_hline(aes(yintercept = y_true2), linetype = "dashed") +
      geom_vline(aes(xintercept = y_true1), linetype = "dashed") +
      theme_bw() +
      theme(text = element_text(size = 7),
            legend.title = element_blank(),
            strip.background = element_rect(fill="white"),
            legend.key.width = unit(0.3, "cm"),
            legend.margin = margin(t=0,r=0,b=0,l=-0.2, unit="cm"),
            strip.text = element_text(
                  margin = margin(t = 1, b = 1, l = 1, r = 1, unit = "pt"))) 

    if(models[i] == "GP") {
        p <- p + geom_density_2d(aes(gp1, gp2, colour = "1"), linewidth = 0.3, breaks = levels) +
             scale_color_manual(labels = c("1" = "Target", "2" = "SBNN-IL", "3" = "BNN (untuned)"),
                            values = c("1" = "#00BFC4", "2" = "#F8766D", "3" = "#FFD700")) 
    } else {
        p <- p + scale_color_manual(labels = c("2" = "SBNN-IP", "3" = "BNN (untuned)"),
                            values = c("2" = "#F8766D", "3" = "#FFD700")) 
    }


      save_image(path = paste0("./src/figures/Section4_4_",
                            models[i], "_joint_histograms_N", N, ".png"),
                  p, width_in = 6.2, height_in = 2.3, res = 300, device = "ggsave_png")

    # Get contour levels
    levels <- get_contour_levels(p)
    nlevels <- length(levels)

    caption <- paste0("Contour levels (from the outer to the inner contours) correspond to ", 
    paste(print_2dec(levels[-nlevels]), collapse = ", "), 
            " and ", print_2dec(levels[nlevels]), ".")
    writeLines(caption, paste0(FIG_DIR, "Section4_4_", 
                            models[i], "_Posterior_contour_levels_N", N, ".tex"))
  }
}

#############################
## Plot traces
#############################

cat(paste0("Figures of MCMC traces\n"))
s1 <- s2 <- seq(-4, 4, length.out = 64L)
sgrid_rounded <- expand.grid(s1_grid = round(s1, 2), s2_grid = round(s2, 2))

n_sub_grid <- 12
sub_grid <- seq(1,64,length.out = n_sub_grid + 2)[-c(1, n_sub_grid+2)]
            
methods <- c("SBNN-IL", "SBNN-IP")
fnames <- c("Section4_1_SBNN-IL", "Section5_2_SBNN-IP")

mean_sd <- np$load("src/intermediates/max_stable_data_Gumbel_mean_std.npy")

for(i in seq_along(fnames)) {

  sgrid_rounded$y_true <- np$load(paste0("src/runs/", fnames[i], "/output/true_field.npy"))
  samples_df <- NULL

  if (fnames[i] == "Section4_1_SBNN-IL") {
    N_list <- c(100, 1000, 5000)
  } else {
    N_list <- 50
    sgrid_rounded$y_true <- sgrid_rounded$y_true * mean_sd[2] + mean_sd[1]
  }

  for(N in N_list) {
      all_samples <- np$load(paste0("src/runs/", fnames[i], "/output/BNN_posterior_samples_N", N, ".npy")) %>% 
                    as.array()
      if(fnames[i] == "Section5_2_SBNN-IP") {
        all_samples <- all_samples <- all_samples * mean_sd[2] + mean_sd[1]
      }
      sub_samples <- all_samples[,sub_grid, sub_grid]
      ll <- list()
      for(j in 1:n_sub_grid) {
        for(k in 1:n_sub_grid) {
          ll <- c(ll, list(sub_samples[,j,k]))
        }
      }

      samples_df <- samples_df %>%
                  rbind(expand.grid(m = N,
                          sample_number = 1:800,
                          s1_grid = round(s1[sub_grid], 2),
                          s2_grid = round(s2[sub_grid], 2)) %>%
                    mutate(samples = do.call("c", ll),
                          s1o_factor =factor(round(s1_grid, 2)),
                          s2o_factor = factor(round(s2_grid, 2),
                                              levels = rev(round(s2[sub_grid], 2)))))           

  }

  samples_df$m <- as.factor(samples_df$m)
  samples_df <- left_join(samples_df, sgrid_rounded)


  g <- ggplot(samples_df) + 
            geom_point(aes(sample_number, samples, colour = m), pch = 46, size = 0.001) +
            facet_grid(s2o_factor ~ s1o_factor, scales = "free_y", labeller = custom_labeller()) +
            geom_hline(aes(yintercept = y_true), linetype = "dashed", linewidth = 0.3) +
            scale_x_continuous(breaks = c(max(samples_df$sample_number)/2, max(samples_df$sample_number))) +
            theme_bw() +
            theme(text = element_text(size = 7),
                  legend.title = element_blank(),
                  legend.position = "none",
                  #legend.key.size = unit(3, "line"),
                  strip.background = element_rect(fill = "white")) +
            ylab(TeX(r"(samples from the predictive distribution)")) +
            xlab("sample number") +
            guides(color = guide_legend(override.aes = list(size = 1)))

  save_image(path = paste0(FIG_DIR, "Section4_4_", methods[i], "-traces.png"),
                g, width_in = 7, height_in = 7, res = 400,
                device = "ggsave_png")
}


################################
## Max-Stable distribution check
################################

## Marginal distribution Gumbel?
cat(paste0("...Plots of samples\n"))

mean_sd <- np$load("src/intermediates/max_stable_data_Gumbel_mean_std.npy")

Target_samples <- (np$load("src/runs/Section5_2_SBNN-IP/output/Target_prior_samples2.npy") %>% as.matrix()) * mean_sd[2] + mean_sd[1] 
SBNN_samples <- (np$load("src/runs/Section5_2_SBNN-IP/output/BNN_prior_samples2_step10000.npy") %>% as.matrix()) * mean_sd[2] + mean_sd[1]

df <- data.frame(SBNN = as.vector(SBNN_samples),
                Target = as.vector(Target_samples[,1:200]))
df_Gumbel <- data.frame(x = seq(-3, 6, by = 0.01)) %>%
             mutate(y = dgumbel(x = seq(-3, 6, by = 0.01)))
g <- ggplot(df) + geom_histogram(aes(x = SBNN, y = ..density..),
                            binwidth = 0.3, fill = "white", colour = "black") +
                geom_line(data = df_Gumbel, aes(x = x, y = y), 
                          colour= "red", alpha = 0.5) +
  xlab(TeX(r"(\log Y(\cdot))")) +
  theme_bw() 

ggsave(g, file = paste0(FIG_DIR, "Section4_4_Max-Stable_Marginal_Gumbel.png"), width = 3, height = 3, dpi = 300)


## Spatial exceedance plots
Dmat <- fields::rdist(as.matrix(expand.grid(1:64, 1:64)))
dists <- c(1,4,8,12,16,20, 24, 28, 32)

#quants <- c(0.95, 0.98, 0.99, 0.999)
quants <- c(0.95, 0.98, 0.99, 0.995)

Dist_list <- list()
for(k in seq_along(dists)) {
   Dist_list[[k]] <- list()
   for(j in 1: nrow(Dmat)) {
    Dist_list[[k]][[j]] <- setdiff(which(Dmat[j,] >= (dists[k] - 0.5) & Dmat[j,] <= (dists[k] + 0.5)), j)
    #Dist_list[[k]][[j]] <- setdiff(which(Dmat[j,] == dists[k]), j)
   }
}

flags_target <- flags_SBNN <- list()
for (l in seq_along(quants)) {
  print(paste0("Doing quantile ", quants[l]))
  target <- qgumbel(quants[l])
  flags_target[[l]] <- flags_SBNN[[l]] <- list()
  for(k in seq_along(dists)) {
    
    
    print(paste0("...Doing distance ", dists[k]))
    flags_target[[l]][[k]] <- flags_SBNN[[l]][[k]] <- cbind(1, 0)
    

    for(i in 1:nrow(Dmat)) {
       nn <- Dist_list[[k]][[i]]
       idx <- which(Target_samples[i,] > target)
       if(length(idx) > 0) 
        flags_target[[l]][[k]] <- rbind(flags_target[[l]][[k]], 
                                        cbind(mean(Target_samples[nn, idx] > target),
                                               length(nn)))
    }
    
    for(i in 1:nrow(Dmat)) {
       nn <- Dist_list[[k]][[i]]
       idx <- which(SBNN_samples[i,] > target)
       if(length(idx) > 0) 
          flags_SBNN[[l]][[k]] <- rbind(flags_SBNN[[l]][[k]], 
                                        cbind(mean(SBNN_samples[nn, idx] > target),
                                               length(nn)))
    }
    

  }
}

exceedance_df = NULL
for (l in seq_along(quants)) {
   exceedance_df <- rbind(exceedance_df, data.frame(model = "target process", q = quants[l], d = dists, 
                    probs = unlist(lapply(flags_target[[l]], function(x) sum(x[,1] * x[,2]) / sum(x[,2])))))
   exceedance_df <- rbind(exceedance_df, data.frame(model = "SBNN-IP", q = quants[l], d = dists, 
                    probs = unlist(lapply(flags_SBNN[[l]], function(x) sum(x[,1] * x[,2]) / sum(x[,2])))))
}

exceedance_df$model <- factor(exceedance_df$model, levels = c("target process", "SBNN-IP"))

quantile_labeller <- function(variable, value) {
    return(
        lapply(value,
                function(v)
                  parse(text = sprintf(paste0("q == %s"), value))
                  ))
}

p <- 
  ggplot(exceedance_df) +
  geom_line(aes(x = d / 8, y = probs, colour = model)) +
  ylab("Conditional proportion of exceedances") +
  xlab("Spatial Lag") +
  facet_wrap(~q, labeller = quantile_labeller) +
  theme_bw() +
  theme(text = element_text(size = 7),
        legend.title = element_blank(),
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.key.width = unit(0.3, "cm"),
        legend.margin = margin(t=-0.4,r=0,b=-0.3,l=-0.2, unit="cm")) +
  ylim(c(0,1))
ggsave(p, file = paste0(FIG_DIR, "Section4_4_Spatial_Exceedance_Plot.png"), width = 3, height = 3, dpi = 300)

###############################
## TABLE 1 - Wasserstein Values
###############################

tabledf <- data.frame(fname = dir("src/runs")) %>% 
            filter(grepl("Section4", fname)) %>%
            mutate(full_path = paste0("src/runs/", fname, "/output/wsr_values.log"), 
            yaml_path = paste0("src/configs/", fname, ".yaml"),
            model = str_split(fname, "_", simplify = TRUE)[, 3],
            target = case_when(
            str_detect(fname, "Section4_1") ~ "Stationary GP",
            str_detect(fname, "Section4_2") ~ "Nonstationary GP",
            str_detect(fname, "Section4_3") ~ "Log-GP"),
            wsr = map_dbl(full_path, mean_last_n_lines, 100),
            yaml_contents = map(yaml_path, read_yaml_file)
          )
          
# Determine the complete set of unique field names across all rows
all_fields <- unique(unlist(sapply(tabledf$yaml_contents, function(x) names(x))))
tabledf2 <- tabledf %>%
            rowwise() %>%
            mutate(yaml_contents = list(fill_missing_fields(yaml_contents, all_fields))) %>%
            unnest_wider(yaml_contents)  %>%
            select(SQRT_EMBEDDING_SIZE, PRIOR_PER, SPATIALLY_VARYING_PARAMETERS, target, model, wsr) %>%
            map_dfr(~ if (is.list(.)) unlist(.) else .) %>%
            mutate(
                  num_params = if_else(is.na(as.numeric(SQRT_EMBEDDING_SIZE)), 
                                    120 + 3280 + 41,  # INPUT (3x40) + INNER 2x(40 x 41) + OUTPUT 41
                                    ((as.numeric(SQRT_EMBEDDING_SIZE)^2 + 1) * 40) + 3280 + 41),
                  num_hyperparams = if_else(PRIOR_PER == 'parameter',
                                            num_params*2,
                                              4*2*2), # 4 mappings * 2 par types (weights, biases) * 2 hypers (mean, variance)
                  num_hyperparams = if_else(SPATIALLY_VARYING_PARAMETERS,
                                              num_hyperparams * as.numeric(SQRT_EMBEDDING_SIZE)^2,
                                              num_hyperparams))
                                          
wsr_table <- tabledf2 %>%
  select(target, model, wsr) %>%
  spread(key = target, value = wsr) 
wsr_table <- wsr_table[, c(1, ncol(wsr_table):2)]
wsr_table_latex <- xtable(wsr_table)

nparam_table <- tabledf2 %>%
  select(target, model, num_params) %>%
  spread(key = target, value = num_params) 
nparam_table <- nparam_table[, c(1, ncol(nparam_table):2)]
nparam_table_latex <- xtable(nparam_table)

nhyperparam_table <- tabledf2 %>%
  select(target, model, num_hyperparams) %>%
  spread(key = target, value = num_hyperparams) 
nhyperparam_table <- nhyperparam_table[, c(1, ncol(nhyperparam_table):2)]
nhyperparam_table_latex <- xtable(nhyperparam_table)

big_table <- cbind(nparam_table[,1:2],
                   nhyperparam_table[,2],
                   wsr_table[,2:4])
names(big_table) <- c("Model",
                      "Num. par.",
                      "Num. hyper-par.",
                      "$W_1(\\psib)$ (Stat. GP)",
                      "$W_1(\\psib)$ \\newline (Non-stat. GP)",
                      "$W_1(\\psib)$ (Log-GP)")
bigtable_latex <- xtable(big_table,
                        align = c("l", "l", "p{1.5cm}", "p{1.75cm}","p{1.85cm}",
                                  "p{2.6cm}","p{1.75cm}"), 
                        digits=c(0,0,0,0,2,2,2),
                        caption = "Number of parameters, number of hyper-parameters, and Wasserstein distance on convergence for each simulation experiment and (S)BNN combination. The reported Wasserstein distance is computed as the average over the distances in the final 100 outer-loop optimisation steps. \\label{tab:results}")
# Print the LaTeX code

results_table <- print(bigtable_latex, type = "latex", 
      include.rownames = FALSE,
      sanitize.text.function = identity,
      table.placement = "!t",
      caption.placement = "top",
      comment = FALSE)

writeLines(results_table, paste0(FIG_DIR, "Section4_results_table.tex"))



# #####################################
# ## Table 2: Results RMSE, CRPS, Cov90
# #####################################

cat(paste0("Scoring\n"))
s1 <- s2 <- seq(-4, 4, length.out = 64L)
sgrid <- expand.grid(s1_grid = s1, s2_grid = s2)

for(model in c("GP", "Max-Stable")) {

  if(model == "GP") {

    model_path <- "src/runs/Section4_1_SBNN-IL/output/"
    sgrid$y_true <- np$load(paste0(model_path, "true_field.npy"))

    fnames <- c(paste0("Target_posterior_samples_N", c(100, 1000, 5000)),
                paste0("BNN_posterior_samples_N", c(100, 1000, 5000)),
                paste0("BNN_Vanilla_posterior_samples_N", c(100, 1000, 5000)))
    N_list <- rep(c(100L,1000L,5000L), 3)
    method <- c(rep("GP", 3), rep("SBNN-IL", 3), rep("BNN-IL (untuned)", 3))

  } else {
    model_path <- "src/runs/Section5_2_SBNN-IP/output/"
    sgrid$y_true <- np$load(paste0(model_path, "true_field.npy"))

    fnames <- c(paste0("BNN_posterior_samples_N", c(50, 100, 1000, 5000)),
              paste0("BNN_Vanilla_posterior_samples_N", c(50, 100, 1000, 5000)))
    N_list <- rep(c(50, 100L, 1000L, 5000L), 2)
    method <- c(rep("SBNN-IP", 4), rep("BNN-IL (untuned)", 4))
  }

  samples_df <- NULL
  for(i in seq_along(fnames)) {

      obs <- np$load(paste0(model_path, "obs_N", N_list[i], ".npy")) %>%
                        as.data.frame() %>% distinct()
      names(obs) <- c("s1_grid", "s2_grid")
      obs$obs <- 1

      all_samples <- np$load(paste0(model_path, fnames[i], ".npy")) %>% 
                    as.array()
      
      ll <- list()
      for(j in 1:64) {
        for(k in 1:64) {
          ll <- c(ll, list(all_samples[,j,k]))
        }
      }

      n_samples <- dim(all_samples)[1]
      
      samples_df <- samples_df %>%
                  rbind(expand.grid(m = N_list[i], 
                          method = method[i],
                          sample_number = 1:n_samples,
                          s1_grid = s1, 
                          s2_grid = s2) %>%
                          mutate(samples = do.call("c", ll)) %>% left_join(obs))
      
  }
  samples_df <- left_join(samples_df, sgrid) %>% filter(is.na(obs))

  Scoring_results <- group_by(samples_df, method, m, s1_grid, s2_grid) %>% 
        summarise(pred = mean(samples), 
                  pred_sd = sd(samples), 
                  y_true = mean(y_true)) %>%
        group_by(method, m) %>%
        summarise(MAPE = mean(abs(pred - y_true)),
                  RMSPE = sqrt(mean((pred - y_true)^2)),
                  CRPS = verification::crps(y_true, cbind(pred, pred_sd))$CRPS) %>%
                  arrange(m)

  if(model == "Max-Stable") {
    print(Scoring_results)
  }               

  if(model == "GP") {

    model_label <- c("GP (True Model)", "SBNN-IL (Calibrated)", "BNN-IL (Uncalibrated)")
    model_name <- c("GP", "SBNN-IL", "BNN-IL (untuned)")

    # for(mm in seq_along(model_name)) {

    #   Scoring_results_mm <-  Scoring_results %>% 
    #                           rename(Model = method) %>%
    #                           filter(Model == model_name[mm]) %>%
    #                           mutate(Model = model_label[mm])

    #   results_table <- xtable(Scoring_results_mm, digits = 3) %>% 
    #                    print(only.contents = TRUE, comment = FALSE, include.rownames = FALSE)
    #   writeLines(results_table, paste0(FIG_DIR, "Section5_2_Scoring_Results_", model_name[mm], ".tex"))

    # }

    map_table <- data.frame(Model_label = model_label, Model = model_name)
    Scoring_results_all <-  Scoring_results %>% 
                              ungroup() %>%
                              rename(Model = method) %>%
                              arrange(m) %>% 
                              left_join(map_table) %>%
                              dplyr::select(-Model)
    Scoring_results_all <- Scoring_results_all[,c(5,1:4)]                          
    names(Scoring_results_all)[1] <- "Model"
    results_table <- xtable(Scoring_results_all, digits = 3) %>% 
                      print(only.contents = TRUE, comment = FALSE, include.rownames = FALSE)
                      

    # Split the output into lines
    lines <- unlist(strsplit(results_table, "\n"))

    # Insert \hline at the desired positions
    lines <- append(lines, "\\hline", 5) # Inserts at 4th line
    lines <- append(lines, "\\hline", 9) # Adjusts to 8th because of previous insertion

    # Combine back into a single string
    modified_table <- paste(lines, collapse = "\n")


    writeLines(modified_table, paste0(FIG_DIR, "Section4_4_Scoring_Results_all.tex"))

  }
}
