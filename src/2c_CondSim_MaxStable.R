## Use this script to see how conditional simulation does against the SBNNs

library(ggplot2)
library(fields)
library(rjson)
library(tidyr)
library(stringr)
library(latex2exp)
library(reticulate)
library(dplyr, warn.conflicts = FALSE)
library(Matrix, warn.conflicts = FALSE)
library(SpatialExtremes)
library(gridExtra)
library(parallel)

set.seed(1)
np <- import('numpy') # Use reticulate to import numpy
source("src/utils.R")
FIG_DIR <- "src/figures/"
dir.create(FIG_DIR, showWarnings = FALSE)

s1 <- s2 <- seq(-4, 4, length.out = 64L)
sgrid <- expand.grid(s1 = s1, s2 = s2)
fnames <- c("Section4_1_SBNN-IL", "Section5_2_SBNN-IP")
models <- c("GP", "Max-Stable")
sgrid$y_true <- np$load(paste0("src/runs/Section5_2_SBNN-IP/output/true_field.npy"))

obs <- np$load(paste0("src/runs/Section5_2_SBNN-IP/output/obs_N50.npy")) %>%
                      as.data.frame()
    names(obs) <- c("s1", "s2")
obs$z <- np$load(paste0("src/runs/Section5_2_SBNN-IP/output/obs_N50_vals.npy")) 
g1 <- ggplot(sgrid) + geom_tile(aes(s1,s2,fill = y_true)) +
    geom_point(data = obs, aes(s1, s2), size = 14) +
    geom_point(data = obs, aes(s1, s2, color = z), size = 10) +
    scale_fill_distiller(palette = "Spectral") +
    scale_colour_distiller(palette = "Spectral")

lscale <- 3

# warning("Choosing subsamples of obs")
# obs <- sample_n(obs, 50L, replace = FALSE)

DistMat_Grid <- fields::rdist(sgrid %>% select(s1, s2) %>% as.matrix(), 
        obs %>% select(s1, s2) %>% as.matrix())
idx_pred_to_rm <- apply(DistMat_Grid, 2, function(x) which(x < 1e-6))
idx_pred_locs <- (1:64^2)[-idx_pred_to_rm]

DistMat_Obs <- fields::rdist(obs %>% select(s1, s2) %>% as.matrix(), 
        obs %>% select(s1, s2) %>% as.matrix())
idx_obs_locs <- apply(DistMat_Obs, 2, function(x) max(which(x < 0.12))) %>% unique()

#idx_sub <- sample(idx_pred_locs, 3999, replace = FALSE)
idx_sub <- idx_pred_locs

sgrid$y_true <- np$load(paste0("src/runs/Section5_2_SBNN-IP/output/true_field.npy"))
mean_sd <- np$load(paste0("src/intermediates/max_stable_data_Gumbel_mean_std.npy"))
obs$z_frechet <- (obs$z * mean_sd[2] + mean_sd[1]) %>% exp()


system.time({X <- lapply(1, function(i) 
        condrmaxstab(30, coord = sgrid[idx_pred_locs, 1:2] %>% as.matrix(),
            cond.coord = obs[idx_obs_locs ,1:2] %>% as.matrix(),
            cond.data = obs[idx_obs_locs, "z_frechet"],
            cov.mod = "powexp", 
            nugget = 0.001, range = lscale,
            smooth = 1.5))})

for(i in 1:nrow(X[[1]]$sim)) {
    sim_name = paste0("sim", i)
    sgrid <- sgrid %>% mutate(!!sim_name := NA)
    sgrid[idx_sub, sim_name] <- (log(X[[1]]$sim[i,]) - mean_sd[1])/mean_sd[2]
}


all_samples <- gather(sgrid[idx_sub, ], sim, samples, -s1,-s2,-y_true)
saveRDS(all_samples, file = "src/intermediates/Max-Stable_CondSim.rds")

Scoring_results <- group_by(all_samples, s1, s2) %>% 
      summarise(pred = mean(samples), 
                pred_sd = sd(samples), 
                y_true = mean(y_true)) %>% ungroup() %>%
      summarise(MAPE = mean(abs(pred - y_true)),
                RMSPE = sqrt(mean((pred - y_true)^2)),
                CRPS = verification::crps(y_true, cbind(pred, pred_sd))$CRPS) 


g2 <- ggplot(sgrid) + geom_tile(aes(s1,s2,fill = y_pred)) +
    geom_point(data = obs, aes(s1, s2), size = 14) +
    geom_point(data = obs, aes(s1, s2, color = z), size = 10) +
    scale_fill_distiller(palette = "Spectral") +
    scale_colour_distiller(palette = "Spectral")


grid.arrange(grobs = list(g1,g2), nrow = 1)
