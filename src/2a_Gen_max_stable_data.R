library("SpatialExtremes")
library("dplyr")
library("ggplot2")
library("parallel")

set.seed(1)

s1 <- s2 <- seq(-4, 4, length.out = 64)
s <- cbind(s1, s2)
df <- expand.grid(s1 = s1, 
                  s2 = s2) %>%
  mutate(z = c(rmaxstab(1, coord = s, cov.mod = "powexp", 
                        nugget = 0, range = 3,
                        smooth = 1.5, grid = TRUE)) %>% log())

g <- ggplot(df) + geom_tile(aes(s1, s2, fill = z)) +
  scale_fill_distiller(palette = "Spectral") + theme_bw()
  print(g)

## Simulate data
N <- 1000000L
#lscales <- runif(N, min = 1, max = 3)
lscales <- rep(3, N)
  
X <- mclapply(lscales, function(l) 
                rmaxstab(1, coord = s, cov.mod = "powexp", 
                    nugget = 0, range = l,
                    smooth = 1.5, grid = TRUE), 
                mc.cores = 50)

all_data <- simplify2array(X) %>%
            aperm(c(3,1,2))

saveRDS(all_data, file = "src/data/max_stable_sims.rds")
saveRDS(all_data[1:10,,], file = "src/data/max_stable_sims_small.rds")
