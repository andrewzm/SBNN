# find_subgrid_indices
#
# Find the indices of a subgrid created by using vectors `s1` and `s2` with 
# a desired number of elements `n1` and `n2` respectively.
#
# @param s1 A numeric vector representing the first grid axis.
# @param s2 A numeric vector representing the second grid axis.
# @param n1 Desired number of elements for the subgrid along the `s1` axis.
# @param n2 Desired number of elements for the subgrid along the `s2` axis.
#
# @details 
# The function begins by creating sequences from the input vectors `s1` and `s2` 
# with lengths `n1 + 2` and `n2 + 2` respectively. The initial and final indices of these 
# sequences are then removed to form the `s1idx` and `s2idx` vectors.
#
# The `focal_axis` dataframe is then formed using the remaining indices from `s1` 
# and `s2`. An expanded grid, `focal_grid`, is generated from the `focal_axis` 
# dataframe.
# 
# The function iterates over each row of the `focal_grid` and finds the index 
# of the closest point from the full grid (`sgrid`), using the Euclidean distance 
# formula. The indices of these closest points are returned.
#
# @return A numeric vector containing indices of the subgrid formed by `s1` and `s2`.
#
# @examples
# s1_vec <- c(1, 2, 3, 4, 5)
# s2_vec <- c(5, 6, 7, 8, 9)
# find_subgrid_indices(s1_vec, s2_vec, 3, 3)
#
# Note: The provided code has a combination of base R and functional programming 
#       styles. Depending on your use-case and the rest of the codebase, you might 
#       consider refactoring or optimizing certain parts for consistency and performance.
find_subgrid_indices <- function(s1, s2, n1 , n2) {

    n_s1 <- length(s1)
    n_s2 <- length(s2)

    s1idx <- round(seq(1, n_s1, length.out = n1 + 2)[c(-1, -(n1 + 2))])
    s2idx <- round(seq(1, n_s2, length.out = n2 + 2)[c(-1, -(n2 + 2))])

    focal_grid <- expand.grid(s1 = s1[s1idx], s2 = s2[s2idx])

    sgrid <- expand.grid(s1 = s1, s2 = s2)
    idx <- NULL
    for(i in 1:nrow(focal_grid)) {
        idx[i] <- which.min((sgrid$s1 - focal_grid$s1[i])^2 +
                            (sgrid$s2 - focal_grid$s2[i])^2) 
    }
    idx
}



custom_labeller <- function(var_num = "0") {
    function(variable, value) {
    
    if (variable == "s1o_factor") {
      return(
        lapply(value,
                function(v)
                  parse(text = sprintf(paste0("s[", var_num, "*','*1] == %s"), v))
                  ))
    } else if (variable == "s2o_factor") {
      return(
        lapply(value,
                function(v)
                  parse(text = sprintf(paste0("s[", var_num, "*','*2] == %s"), v))
      ))
    }
  }
}

custom_labeller2 <- function(var_num = "0") {
  function(labels) {
    lapply(names(labels), function(variable) {
      setNames(
        sapply(labels[[variable]], function(value) {
          if (variable == "s1o_factor") {
            return(sprintf(paste0("$s_{", var_num, ",1} == %s$"), value))
          } else if (variable == "s2o_factor") {
            return(sprintf(paste0("$s_{", var_num, ",2} == %s$"), value))
          } else {
            return(value)
          }
        }, USE.NAMES = TRUE),
        labels[[variable]]
      )
    })
  }
}

get_contour_levels <- function(p) {

  # Build the plot
  plot_build <- ggplot2::ggplot_build(p)

  # Extract the data for the contour layer (assuming it's the first layer)
  contour_levels1 <- unique(plot_build$data[[1]]$level)
  contour_levels2 <- unique(plot_build$data[[2]]$level)
  
  # Check that one is equal, or an extension to, the other
  if(!(length(setdiff(contour_levels1, contour_levels2)) == 0 | 
       length(setdiff(contour_levels2, contour_levels1)) == 0))
   stop("Contour levels do not line up")

  # Get the levels
  unique(c(contour_levels1, contour_levels2))

}

print_1dec <- function(x) sprintf("%.1f", round(x, 1))
print_2dec <- function(x) sprintf("%.2f", round(x, 2))


# Define a function to read the last line of a file and extract the number
mean_last_n_lines <- function(path, n = 1) {
  if (file.exists(path)) {
    lines <- tail(readLines(path), n)
    mean(as.numeric(str_extract(lines, "[-+]?\\d*\\.?\\d+([eE][-+]?\\d+)?")))
  } else {
    NA
  }
}


# Define a function to read a YAML file and convert it to a list
read_yaml_file <- function(path) {
  if (file.exists(path)) {
    suppressWarnings({yaml.load_file(path)})
  } else {
    list(NA)  # Returns a list with NA if file does not exist
  }
}


# Create a function to fill missing fields with NA for each row in YAML contents
fill_missing_fields <- function(yaml_content, all_fields) {
  missing_fields <- setdiff(all_fields, names(yaml_content))
  if (length(missing_fields) > 0) {
    yaml_content[missing_fields] <- NA
  }
  yaml_content
}

read_last_100_lines_avg <- function(path) {
  lines <- readLines(path)
  
  # Take the last 100 lines (or fewer if the file has less than 100 lines)
  last_100_lines <- tail(lines, n = min(100, length(lines)))
  
  # Convert to numeric values
  values <- as.numeric(last_100_lines)
  
  # Return the average
  mean(values, na.rm = TRUE)
}


## Find the maximum level from samples from two bivariate variables
find_max_level <- function(x1, x2, y1, y2, n = 100) {
  kde1 <- kde2d(x1, x2, n = 100)
  kde2 <- kde2d(y1, y2, n = 100)
  max(c(kde1$z, kde2$z))
}

find_min_max_level <- function(x1, x2, y1, y2, n = 100) {
  kde1 <- kde2d(x1, x2, n = 100)
  kde2 <- kde2d(y1, y2, n = 100)
  min(c(max(kde1$z), max(kde2$z)))
}

## Save image
save_image <- function(path, p, width_in, height_in, res = 300, device = "png") {

    if(device == "png") {
        png(path, type = "cairo", 
            width = width_in * res, height = height_in * res, res = res)
        plot(p)
        dev.off()
    } else if(device == "ggsave_png") {
       ggsave(file = path, 
              p, width = width_in, height = height_in, dpi = res)
    } else {
      stop("device must be png or ggsave_png")
    }
}

format_two_decimals <- function(x) {
      sprintf("%.2f", as.numeric(x))
    }