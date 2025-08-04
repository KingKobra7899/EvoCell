library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(scales)
library(RColorBrewer)

# Create time steps (assuming each row represents one time step)
simulation_data$steps <- 1
simulation_data$seconds <- cumsum(simulation_data$steps)

simulation_data$avg_drive <- apply(
  simulation_data[, c("avg_hunger", "avg_social", "avg_isolation")],
  1,
  function(x) {
    c("hunger", "social", "isolation")[which.max(x)]
  }
)
