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


ggplot(simulation_data, aes(x = seconds)) +
  geom_line(aes(y = num_cells, color = "Cells"), size = 1.2) +
  geom_line(aes(y = num_plants, color = "Plants"), size = 1.2) +
  scale_color_manual(values = c("Cells" = "#2E86AB", "Plants" = "#A23B72")) +
  labs(x = "Time (seconds)", y = "Count", color = "") +
  theme_minimal() +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "top",
    plot.background = element_rect(fill = "white", color = NA),
    text = element_text(family = "Arial", color = "#2c3e50"),
    axis.title = element_text(size = 12),
    legend.text = element_text(size = 11)
  )
