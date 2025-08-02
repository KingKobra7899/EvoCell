library(ggplot2)

# Create time steps (assuming each row represents one time step)
simulation_data$steps <- 1
simulation_data$seconds <- cumsum(simulation_data$steps)

# Create the plot with a modern aesthetic
ggplot(simulation_data, aes(x = seconds)) +
  geom_line(aes(y = num_cells, color = "Cells"), size = 1.2, alpha = 0.8) +
  geom_line(aes(y = num_plants, color = "Plants"), size = 1.2, alpha = 0.8) +
  scale_color_manual(values = c("Cells" = "#3B528BFF", "Plants" = "#5DC863FF")) +
  labs(
    title = "Simulation: Cell and Plant Populations Over Time",
    x = "Time (seconds)",
    y = "Population Count",
    color = "Population Type"
  ) +
  ylim(0, 500) +
  theme_light() + # Use theme_light for a clean, professional look
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title = element_text(size = 12),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95")
  )