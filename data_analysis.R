library(ggplot2)
library(tidyr)
library(dplyr)


# Enhanced color palette
colors <- c("#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4")

# Prepare data with refined variable names
plot_data <- simulation_data %>%
  select(time, avg_speed, num_cells, avg_brain_size, avg_sight_r) %>%
  pivot_longer(cols = -time, names_to = "variable", values_to = "value") %>%
  mutate(
    variable = factor(variable,
                      levels = c("avg_brain_sizex", "avg_sight_r", "avg_speed", "num_cells"),
                      labels = c("Neural Complexity", "Vision Range", "Locomotion Speed", "Population Size")
    )
  )

# Create color mapping
color_mapping <- setNames(colors, levels(plot_data$variable))

# Enhanced visualization
ggplot(plot_data, aes(x = time, y = value, color = variable)) +
  geom_line(size = 2.5, alpha = 0.85) +
  geom_point(size = 1.8, alpha = 0.7, stroke = 0) +
  facet_wrap(~ variable, scales = "free_y", ncol = 2, 
             labeller = labeller(variable = function(x) paste("●", x))) +
  scale_color_manual(values = color_mapping) +
  theme_void(base_size = 12) +
  labs(
    x = "Simulation Time",
    y = NULL
  ) +
  theme(
    # Facet styling
    strip.text = element_text(size = 16, face = "bold", 
                              margin = margin(t = 20, b = 15),
                              color = "#34495e"),
    strip.background = element_blank(),
    
    # Axis styling
    axis.title.x = element_text(size = 13, face = "bold", 
                                margin = margin(t = 25),
                                color = "#34495e"),
    axis.text.x = element_text(size = 11, color = "#7f8c8d",
                               margin = margin(t = 8)),
    axis.text.y = element_text(size = 11, color = "#7f8c8d",
                               margin = margin(r = 8)),
    axis.line.x = element_line(color = "#bdc3c7", size = 0.8),
    axis.ticks.x = element_line(color = "#bdc3c7", size = 0.6),
    axis.ticks.length.x = unit(6, "pt"),
    
    # Grid customization
    panel.grid.major.y = element_line(color = "#ecf0f1", size = 0.6),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    
    # Layout
    panel.spacing = unit(2, "lines"),
    plot.margin = margin(30, 40, 30, 40),
    
    # Background
    plot.background = element_rect(fill = "#fafafa", color = NA),
    panel.background = element_rect(fill = "white", color = NA,
                                    size = 1.2),
    
    # Remove legend
    legend.position = "none"
  ) +
  # Add subtle shadow effect with secondary geom
  geom_line(aes(y = value), size = 3, alpha = 0.15, color = "black") +
  geom_line(aes(color = variable), size = 2.5, alpha = 0.85)