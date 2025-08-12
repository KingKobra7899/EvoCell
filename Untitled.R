library(dplyr)
library(ggplot2)
simulation_data <- simulation_data[simulation_data$avg_predation > 0,]

ggplot(simulation_data, aes(x = time, y = num_cells))+
  geom_line()
