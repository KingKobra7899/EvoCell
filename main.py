import pygame as pg
import pygame_widgets as pw
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import numpy as np
import json

class Particle:
    """Class representing a particle in the creature's vision."""
    def __init__(self, position, is_plant):
        # position in normalized coordinates: (0,0) center, (1,0) right edge
        self.position = np.array(position, dtype=np.float32)
        self.is_plant = is_plant

    def display(self, screen):
        width, height = screen.get_size()
        sim_area_width = width - 450  # Simulation area width (right side only)

        # Convert normalized coordinates to pixel coordinates in the simulation area
        pixel_x = int(450 + (self.position[0] + 1) * 0.5 * sim_area_width)
        pixel_y = int((1 - (self.position[1] + 1) * 0.5) * height)  # invert Y

        if self.is_plant:
            # Enhanced plant appearance with gradient effect
            base_color = (34, 139, 34)  # Forest green
            outline_color = (0, 100, 0)  # Darker green
            pg.draw.circle(screen, base_color, (pixel_x, pixel_y), 8)
            pg.draw.circle(screen, outline_color, (pixel_x, pixel_y), 8, 2)
            # Add a small highlight
            pg.draw.circle(screen, (144, 238, 144), (pixel_x - 2, pixel_y - 2), 3)
        else:
            # Enhanced non-plant appearance
            base_color = (70, 130, 180)  # Steel blue
            outline_color = (25, 25, 112)  # Midnight blue
            pg.draw.circle(screen, base_color, (pixel_x, pixel_y), 8)
            pg.draw.circle(screen, outline_color, (pixel_x, pixel_y), 8, 2)
            # Add a small highlight
            pg.draw.circle(screen, (176, 196, 222), (pixel_x - 2, pixel_y - 2), 3)

class BrainAnalyzer:
    def __init__(self, json_filepath):
        self.json_filepath = json_filepath
        brain_data = json.load(open(json_filepath, 'r'))
        self.brain_size = brain_data['brain_size']
        input_size = self.brain_size + 65
        output_size = self.brain_size

        self.encoder_weights = np.array(brain_data['encoder_weights'], dtype=np.float32).reshape(output_size, input_size)
        self.encoder_bias = np.array(brain_data['encoder_bias'], dtype=np.float32).reshape(output_size, 1)
        self.social_weights = np.array(brain_data['social_weights'], dtype=np.float32)
        self.isolation_weights = np.array(brain_data['isolation_weights'], dtype=np.float32)
        self.hunger_weights = np.array(brain_data['hunger_weights'], dtype=np.float32)
        self.predation = brain_data['predation']
    
    def create_environment(self, particles, old_encoding, old_h, old_i, old_s, energy_deficit, metabolic_rate):
        full_env = np.zeros((self.brain_size + 65, 1), dtype=np.float32)
        
        for idx, particle in enumerate(particles[:20]):
            base = idx * 3
            full_env[base] = particle.position[0]
            full_env[base + 1] = particle.position[1]
            full_env[base + 2] = 1.0 if particle.is_plant else 0.0

        full_env[60] = old_h
        full_env[61] = old_i
        full_env[62] = old_s
        full_env[63] = energy_deficit
        full_env[64] = metabolic_rate
        
        full_env[65:] = old_encoding.reshape(-1, 1)
        
        return full_env

    def get_desired_movement(self, full_env, particles):
        encoder_output = np.tanh(self.encoder_weights @ full_env + self.encoder_bias)
        
        social_output = np.dot(self.social_weights, encoder_output)
        isolation_output = np.dot(self.isolation_weights, encoder_output)
        hunger_output = np.dot(self.hunger_weights, encoder_output)
        
        desired_movement = np.zeros(3, dtype=np.float32)
        desired_movement[0] = social_output[0]  
        desired_movement[1] = hunger_output[0]
        desired_movement[2] = isolation_output[0]
        
        social_position = np.array([0.0, 0.0], dtype=np.float32)
        for particle in particles:
            if not particle.is_plant:
                social_position += particle.position
        if len(particles) > 0:
            social_position /= len(particles)
        isolation_position = -1 * social_position
        
        hunger_position = np.array([0.0, 0.0], dtype=np.float32)
        for particle in particles:
            if particle.is_plant or self.predation > 0:
                hunger_position += particle.position
        if len(particles) > 0:
            hunger_position /= len(particles)
        
        max_drive = sum(desired_movement)
        if max_drive > 0:
            desired_movement /= max_drive
        else:
            desired_movement = np.array([0.333, 0.333, 0.333], dtype=np.float32)
        
        movement_vector = np.zeros(2, dtype=np.float32)
        if desired_movement[0] > 0:
            movement_vector += social_position * desired_movement[0]
        if desired_movement[1] > 0:
            movement_vector += hunger_position * desired_movement[1]
        if desired_movement[2] > 0:
            movement_vector += isolation_position * desired_movement[2]
        
        return movement_vector, desired_movement

def draw_rounded_rect(surface, color, rect, radius):
    """Draw a rounded rectangle"""
    pg.draw.rect(surface, color, rect, border_radius=radius)

def draw_gradient_background(screen, color1, color2):
    """Draw a subtle gradient background"""
    width, height = screen.get_size()
    for y in range(height):
        ratio = y / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pg.draw.line(screen, (r, g, b), (0, y), (width, y))

def draw_brain_with_neural_network(screen, center, radius, activity_level):
    """Draw an enhanced brain visualization with neural network appearance"""
    # Main brain body with gradient
    brain_color = (60, 60, 80)
    highlight_color = (120, 120, 140)
    
    # Draw main brain circle with gradient effect
    pg.draw.circle(screen, brain_color, center, radius)
    pg.draw.circle(screen, highlight_color, center, radius, 3)
    
    # Draw neural network pattern inside
    for i in range(8):
        angle = i * np.pi / 4
        inner_x = center[0] + int(np.cos(angle) * radius * 0.6)
        inner_y = center[1] + int(np.sin(angle) * radius * 0.6)
        pg.draw.circle(screen, (100, 100, 120), (inner_x, inner_y), 3)
        pg.draw.line(screen, (80, 80, 100), center, (inner_x, inner_y), 1)
    
    # Central processing node
    pg.draw.circle(screen, (200, 200, 220), center, 5)

def draw_vector_arrow(screen, start, end, color, thickness=3):
    """Draw an arrow from start to end point"""
    if np.linalg.norm(np.array(end) - np.array(start)) < 1:
        return
    
    # Draw main line
    pg.draw.line(screen, color, start, end, thickness)
    
    # Calculate arrow head
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    arrow_length = 15
    arrow_angle = np.pi / 6
    
    # Arrow head points
    point1 = (
        end[0] - arrow_length * np.cos(angle - arrow_angle),
        end[1] - arrow_length * np.sin(angle - arrow_angle)
    )
    point2 = (
        end[0] - arrow_length * np.cos(angle + arrow_angle),
        end[1] - arrow_length * np.sin(angle + arrow_angle)
    )
    
    pg.draw.polygon(screen, color, [end, point1, point2])

# Initialize Pygame
pg.init()
font = pg.font.Font(None, 24)
title_font = pg.font.Font(None, 36)

# Screen setup with better dimensions
screen = pg.display.set_mode((1200, 900))
pg.display.set_caption("Neural Brain Behavior Analyzer")

# Color scheme
BACKGROUND_LIGHT = (248, 249, 250)
BACKGROUND_DARK = (240, 242, 245)
PANEL_COLOR = (255, 255, 255)
TEXT_COLOR = (33, 37, 41)
ACCENT_COLOR = (0, 123, 255)
SUCCESS_COLOR = (40, 167, 69)
WARNING_COLOR = (255, 193, 7)
DANGER_COLOR = (220, 53, 69)

running = True
particles = []
brain = BrainAnalyzer('brain.json')

# Create control panel with better positioning
panel_x = 50
panel_y = 50
panel_width = 350
slider_width = 200
slider_height = 25

# Enhanced sliders with better styling
old_h_slider = Slider(screen, panel_x + 20, panel_y + 80, slider_width, slider_height, 
                     min=0, max=1, step=0.01, initial=0.5,
                     colour=(200, 200, 200), handleColour=DANGER_COLOR)
old_i_slider = Slider(screen, panel_x + 20, panel_y + 130, slider_width, slider_height,
                     min=0, max=1, step=0.01, initial=0.5,
                     colour=(200, 200, 200), handleColour=WARNING_COLOR)
old_s_slider = Slider(screen, panel_x + 20, panel_y + 180, slider_width, slider_height,
                     min=0, max=1, step=0.01, initial=0.5,
                     colour=(200, 200, 200), handleColour=SUCCESS_COLOR)
energy_deficit_slider = Slider(screen, panel_x + 20, panel_y + 230, slider_width, slider_height,
                              min=-1, max=1, step=0.01, initial=0.0,
                              colour=(200, 200, 200), handleColour=ACCENT_COLOR)
metabolic_rate_slider = Slider(screen, panel_x + 20, panel_y + 280, slider_width, slider_height,
                              min=0, max=1, step=0.01, initial=0.5,
                              colour=(200, 200, 200), handleColour=(128, 0, 128))

old_encoding = np.zeros(brain.brain_size, dtype=np.float32)

clock = pg.time.Clock()

while running:
    events = pg.event.get()
    
    for event in events:
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            width, height = screen.get_size()
            mouse_x, mouse_y = pg.mouse.get_pos()
            
            # Only add particles if clicking in the main area (not on controls)
            if mouse_x > 450:  # Right side of screen
                # Convert to normalized coordinates: center = (0,0), range [-1, 1]
                sim_area_width = width - 450
                norm_x = ((mouse_x - 450) / sim_area_width) * 2 - 1
                norm_y = -((mouse_y / height) * 2 - 1)  # invert Y so up is positive
                
                pos = (norm_x, norm_y)
                
                if event.button == 1:  # Left click = plant
                    particles.append(Particle(pos, True))
                elif event.button == 3:  # Right click = non-plant
                    particles.append(Particle(pos, False))
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_c:  # Clear particles with 'C' key
                particles.clear()

    # Draw gradient background
    draw_gradient_background(screen, BACKGROUND_LIGHT, BACKGROUND_DARK)
    
    # Draw control panel background
    panel_rect = pg.Rect(panel_x, panel_y, panel_width, 350)
    draw_rounded_rect(screen, PANEL_COLOR, panel_rect, 10)
    pg.draw.rect(screen, (220, 220, 220), panel_rect, 2, border_radius=10)
    
    # Draw title
    title_text = title_font.render("Neural Behavior Control", True, TEXT_COLOR)
    screen.blit(title_text, (panel_x + 20, panel_y + 15))
    
    # Update slider values
    old_h = old_h_slider.getValue()
    old_i = old_i_slider.getValue()
    old_s = old_s_slider.getValue()
    energy_deficit = energy_deficit_slider.getValue()
    metabolic_rate = metabolic_rate_slider.getValue()
    
    # Draw parameter labels and values
    labels = [
        (f"Hunger Drive: {old_h:.2f}", DANGER_COLOR, panel_y + 65),
        (f"Isolation Drive: {old_i:.2f}", WARNING_COLOR, panel_y + 115),
        (f"Social Drive: {old_s:.2f}", SUCCESS_COLOR, panel_y + 165),
        (f"Energy Deficit: {energy_deficit:.2f}", ACCENT_COLOR, panel_y + 215),
        (f"Metabolic Rate: {metabolic_rate:.2f}", (128, 0, 128), panel_y + 265)
    ]
    
    for label_text, color, y_pos in labels:
        text = font.render(label_text, True, color)
        screen.blit(text, (panel_x + 20, y_pos))
    
    # Draw instructions
    instructions = [
        "Left Click: Add Plant",
        "Right Click: Add Animal", 
        "Press 'C': Clear All"
    ]
    
    for i, instruction in enumerate(instructions):
        text = font.render(instruction, True, TEXT_COLOR)
        screen.blit(text, (panel_x + 20, panel_y + 320 + i * 25))
    
    # Process brain analysis
    full_env = brain.create_environment(particles, old_encoding, old_h, old_i, old_s, energy_deficit, metabolic_rate)
    movement_vector, desired_movement = brain.get_desired_movement(full_env, particles)
    
    # Draw simulation area background
    sim_area = pg.Rect(450, 0, 750, 900)
    draw_rounded_rect(screen, (250, 250, 250), sim_area, 0)
    
    # Draw grid for better spatial reference
    grid_color = (230, 230, 230)
    for i in range(450, 1200, 50):
        pg.draw.line(screen, grid_color, (i, 0), (i, 900), 1)
    for i in range(0, 900, 50):
        pg.draw.line(screen, grid_color, (450, i), (1200, i), 1)
    
    # Draw particles with enhanced appearance
    for particle in particles:
        particle.display(screen)
    
    # Draw the brain at the center of the simulation area
    brain_center = (825, 450)  # Center of right side
    brain_radius = 25
    
    # Calculate activity level based on movement magnitude
    activity_level = min(1.0, np.linalg.norm(movement_vector) * 2)
    
    draw_brain_with_neural_network(screen, brain_center, brain_radius, activity_level)
    
    # Draw movement vector as arrow
    if np.linalg.norm(movement_vector) > 0.01:
        vector_scale = 200
        end_x = brain_center[0] + int(movement_vector[0] * vector_scale)
        end_y = brain_center[1] - int(movement_vector[1] * vector_scale)  # invert Y axis
        
        # Color code the arrow based on dominant drive
        max_drive_idx = np.argmax(desired_movement)
        arrow_colors = [SUCCESS_COLOR, DANGER_COLOR, WARNING_COLOR]  # social, hunger, isolation
        arrow_color = arrow_colors[max_drive_idx] if max_drive_idx < 3 else TEXT_COLOR
        
        draw_vector_arrow(screen, brain_center, (end_x, end_y), arrow_color, 4)
    
    # Draw drive indicators
    drive_labels = ["Social", "Hunger", "Isolation"]
    drive_colors = [SUCCESS_COLOR, DANGER_COLOR, WARNING_COLOR]
    
    for i, (label, color, value) in enumerate(zip(drive_labels, drive_colors, desired_movement)):
        bar_x = panel_x + 250
        bar_y = panel_y + 80 + i * 40
        bar_width = int(abs(value) * 80)
        
        # Draw drive bar
        if value > 0:
            pg.draw.rect(screen, color, (bar_x, bar_y, bar_width, 15))
        pg.draw.rect(screen, (150, 150, 150), (bar_x, bar_y, 80, 15), 1)
        
        # Draw label
        text = font.render(f"{label}: {value:.2f}", True, color)
        screen.blit(text, (bar_x - 80, bar_y - 2))
    
    # Update and draw sliders
    for slider in [old_h_slider, old_i_slider, old_s_slider, energy_deficit_slider, metabolic_rate_slider]:
        slider.listen(events)
        slider.draw()
    
    # Draw legend
    legend_y = 750
    legend_items = [
        ("🟢 Plants", (34, 139, 34)),
        ("🔵 Animals", (70, 130, 180)),
        ("🧠 Neural Agent", (60, 60, 80))
    ]
    
    for i, (item, color) in enumerate(legend_items):
        text = font.render(item, True, color)
        screen.blit(text, (panel_x + 20, legend_y + i * 25))
    
    pg.display.flip()
    clock.tick(60)  # 60 FPS for smooth interaction

pg.quit()