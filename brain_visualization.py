import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load JSON
with open("brain.json", "r") as f:
    brain = json.load(f)

brain_size = brain["brain_size"]
weights_1d = np.array(brain["encoder_weights"])

# In Rust: DMatrix::zeros(output_size, input_size) means (rows=output, cols=input)
# So we reshape to (output_size, input_size)
output_size = 8
input_size = len(weights_1d) // output_size
weights = weights_1d.reshape(output_size, input_size)

# Seaborn heatmap expects rows as y, cols as x, so we transpose:
weights_T = weights.T  # shape: (input_size, output_size)

# Create heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(weights_T, cmap="coolwarm", center=0, cbar_kws={'label': 'Weight Value'})

# Highlight triplet groups in first 60 inputs
for triplet_boundary in range(3, 61, 3):
    ax.axhline(triplet_boundary, color='black', lw=0.5)

# Add labels
ax.set_xlabel("Hidden Neuron Index")
ax.set_ylabel("Input Neuron Index")
ax.set_title("First Layer Weights (Input → Hidden)")

plt.tight_layout()
plt.show()
