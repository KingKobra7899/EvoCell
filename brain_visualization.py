import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Load JSON
with open("brain.json", "r") as f:
    brain = json.load(f)

brain_size = brain["brain_size"]
weights_1d = np.array(brain["encoder_weights"])

# In Rust: DMatrix::zeros(output_size, input_size) means (rows=output, cols=input)
# So we reshape to (output_size, input_size)
output_size = 11
input_size = len(weights_1d) // output_size
weights = weights_1d.reshape(output_size, input_size)

# Seaborn heatmap expects rows as y, cols as x, so we transpose:
weights_T = weights.T  # shape: (input_size, output_size)

pca = PCA(n_components=2)
weights_2d = pca.fit_transform(weights_T)  # shape: (input_size, 2)

# Color groups: triplet index for first 60, else gray
colors = []
for i in range(input_size):
    if i < 60:
        colors.append(i // 3)  # triplet group index
    else:
        colors.append(-1)  # rest

# Scatter points
plt.figure(figsize=(8, 6))
scatter = plt.scatter(weights_2d[:, 0], weights_2d[:, 1],
                      c=colors, cmap='tab20', s=50, edgecolor='k', zorder=2)

# Connect triplet members
for triplet_idx in range(20):  # first 60 inputs → 20 triplets
    indices = [triplet_idx * 3 + j for j in range(3)]
    triplet_points = weights_2d[indices]
    plt.plot(triplet_points[:, 0], triplet_points[:, 1],
             color='black', lw=0.5, alpha=0.5, zorder=1)

plt.title("PCA of Input Weight Vectors (Triplets Connected)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)

# Legend for triplets
handles, labels = scatter.legend_elements(num=20)
plt.legend(handles, [f"Triplet {i}" for i in range(20)] + ["Other"],
           loc="best", fontsize=8, ncol=2)

plt.tight_layout()
plt.show()