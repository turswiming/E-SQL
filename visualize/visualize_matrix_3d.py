import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Data from the table
query_languages = ["English", "Chinese", "Cantonese", "Japanese"]
schema_languages = ["English", "Chinese", "Cantonese", "Japanese"]
data = np.array([
    [50.0, 58.8, 51.2, 50.0],
    [52.4, 46.4, 47.6, 48.8],
    [48.8, 51.2, 44.1, 52.4],
    [50.0, 47.6, 46.4, 57.1]
])

# Create meshgrid for the 3D plot
X, Y = np.meshgrid(np.arange(len(schema_languages)), np.arange(len(query_languages)))
Z = data

# Create the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

# Customize the axes
ax.set_xticks(np.arange(len(schema_languages)))
ax.set_xticklabels(schema_languages)
ax.set_yticks(np.arange(len(query_languages)))
ax.set_yticklabels(query_languages)
ax.set_zlabel('Score')

ax.set_zlim(20, np.max(Z) + 10)
# Add labels and title
ax.set_title("3D Surface Plot of Query Language vs Schema Language")
ax.set_xlabel("Schema Language")
ax.set_ylabel("Query Language")

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Show the plot
plt.show()