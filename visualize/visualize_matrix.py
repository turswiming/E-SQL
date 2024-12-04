import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data from the table
data = {
    "query language": ["English", "Chinese", "Cantonese", "Japanese"],
    "English": [50.0, 52.4, 48.8, 50.0],
    "Chinese": [58.8, 46.4, 51.2, 47.6],
    "Cantonese": [51.2, 47.6, 44.1, 46.4],
    "Japanese": [50.0, 48.8, 52.4, 57.1]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)
df.set_index("query language", inplace=True)

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True, linewidths=.5)

# Add labels and title
plt.title("Heatmap of Query Language vs Schema Language")
plt.xlabel("Schema Language")
plt.ylabel("Query Language")

# Show the plot
plt.show()