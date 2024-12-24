import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('pose_angles.csv')

# Separate the header and the data
header = df.columns
data = df.values

# Shuffle the rows of the data
import numpy as np
np.random.shuffle(data)

# Create a new DataFrame with the shuffled data and the original header
shuffled_df = pd.DataFrame(data, columns=header)

# Save the shuffled DataFrame back to a new CSV file
shuffled_df.to_csv('shuffled_file.csv', index=False)
