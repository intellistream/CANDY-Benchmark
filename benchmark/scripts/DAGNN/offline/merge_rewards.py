import os
import csv

# Define the root directory
root_dir = './'

# List to hold the 27th column (rewards)
rewards = []

# Traverse the directory structure
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'states.csv':
            file_path = os.path.join(subdir, file)
            with open(file_path, mode='r') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    # Extract the 27th column (index 26 in zero-based indexing)
                    reward_value = row[26]  # Assuming the values are already in string format
                    rewards.append([reward_value])  # Append as a list to keep the format for CSV writing

# Write rewards to rewards.csv
with open('rewards.csv', mode='w', newline='') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerows(rewards)  # Write the reward values row by row

