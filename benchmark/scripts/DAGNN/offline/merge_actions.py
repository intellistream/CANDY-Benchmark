import os
import csv

# Define the root directory
root_dir = './'

# List to hold the 25th column (actions)
actions = []

# Traverse the directory structure
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'states.csv':
            file_path = os.path.join(subdir, file)
            with open(file_path, mode='r') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    # Extract the 25th column (index 24 in zero-based indexing)
                    action_value = row[24]  # Assuming the values are already in string format
                    actions.append([action_value])  # Append as a list to keep the format for CSV writing

# Write actions to actions.csv
with open('actions.csv', mode='w', newline='') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerows(actions)  # Write the action values row by row

