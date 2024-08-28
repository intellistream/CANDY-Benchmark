import os
import csv

# Define the root directory
root_dir = './'

# List to hold the last column (constraints)
constraints = []

# Traverse the directory structure
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'states.csv':
            file_path = os.path.join(subdir, file)
            with open(file_path, mode='r') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    # Extract the last column
                    constraint_value = row[-1]  # -1 for the last column
                    constraints.append([constraint_value])  # Append as a list for CSV writing

# Write constraints to constraints.csv
with open('constraints.csv', mode='w', newline='') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(['Constraint'])  # Optional header row
    csvwriter.writerows(constraints)  # Write the constraint values row by row

