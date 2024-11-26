import os
import csv

# Define the root directory
root_dir = './'

# List to hold the last column (constraints)
constraints = []

# Iterate through directories named from "0" to "60"
for i in range(61):
    dir_name = str(i)  # Directory names are "0", "1", ..., "60"
    file_path = os.path.join(root_dir, dir_name, 'states.csv')

    # Check if the file exists before processing
    if os.path.exists(file_path):
        with open(file_path, mode='r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            # Read each row in the current states.csv file
            for row in csvreader:
                # Extract the last column (no matter how many columns there are)
                constraint_value = row[-1]  # -1 refers to the last column
                constraints.append([constraint_value])  # Append as a list for CSV writing

# Write the constraints to constraints.csv
with open('constraints.csv', mode='w', newline='') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerows(constraints)  # Write the constraints row by row

