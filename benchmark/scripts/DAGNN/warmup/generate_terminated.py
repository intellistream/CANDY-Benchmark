

import os
import csv

# Define the root directory
root_dir = './'

# List to hold the 25th column (actions)
terminated = []

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
                #last_row = row
                # Extract the 25th column (index 24 in zero-based indexing)
                #action_value = row[24]  # Convert to the appropriate type if necessary
                terminated.append(['0'])  # Append as a list for CSV writing
            terminated.pop()
            terminated.append(['1'])


# Write the actions to actions.csv
with open('terminated.csv', mode='w', newline='') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerows(terminated)  # Write the action values row by row