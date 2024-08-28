import os
import csv

# Define the root directory
root_dir = './'

# List to hold the next_observations data
next_observations = []

# Traverse the directory structure
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'states.csv':
            file_path = os.path.join(subdir, file)
            with open(file_path, mode='r') as csvfile:
                csvreader = csv.reader(csvfile)
                rows = list(csvreader)
                
                # Skip if the file is empty or has only one row
                if len(rows) < 2:
                    continue
                
                # Define the number of columns in combined_state
                num_gs = 8
                num_bs = 12
                num_ws = 6
                total_columns = num_gs + num_bs + num_ws
                
                # Function to compute combined_gs, combined_bs, combined_ws
                def compute_combined(row):
                    # Extract gs values
                    gs_degree_sum = float(row[1])
                    gs_degree_variance = float(row[2])
                    gs_neighbor_distance_sum = float(row[3])
                    gs_neighbor_distance_variance = float(row[4])
                    gs_ntotal = float(row[0])
                    gs_steps_expansion_average = float(row[7])
                    gs_steps_taken_avg = float(row[6])
                    gs_steps_taken_max = float(row[5])

                    # Extract bs values
                    bs_ntotal = float(row[8])
                    bs_old_ntotal = float(row[9])
                    bs_degree_sum_new = float(row[10])
                    bs_degree_variance_new = float(row[11])
                    bs_degree_variance_old = float(row[12])
                    bs_degree_sum_old = float(row[13])
                    bs_neighbor_distance_sum_new = float(row[14])
                    bs_neighbor_distance_variance_new = float(row[15])
                    bs_neighbor_distance_sum_old = float(row[16])
                    bs_neighbor_distance_variance_old = float(row[17])
                    bs_steps_taken_sum = float(row[18])
                    bs_steps_taken_max = float(row[19])
                    bs_steps_expansion_sum = float(row[20])

                    # Extract ws values
                    ws_get_count_0 = float(row[21])
                    ws_get_count_1 = float(row[22])
                    ws_get_count_2 = float(row[23])

                    # Compute combined_gs
                    combined_gs = [
                        gs_degree_sum / (bs_ntotal + bs_old_ntotal),
                        gs_degree_variance,
                        gs_neighbor_distance_sum / (bs_ntotal + bs_old_ntotal),
                        gs_neighbor_distance_variance,
                        gs_ntotal,
                        gs_steps_expansion_average,
                        gs_steps_taken_avg,
                        gs_steps_taken_max
                    ]

                    # Compute combined_bs
                    combined_bs = [
                        bs_degree_sum_new / bs_ntotal,
                        bs_degree_sum_old / bs_old_ntotal if bs_old_ntotal != 0 else 0,
                        bs_degree_variance_new,
                        bs_degree_variance_old,
                        bs_neighbor_distance_sum_new / bs_ntotal,
                        bs_neighbor_distance_sum_old / bs_old_ntotal if bs_old_ntotal != 0 else 0,
                        bs_neighbor_distance_variance_new,
                        bs_neighbor_distance_variance_old,
                        bs_ntotal,
                        bs_steps_expansion_sum / bs_ntotal,
                        bs_steps_taken_max,
                        bs_steps_taken_sum / bs_ntotal
                    ]

                    # Compute combined_ws
                    combined_ws = [15, 100, 50, ws_get_count_0, ws_get_count_1, ws_get_count_2]

                    # Combine all states
                    combined_state = combined_gs + combined_bs + combined_ws

                    return combined_state
                
                # Process each row
                for i, row in enumerate(rows):
                    if i == len(rows) - 1:
                        # Last row should be all zeros
                        last_row_length = len(compute_combined(rows[0]))  # Compute the length based on the first row
                        next_observations.append(['0'] * last_row_length)
                    else:
                        # Use the second row for the next_observations
                        next_observations.append(compute_combined(row))

# Write next_observations to next_observations.csv
with open('next_observations.csv', mode='w', newline='') as outfile:
    csvwriter = csv.writer(outfile)
    # Write the data rows
    csvwriter.writerows(next_observations)

