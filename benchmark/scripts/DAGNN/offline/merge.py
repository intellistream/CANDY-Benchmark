import os
import csv

# Define the root directory
root_dir = './'

# List to hold all combined_state entries
combined_states = []

# Traverse the directory structure
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'states.csv':
            file_path = os.path.join(subdir, file)
            with open(file_path, mode='r') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
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

                    # Add the combined state to the list
                    combined_states.append(combined_state)

# Write combined states to observations.csv
with open('observations.csv', mode='w', newline='') as outfile:
    csvwriter = csv.writer(outfile)
    # Assuming you want to write headers, otherwise comment this out
    headers = [
        'gs.degree_sum_normalized', 'gs.degree_variance', 'gs.neighbor_distance_sum_normalized', 'gs.neighbor_distance_variance',
        'gs.ntotal', 'gs.steps_expansion_average', 'gs.steps_taken_avg', 'gs.steps_taken_max',
        'bs.degree_sum_new_normalized', 'bs.degree_sum_old_normalized', 'bs.degree_variance_new', 'bs.degree_variance_old',
        'bs.neighbor_distance_sum_new_normalized', 'bs.neighbor_distance_sum_old_normalized', 'bs.neighbor_distance_variance_new',
        'bs.neighbor_distance_variance_old', 'bs.ntotal', 'bs.steps_expansion_sum_normalized', 'bs.steps_taken_max',
        'bs.steps_taken_sum_normalized', 'fixed_1', 'fixed_2', 'fixed_3', 'ws.get_count(0)', 'ws.get_count(1)', 'ws.get_count(2)'
    ]
    csvwriter.writerow(headers)  # Write the headers
    csvwriter.writerows(combined_states)  # Write the data rows

