import os
import csv

# Define the root directory
root_dir = './'

# List to hold the calculated rewards
rewards = []

# Traverse the directory structure
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'states.csv':
            file_path = os.path.join(subdir, file)
            with open(file_path, mode='r') as csvfile:
                csvreader = csv.reader(csvfile)

                previous_reward = 0  # Initialize the first reward as 0

                for row_idx, row in enumerate(csvreader):
                    current_reward = float(row[26])  # Convert reward value to float
                    if row_idx == 0:
                        # For the first row, reward is 0
                        rewards.append([0])
                    else:
                        # Compute the difference between the current and previous reward
                        reward_difference = current_reward - previous_reward
                        rewards.append([reward_difference])

                    # Update the previous_reward for the next iteration
                    previous_reward = current_reward

# Write the calculated rewards to rewards.csv
with open('rewards.csv', mode='w', newline='') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerows(rewards)  # Write the reward values row by row


