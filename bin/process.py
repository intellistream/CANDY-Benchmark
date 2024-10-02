import sys
import os
import shutil
import torch
import multiprocessing
import logging

# wait for user to input the data path
DATA_PATH = input("Enter the path to the data root (e.g. ~/data/Glove): ")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")

# set the logging level, format and log file
logfile = f"process_{DATA_PATH.split('/')[-1]}.log"
os.remove(logfile) if os.path.exists(logfile) else None
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename=logfile)

# wait for user to input the number of time steps
TIME_STEPS = int(input("Enter the number of time steps (e.g. 10): "))

# wait for user to input the ratio of insert operations and delete operations
INSERT_RATIO = float(input("Enter the ratio of insert operations (e.g. 0.05) : "))
while INSERT_RATIO > 1 or INSERT_RATIO < 0:
    print("Error: Insert ratio must be between 0 and 1.")
    INSERT_RATIO = float(input("Enter the ratio of insert operations (e.g. 0.05) : "))
DELETE_RATIO = float(input("Enter the ratio of delete operations (e.g. 0.05) : "))
while DELETE_RATIO > 1 or DELETE_RATIO < 0:
    print("Error: Delete ratio must be between 0 and 1.")
    DELETE_RATIO = float(input("Enter the ratio of delete operations (e.g. 0.05) : "))

# wait for user to input the number to query
QUERY_NUM = int(input("Enter the number of queries (e.g. 100): "))
while QUERY_NUM < 0:
    print("Error: Query number must be greater than 0.")
    QUERY_NUM = int(input("Enter the number of queries (e.g. 100) : "))

# check if the raw data path exists
if not os.path.exists(RAW_DATA_PATH):
    print(f"Error: {RAW_DATA_PATH} does not exist.")
    sys.exit(1)

# create the processed data directory and clear the files and directories
if os.path.exists(PROCESSED_DATA_PATH):
    try:
        shutil.rmtree(PROCESSED_DATA_PATH)
    except OSError as e:
        print(f"Error: {PROCESSED_DATA_PATH} could not be cleared.")
        sys.exit(1)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def get_querys_top_k(query_tensors : list[torch.Tensor], tensors: list[torch.Tensor], k_nums : list[int]) -> list:
    # querys_top : list[list[torch.Tensor]] in length of k_nums
    querys_top = []
    for _ in k_nums:
        querys_top.append([])

    # get the top k tensors for each query
    for query in query_tensors:
        query = query.view(1, -1)
        distances = torch.cdist(query, torch.stack(tensors, dim=0), p=2)
        for idx, k in enumerate(k_nums):
            top_k_indexes = torch.topk(distances, k, largest=False).indices.squeeze().tolist()
            querys_top[idx].append([tensors[index] for index in top_k_indexes])
    return querys_top

# create the time steps
def create_time_steps(tensors : list, processed_data_path : str) -> None:
    for i in range(TIME_STEPS):
        logging.info(f"Creating time step {i + 1} in {processed_data_path} ...")
        # create a step directory
        step_dir = os.path.join(processed_data_path, f"step_{i + 1}")
        os.makedirs(step_dir, exist_ok=True)
        
        # insert and delete operations
        delete_count = int(len(tensors) * DELETE_RATIO)
        insert_count = int(len(tensors) * INSERT_RATIO)

        # delete operations
        delete_tensors = []
        for _ in range(delete_count):
            index = torch.randint(0, len(tensors), (1,)).item()
            delete_tensors.append(tensors.pop(index))

        # insert operations
        insert_tensors = []
        for _ in range(insert_count):
            # randomly generate a tensor
            new_tensor = torch.randn(len(tensors[0]))
            insert_tensors.append(new_tensor)
            tensors.insert(torch.randint(0, len(tensors), (1,)).item(), new_tensor)

        # save the delete and insert operations
        torch.save(torch.cat(delete_tensors, dim=0), os.path.join(step_dir, "delete_tensor.pt"))
        torch.save(torch.cat(insert_tensors, dim=0), os.path.join(step_dir, "insert_tensor.pt"))

        # query operations
        query_tensors = []
        for _ in range(QUERY_NUM):
            # randomly generate a tensor
            query_tensors.append(torch.randn(len(tensors[0])))
        
        # save the query operations
        torch.save(torch.cat(query_tensors, dim=0), os.path.join(step_dir, "query_tensor.pt"))

        # save the top k tensors for each query
        k_nums = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
        querys_top = get_querys_top_k(query_tensors, tensors, k_nums)
        for idx, k in enumerate(k_nums):
            querys_top_k = querys_top[idx]
            torch.save(querys_top_k, os.path.join(step_dir, f"ground_truth_query_top_{k}.pt"))

# process the data
def process_glove_file(filename):
    logging.info(f"Processing {filename} ...")

    output_root = os.path.join(PROCESSED_DATA_PATH, filename.replace(".txt", ""))
    os.makedirs(output_root, exist_ok=True)

    with open(os.path.join(RAW_DATA_PATH, filename), "r") as f:
        lines = f.readlines()

    tensors = []
    for line in lines:
        if not line.strip():
            continue

        parts = line.strip().split()  # split the line into parts
        if len(parts) < 2:  # ensure that there is at least one word and one vector
            logging.debug(f"Invalid line in file {filename} at line {line}")
            continue

        vector = parts[1:]  # get the vector part of the line

        try:
            # convert the vector to a list of floats
            vector = list(map(float, vector))
            tensors.append(torch.tensor(vector, dtype=torch.float32))
        except ValueError as e:
            logging.debug(f"{e} in file {filename} at line {line}")
            continue

    # check tensors dimension and pad if necessary
    max_dim = max([len(tensor) for tensor in tensors])
    for i in range(len(tensors)):
        if len(tensors[i]) < max_dim:
            tensors[i] = torch.cat((tensors[i], torch.zeros(max_dim - len(tensors[i]))))
            logging.info(f"Padded tensor {i} in file {filename}")

    # save the initial data
    output_path = os.path.join(output_root, "initialData.pt")
    torch.save(tensors, output_path)
    logging.info(f"Saved initial data to {output_path}")

    # create the time steps
    create_time_steps(tensors, output_root)

if DATA_PATH.find("Glove") != -1:
    filenames = os.listdir(RAW_DATA_PATH)
    poolSize = 10 if len(filenames) > 10 else len(filenames)
    with multiprocessing.Pool(poolSize) as pool:
        pool.map(process_glove_file, filenames)

elif DATA_PATH.find("Audio") != -1:
    # TODO: process Audio data
    pass