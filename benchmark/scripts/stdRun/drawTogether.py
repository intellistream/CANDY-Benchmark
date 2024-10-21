import numpy as np
import torch
import os
import struct
def load_fvecs(filename):
    """Read .fvecs file and return a list of vectors."""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            # Read dimension of the vector (first 4 bytes, int32)
            data = f.read(4)
            if not data:
                break
            dim = struct.unpack('i', data)[0]
            # Read the floats (dim * 4 bytes)
            vector = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vector)
    return np.array(vectors)
def l2_normalize(a):
   
    
    # Compute the L2 norm along the dimension 0
    norm = torch.norm(a, p=2, dim=0, keepdim=True)
    
    # Return the tensor divided by its L2 norm
    return a / norm
def compute_standard_deviation(fvecs_data):
    """
    Compute the standard deviation of the vectors loaded from the .fvecs file using PyTorch.
    """
    # Convert numpy array to PyTorch tensor
    num_elements = fvecs_data.size
    num_rows = num_elements // 768
    tensor_data = torch.tensor(fvecs_data[:num_rows * 768].reshape(num_rows, 768))  # Reshape to have `target_cols` columns)
    print(tensor_data.shape)
    # Compute standard deviation
    std_dev = torch.std((tensor_data))
    
    return std_dev
def getStd(fname):
    fvecs_data = load_fvecs(fname)
    std_dev = compute_standard_deviation(fvecs_data)
    return std_dev

dataset_dataPath_mapping = {
    'DPR': 'datasets/DPR/DPR100KC4.fvecs',
    'COCO-.05': 'results/scanMultiModalPropotion/multiModalProp/data_0.05.fvecs',
    'COCO-.8': 'results/scanMultiModalPropotion/multiModalProp/data_0.8.fvecs',
    'COCO-I': 'datasets/coco/data_image.fvecs',
    'COCO-C': 'datasets/coco/data_captions.fvecs',
  
}
def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    stdRu = {}
    for dataset_name, file_path in dataset_dataPath_mapping.items(): 
        stdi = getStd(exeSpace+file_path)
        stdRu[dataset_name] = stdi.item()
    print(stdRu)

if __name__ == "__main__":
    main()