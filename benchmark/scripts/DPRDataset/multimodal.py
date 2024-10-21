import torch
from datasets import load_dataset
from PIL import Image
from torchmultimodal.models.flava.model import flava_model
from torchmultimodal.transforms.flava_transform import FLAVAImageTransform
from transformers import BertTokenizer
import os
import numpy as np
import random
exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
targetPathBase = exeSpace + 'datasets/coco'
PATH_TO_IMAGE_FOLDER = targetPathBase
def create_full_path(example):
    """Create full path to image using `base_path` to COCO2017 folder."""
    example["image_path"] = os.path.join(PATH_TO_IMAGE_FOLDER, example["file_name"])
    return example
def getCoco(cocoPath):
    tryDownload = 0
    if os.path.exists(cocoPath + "/" +"coco1.txt" ):
        print('download of coco is done')
    else:
        os.system("cd "+cocoPath+ '&& wget http://images.cocodataset.org/zips/train2017.zip')
        os.system("cd "+cocoPath+'&& wget http://images.cocodataset.org/zips/val2017.zip')
        os.system("cd "+cocoPath+'&& touch coco1.txt')
    if os.path.exists(cocoPath + "/" +"coco2.txt" ):
        print('unzip of coco is done')
    else:
        os.system("cd "+cocoPath+'&& unzip train2017.zip && unzip val2017.zip')
        os.system("cd "+cocoPath+'&& touch coco2.txt')
def encode_images(image_paths, N):
    # Specify the GPU by index (e.g., use GPU 1)
    gpu_index = 0  # Change this to the desired GPU index
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained FLAVA model and move it to the correct device
    model = flava_model(pretrained=True).to(device)
    model.eval()
    # Define the image transform using FLAVA's image preprocessing
    image_transform = FLAVAImageTransform(is_train=False)
    image_tensors = []
    
    # Process the first N images in the list
    for image_path in image_paths[:N]:
        # Open the image file
        image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode
        # Apply the FLAVA image transform
        image_tensor = image_transform(image)["image"].unsqueeze(0)  # Add batch dimension
        image_tensors.append(image_tensor)
    
    # Stack the image tensors into a batch
    image_tensors = torch.cat(image_tensors, dim=0).to(device)  # Move to GPU if available
    
    # Encode the images using FLAVA's image encoder
    with torch.no_grad():
        _, image_embeddings = model.encode_image(image_tensors, projection=True)
    
    return image_embeddings
import numpy as np
def append_to_fvecs(file_path, vectors):
        """ Appends the vectors to an .fvecs file. """
        with open(file_path, 'ab') as f:
            for vec in vectors:
                dim = np.array([vec.shape[0]], dtype=np.int32)  # First write the dimension
                vec = vec.cpu().numpy().astype(np.float32)       # Convert to numpy float32
                dim.tofile(f)                                    # Write dimension
                vec.tofile(f)       
# Micro-batched image encoder with flushing after every batch
def encode_images_to_fvecs(image_paths, N, batch_size, output_file):
    total_images = min(N, len(image_paths))  # Ensure we don't exceed available images
     # Specify the GPU by index (e.g., use GPU 1)
    gpu_index = 0  # Change this to the desired GPU index
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained FLAVA model and move it to the correct device
    model = flava_model(pretrained=True).to(device)
    model.eval()
    # Define the image transform using FLAVA's image preprocessing
    image_transform = FLAVAImageTransform(is_train=False)
    image_tensors = []
    # Process images in micro-batches and flush after every batch
    for i in range(0, total_images, batch_size):
        # Get the paths for the current micro-batch
        batch_paths = image_paths[i:min(i + batch_size, total_images)]
        
        image_tensors = []
        for image_path in batch_paths:
            # Open the image file
            image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode
            # Apply the FLAVA image transform
            image_tensor = image_transform(image)["image"].unsqueeze(0)  # Add batch dimension
            image_tensors.append(image_tensor)
        
        # Stack the image tensors into a batch
        image_tensors = torch.cat(image_tensors, dim=0).to(device)  # Move to GPU if available
        
        # Encode the images using FLAVA's image encoder
        with torch.no_grad():
            _, image_embeddings = model.encode_image(image_tensors, projection=True)
        
        # Append the encoded embeddings to the .fvecs file
        append_to_fvecs(output_file, image_embeddings)

        print(f"Processed batch {i // batch_size + 1}, flushed to {output_file}")

    print(f"Finished encoding {total_images} images and saved to {output_file}")
# Micro-batched caption encoder with flushing after every batch
def encode_captions(captions, N, batch_size, output_file):
    total_captions = min(N, len(captions))  # Ensure we don't exceed available captions
    # Load BERT tokenizer from Hugging Face for text tokenization
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
     # Specify the GPU by index (e.g., use GPU 1)
    gpu_index = 0  # Change this to the desired GPU index
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained FLAVA model and move it to the correct device
    model = flava_model(pretrained=True).to(device)
    model.eval()
    # Process captions in micro-batches and flush after every batch
    for i in range(0, total_captions, batch_size):
        # Get the captions for the current micro-batch
        batch_captions = captions[i:min(i + batch_size, total_captions)]
        
        # Tokenize captions and convert to tensors
        inputs = tokenizer(batch_captions, return_tensors="pt", padding=True, truncation=True, max_length=128)
        text_tensors = inputs['input_ids'].to(device)  # Move to GPU if available
        
        # Encode the texts using FLAVA's text encoder
        with torch.no_grad():
            _, text_embeddings = model.encode_text(text_tensors, projection=True)
        
        # Append the encoded embeddings to the .fvecs file
        append_to_fvecs(output_file, text_embeddings)

        print(f"Processed batch {i // batch_size + 1}, flushed to {output_file}")

    print(f"Finished encoding {total_captions} captions and saved to {output_file}")
    return text_embeddings
# Function to read vectors from an *.fvecs file
def read_fvecs(file_path):
    vectors = []
    with open(file_path, 'rb') as f:
        while True:
            # Read the dimension (first 4 bytes)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # End of file
            dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            
            # Read the vector based on the dimension
            vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
            vectors.append(vec)
    return vectors

# Function to append two *.fvecs files and save the result into a new file
def append_fvecs(file1, file2, output_file):
    # Read vectors from both fvecs files
    vectors1 = read_fvecs(file1)
    vectors2 = read_fvecs(file2)
    
    # Combine the vectors
    combined_vectors = vectors1 + vectors2
    
    # Save the combined vectors to a new .fvecs file
    with open(output_file, 'wb') as f:
        for vec in combined_vectors:
            dim = np.array([vec.shape[0]], dtype=np.int32)  # Write dimension
            dim.tofile(f)
            vec.astype(np.float32).tofile(f)  # Write vector values

    print(f"Appended {len(vectors2)} vectors from {file2} to {file1}, saved to {output_file}")

# Function to shuffle and save combined vectors into a new *.fvecs file
def shuffle_and_save_fvecs(file1, file2, output_file):
    # Read vectors from both fvecs files
    vectors1 = read_fvecs(file1)
    vectors2 = read_fvecs(file2)
    
    # Combine the vectors from both files
    combined_vectors = vectors1 + vectors2
    
    # Shuffle the combined vectors
    random.shuffle(combined_vectors)
    
    # Save the shuffled vectors to a new .fvecs file
    with open(output_file, 'wb') as f:
        for vec in combined_vectors:
            dim = np.array([vec.shape[0]], dtype=np.int32)  # Write dimension
            dim.tofile(f)
            vec.astype(np.float32).tofile(f)  # Write vector values

    print(f"Shuffled {len(combined_vectors)} vectors from {file1} and {file2}, saved to {output_file}")
def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    targetPathBase = exeSpace + 'datasets/coco'
    os.system('mkdir ' + targetPathBase)
    os.system('rm -rf ' + targetPathBase+"/*.fvecs")
    getCoco(targetPathBase)
    
    dataset = load_dataset("phiyodr/coco2017")
    dataset = dataset.map(create_full_path)
    #generate queries 
    captions = dataset['validation']['captions']
    captionFirst =  [caption[0] for caption in captions[:len(captions)]]
    print(captionFirst[1])
    caps = encode_captions(captionFirst,500,128,targetPathBase+'/query_captions.fvecs')
    images = encode_images_to_fvecs(dataset['validation']['image_path'],500,128,targetPathBase+'/query_image.fvecs')
    shuffleQuery = shuffle_and_save_fvecs(targetPathBase+'/query_captions.fvecs',targetPathBase+'/query_image.fvecs',targetPathBase+'/query_shuffle.fvecs')
    appendQuery = append_fvecs(targetPathBase+'/query_captions.fvecs',targetPathBase+'/query_image.fvecs',targetPathBase+'/query_append.fvecs')
    # generate database
    captions = dataset['train']['captions']
    captionFirst =  [caption[0] for caption in captions[:len(captions)]]
    print(captionFirst[1])
    caps = encode_captions(captionFirst,100000,128,targetPathBase+'/data_captions.fvecs')
    images = encode_images_to_fvecs(dataset['train']['image_path'],100000,128,targetPathBase+'/data_image.fvecs')
    shuffleData = shuffle_and_save_fvecs(targetPathBase+'/data_captions.fvecs',targetPathBase+'/data_image.fvecs',targetPathBase+'/data_shuffle.fvecs')
    appendData = append_fvecs(targetPathBase+'/data_captions.fvecs',targetPathBase+'/data_image.fvecs',targetPathBase+'/data_append.fvecs')
if __name__ == "__main__":
    main()

