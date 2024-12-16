import torch
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
import numpy as np
import random
import os
def append_to_fvecs(file_path, vectors):
        """ Appends the vectors to an .fvecs file. """
        with open(file_path, 'ab') as f:
            for vec in vectors:
                dim = np.array([vec.shape[0]], dtype=np.int32)  # First write the dimension
                vec = vec.cpu().numpy().astype(np.float32)       # Convert to numpy float32
                dim.tofile(f)                                    # Write dimension
                vec.tofile(f)       
def save_to_fvecs(file_path, vectors):
        """ Appends the vectors to an .fvecs file. """
        with open(file_path, 'wb') as f:
            for vec in vectors:
                dim = np.array([vec.shape[0]], dtype=np.int32)  # First write the dimension
                vec = vec.cpu().numpy().astype(np.float32)       # Convert to numpy float32
                dim.tofile(f)                                    # Write dimension
                vec.tofile(f)       
# Step 3: Define a function to generate embeddings for a batch of texts
def generate_embeddings(batch_texts, model, tokenizer, device,max_length=128):
    """Generate embeddings for a batch of texts."""
    # Tokenize the batch of texts
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move to device

    # Generate embeddings (take the last hidden state or pooler output)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the pooled output as sentence embedding (or use other strategies like mean pooling)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu()  # Move back to CPU
    return embeddings
def encode_texts(texts, N,batch_size, output_file):
    # Specify the GPU by index (e.g., use GPU 0)
    total_texts = min(N, len(texts))  # Ensure we don't exceed available captions
    gpu_index = 0  # Change this to the desired GPU index
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained Roberta model and move it to the correct device
    model = RobertaModel.from_pretrained('roberta-base').to(device)
    model.eval()

    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Load the pre-trained Roberta model and move it to the correct device
    model = RobertaModel.from_pretrained('roberta-base').to(device)
    model.eval()
    
    
    for i in range(0, total_texts, batch_size):
            # Extract micro-batch from dataset
        with torch.no_grad():  # Ensure no gradients are calculated
            endPos  =min(i + batch_size,total_texts-1)
            batch = texts[i:endPos]
            # Extract the text field from the batch
            # Tokenize the input texts
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            
            # Move tensors to GPU if available
            if torch.cuda.is_available():
                inputs = {key: val.cuda() for key, val in inputs.items()}
                model.cuda()

            # Forward pass through the model to get embeddings
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1) # Mean pooling of token embeddings
            # Append the embeddings for this batch
        append_to_fvecs(output_file,embeddings)   
            #append_to_fvecs(output_file, embeddings)
        print(f"Processed batch {i // batch_size + 1}, flushed to {output_file},size {embeddings.shape}")
   
    #print(f"Processed batch {i // batch_size + 1}, flushed to {output_file},size {embeddings.shape}")

def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    targetPathBase = exeSpace + 'datasets/reddit'
    os.system('mkdir ' + targetPathBase)
    # Step 1: Load the test split of the Reddit dataset
   

    # Step 2: Get the list of comments from the test split
    if(os.path.exists(targetPathBase+'/data_reddit.fvecs')):
            print('skip '+targetPathBase+'/data_reddit.fvecs')
    else: 
        dataset = load_dataset("reddit", split="train[:200001]")  # Load the test split
        comments = dataset['body']  # Extract the 'body' field which contains the comments
        encode_texts(comments,100000+1,128,targetPathBase+'/data_reddit.fvecs')
    if(os.path.exists(targetPathBase+'/query_reddit.fvecs')):
            print('skip '+targetPathBase+'/query_reddit.fvecs')
    else: 
        dataset = load_dataset("reddit", split="train[250000:]")  # Load the test split
        comments = dataset['body']  # Extract the 'body' field which contains the comments
        encode_texts(comments,2000+1,128,targetPathBase+'/query_reddit.fvecs')
    
if __name__ == "__main__":
    main()