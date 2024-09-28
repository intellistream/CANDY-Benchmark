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
    for i in range(0, total_texts, batch_size):
        for text in texts[i:min(i + batch_size, total_texts)]:
            text_tensors = []
            # Tokenize and encode the text, truncating and padding to a max length of 512
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            text_tensors.append(inputs)
            # Concatenate all input tensors into a single batch and move to the GPU (if available)
        input_ids = torch.cat([t['input_ids'] for t in text_tensors], dim=0).to(device)
        attention_mask = torch.cat([t['attention_mask'] for t in text_tensors], dim=0).to(device)
            
            # Encode the texts using Roberta model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Use the mean of the last hidden state to get sentence embeddings
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
        print(f"Processed batch {i // batch_size + 1}, flushed to {output_file}")

    print(f"Finished encoding {total_texts} comments and saved to {output_file}")
    # Concatenate all input tensors into a single batch and move to the GPU (if available)
    input_ids = torch.cat([t['input_ids'] for t in text_tensors], dim=0).to(device)
    attention_mask = torch.cat([t['attention_mask'] for t in text_tensors], dim=0).to(device)
    
    # Encode the texts using Roberta model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the mean of the last hidden state to get sentence embeddings
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
    append_to_fvecs(output_file, sentence_embeddings)

def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    targetPathBase = exeSpace + 'datasets/reddit'
    os.system('mkdir ' + targetPathBase)
    os.system('rm -rf ' + targetPathBase+"/*.fvecs")
    # Step 1: Load the test split of the Reddit dataset
    dataset = load_dataset("reddit", split="train[:50%]")  # Load the test split

    # Step 2: Get the list of comments from the test split
    comments = dataset['body']  # Extract the 'body' field which contains the comments
    encode_texts(comments,200000,128,targetPathBase+'/data_reddit.fvecs')

    dataset = load_dataset("reddit", split="train[51%:]")  # Load the test split

    # Step 2: Get the list of comments from the test split
    comments = dataset['body']  # Extract the 'body' field which contains the comments
    encode_texts(comments,2000,128,targetPathBase+'/query_reddit.fvecs')
    
if __name__ == "__main__":
    main()