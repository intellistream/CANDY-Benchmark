from datasets import load_dataset
from torch.utils.data import random_split
import torch
from torch.utils.data import DataLoader
from classifier import MultimodalSentimentClassifier
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm  # For progress bar
from disasterImage import DisasterImageDataset
import PyCANDY as candy

import os,time,random
# Prepare DataLoader
def collate_fn(batch):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    return texts, torch.tensor(labels)

# Training Loop with Progress Bar
def trainText(model, train_loader, optimizer, criterion,device):
    model.to(device)
    model.train()  # Set model to training mode
    running_loss = 0.0
    #correct_predictions = 0
    total_samples = 0

        # Progress bar for monitoring
    for texts, labels in tqdm(train_loader,desc="Training text"):
            labels = labels.to(device)

            # Convert images to embeddings using the model's convert_image_to_tensor function
            txt_embeddings = [model.convert_text_to_tensor(text) for text in texts]
            txt_embeddings = torch.stack(txt_embeddings).to(device)  # Stack to create a batch

            # Forward pass through the classifier
            optimizer.zero_grad()
            outputs = model(txt_embeddings)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item() * labels.size(0)
           # _, predicted = torch.max(outputs, 1)
           # correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    return epoch_loss
def trainImg(model, train_loader, optimizer, criterion, device='cuda:0'):
    model.to(device)
    model.train()  # Set model to training mode
    running_loss = 0.0
    #correct_predictions = 0
    total_samples = 0

        # Progress bar for monitoring
    for img_paths, labels in tqdm(train_loader,desc="Training img"):
            labels = labels.to(device)

            # Convert images to embeddings using the model's convert_image_to_tensor function
            image_embeddings = [model.convert_image_to_tensor(img_path) for img_path in img_paths]
            image_embeddings = torch.stack(image_embeddings).to(device)  # Stack to create a batch

            # Forward pass through the classifier
            optimizer.zero_grad()
            outputs = model(image_embeddings)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item() * labels.size(0)
           # _, predicted = torch.max(outputs, 1)
           # correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    return epoch_loss

# Evaluation Loop with Progress Bar
def evaluateImg(model, test_loader,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in tqdm(test_loader, desc="Evaluating img", leave=False):
            labels = labels.to(device)
            
            # Convert texts to tensors
            text_embeddings = [model.convert_image_to_tensor(text) for text in texts]
            text_embeddings = torch.stack(text_embeddings)
            
            outputs = model(text_embeddings)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

# Saving the model's state_dict after tr
# Saving the model's state_dict after training
def save_model(model, file_path="model.pth"):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
# Loading the model's state_dict
def set_seed(seed):
    torch.manual_seed(seed)               # Set seed for PyTorch
    torch.cuda.manual_seed(seed)          # If you're using a GPU
    torch.cuda.manual_seed_all(seed)      # For multi-GPU setups
def load_model(model, file_path="model.pth"):
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Set the model to evaluation mode after loading
    print(f"Model loaded from {file_path}")
    return model
def main():
    print('load image data')
    set_seed(114514)
    # Initialize dataset and dataloader
    # Initialize dataset and dataloader
    img_dir = "sentiment/sentiment-dataset/images"  # Specify the directory containing images
    annotations_file = "sentiment/sentiment-dataset/annotations.csv"
    full_dataset = DisasterImageDataset(annotations_file=annotations_file, img_dir=img_dir, neutral_label_handling="ignore")

    # Split the dataset into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


    # Initialize DataLoaders
    train_loader_img = DataLoader(train_dataset, batch_size=8, shuffle=True)
    print('loading model')
    # Initialize model, optimizer, and loss function
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = MultimodalSentimentClassifier(device=device).to(device)
    optimizer = Adam(model.parameters(),lr=1e-4)  # Only train classifier
    criterion = nn.CrossEntropyLoss()
    if os.path.exists('model.pth'):
        print('load model')
        model = load_model(model,'model.pth')
        set_seed(114514)
    else:
        print("run trainning")
        # Run Training and Evaluation
        epochs = 5
        set_seed(114514)
        for epoch in range(epochs):
            #train_loss = trainText(model, train_loader_txt, optimizer, criterion,device)
            #accuracy = evaluate(model, test_loader)
            #print(f"Epoch {epoch+1}/{epochs}, Loss of text: {train_loss:.4f}")
            train_loss = trainImg(model, train_loader_img, optimizer, criterion)
        #accuracy = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss of image: {train_loss:.4f}")
        print('trainning is done')
        save_model(model,'model.pth')
        set_seed(114514)
if __name__ == "__main__":
    main()

