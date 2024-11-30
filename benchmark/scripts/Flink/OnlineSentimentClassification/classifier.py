import torch
import torch.nn as nn
from torchmultimodal.models.flava.model import flava_model
from transformers import BertTokenizer
from torchmultimodal.transforms.flava_transform import FLAVAImageTransform
from PIL import Image
from torchmultimodal.models.flava.model import flava_model
from torchmultimodal.transforms.flava_transform import FLAVAImageTransform
from transformers import BertTokenizer
import os
import numpy as np
import random
from PIL import ImageFile
import torch.nn.functional as F
# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
class MultimodalSentimentClassifier(nn.Module):
    def __init__(self, device='cuda:0'):
        super(MultimodalSentimentClassifier, self).__init__()
        # Load pretrained FLAVA model
        self.flava_model = flava_model(pretrained=True).to(device)
        self.flava_model.eval()  # Set to evaluation mode
          # Freeze all parameters in the FLAVA model
        for param in self.flava_model.parameters():
            param.requires_grad = False
        # Initialize image transform and tokenizer
        self.image_transform = FLAVAImageTransform(is_train=False)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Define the classification decoder (MLP)
        self.decoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 768),  
            nn.ReLU(),
        )
        self.logits = nn.Sequential(
            nn.Linear(768, 2),# Assuming 2 sentiment classes (negative, neutral, positive)
        )
        self.device = device

    def convert_text_to_tensor(self, text):
        """Convert raw text to a 768D tensor using FLAVA text encoder."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        text_tensor = inputs['input_ids'].to(self.device)
        with torch.no_grad():
            _, text_embedding = self.flava_model.encode_text(text_tensor, projection=True)
        return text_embedding.squeeze(0)

    def convert_image_to_tensor(self, image_path):
        """Convert image file to a 768D tensor using FLAVA image encoder."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, image_embedding = self.flava_model.encode_image(image_tensor, projection=True)
        return image_embedding.squeeze(0)

    def forward(self, embedding_tensor):
        """Forward pass that takes a 768D embedding tensor."""
        x = self.decoder(embedding_tensor)
        output = self.logits((x+embedding_tensor)/2)
        return (output)
    def decoderOutput(self,embedding_tensor):
        """Forward pass that takes a 768D embedding tensor."""
        x = (self.decoder(embedding_tensor)+embedding_tensor)/2
        embeddings = torch.flatten(x, 1)  # This is the embedding (512-dimensional vector)
           # Normalize the embeddings to unit length (L2 normalization)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize along the feature dimension
    
        return normalized_embeddings  # Return L2-normalized embedding
def main():
    # Example usage
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = MultimodalSentimentClassifier(device=device)
    image_path = 'path_to_image.jpg'
    text = "This is a sample caption for sentiment analysis."
    output = model(image_path, text)
    print("Sentiment output:", output)