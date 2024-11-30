import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class DisasterImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, neutral_label_handling="ignore"):
        self.annotations = pd.read_csv(annotations_file,sep=";")
        self.img_dir = img_dir
        self.neutral_label_handling = neutral_label_handling

        #print(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image
        img_name = self.annotations.iloc[idx, 0]  # Assuming first column is the image filename
        img_path = f"{self.img_dir}/{img_name}.jpg"

        
        
        # Extract and process Q1 sentiment for binary classification
        q1_sentiment = self.annotations.filter(regex="^A\d+\.Q1").iloc[idx].astype(float).mean()
        
        # Convert Q1 to binary sentiment label: 0 (negative) for 1-4, 1 (positive) for 6-9
        if q1_sentiment <= 5:
            q1_label = torch.tensor(0, dtype=torch.long)  # Negative sentiment
        elif q1_sentiment > 5:
            q1_label = torch.tensor(1, dtype=torch.long)  # Positive sentiment
        else:
            print(q1_sentiment)
            # If neutral handling is 'ignore', this case should already be filtered out
            raise ValueError("Unexpected neutral Q1 sentiment rating when using binary classification.")
        
        # Extract Q2 for additional arousal label if needed
        #q2_arousal = self.annotations.filter(regex="^A\d+\.Q2").iloc[idx].astype(float).mean()
        #q2_label = torch.tensor(q2_arousal, dtype=torch.float32)
        
        return img_path, q1_label
