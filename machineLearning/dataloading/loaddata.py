# code for loading data
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
    """
    Custom dataset for loading images and labels which also applies transforms
    """
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_column = "label"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        label = float(self.dataframe[self.label_column][idx])

        # Move tensors to the GPU
        image = image.to(self.device)
        label = torch.tensor(label, dtype=torch.float32).to(self.device)
        return image, label
    

def prepare_data(csv_file, image_folder, type):
    labels = pd.read_csv(csv_file)
    # set to boole if value 999
    labels["eelgrass"] = labels["eelgrass"].replace(999, 0)
    labels["eelgrass"] = labels["eelgrass"].astype("float64")
    # replace 999 in label with -1
    labels["label"] = labels["label"].replace(999, -1)
    labels["label"] = labels["label"].astype("float64")
    # divide labels by 100 for sigmoid output
    labels["label"] = labels["label"]/100

    folder = image_folder
    if folder[-1] != "/":
        folder += "/"
    if type == "None":
        labels["filename"] = [folder + i + '.jpg' for i in labels["filename"]]
    if type == "hess":
        labels["filename"] = [folder + i + '_hess.png' for i in labels["filename"]]

    binary_labels = labels
    continous_labels = labels[labels["label"] >= 0].reset_index(drop=True)
    # drop eelgrass and transect column in continous_labels
    continous_labels = continous_labels.drop(columns=["eelgrass"])
    binary_labels = binary_labels.drop(columns=["label"])
    # rename eelgrass column to label
    binary_labels = binary_labels.rename(columns={"eelgrass": "label"})
    return binary_labels, continous_labels

def timeseries_eval_split(df):
    df = df.sort_values(by=["filename"])
    df = df.reset_index(drop=True)
    return df

def timeseries_split(df):# split binary labels into train and val by transect
    train = df[df["transect"] != "R6"]
    val = df[df["transect"] == "R6"]
    # drop "transect" columns
    train = train.drop(columns=["transect"])
    val = val.drop(columns=["transect"])
    val = val.reset_index(drop=True)
    train = train.reset_index(drop=True)
    # sort by filename
    train = train.sort_values(by=["filename"])
    val = val.sort_values(by=["filename"])
    return train, val

def random_split(df, split_ratio=0.8):
    gen = torch.Generator().manual_seed(42)
    train_size = int(split_ratio * len(df))  # Adjust the split ratio as needed
    val_size = len(df) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(df, [train_size, val_size], generator=gen)
    return train_dataset, val_dataset