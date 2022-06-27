import time
import torch
from torch.utils.data import DataLoader, Dataset
import PIL
import os

from torchvision import transforms as transforms
import torch.optim as optim
from torchvision import models
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
import pickle
import torch.nn.functional as F
import seaborn as sns

from utils import MHCoverDataset, get_dataloader
from pathlib import Path
from utils.training import TrainingInterface

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pkl_url_file = "utils/dataset/label_translate.pkl"
root_dir = "data/images_transformed/"
with open(pkl_url_file, 'rb') as pkl_file:
    label_dict = pickle.load(pkl_file)
    
df = pd.read_csv("data/labels.csv")

df_new = pd.read_csv("data/new_labels.csv")
df_new['label'] = df_new['type'] + '/' + df_new['subtype']
df_new = df_new.drop(['type', 'subtype'], axis=1)
df_new["set"] = "train"

df = pd.concat([df, df_new])
df = df.reset_index()
print("shape dataframe:", df.shape)

new_model = models.resnet152(pretrained=False, progress=True)
new_model = TrainingInterface(model=new_model, name="resnet152")

my_train_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=45),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomPerspective(p=0.2),
        transforms.RandomEqualize(p=0.2),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
    ]
)

trainloader = get_dataloader(
    root_dir=root_dir,
    df=df[df.set =="train"].reset_index(),
    fp_label_translator=pkl_url_file,
    transformations=my_train_transforms,
    batch_size=32,
    workers=0,
    pin_memory=True,
    shuffle=True
)

valloader = get_dataloader(
    root_dir=root_dir,
    df=df[df.set =="val"].reset_index(),
    fp_label_translator=pkl_url_file,
    transformations=my_train_transforms,
    batch_size=32,
    workers=0,
    pin_memory=True,
    shuffle=True
)

n_epochs = 200
learning_rate = 0.005


model = new_model.model
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.0001)

new_model.train(criterion = criterion, optimizer = optimizer, n_epochs = n_epochs, dataloader_train = trainloader ,dataloader_val= valloader)

torch.save(new_model.model, 'model.pth')