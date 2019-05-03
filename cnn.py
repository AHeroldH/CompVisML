import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pytorchtools import EarlyStopping
import numpy as np
from PIL import Image
import os
import re
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import csv

# Device configuration
device = torch.device('cuda:0')

# Hyper parameters
num_epochs = 60
num_classes = 29
batch_size = 80
learning_rate = 0.001

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor()
])

valid_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(root='Train/TrainImages',
                                                 transform=train_transforms)

valid_dataset = torchvision.datasets.ImageFolder(root='Validation/ValidationImages',
                                                 transform=valid_transform)

class_names = train_dataset.classes

dataset = 'Test/TestImages'


def make_dataset(dir):
    images = []
    ids = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            images.append(path)
            id = int(re.search('\d+', fname).group())
            ids.append(id)

    return ids, images


def loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DatasetFolder:
    def __init__(self):
        super(DatasetFolder, self).__init__()
        ids, samples = make_dataset(dataset)

        self.samples = samples
        self.ids = ids

    def __getitem__(self, index):
        ids = self.ids
        path = self.samples[index]
        sample = loader(path)
        sample = valid_transform(sample)

        return ids[index], sample

    def __len__(self):
        return len(self.samples)


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=DatasetFolder(),
                                          batch_size=1,
                                          shuffle=False)

model_conv = torchvision.models.densenet161(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.classifier.in_features
model_conv.classifier = nn.Linear(num_ftrs, num_classes)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.Adam(model_conv.classifier.parameters(), lr=learning_rate)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_conv, num_epochs)

# Train the model

since = time.time()

best_model_wts = copy.deepcopy(model_conv.state_dict())
best_acc = 0.0

total_step = len(train_loader)
# early_stopping = EarlyStopping(patience=20,
#                             verbose=True)  # early stopping patience; how long to wait after last time validation loss improved

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # exp_lr_scheduler.step()
    model_conv.train()

    running_loss = 0.0
    running_corrects = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer_conv.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model_conv(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer_conv.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / 5380
    epoch_acc = running_corrects.double() / 5380

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    model_conv.eval()

    running_loss = 0.0
    running_corrects = 0

    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer_conv.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model_conv(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / 2298
    epoch_acc = running_corrects.double() / 2298

    print('Valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model_conv.state_dict())

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# Test the model

model_conv.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

row = ['ID', 'Label']

os.remove('not_sorted_submission.csv')

with open("not_sorted_submission.csv", "w") as submission_csv:
    writer = csv.writer(submission_csv)
    writer.writerow(row)

submission_csv.close()

for ids, images in test_loader:
    images = images.to(device)
    ids = ids.to(device)

    optimizer_conv.zero_grad()

    with torch.set_grad_enabled(False):
        outputs = model_conv(images)
        _, preds = torch.max(outputs, 1)

    predictions = [ids.item(), class_names[preds.item()]]

    with open("not_sorted_submission.csv", "a") as submission_csv:
        writer = csv.writer(submission_csv)
        writer.writerow(predictions)

submission_csv.close()

read = pd.read_csv("not_sorted_submission.csv", usecols=['ID', 'Label'], index_col=0)

# sorting based on column labels
df = read.sort_index()
df['ID'] = df.index
df = df[['ID', 'Label']]
df.to_csv('submission.csv', index=False)

# Save the model checkpoint
torch.save(model_conv.state_dict(), 'model.ckpt')
