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

# Device configuration
device = torch.device('cuda:0')

# Hyper parameters
num_epochs = 100
num_classes = 29
batch_size = 32
learning_rate = 0.00075

train_dataset = torchvision.datasets.ImageFolder(root='Train/TrainImages',
                                                 transform=transforms.ToTensor())

valid_dataset = torchvision.datasets.ImageFolder(root='Validation/ValidationImages',
                                                 transform=transforms.ToTensor())

test_dataset = 'Test/TestImages'
test_data_files = os.listdir(test_dataset)
im = Image.open(f'{test_dataset}/{test_data_files[0]}')

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)


class ConvNet(nn.Module):
    def __init__(self, num_classes=29):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4 * 4 * 256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

# Train the model

train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []
running_loss = 0.0
running_corrects = 0

total_step = len(train_loader)
early_stopping = EarlyStopping(patience=20,
                               verbose=True)  # early stopping patience; how long to wait after last time validation loss improved

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

    model.eval()
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(images)
        # Calculate the loss
        loss = criterion(output, labels)
        # Record validation loss
        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    valid_epoch_loss = running_loss / 2298
    valid_epoch_acc = running_corrects.double() / 2298
    train_epoch_loss = running_loss / 5380
    train_epoch_acc = running_corrects.double() / 5380

    print('Train Loss: {:.4f} Acc: {:.4f}'.format(
        train_epoch_loss, train_epoch_acc))

    print('Valid Loss: {:.4f} Acc: {:.4f}'.format(
        valid_epoch_loss, valid_epoch_acc))

    print('Epoch [{}/{}] train_loss: {:.5f} valid_loss: {:.5f}'.format(epoch + 1, num_epochs, train_loss, valid_loss))

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []

    # early_stopping needs the validation loss to check if it has decreased,
    # and if it has, it will make a checkpoint of the current model
    early_stopping(valid_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break


# Test the model

def apply_test_transforms(inp):
    # out = transforms.functional.resize(inp, [224, 224])
    out = transforms.functional.to_tensor(inp)
    # mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)
    # std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)
    # out = transforms.functional.normalize(out, mean, std)
    return out.to(device)


def predict_single_instance(model, tensor):
    batch = torch.stack([tensor])
    preds = model(batch)
    _, predictions = torch.max(preds, 1)
    return predictions.item() + 1


def test_data_from_fname(fname):
    im = Image.open('{}/{}'.format(test_dataset, fname))
    return apply_test_transforms(im)


def extract_file_id(fname):
    return int(re.search('\d+', fname).group())


model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

predicts = {extract_file_id(fname): predict_single_instance(model, test_data_from_fname(fname))
            for fname in test_data_files}

ds = pd.Series({id: label for (id, label) in zip(predicts.keys(), predicts.values())})
df = pd.DataFrame(ds, columns=['Label']).sort_index()
df['ID'] = df.index
df = df[['ID', 'Label']]

df.to_csv('submission.csv', index=False)

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
