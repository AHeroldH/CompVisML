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

# Device configuration
device = torch.device('cuda:0')

# Hyper parameters
num_epochs = 25
num_classes = 29
batch_size = 90
learning_rate = 0.001

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


'''class ConvNet(nn.Module):
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
'''

model_conv = torchvision.models.densenet201(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.classifier.in_features
model_conv.classifier = nn.Linear(num_ftrs, num_classes)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# Train the model

since = time.time()

best_model_wts = copy.deepcopy(model_conv.state_dict())
best_acc = 0.0

#early_stopping = EarlyStopping(patience=20,
 #                              verbose=True)  # early stopping patience; how long to wait after last time validation loss improved

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    exp_lr_scheduler.step()
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


    # clear lists to track next epoch
    #train_losses = []
    #valid_losses = []

    # early_stopping needs the validation loss to check if it has decreased,
    # and if it has, it will make a checkpoint of the current model
    #early_stopping(valid_loss, model_conv)

    #if early_stopping.early_stop:
    #    print("Early stopping")
    #    break


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


model_conv.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

predicts = {extract_file_id(fname): predict_single_instance(model_conv, test_data_from_fname(fname))
            for fname in test_data_files}

ds = pd.Series({id: label for (id, label) in zip(predicts.keys(), predicts.values())})
df = pd.DataFrame(ds, columns=['Label']).sort_index()
df['ID'] = df.index
df = df[['ID', 'Label']]

df.to_csv('submission.csv', index=False)

# Save the model checkpoint
torch.save(model_conv.state_dict(), 'model.ckpt')
