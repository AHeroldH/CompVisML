import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pytorchtools import EarlyStopping
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import re
import pandas as pd

# Device configuration
device = torch.device('cuda:0')

# Hyper parameters
num_epochs = 3
num_classes = 29
batch_size = 34
learning_rate = 0.0005

train_dataset = torchvision.datasets.ImageFolder(root='Train/TrainImages',
                                                 transform=transforms.ToTensor())

valid_dataset = torchvision.datasets.ImageFolder(root='Validation/ValidationImages',
                                                 transform=transforms.ToTensor())

#test_dataset = torchvision.datasets.ImageFolder(root='Test',
#                                                transform=transforms.ToTensor())

test_dataset = 'Test/TestImages'
test_data_files = os.listdir(test_dataset)
im = Image.open('{}/{}'.format(test_dataset, test_data_files[0]))
plt.imshow(im)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Convolutional neural network (two convolutional layers)
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
        # self.layer7 = nn.Sequential(
        #    nn.Conv2d(160, 192, kernel_size=8, stride=1, padding=4),
        #    nn.BatchNorm2d(192),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer8 = nn.Sequential(
        #   nn.Conv2d(192, 224, kernel_size=8, stride=1, padding=4),
        #   nn.BatchNorm2d(224),
        #   nn.ReLU(),
        #   nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer9 = nn.Sequential(
        #   nn.Conv2d(224, 256, kernel_size=8, stride=1, padding=4),
        #   nn.BatchNorm2d(256),
        #   nn.ReLU(),
        #   nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4 * 4 * 256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # out = self.layer7(out)
        # out = self.layer8(out)
        # out = self.layer9(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

# Train the model

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []

total_step = len(train_loader)
early_stopping = EarlyStopping(patience=2,
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

    model.eval()
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(images)
        # calculate the loss
        loss = criterion(output, labels)
        # record validation loss
        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    print('Epoch [{}/{}] train_loss: {:.5f} valid_loss: {:.5f}'.format(epoch + 1, num_epochs, train_loss, valid_loss))

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []

    # early_stopping needs the validation loss to check if it has decresed,
    # and if it has, it will make a checkpoint of the current model
    early_stopping(valid_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# Test the model

# def apply_test_transforms(inp):
#    out = transforms.functional.resize(inp, [255, 255])
#    out = transforms.functional.to_tensor(out)
#    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)
#    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)
#    out = transforms.functional.normalize(out, mean, std)
#    return out
#
#
# def test_data_from_fname(fname):
#    im = Image.open('{}/{}'.format(test_dataset, fname))
#    return apply_test_transforms(im)
#
#
# def extract_file_id(fname):
#     print("Extracting id from " + fname)
#     return int(re.search('\d+', fname).group())
#
#
# im_as_tensor = apply_test_transforms(im)
# print(im_as_tensor.size())
# minibatch = torch.stack([im_as_tensor])
# print(minibatch.size())
#
# model.cuda()
#
# for inp in im_as_tensor:
#     x = inp.cuda()
#     model(x)
#
# #model(minibatch)
#
# model.eval()
# predictions = {extract_file_id(fname): test_data_from_fname(fname)
#               for fname in test_data_files}
#
# ds = pd.Series({id: label for (id, label) in zip(predictions.keys(), predictions.values())})
# ds.head()
# df = pd.DataFrame(ds, columns=['label']).sort_index()
# df['id'] = df.index
# df = df[['id', 'label']]
# df.head()
#
# df.to_csv('submission.csv', index=False)

test_loss = 0.0
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

with torch.no_grad():
   for i, images in enumerate(test_loader):
       if len(labels.data) != batch_size:
          break

       images = images.to(device)
       # labels = labels.to(device)

       outputs = model(images)

#ds = pd.Series((outputs.cpu()).numpy(), )
#ds.head()
#print(outputs)
#print((outputs.cpu()).tolist())
df = pd.DataFrame(outputs.cpu().detach().numpy(), columns=['ID', 'Label']).sort_index()
df['ID'] = df.index
df = df[['ID', 'Label']]
df.head()

df.to_csv('submission.csv', index=False)

#        loss = criterion(outputs, labels)
#        test_loss += loss.item() * images.size(0)
#        _, predicted = torch.max(outputs.data, 1)
#        correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
#        # calculate test accuracy for each object class
#        for i in range(batch_size):
#            label = labels.data[i]
#            class_correct[label] += correct[i].item()
#            class_total[label] += 1
#
#    test_loss = test_loss / len(test_loader.dataset)
#    print('Test Loss: {:.6f}\n'.format(test_loss))
#
#    for i in range(num_classes):
#        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
#            str(i), 100 * class_correct[i] / class_total[i],
#            np.sum(class_correct[i]), np.sum(class_total[i])))
#
#    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
#        100. * np.sum(class_correct) / np.sum(class_total),
#       np.sum(class_correct), np.sum(class_total)))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
