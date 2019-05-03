import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0")
model = torchvision.models.densenet161(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 29)
model.load_state_dict(torch.load('model.ckpt'))
model.to(device)

# Hyper parameters
num_epochs = 1
num_classes = 29
batch_size = 90
learning_rate = 0.001


valid_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

valid_dataset = torchvision.datasets.ImageFolder(root='Validation/ValidationImages',
                                                 transform=valid_transform)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model.eval()

running_loss = 0.0
running_corrects = 0
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

for images, labels in valid_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer_conv.zero_grad()

    with torch.set_grad_enabled(False):
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

    running_loss += loss.item() * images.size(0)
    running_corrects += torch.sum(preds == labels.data)
    correct = np.squeeze(preds.eq(labels.data.view_as(preds)))

    for i in range(batch_size):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

epoch_loss = running_loss / 2298
epoch_acc = running_corrects.double() / 2298

print('Valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

for i in range(num_classes):
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
