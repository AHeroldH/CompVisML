import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim

device = torch.device("cuda:0")
model = torchvision.models.densenet201(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 29)
model.load_state_dict(torch.load('model.ckpt'))
model.to(device)

# Hyper parameters
num_epochs = 1
num_classes = 29
batch_size = 90
learning_rate = 0.001

valid_dataset = torchvision.datasets.ImageFolder(root='Validation/ValidationImages',
                                                 transform=transforms.ToTensor())

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)

model.eval()

running_loss = 0.0
running_corrects = 0
for images, labels in valid_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer_conv.zero_grad()

    with torch.set_grad_enabled(False):
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

    running_loss += loss.item() * images.size(0)
    running_corrects += torch.sum(preds == labels.data)

epoch_loss = running_loss / 2298
epoch_acc = running_corrects.double() / 2298

print('Valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

im = Image.open('Validation/ValidationImages/1/Image5.jpg')
im = transforms.functional.to_tensor(im).to(device)
im = torch.stack([im])

preds = model(im)
_, predictions = torch.max(preds, 1)
print(preds)
print("Label: ", str(predictions.item()+1))
