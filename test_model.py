import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import os

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

dataset = 'Test/TestImages'


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class DatasetFolder(dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None):
        super(DatasetFolder, self).__init__(root)
        self.transform = transform
        samples = make_dataset(self.root)

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)


test_loader = torch.utils.data.DataLoader(dataset=DatasetFolder(dataset),
                                          batch_size=batch_size,
                                          shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)

model.eval()

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer_conv.zero_grad()

    with torch.set_grad_enabled(False):
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        print(outputs)

im = Image.open('Validation/ValidationImages/1/Image5.jpg')
im = transforms.functional.to_tensor(im).to(device)
im = torch.stack([im])

preds = model(im)
_, predictions = torch.max(preds, 1)
print(preds)
print("Label: ", str(predictions.item() + 1))
