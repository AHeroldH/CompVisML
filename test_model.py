import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import os
import csv
import pandas as pd

device = torch.device("cuda:0")
model = torchvision.models.densenet201(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 29)
model.load_state_dict(torch.load('model.pt'))
model.to(device)

# Hyper parameters
num_epochs = 1
num_classes = 29
batch_size = 90
learning_rate = 0.001

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

    def __init__(self):
        super(DatasetFolder, self).__init__()
        ids, samples = make_dataset(dataset)

        self.samples = samples
        self.ids = ids

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        ids = self.ids
        path = self.samples[index]
        sample = loader(path)
        sample = transforms.functional.to_tensor(sample)

        return ids[index], sample

    def __len__(self):
        return len(self.samples)


test_loader = torch.utils.data.DataLoader(dataset=DatasetFolder(),
                                          batch_size=1,
                                          shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)

model.eval()

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
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    predictions = [ids.item(), preds.item()+1]

    with open("not_sorted_submission.csv", "a") as submission_csv:
        writer = csv.writer(submission_csv)
        writer.writerow(predictions)

submission_csv.close()

read = pd.read_csv("not_sorted_submission.csv", usecols=['ID', 'Label'], index_col=0)

# sorting based on column labels
df = read.sort_index()
df['ID'] = df.index
df = df[['ID','Label']]
df.to_csv('submission.csv', index=False)