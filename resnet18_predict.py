from __future__ import print_function, division
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn

def default_loader(path):
    return Image.open(path)


class TestDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        a = '0'
        b = '\n'
        for line in fh:
            line = line + b + a
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

data_transforms = {
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
test_data = TestDataset(txt='test.txt', transform = data_transforms['test'])
print(len(test_data))

test_data_loader = DataLoader(test_data, batch_size =5, shuffle = False, num_workers = 4)
print(len(test_data_loader))

for i, (batch_x, batch_y) in enumerate(test_data_loader):
    if(i<4):
        print(i, batch_x.size(),batch_y.size())

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model.load_state_dict(torch.load('params1.pkl'))
model_dict = model.state_dict()
for k, v in model_dict.items():
    print(k)

name = 'label'
path = './' + name + '.txt'
file = open(path, 'w')

for data in test_data_loader:
    # get the inputs
    inputs,labels = data
    # wrap them in Variable
    inputs, labels = Variable(inputs), Variable(labels)
    # forward
    outputs = model(inputs)
    _, preds_top1 = torch.max(outputs.data, 1)

    a = str(preds_top1)

    file.write(a)
file.close()

def adjust_txt(infile, outfile):
    infopen = open(infile,'r')
    outfopen = open(outfile,'w')
    lines = infopen.readlines()
    for line in lines:
        line = line.lstrip()
        if '.' in line or line.split()==False:
            outfopen.writelines('')
        else:
            outfopen.writelines(line)
    infopen.close()
    outfopen.close()

adjust_txt('label.txt','Project2_20472766.txt')























