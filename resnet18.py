from __future__ import print_function, division
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import numpy as np
import torch
import time
from torch.autograd import Variable
import torch.nn as nn

plt.ion()

def default_loader(path):
    return Image.open(path)

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
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
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_data=MyDataset(txt='train.txt', transform=data_transforms['train'])
print(len(train_data))
val_data=MyDataset(txt='val.txt', transform=data_transforms['val'])
print(len(val_data))
train_data_loader = DataLoader(train_data, batch_size=5,shuffle=True,num_workers=4)
print(len(train_data_loader))
val_data_loader = DataLoader(val_data, batch_size=5,shuffle=True,num_workers=4)
print(len(val_data_loader))
dsets = {'train':train_data, 'val': val_data}
dset_sizes = {x: len(dsets[x]) for x in ['train','val']}
print(dset_sizes)


for i, (batch_x, batch_y) in enumerate(train_data_loader):
    if(i<4):
        print(i, batch_x.size(),batch_y.size())


def imshow(imgs, title=None):
    inp = imgs.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(train_data_loader))
print(classes)
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out)


def train_model(model, criterion, optimizer, scheduler, num_epoch=6):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc_1 = 0.0
    val_loss = []
    val_accuracy_top1 = []
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects_1 = 0

            if phase == 'train':
                scheduler.step()
                model.train(True) # Set model to training mode
                # Iterate over data
                for data in train_data_loader:
                    # get the inputs
                    inputs, labels = data
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    outputs = model(inputs)
                    _, preds_top1 = torch.max(outputs.data, 1)

                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
                    # statistics
                    running_loss += loss.data[0]
                    running_corrects_1 += torch.sum(preds_top1 == labels.data)
            else:
                model.train(False) # Set model to evaluate mode
                # Iterate over data
                for data in val_data_loader:
                    # get the inputs
                    inputs, labels = data
                    # wrap them in Variable

                    inputs, labels = Variable(inputs), Variable(labels)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    outputs = model(inputs)
                    _, preds_top1 = torch.max(outputs.data, 1)

                    loss = criterion(outputs, labels)
                    # statistics
                    running_loss += loss.data[0]
                    running_corrects_1 += torch.sum(preds_top1 == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_loss_val = running_loss / dset_sizes['val']
            epoch_acc_1 = running_corrects_1 / dset_sizes[phase]
            epoch_acc_1_val = running_corrects_1 / dset_sizes['val']
            val_loss.append(epoch_loss_val)
            val_accuracy_top1.append(epoch_acc_1_val)

            print('{} Loss: {:.4f}  Top1_Accuracy: {:.4f}  '.format(
                phase, epoch_loss, epoch_acc_1))
            # deep copy the model
            if phase == 'val' and epoch_acc_1 > best_acc_1:
                best_acc_1 = epoch_acc_1
                best_model_wts = model.state_dict()
        print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Top1_Accuracy: {:4f}".format(best_acc_1))

    torch.save(best_model_wts, 'params1.pkl')

    # load best model weights
    model.load_state_dict(torch.load('params1.pkl'))
    return model


def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(val_data_loader):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(classes[labels.data[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

# Finetuning the convnet
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 5)

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 6 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epoch=14)

visualize_model(model_ft)
plt.ioff()
plt.show()