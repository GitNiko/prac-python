from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time 
import copy
import os

plt.ion()

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

data_dir = 'hymenoptera_data'
dsets = {
  x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']}
dset_loaders = {
  x: torch.utils.data.DataLoader(dsets[x], batch_size=4, shuffle=True, num_workers=4)
    for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes


use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
  since = time.time()

  best_model = model
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
      if phase == 'train':
        # optimizer = lr_scheduler(optimizer, epoch)
        lr_scheduler.step()
        model.train(True)
      else:
        model.train(False)

      running_loss = 0.0
      running_corrects = 0
      
      for data in dset_loaders[phase]:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        if phase == 'train':
          loss.backward()
          optimizer.step()
        
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)
      
      epoch_loss = running_loss / dset_sizes[phase]
      epoch_acc = running_corrects / dset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = copy.deepcopy(model)
    
    print()
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))
  return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
  lr = init_lr * (0.1**(epoch // lr_decay_epoch))
  if epoch % lr_decay_epoch == 0:
    print('LR is set to {}'.format(lr))
  
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  
  return optimizer

def visualize_model(model, num_images=6):
  images_so_far = 0
  fig = plt.figure()

  for i, data in enumerate(dset_loaders['val']):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)

    for j in range(inputs.size()[0]):
      images_so_far += 1
      ax = plt.subplot(num_images//2, 2, images_so_far)
      ax.axis('off')
      ax.set_title('predicted: {}'.format(dset_classes[preds[j]]))
      imshow(inputs.cpu().data[j])

      if images_so_far == num_images:
        return

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(99999)  # pause a bit so that plots are updated

# inputs, classes = next(iter(dset_loaders['train']))

# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[dset_classes[x] for x in classes])

# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)

# criterion = nn.CrossEntropyLoss()

# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
  param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)



# Epoch 0/24
# ----------
# LR is set to 0.001
# train Loss: 0.1361 Acc: 0.7377
# val Loss: 0.0423 Acc: 0.9281

# Epoch 1/24
# ----------
# train Loss: 0.1258 Acc: 0.7541
# val Loss: 0.0557 Acc: 0.9150

# Epoch 2/24
# ----------
# train Loss: 0.1117 Acc: 0.8156
# val Loss: 0.0946 Acc: 0.8562

# Epoch 3/24
# ----------
# train Loss: 0.1204 Acc: 0.8279
# val Loss: 0.0732 Acc: 0.8954

# Epoch 4/24
# ----------
# train Loss: 0.1108 Acc: 0.8361
# val Loss: 0.0832 Acc: 0.8693

# Epoch 5/24
# ----------
# train Loss: 0.1073 Acc: 0.8238
# val Loss: 0.0762 Acc: 0.8824

# Epoch 6/24
# ----------
# train Loss: 0.1423 Acc: 0.7869
# val Loss: 0.0532 Acc: 0.9346

# Epoch 7/24
# ----------
# LR is set to 0.0001
# train Loss: 0.0955 Acc: 0.8320
# val Loss: 0.0615 Acc: 0.9346

# Epoch 8/24
# ----------
# train Loss: 0.0682 Acc: 0.8689
# val Loss: 0.0518 Acc: 0.9412

# Epoch 9/24
# ----------
# train Loss: 0.0705 Acc: 0.8770
# val Loss: 0.0567 Acc: 0.9281

# Epoch 10/24
# ----------
# train Loss: 0.0685 Acc: 0.8852
# val Loss: 0.0561 Acc: 0.9216

# Epoch 11/24
# ----------
# train Loss: 0.0722 Acc: 0.8852
# val Loss: 0.0482 Acc: 0.9477

# Epoch 12/24
# ----------
# train Loss: 0.0907 Acc: 0.8115
# val Loss: 0.0533 Acc: 0.9281

# Epoch 13/24
# ----------
# train Loss: 0.0788 Acc: 0.8566
# val Loss: 0.0640 Acc: 0.9412

# Epoch 14/24
# ----------
# LR is set to 1.0000000000000003e-05
# train Loss: 0.0753 Acc: 0.8852
# val Loss: 0.0513 Acc: 0.9412

# Epoch 15/24
# ----------
# train Loss: 0.0763 Acc: 0.8566
# val Loss: 0.0509 Acc: 0.9477

# Epoch 16/24
# ----------
# train Loss: 0.0613 Acc: 0.8975
# val Loss: 0.0536 Acc: 0.9412

# Epoch 17/24
# ----------
# train Loss: 0.0636 Acc: 0.8811
# val Loss: 0.0495 Acc: 0.9477

# Epoch 18/24
# ----------
# train Loss: 0.0807 Acc: 0.8484
# val Loss: 0.0484 Acc: 0.9477

# Epoch 19/24
# ----------
# train Loss: 0.0584 Acc: 0.9016
# val Loss: 0.0523 Acc: 0.9281

# Epoch 20/24
# ----------
# train Loss: 0.0808 Acc: 0.8525
# val Loss: 0.0595 Acc: 0.9020

# Epoch 21/24
# ----------
# LR is set to 1.0000000000000002e-06
# train Loss: 0.0672 Acc: 0.8893
# val Loss: 0.0536 Acc: 0.9346

# Epoch 22/24
# ----------
# train Loss: 0.0806 Acc: 0.8648
# val Loss: 0.0553 Acc: 0.9085

# Epoch 23/24
# ----------
# train Loss: 0.0633 Acc: 0.9057
# val Loss: 0.0548 Acc: 0.9216

# Epoch 24/24
# ----------
# train Loss: 0.0557 Acc: 0.9016
# val Loss: 0.0616 Acc: 0.9150

# Training complete in 29m 14s
# Best val Acc: 0.947712


# Epoch 0/24
# ----------
# train Loss: 0.1389 Acc: 0.7090
# val Loss: 0.0655 Acc: 0.9216

# Epoch 1/24
# ----------
# train Loss: 0.1477 Acc: 0.7418
# val Loss: 0.0628 Acc: 0.9150

# Epoch 2/24
# ----------
# train Loss: 0.1558 Acc: 0.7623
# val Loss: 0.0495 Acc: 0.9216

# Epoch 3/24
# ----------
# train Loss: 0.1111 Acc: 0.8238
# val Loss: 0.0617 Acc: 0.9281

# Epoch 4/24
# ----------
# train Loss: 0.1043 Acc: 0.8156
# val Loss: 0.0621 Acc: 0.9216

# Epoch 5/24
# ----------
# train Loss: 0.1599 Acc: 0.7746
# val Loss: 0.0581 Acc: 0.9085

# Epoch 6/24
# ----------
# train Loss: 0.1381 Acc: 0.7664
# val Loss: 0.0775 Acc: 0.8758

# Epoch 7/24
# ----------
# train Loss: 0.1015 Acc: 0.8279
# val Loss: 0.0551 Acc: 0.9216

# Epoch 8/24
# ----------
# train Loss: 0.0944 Acc: 0.8238
# val Loss: 0.0608 Acc: 0.9085

# Epoch 9/24
# ----------
# train Loss: 0.0850 Acc: 0.8402
# val Loss: 0.0480 Acc: 0.9477

# Epoch 10/24
# ----------
# train Loss: 0.0954 Acc: 0.8238
# val Loss: 0.0446 Acc: 0.9477

# Epoch 11/24
# ----------
# train Loss: 0.1038 Acc: 0.8115
# val Loss: 0.0443 Acc: 0.9477

# Epoch 12/24
# ----------
# train Loss: 0.0793 Acc: 0.8402
# val Loss: 0.0453 Acc: 0.9477

# Epoch 13/24
# ----------
# train Loss: 0.0785 Acc: 0.8361
# val Loss: 0.0436 Acc: 0.9477

# Epoch 14/24
# ----------
# train Loss: 0.0987 Acc: 0.8238
# val Loss: 0.0478 Acc: 0.9477

# Epoch 15/24
# ----------
# train Loss: 0.1074 Acc: 0.8074
# val Loss: 0.0494 Acc: 0.9346

# Epoch 16/24
# ----------
# train Loss: 0.0825 Acc: 0.8402
# val Loss: 0.0651 Acc: 0.8824

# Epoch 17/24
# ----------
# train Loss: 0.0761 Acc: 0.8852
# val Loss: 0.0459 Acc: 0.9542

# Epoch 18/24
# ----------
# train Loss: 0.0826 Acc: 0.8402
# val Loss: 0.0468 Acc: 0.9477

# Epoch 19/24
# ----------
# train Loss: 0.0639 Acc: 0.8893
# val Loss: 0.0454 Acc: 0.9412

# Epoch 20/24
# ----------
# train Loss: 0.0790 Acc: 0.8648
# val Loss: 0.0462 Acc: 0.9542

# Epoch 21/24
# ----------
# train Loss: 0.1000 Acc: 0.8279
# val Loss: 0.0480 Acc: 0.9281

# Epoch 22/24
# ----------
# train Loss: 0.0684 Acc: 0.8893
# val Loss: 0.0496 Acc: 0.9477

# Epoch 23/24
# ----------
# train Loss: 0.0672 Acc: 0.8893
# val Loss: 0.0454 Acc: 0.9477

# Epoch 24/24
# ----------
# train Loss: 0.0822 Acc: 0.8402
# val Loss: 0.0495 Acc: 0.9216

# Training complete in 11m 56s
# Best val Acc: 0.954248


visualize_model(model_conv)

plt.ioff()
plt.show()