import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
# from image_processing import list_files
# from image_processing import RankData
import torch.distributed
import torch.optim
from torch.utils.data import DataLoader
import os
import timm
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from IPython.display import Image
import matplotlib.pyplot as plt
import random
from random import randint
import imageio
import numpy as np
import scipy.stats as stats
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
import scipy.stats as stats
import pandas as pd

class OCTData(Dataset):


    def __init__(self,image_paths,labels_rank, transform = None):
        self.image_paths = image_paths
        self.transform = transform
        self.labels_rank = labels_rank

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        image = self.image_paths[index]

        ## get the rank to which image is associated
        # [86:-4] to only get the relevant ID bit of path
        rank = self.labels_rank['normalisedranking'][self.labels_rank['id'] == image[-76:-4]].values ##[0]

        image = imageio.imread(image)

        if self.transform:
            image = self.transform(image)
            #image = TF.adjust_gamma(image, gamma = 0.1 * random.randint(5,15))

        else: ## for tuning and validation dataset
            convert_tensor = transforms.ToTensor()
            image = convert_tensor(image)

        return image, rank


## Load label data from json file

labels_path = Path("/home/anissa/oct/scantensus-imp-coro-seligman-oct-rank-train-export-mar11.json")

with open(labels_path, 'r') as json_f:
    rankf = json.load(json_f)

keypoint_names = list(rankf.keys())

ids = []
rankings = []
volatilities = []
ratingdeviations = []
data = rankf['rankings']['list']

for item in data:
    id = item['id']
    ranking = item['rating']
    volatility = item['volatility']
    ratingdeviation =item['rd']
    ids.append(id)
    rankings.append(ranking)
    volatilities.append(volatility)
    ratingdeviations.append(ratingdeviation)

idranking = pd.DataFrame()
idranking['id'] = ids
idranking['ranking'] = rankings
idranking['volatilities'] = volatilities
idranking['ratingdeviations'] = ratingdeviations
idranking['normalisedrds'] = idranking['ratingdeviations']/(max(idranking['ranking'])-min(idranking['ranking']))

## add new column with normalised rankings between 0 and 1: ## for now, max ranking =1890.94 and min = 1205.87
#could normalise in a different way, for ex sigmoid function
idranking['normalisedranking'] = (idranking['ranking'] - min(idranking['ranking']))/(max(idranking['ranking'])-min(idranking['ranking']))

## idranking = dataframe with each flow trace id and corresponding ranking

print(idranking['normalisedranking'].std())

## Load image data
png_dir = "/home/anissa/oct/03"

all_images = os.listdir(png_dir)
all_images_paths = [os.path.join(png_dir,element) for element in all_images if element.endswith('.png')]

print(len(all_images_paths))

# images_folders = pd.DataFrame()
# images_folders['image_paths'] = all_images_paths
#
# folders = []
# for path in all_images_paths:
#     folder = path[23:27]
#     folders.append(folder)
#
# images_folders['folders'] = folders
#
# # split into development and test set (can only do 2)
# gs = GroupShuffleSplit(n_splits=2, test_size=.15, random_state=423)
# dev_ix, val_ix = next(gs.split(images_folders, groups=images_folders.folders))
#
# val_images_paths = []
# for i in val_ix:
#     val_images_path = images_folders['image_paths'][i]
#     val_images_paths.append(val_images_path)
#
# dev_images_folders = pd.DataFrame()
# dev_images_folders['image_paths'] = images_folders['image_paths'][dev_ix]
# dev_images_folders['folders'] = images_folders['folders'][dev_ix]
# dev_images_folders = dev_images_folders.reset_index(drop=True)
#
# # split development data into train and tune (test_size = 0.15/0.85)
#
# gs = GroupShuffleSplit(n_splits=2, test_size=.17647, random_state=423)
# train_ix, tune_ix = next(gs.split(dev_images_folders, groups=dev_images_folders.folders))
#
# train_images_paths = []
# for i in train_ix:
#     train_images_path = dev_images_folders['image_paths'][i]
#     train_images_paths.append(train_images_path)
#
# tune_images_paths = []
# for i in tune_ix:
#     tune_images_path = dev_images_folders['image_paths'][i]
#     tune_images_paths.append(tune_images_path)

seed = 423

random.Random(seed).shuffle(all_images_paths)
train_images_paths = all_images_paths[0:757]
tune_images_paths = all_images_paths[757:919]
val_images_paths = all_images_paths[919:1080]

# decide what transforms you want
transform = transforms.Compose([  # choose parameters
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=180)])

train_data = OCTData(train_images_paths, labels_rank = idranking, transform = transform)

tune_data = OCTData(tune_images_paths, labels_rank = idranking, transform = None)

val_data = OCTData(val_images_paths, labels_rank = idranking, transform = None)

num_workers = 0
initial_learning_rate = 0.004 #?? # maybe try higher learning rate.
batch_size = 16 #??

## GPU
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    device = torch.device('cuda')
else:
    device = 'cpu'

## Need to get model and configure it
num_classes = 1
model = timm.create_model('resnet34', num_classes = num_classes)
#model = timm.create_model('resnet18', num_classes = num_classes)
#model = timm.create_model('inception_v3', num_classes = num_classes)

model.default_cfg['input_size'] = (3,1024,1024)

#for inceptionv3 - model.Conv2d_1a_3x3.conv = nn.Conv2d(1,32,kernel_size=(3, 3), stride=(2, 2), bias=False)
model.conv1 = nn.Conv2d(3,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#are there other default configs to change?

writer = SummaryWriter()

if train_on_gpu:
    model.cuda()

## Define metrics

loss_fn = nn.MSELoss() #mean squared error

train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers= num_workers,
                                               pin_memory=False)
#sampler=train_sampler)

tune_dataloader = torch.utils.data.DataLoader(tune_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=False)
#sampler=tune_sampler)

val_dataloader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=False)
#sampler=tune_sampler)
print(len(train_dataloader.dataset))
print(len(tune_dataloader.dataset))

optimizer = Adam(params=model.parameters(), lr=initial_learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

n_epochs = 200 #??

tune_loss_min = np.Inf  # track change in validation loss

for epoch in range(1, n_epochs + 1):

    train_loss = 0.0
    tune_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()


    for data, target in train_dataloader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        target = target.float() #trying to fix error found dtype double but expected float
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = loss_fn(output, target)
        loss = loss.to(torch.float32)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    ######################
    # validate the model #
    ######################

    model.eval()
    for data, target in tune_dataloader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        target = target.float()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = loss_fn(output, target)
        loss = loss.to(torch.float32)
        # update average tune loss
        tune_loss += loss.item() * data.size(0)


    # calculate average losses
    train_loss = train_loss / len(train_dataloader.dataset)
    tune_loss = tune_loss / len(tune_dataloader.dataset)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tTune Loss: {:.6f}'.format(
        epoch, train_loss, tune_loss))
    writer.add_scalar("Train Loss", train_loss, epoch)
    writer.add_scalar("Tune Loss", tune_loss, epoch)

    # save model if validation loss has decreased
    if tune_loss <= tune_loss_min:
        print('Tune loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            tune_loss_min,
            tune_loss))
        torch.save(model.state_dict(), 'oct_model.pt')
        tune_loss_min = tune_loss

## TESTS

## View transformed images
# from PIL import Image
import matplotlib.pyplot as plt
#
# imagetransform = transforms.ToPILImage()
#
# for i in range(30):
#     tensor = train_data[i]    [0]
#     plt.imshow(tensor.permute(1, 2, 0), cmap='gray')
#     plt.show()

model.load_state_dict(torch.load('/home/anissa/oct/oct_model.pt'))
model.eval()
model.cpu()


import matplotlib.pyplot as plt
preds = []
actual = []
for i in range(len(train_data)):
    x = model(train_data[i][0][None, ...])
    preds.append(x[0].item())
    actual.append(train_data[i][1].item())

df = pd.DataFrame()
df['preds'] = preds
df['actualrank'] = actual
df['squarederror'] = (df['preds'] - df['actualrank']) ** 2

meansquarederror = df['squarederror'].mean()

print(stats.spearmanr(df['actualrank'], df['preds']))

preds = []
actual = []
for i in range(len(tune_data)):
    x = model(tune_data[i][0][None, ...])
    preds.append(x[0].item())
    actual.append(tune_data[i][1].item())

dftune = pd.DataFrame()
dftune['preds'] = preds
dftune['actualrank'] = actual
dftune['squarederror'] = (dftune['preds'] - dftune['actualrank']) ** 2

meansquarederrortune = dftune['squarederror'].mean()
print(meansquarederrortune)

print(stats.spearmanr(dftune['actualrank'], dftune['preds']))

preds = []
actual = []
for i in range(len(val_data)):
    x = model(val_data[i][0][None, ...])
    preds.append(x[0].item())
    actual.append(val_data[i][1].item())

import pandas as pd
dfval = pd.DataFrame()
dfval['preds'] = preds
dfval['actualrank'] = actual
dfval['squarederror'] = (dfval['preds'] - dfval['actualrank']) ** 2

meansquarederrorval = dfval['squarederror'].mean()
print(meansquarederrorval)

import scipy.stats as stats

print(stats.spearmanr(dfval['actualrank'], dfval['preds']))

plt.scatter(df['actualrank'], df['preds'],alpha=0.5)
plt.xlabel("Human ranking score")
plt.ylabel("AI-predicted score")
plt.title("Network Performance on Training Data")
plt.ylim([0,1])
x = np.linspace(0,1,10)
y = x
plt.plot(x,y,'-r')
plt.show()

plt.scatter(dftune['actualrank'], dftune['preds'], alpha=0.5)
plt.xlabel("Human ranking score")
plt.ylabel("AI-predicted score")
plt.title("Network Performance on Tuning Data")
plt.plot(x,y,'-r')
plt.show()
#
plt.scatter(dfval['actualrank'], dfval['preds'], alpha=0.5)
plt.xlabel("Human ranking score")
plt.ylabel("AI-predicted score")
plt.title("Network Performance on Internal Validation Data")
plt.ylim([0,1])
x = np.linspace(0,1,10)
y = x
plt.plot(x,y,'-r')
plt.show()


