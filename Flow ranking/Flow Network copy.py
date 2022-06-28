import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import random
import torch.distributed
import torch.optim
from torch.utils.data import DataLoader
import os
import timm
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import json


import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from random import randint
import os
import imageio
import sklearn
from sklearn.model_selection import GroupShuffleSplit

## find size of images - image size = 968, 712 (width, height), dim = 2
# import PIL
# from PIL import Image
# #image = PIL.Image.open("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/scantensus-ranking/01/0e/84/01-0e84ef3dae104a1b2345009d7c21a7131d14a1385aa94d7158e69cd08ba649f9-0000.png")
# image = PIL.Image.open("/home/anissa/scantensus-ranking/01/0e/84/01-0e84ef3dae104a1b2345009d7c21a7131d14a1385aa94d7158e69cd08ba649f9-0000.png")
# width, height = image.size
# print(width, height)

# looks like dimensions of actual flow trace are 768 x 512

def list_files(dir):
    r = []
    png_dir_list = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    png_dir_list = [element for element in r if element.endswith('.png')]
    return png_dir_list

class RankData(Dataset):

    actual_image_width = 768

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
        image_width = image.shape[1]


        if self.transform:

            random_starting_x = randint(100, self.actual_image_width - 256)

            # height: crops black bit on the top, width: within the trace portion (100,868), selects random interval 256 units long
            image = image[200:712, random_starting_x:random_starting_x + 256]
            image = self.transform(image)
            image = TF.adjust_gamma(image, gamma = 0.1 * random.randint(5,15))

        else: ## for tuning and validation dataset
            center_image = int(image_width / 2)
            image = image[:, center_image - 128: center_image + 128]

            convert_tensor = transforms.ToTensor()
            image = convert_tensor(image)

        return image, rank


## Load label data from json file

labels_path = Path("/home/anissa/scantensus-ranking/scantensus-imp-coro-seligman-flow-ranking-dev-final-export-22030.json")


with open(labels_path, 'r') as json_f:
    rankf = json.load(json_f)

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

import pandas as pd

idranking = pd.DataFrame()
idranking['id'] = ids
idranking['ranking'] = rankings
idranking['volatilities'] = volatilities
idranking['ratingdeviations'] = ratingdeviations
idranking['normalisedrds'] = idranking['ratingdeviations']/(max(idranking['ranking'])-min(idranking['ranking']))
idranking['normalisedranking'] = (idranking['ranking'] - min(idranking['ranking']))/(max(idranking['ranking'])-min(idranking['ranking']))

## Load image data

#png_dir = "/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/scantensus-ranking/01"
png_dir = "/home/anissa/scantensus-ranking/01"

all_images_paths = list_files(png_dir)
len(all_images_paths)

seed = 423
random.Random(seed).shuffle(all_images_paths)

images_folders = pd.DataFrame()
images_folders['image_paths'] = all_images_paths

folders = []
for path in all_images_paths:
    folder = path[44:48]
    folders.append(folder)

images_folders['folders'] = folders

# split into development and test set (can only do 2)
gs = GroupShuffleSplit(n_splits=2, test_size=.14, random_state=423)
dev_ix, val_ix = next(gs.split(images_folders, groups=images_folders.folders))

val_images_paths = []
for i in val_ix:
    val_images_path = images_folders['image_paths'][i]
    val_images_paths.append(val_images_path)

dev_images_folders = pd.DataFrame()
dev_images_folders['image_paths'] = images_folders['image_paths'][dev_ix]
dev_images_folders['folders'] = images_folders['folders'][dev_ix]
dev_images_folders = dev_images_folders.reset_index(drop=True)

# split development data into train and tune (test_size = 0.15/0.85)

gs = GroupShuffleSplit(n_splits=2, test_size=.17, random_state=423)
train_ix, tune_ix = next(gs.split(dev_images_folders, groups=dev_images_folders.folders))

train_images_paths = []
for i in train_ix:
    train_images_path = dev_images_folders['image_paths'][i]
    train_images_paths.append(train_images_path)

tune_images_paths = []
for i in tune_ix:
    tune_images_path = dev_images_folders['image_paths'][i]
    tune_images_paths.append(tune_images_path)

##separate into train, tune and validate (0.7, 0.15, 0.15) sets - total =519, so 363,78,78
#
# seed = 423
#
# random.Random(seed).shuffle(all_images_paths)
# train_images_paths = all_images_paths[0:743]
# tune_images_paths = all_images_paths[743:903]
# val_images_paths = all_images_paths[903:1062]

# need to figure out how to get transforms to work, right now error is = img should be PIL Image. Got <class 'imageio.core.util.Array'>
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1), scale = (0.66,1.5))])

train_data = []
train_data = RankData(train_images_paths, labels_rank = idranking, transform = transform)

tune_data = RankData(tune_images_paths, labels_rank = idranking, transform = None)

val_data = RankData(val_images_paths, labels_rank = idranking, transform = None)

print(train_data[0])

print(len(train_data))
print(len(tune_data))
print(len(val_data))

num_workers = 0
initial_learning_rate = 0.004
batch_size = 16

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

model.default_cfg['input_size'] = (1,512,256)

#for inceptionv3 - model.Conv2d_1a_3x3.conv = nn.Conv2d(1,32,kernel_size=(3, 3), stride=(2, 2), bias=False)
model.conv1 = nn.Conv2d(1,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#are there other default configs to change?


if train_on_gpu:
    model.cuda()

## Define metrics

loss_fn = nn.MSELoss() #mean squared error

#train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers= num_workers,
                                               pin_memory=False)
                                               #sampler=train_sampler)

#tune_sampler = torch.utils.data.distributed.DistributedSampler(tune_data)

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
        train_loss += loss.item() * data.size(0) # why did i multiply by data.size


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
    #writer.add_scalar("Train Loss", train_loss, epoch)
    #writer.add_scalar("Tune Loss", tune_loss, epoch)

    # save model if validation loss has decreased
    if tune_loss <= tune_loss_min:
        print('Tune loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            tune_loss_min,
            tune_loss))
        torch.save(model.state_dict(), '/home/anissa/scantensus-ranking/final_flow_quality_model.pt')
        tune_loss_min = tune_loss

## TESTS

## View transformed images
# from PIL import Image
import matplotlib.pyplot as plt
#
# imagetransform = transforms.ToPILImage()
#
# for i in range(30):
#     tensor = train_data[i][0]
#     plt.imshow(tensor.permute(1, 2, 0), cmap='gray')
#     plt.show()
# #
model.load_state_dict(torch.load('/home/anissa/scantensus-ranking/final_flow_quality_model.pt'))
model.eval()
model.cpu()

preds = []
actual = []
for i in range(len(train_data)):
    x = model(train_data[i][0][None, ...])
    preds.append(x[0].item())
    actual.append(train_data[i][1].item())

import pandas as pd
df = pd.DataFrame()
df['preds'] = preds
df['actualrank'] = actual
df['squarederror'] = (df['preds'] - df['actualrank']) ** 2

meansquarederror = df['squarederror'].mean()

import scipy.stats as stats

tau, p_value = stats.kendalltau(df['actualrank'], df['preds'])

print(tau)

print(stats.spearmanr(df['actualrank'], df['preds']))

preds = []
actual = []
for i in range(len(tune_data)):
    x = model(tune_data[i][0][None, ...])
    preds.append(x[0].item())
    actual.append(tune_data[i][1].item())

import pandas as pd
dftune = pd.DataFrame()
dftune['preds'] = preds
dftune['actualrank'] = actual
dftune['squarederror'] = (dftune['preds'] - dftune['actualrank']) ** 2

meansquarederrortune = dftune['squarederror'].mean()
print(meansquarederrortune)

import scipy.stats as stats

tau, p_value = stats.kendalltau(dftune['actualrank'], dftune['preds'])

print(tau)

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

tau, p_value = stats.kendalltau(dfval['actualrank'], dfval['preds'])

print(tau)

print(stats.spearmanr(dfval['actualrank'], dfval['preds']))

import numpy as np
import matplotlib.pyplot as plt
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

plt.scatter(dfval['actualrank'], dfval['preds'], alpha=0.5)
plt.xlabel("Human ranking score")
plt.ylabel("AI-predicted score")
plt.title("Network Performance on Internal Validation Data")
plt.ylim([0,1])
x = np.linspace(0,1,10)
y = x
plt.plot(x,y,'-r')
plt.show()

## Both have pretty significant discontinuities
# frames = [dftune, df]
# dftraintune = pd.concat(frames)

# plt.hist(dftraintune['preds'],bins=50)
# plt.title("Histogram AI Predictions")
# plt.show()
#
#
# import seaborn as sns
# sns.distplot(dftraintune['preds'])
# plt.title("Distribution of AI Predictions")
# plt.show()
#
# plt.hist(dftraintune['preds'],bins=50)
# plt.title("Histogram AI Predictions")
# plt.show()
#
#
# import seaborn as sns
# sns.distplot(dftraintune['actualrank'])
# plt.title("Distribution of human scores")
# plt.show()
#
# plt.hist(dftraintune['actualrank'],bins=50)
# plt.title("Histogram human scores")
# plt.show()

