import torch
import pandas as pd
import numpy as np
import timm
from pathlib import Path
import json
import imageio
import os
import random
from scipy.stats import stats
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import time
import sklearn
from sklearn.model_selection import GroupShuffleSplit
import scipy.stats as stats

class OCTData(Dataset):

    # actual_image_width = 768

    def __init__(self, image_paths, labels_rank, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.labels_rank = labels_rank

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.image_paths[index]

        ## get the rank to which image is associated
        # [86:-4] to only get the relevant ID bit of path
        rank = self.labels_rank['normalisedranking'][self.labels_rank['id'] == image[-76:-4]].values  ##[0]

        image = imageio.imread(image)

        if self.transform:
            image = self.transform(image)
            # image = TF.adjust_gamma(image, gamma = 0.1 * random.randint(5,15))

        else:  ## for tuning and validation dataset
            convert_tensor = transforms.ToTensor()
            image = convert_tensor(image)

        return image, rank

    ## Load label data from json file

labels_path = Path("/home/anissa/oct/scantensus-imp-coro-seligman-oct-rank-train-export-mar1.json")

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
    ratingdeviation = item['rd']
    ids.append(id)
    rankings.append(ranking)
    volatilities.append(volatility)
    ratingdeviations.append(ratingdeviation)

idranking = pd.DataFrame()
idranking['id'] = ids
idranking['ranking'] = rankings
idranking['volatilities'] = volatilities
idranking['ratingdeviations'] = ratingdeviations
idranking['normalisedrds'] = idranking['ratingdeviations'] / (max(idranking['ranking']) - min(idranking['ranking']))
idranking['normalisedranking'] = (idranking['ranking'] - min(idranking['ranking'])) / (max(idranking['ranking']) - min(idranking['ranking']))

## idranking = dataframe with each flow trace id and corresponding ranking

print(idranking['normalisedranking'].std())

## Load image data
png_dir = "/home/anissa/oct/03"
#png_dir = "/home/anissa/oct/valid_oct"
#png_dir = "/home/anissa/oct/valid_oct2"
#png_dir = "/home/anissa/oct/oct_20011"

all_images = os.listdir(png_dir)
all_images_paths = [os.path.join(png_dir, element) for element in all_images if element.endswith('.png')]
all_images_paths = sorted(all_images_paths, key=lambda x: int(x[-8:-4]))

# print(len(all_images_paths))
#
images_folders = pd.DataFrame()
images_folders['image_paths'] = all_images_paths

folders = []
for path in all_images_paths:
    folder = path[23:27]
    folders.append(folder)

images_folders['folders'] = folders

# split into development and test set (can only do 2)
gs = GroupShuffleSplit(n_splits=2, test_size=.15, random_state=423)
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

gs = GroupShuffleSplit(n_splits=2, test_size=.17647, random_state=423)
train_ix, tune_ix = next(gs.split(dev_images_folders, groups=dev_images_folders.folders))

train_images_paths = []
for i in train_ix:
    train_images_path = dev_images_folders['image_paths'][i]
    train_images_paths.append(train_images_path)

tune_images_paths = []
for i in tune_ix:
    tune_images_path = dev_images_folders['image_paths'][i]
    tune_images_paths.append(tune_images_path)

##separate into train, tune and validate (0.7, 0.15, 0.15) sets - total =1080, so 757,162,162

# seed = 423
# random.Random(seed).shuffle(all_images_paths)
# train_images_paths = all_images_paths[0:757]
# tune_images_paths = all_images_paths[757:919]
# val_images_paths = all_images_paths[919:1080]

# decide what transforms you want
transform = transforms.Compose([  # choose parameters
    transforms.ToTensor(),
    # transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1), scale = (0.66,1.5))])
    transforms.RandomRotation(degrees=180)])

train_data = OCTData(train_images_paths, labels_rank=idranking, transform=transform)
tune_data = OCTData(tune_images_paths, labels_rank=idranking, transform=None)
val_data = OCTData(val_images_paths, labels_rank=idranking, transform=None)
#external_val_data = OCTData(all_images_paths, labels_rank=idranking, transform=None)

num_classes = 1
model = timm.create_model('resnet34', num_classes = num_classes)
model.default_cfg['input_size'] = (3,1024,1024)
model.conv1 = nn.Conv2d(3,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model.init_weights()
#state_dict = load_and_fix_state_dict('oct_model.pt')
model.load_state_dict(torch.load('/home/anissa/oct/oct_model.pt'))
model.eval()
model.cpu()
#
# preds = []
# expert_rank = []
# ids = []
#
# for i in range(len(train_data)):
#     x = model(train_data[i][0][None, ...])
#     preds.append(x[0].item())
#     expert_rank.append(train_data[i][1].item())
#     ids.append(train_data.image_paths[i][-76:-4])
#
# df_preds = pd.DataFrame(preds)
# df_preds['ids'] = ids
# df_preds['expert rank'] = expert_rank
#
# import shutil
# import os
#
# # put reference images in a folder
# folder_path = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/oct/reference OCT'
# image_path = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/oct/03/'
# for a in range(len(reference)):
#     for i in range(11):
#         if reference['0'][a] == i:
#             image = reference['03-82984701116ab7ada2feb0d63405fb23ac24636d63b7209ea92c16d55ab76c1e-0268'][a] + '.png'
#             path = os.path.join(image_path,image)
#             target_path = os.path.join(folder_path,str(i),image)
#             shutil.copy(path, target_path)
#


# inference time is about 0.4 seconds on cpu
# a = time.time()
# preds = []
# for i in range(len(external_val_data)):
#     x = model(external_val_data[i][0][None, ...])
#     b = time.time() - a
#     print(b)
#     a = time.time()
#     preds.append(x[0].item())
#
# df_preds = pd.DataFrame(preds)

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

tau, p_value = stats.kendalltau(df['actualrank'], df['preds'])

print(tau)

print(stats.spearmanr(df['actualrank'], df['preds']))
#
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

print(stats.spearmanr(dftune['actualrank'],dftune['preds']))

preds = []
actual = []
for i in range(len(val_data)):
    x = model(val_data[i][0][None, ...])
    preds.append(x[0].item())
    actual.append(val_data[i][1].item())

dfval = pd.DataFrame()
dfval['preds'] = preds
dfval['actualrank'] = actual
dfval['squarederror'] = (dfval['preds'] - dfval['actualrank']) ** 2

print(stats.spearmanr(dfval['actualrank'], dfval['preds']))
#
plt.scatter(df['actualrank'], df['preds'],alpha=0.5)
plt.xlabel("Human ranking score")
plt.ylabel("AI-predicted score")
plt.title("OCT Training data")
plt.ylim([0,1])
x = np.linspace(0,1,10)
y = x
plt.plot(x,y,'-r')
plt.show()

plt.scatter(dftune['actualrank'], dftune['preds'], alpha=0.5)
plt.xlabel("Human ranking score")
plt.ylabel("AI-predicted score")
plt.title("OCT Tuning data")
plt.plot(x,y,'-r')
plt.show()

plt.scatter(dfval['actualrank'], dfval['preds'], alpha=0.5)
plt.xlabel("Human ranking score")
plt.ylabel("AI-predicted score")
plt.title("OCT Val data")
plt.plot(x,y)
