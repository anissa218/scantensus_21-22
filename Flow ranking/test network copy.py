import torch
import timm
from torch import nn
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import imageio
import random
from random import randint
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from scipy.stats import stats

def list_files(dir):
    r = []
    png_dir_list = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    png_dir_list = [element for element in r if element.endswith('.png')]
    return png_dir_list

class RankExternalValData(Dataset):
    actual_image_width = 320
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
        id = image[-76:-4]
        image = imageio.imread(image)
        image_width = image.shape[1]
        center_image = int(image_width /2)
        image = image[:,center_image - 128 : center_image + 128]
        if self.transform:
            image = self.transform(image)
            image = TF.adjust_gamma(image, gamma = 0.1 * random.randint(5,15))
        else: ## for tuning and validation dataset
            convert_tensor = transforms.ToTensor()
            image = convert_tensor(image)
        return image, rank, id

## Load val label data for pooled users from json file
ranking_path = Path("/home/anissa/scantensus-ranking/ratings_all_users.json")
with open(ranking_path, 'r') as json_f:
    ranking_all_users = json.load(json_f)
# Load val data from each individual user from json file

# Separates data into a dictionary with dataframes for each user's data
ids = []
rankings = []
volatilities = []
ratingdeviations = []
n_pairs = []
idranking_dict = {}
i = 0

for user in ranking_all_users:

    idranking_dict[str(i)] = pd.DataFrame()

    for item in user:
        id = item['id']
        ranking = item['rating']
        volatility = item['vol']
        ratingdeviation = item['rd']
        pairs = item['n_pairs']
        ids.append(id)
        rankings.append(ranking)
        volatilities.append(volatility)
        ratingdeviations.append(ratingdeviation)
        n_pairs.append(pairs)

    idranking_dict[str(i)]['id'] = ids
    idranking_dict[str(i)]['ranking'] = rankings
    idranking_dict[str(i)]['volatilities'] = volatilities
    idranking_dict[str(i)]['ratingdeviations'] = ratingdeviations
    idranking_dict[str(i)]['normalisedrds'] = idranking_dict[str(i)]['ratingdeviations'] / (max(idranking_dict[str(i)]['ranking']) - min(idranking_dict[str(i)]['ranking']))
    idranking_dict[str(i)]['normalisedranking'] = (idranking_dict[str(i)]['ranking'] - min(idranking_dict[str(i)]['ranking'])) / (max(idranking_dict[str(i)]['ranking']) - min(idranking_dict[str(i)]['ranking']))
    idranking_dict[str(i)]['n_pairs'] = n_pairs

    ids = []
    rankings = []
    volatilities = []
    ratingdeviations = []
    n_pairs = []

    i = i + 1

## Load image data

external_val_png_dir = "/home/anissa/scantensus-ranking/32"
sapna_png_dir = "/home/anissa/scantensus-ranking/selected_sapna_sdys"
external_val_images_paths = list_files(external_val_png_dir)
sapna_images_paths = list_files(sapna_png_dir)
len(external_val_images_paths)

# labels are currently from pooled users - can change which element of dictionary list is used
external_val_data = RankExternalValData(external_val_images_paths, labels_rank = idranking_dict['12'], transform = None)
#sapna_val_data = RankExternalValData(sapna_images_paths, labels_rank = idranking_dict['0'], transform = None)

num_classes = 1
model = timm.create_model('resnet34', num_classes = num_classes)

model.default_cfg['input_size'] = (1,512,256)

model.conv1 = nn.Conv2d(1,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#or flow model 3??
model.load_state_dict(torch.load('/home/anissa/scantensus-ranking/final_flow_quality_model.pt'))
model.eval()

preds = []
actual = []
id = []

for i in range(len(external_val_data)):
    x = model(external_val_data[i][0][None, ...])
    preds.append(x[0].item())
    actual.append(external_val_data[i][1].item())
    ids.append(external_val_data[i][2])

print(model(external_val_data[15][0][None, ...]))
print(model(external_val_data[15][0][None, ...]))
print(model(external_val_data[15][0][None, ...]))

df_external_val = pd.DataFrame()
df_external_val['preds'] = preds
df_external_val['actualrank'] = actual
df_external_val['squarederror'] = (df_external_val['preds'] - df_external_val['actualrank']) ** 2
df_external_val['id'] = ids

meansquarederror = df_external_val['squarederror'].mean()

df_external_val["Rank"] = df_external_val["preds"].rank()
df_external_val["Expert Rank"] = df_external_val["actualrank"].rank()

#
# preds = []
# actual = []
# ids = []
#
# for i in range(len(sapna_val_data)):
#     x = model(sapna_val_data[i][0][None, ...])
#     preds.append(x[0].item())
#     #actual.append(sapna_val_data[i][1].item())
#     ids.append(sapna_val_data[i][2])
#
# df_sapna_val = pd.DataFrame()
# df_sapna_val['preds'] = preds
# #df_sapna_val['actualrank'] = actual
# #df_sapna_val['squarederror'] = (df_sapna_val['preds'] - df_sapna_val['actualrank']) ** 2
# df_sapna_val['id'] = ids
#
# #meansquarederror = df_sapna_val['squarederror'].mean()
#
# df_sapna_val.to_csv('sapna_val.csv')

import scipy.stats as stats


print(stats.spearmanr(df_external_val['actualrank'], df_external_val['preds']))

# Graph of network prediction vs human ranking score
plt.scatter(df_external_val['Rank'], df_external_val['Expert Rank'],alpha=0.5)
plt.xlabel("Human ranking")
plt.ylabel("AI-predicted ranking")
plt.title("Network performance on External Validation Data")
#plt.ylim([0,1])
x = np.linspace(0,200,10)
y = x
plt.plot(x,y,'-r')
plt.show()

# preds = []
#
# for i in range(len(sapna_val_data)):
#     x = model(sapna_val_data[i][0][None, ...])
#     preds.append(x[0].item())
#
# df_sapna_val = pd.DataFrame()
# df_sapna_val['preds'] = preds


# plt.hist(df_sapna_val['preds'],bins=50)
# plt.title("Histogram AI Predictions")
# plt.xlim([0, 1])
# plt.show()

#import seaborn as sns
# sns.displot(df_sapna_val['preds'])
# plt.title("Distribution of AI Predictions on 30 Tracking Traces")
# plt.xlim([0, 1])
# plt.show()
#
# sns.displot(df_external_val['preds'], kind = "kde")
# plt.xlabel("AI Predictions on Validation Images")
# #plt.title("Distribution of AI Predictions on 200 Expert Validation Traces")
# plt.xlim([0, 1])
# plt.show()
#
# sns.displot(df_external_val['actualrank'], kind = "kde")
# plt.xlabel("Expert scores on Validation Images")
# #plt.title("Distribution of AI Predictions on 200 Expert Validation Traces")
# plt.xlim([0, 1])
# plt.show()
#
for i in range(13):
    idranking_dict[str(i)]["Expert Rank"] = idranking_dict[str(i)]["ranking"].rank()
sorted_external_val = df_external_val.sort_values("id")

sorted_idranking_dict = {}

for i in range(13):
    sorted_df = idranking_dict[str(i)].sort_values("id")
    sorted_idranking_dict[str(i)] = sorted_df


#num of pairwise comparisons per expert
comparisons = []
for i in range(13):
    x = sorted_idranking_dict[str(i)]['n_pairs'].sum()
    comparisons.append(x)

# Plot of rankings, each user and AI vs expert consensus

# change legend so it says experts instead of user, and maybe no need to have all of them (ie 9, 10 and 11)

for i in range(13):
    plt.scatter(sorted_idranking_dict[str(i)]['Expert Rank'], sorted_idranking_dict['12']['Expert Rank'], alpha=0.5, label = "individual user")
plt.scatter(sorted_external_val['Rank'], sorted_idranking_dict['12']['Expert Rank'], marker = "+", color = "black", label= "AI")
plt.xlabel("User / AI ranking")
plt.ylabel("Expert consensus ranking")
plt.legend()
plt.title("External validation data")
# plt.ylim([0,1])
x = np.linspace(0, 200, 10)
y = x
plt.plot(x, y, '-r')
plt.show()

sorted_external_val['squared error'] = (sorted_external_val['Rank'] - sorted_external_val['Expert Rank']) **2

for i in range(200):
    if sorted_external_val['squared error'][i] > 2500:
        print(sorted_external_val['id'][i])
        print(i)
        print(sorted_external_val['Rank'][i])
        print(sorted_external_val['Expert Rank'][i])

sorted_ranks = pd.DataFrame()
sorted_ranks['ID'] = sorted_external_val['id'].values
sorted_ranks['Consensus Rank'] = sorted_idranking_dict['12']['Expert Rank'].values
sorted_ranks['AI Rank'] = sorted_external_val['Rank'].values

for i in range(12):
    sorted_ranks[str(i)] = sorted_idranking_dict[str(i)]['Expert Rank'].values

final_sorted_ranks = sorted_ranks.sort_values("Consensus Rank")

# zoom into graph of consensus vs each user/AI so you can clearly se range
for i in range(12):
    plt.scatter(sorted_idranking_dict['12']['Expert Rank'], sorted_idranking_dict[str(i)]['Expert Rank'], s=10, alpha=0.25, label = "individual user")
plt.scatter(sorted_idranking_dict['12']['Expert Rank'],sorted_external_val['Rank'], marker = "+", s=30, color = "red", label= "AI")
plt.xlabel("User / AI ranking")
plt.ylabel("Expert consensus ranking")
plt.legend()
plt.title("External validation data")
# plt.ylim([0,1])
x = np.linspace(0, 200, 10)
y = x
#plt.plot(x, y, '-r')
plt.xlim([0,30])
plt.show()

# User/AI vs consensus but only for users who have done a reasonable number of comparisons
plt.scatter(sorted_idranking_dict['12']['Expert Rank'], sorted_idranking_dict['0']['Expert Rank'], s=10, alpha=0.25, label = "Individual Experts")
for i in range(1,9):
    plt.scatter(sorted_idranking_dict['12']['Expert Rank'], sorted_idranking_dict[str(i)]['Expert Rank'], s=10, alpha=0.25, label = "_Individual Expert")
plt.scatter(sorted_idranking_dict['12']['Expert Rank'],sorted_external_val['Rank'], marker = "+", s=30, color = "red", label= "AI")
plt.xlabel("Expert consensus ranking")
plt.ylabel("User / AI ranking")
plt.legend(prop={'size': 8})
plt.title("Expert and AI ranking of external validation data compared to consensus")
# plt.ylim([0,1])
x = np.linspace(0, 200, 10)
y = x
#plt.plot(x, y, '-r')
plt.show()

# Graph with only best and worst expert
plt.scatter(sorted_idranking_dict['7']['Expert Rank'], sorted_idranking_dict['12']['Expert Rank'],
            s=10, alpha=0.25, label="Expert 7")
plt.scatter(sorted_idranking_dict['3']['Expert Rank'], sorted_idranking_dict['12']['Expert Rank'],
            s=10, alpha=0.25, label="Expert 3")
plt.scatter(sorted_external_val['Rank'], sorted_idranking_dict['12']['Expert Rank'], marker = "+", color = "red", label= "AI")
plt.xlabel("User / AI ranking")
plt.ylabel("Expert consensus ranking")
plt.legend()
plt.title("Expert and AI ranking of external validation data compared to consensus")
plt.show()

# # DO the same with the new json file, where each user is actually the results of every other user minus that user
#
# import json
# from pathlib import Path
# import pandas as pd
# import scipy.stats as stats
# import numpy as np
#
# ranking_path = Path("/home/anissa/scantensus-ranking/ratings_excluded_users.json")
# with open(ranking_path, 'r') as json_f:
#     ranking_all_users = json.load(json_f)
#
# # Separates data into a dictionary with dataframes for each user's data
# ids = []
# rankings = []
# volatilities = []
# ratingdeviations = []
# n_pairs = []
# exc_idranking_dict = {}
# i = 0
#
# for user in ranking_all_users:
#     exc_idranking_dict[str(i)] = pd.DataFrame()
#     for item in user:
#         id = item['id']
#         ranking = item['rating']
#         volatility = item['vol']
#         ratingdeviation = item['rd']
#         pairs = item['n_pairs']
#         ids.append(id)
#         rankings.append(ranking)
#         volatilities.append(volatility)
#         ratingdeviations.append(ratingdeviation)
#         n_pairs.append(pairs)
#     exc_idranking_dict[str(i)]['id'] = ids
#     exc_idranking_dict[str(i)]['ranking'] = rankings
#     exc_idranking_dict[str(i)]['volatilities'] = volatilities
#     exc_idranking_dict[str(i)]['ratingdeviations'] = ratingdeviations
#     exc_idranking_dict[str(i)]['normalisedrds'] = exc_idranking_dict[str(i)]['ratingdeviations'] / (max(exc_idranking_dict[str(i)]['ranking']) - min(exc_idranking_dict[str(i)]['ranking']))
#     exc_idranking_dict[str(i)]['normalisedranking'] = (exc_idranking_dict[str(i)]['ranking'] - min(exc_idranking_dict[str(i)]['ranking'])) / (max(exc_idranking_dict[str(i)]['ranking']) - min(exc_idranking_dict[str(i)]['ranking']))
#     exc_idranking_dict[str(i)]['n_pairs'] = n_pairs
#     ids = []
#     rankings = []
#     volatilities = []
#     ratingdeviations = []
#     n_pairs = []
#     i = i + 1
#
# for i in range(13):
#     exc_idranking_dict[str(i)]["Expert Rank"] = exc_idranking_dict[str(i)]["ranking"].rank()
#
# exc_sorted_idranking_dict = {}
#
# for i in range(13):
#     exc_sorted_df = exc_idranking_dict[str(i)].sort_values("id")
#     exc_sorted_idranking_dict[str(i)] = exc_sorted_df
#
# import scipy.stats as stats
# import pandas as pd
#
# x=0
# y=0
# correlation = []
# correlation_df = pd.DataFrame()
#
# # first correlation between each 12 users and each other
# for a in range(12):
#     for i in range(12):
#         x,y = stats.spearmanr(sorted_idranking_dict[str(a)]['Expert Rank'],sorted_idranking_dict[str(i)]['Expert Rank'])
#         correlation.append(x)
#     correlation_df[str(a)] = correlation
#     correlation = []
#
# # then correlation between each 12 users and the consensus
# Expert_correlation = []
# for i in range(12):
#     # excluded idranking dataframe is the consensus minus the said user
#     x,y = stats.spearmanr(sorted_idranking_dict[str(i)]['Expert Rank'],exc_sorted_idranking_dict[str(i)]['Expert Rank'])
#     Expert_correlation.append(x)
# correlation_df['correlation w expert consensus minus each expert'] = Expert_correlation
#
# AI_correlation = []
# for i in range(12):
#     x,y = stats.spearmanr(sorted_external_val['Rank'],sorted_idranking_dict[str(i)]['Expert Rank'])
#     AI_correlation.append(x)
#
# correlation_df['AI'] = AI_correlation

