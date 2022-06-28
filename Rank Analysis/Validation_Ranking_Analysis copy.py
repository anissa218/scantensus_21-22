import json
from pathlib import Path
import pandas as pd
import scipy.stats as stats

ranking_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/scantensus-ranking/ratings_all_users.json")
with open(ranking_path, 'r') as json_f:
    ranking_all_users = json.load(json_f)

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

# # Find mean rating deviation for each user
rd_comparisons = pd.DataFrame()
mean_norm_rd = []
sum_comparisons = []
rd_mean = 0
pairs = 0

for user in idranking_dict:
    rd_mean = idranking_dict[user]['normalisedrds'].mean()
    mean_norm_rd.append(rd_mean)
    pairs = (idranking_dict[user]['n_pairs'].sum())/2 # need to divide by 2 to to get number of paired comparisons
    sum_comparisons.append(pairs)
    pairs = 0
    rd_mean = 0


rd_comparisons['mean_norm_rd'] = mean_norm_rd
rd_comparisons['pairwise_comparisons'] = sum_comparisons

# Plot normalised RD for each user

import matplotlib.pyplot as plt
plt.scatter(rd_comparisons['pairwise_comparisons'], rd_comparisons['mean_norm_rd'])
plt.xlabel("Paired Comparisons",fontsize=14)
plt.ylabel("Ranking Deviation",fontsize=14)
plt.ylim(0,0.125)
plt.title("Ranking deviation for each user")
plt.show()

import json
from pathlib import Path
import numpy as np
import pandas as pd

# Separates data into a dictionary with dataframes for each user's data
ids = []
rankings = []
volatilities = []
ratingdeviations = []
n_pairs = []
split_idranking_dict = {}
i = 0

for i in range(1,9):

    ranking_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/james_ranking_functions/split_ratings" + str(i) + ".json")

    with open(ranking_path, 'r') as json_f:
        ranking_all_users = json.load(json_f)

    split_idranking_dict[str(i)] = pd.DataFrame()

    for item in ranking_all_users:
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

    split_idranking_dict[str(i)]['id'] = ids
    split_idranking_dict[str(i)]['ranking'] = rankings
    split_idranking_dict[str(i)]['volatilities'] = volatilities
    split_idranking_dict[str(i)]['ratingdeviations'] = ratingdeviations
    split_idranking_dict[str(i)]['normalisedrds'] = split_idranking_dict[str(i)]['ratingdeviations'] / (max(split_idranking_dict[str(i)]['ranking']) - min(split_idranking_dict[str(i)]['ranking']))
    split_idranking_dict[str(i)]['normalisedranking'] = (split_idranking_dict[str(i)]['ranking'] - min(split_idranking_dict[str(i)]['ranking'])) / (max(split_idranking_dict[str(i)]['ranking']) - min(split_idranking_dict[str(i)]['ranking']))
    split_idranking_dict[str(i)]['n_pairs'] = n_pairs

    ids = []
    rankings = []
    volatilities = []
    ratingdeviations = []
    n_pairs = []

    i = i + 1

# # Find mean rating deviation for each user
split_rd_comparisons = pd.DataFrame()
split_mean_norm_rd = []
split_sum_comparisons = []
split_rd_mean = 0
split_pairs = 0

for user in split_idranking_dict:
    split_rd_mean = split_idranking_dict[user]['normalisedrds'].mean()
    split_mean_norm_rd.append(split_rd_mean)
    split_pairs = (split_idranking_dict[user]['n_pairs'].sum())/2 # need to divide by 2 to to get number of paired comparisons
    split_sum_comparisons.append(split_pairs)
    split_pairs = 0
    split_rd_mean = 0


split_rd_comparisons['mean_norm_rd'] = split_mean_norm_rd
split_rd_comparisons['pairwise_comparisons'] = split_sum_comparisons

# Plot normalised RD for pooled users as total comparisons increase

import matplotlib.pyplot as plt
plt.scatter(split_rd_comparisons['pairwise_comparisons'], split_rd_comparisons['mean_norm_rd'])
plt.xlabel("Paired Comparisons",fontsize=14)
plt.ylabel("Ranking Deviation",fontsize=14)
#plt.ylim(0,0.125)
plt.title("Pooled users")
plt.show()

# ANALYZE ONE USER

import json
from pathlib import Path
import numpy as np
import pandas as pd

# Separates data into a dictionary with dataframes for each user's data
ids = []
rankings = []
volatilities = []
ratingdeviations = []
n_pairs = []
user0_split_idranking_dict = {}
i = 0

for i in range(1,9):

    ranking_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/james_ranking_functions/user0_split_ratings" + str(i) + ".json")

    with open(ranking_path, 'r') as json_f:
        ranking_all_users = json.load(json_f)

    user0_split_idranking_dict[str(i)] = pd.DataFrame()

    for item in ranking_all_users:
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

    user0_split_idranking_dict[str(i)]['id'] = ids
    user0_split_idranking_dict[str(i)]['ranking'] = rankings
    user0_split_idranking_dict[str(i)]['volatilities'] = volatilities
    user0_split_idranking_dict[str(i)]['ratingdeviations'] = ratingdeviations
    user0_split_idranking_dict[str(i)]['normalisedrds'] = user0_split_idranking_dict[str(i)]['ratingdeviations'] / (max(user0_split_idranking_dict[str(i)]['ranking']) - min(user0_split_idranking_dict[str(i)]['ranking']))
    user0_split_idranking_dict[str(i)]['normalisedranking'] = (user0_split_idranking_dict[str(i)]['ranking'] - min(user0_split_idranking_dict[str(i)]['ranking'])) / (max(user0_split_idranking_dict[str(i)]['ranking']) - min(user0_split_idranking_dict[str(i)]['ranking']))
    user0_split_idranking_dict[str(i)]['n_pairs'] = n_pairs

    ids = []
    rankings = []
    volatilities = []
    ratingdeviations = []
    n_pairs = []

    i = i + 1

# # Find mean rating deviation for each user
user0_split_rd_comparisons = pd.DataFrame()
user0_split_mean_norm_rd = []
user0_split_sum_comparisons = []
user0_split_rd_mean = 0
user0_split_pairs = 0

for user in user0_split_idranking_dict:
    user0_split_rd_mean = user0_split_idranking_dict[user]['normalisedrds'].mean()
    user0_split_mean_norm_rd.append(user0_split_rd_mean)
    user0_split_pairs = (user0_split_idranking_dict[user]['n_pairs'].sum())/2 # need to divide by 2 to to get number of paired comparisons
    user0_split_sum_comparisons.append(user0_split_pairs)
    user0_split_pairs = 0
    user0_split_rd_mean = 0


user0_split_rd_comparisons['mean_norm_rd'] = user0_split_mean_norm_rd
user0_split_rd_comparisons['pairwise_comparisons'] = user0_split_sum_comparisons

# Plot normalised RD for each user as total comparisons increase

import matplotlib.pyplot as plt
plt.scatter(user0_split_rd_comparisons['pairwise_comparisons'], user0_split_rd_comparisons['mean_norm_rd'])
plt.xlabel("Paired Comparisons",fontsize=14)
plt.ylabel("Ranking Deviation",fontsize=14)
#plt.ylim(0,0.125)
plt.title("User 0")
plt.show()

