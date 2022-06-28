import json
from pathlib import Path
import pandas as pd
import scipy.stats as stats
import numpy as np

ranking_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/james_ranking_functions/ratings_all_users.json")
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

# # Find correlation coefficient for each user compared to pooled users
correlation_users = pd.DataFrame()
spearman_coefs = []
sum_comparisons = []
spearman_coef = 0
pairs = 0

for user in idranking_dict:
    # check that 10 is the pooled users
    spearman_coef,x = stats.spearmanr(idranking_dict[user]['normalisedranking'], idranking_dict['10']['normalisedranking'])
    spearman_coefs.append(spearman_coef)
    pairs = (idranking_dict[user]['n_pairs'].sum())/2 # need to divide by 2 to to get number of paired comparisons
    sum_comparisons.append(pairs)
    pairs = 0
    spearman_coef = 0


correlation_users['spearman_coefs'] = spearman_coefs
correlation_users['pairwise_comparisons'] = sum_comparisons

# Make dataframe for internal experts
internal_correlation_users = pd.DataFrame()
internal_correlation_users['spearman_coefs'] = [spearman_coefs[index] for index in [2,3,5,6,7,8,9]]
internal_correlation_users['pairwise_comparisons'] = [sum_comparisons[index] for index in [2,3,5,6,7,8,9]]


# Make dataframe for external experts
external_correlation_users = pd.DataFrame()
external_spearman_coefs = [spearman_coefs[idx] for idx in [0,1,4]]
external_correlation_users['spearman_coefs'] = external_spearman_coefs
external_comparisons = [sum_comparisons[idx] for idx in [0,1,4]]
external_correlation_users['pairwise_comparisons'] = external_comparisons

# # add external consensus - NOT THE RIGHT WAY TO DO IT, NEED TO GET RATING FOR ALL EXPERTS COMBINED
# external_correlation_users_multiplication = external_correlation_users['spearman_coefs']*external_correlation_users['pairwise_comparisons']
# external_spearman = (external_correlation_users_multiplication.sum())/(external_correlation_users['pairwise_comparisons'].sum())
# external_spearman_coefs.append(external_spearman)
#
# external_comparison = external_correlation_users['pairwise_comparisons'].sum()
# external_comparisons.append(external_comparison)
#
# external_correlation_users = pd.DataFrame()
# external_correlation_users['spearman_coefs'] = external_spearman_coefs
# external_correlation_users['pairwise_comparisons'] = external_comparisons

# Plot normalised RD for each user

import matplotlib.pyplot as plt
plt.scatter(internal_correlation_users['pairwise_comparisons'], internal_correlation_users['spearman_coefs'],alpha=0.5, color = "red")
plt.scatter(external_correlation_users['pairwise_comparisons'], external_correlation_users['spearman_coefs'],alpha=0.5, color = "blue")
plt.xlabel("Paired Comparisons",fontsize=14)
plt.ylim(0.9,1.0)
plt.ylabel("Spearman Correlation Coefficient",fontsize=14)
plt.title("Spearman Correlation Coefficient for each user compared to consensus")
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

split_correlation_users = pd.DataFrame()
split_spearman_coefs = []
split_sum_comparisons = []
split_spearman_coef = 0
split_pairs = 0

for comparison in split_idranking_dict:
    split_spearman_coef,x = stats.spearmanr(split_idranking_dict[comparison]['normalisedranking'],split_idranking_dict['8']['normalisedranking'])
    split_spearman_coefs.append(split_spearman_coef)
    split_pairs = (split_idranking_dict[comparison]['n_pairs'].sum())/2 # need to divide by 2 to to get number of paired comparisons
    split_sum_comparisons.append(split_pairs)
    split_pairs = 0
    split_spearman_coef = 0

split_correlation_users['spearman_coefs'] = split_spearman_coefs
split_correlation_users['pairwise_comparisons'] = split_sum_comparisons

# Plot normalised RD for pooled users as total comparisons increase

import matplotlib.pyplot as plt
plt.scatter(split_correlation_users['pairwise_comparisons'][0:7], split_correlation_users['spearman_coefs'][0:7],alpha=0.5)
plt.xlabel("Paired Comparisons",fontsize=14)
plt.ylabel("Spearman Correlation Coefficient",fontsize=14)

# not sure what way is best to plot line of best fit
z = np.polyfit(split_correlation_users['pairwise_comparisons'][0:7], split_correlation_users['spearman_coefs'][0:7], 3)
f = np.poly1d(z)
a = np.linspace(3300,26000,num = 1000)
plt.plot(a,f(a),color="blue")
plt.title("Correlation coefficients for pooled users as comparisons increase")
plt.show()

# ANALYZE ONE USER

import json
from pathlib import Path
import numpy as np
import pandas as pd

# Separates data into a dictionary with dataframes for each user's data

all_users_split_correlation = []

for a in range(0,10):
    ids = []
    rankings = []
    volatilities = []
    ratingdeviations = []
    n_pairs = []
    user_split_idranking_dict = {}

    i = 0

    for i in range(1,9):

        ranking_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/james_ranking_functions/user" + str(a) + "_split_ratings" + str(i) + ".json")

        with open(ranking_path, 'r') as json_f:
            ranking_all_users = json.load(json_f)

        user_split_idranking_dict[str(i)] = pd.DataFrame()

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

        user_split_idranking_dict[str(i)]['id'] = ids
        user_split_idranking_dict[str(i)]['ranking'] = rankings
        user_split_idranking_dict[str(i)]['volatilities'] = volatilities
        user_split_idranking_dict[str(i)]['ratingdeviations'] = ratingdeviations
        user_split_idranking_dict[str(i)]['normalisedrds'] = user_split_idranking_dict[str(i)]['ratingdeviations'] / (max(user_split_idranking_dict[str(i)]['ranking']) - min(user_split_idranking_dict[str(i)]['ranking']))
        user_split_idranking_dict[str(i)]['normalisedranking'] = (user_split_idranking_dict[str(i)]['ranking'] - min(user_split_idranking_dict[str(i)]['ranking'])) / (max(user_split_idranking_dict[str(i)]['ranking']) - min(user_split_idranking_dict[str(i)]['ranking']))
        user_split_idranking_dict[str(i)]['n_pairs'] = n_pairs

        ids = []
        rankings = []
        volatilities = []
        ratingdeviations = []
        n_pairs = []

        i = i + 1

    # # Find mean rating deviation for each user
    user_split_correlation_users = pd.DataFrame()
    user_split_spearman_coefs = []
    user_split_sum_comparisons = []
    user_split_spearman_coef = 0
    user_split_pairs = 0

    for user in user_split_idranking_dict:
        # get correlation to final consensus instead of individual consensus
        user_split_spearman_coef, x = stats.spearmanr(user_split_idranking_dict[user]['normalisedranking'],split_idranking_dict['8']['normalisedranking'])
        # user_split_spearman_coef,x = stats.spearmanr(user_split_idranking_dict[user]['normalisedranking'],user_split_idranking_dict['8']['normalisedranking'])
        user_split_spearman_coefs.append(user_split_spearman_coef)
        user_split_pairs = (user_split_idranking_dict[user]['n_pairs'].sum())/2 # need to divide by 2 to to get number of paired comparisons
        user_split_sum_comparisons.append(user_split_pairs)
        user_split_pairs = 0
        user_split_spearman_coef = 0

    user_split_correlation_users['spearman_coefs'] = user_split_spearman_coefs
    user_split_correlation_users['pairwise_comparisons'] = user_split_sum_comparisons

    all_users_split_correlation.append(user_split_correlation_users)

# Plot normalised RD for each user as total comparisons increase

import matplotlib.pyplot as plt
for i in range(0,10):
    if all_users_split_correlation[i]['pairwise_comparisons'][7] > 1000:
        plt.scatter(all_users_split_correlation[i]['pairwise_comparisons'], all_users_split_correlation[i]['spearman_coefs'])
plt.xlabel("Paired Comparisons",fontsize=14)
plt.ylabel("Spearman Correlation Coefficient",fontsize=14)
plt.ylim(0.3,1.0)
plt.title("Correlation between each user's ratings and pooled consensus")
plt.show()

# for this plot above, does it make more sense to do correlation to full

## alpha=0.5

# RETRY POOLED CORRELATION PLOT, BUT SPLITTING DATA IN HALF

# will split data in half
# one half = ranking consensus
# other half will split into sets of increasing sizes, and plot correlation with consensus

# consensus half

import json
from pathlib import Path
import pandas as pd
import scipy.stats as stats
import numpy as np

ranking_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/james_ranking_functions/first_half_split_ratings.json")
with open(ranking_path, 'r') as json_f:
    ranking_all_users = json.load(json_f)

# Separates data into a dictionary with dataframes for each user's data
ids = []
rankings = []
volatilities = []
ratingdeviations = []
n_pairs = []

half_idranking_dict = pd.DataFrame()

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

half_idranking_dict['id'] = ids
half_idranking_dict['ranking'] = rankings
half_idranking_dict['volatilities'] = volatilities
half_idranking_dict['ratingdeviations'] = ratingdeviations
half_idranking_dict['normalisedrds'] = half_idranking_dict['ratingdeviations'] / (max(half_idranking_dict['ranking']) - min(half_idranking_dict['ranking']))
half_idranking_dict['normalisedranking'] = (half_idranking_dict['ranking'] - min(half_idranking_dict['ranking'])) / (max(half_idranking_dict['ranking']) - min(half_idranking_dict['ranking']))
half_idranking_dict['n_pairs'] = n_pairs


# Separates data into a dictionary with dataframes for each user's data
ids = []
rankings = []
volatilities = []
ratingdeviations = []
n_pairs = []
half_split_idranking_dict = {}
i = 0

for i in range(1,9):

    ranking_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/james_ranking_functions/half_split_ratings" + str(i) + ".json")

    with open(ranking_path, 'r') as json_f:
        ranking_all_users = json.load(json_f)

    half_split_idranking_dict[str(i)] = pd.DataFrame()

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

    half_split_idranking_dict[str(i)]['id'] = ids
    half_split_idranking_dict[str(i)]['ranking'] = rankings
    half_split_idranking_dict[str(i)]['volatilities'] = volatilities
    half_split_idranking_dict[str(i)]['ratingdeviations'] = ratingdeviations
    half_split_idranking_dict[str(i)]['normalisedrds'] = half_split_idranking_dict[str(i)]['ratingdeviations'] / (max(half_split_idranking_dict[str(i)]['ranking']) - min(half_split_idranking_dict[str(i)]['ranking']))
    half_split_idranking_dict[str(i)]['normalisedranking'] = (half_split_idranking_dict[str(i)]['ranking'] - min(half_split_idranking_dict[str(i)]['ranking'])) / (max(half_split_idranking_dict[str(i)]['ranking']) - min(half_split_idranking_dict[str(i)]['ranking']))
    half_split_idranking_dict[str(i)]['n_pairs'] = n_pairs

    ids = []
    rankings = []
    volatilities = []
    ratingdeviations = []
    n_pairs = []

    i = i + 1

# # Find mean rating deviation for each user

half_split_correlation_users = pd.DataFrame()
half_split_spearman_coefs = []
half_split_sum_comparisons = []
half_split_spearman_coef = 0
half_split_pairs = 0

for comparison in half_split_idranking_dict:
    half_split_spearman_coef,x = stats.spearmanr(half_split_idranking_dict[comparison]['normalisedranking'],half_idranking_dict['normalisedranking'])
    half_split_spearman_coefs.append(half_split_spearman_coef)
    half_split_pairs = (half_split_idranking_dict[comparison]['n_pairs'].sum())/2 # need to divide by 2 to to get number of paired comparisons
    half_split_sum_comparisons.append(half_split_pairs)
    half_split_pairs = 0
    half_split_spearman_coef = 0

half_split_correlation_users['spearman_coefs'] = half_split_spearman_coefs
half_split_correlation_users['pairwise_comparisons'] = half_split_sum_comparisons

# Plot normalised RD for pooled users as total comparisons increase

import matplotlib.pyplot as plt
plt.scatter(half_split_correlation_users['pairwise_comparisons'][0:8], half_split_correlation_users['spearman_coefs'][0:8],alpha=0.5)
plt.xlabel("Paired Comparisons",fontsize=14)
plt.ylabel("Spearman Correlation Coefficient",fontsize=14)

# not sure what way is best to plot line of best fit
z = np.polyfit(half_split_correlation_users['pairwise_comparisons'][0:8], half_split_correlation_users['spearman_coefs'][0:8], 4)
f = np.poly1d(z)
a = np.linspace(1800,16000,num = 1000)
plt.plot(a,f(a),color="blue")
plt.title("Correlation of half the rankings as comparisons increase with the other half")
plt.show()


# Try the same but looking at correlation with