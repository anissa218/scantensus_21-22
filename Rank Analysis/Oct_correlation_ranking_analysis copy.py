import json
from pathlib import Path
import pandas as pd
import scipy.stats as stats
import numpy as np

# Separates data into a dictionary with dataframes for each user's data
ids = []
rankings = []
volatilities = []
ratingdeviations = []
n_pairs = []
split_idranking_dict = {}
i = 0

for i in range(1,9):

    ranking_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/james_ranking_functions/oct_split_ratings" + str(i) + ".json")

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
plt.scatter(split_correlation_users['pairwise_comparisons'], split_correlation_users['spearman_coefs'],alpha=0.5)
plt.xlabel("Paired Comparisons",fontsize=14)
plt.ylabel("Spearman Correlation Coefficient",fontsize=14)

# not sure what way is best to plot line of best fit
z = np.polyfit(split_correlation_users['pairwise_comparisons'], split_correlation_users['spearman_coefs'], 3)
f = np.poly1d(z)
a = np.linspace(2500,25000,num = 1000)
plt.plot(a,f(a),color="blue")
plt.title("Correlation coefficients for pooled users as comparisons increase")
plt.show()


import json
from pathlib import Path
import numpy as np
import pandas as pd

# Separates data into a dictionary with dataframes for each user's data
#
# all_users_split_correlation = []
#
# for a in range(0,4):
#     ids = []
#     rankings = []
#     volatilities = []
#     ratingdeviations = []
#     n_pairs = []
#     user_split_idranking_dict = {}
#
#     i = 0
#
#     for i in range(1,9):
#
#         ranking_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/james_ranking_functions/oct_user" + str(a) + "_split_ratings" + str(i) + ".json")
#
#         with open(ranking_path, 'r') as json_f:
#             ranking_all_users = json.load(json_f)
#
#         user_split_idranking_dict[str(i)] = pd.DataFrame()
#
#         for item in ranking_all_users:
#             id = item['id']
#             ranking = item['rating']
#             volatility = item['vol']
#             ratingdeviation = item['rd']
#             pairs = item['n_pairs']
#             ids.append(id)
#             rankings.append(ranking)
#             volatilities.append(volatility)
#             ratingdeviations.append(ratingdeviation)
#             n_pairs.append(pairs)
#
#         user_split_idranking_dict[str(i)]['id'] = ids
#         user_split_idranking_dict[str(i)]['ranking'] = rankings
#         user_split_idranking_dict[str(i)]['volatilities'] = volatilities
#         user_split_idranking_dict[str(i)]['ratingdeviations'] = ratingdeviations
#         user_split_idranking_dict[str(i)]['normalisedrds'] = user_split_idranking_dict[str(i)]['ratingdeviations'] / (max(user_split_idranking_dict[str(i)]['ranking']) - min(user_split_idranking_dict[str(i)]['ranking']))
#         user_split_idranking_dict[str(i)]['normalisedranking'] = (user_split_idranking_dict[str(i)]['ranking'] - min(user_split_idranking_dict[str(i)]['ranking'])) / (max(user_split_idranking_dict[str(i)]['ranking']) - min(user_split_idranking_dict[str(i)]['ranking']))
#         user_split_idranking_dict[str(i)]['n_pairs'] = n_pairs
#
#         ids = []
#         rankings = []
#         volatilities = []
#         ratingdeviations = []
#         n_pairs = []
#
#         i = i + 1
#
#     # # Find mean rating deviation for each user
#     user_split_correlation_users = pd.DataFrame()
#     user_split_spearman_coefs = []
#     user_split_sum_comparisons = []
#     user_split_spearman_coef = 0
#     user_split_pairs = 0
#
#     for user in user_split_idranking_dict:
#         # get correlation to final consensus instead of individual consensus
#         user_split_spearman_coef, x = stats.spearmanr(user_split_idranking_dict[user]['normalisedranking'],split_idranking_dict['8']['normalisedranking'])
#         # user_split_spearman_coef,x = stats.spearmanr(user_split_idranking_dict[user]['normalisedranking'],user_split_idranking_dict['8']['normalisedranking'])
#         user_split_spearman_coefs.append(user_split_spearman_coef)
#         user_split_pairs = (user_split_idranking_dict[user]['n_pairs'].sum())/2 # need to divide by 2 to to get number of paired comparisons
#         user_split_sum_comparisons.append(user_split_pairs)
#         user_split_pairs = 0
#         user_split_spearman_coef = 0
#
#     user_split_correlation_users['spearman_coefs'] = user_split_spearman_coefs
#     user_split_correlation_users['pairwise_comparisons'] = user_split_sum_comparisons
#
#     all_users_split_correlation.append(user_split_correlation_users)
#
# # Plot normalised RD for each user as total comparisons increase
#
# import matplotlib.pyplot as plt
# for i in range(0,4):
#     if all_users_split_correlation[i]['pairwise_comparisons'][7] > 1000:
#         plt.scatter(all_users_split_correlation[i]['pairwise_comparisons'], all_users_split_correlation[i]['spearman_coefs'])
# plt.xlabel("Paired Comparisons",fontsize=14)
# plt.ylabel("Spearman Correlation Coefficient",fontsize=14)
# plt.ylim(0.3,1.0)
# plt.title("Correlation coefficient for each user compared to pooled consensus")
# plt.show()
