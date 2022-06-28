import json
from pathlib import Path
import numpy as np
from Get_rankings_development import split_rankings

i = float(input()) #variable between 0 and 1

split_rankings(i)

ratings_first_half_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/james_ranking_functions/" + str(i) + "ratings_first_half.json")
with open(ratings_first_half_path, 'r') as json_f:
    ratings_first_half = json.load(json_f)

ratings_second_half_path = Path("/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/james_ranking_functions/" + str(i) + "ratings_second_half.json")
with open(ratings_second_half_path, 'r') as json_f:
    ratings_second_half = json.load(json_f)

# # Make array of ratings
# ratings = []
# ids = []
#
# for image in ratings_first_half:
#     ids.append(image['id'])
#     ratings.append(image['rating'])
#
# idratings_first_half = list(zip(ids,ratings))
#
# ratings = []
# ids = []
#
# for image in ratings_second_half:
#     ids.append(image['id'])
#     ratings.append(image['rating'])
#
# idratings_second_half = list(zip(ids, ratings))

    # ratings_one_user = np.array(ratings_one_user)
    # ratings_all_users = np.vstack([ratings_all_users,ratings_one_user])
    # ratings_one_user = []

# Try sorting by rating

def get_rating(ratings):
    return ratings.get('rating')

ratings_first_half.sort(key=get_rating)
ratings_second_half.sort(key=get_rating)

ids_first_half = []

for image in ratings_first_half:
    ids_first_half.append(image['id'])

ids_second_half = []

for image in ratings_second_half:
    ids_second_half.append(image['id'])

import scipy.stats as stats

tau, p_value = stats.kendalltau(ids_first_half, ids_second_half)

print(tau)

print(stats.spearmanr(ids_first_half, ids_second_half))



