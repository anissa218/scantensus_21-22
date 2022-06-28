#from utils_copy import create_rateable_pool_from_video_ids, process_unweighted_results, get_users_and_pairs_from_results
from glicko2.utils import create_rateable_pool_from_video_ids, process_unweighted_results, get_users_and_pairs_from_results
import json
from pathlib import Path

## data
#responses_path = Path("jan10_expert_responses.json")
#responses_path = Path("jan24_expert_responses.json")
responses_path = Path("feb17_expert_responses.json")

with open(responses_path, 'r') as json_f:
    responses = json.load(json_f)

config = responses['config']
results = responses['results']

# Get users
users = get_users_and_pairs_from_results(results)
user_list = list(users.keys())

# Filter out the results for each ID
# currently 8 different ids:
# ['z4kmAlTyIydg4tV3eOmeK6U8cCM2',
#  'bmEhwNCZT9Wiftgvsopb7vBjO9o1',
#  'VJDApcT8ehTeWv2llV8QGzHTEWS2',
#  'KP2b8tn6m1aTexnbfoII2OHboe03',
#  'PosPfN1VDhgHfbRQxEKuXFTvtxj1',
#  'fZSEvTemhNgVhCqcMp76IzK1RLv2',
#  'teAZL0rNDyWevenTcbHQTdzCOGi2',
#  'vJd46Vaq1pOCAMW1bwxfe8gFKDd2']

results_0 = []
results_1 = []
results_2 = []
results_3 = []
results_4 = []
results_5 = []
results_6 = []
results_7 = []
results_8 = []
results_9 = []
results_10 = []
results_11 = []

for element in results:
    if element['user'] == user_list[0]:
        results_0.append(element)
    if element['user'] == user_list[1]:
        results_1.append(element)
    if element['user'] == user_list[2]:
        results_2.append(element)
    if element['user'] == user_list[3]:
        results_3.append(element)
    if element['user'] == user_list[4]:
        results_4.append(element)
    if element['user'] == user_list[5]:
        results_5.append(element)
    if element['user'] == user_list[6]:
        results_6.append(element)
    if element['user'] == user_list[7]:
        results_7.append(element)
    if element['user'] == user_list[8]:
        results_8.append(element)
    if element['user'] == user_list[9]:
        results_9.append(element)
    if element['user'] == user_list[10]:
        results_10.append(element)
    if element['user'] == user_list[11]:
        results_11.append(element)

results_per_user = []
results_per_user.append(results_0)
results_per_user.append(results_1)
results_per_user.append(results_2)
results_per_user.append(results_3)
results_per_user.append(results_4)
results_per_user.append(results_5)
results_per_user.append(results_6)
results_per_user.append(results_7)
results_per_user.append(results_8)
results_per_user.append(results_9)
results_per_user.append(results_10)
results_per_user.append(results_11)

# total = len(results_0) + len(results_1) + len(results_2) + len(results_3) + len(results_4) + len(results_5) + len(results_6) + len(results_7)
# total should be equal to len(results)

# Create rating pool and rate
video_ids = config[0]['cases']
rateable_pool = create_rateable_pool_from_video_ids(video_ids)
rateable_pool = process_unweighted_results(rateable_pool, results)
total = rateable_pool

## do the same for individual results
rateable_pool_users = []

for element in results_per_user:
    video_ids = config[0]['cases']
    rateable_pool = create_rateable_pool_from_video_ids(video_ids)
    rateable_pool = process_unweighted_results(rateable_pool, element)
    #rateable_pool = process_unweighted_results(rateable_pool, ('results_'+str(i)))
    rateable_pool_users.append(rateable_pool)

rateable_pool_users.append(total)

ratings_all_users = []
ratings_per_user = []
for i in range(len(users)+1):
    for element in rateable_pool_users[i]:
        ratings_per_user.append({'id':rateable_pool_users[i][element].name,'n_pairs':rateable_pool_users[i][element].n_pairs, 'rating':rateable_pool_users[i][element].rating, 'rd':rateable_pool_users[i][element].rd,'vol':rateable_pool_users[i][element].vol})
    ratings_all_users.append(ratings_per_user)
    ratings_per_user = []

import json
with open('ratings_all_users.json', 'w') as f:
    json.dump(ratings_all_users,f,indent=4)

# rateable_pool_users now has lines 0 to 11 of user ratings, + line 12 of total ratings
