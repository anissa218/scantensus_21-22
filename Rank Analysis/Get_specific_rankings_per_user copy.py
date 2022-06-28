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
# currently 10 different ids:
#['HP0p4yvanwdQboCm1gjSBOTqvwl2', 'ivSEiVd6yEMmu7qEgfzTkqPs38F3', 'bmEhwNCZT9Wiftgvsopb7vBjO9o1', 'Sja6bDxzTcTFwVube7SHEhjxwbx2', 'idhGeYnN7XOTysl6R0LrC2Sdqlv2', 'KP2b8tn6m1aTexnbfoII2OHboe03', 'fZSEvTemhNgVhCqcMp76IzK1RLv2', 'VJDApcT8ehTeWv2llV8QGzHTEWS2', 'teAZL0rNDyWevenTcbHQTdzCOGi2', '8trmtB9UNhVR9hYf6MGqTSCrzV93']


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

for b in range(0,12):

    results = results_per_user[b]

    length = len(results)

    for i in range(1,9):
        a = round(i*0.125*length)
        split_results = results[:a]
        rateable_pool = create_rateable_pool_from_video_ids(video_ids)
        rateable_pool = process_unweighted_results(rateable_pool, split_results)
        split_ratings = rateable_pool

        relevant_ratings = []
        for element in split_ratings:
            relevant_ratings.append({'id': split_ratings[element].name, 'n_pairs': split_ratings[element].n_pairs,
                                     'rating': split_ratings[element].rating, 'rd': split_ratings[element].rd,
                                     'vol': split_ratings[element].vol})

        with open('user' + str(b) + '_split_ratings' + str(i) + '.json', 'w') as f:
            json.dump(relevant_ratings, f, indent=4)


