#from utils_copy import create_rateable_pool_from_video_ids, process_unweighted_results, get_users_and_pairs_from_results
from glicko2.utils import create_rateable_pool_from_video_ids, process_unweighted_results, get_users_and_pairs_from_results
import json
from pathlib import Path

# Get ratings for all users
#responses_path = Path("jan24_expert_responses.json")
responses_path = Path("feb17_expert_responses.json")
with open(responses_path, 'r') as json_f:
    responses = json.load(json_f)

config = responses['config']
results = responses['results']

# Get users
users = get_users_and_pairs_from_results(results)


video_ids = config[0]['cases']

# Split results in increasing sizes (8 different sets)

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

    with open('split_ratings' + str(i) + '.json', 'w') as f:
        json.dump(relevant_ratings, f, indent=4)
