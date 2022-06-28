#from utils_copy import create_rateable_pool_from_video_ids, process_unweighted_results, get_users_and_pairs_from_results
from glicko2.utils import create_rateable_pool_from_video_ids, process_unweighted_results, get_users_and_pairs_from_results
import json
from pathlib import Path

# Get ratings for all users
responses_path = Path("development_responses_jan11.json")
with open(responses_path, 'r') as json_f:
    responses = json.load(json_f)

config = responses['config']
results = responses['results']

# Get users
users = get_users_and_pairs_from_results(results)

# Create rating pool and rate
video_ids = config[0]['cases']
rateable_pool = create_rateable_pool_from_video_ids(video_ids)
rateable_pool = process_unweighted_results(rateable_pool, results)
total = rateable_pool

all_ratings = []
for element in total:
    all_ratings.append({'id':total[element].name,'n_pairs':total[element].n_pairs, 'rating':total[element].rating, 'rd':total[element].rd,'vol':total[element].vol})

# save ratings
with open('all_ratings.json','w') as f:
    json.dump(all_ratings,f,indent=4)

def split_rankings(i): # i is a number between 0 and 1 which indicates how much overlap there is in pairwise comparisons
    ## data
    responses_path = Path("development_responses_jan11.json")
    with open(responses_path, 'r') as json_f:
        responses = json.load(json_f)

    config = responses['config']
    results = responses['results']

    # Get users
    users = get_users_and_pairs_from_results(results)

    # Create rating pool and rate
    video_ids = config[0]['cases']
    rateable_pool = create_rateable_pool_from_video_ids(video_ids)
    rateable_pool = process_unweighted_results(rateable_pool, results)
    total = rateable_pool

    # Split results in half (length of 1403)

    a = round(702 + i*701)
    b = round(702 - i*702)
    results_part1 = results[:a] # length of 702
    results_part2 = results[b:1404] # length of 701

    rateable_pool = create_rateable_pool_from_video_ids(video_ids)
    rateable_pool = process_unweighted_results(rateable_pool, results_part1)
    ratings_part1 = rateable_pool

    rateable_pool = create_rateable_pool_from_video_ids(video_ids)
    rateable_pool = process_unweighted_results(rateable_pool, results_part2)
    ratings_part2 = rateable_pool

    # ratings_all_users = []
    # ratings_per_user = []
    # for i in range(8):

    all_ratings = []
    for element in total:
        all_ratings.append({'id':total[element].name,'n_pairs':total[element].n_pairs, 'rating':total[element].rating, 'rd':total[element].rd,'vol':total[element].vol})

    ratings_first_half = []
    for element in ratings_part1:
        ratings_first_half.append({'id':ratings_part1[element].name,'n_pairs':ratings_part1[element].n_pairs, 'rating':ratings_part1[element].rating, 'rd':ratings_part1[element].rd,'vol':ratings_part1[element].vol})

    ratings_second_half = []
    for element in ratings_part2:
        ratings_second_half.append({'id':ratings_part2[element].name,'n_pairs':ratings_part2[element].n_pairs, 'rating':ratings_part2[element].rating, 'rd':ratings_part2[element].rd,'vol':ratings_part2[element].vol})

    # with open('all_ratings.json','w') as f:
    #     json.dump(all_ratings,f,indent=4)
    with open(str(i)+'ratings_first_half.json','w') as f:
        json.dump(ratings_first_half,f,indent=4)
    with open(str(i)+'ratings_second_half.json','w') as f:
        json.dump(ratings_second_half,f,indent=4)