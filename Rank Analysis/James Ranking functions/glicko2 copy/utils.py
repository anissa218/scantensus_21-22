import copy
import pandas as pd
import numpy as np
from collections import defaultdict
from glicko2.Rateable import Rateable


def create_rateable_pool_from_video_ids(player_list):
    pool = {}
    for player_name in player_list:
        pool[player_name] = Rateable(name=player_name)
    return pool


def process_unweighted_results(rateable_pool, results_list, omit_user=None, specific_user=None):
    for result in results_list:
        assert not (omit_user and specific_user), "Can't have both omitting and specifying a user"
        if omit_user:
            if result['user'] == omit_user:
                continue
        if specific_user:
            if result['user'] != specific_user:
                continue
        for i_player, player_id in enumerate(result['opinion']):
            # Cache the ratings & rds before we start so the last person to play A still plays vs A's original score
            bl_ratings = {pid: rateable_pool[pid].getRating() for pid in result['opinion'] if pid in rateable_pool}
            bl_rds = {pid: rateable_pool[pid].getRd() for pid in result['opinion'] if pid in rateable_pool}
            if player_id in rateable_pool:  # Image may have been rated but later removed
                rateable = rateable_pool[player_id]
                player_vs_ratings = []
                player_vs_deviations = []
                player_vs_results = []
                for i_vs_player, vs_player_id, in enumerate(result['opinion']):
                    # Image may have been rated but later removed, and can't play itself
                    if vs_player_id in rateable_pool and player_id != vs_player_id:
                        res = 1 if i_player < i_vs_player else 0
                        player_vs_ratings.append(bl_ratings[vs_player_id])
                        player_vs_deviations.append(bl_rds[vs_player_id])
                        player_vs_results.append(res)
                if player_vs_ratings:
                    rateable.update_rateable(player_vs_ratings, player_vs_deviations, player_vs_results)
    return rateable_pool


def process_user_volatilities(users, results_list, video_ids):
    pool = create_rateable_pool_from_video_ids(player_list=video_ids)
    for user_id in users.keys():
        process_unweighted_results(pool, results_list, omit_user=user_id)
        volalities_pre = {player_id: pool[player_id].vol for player_id in pool.keys()}
        process_unweighted_results(pool, results_list, specific_user=user_id)
        volalities_post = {player_id: pool[player_id].vol for player_id in pool.keys()}
        volatility_delta = []
        for player_id, volatility_pre in volalities_pre.items():
            if volalities_post[player_id] != volatility_pre:  # Only look for a delta if a change (ie rated)
                volatility_delta.append(volalities_post[player_id] - volatility_pre)
        users[user_id]['volatility'] = np.mean(volatility_delta)
    return users


def get_users_and_pairs_from_results(results):
    pairs_by_user = defaultdict(int)
    for result in results:
        opinions = result['opinion']
        n_pairs = len(opinions) * (len(opinions)-1) / 2
        pairs_by_user[result['user']] += n_pairs

    users = {}
    for user_id, n in pairs_by_user.items():
        users[user_id] = {}
        users[user_id]['pairs'] = n
        users[user_id]['volatility'] = 0

    return users


def pool_to_sorted_dataframe(pool):
    pool_sorted = {value.name: value for value in sorted(pool.values(), key=lambda x: x.getRating(), reverse=True)}

    sorted_names = [video.name for video in pool_sorted.values()]
    sorted_ratings = [video.getRating() for video in pool_sorted.values()]
    sorted_n_ratings = [video.n_ratings for video in pool_sorted.values()]
    sorted_n_pairs = [video.n_pairs for video in pool_sorted.values()]
    sorted_rd = [video.getRd() for video in pool_sorted.values()]
    sorted_cis_lower = [video.lowerCI() for video in pool_sorted.values()]
    sorted_cis_upper = [video.upperCI() for video in pool_sorted.values()]
    sorted_volatility = [video.vol for video in pool_sorted.values()]

    df = pd.DataFrame({'id': sorted_names,
                       'rating': sorted_ratings,
                       'n_ratings': sorted_n_ratings,
                       'n_pairs': sorted_n_pairs,
                       'rd': sorted_rd,
                       'ci_lower': sorted_cis_lower,
                       'ci_upper': sorted_cis_upper,
                       'volatility': sorted_volatility})
    return df


def get_n_highest_rd_from_pool(pool, n):
    pool = copy.deepcopy(pool)
    pool_sorted = {value.name: value for value in sorted(pool.values(), key=lambda x: x.getRd(), reverse=True)}
    return list(pool_sorted.keys())[:n]