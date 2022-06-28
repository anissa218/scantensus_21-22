import random
from glicko2.utils import *
from firebase import firebase
from datetime import datetime
from google.cloud import datastore


def compare_answer(request, local=False):
    if local:
        project_id = "imp-echo-stowell-a4c-a5c"
        user = "vNrSHLRL0IT6GICFfsQ7ENEqhmz1"
        opinion = ['01-208cb0dce0d6faa7d20a6903e5867cce132a7567466f8a58044f24b88454a7e2-0001', '01-02e9e674e3b41d5095e07837c42f63f9f872ff0662d8c31b1f7868cd997840ef-0001', '01-c7f17812d5d5a483787898240475b7fe75e562f18a0fd14a0b25b466703d7fa8-0001', '01-e8d8c69bba145eea73523f0bdfa6bf19ab8613659d594ab4a2e85f9cb022ef9f-0001', '01-0ab04ee76f5ad088c4918f8ce322b3dac9cc868522f28e521e569d59bf366fe1-0001']
        sync_with_firebase = False
    else:
        sync_with_firebase = True
        json_data = request.get_json(force=True)
        project_id = json_data.get('project_id', None)
        user = json_data.get('user', None)
        opinion = json_data.get('opinion', None)
        if project_id is None or user is None or opinion is None:
            return f"Expected to be passed project_id ({project_id}), user ({user}) and opinion ({opinion})"

    client = datastore.Client('imagine-d6819')
    key = client.key("compare_results")
    entity = datastore.Entity(key=key)
    entity.update({
        'project_id': project_id,
        'opinion': opinion,
        'time': datetime.now(),
        'user': user
    })
    client.put(entity)

    response = compare_rate_and_sync(request, sync_with_firebase=sync_with_firebase, local=local)
    return response


def compare_config_write(request):
    json_data = request.get_json(force=True)
    project_id = json_data.get('project_id', None)
    cases = json_data.get('items', None)
    if project_id is None or cases is None:
        return f"Expected to be passed project_id ({project_id}, and cases ({cases})"

    client = datastore.Client('imagine-d6819')
    query = client.query(kind="compare_config")
    query.add_filter('project_id', '=', project_id)
    results_config = list(query.fetch())

    if len(results_config) == 1:
        config = results_config[0]
        key = client.key("compare_config", config.id)
    elif len(results_config) == 0:
        key = client.key("compare_config")
    else:
        raise ValueError(f"expected 1 config entry for project {project_id}, found {len(results_config)}")

    entity = datastore.Entity(key)
    entity.update({
        'project_id': project_id,
        'cases': cases,
    })
    client.put(entity)
    print(f"Put {entity} -> {entity.id}")
    return {k: v for k, v in entity.items()}


def compare_question(request, local=False):
    if local==True:
        project_id = "imp-echo-francis-lefty-righty-a4c"
        eligible_pool_proportion = 0.25
        n = 6
    else:
        json_data = request.get_json(force=True)
        project_id = json_data.get('project_id', None)
        eligible_pool_proportion = json_data.get('eligible_pool_proportion', None)
        n = json_data.get('n', None)
    if project_id is None or n is None:
        return f"Expected to be passed project_id ({project_id}, and n ({n})"

    client = datastore.Client('imagine-d6819')

    # Get config
    query = client.query(kind="compare_config")
    query.add_filter('project_id', '=', project_id)
    results = list(query.fetch())
    assert len(results) == 1, f"expected 1 config entry for project {project_id}, found {len(results)}"
    config = results[0]

    # Get results
    query = client.query(kind="compare_results")
    query.add_filter('project_id', '=', project_id)
    results = query.fetch()
    results = [{k: v for k, v in result.items()} for result in results]  # Turn from an entity to a plain dictionary

    # Create rating pool & rate
    video_ids = config['cases']
    rateable_pool = create_rateable_pool_from_video_ids(video_ids)
    rateable_pool = process_unweighted_results(rateable_pool, results)

    # Get IDs for highest RD
    if eligible_pool_proportion:
        eligible_pool_n = int(eligible_pool_proportion * len(rateable_pool))
        highest_rds = get_n_highest_rd_from_pool(rateable_pool, max(eligible_pool_n, n))
        highest_rds = random.sample(highest_rds, n)
    else:
        highest_rds = get_n_highest_rd_from_pool(rateable_pool, n)
    #rds = {value.name: value for value in sorted(rateable_pool.values(), key=lambda x: x.getRd(), reverse=True)}
    response = {'items': highest_rds}
    return response


def compare_rate_and_sync(request, sync_with_firebase=True, local=False, calc_volatility=False):
    if local:
        project_id = "imp-echo-stowell-a4c-a5c"
    else:
        json_data = request.get_json(force=True)
        project_id = json_data.get('project_id', None)
        assert project_id is not None, f"expected project_id to be passed in request"

    client = datastore.Client('imagine-d6819')

    # Get config
    query = client.query(kind="compare_config")
    query.add_filter('project_id', '=', project_id)
    results = list(query.fetch())
    assert len(results) == 1, f"expected 1 config entry for project {project_id}, found {len(results)}"
    config = results[0]

    # Get results
    query = client.query(kind="compare_results")
    query.add_filter('project_id', '=', project_id)
    results = query.fetch()
    results = [{k: v for k, v in result.items()} for result in results]  # Turn from an entity to a plain dictionary

    # Get users
    users = get_users_and_pairs_from_results(results)

    # Create rating pool & rate
    video_ids = config['cases']
    rateable_pool = create_rateable_pool_from_video_ids(video_ids)
    rateable_pool = process_unweighted_results(rateable_pool, results)

    # Calculate user metrics
    if calc_volatility:
        print(f"Processing user volatilities")
        users = process_user_volatilities(users=users, results_list=results, video_ids=video_ids)
    else:
        print(f"Not calculating volatility")

    # Prepare for FB upload
    df = pool_to_sorted_dataframe(rateable_pool)
    results_as_list_of_dicts = list(df.T.to_dict().values())
    timestring = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    pooled_pairs = {'pairs': sum((user['pairs'] for user in users.values()))}

    if sync_with_firebase:
        print(f"Syncing with FB...")
        fb = firebase.FirebaseApplication('https://scantensus.firebaseio.com', None)
        response_rankings = fb.put('/fiducial/' + project_id, 'rankings',
                                   {'list': results_as_list_of_dicts,
                                    'time': timestring})
        response_credit = fb.put('/fiducial/' + project_id, 'credit',
                                 {'user': users,
                                  'pooled': pooled_pairs})
        response = {'rankings': response_rankings,
                    'credit': response_credit}
        return response
    else:
        return {
            'rankings': {
                'list': results_as_list_of_dicts,
                'time': timestring},
            'credit': {
                'user': users,
                'pooled': pooled_pairs}}


if __name__ == "__main__":
    compare_question(None, local=True)
