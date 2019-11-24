import argparse
import requests
import pandas as pd
from config import get_config
from datetime import datetime, timedelta

request_prefix = "https://api.pushshift.io/reddit/search/submission/?subreddit="
requested_attrs = get_config('fetch_attributes')


"""
Makes a request call for a maximum of 1000 posts that are either less than or greater than the threshold score. Returns the json data of request if the request was successful.
"""

def _request_posts(score_query, subreddit, before_time):
    request_prefix_subreddit = request_prefix + subreddit

    request_suffix = f'&before={str(before_time)}&sort=desc&sort_type=created_utc&size=1000&score{score_query}'

    request = requests.get(request_prefix_subreddit + request_suffix)

    # print(f'Request status: {request.status_code}, req:{request_prefix_subreddit + request_suffix}')
    assert request.status_code == 200, f'Failed request with status code {request.status_code} with request: {request_prefix_subreddit + request_suffix}'

    return request.json()


"""
Returns a DataFrame of all the posts (up to max_samples) that satisfy the score_query and sets each of the score values to class_num.
"""

def scrape_posts(max_samples, score_query, class_num, subreddit):
    earliest_post = int((datetime.now() - timedelta(days=2)).timestamp())
    posts_scraped = []

    posts = _request_posts(score_query, subreddit, earliest_post)

    while len(posts['data']) != 0:
        # print(f'Earliest post: {earliest_post}')
        print(f'Posts remaining: {max_samples - len(posts_scraped)}')

        num_posts_left = min(len(posts['data']),
                             max_samples - len(posts_scraped))

        if num_posts_left == 0:
            break

        for post_idx in range(num_posts_left):
            post_data = posts['data'][post_idx]
            requested_post_info = []
            earliest_post = min(earliest_post, post_data['created_utc'])

            for attr in requested_attrs:
                if attr == 'score':
                    requested_post_info.append(class_num)
                elif not attr in post_data:
                    if attr == 'link_flair_text':
                        requested_post_info.append('')
                    else:
                        raise AttributeError(
                            f'Attribute not found in post data: "{attr}"')
                else:
                    requested_post_info.append(post_data[attr])

            posts_scraped.append(requested_post_info)

        posts = _request_posts(score_query, subreddit, earliest_post)
        # print(f'Fetching with t:{threshold}, lt:{lt}, ep:{earliest_post}')

    posts_df = pd.DataFrame(posts_scraped, columns=requested_attrs)
    return posts_df