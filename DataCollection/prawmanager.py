import praw
import pandas as pd
from datetime import datetime

from config import get_config


class PRAWManager:

    """ 
    Initializes the PRAW client using the authorization info provided
    Also sets the attributes to collect when fetching posts from a subreddit
    """

    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret,
            user_agent=user_agent)

        self.collecting_attributes = get_config('fetch_attributes')

    """
    Converts a set of submissions to a DataFrame object with the columns
    as the requested attributes.

    :param submissions: ListingGenerator or other set of submissions
    :param attributes: list of attribute strings to extract from each submission
    """

    def submissions_to_df(self, submissions, attributes):
        result = []
        for post in submissions:
            post_data = []
            for attribute in attributes:
                if hasattr(post, attribute):
                    post_data.append(getattr(post, attribute))
                else:
                    raise AttributeError(
                        f'Invalid attribute given: {attribute}')
            result.append(post_data)

        posts_df = pd.DataFrame(result, columns=attributes)
        return posts_df

    """
    Retrieves num_posts posts from the specified subreddit from the section
    specified by the parameter `sorting` which can be 'hot', 'top' or 'new'.
    The attributes obtained are specified by a list `attributes` which 
    contains the names of the attributes requested.
    The result is a ListingGenerator object of the requested posts.
    """

    def get_from_subreddit(self, subreddit, num_posts, sorting='hot'):

        subred = self.reddit.subreddit(subreddit)

        if sorting == 'hot' or sorting == 'top' or sorting == 'new':
            # Applies subred.hot(), subred.top() or subred.new()
            return getattr(subred, sorting)(limit=num_posts)
        else:
            raise ValueError(
                "Invalid sorting type provided. Try 'hot', 'top' or 'new'.")

    """
    Obtains the ids for the requested selection of posts
    """

    def get_submission_ids(self, subreddit, num_posts=1000, sort_by='hot'):

        submission_ids = self.get_from_subreddit(
            subreddit, num_posts, sorting=sort_by)
        submission_ids = self.submissions_to_df(submission_ids, ['id'])
        return submission_ids

    """
    Obtains a set of post information for the requested selection of posts
    """

    def get_posts(self, subreddit, num_posts=1000, sort_by='hot'):

        posts = self.get_from_subreddit(subreddit, num_posts, sort_by)
        posts = self.submissions_to_df(posts, self.collecting_attributes)
        return posts

    """
    Obtains the request_id used for requesting submissions by id.
    The prefix t3_ is used for submissions.
    """

    def _get_req_id_from_submission_id(self, post_id):
        # Submission API requests by ID are made using t3_<id>
        return post_id if post_id.startswith('t3_') else f't3_{post_id}'

    """
    Obtains a set of post information from a set of ids
    """

    def get_posts_from_ids(self, df_ids):

        if not isinstance(df_ids, pd.DataFrame):
            raise TypeError('Submission IDs must be given in a DataFrame')
        if not 'id' in df_ids:
            print('Submission ids must have a column header')
            raise AttributeError(
                'Given dataframe does not have a column for "ids"')

        # Extract the list of ids from the Dataframe by reshaping it to 1D
        req_ids = df_ids.values.reshape(-1).tolist()
        req_ids = [self._get_req_id_from_submission_id(pid) for pid in req_ids]

        submissions = self.reddit.info(req_ids)
        return self.submissions_to_df(submissions, self.collecting_attributes)
