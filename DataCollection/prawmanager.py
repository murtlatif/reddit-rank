import praw
import pandas as pd
from datetime import datetime


class PRAWManager:

    """ 
    Initializes the PRAW client using the authorization info provided
    Also sets the attributes to collect when fetching posts from a subreddit
    """

    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent)

        self.collecting_attributes = [
            'title', 'num_comments', 'spoiler', 'over_18', 'created_utc', 'score']

    """
    Retrieves num_posts posts from the specified subreddit from the section
    specified by the parameter `sorting` which can be 'hot', 'top' or 'new'.
    The attributes obtained are specified by a list `attributes` which 
    contains the names of the attributes requested.
    The result is a DataFrame object of the requested posts.
    """

    def get_from_subreddit(self, subreddit, num_posts, attributes, sorting='hot'):

        result = []
        subred = self.reddit.subreddit(subreddit)

        if sorting == 'hot':
            posts = subred.hot(limit=num_posts)
        elif sorting == 'top':
            posts = subred.top(limit=num_posts)
        elif sorting == 'new':
            posts = subred.new(limit=num_posts)
        else:
            raise Exception(
                "Invalid sorting type provided. Try 'hot', 'top' or 'new'.")

        for post in posts:
            post_data = []
            for attribute in attributes:
                if hasattr(post, attribute):
                    post_data.append(getattr(post, attribute))
                else:
                    raise Exception(f'Invalid attribute given: {attribute}')
            result.append(post_data)

        posts_df = pd.DataFrame(result, columns=attributes)
        return posts_df

    """
    Obtains the ids for the requested selection of posts
    """

    def get_submission_ids(self, subreddit, num_posts=1000, sort_by='hot'):

        submission_ids = self.get_from_subreddit(subreddit, num_posts, ['id'], sorting=sort_by)
        return submission_ids


    def get_posts(self, subreddit, num_posts=1000, sort_by='hot'):
        
        posts = self.get_from_subreddit(subreddit, num_posts, self.collecting_attributes, sort_by)
        return posts