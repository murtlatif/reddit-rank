import argparse
import schedule
import sys
import time

from config import get_config
from clientauth import authorization_info_from_json
from prawmanager import PRAWManager
from datamanager import DataManager


class DataCollector:

    """
    Initializes the PRAW Manager and Data Manager, and parses arguments
    to perform any subcommands available.
    """

    def __init__(self):

        parser = argparse.ArgumentParser(
            description='RedditRank - Data Collecting Service: collectdata.py',
            usage='''collectdata.py <command> [<args>]

            The available commands are:
                autocollect     Periodically fetches and saves reddit posts
            ''')

        parser.add_argument(
            'command', help='Subcommand to run', metavar='command', choices=['autocollect'])
        args = parser.parse_args(sys.argv[1:2])

        # Only accept commands that are methods of this class
        if not hasattr(self, args.command):
            print('Unrecognized command.')
            parser.print_help()
            exit(1)

        # Get authorization info from client_auth.json
        client_auth_info = get_config('client_auth')
        client_id = client_auth_info['client_id']
        client_secret = client_auth_info['client_secret']
        user_agent = client_auth_info['user_agent']

        # Initialize the PRAW Manager and Data Manager
        self.praw_manager = PRAWManager(client_id, client_secret, user_agent)
        self.data_manager = DataManager()

        # Run the requested method
        getattr(self, args.command)()

    """
    Fetches the latest 1000 posts and saves the data in a CSV file.

    fetch-type: determines whether the data obtained is actual post data or
        the IDs of the submissions
    subreddit: which subreddit to extract the submissions from
    sort-by: the sorting method of the submissions (hot, new or top)
    
    The data is exported into the directory specified by config.json.
    """

    def autocollect(self):
        # Create an argument parser to extract subcommand args
        parser = argparse.ArgumentParser(
            description='Periodically fetches and saves reddit posts',
            usage='collectdata.py autocollect --fetch-type [-h, --help] [-r, --subreddit] [-s, --sort-by]')
        parser.add_argument('--fetch-type',
                            choices=['ids', 'posts'], required=True)
        parser.add_argument('--subreddit', '-r', default='AskReddit')
        parser.add_argument('--sort-by', '-s', default='new',
                            choices=['new', 'top', 'hot'])

        # Only take arguments after the subcommand
        args = parser.parse_args(sys.argv[2:])

        if args.fetch_type == 'ids':
            submissions = self.praw_manager.get_submission_ids(args.subreddit, sort_by=args.sort_by)
            self.data_manager.save_to_csv(
                submissions, file_identifier=args.subreddit+'_submission_id_')

        elif args.fetch_type == 'posts':
            posts = self.praw_manager.get_posts(args.subreddit, sort_by=args.sort_by)
            self.data_manager.save_to_csv(
                posts, file_identifier=args.subreddit+'_posts_')


def main(*args, **kwargs):
    dc = DataCollector()


if __name__ == '__main__':
    main()
