import argparse
import schedule
import sys
import time

from config import get_config
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
                convert         Generates post data from a set of submission_ids
            ''')

        parser.add_argument(
            'command', help='Subcommand to run', metavar='command', choices=['autocollect', 'convert'])
        args = parser.parse_args(sys.argv[1:2])

        # Only accept commands that are methods of this class
        if not hasattr(self, args.command):
            print('Unrecognized command.')
            parser.print_help()
            sys.exit(1)

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
    The data is exported into the directory specified by config.json.

    :param fetch-type: determines whether the data obtained is actual post data     or the IDs of the submissions
    :param subreddit: which subreddit to extract the submissions from
    :param sort-by: the sorting method of the submissions (hot, new or top)
    :return: DataFrame of the collected data
    """

    def __collect(self, fetch_type, subreddit, sort_by):

        print(
            f'Fetching posts... type:{fetch_type}, subreddit:{subreddit}, '
            f'sort_by:{sort_by}')

        if fetch_type == 'ids':
            submissions = self.praw_manager.get_submission_ids(
                subreddit, sort_by=sort_by)
            self.data_manager.save_to_csv(
                submissions, file_identifier=subreddit+'_submission_id_')
            print(f'Successfully collected and saved submissions.')
            return submissions

        elif fetch_type == 'posts':
            posts = self.praw_manager.get_posts(subreddit, sort_by=sort_by)
            self.data_manager.save_to_csv(
                posts, file_identifier=subreddit+'_posts_')
            print(f'Successfully collected and saved submissions.')
            return posts

        else:
            print('Invalid fetch_type for collecting data:', fetch_type)
            return False

    """
    Periodically performs __collect to obtain the requested data
    every 2 hours.
    To-do: allow customization of periodicity
    """

    def autocollect(self):
        # Create an argument parser to extract subcommand args
        parser = argparse.ArgumentParser(
            description='Periodically fetches and saves reddit posts',
            usage=('collectdata.py autocollect --fetch-type [-h, --help] '
                   '[-r, --subreddit] [-s, --sort-by]'))
        parser.add_argument('--fetch-type',
                            choices=['ids', 'posts'], required=True)
        parser.add_argument('--subreddit', '-r', default='AskReddit')
        parser.add_argument('--sort-by', '-s', default='new',
                            choices=['new', 'top', 'hot'])

        # Only take arguments after the subcommand
        args = parser.parse_args(sys.argv[2:])

        schedule.every(2).hours.do(self.__collect, fetch_type=args.fetch_type,
                                   subreddit=args.subreddit, sort_by=args.sort_by)

        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                print('Stopping automatic collection...')
                sys.exit()

    """
    Generates a table of submission data from a set of submission IDs.
    The submission IDs are passed in through a CSV file.
    The attributes obtained for each submission is the set of attributes from declared in config.json
    """

    def convert(self):
        # Create an argument parser to extract subcommand args
        parser = argparse.ArgumentParser(
            description='Generates a CSV of submission details from a CSV of submission IDs',
            usage='collectdata.py convert file [-h, --help]')
        parser.add_argument('file')

        # Only take arguments after the subcommand
        args = parser.parse_args(sys.argv[2:])

        submission_ids = self.data_manager.load_from_csv(args.file)
        if submission_ids is False:
            raise Exception('Failed to load CSV file:' + args.file)

        submissions = self.praw_manager.get_posts_from_ids(submission_ids)

        # Extract the file name without the rest of the path
        new_file_name = args.file.split('/')[-1][:-4]
        new_file_name = 'converted_' + new_file_name
        
        self.data_manager.save_to_csv(submissions, timestamp=False, file_identifier=new_file_name)


def main(*args, **kwargs):
    dc = DataCollector()


if __name__ == '__main__':
    main()
