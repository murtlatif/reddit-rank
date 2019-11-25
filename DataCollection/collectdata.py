import argparse
import schedule
import sys
import time

from config import get_config
from prawmanager import PRAWManager
from datamanager import DataManager
from scrapetool import scrape_posts

_default_seed = 29


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
                autocollect     Periodically fetches and saves post data
                manualcollect   Immediately fetches and saves post data
                convert         Generates post data from a set of submission_ids
                clean           Generates clean post data from a post data file
                split           Generates training files from a post data file
            ''')

        parser.add_argument(
            'command', help='Subcommand to run', metavar='command',
            choices=['autocollect',
                     'manualcollect',
                     'convert',
                     'clean',
                     'split',
                     'scrape',
                     'stats'
                     ])
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

    def __collect(self, fetch_type, subreddit, sort_by, time_filter=None):

        print(
            f'Fetching posts... type:{fetch_type}, subreddit:{subreddit}, '
            f'sort_by:{sort_by}')

        if fetch_type == 'ids':
            submissions = self.praw_manager.get_submission_ids(
                subreddit, sort_by=sort_by, time_filter=time_filter)
            self.data_manager.save_to_csv(
                submissions, file_identifier=subreddit+'_submission_id_')
            print(f'Successful IDs fetch at {time.ctime()}')
            return submissions

        elif fetch_type == 'posts':
            posts = self.praw_manager.get_posts(
                subreddit, sort_by=sort_by, time_filter=time_filter)
            self.data_manager.save_to_csv(
                posts, file_identifier=subreddit+'_posts_')
            print(f'Successful posts fetch at {time.ctime()}')
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
        parser.add_argument('--time-filter', default='all',
                            choices=['all', 'day', 'hour', 'month', 'week', 'year'])
        parser.add_argument('--every', default=2, type=int)

        # Only take arguments after the subcommand
        args = parser.parse_args(sys.argv[2:])

        print(f'Starting autocollection at {time.ctime()}')
        schedule.every(args.every).hours.do(self.__collect, fetch_type=args.fetch_type,
                                   subreddit=args.subreddit, sort_by=args.sort_by,
                                   time_filter=args.time_filter)

        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                print('Stopping automatic collection...')
                sys.exit()

    """
    Immediately performs __collect to obtain the requested data.
    """

    def manualcollect(self):
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
        parser.add_argument('--time-filter', default='all',
                            choices=['all', 'day', 'hour', 'month', 'week', 'year'])

        # Only take arguments after the subcommand
        args = parser.parse_args(sys.argv[2:])

        return self.__collect(fetch_type=args.fetch_type, subreddit=args.subreddit, sort_by=args.sort_by, time_filter=args.time_filter)

    """
    Scrapes submission data based on the classes and fetch_attributes specified
    in config.json. The 'samples' argument specifies the maximum number of
    submissions to obtain for each class.
    The data is saved as a .csv in the directory specified in config.json.
    """
    
    def scrape(self):
        # Create an argument parser to extract subcommand args
        parser = argparse.ArgumentParser()
        parser.add_argument('--samples', '-n', default=10000, type=int)
        parser.add_argument('--subreddit', '-r',
                            default='AskReddit', choices=['AskReddit'])
        
        # Only take arguments after the subcommand
        args = parser.parse_args(sys.argv[2:])
        
        score_queries = get_config('classes')

        dfs = []
        try:
            for class_num in range(len(score_queries)):
                class_posts = scrape_posts(args.samples, score_queries[class_num], class_num, args.subreddit)

                dfs.append(class_posts)
                # class_file_name = f'{class_num}_{args.samples}s_{args.subreddit}'

                # self.data_manager.save_to_csv(class_posts, timestamp=False, file_identifier=class_file_name)

        except AssertionError as e:
            print(f'Request was unsuccessful: {e}')
            return False

        merged_dfs = self.data_manager.merge(dfs)
        merged_file_name = f'{args.samples}s_{args.subreddit}'

        self.data_manager.save_to_csv(merged_dfs, timestamp=False, file_identifier=merged_file_name)

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
            raise IOError('Failed to load CSV file:' + args.file)

        submissions = self.praw_manager.get_posts_from_ids(submission_ids)

        # Extract the file name without the rest of the path
        new_file_name = args.file.split('/')[-1][:-4]
        new_file_name = 'converted_' + new_file_name

        self.data_manager.save_to_csv(
            submissions, timestamp=False, file_identifier=new_file_name)

    """
    Applies all data processing filters to the dataset given by the input file and saves the resulting dataset as a new CSV file (prefixed with clean_)
    """

    def clean(self):
        # Create an argument parser to extract subcommand args
        parser = argparse.ArgumentParser(
            description='Generates a CSV of the inputted posts after data cleaning is applied',
            usage='collectdata.py clean file [-h, --help] [--no-age]'
                  '[--threshold 100] [--ids]')
        parser.add_argument('file')
        parser.add_argument('--no-age', action='store_false')
        parser.add_argument('--threshold', type=int, default=100)
        parser.add_argument('--ids', help='Remove duplicate ids only',
                            action='store_true')

        # Only take arguments after the subcommand
        args = parser.parse_args(sys.argv[2:])

        posts = self.data_manager.load_from_csv(args.file)
        if posts is False:
            raise IOError('Failed to load CSV file:' + args.file)

        if args.ids:
            clean_ids = posts.drop_duplicates(subset='id')
            print(f'Removed {len(posts) - len(clean_ids)} duplicate IDs.')

            new_file_name = args.file.split('/')[-1][:-4]
            new_file_name = 'cleanid_' + new_file_name
            self.data_manager.save_to_csv(
                clean_ids, timestamp=False, file_identifier=new_file_name)
            return

        clean_posts = self.data_manager.clean_data(posts)

        print(f'Filtered out {len(posts) - len(clean_posts)} posts.')

        print(f'Frequencies of clean post scores:')
        print(clean_posts['score'].value_counts())

        # Extract the file name without the rest of the path
        new_file_name = args.file.split('/')[-1][:-4]
        new_file_name = 'clean_' + new_file_name

        self.data_manager.save_to_csv(
            clean_posts, timestamp=False, file_identifier=new_file_name)

    """
    Splits a dataset into training, validation, testing and overfitting data,
    saving the resulting datasets into new CSV files with the same name but
    prefixed with 'train_', 'valid_', 'test_', and 'overfit_' respectively.
    
    Percentage of total dataset for each created dataset:
    train:  64%
    valid:  16%
    test:   16%
    overfit: 4%
    """

    def split(self):
        # Create an argument parser to extract subcommand args
        parser = argparse.ArgumentParser(
            description='Generates split training files.',
            usage='collectdata.py split file [-h, --help]')
        parser.add_argument('file')

        # Only take arguments after the subcommand
        args = parser.parse_args(sys.argv[2:])

        posts = self.data_manager.load_from_csv(args.file)
        if posts is False:
            raise IOError('Failed to load CSV file:' + args.file)

        post_data = posts.drop(columns='id')
        # balanced_posts = self.data_manager.balance_dataset(post_data)

        train_valid, test_overfit = self.data_manager.split_data(
            post_data, 0.8)
        train, valid = self.data_manager.split_data(train_valid, 0.8)
        test, overfit = self.data_manager.split_data(test_overfit, 0.8)

        # Display the frequency of each classification for each split dataset
        print(f"==train frequencies==\n{train['score'].value_counts()}\n")
        print(f"==valid frequencies==\n{valid['score'].value_counts()}\n")
        print(f"==test frequencies==\n{test['score'].value_counts()}\n")
        print(f"==overfit frequencies==\n{overfit['score'].value_counts()}\n")

        # Onehot-encode the score for each of the split files
        # train = self.data_manager.onehot_score(train)
        # valid = self.data_manager.onehot_score(valid)
        # test = self.data_manager.onehot_score(test)
        # overfit = self.data_manager.onehot_score(overfit)

        # Get new file names for each
        new_file_name = args.file.split('/')[-1][:-4]

        self.data_manager.save_to_csv(
            train, timestamp=False, file_identifier='train_' + new_file_name)
        self.data_manager.save_to_csv(
            valid, timestamp=False, file_identifier='valid_' + new_file_name)
        self.data_manager.save_to_csv(
            test, timestamp=False, file_identifier='test_' + new_file_name)
        self.data_manager.save_to_csv(
            overfit, timestamp=False, file_identifier='overfit_' + new_file_name)

    def stats(self):
        # Create an argument parser to extract subcommand args
        parser = argparse.ArgumentParser(
            description='Retrieves statistics about a particular dataset',
            usage='collectdata.py split file [-h, --help]')
        parser.add_argument('file')

        # Only take arguments after the subcommand
        args = parser.parse_args(sys.argv[2:])

        post_dataset = self.data_manager.load_from_csv(args.file)
        if post_dataset is False:
            raise IOError('Failed to load CSV file:' + args.file)

        data_stats = self.data_manager.get_stats(post_dataset)

        for data_freq in data_stats['frequency']:
            print(f'Value frequencies of {data_freq}: \n{data_stats["frequency"][data_freq]}')

        print(f'Average title length: {data_stats["avg_title_len"]}')



def main(*args, **kwargs):
    dc = DataCollector()


if __name__ == '__main__':
    main()
