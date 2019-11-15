import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import filters
from config import get_config

_default_seed = 29


class DataManager:

    def __init__(self, seed=_default_seed):
        self.seed = seed

    """
    Saves a Pandas DataFrame object to a CSV. The exported file will be 
    located in the directory specified by the `data_export_path` attribute
    from config.json. 
    The file name is optional by the caller, with the default
    file name being the timestamp of the export. Timestamps may still be
    enabled if an output_file name is specified.
    If wrap_dir is True, then the file identifier will add the export 
    directory as a prefix and .csv as a suffix.

    """

    def save_to_csv(self, data_to_save, timestamp=True, file_identifier='', wrap_dir=True):
        if not isinstance(data_to_save, pd.DataFrame):
            print('save_to_csv failed: Attempted to save a non DataFrame object.')
            return False

        if (not file_identifier) or timestamp:
            file_identifier += datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        if wrap_dir:
            output_dir = get_config('data_export_path') + \
                file_identifier + '.csv'
        else:
            output_dir = file_identifier

        data_to_save.to_csv(output_dir, index=False)
        return True

    """
    Creates a new Pandas DataFrame object corresponding to the CSV file that
    was passed in.
    """

    def load_from_csv(self, csv_path):
        if not csv_path:
            print('No CSV file specified.')
            return False

        return pd.read_csv(csv_path)

    """
    Applies filters to process the inputted dataset.

    Filters applied:
    - Duplication removal
    - Post age filter
    - Title processing update filter
    - (str)flair text -> (bool)serious_flair converter
    """

    def clean_data(self, df, age_filter=True, threshold=100):

        # Remove duplicates
        clean_df = df.drop_duplicates()

        # Apply filters.filter_post_age to only keep posts more than 2 days old
        if age_filter:
            clean_df = clean_df[clean_df['created_utc'].apply(
                filters.filter_post_age)]

        # Apply modification filters on the title
        clean_df['title'] = clean_df['title'].apply(filters.update_title)

        # Modify link_flair_text to bools checking existence of a Serious flair
        clean_df = clean_df.rename(
            columns={'link_flair_text': 'serious_flair'})
        clean_df['serious_flair'] = clean_df['serious_flair'].apply(
            filters.has_flair)

        # Update score to a 1/0 for high/low score
        clean_df['score'] = clean_df['score'].apply(
            filters.classify_score, threshold=threshold)

        return clean_df

    """
    Extracts the inputs (all columns excluding score and id) and the 
    labels (score column) from the dataset.
    """

    def extract_labels(self, df, drop_score=False):

        if not isinstance(df, pd.DataFrame):
            raise TypeError('Input must be a DataFrame object')

        if (not 'score' in df):
            raise KeyError(
                '"score" column not found when extracting labels')

        labels = df['score']

        return labels

    """
    Splits a binary classified dataset into two equally sized datasets for each
    classification, and remerges them to produce a dataset with an equal number
    of samples for each class.
    """

    def balance_dataset(self, df):
        df_low = df[df['score'] == 0]
        df_high = df[df['score'] == 1]

        sample_size = min(len(df_low), len(df_high))
        df_low_samples = df_low.sample(n=sample_size, random_state=self.seed)
        df_high_samples = df_high.sample(n=sample_size, random_state=self.seed)

        return pd.concat([df_low_samples, df_high_samples])

    """
    Splits data into two sets. The split percentage size is determined by 
    left_percentage [from 0 to 1], where the returned left and right returned 
    sets will have (left_percentage) and (1 - left_percentage) of the total 
    datasets respectively. The seed of the datamanger is used for consistency. 
    The labels are used to stratify the split.
    """

    def split_data(self, df, left_percentage, stratified=True):
        strat = df['score'] if stratified else None
        df_left, df_right = train_test_split(
            df, train_size=left_percentage, random_state=self.seed, shuffle=True, stratify=strat)

        return df_left, df_right
