import pandas as pd
from datetime import datetime, timedelta

import filters
from config import get_config


class DataManager:

    def __init__(self):
        pass

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
    """

    def clean_data(self, df):

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Apply filters.filter_post_age to only keep posts more than 2 days old
        df = df[df['created_utc'].apply(filters.filter_post_age)]

        # Apply modification filters on the title
        df['title'] = df['title'].apply(filters.update_title)
