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
    from config.json. The file name is optional by the caller, with the default
    file name being the timestamp of the export. Timestamps may still be
    enabled if an output_file name is specified.
    """

    def save_to_csv(self, data_to_save, timestamp=True, file_identifier=''):
        if not isinstance(data_to_save, pd.DataFrame):
            print('save_to_csv failed: Attempted to save a non DataFrame object.')
            return False

        output_dir = get_config('data_export_path')

        if file_identifier:
            output_dir += file_identifier
        if (not file_identifier) or timestamp:
            output_dir += datetime.now().strftime('%Y-%m-%dT%H-%M-%S.csv')

        data_to_save.to_csv(output_dir, index=False)
        return True

    def clean_data(self, df):
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Apply filters.filter_post_age to only keep posts more than 2 days old
        df = df[df['created_utc'].apply(filters.filter_post_age)]

        # Apply modification filters on the title
        df['title'] = df['title'].apply(filters.update_title)


        
