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
    - Title processing update filter
    - (str)flair text -> (bool)serious_flair converter
    """

    def clean_data(self, df):

        # Remove duplicates by id
        clean_df = df.drop_duplicates(subset='id')

        # Remove posts that are atypical (e.g. breaking news, mod posts)
        clean_df = clean_df[clean_df['link_flair_text'].apply(
            filters.filter_special_content)]

        # Apply modification filters on the title
        clean_df['title'] = clean_df['title'].apply(filters.update_title)

        # Modify link_flair_text to bools checking existence of a Serious flair
        clean_df = clean_df.rename(
            columns={'link_flair_text': 'serious_flair'})
        clean_df['serious_flair'] = clean_df['serious_flair'].apply(
            filters.has_flair)

        clean_df['over_18'] = clean_df['over_18'].apply(filters.nsfw_bool_to_num)

        # Replace created_utc with weekday and hour of the day
        weekday_hour_list = clean_df['created_utc'].apply(filters.convert_date_time).values.tolist()

        weekday_hour_df = pd.DataFrame(weekday_hour_list, columns=['created_utc', 'hour', 'weekday'])

        weekday_hour_df = weekday_hour_df.drop_duplicates()
        clean_df = pd.merge(clean_df, weekday_hour_df)

        # Drop created_utc and reorder columns
        clean_df = clean_df.drop(columns=['created_utc'])

        clean_df = clean_df[['id','title','over_18','serious_flair','hour','weekday', 'score']]

        # One hot encode hour and weekday columns (leave score until after split)
        # clean_df = pd.concat([clean_df, pd.get_dummies(clean_df['score'], prefix='score')], axis=1)
        clean_df = pd.concat([clean_df, pd.get_dummies(clean_df['hour'], prefix='hour')], axis=1)
        clean_df = pd.concat([clean_df, pd.get_dummies(clean_df['weekday'], prefix='weekday')], axis=1)

        clean_df = clean_df.drop(columns=['hour', 'weekday'])

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
    DEPRECATED
    Splits a binary classified dataset into two equally sized datasets for each
    classification, and remerges them to produce a dataset with an equal number
    of samples for each class.
    """

    def balance_dataset_deprecated(self, df):
        df_low = df[df['score'] == 0]
        df_high = df[df['score'] == 1]

        sample_size = min(len(df_low), len(df_high))
        df_low_samples = df_low.sample(n=sample_size, random_state=self.seed)
        df_high_samples = df_high.sample(n=sample_size, random_state=self.seed)

        return pd.concat([df_low_samples, df_high_samples])

    """
    Merges a list of DataFrames.
    """
    def merge(self, dfs):
        if len(dfs) == 0:
            return None

        return pd.concat(dfs)

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

    """
    Onehot encode the score column into multiple columns depending on the number of classes defined in config.json
    """
    def onehot_score(self, df):
        score_classes = get_config('classes')
        score_columns = [f'score_{class_num}' for class_num in range(len(score_classes))]

        score_onehot_list = df['score'].apply(filters.convert_onehot).values.tolist()

        score_onehot_df = pd.DataFrame(score_onehot_list, columns=score_columns)
        merged_onehot_df = pd.concat([df.reset_index(drop=True), score_onehot_df.reset_index(drop=True)], axis=1)
        return merged_onehot_df.drop(columns=['score'])

    def get_stats(self, df):
        stats = {'frequency' : {}}

        weekday_hour_list = df['created_utc'].apply(filters.convert_date_time).values.tolist()

        weekday_hour_df = pd.DataFrame(weekday_hour_list, columns=['created_utc', 'hour', 'weekday'])

        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        temp_df = df
        temp_df['hour'] = weekday_hour_df['hour']
        temp_df['weekday'] = weekday_hour_df['weekday'].apply(lambda x: weekday_names[x])


        # Frequencies of value occurrences
        freq_attrs = ['score', 'hour', 'weekday', 'spoiler']
        for freq_attr in freq_attrs:
            if freq_attr in temp_df:
                stats['frequency'][freq_attr] = df[freq_attr].value_counts()
        
        # Average title length
        total_length = 0
        for post_title in df['title']:
            total_length += len(post_title)
        
        stats['avg_title_len'] = total_length/len(df['title'])

        return stats

