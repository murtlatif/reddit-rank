from datetime import datetime, timedelta
import re
from pandas import isna
from math import log10


"""
Filters post creation time.

Returns True if the post is greater than 2 days old at the time of applying the filter. Returns False otherwise.
"""

def filter_post_age(created_utc):
    return created_utc <= (datetime.now() - timedelta(days=2)).timestamp()


"""
Modifies title to be more suitable for word vector transformation.
"""

def update_title(title):
    new_title = replace_subreddit(title)
    new_title = update_currency_to_word(new_title)
    new_title = update_numbers(new_title)
    new_title = replace_common_words(new_title)
    return new_title


"""
Replaces every occurrence of 'r/...' with 'subreddit' within a title
"""

def replace_subreddit(title):
    return re.sub(r'\sr/\w+', 'subreddit', title)


"""
Replaces currency symbols with their respective english words.
"""

def update_currency_to_word(title):
    updated_title = re.sub(r'\$', r' dollars ', title)
    updated_title = re.sub(r'€', r' euros ', updated_title)
    updated_title = re.sub(r'¥', r' yen ', updated_title)
    updated_title = re.sub(r'£', r' pounds ', updated_title)
    return updated_title


"""
Internally used function that decides how a number is transformed.
Any numbers less than 100 are kept as their original number.
The number 420 is kept as its original number.
Any numbers between 101 and 9,999,999,999 are rounded down to the nearest
order of magnitude (e.g. 4266 -> 1000, 7412233 -> 1000000)
"""

def _replace_number(num_match):
    num_as_str = num_match.group()
    num = float(num_as_str)

    # Handle numbers that are not within the logarithmic domain
    if num == 0:
        return num_as_str
    elif num < 0:
        num = -num

    log_num = int(log10(num))
    if log_num <= 1 or (num_as_str == '420'):
        return num_as_str
    elif log_num <= 9:
        return str(10 ** log_num)
    else:
        return 'many'


"""
Removes commas from numbers and replaces numbers with their updated
values decided using _replace_number
"""

def update_numbers(title):
    # Remove commas from numbers
    no_comma_nums_title = re.sub(r'(\d),(\d)', r'\1\2', title)

    # Replace the numbers with nearest magnitude or 'many' if too large
    return re.sub(r'\d+', _replace_number, no_comma_nums_title)


"""
Replaces acronyms and redundant words within a title
"""

def replace_common_words(title):
    # Replace SO -> significant other
    new_title = re.sub(r'(\W)SO(\W)', r'\1significant other\2', title)

    # Replace OP with overpowered
    new_title = re.sub(r'(\W)OP(\W)', r'\1original poster\2', new_title)

    # Replace 'redditors' with 'people'
    new_title = re.sub(r'redditors', r'people', new_title, flags=re.IGNORECASE)

    # Remove the [Serious] or [serious] tag
    new_title = re.sub(r'[\[\(]serious[\)\]]', '', new_title,
                       flags=re.IGNORECASE)

    return new_title


"""
Replaces the link_flair_text column to a boolean value column determining
whether or not the link contains a 'Serious Replies Only' flair.
"""

def has_flair(flair_text):
    if flair_text == 'Serious Replies Only':
        return True
    else:
        return False


"""
Filter posts with a flair that indicates breaking news or a moderator post.
Returns true if there is no flair, or a 'Serious Replies Only' flair, and
returns false otherwise.
"""

def filter_special_content(flair_text):
    if (isna(flair_text)) or (flair_text == 'Serious Replies Only'):
        return True
    else:
        return False

"""
Replaces the score values with a 1 for a 'high' score, or 0 for a 'low' score.
"""

def classify_score(score, threshold):
    if score >= threshold:
        return 1
    else:
        return 0
