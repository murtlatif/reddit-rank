import json

config = {}

with open('config.json') as json_file:
    config = json.load(json_file)


def get_config(setting):
    if setting in config:
        return config[setting]
    else:
        raise ValueError(f'Setting does not exist: {setting}')